"""JevBench v1.5 scoring and the serving policies on runs' per-item predictions.

    JAX_PLATFORMS=cpu python scripts/v15_study.py NAME=RUN/post ... --tag -k2 \\
        --out reports/v15-study

For each run (its ``post/`` directory: ``calibration-jax{tag}.json`` and
``predictions/{jevbench,generators,validation}{tag}.jsonl.gz``, whichever exist):

1. the v1.5 Intelligence (per type × tier) and Calibration (per type) as calibrated, and
   at T = 1, with the noul abstention rate and the per-family accuracy
   (:mod:`tjev.eval.v15`);
2. the one-knob policy sweeps (noul commit margin, noul T, score T) on the validation
   split (dev) and the held-out generators;
3. each knob chosen on the validation split by the preset-A composite with Speed and
   Cost fixed (minimise 1/I + 1/C), and the combined policy scored on both splits;
4. the calibration pool: per-type temperatures fitted on held-out *sources* instead of
   the in-distribution calibration split, leave-one-source-out over the generators (each
   generator scored at T fitted on the ten others; a type the pool lacks keeps the
   in-distribution T), against the in-distribution T and T = 1; and the pool T (all
   generators) applied in distribution (the validation split).

Writes ``OUT/v15-study.json`` and ``OUT/v15-study.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tjev.eval import calibrate as cal
from tjev.eval import predictions as dump
from tjev.eval import v15
from tjev.eval.metrics import label_mask, lookup, summarize, view_probs
from tjev.serve.policy import AbstentionPolicy, ServingPolicy

SPLITS = ("jevbench", "validation", "generators")
COMMIT = ServingPolicy(abstention=AbstentionPolicy(into_noul=True))
KINDS = ("choice", "noul", "score")


def objective(h: dict) -> float:
    """1/I + 1/C: what preset A's harmonic mean depends on when Speed and Cost are fixed."""
    i, c = max(h["I"], 1e-6), max(h["C"], 1e-6)
    return 1 / i + 1 / c


def choose(points: list[dict], knob: str) -> float:
    return min(points, key=objective)[knob]


def tjev_metrics(records: list[dict], temps: dict) -> dict:
    """tjev's own 15-bin ECE / NLL / accuracy (all and per type) at ``temps``."""
    a = dump.arrays(records)
    t = np.asarray([lookup(temps, k, g) for k, g in zip(a["type"], a["lang"], strict=True)])
    probs = view_probs(a["logits"], a["mask"], t)
    mask = label_mask(a["mask"])
    out = {"all": summarize(probs, a["target"], mask)}
    kinds = np.asarray(a["type"])
    for k in KINDS:
        sel = kinds == k
        if sel.any():
            out[k] = summarize(probs[sel], a["target"][sel], mask[sel])
    return out


def loso_pool(records: list[dict], in_dist: dict) -> dict:
    """Leave one generator source out: T fitted on the other sources (types the pool lacks:
    the in-distribution T), the held-out source's records re-scored at it."""
    sources = sorted({r["source"] for r in records})
    rescored, temps_by = [], {}
    for s in sources:
        pool = [r for r in records if r["source"] != s]
        fit = cal.fit_from_predictions(pool)
        temps = in_dist | {k: v for k, v in fit["temperatures"].items()
                           if k in {r["type"] for r in pool}}  # fmt: skip
        temps_by[s] = temps
        rescored += [(r, temps) for r in records if r["source"] == s]
    # one row set: each record at its own fold's temperatures
    rows = [v15.rows_from_predictions([r], temps)[0][0] for r, temps in rescored]
    return {"rows": rows, "temperatures": temps_by}


def per_record_metrics(pairs: list[tuple[dict, dict]]) -> dict:
    """tjev metrics of records each at its own temperatures."""
    probs, targets, masks, kinds = [], [], [], []
    width = max(len(r["labels"]) for r, _ in pairs)
    for r, temps in pairs:
        a = dump.arrays([r])
        t = lookup(temps, r["type"], r.get("lang"))
        p = view_probs(a["logits"], a["mask"], t)[0]
        probs.append(np.pad(p, (0, width - len(p))))
        targets.append(np.pad(a["target"][0], (0, width - len(p))))
        masks.append(np.arange(width) < len(p))
        kinds.append(r["type"])
    probs, targets, masks, kinds = map(np.asarray, (probs, targets, masks, kinds))
    out = {"all": summarize(probs, targets, masks)}
    for k in KINDS:
        sel = kinds == k
        if sel.any():
            out[k] = summarize(probs[sel], targets[sel], masks[sel])
    return out


def study_run(post: Path, tag: str) -> dict:
    art = json.loads((post / f"calibration-jax{tag}.json").read_text())
    in_dist = art["temperatures"]
    out: dict = {"temperatures": in_dist, "splits": {}}
    recs = {}
    for split in SPLITS:
        path = post / "predictions" / f"{split}{tag}.jsonl.gz"
        if not path.exists():
            out["splits"][split] = None
            continue
        recs[split] = dump.read(path)
        rows, skipped = v15.rows_from_predictions(recs[split], in_dist)
        raw, _ = v15.rows_from_predictions(recs[split], {})
        out["splits"][split] = {
            "skipped": skipped,
            "calibrated": v15.report(rows),
            "t1": v15.headline(v15.report(raw, with_families=False)),
            "sweep": v15.sweep(rows),
        }
    dev = out["splits"].get("validation")
    if dev:
        sw = dev["sweep"]
        # sequential: the margin, then the noul T *with that commit on* (alone, a lower noul
        # T only buys what the commit already does), then the score T (independent)
        margin = choose(sw["commit_margin"], "margin")
        val_rows, _ = v15.rows_from_predictions(recs["validation"], in_dist)
        given = [
            {"noul_t": t} | v15.headline(v15.report(val_rows, ServingPolicy(
                abstention=AbstentionPolicy(into_noul=True, margin=margin),
                temperatures={"noul": t}), with_families=False))
            for t in v15.NOUL_TEMPERATURES
        ]  # fmt: skip
        dev["noul_t_given_commit"] = given
        chosen = {
            "margin": margin,
            "noul_t": choose(given, "noul_t"),
            "score_t": choose(sw["score_temperature"], "score_t"),
        }
        commit_only = ServingPolicy(abstention=AbstentionPolicy(into_noul=True, margin=margin))
        combined = ServingPolicy(
            abstention=AbstentionPolicy(into_noul=True, margin=margin),
            temperatures={"noul": chosen["noul_t"], "score": chosen["score_t"]},
        )
        out["chosen"] = chosen
        out["policies"] = {}
        for split, r in recs.items():
            rows, _ = v15.rows_from_predictions(r, in_dist)
            out["policies"][split] = {
                "commit": v15.headline(v15.report(rows, commit_only, with_families=False)),
                "combined": v15.headline(v15.report(rows, combined, with_families=False)),
            }
    if "generators" in recs:
        gen = recs["generators"]
        loso = loso_pool(gen, in_dist)
        in_rows, _ = v15.rows_from_predictions(gen, in_dist)
        t1_rows, _ = v15.rows_from_predictions(gen, {})
        pairs_pool = [(r, loso["temperatures"][r["source"]]) for r in gen]
        out["pool"] = {
            "loso_temperatures": loso["temperatures"],
            "new_source": {
                "t1": {"v15": v15.headline(v15.report(t1_rows, with_families=False)),
                       "tjev": tjev_metrics(gen, {})},
                "in_dist": {"v15": v15.headline(v15.report(in_rows, with_families=False)),
                            "tjev": tjev_metrics(gen, in_dist)},
                "pool_loso": {"v15": v15.headline(v15.report(loso["rows"], with_families=False)),
                              "tjev": per_record_metrics(pairs_pool)},
            },
            "with_commit": {  # the noul commit (margin 0) on top of each temperature
                "in_dist": v15.headline(v15.report(in_rows, COMMIT, with_families=False)),
                "pool_loso": v15.headline(v15.report(loso["rows"], COMMIT, with_families=False)),
            },
        }  # fmt: skip
        full_pool = cal.fit_from_predictions(gen)
        pool_t = in_dist | full_pool["temperatures"]
        out["pool"]["all_generators_t"] = pool_t
        if "validation" in recs:
            val = recs["validation"]
            vrows, _ = v15.rows_from_predictions(val, pool_t)
            out["pool"]["in_distribution"] = {
                "in_dist": tjev_metrics(val, in_dist),
                "pool": tjev_metrics(val, pool_t),
                "pool_v15": v15.headline(v15.report(vrows, with_families=False)),
            }
    return out


# --- tables -----------------------------------------------------------------------------------


def f(x, nd=1) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def markdown(results: dict) -> str:
    lines = ["# JevBench v1.5 on per-item predictions", ""]
    lines += ["## As calibrated (in-distribution per-type T)", "",
              ("| run | split | I (equal) | I choice | I noul | I score | C (equal) | C choice "
               "| C noul | C score | noul abstain | I at T=1 | C at T=1 |"),
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, res in results.items():
        for split, s in res["splits"].items():
            if s is None:
                lines.append(f"| {name} | {split} | not dumped | | | | | | | | | | |")
                continue
            h = v15.headline(s["calibrated"])
            lines.append(
                f"| {name} | {split} | {f(h['I'])} | {f(h['I_choice'])} | {f(h['I_noul'])} "
                f"| {f(h['I_score'])} | {f(h['C'])} | {f(h['C_choice'])} | {f(h['C_noul'])} "
                f"| {f(h['C_score'])} | {f(100 * h['abstain'])} % | {f(s['t1']['I'])} "
                f"| {f(s['t1']['C'])} |"
            )
    lines += ["", "## Per tier (CC)", ""]
    for name, res in results.items():
        for split, s in res["splits"].items():
            if s is None:
                continue
            for kind, t in s["calibrated"]["intelligence"]["types"].items():
                tiers = ", ".join(f"{k} {f(v['cc'])} (n {v['n']})" for k, v in t["tiers"].items())
                lines.append(f"- {name} / {split} / {kind}: {tiers}")
    lines += ["", "## Per family (argmax accuracy / v1.5 accuracy of choice + noul)", ""]
    for name, res in results.items():
        for split, s in res["splits"].items():
            if s is None or split == "validation":  # validation: ~130 families, in the JSON
                continue
            fams = s["calibrated"]["families"]
            lines.append(f"- {name} / {split}: " + "; ".join(
                f"{k} {f(100 * v['accuracy'])}/{f(None if v['v15_accuracy'] is None else 100 * v['v15_accuracy'])} (n {v['n']})"
                for k, v in fams.items()))  # fmt: skip
    lines += ["", "## Sweeps (one knob at a time; I and C equal-type, noul abstention)", ""]
    for name, res in results.items():
        for split, s in res["splits"].items():
            if s is None:
                continue
            lines += [f"### {name}, {split}", "",
                      "| knob | value | I | I noul | I score | C | C noul | C score | abstain |",
                      "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
            for knob, key in (("commit_margin", "margin"), ("noul_temperature", "noul_t"),
                              ("score_temperature", "score_t")):  # fmt: skip
                for p in s["sweep"][knob]:
                    lines.append(
                        f"| {key} | {p[key]} | {f(p['I'])} | {f(p['I_noul'])} "
                        f"| {f(p['I_score'])} | {f(p['C'])} | {f(p['C_noul'])} "
                        f"| {f(p['C_score'])} | {f(100 * p['abstain'])} % |"
                    )
            lines.append("")
    lines += ["## Noul T with the commit on (validation)", "",
              "| run | noul T | I | I noul | C | C noul |", "|---|---|---|---|---|---|"]  # fmt: skip
    for name, res in results.items():
        for p in (res["splits"].get("validation") or {}).get("noul_t_given_commit", []):
            lines.append(f"| {name} | {p['noul_t']} | {f(p['I'])} | {f(p['I_noul'])} "
                         f"| {f(p['C'])} | {f(p['C_noul'])} |")  # fmt: skip
    lines += ["", "## Chosen on the validation split (min 1/I + 1/C), scored", "",
              ("| run | chosen | split | policy | I | I noul | I score | C | C noul | C score "
               "| abstain |"), "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, res in results.items():
        if "chosen" not in res:
            continue
        ch = res["chosen"]
        desc = f"m {ch['margin']}, T_noul {ch['noul_t']}, T_score {ch['score_t']}"
        for split, pol in res["policies"].items():
            base = v15.headline(res["splits"][split]["calibrated"])
            for pname, h in (("none", base), ("commit", pol["commit"]),
                             ("combined", pol["combined"])):  # fmt: skip
                lines.append(
                    f"| {name} | {desc} | {split} | {pname} | {f(h['I'])} | {f(h['I_noul'])} "
                    f"| {f(h['I_score'])} | {f(h['C'])} | {f(h['C_noul'])} "
                    f"| {f(h['C_score'])} | {f(100 * h['abstain'])} % |"
                )
    lines += ["", "## Calibration pool (held-out generators, leave one source out)", "",
              ("| run | T | ECE all | ECE choice | ECE noul | ECE score | NLL | v1.5 C | C noul "
               "| I |"), "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, res in results.items():
        if "pool" not in res:
            continue
        for key, label in (("t1", "T = 1"), ("in_dist", "in-distribution"),
                           ("pool_loso", "pool (LOSO)")):  # fmt: skip
            m = res["pool"]["new_source"][key]
            t, h = m["tjev"], m["v15"]
            lines.append(
                f"| {name} | {label} | {f(t['all']['ece'], 3)} "
                f"| {f(t.get('choice', {}).get('ece'), 3)} | {f(t.get('noul', {}).get('ece'), 3)} "
                f"| {f(t.get('score', {}).get('ece'), 3)} | {f(t['all']['nll'], 3)} "
                f"| {f(h['C'])} | {f(h['C_noul'])} | {f(h['I'])} |"
            )
        for key, label in (("in_dist", "in-distribution + commit"),
                           ("pool_loso", "pool (LOSO) + commit")):  # fmt: skip
            h = res["pool"]["with_commit"][key]
            lines.append(f"| {name} | {label} | | | | | | {f(h['C'])} | {f(h['C_noul'])} "
                         f"| {f(h['I'])} |")  # fmt: skip
        ind = res["pool"].get("in_distribution")
        if ind:
            lines.append(
                f"| {name} | pool T on validation (in-dist cost) | {f(ind['pool']['all']['ece'], 3)} "
                f"(in-dist T {f(ind['in_dist']['all']['ece'], 3)}) | | | | "
                f"{f(ind['pool']['all']['nll'], 3)} (in-dist T {f(ind['in_dist']['all']['nll'], 3)}) "
                f"| {f(ind['pool_v15']['C'])} | | |"
            )
    lines += ["", "Temperatures: in-distribution / pool (all generators):", ""]
    for name, res in results.items():
        pool_t = res.get("pool", {}).get("all_generators_t", {})
        lines.append(f"- {name}: " + ", ".join(
            f"{k} {res['temperatures'][k]:.3f} / {pool_t.get(k, float('nan')):.3f}"
            for k in KINDS))  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="NAME=POST_DIR")
    ap.add_argument("--tag", default="-k2")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    results = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        print(f"[v15-study] {name}", flush=True)
        results[name] = study_run(Path(path), args.tag)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v15-study.json").write_text(json.dumps(results, indent=1, default=str))
    (args.out / "v15-study.md").write_text(markdown(results))
    print(markdown(results))


if __name__ == "__main__":
    main()
