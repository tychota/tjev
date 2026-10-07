"""jev_fr_mt as a mix folder that ``tjev data extend --import-from`` (mix-v7.5) adds.

    python scripts/fr_mt_mix.py --run /content/frmt --out ~/tjev-work/data/frmt-mix

``--run`` is the output of ``scripts/fr_mt_gpu.sh``: train/shard-NN/ (translations of the
train rows of plumb, decider_teacher, mghafiri_scenarios) and held/{validation,calibration,
test}/ (translations of their held-out rows, kept in the same part). Writes OUT/train/
jev_fr_mt.jsonl, OUT/validation.jsonl, OUT/calibration.jsonl, OUT/test/jev_fr_mt.jsonl,
OUT/meta/jev_fr_mt.jsonl.gz (per row: base id and source, MT / QE models, chrF, CometKiwi,
checks) and OUT/manifest.json (block jev, keep rates, score distributions, licences).

Each row keeps its base source's ``license_use`` (commercial); ``license`` names the upstream
licence and the MT model's. The QE model (CometKiwi, CC BY-NC-SA 4.0) only scored the rows.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from tjev.data.item import parse_item, write_jsonl
from tjev.data.licences import licence_of, license_use_of

SOURCE = "jev_fr_mt"


def _q(xs: list[float]) -> dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {}
    return {q: round(xs[int(q / 100 * (len(xs) - 1))], 3) for q in (5, 10, 25, 50, 75, 90, 95)}


def load(run: Path) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    for d in sorted(run.glob("shard-*")) if run.is_dir() else []:
        if not (d / "stats.json").exists():
            raise SystemExit(f"{d}: not finished (no stats.json)")
        accepted += [json.loads(x) for x in (d / "translated.jsonl").open(encoding="utf-8")]
        rejected += [json.loads(x) for x in (d / "rejected.jsonl").open(encoding="utf-8")]
    return accepted, rejected


def to_item_row(r: dict) -> dict:
    base = r["meta"]["base_source"]
    row = {k: v for k, v in r.items() if k != "meta"}
    row["license"] = r["license_use"]  # the descriptive licence string of fr_mt
    row["license_use"] = license_use_of(base)
    parse_item(row)  # fail closed
    return row


def summary(accepted: list[dict], rejected: list[dict], stats: list[dict]) -> dict:
    by = {}
    for r in accepted:
        by.setdefault(r["meta"]["base_source"], Counter())["kept"] += 1
    for r in rejected:
        c = by.setdefault(r["base_source"], Counter())
        c["skipped_english_specific" if "chrf" not in r else "rejected"] += 1
    rates = {
        s: {**c, "keep_rate_of_gated": round(c["kept"] / max(1, c["kept"] + c["rejected"]), 3)}
        for s, c in sorted(by.items())
    }
    gated = [r["meta"] for r in accepted] + [r for r in rejected if "chrf" in r]
    checks = Counter()
    for g in gated:
        for k, v in g.get("checks", {}).items():
            checks[k] += not v
    return {
        "rows": len(accepted) + len(rejected),
        "kept": len(accepted),
        "by_base_source": rates,
        "failed_check (of gated rows)": dict(checks),
        "chrf_kept": _q([r["meta"]["chrf"] for r in accepted]),
        "chrf_all_gated": _q([g["chrf"] for g in gated]),
        "kiwi_kept": _q([r["meta"].get("kiwi") for r in accepted]),
        "kiwi_all_gated": _q([g.get("kiwi") for g in gated]),
        "kiwi_worst_segment_kept": _q([r["meta"].get("kiwi_worst_segment") for r in accepted]),
        "wall_seconds": round(sum(s.get("wall_seconds", 0) for s in stats), 1),
        "model_load_seconds": round(sum(s.get("model_load_seconds", 0) for s in stats), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run, out = Path(args.run).expanduser(), Path(args.out).expanduser()
    train, rejected = load(run / "train")
    stats = [
        json.loads((d / "stats.json").read_text()) for d in sorted((run / "train").glob("shard-*"))
    ]
    rows = [to_item_row(r) for r in train]
    write_jsonl(out / "train" / f"{SOURCE}.jsonl", [parse_item(r) for r in rows])
    meta = [{"id": r["id"], **r["meta"]} for r in train]
    held = {}
    for part in ("validation", "calibration", "test"):
        d = run / "held" / part
        got = [json.loads(x) for x in (d / "translated.jsonl").open(encoding="utf-8")]
        held[part] = [to_item_row(r) for r in got]
        meta += [{"id": r["id"], "part": part, **r["meta"]} for r in got]
    write_jsonl(out / "validation.jsonl", [parse_item(r) for r in held["validation"]])
    write_jsonl(out / "calibration.jsonl", [parse_item(r) for r in held["calibration"]])
    write_jsonl(out / "test" / f"{SOURCE}.jsonl", [parse_item(r) for r in held["test"]])
    (out / "meta").mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "meta" / f"{SOURCE}.jsonl.gz", "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in meta)
    first = train[0]["meta"] if train else {}
    card, redistributable, note = licence_of(SOURCE)
    uses = Counter(r["license_use"] for r in rows)
    report = summary(train, rejected, stats)
    entry = {
        "block": "jev", "train": len(rows), "train_built": len(rows),
        **{p: len(v) for p, v in held.items()},
        "lang": {"fr": len(rows)}, "family": "translated", "repo": "derived (gen_v7.fr_mt)",
        "license": card, "card_licence": card, "redistributable": redistributable,
        "licence_note": note, "license_use": max(uses, key=uses.get) if uses else "commercial",
        "commercial_ok": True,
        "mt_model": first.get("mt_model"), "mt_licence": "CC-BY-4.0",
        "qe_model": first.get("qe_model"), "qe_licence": "CC-BY-NC-SA-4.0 (scores only)",
        "qe": first.get("qe"), "min_kiwi": first.get("min_kiwi"),
        "min_kiwi_segment": first.get("min_kiwi_segment"),
        "base_sources": ["plumb", "decider_teacher", "mghafiri_scenarios"],
        "translation": report, "drops": {},
    }  # fmt: skip
    manifest = {"sources": {SOURCE: entry}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({"train": len(rows), **{p: len(v) for p, v in held.items()}, **report},
                     indent=1, ensure_ascii=False))  # fmt: skip


if __name__ == "__main__":
    main()
