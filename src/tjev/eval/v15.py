"""JevBench v1.5 scoring (the frozen ``docs/METHOD-v1.5.md`` and the equal-types headline
addendum) on per-item predictions, and the serving policies' effect on it.

The harness (pinned in :mod:`tjev.eval.official`) has no v1.5 scorer; this is tjev's
implementation, shared by ``tjev eval-official`` (harness records) and ``tjev eval-v15``
(``tjev post`` prediction dumps, :mod:`tjev.eval.predictions`). Per item (§3.1):

- **choice**: correct iff the argmax (the harness's: ties to the smallest label) is gold;
  chance 1 / options.
- **noul**: P(yes) <= 0.2 is No, >= 0.8 is Yes, anything between is an abstention,
  counted wrong; chance 0.5.
- **score**: nMAE = |E[level] − gold| / (K − 1) (the expected level of the returned
  distribution); chance nMAE = the uniform guess's, mean over levels of |l − gold| / (K − 1).

A failed or invalid answer is wrong (score: the worst level's nMAE). Per type × tier:
CC = 100 · (acc − c̄) / (1 − c̄) (score: 100 · (1 − mean nMAE / mean nMAE_chance)), not
clipped; per type: tiers weighted easy 10 / standard 20 / judge 30 / hard 40, renormalised
over the tiers present (items with no known tier form one tier, ``all``); Intelligence:
the types' CC with equal weights (headline) or 50/25/25 (the frozen option B). The public
items are the open set only: I = 0.5 · I_open + 0.5 · I_sealed needs the sealed half.

Calibration (§5, "typed calibration as in the draft"; the draft that maps a part to
points is not published, so each part is scored as in v1.3, ECE → 100 · (1 − ECE/0.5) =
100 − 200 · ECE, TVD or RPS → 100 · (1 − x), and a type's parts are averaged); pooled over
tiers, harness ECE (10 equal-width bins):

- **choice**: top-label ECE and the mean TVD to the gold distributions of the items that
  have one (JevBench: 10 hard items); ``ece_hard`` (the v1.3 hard-tier-only ECE) is reported
  beside it, not scored;
- **noul**: ECE of P(yes) against the yes rate (Brier reported);
- **score**: normalised RPS and top-level ECE.

docs/official-eval.md#v15-scoring.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from tjev.serve.policy import AbstentionPolicy, ServingPolicy, as_policy

TIER_WEIGHTS = {"easy": 0.10, "standard": 0.20, "judge": 0.30, "hard": 0.40}
TYPE_WEIGHTS = {  # headline (the equal-types addendum) and the frozen option B
    "equal": {"choice": 1 / 3, "noul": 1 / 3, "score": 1 / 3},
    "frozen": {"choice": 0.50, "noul": 0.25, "score": 0.25},
}
NOUL_BAND = (0.20, 0.80)
UNTIERED = "all"
# tjev's JevBench sources (tjev.data.jevbench) -> v1.5 tiers ("original" is standard)
SOURCE_TIERS = {"jevbench_easy": "easy", "jevbench_original": "standard", "jevbench_hard": "hard"}


@dataclass(frozen=True)
class Row:
    """One answered (or failed: ``probs`` None) item; ``probs`` in ``labels`` order."""

    id: str
    type: str
    labels: tuple[str, ...]
    gold: str
    probs: np.ndarray | None
    tier: str = UNTIERED
    family: str = "unknown"
    source: str = ""
    gold_probs: np.ndarray | None = None  # a gold distribution (choice TVD), if any

    def dist(self) -> dict[str, float]:
        assert self.probs is not None
        return {label: float(p) for label, p in zip(self.labels, self.probs, strict=True)}


# --- per item -------------------------------------------------------------------------------


def argmax_label(probs: dict[str, float]) -> str:
    """The harness's argmax (``jevbench.scoring.argmax_label``): ties to the smallest label."""
    best, best_p = "", -1.0
    for k in sorted(probs):
        if probs[k] > best_p:
            best, best_p = k, probs[k]
    return best


def score_item(
    kind: str,
    labels: Sequence[str],
    gold: str,
    probs: dict[str, float] | None,
    band: tuple[float, float] = NOUL_BAND,
) -> dict[str, Any]:
    """One item under v1.5 §3.1 (``probs`` None: a failed or invalid answer, wrong)."""
    out: dict[str, Any] = {"type": kind}
    if kind == "choice":
        correct = probs is not None and argmax_label(probs) == gold
        return out | {"correct": correct, "chance": 1 / len(labels)}
    if kind == "noul":
        p = None if probs is None else float(probs["yes"])
        yes = p is not None and p >= band[1]
        no = p is not None and p <= band[0]
        return out | {
            "correct": (yes and gold == "yes") or (no and gold == "no"),
            "abstained": p is not None and not (yes or no),
            "p_yes": p,
        }
    k, g = len(labels), int(gold)
    if probs is None:
        nmae = max(g, k - 1 - g) / (k - 1)
    else:
        ev = sum(float(label) * float(v) for label, v in probs.items())
        nmae = abs(ev - g) / (k - 1)
    return out | {
        "nmae": nmae,
        "nmae_chance": sum(abs(level - g) for level in range(k)) / (k * (k - 1)),
    }


# --- Intelligence ---------------------------------------------------------------------------


def cc(items: list[dict]) -> float | None:
    """Chance-corrected competence of one type × tier cell (not clipped)."""
    if not items:
        return None
    kind = items[0]["type"]
    if kind == "score":
        chance = float(np.mean([i["nmae_chance"] for i in items]))
        return 100 * (1 - float(np.mean([i["nmae"] for i in items])) / chance)
    acc = float(np.mean([i["correct"] for i in items]))
    chance = 0.5 if kind == "noul" else float(np.mean([i["chance"] for i in items]))
    return 100 * (acc - chance) / (1 - chance)


def weighted(values: dict[str, Any], weights: dict[str, float]) -> float | None:
    """The weighted mean of the non-None values with a weight, renormalised over them."""
    present = {k: v for k, v in values.items() if v is not None and weights.get(k)}
    if not present:
        return None
    return sum(weights[k] * v for k, v in present.items()) / sum(weights[k] for k in present)


def tier_weights(tiers: Iterable[str]) -> dict[str, float]:
    """v1.5's tier weights; an untiered set (no known tier) is one tier of weight 1."""
    tiers = set(tiers)
    if tiers & set(TIER_WEIGHTS):
        return TIER_WEIGHTS
    return dict.fromkeys(tiers, 1.0)


def intelligence(scored: list[tuple[str, dict]]) -> dict:
    """Per type × tier CC, per-type CC (tier weights 10/20/30/40, renormalised over the
    tiers that have items) and I_open on these items under both type weightings."""
    cells: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for tier, item in scored:
        cells[item["type"]][tier].append(item)
    per_type: dict[str, dict[str, Any]] = {}
    for kind, tiers in sorted(cells.items()):
        by_tier = {tier: cc(items) for tier, items in tiers.items()}
        per_type[kind] = {
            "cc": weighted(by_tier, tier_weights(by_tier)),
            "tiers": {t: {"n": len(tiers[t]), "cc": by_tier[t]} for t in sorted(tiers)},
            "n": sum(len(v) for v in tiers.values()),
        }
        items = [i for v in tiers.values() for i in v]
        if kind != "score":
            per_type[kind]["accuracy"] = float(np.mean([i["correct"] for i in items]))
        else:
            per_type[kind]["nmae"] = float(np.mean([i["nmae"] for i in items]))
        if kind == "noul":
            per_type[kind]["abstentions"] = sum(i["abstained"] for i in items)
            per_type[kind]["abstention_rate"] = per_type[kind]["abstentions"] / len(items)
    by_type = {k: v["cc"] for k, v in per_type.items()}
    return {"types": per_type} | {
        f"intelligence_{name}": weighted(by_type, w) for name, w in TYPE_WEIGHTS.items()
    }


# --- Calibration ----------------------------------------------------------------------------


def ece(conf: Sequence[float], hit: Sequence[float], bins: int = 10) -> float | None:
    """The harness's ECE (``jevbench.metrics.ece_top_label``: 10 equal-width bins, the last
    one closed)."""
    if not len(conf):
        return None
    sums: dict[int, list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    for c, h in zip(conf, hit, strict=True):
        b = sums[min(int(min(max(c, 0.0), 1.0) * bins), bins - 1)]
        b[0] += 1
        b[1] += c
        b[2] += h
    return sum(n / len(conf) * abs(h / n - c / n) for n, c, h in sums.values())


def tvd(p: dict, q: dict, labels: Sequence[str]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in labels)


def rps(probs: dict, gold: int, k: int) -> float:
    """Ranked probability score over k ordered levels, normalised to [0, 1]."""
    cdf = np.cumsum([probs[str(level)] for level in range(k)])
    obs = np.cumsum([1.0 if level == gold else 0.0 for level in range(k)])
    return float(np.sum((cdf[:-1] - obs[:-1]) ** 2) / (k - 1))


def ece_points(value: float | None) -> float | None:
    """ECE → points as in v1.3: 100 · (1 − ECE / 0.5), floored at 0."""
    return None if value is None else max(0.0, 100 * (1 - value / 0.5))


def calibration(rows: list[Row]) -> dict:
    """Typed calibration (v1.5 §5) on the rows with a distribution (module docstring)."""
    by: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r.probs is not None:
            by[r.type].append(r)
    out: dict[str, Any] = {}
    for kind, sel in sorted(by.items()):
        dists = [r.dist() for r in sel]
        conf = [float(max(p.values())) for p in dists]
        hit = [float(argmax_label(p) == r.gold) for r, p in zip(sel, dists, strict=True)]
        part: dict[str, Any] = {"n": len(sel)}
        if kind == "choice":
            part["ece"] = ece(conf, hit)
            hard = [(c, h) for r, c, h in zip(sel, conf, hit, strict=True) if r.tier == "hard"]
            part["ece_hard"] = ece(*zip(*hard, strict=True)) if hard else None
            gold = [(r, p) for r, p in zip(sel, dists, strict=True) if r.gold_probs is not None]
            part["tvd_n"] = len(gold)
            part["tvd"] = None
            if gold:
                part["tvd"] = float(np.mean([
                    tvd(p, dict(zip(r.labels, map(float, r.gold_probs), strict=True)), r.labels)
                    for r, p in gold]))  # fmt: skip
            points = [ece_points(part["ece"])]
            points += [] if part["tvd"] is None else [100 * (1 - part["tvd"])]
        elif kind == "noul":
            p_yes = [p["yes"] for p in dists]
            truth = [float(r.gold == "yes") for r in sel]
            part["ece_p_yes"] = ece(p_yes, truth)
            part["brier"] = float(
                np.mean([2 * (p - y) ** 2 for p, y in zip(p_yes, truth, strict=True)])
            )
            points = [ece_points(part["ece_p_yes"])]
        else:
            part["rps"] = float(np.mean([rps(p, int(r.gold), len(r.labels))
                                         for r, p in zip(sel, dists, strict=True)]))  # fmt: skip
            part["ece"] = ece(conf, hit)
            points = [100 * (1 - part["rps"]), ece_points(part["ece"])]
        part["points"] = float(np.mean([x for x in points if x is not None]))
        out[kind] = part
    points = {k: v["points"] for k, v in out.items()}
    return {"types": out} | {
        f"calibration_{name}": weighted(points, w) for name, w in TYPE_WEIGHTS.items()
    }


# --- serving policies -----------------------------------------------------------------------


def apply_policy(rows: list[Row], policy: ServingPolicy | AbstentionPolicy | None) -> list[Row]:
    """What the JevBench TypeSafe adapter would score under a serving ``policy``: the
    calibrated distributions sharpened at the per-type serving temperature, and a noul
    P(yes) committed when the abstention policy reports it as ``noul`` (``into_noul``;
    ``field`` mode does not change what is scored)."""
    policy = as_policy(policy)
    if policy is None:
        return rows
    out = []
    for r in rows:
        if r.probs is None:
            out.append(r)
            continue
        p = policy.serve(r.type, r.probs)
        a = policy.abstention
        if r.type == "noul" and a is not None and a.into_noul:
            yes = r.labels.index("yes")
            committed = a.commit(float(p[yes]))
            p = np.full(2, 1 - committed)
            p[yes] = committed
        out.append(replace(r, probs=p))
    return out


def families(rows: list[Row], band: tuple[float, float] = NOUL_BAND) -> dict:
    """Per family: argmax accuracy (the harness's per-family number; score by its argmax
    level) and the v1.5 accuracy of its choice and noul items (noul: the band rule)."""
    by: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by[r.family].append(r)
    out = {}
    for fam, sel in sorted(by.items()):
        argmax = [r.probs is not None and argmax_label(r.dist()) == r.gold for r in sel]
        v15 = [score_item(r.type, r.labels, r.gold, r.dist() if r.probs is not None else None,
                          band)["correct"] for r in sel if r.type != "score"]  # fmt: skip
        out[fam] = {"n": len(sel), "accuracy": float(np.mean(argmax)),
                    "v15_accuracy": float(np.mean(v15)) if v15 else None}  # fmt: skip
    return out


def report(
    rows: list[Row],
    policy: ServingPolicy | AbstentionPolicy | None = None,
    *,
    with_families: bool = True,
) -> dict:
    """v1.5 Intelligence (per type × tier) and Calibration (per type) of ``rows`` as served
    under ``policy``, the noul abstention rate and the per-family accuracy."""
    served = apply_policy(rows, policy)
    scored = [(r.tier, score_item(r.type, r.labels, r.gold,
                                  r.dist() if r.probs is not None else None)) for r in served]  # fmt: skip
    intel = intelligence(scored)
    noul = intel["types"].get("noul", {})
    out = {
        "n": len(rows),
        "policy": None if as_policy(policy) is None else as_policy(policy).describe(),  # type: ignore[union-attr]
        "intelligence": intel,
        "calibration": calibration(served),
        "noul_abstention_rate": noul.get("abstention_rate"),
    }
    if with_families:
        out["families"] = families(served)
    return out


def headline(rep: dict) -> dict:
    """The flat numbers of a :func:`report` (for tables and sweeps)."""
    intel, cal = rep["intelligence"], rep["calibration"]
    flat: dict[str, Any] = {
        "I": intel["intelligence_equal"],
        "C": cal["calibration_equal"],
        "abstain": rep["noul_abstention_rate"],
    }
    for kind in ("choice", "noul", "score"):
        flat[f"I_{kind}"] = intel["types"].get(kind, {}).get("cc")
        flat[f"C_{kind}"] = cal["types"].get(kind, {}).get("points")
    return flat


# --- prediction dumps -----------------------------------------------------------------------


def rows_from_predictions(
    records: list[dict],
    temperatures: dict[str, float] | None = None,
    *,
    gold_probs: dict[str, dict[str, float]] | None = None,
    tiers: dict[str, str] | None = None,
) -> tuple[list[Row], dict]:
    """Rows of a ``tjev post`` prediction dump. ``temperatures``: a calibration artifact's
    (re-applied to the raw logits, order-averaged as served); None: the dump's own
    ``probs`` (the temperatures its report applied). Gold: the argmax of the target; an
    item whose target has a tied maximum (no single gold label) is skipped. A soft target
    (max < 1) is also the item's gold distribution (choice TVD); ``gold_probs`` (by id,
    e.g. the JevBench hard items' published distributions) overrides it. Tier: ``tiers``
    by id, else a JevBench source's tier, else ``all``. Returns (rows, skipped counts)."""
    from tjev.eval.metrics import lookup, view_probs

    rows, skipped = [], defaultdict(int)
    for rec in records:
        labels = tuple(rec["labels"])
        target = np.asarray(rec["target"], np.float64)
        top = np.flatnonzero(target >= target.max() - 1e-9)
        if len(top) != 1:
            skipped["tied_gold"] += 1
            continue
        if temperatures is None:
            probs = np.asarray(rec["probs"], np.float64)
        else:
            z = np.asarray(rec["logits"], np.float64)[None]  # [1,V,K]
            t = lookup(temperatures, rec["type"], rec.get("lang"))
            probs = view_probs(z, np.ones_like(z, bool), t)[0]
        gp = None
        if gold_probs and rec["id"] in gold_probs:
            gp = np.asarray([float(gold_probs[rec["id"]].get(k, 0.0)) for k in labels])
        elif target.max() < 1 - 1e-9:
            gp = target
        tier = (
            (tiers or {}).get(rec["id"])
            or rec.get("tier")
            or SOURCE_TIERS.get(rec.get("source", ""), UNTIERED)
        )
        rows.append(Row(id=rec["id"], type=rec["type"], labels=labels, gold=labels[int(top[0])],
                        probs=probs / probs.sum(), tier=tier, family=rec.get("family") or "unknown",
                        source=rec.get("source", ""), gold_probs=gp))  # fmt: skip
    return rows, dict(skipped)


# --- the policy sweep -----------------------------------------------------------------------

MARGINS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)  # 0.3: never commit (the band's half-width)
NOUL_TEMPERATURES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SCORE_TEMPERATURES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def sweep(
    rows: list[Row],
    *,
    margins: Sequence[float] = MARGINS,
    noul_temperatures: Sequence[float] = NOUL_TEMPERATURES,
    score_temperatures: Sequence[float] = SCORE_TEMPERATURES,
) -> dict[str, list[dict]]:
    """One policy knob at a time (the others off), each point's :func:`headline`:
    the noul commit margin (``margin`` 0.3 = off), the noul and the score temperature."""

    def point(policy: ServingPolicy | None, **knob: float) -> dict:
        return knob | headline(report(rows, policy, with_families=False))

    commit = [point(ServingPolicy(abstention=AbstentionPolicy(into_noul=True, margin=m)),
                    margin=m) for m in margins]  # fmt: skip
    noul_t = [point(ServingPolicy(temperatures={"noul": t}), noul_t=t) for t in noul_temperatures]
    score_t = [point(ServingPolicy(temperatures={"score": t}), score_t=t)
               for t in score_temperatures]  # fmt: skip
    return {"commit_margin": commit, "noul_temperature": noul_t, "score_temperature": score_t}


# --- whole dumps ----------------------------------------------------------------------------


def harness_gold_probs(root: str | Path) -> dict[str, dict[str, float]]:
    """The published gold distributions of the JevBench public items (by id), from a
    harness checkout's ``datasets/public/*.jsonl`` (``provenance.gold_probs``)."""
    out = {}
    for path in sorted((Path(root) / "datasets" / "public").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw = json.loads(line)
                if gold := (raw.get("provenance") or {}).get("gold_probs"):
                    out[raw["id"]] = gold
    return out


def score_dumps(
    paths: Sequence[str | Path],
    *,
    temperatures: dict[str, float] | None = None,
    policy: ServingPolicy | AbstentionPolicy | None = None,
    gold_probs: dict[str, dict[str, float]] | None = None,
    with_sweep: bool = False,
) -> dict:
    """v1.5 reports of prediction dumps (one per file): as calibrated (``temperatures``,
    else the dump's own probabilities), under ``policy`` if given, and the policy
    :func:`sweep` if ``with_sweep``."""
    from tjev.eval.predictions import read

    out: dict[str, Any] = {}
    for path in map(Path, paths):
        rows, skipped = rows_from_predictions(read(path), temperatures, gold_probs=gold_probs)
        rep: dict[str, Any] = {"file": str(path), "skipped": skipped,
                               "calibrated": report(rows)}  # fmt: skip
        if as_policy(policy) is not None:
            rep["policy"] = report(rows, policy)
        if with_sweep:
            rep["sweep"] = sweep(rows)
        out[path.name.removesuffix(".jsonl.gz")] = rep
    return out
