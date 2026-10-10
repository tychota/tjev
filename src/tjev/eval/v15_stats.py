"""Statistical rigour for the v1.5 numbers (:mod:`tjev.eval.v15`): item-bootstrap confidence
intervals, paired comparisons of two runs on the same items, and calibration diagnostics
beyond the scored ECE.

**Bootstrap.** Items are resampled with replacement; a resample is a multiplicity vector
``w`` (multinomial, sums to N), and every headline number of :func:`v15.headline` (I, C,
the per-type I and C, the noul abstention rate) is recomputed as a ``w``-weighted aggregate
of per-item quantities (vectorised over resamples; ``w = 1`` reproduces :func:`v15.report`
exactly). The interval is the 95% percentile interval. Tiers and types stay as in the full
set (a resample that loses a whole cell renormalises over the cells present, as the scorer
does). A paired comparison draws one ``w`` per resample for both runs, on the items present in
both (matched by id), so the difference's interval accounts for the shared item draw; it
also reports the fraction of resamples where A > B.

**Calibration diagnostics** (top-label confidence vs. correctness of the argmax):

- smooth ECE (Błasiok & Nakkiran 2024, "Smooth ECE: principled reliability diagrams via
  kernel smoothing"): the integral of the absolute residual ``y − f`` smoothed by a Gaussian
  kernel (bandwidth 0.05, reflected at 0 and 1) under the kernel density of ``f``, on a grid
  (here at a fixed bandwidth, not the paper's self-consistent one);
- adaptive ECE: 15 equal-mass bins;
- the Brier score of top-label confidence and its Murphy decomposition over the same 15
  bins: reliability − resolution + uncertainty (equal to the Brier score up to the
  within-bin variance of the confidence);
- AURC: the mean over coverages k/N of the error rate of the k most confident items (and
  E-AURC, its excess over an oracle ordering);
- RPS (normalised, :func:`v15.rps`) for score items.

docs/official-eval.md#statistical-rigour.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tjev.eval import v15
from tjev.serve.policy import AbstentionPolicy, ServingPolicy

TYPES = ("choice", "noul", "score")
METRICS = ("I", "C", "abstain", *(f"{p}_{k}" for p in ("I", "C") for k in TYPES))
ECE_BINS = 10  # the harness's
LEVEL = 0.95
CHUNK = 128  # resamples per vectorised block

Policy = ServingPolicy | AbstentionPolicy | None


# --- per-item quantities ---------------------------------------------------------------------


@dataclass
class Items:
    """Per-item arrays of a set of (served) rows, enough to recompute :func:`v15.headline`
    as weighted aggregates."""

    ids: list[str]
    type: np.ndarray  # str
    tier: np.ndarray  # str
    x: np.ndarray  # choice/noul: correct; score: nMAE
    chance: np.ndarray  # choice: 1/options; noul: 0.5; score: nMAE of the uniform guess
    abstained: np.ndarray
    has_probs: np.ndarray
    cal_conf: np.ndarray  # the ECE's confidence (choice/score: top label; noul: P(yes))
    cal_hit: np.ndarray  # and its outcome (choice/score: argmax right; noul: gold is yes)
    extra: np.ndarray  # choice: TVD to the gold distribution; score: RPS
    has_extra: np.ndarray
    tier_weights: dict[str, dict[str, float]]

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, idx: np.ndarray) -> Items:
        arrays = {k: getattr(self, k)[idx] for k in ("type", "tier", "x", "chance", "abstained",
                                                     "has_probs", "cal_conf", "cal_hit", "extra",
                                                     "has_extra")}  # fmt: skip
        return Items(ids=[self.ids[i] for i in idx], tier_weights=self.tier_weights, **arrays)


def items(rows: list[v15.Row], policy: Policy = None) -> Items:
    """The per-item quantities of ``rows`` as served under ``policy``."""
    served = v15.apply_policy(rows, policy)
    n = len(served)
    out: dict[str, Any] = {
        "x": np.zeros(n), "chance": np.zeros(n), "abstained": np.zeros(n),
        "has_probs": np.zeros(n, bool), "cal_conf": np.zeros(n), "cal_hit": np.zeros(n),
        "extra": np.zeros(n), "has_extra": np.zeros(n, bool),
    }  # fmt: skip
    tiers: dict[str, set[str]] = defaultdict(set)
    for i, r in enumerate(served):
        dist = r.dist() if r.probs is not None else None
        s = v15.score_item(r.type, r.labels, r.gold, dist)
        tiers[r.type].add(r.tier)
        if r.type == "score":
            out["x"][i], out["chance"][i] = s["nmae"], s["nmae_chance"]
        else:
            out["x"][i] = float(s["correct"])
            out["chance"][i] = 0.5 if r.type == "noul" else s["chance"]
            out["abstained"][i] = float(s.get("abstained", False))
        if dist is None:
            continue
        out["has_probs"][i] = True
        if r.type == "noul":
            out["cal_conf"][i], out["cal_hit"][i] = dist["yes"], float(r.gold == "yes")
            continue
        out["cal_conf"][i] = max(dist.values())
        out["cal_hit"][i] = float(v15.argmax_label(dist) == r.gold)
        if r.type == "choice" and r.gold_probs is not None:
            gold = dict(zip(r.labels, map(float, r.gold_probs), strict=True))
            out["extra"][i], out["has_extra"][i] = v15.tvd(dist, gold, r.labels), True
        elif r.type == "score":
            out["extra"][i] = v15.rps(dist, int(r.gold), len(r.labels))
            out["has_extra"][i] = True
    return Items(
        ids=[r.id for r in served],
        type=np.asarray([r.type for r in served], dtype=object),
        tier=np.asarray([r.tier for r in served], dtype=object),
        tier_weights={k: dict(v15.tier_weights(t)) for k, t in tiers.items()},
        **out,
    )


# --- weighted headline -----------------------------------------------------------------------


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _mean_present(values: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Per resample: the weighted mean of the non-NaN values (NaN if none)."""
    v = np.stack(values)  # [parts, B]
    w = np.asarray(weights)[:, None] * ~np.isnan(v)
    return _ratio(np.nansum(np.nan_to_num(v) * w, 0), w.sum(0))


def weighted_headline(it: Items, w: np.ndarray) -> dict[str, np.ndarray]:
    """:func:`v15.headline` of ``it`` with item multiplicities ``w`` [B, N], per resample."""
    w = np.atleast_2d(np.asarray(w, np.float64))
    out: dict[str, np.ndarray] = {}
    i_types, c_types = [], []
    for kind in TYPES:
        m = it.type == kind
        nan = np.full(len(w), np.nan)
        if not m.any():
            out[f"I_{kind}"] = out[f"C_{kind}"] = nan
            i_types.append(nan)
            c_types.append(nan)
            continue
        # Intelligence: CC per tier, tier-weighted over the tiers present
        cells, tw = [], []
        for tier, weight in it.tier_weights[kind].items():
            cell = m & (it.tier == tier)
            if not cell.any() or not weight:
                continue
            sw = w[:, cell].sum(1)
            mean_x = _ratio(w[:, cell] @ it.x[cell], sw)
            mean_c = _ratio(w[:, cell] @ it.chance[cell], sw)
            cells.append(100 * (1 - mean_x / mean_c) if kind == "score"
                         else 100 * (mean_x - mean_c) / (1 - mean_c))  # fmt: skip
            tw.append(weight)
        out[f"I_{kind}"] = _mean_present(cells, tw) if cells else nan
        if kind == "noul":
            out["abstain"] = _ratio(w[:, m] @ it.abstained[m], w[:, m].sum(1))
        # Calibration: harness ECE (10 equal-width bins) and the type's other part
        c = m & it.has_probs
        if not c.any():
            out[f"C_{kind}"] = nan
            i_types.append(out[f"I_{kind}"])
            c_types.append(nan)
            continue
        conf, hit = it.cal_conf[c], it.cal_hit[c]
        wc = w[:, c]
        n = wc.sum(1)
        bins = np.minimum((np.clip(conf, 0.0, 1.0) * ECE_BINS).astype(int), ECE_BINS - 1)
        gap = np.zeros(len(w))
        for b in np.unique(bins):
            sel = bins == b
            gap += np.abs(wc[:, sel] @ (hit[sel] - conf[sel]))
        ece_pts = np.maximum(0.0, 100 * (1 - _ratio(gap, n) / 0.5))
        parts = [ece_pts]
        if kind != "noul":
            e = it.has_extra[c]
            if e.any():
                mean_extra = _ratio(wc[:, e] @ it.extra[c][e], wc[:, e].sum(1))
                parts.append(100 * (1 - mean_extra))
        out[f"C_{kind}"] = _mean_present(parts, [1.0] * len(parts))
        out[f"C_{kind}"] = np.where(n > 0, out[f"C_{kind}"], np.nan)
        i_types.append(out[f"I_{kind}"])
        c_types.append(out[f"C_{kind}"])
    out.setdefault("abstain", np.full(len(w), np.nan))
    out["I"] = _mean_present(i_types, [1.0] * 3)
    out["C"] = _mean_present(c_types, [1.0] * 3)
    return {k: out[k] for k in METRICS}


def _weights(rng: np.random.Generator, n_items: int, n: int):
    """Resample multiplicities, ``CHUNK`` resamples at a time."""
    for start in range(0, n, CHUNK):
        yield rng.multinomial(n_items, np.full(n_items, 1 / n_items), size=min(CHUNK, n - start))


def _scalar(x: float) -> float | None:
    return None if np.isnan(x) else float(x)


def _interval(samples: np.ndarray, level: float = LEVEL) -> tuple[float | None, float | None]:
    ok = samples[~np.isnan(samples)]
    if not len(ok):
        return None, None
    lo, hi = np.percentile(ok, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def bootstrap(it: Items, n: int = 2000, seed: int = 0) -> dict[str, dict]:
    """Per headline metric: the point estimate, the bootstrap standard error and bias (mean
    of the resamples − point), the 95% percentile interval (``lo``, ``hi``) over ``n`` item
    resamples. The binned ECE inside C is biased upward under resampling (duplicated items
    add noise to every bin), so C's resamples sit low: on a large set its percentile
    interval can exclude the point (``bias`` says by how much). Compare runs with
    :func:`paired`, where the shift is shared by both sides."""
    point = weighted_headline(it, np.ones((1, len(it))))
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for w in _weights(rng, len(it), n):
        for k, v in weighted_headline(it, w).items():
            draws[k].append(v)
    out = {}
    for k in METRICS:
        s = np.concatenate(draws[k])
        lo, hi = _interval(s)
        valid = (~np.isnan(s)).sum() > 1
        out[k] = {"point": _scalar(point[k][0]), "lo": lo, "hi": hi,
                  "se": _scalar(np.nanstd(s, ddof=1)) if valid else None,
                  "bias": _scalar(np.nanmean(s) - point[k][0]) if valid else None}  # fmt: skip
    return out


def match(a: Items, b: Items) -> tuple[Items, Items, dict[str, int]]:
    """The items of ``a`` and ``b`` with the same id, in ``a``'s order (a repeated id: its
    first occurrence), and the counts dropped from each side."""
    first_b: dict[str, int] = {}
    for i, k in enumerate(b.ids):
        first_b.setdefault(k, i)
    seen: set[str] = set()
    ia, ib = [], []
    for i, k in enumerate(a.ids):
        if k in first_b and k not in seen:
            ia.append(i)
            ib.append(first_b[k])
        seen.add(k)
    common = len(ia)
    counts = {"n_common": common, "only_a": len(set(a.ids)) - common,
              "only_b": len(first_b) - common,
              "duplicates": (len(a.ids) - len(set(a.ids))) + (len(b.ids) - len(first_b))}  # fmt: skip
    return a.take(np.asarray(ia, int)), b.take(np.asarray(ib, int)), counts


def paired(a: Items, b: Items, n: int = 2000, seed: int = 0) -> dict:
    """A − B on the items present in both (by id), with one item resample shared by both
    sides: per metric, each side's point, the difference, its 95% percentile interval and
    the fraction of resamples where A > B (ties count half)."""
    a, b, counts = match(a, b)
    if not counts["n_common"]:
        return counts | {"metrics": {}}
    ones = np.ones((1, counts["n_common"]))
    pa, pb = weighted_headline(a, ones), weighted_headline(b, ones)
    rng = np.random.default_rng(seed)
    diffs = defaultdict(list)
    for w in _weights(rng, counts["n_common"], n):
        ha, hb = weighted_headline(a, w), weighted_headline(b, w)
        for k in METRICS:
            diffs[k].append(ha[k] - hb[k])
    metrics = {}
    for k in METRICS:
        d = np.concatenate(diffs[k])
        ok = d[~np.isnan(d)]
        lo, hi = _interval(d)
        metrics[k] = {
            "a": _scalar(pa[k][0]), "b": _scalar(pb[k][0]), "diff": _scalar(pa[k][0] - pb[k][0]),
            "lo": lo, "hi": hi,
            "p_a_gt_b": float(np.mean(ok > 0) + 0.5 * np.mean(ok == 0)) if len(ok) else None,
        }  # fmt: skip
    return counts | {"metrics": metrics}


# --- calibration diagnostics ------------------------------------------------------------------


def smooth_ece(conf: Sequence[float], hit: Sequence[float], bandwidth: float = 0.05,
               grid: int = 1001) -> float | None:  # fmt: skip
    """Smooth ECE (Błasiok & Nakkiran 2024) at a fixed bandwidth: ∫ |Σ_i K(t, f_i)(y_i − f_i)|
    dt / ∫ Σ_i K(t, f_i) dt on [0, 1], K a Gaussian reflected at 0 and 1."""
    f = np.clip(np.asarray(conf, np.float64), 0.0, 1.0)
    y = np.asarray(hit, np.float64)
    if not len(f):
        return None
    t = np.linspace(0.0, 1.0, grid)[:, None]
    num, den = np.zeros(grid), np.zeros(grid)
    step = max(1, 2_000_000 // grid)
    for s in range(0, len(f), step):
        fi, ri = f[s : s + step][None], (y - f)[s : s + step]
        k = sum(np.exp(-0.5 * ((t - c) / bandwidth) ** 2) for c in (fi, -fi, 2 - fi))
        num += k @ ri
        den += k.sum(1)
    return float(np.abs(num).sum() / den.sum())


def _equal_mass(conf: np.ndarray, bins: int) -> list[np.ndarray]:
    order = np.argsort(conf, kind="stable")
    return [b for b in np.array_split(order, min(bins, len(conf))) if len(b)]


def adaptive_ece(conf: Sequence[float], hit: Sequence[float], bins: int = 15) -> float | None:
    """Top-label ECE over ``bins`` equal-mass bins (sorted by confidence)."""
    c, h = np.asarray(conf, np.float64), np.asarray(hit, np.float64)
    if not len(c):
        return None
    return float(sum(len(b) * abs(h[b].mean() - c[b].mean()) for b in _equal_mass(c, bins))
                 / len(c))  # fmt: skip


def brier_decomposition(conf: Sequence[float], hit: Sequence[float], bins: int = 15) -> dict:
    """The Brier score of the top-label confidence and its Murphy decomposition
    (reliability, resolution, uncertainty) over ``bins`` equal-mass bins."""
    c, h = np.asarray(conf, np.float64), np.asarray(hit, np.float64)
    if not len(c):
        return {"brier": None, "reliability": None, "resolution": None, "uncertainty": None}
    base = h.mean()
    groups = _equal_mass(c, bins)
    return {
        "brier": float(np.mean((c - h) ** 2)),
        "reliability": float(sum(len(b) * (c[b].mean() - h[b].mean()) ** 2 for b in groups)
                             / len(c)),
        "resolution": float(sum(len(b) * (h[b].mean() - base) ** 2 for b in groups) / len(c)),
        "uncertainty": float(base * (1 - base)),
    }  # fmt: skip


def aurc(conf: Sequence[float], hit: Sequence[float]) -> dict:
    """Area under the risk-coverage curve: the mean, over k = 1..N, of the error rate of the
    k most confident items (ties: in input order), and E-AURC (minus an oracle's)."""
    c, h = np.asarray(conf, np.float64), np.asarray(hit, np.float64)
    if not len(c):
        return {"aurc": None, "eaurc": None}
    k = np.arange(1, len(c) + 1)
    err = 1 - h[np.argsort(-c, kind="stable")]
    value = float(np.mean(np.cumsum(err) / k))
    oracle = float(np.mean(np.cumsum(np.sort(1 - h)) / k))
    return {"aurc": value, "eaurc": value - oracle}


def diagnostics(rows: list[v15.Row]) -> dict:
    """The calibration diagnostics of rows with a distribution (top label)."""
    sel = [r for r in rows if r.probs is not None]
    dists = [r.dist() for r in sel]
    conf = [max(d.values()) for d in dists]
    hit = [float(v15.argmax_label(d) == r.gold) for r, d in zip(sel, dists, strict=True)]
    out: dict[str, Any] = {
        "n": len(sel),
        "accuracy": float(np.mean(hit)) if hit else None,
        "confidence": float(np.mean(conf)) if conf else None,
        "smece": smooth_ece(conf, hit),
        "adaptive_ece": adaptive_ece(conf, hit),
        **brier_decomposition(conf, hit),
        **aurc(conf, hit),
    }
    score = [(r, d) for r, d in zip(sel, dists, strict=True) if r.type == "score"]
    if score:
        out["rps"] = float(np.mean([v15.rps(d, int(r.gold), len(r.labels)) for r, d in score]))
    return out


def calibration_diagnostics(rows: list[v15.Row]) -> dict:
    """:func:`diagnostics` per question type, and per language × type where rows carry one."""
    by_type: dict[str, list[v15.Row]] = defaultdict(list)
    by_lang: dict[str, dict[str, list[v15.Row]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_type[r.type].append(r)
        if r.lang:
            by_lang[r.lang][r.type].append(r)
    return {
        "types": {k: diagnostics(v) for k, v in sorted(by_type.items())},
        "langs": {lang: {k: diagnostics(v) for k, v in sorted(t.items())}
                  for lang, t in sorted(by_lang.items())},
    }  # fmt: skip


# --- dumps ----------------------------------------------------------------------------------


def dump_name(path: str | Path) -> str:
    return Path(path).name.removesuffix(".jsonl.gz")


def pair_dumps(a: Sequence[str | Path], b: Sequence[str | Path]) -> list[tuple[Path, Path]]:
    """Each A dump with the B dump of the same name: a B file, or a file of that name in a
    B directory. Raises if an A dump has no counterpart."""
    files = {dump_name(p): Path(p) for p in b if not Path(p).is_dir()}
    dirs = [Path(p) for p in b if Path(p).is_dir()]
    out = []
    for path in map(Path, a):
        name = path.name
        other = files.get(dump_name(path)) or next(
            (d / name for d in dirs if (d / name).exists()), None
        )
        if other is None:
            raise FileNotFoundError(f"no B dump named {name}")
        out.append((path, other))
    return out


def compare_dumps(
    a: Sequence[str | Path],
    b: Sequence[str | Path],
    *,
    policy: Policy = None,
    gold_probs: dict[str, dict[str, float]] | None = None,
    n: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired A − B of prediction dumps (each with its own probabilities), per dump name,
    as calibrated and under ``policy`` if given (:func:`paired`)."""
    from tjev.serve.policy import as_policy

    out: dict[str, Any] = {}
    for pa, pb in pair_dumps(a, b):
        rows_a, _ = v15.load_rows(pa, gold_probs=gold_probs)
        rows_b, _ = v15.load_rows(pb, gold_probs=gold_probs)
        rep: dict[str, Any] = {"a": str(pa), "b": str(pb), "n_resamples": n, "seed": seed,
                               "level": LEVEL}  # fmt: skip
        views = [("calibrated", None)] + ([("policy", policy)] if as_policy(policy) else [])
        for key, p in views:
            res = paired(items(rows_a, p), items(rows_b, p), n, seed)
            rep |= {k: res[k] for k in ("n_common", "only_a", "only_b", "duplicates")}
            rep[key] = res["metrics"]
        out[dump_name(pa)] = rep
    return out
