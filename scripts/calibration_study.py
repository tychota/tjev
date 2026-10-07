"""Calibration study on a run's per-item predictions (``tjev post`` → ``post/predictions/``).

    JAX_PLATFORMS=cpu python scripts/calibration_study.py RUN/post/predictions \\
        --out reports/calibration/RUN [--folds 5]

Every method is fitted on the mix calibration split and scored on the mix validation split
(in distribution) and the held-out generators (another seed: domain-shifted), plus a
K-fold CV on the calibration split itself. Methods (docs/calibration.md):

- temperature groupings: none, global, per type (the old artifact), per language, per
  (type × lang) unshrunk, per (type × lang) ridge-shrunk to per type with tau by CV (the
  implementation, :func:`tjev.eval.calibrate.fit_grouped`), the same with an empirical-
  Bayes tau (normal-normal on log T, Laplace curvature), the same applied with the
  *guessed* language (what the server sees), T(n) = a + b ln n (option count, global and
  with a per-type a), per source (an upper bound: sources are unknown at serving);
- beyond temperature, on top of per-type T: a bias per displayed option position (per
  type), beta calibration of noul P(yes), histogram binning of the top-label confidence;
- oracle rows: per type / per (type × lang) T fitted on the evaluated set itself (the
  floor any temperature grouping fitted elsewhere can reach: the remaining gap).

Writes ``OUT/study.json`` and ``OUT/study.md`` (the tables of docs/calibration.md).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from tjev.eval import calibrate as cal
from tjev.eval import predictions as dump
from tjev.eval.metrics import label_mask, slot_temperatures, summarize, view_probs

Data = dict
Predictor = Callable[[Data], np.ndarray]  # data -> probs [N,K]
SPLITS = ("validation", "generators")


def load(directory: Path, split: str, tag: str = "") -> Data:
    d = dump.arrays(dump.read(directory / f"{split}{tag}.jsonl.gz"))
    d["type"], d["lang"], d["source"] = (np.asarray(d[k]) for k in ("type", "lang", "source"))
    d["lang_guess"] = np.asarray([g or "?" for g in d["lang_guess"]])
    d["pos"] = positions(d)
    noul = [lab for lab, k in zip(d["labels"], d["type"], strict=True) if k == "noul"]
    if any(list(lab) != ["no", "yes"] for lab in noul):
        raise ValueError("noul labels must be ('no', 'yes')")
    return d


def positions(d: Data) -> np.ndarray:
    """[N,V,K] displayed position of each label in each view (-1 absent)."""
    pos = -np.ones(d["logits"].shape, np.int64)
    for i, views in enumerate(d["orders"]):
        for v, order in enumerate(views):
            pos[i, v, order] = np.arange(len(order))
    return pos


def take(d: Data, sel: np.ndarray) -> Data:
    """The items ``sel`` (a boolean mask or indices) of ``d`` (and of its aligned
    ``second`` checkpoint)."""
    n = len(d["type"])
    sel = np.asarray(sel)
    idx = np.flatnonzero(sel) if sel.dtype == bool else sel
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and len(v) == n:
            out[k] = v[idx]
        elif isinstance(v, list) and len(v) == n:
            out[k] = [v[i] for i in idx]
        elif isinstance(v, dict) and "type" in v:
            out[k] = take(v, idx)
        else:
            out[k] = v
    return out


def align(a: Data, b: Data) -> Data:
    """``a`` restricted to the ids ``b`` also has, with ``b`` in the same order as
    ``a["second"]``."""
    where = {i: j for j, i in enumerate(b["id"])}
    a = take(a, np.asarray([i in where for i in a["id"]]))
    return a | {"second": take(b, np.asarray([where[i] for i in a["id"]], np.int64))}


def metrics(d: Data, probs: np.ndarray) -> dict:
    m = label_mask(d["mask"])
    out = {"all": summarize(probs, d["target"], m)}
    for g in ("lang", "type"):
        for value in sorted(set(d[g].tolist())):
            sel = d[g] == value
            out[f"{g}={value}"] = summarize(probs[sel], d["target"][sel], m[sel])
    if (d["type"] == "noul").any():
        out["noul_band"] = noul_band(probs[d["type"] == "noul"], d["target"][d["type"] == "noul"])
    return out


BAND = (0.2, 0.8)  # JevBench v1.5: P(yes) <= 0.2 no, >= 0.8 yes, in between an abstention


def noul_band(probs: np.ndarray, target: np.ndarray) -> dict:
    """The JevBench v1.5 reading of noul P(yes): abstention rate, accuracy with
    abstentions counted wrong (v1.5), accuracy of the committed answers."""
    p, gold_yes = probs[:, 1], target[:, 1] >= 0.5
    committed = (p <= BAND[0]) | (p >= BAND[1])
    correct = ((p >= BAND[1]) & gold_yes) | ((p <= BAND[0]) & ~gold_yes)
    return {"abstain": float(1 - committed.mean()), "v15_accuracy": float(correct.mean()),
            "committed_accuracy": float(correct.sum() / max(committed.sum(), 1))}  # fmt: skip


def flags(d: Data, probs: np.ndarray, base: np.ndarray) -> dict:
    """Degenerate outputs: probability ~0 on a gold label (NLL blow-up) or ~1 at the top,
    and how often the argmax moved from the uncalibrated one (accuracy change)."""
    m = label_mask(d["mask"])
    p0 = int(np.sum(np.any(m & (d["target"] > 0) & (probs < 1e-6), -1)))
    p1 = int(np.sum(probs.max(-1) > 1 - 1e-9))
    moved = float(np.mean(probs.argmax(-1) != base.argmax(-1)))
    acc = summarize(probs, d["target"], m)["accuracy"] - summarize(base, d["target"], m)["accuracy"]
    return {"p0_on_gold": p0, "p1": p1, "argmax_changed": moved, "accuracy_delta": float(acc)}


def sum_nll(d: Data, probs: np.ndarray) -> float:
    m = label_mask(d["mask"])
    return float(-np.sum(d["target"] * np.log(np.where(m, probs, 1.0) + 1e-12)))


# ---------------------------------------------------------------- temperature groupings


def by_key(key: Callable[[Data], np.ndarray]) -> Callable[[Data], Predictor]:
    """One temperature per key value (unseen values: the global temperature)."""

    def fit(d: Data) -> Predictor:
        keys = key(d)
        glob, *_ = cal.fit_temperature(d["logits"], d["mask"], d["target"])
        temps = {}
        for k in sorted(set(keys.tolist())):
            sel = keys == k
            temps[k], *_ = cal.fit_temperature(d["logits"][sel], d["mask"][sel], d["target"][sel])

        def predict(e: Data) -> np.ndarray:
            t = np.asarray([temps.get(k, glob) for k in key(e).tolist()])
            return view_probs(e["logits"], e["mask"], t)

        predict.params = {"temperatures": temps, "global": glob}
        return predict

    return fit


def none(d: Data) -> Predictor:
    return lambda e: view_probs(e["logits"], e["mask"], 1.0)


def global_t(d: Data) -> Predictor:
    return by_key(lambda e: np.zeros(len(e["type"]), int))(d)


def grouped(grouping: str, *, lang_key: str = "lang", tau: float | None = None):
    """The implementation (:func:`tjev.eval.calibrate.fit_grouped`), applied with the
    slots' ``lang_key`` (``lang_guess``: what the server sees)."""

    def fit(d: Data) -> Predictor:
        if tau is None:
            f = cal.fit_grouped(d["logits"], d["mask"], d["target"], list(d["type"]),
                                list(d["lang"]), grouping=grouping)  # fmt: skip
        else:
            f = cal.fit_grouped(d["logits"], d["mask"], d["target"], list(d["type"]),
                                list(d["lang"]), grouping=grouping, taus=(tau,))  # fmt: skip

        def predict(e: Data) -> np.ndarray:
            langs = [None if g == "?" else g for g in e[lang_key].tolist()]
            t = slot_temperatures(f["temperatures"], list(e["type"]), langs)
            return view_probs(e["logits"], e["mask"], t)

        predict.params = {k: f.get(k) for k in ("temperatures", "tau", "cv_nll")}
        return predict

    return fit


def empirical_bayes(d: Data) -> Predictor:
    """Normal-normal on log T per (type, lang) around the type's: tau = 1 / (2 sigma^2),
    sigma^2 = mean squared deviation of the cells' MLE log T minus their mean variance
    (1 / curvature of the summed NLL); then the ridge fit with that tau."""
    base = cal.fit_per_type(d["logits"], d["mask"], d["target"], list(d["type"]))["temperatures"]
    dev, var = [], []
    for k in sorted(set(d["type"].tolist())):
        for g in sorted(set(d["lang"][d["type"] == k].tolist())):
            sel = (d["type"] == k) & (d["lang"] == g)
            lg, mk, tg = d["logits"][sel], d["mask"][sel], d["target"][sel]
            n = int(sel.sum())
            if n < 50:  # as fit_grouped: no cell of its own
                continue

            def f(u, lg=lg, mk=mk, tg=tg, n=n):
                return n * cal.nll(lg, mk, tg, float(np.exp(u)))

            u = minimize_scalar(f, bounds=cal.LOG_T_BOUNDS, method="bounded").x
            h = 1e-3
            curv = max((f(u + h) - 2 * f(u) + f(u - h)) / h**2, 1e-6)
            dev.append((u - np.log(base[k])) ** 2)
            var.append(1.0 / curv)
    sigma2 = max(float(np.mean(dev) - np.mean(var)), 1e-8)
    tau = 1.0 / (2.0 * sigma2)
    predict = grouped("type_lang", tau=tau)(d)
    predict.params = predict.params | {"eb_tau": tau, "sigma_log_t": float(np.sqrt(sigma2))}
    return predict


def option_count(per_type: bool) -> Callable[[Data], Predictor]:
    """T(n) = a (+ a_type) + b ln n, fitted by NLL (n: the item's option count)."""

    def fit(d: Data) -> Predictor:
        types = sorted(set(d["type"].tolist()))

        def temps(params, e):
            a = params[0] + (np.asarray([dict(zip(types, [0.0, *params[2:]], strict=False))
                                         .get(k, 0.0) for k in e["type"]]) if per_type else 0)  # fmt: skip
            return np.clip(a + params[1] * np.log(e["n"]), 0.05, 20.0)

        x0 = np.r_[1.0, 0.0, np.zeros(len(types) - 1 if per_type else 0)]
        res = minimize(lambda p: sum_nll(d, view_probs(d["logits"], d["mask"], temps(p, d))),
                       x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4})  # fmt: skip

        def predict(e: Data) -> np.ndarray:
            return view_probs(e["logits"], e["mask"], temps(res.x, e))

        predict.params = {"a_b_atype": res.x.tolist(), "types": types}
        return predict

    return fit


# ---------------------------------------------------------------- beyond temperature

POSITIONS = 8  # positions >= 7 share one bias


def vector_scaling(kinds: tuple[str, ...] | None = None) -> Callable[[Data], Predictor]:
    """Vector scaling in *position* space (options are shuffled, so a label-space vector
    does not exist): per type, log T and a bias per displayed letter position (position 0
    the reference, >= 7 shared) added to the scaled logits of every view. Types not in
    ``kinds`` (None: all) get their temperature only. With variable options, this is the
    part of Dirichlet calibration (Kull et al. 2019) that is defined."""

    def fit(d: Data) -> Predictor:
        return _vector_scaling(d, kinds)

    return fit


def _vector_scaling(d: Data, kinds) -> Predictor:
    params = {}
    for k in sorted(set(d["type"].tolist())):
        s = take(d, d["type"] == k)
        pos = np.clip(s["pos"], 0, POSITIONS - 1)
        free = kinds is None or k in kinds

        def loss(p, s=s, pos=pos, free=free):
            bias = np.r_[0.0, p[1:]] if free else np.zeros(POSITIONS)
            z = s["logits"] / np.exp(p[0]) + bias[pos]
            return sum_nll(s, view_probs(z, s["mask"]))

        x0 = np.zeros(POSITIONS if free else 1)
        p = minimize(loss, x0, method="L-BFGS-B").x
        params[k] = np.r_[p, np.zeros(POSITIONS - len(p))]

    def predict(e: Data) -> np.ndarray:
        z = e["logits"].copy()
        pos = np.clip(e["pos"], 0, POSITIONS - 1)
        for k, p in params.items():
            sel = e["type"] == k
            z[sel] = z[sel] / np.exp(p[0]) + np.r_[0.0, p[1:]][pos[sel]]
        return view_probs(z, e["mask"])

    predict.params = {k: {"T": float(np.exp(p[0])), "bias": p[1:].round(3).tolist()}
                      for k, p in params.items()}  # fmt: skip
    return predict


def beta_noul(d: Data) -> Predictor:
    """Per-type T, then noul P(yes) recalibrated by beta calibration:
    logit p' = a ln p - b ln(1 - p) + c (a = b, c = 0: identity), fitted by NLL."""
    base = grouped("type")(d)
    yes = 1  # noul labels are ("no", "yes")
    s = d["type"] == "noul"
    p = np.clip(base(d)[s, yes], 1e-6, 1 - 1e-6)
    t = d["target"][s, yes]

    def loss(x):
        q = 1 / (1 + np.exp(-(x[0] * np.log(p) - x[1] * np.log(1 - p) + x[2])))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -np.sum(t * np.log(q) + (1 - t) * np.log(1 - q))

    x = minimize(loss, np.r_[1.0, 1.0, 0.0], method="L-BFGS-B").x

    def predict(e: Data) -> np.ndarray:
        probs = base(e).copy()
        sel = e["type"] == "noul"
        p = np.clip(probs[sel, yes], 1e-6, 1 - 1e-6)
        q = 1 / (1 + np.exp(-(x[0] * np.log(p) - x[1] * np.log(1 - p) + x[2])))
        probs[sel, yes], probs[sel, 0] = q, 1 - q
        return probs

    predict.params = {"a_b_c": x.round(4).tolist()}
    return predict


def histogram(bins: int = 10) -> Callable[[Data], Predictor]:
    """Per-type T, then per type the top-label probability replaced by the expected
    correctness of its equal-mass bin (the rest rescaled to sum to 1)."""

    def fit(d: Data) -> Predictor:
        base = grouped("type")(d)
        probs = base(d)
        table = {}
        for k in sorted(set(d["type"].tolist())):
            sel = d["type"] == k
            p = probs[sel]
            conf = p.max(-1)
            correct = np.take_along_axis(d["target"][sel], p.argmax(-1)[:, None], -1)[:, 0]
            edges = np.quantile(conf, np.linspace(0, 1, bins + 1)[1:-1])
            idx = np.searchsorted(edges, conf)
            # a bin's value: its mean correctness, smoothed towards its mean confidence
            value = np.array([(correct[idx == b].sum() + 2 * conf[idx == b].mean())
                              / ((idx == b).sum() + 2) if (idx == b).any() else np.nan
                              for b in range(bins)])  # fmt: skip
            table[k] = (edges, value)

        def predict(e: Data) -> np.ndarray:
            probs = base(e).copy()
            for k, (edges, value) in table.items():
                sel = np.flatnonzero(e["type"] == k)
                p = probs[sel]
                top = p.argmax(-1)
                conf = p.max(-1)
                new = np.clip(value[np.searchsorted(edges, conf)], 1e-4, 1 - 1e-4)
                new = np.where(np.isnan(new), conf, new)
                rest = np.where(conf < 1, (1 - new) / np.maximum(1 - conf, 1e-12), 0.0)
                q = p * rest[:, None]
                q[np.arange(len(sel)), top] = new
                m = label_mask(e["mask"][sel])
                q = np.where(m, q, 0.0)
                probs[sel] = q / q.sum(-1, keepdims=True)
            return probs

        return predict

    return fit


METHODS: dict[str, Callable[[Data], Predictor]] = {
    "none (T = 1)": none,
    "global T": global_t,
    "per type (old artifact)": grouped("type"),
    "per lang": by_key(lambda e: e["lang"]),
    "per type × lang, unshrunk": grouped("type_lang", tau=0.0),
    "per type × lang, ridge (CV tau) [impl]": grouped("type_lang"),
    "per type × lang, ridge, served (guessed lang)": grouped("type_lang", lang_key="lang_guess"),
    "per type × lang, empirical Bayes": empirical_bayes,
    "T(n) = a + b ln n": option_count(per_type=False),
    "T(n) = a_type + b ln n": option_count(per_type=True),
    "per source (upper bound)": by_key(lambda e: e["source"]),
    "feature T (thermometer-lite)": None,  # set below (defined after METHODS' helpers)
    "vector scaling: T + position bias, choice": vector_scaling(("choice",)),
    "vector scaling: T + position bias, every type": vector_scaling(),
    "batch calibration (eval batch), choice+noul": None,
    "batch calibration (calibration split), choice+noul": None,
    "per type T + Platt with yes-bias (noul)": None,
    "per type T + beta (noul)": beta_noul,
    "orderings: first only, per type T": None,
    "orderings: mean of probabilities, per type T": None,
    "orderings: log-space mean, per type T": None,
    "checkpoint 2 alone, per type T": None,
    "checkpoints 1 + 2, mean of calibrated probabilities": None,
    "[diagnostic] per type T + histogram binning": histogram(),
    "[diagnostic] per type T + isotonic (top label)": None,
}


# ---------------------------------------------------------------- the follow-up methods


def feats(e: Data) -> np.ndarray:
    return cal.temperature_features(e["logits"], e["mask"], e["n"], e["lang_guess"], e["tokens"])


def feature_t(d: Data) -> Predictor:
    """log T = u_type + w · features (:func:`tjev.eval.calibrate.fit_features`)."""
    f = cal.fit_features(d["logits"], d["mask"], d["target"], d["type"], feats(d))

    def predict(e: Data) -> np.ndarray:
        t = np.exp(cal.feature_log_t(f, e["type"], feats(e)))
        return view_probs(e["logits"], e["mask"], t)

    predict.params = {k: f.get(k) for k in ("type_log_t", "w", "features", "lambda", "cv_nll")}
    return predict


BC_KINDS = ("choice", "noul")  # score levels keep their order: their "bias" is the prior
BC_MIN = 20  # items per (type, option count) group for a bias estimate


def _view_softmax(e: Data) -> np.ndarray:
    z = np.where(e["mask"], e["logits"], -np.inf)
    z = z - np.where(np.any(e["mask"], -1, keepdims=True), z.max(-1, keepdims=True), 0.0)
    x = np.where(e["mask"], np.exp(z), 0.0)
    return x / np.maximum(x.sum(-1, keepdims=True), 1e-300)


def bc_table(e: Data) -> dict:
    """Batch calibration (Zhou et al. 2024): per (type, option count), the mean T = 1
    probability of each displayed position over the batch (label-free)."""
    s, table = _view_softmax(e), {}
    for k in BC_KINDS:
        for n in sorted(set(e["n"][e["type"] == k].tolist())):
            sel = (e["type"] == k) & (e["n"] == n)
            if sel.sum() < BC_MIN:
                continue
            pos, mk, sp = e["pos"][sel], e["mask"][sel], s[sel]
            table[(k, int(n))] = np.asarray([sp[(pos == j) & mk].mean() for j in range(n)])
    return table


def debias(e: Data, table: dict) -> np.ndarray:
    """Logits minus log(mean probability of their displayed position)."""
    z = e["logits"].copy()
    for (k, n), mean in table.items():
        sel = (e["type"] == k) & (e["n"] == n)
        pos = np.clip(e["pos"][sel], 0, n - 1)
        z[sel] = z[sel] - np.log(np.maximum(mean, 1e-12))[pos]
    return z


def batch_calibration(source: str) -> Callable[[Data], Predictor]:
    """``source="batch"``: the position means of the evaluated batch itself (transductive,
    as in the paper); ``"calibration"``: those of the calibration split. Then per-type T,
    fitted on the debiased calibration logits."""

    def fit(d: Data) -> Predictor:
        own = bc_table(d)
        temps = cal.fit_per_type(debias(d, own), d["mask"], d["target"], list(d["type"]))
        temps = temps["temperatures"]

        def predict(e: Data) -> np.ndarray:
            z = debias(e, bc_table(e) if source == "batch" else own)
            return view_probs(z, e["mask"], slot_temperatures(temps, list(e["type"])))

        predict.params = {"temperatures": temps,
                          "table": {f"{k}/{n}": v.round(4).tolist() for (k, n), v in own.items()}}  # fmt: skip
        return predict

    return fit


def platt_noul(d: Data) -> Predictor:
    """Platt scaling of noul (Platt 1999) on the T = 1 order-averaged P(yes):
    logit p' = a logit p + c; a plays 1 / T, c is the explicit yes-bias (c < 0: the model
    says yes too easily). Choice / score: per-type T."""
    base = grouped("type")(d)
    s = d["type"] == "noul"
    p = np.clip(view_probs(d["logits"][s], d["mask"][s])[:, 1], 1e-9, 1 - 1e-9)
    t = d["target"][s, 1]
    lg = np.log(p) - np.log1p(-p)

    def loss(x):
        q = np.clip(1 / (1 + np.exp(-(x[0] * lg + x[1]))), 1e-12, 1 - 1e-12)
        return -np.sum(t * np.log(q) + (1 - t) * np.log(1 - q))

    x = minimize(loss, np.r_[1.0, 0.0], method="L-BFGS-B").x

    def predict(e: Data) -> np.ndarray:
        probs = base(e).copy()
        sel = e["type"] == "noul"
        p = np.clip(view_probs(e["logits"][sel], e["mask"][sel])[:, 1], 1e-9, 1 - 1e-9)
        q = 1 / (1 + np.exp(-(x[0] * (np.log(p) - np.log1p(-p)) + x[1])))
        probs[sel, 1], probs[sel, 0] = q, 1 - q
        return probs

    predict.params = {"a": float(x[0]), "yes_bias_c": float(x[1]), "T_equiv": float(1 / x[0])}
    return predict


def orderings(mode: str) -> Callable[[Data], Predictor | None]:
    """Order ensembles (needs a dump with K > 1 views): ``first`` (one ordering),
    ``mean`` (mean of the views' probabilities: what tjev serves) or ``logmean`` (the
    renormalised geometric mean: softmax of the mean log-probability); then per-type T."""

    def view(e: Data) -> Data:
        if mode == "first":
            m = e["mask"].copy()
            m[:, 1:] = False
            return e | {"mask": m}
        if mode == "logmean":
            present = np.any(e["mask"], -1, keepdims=True)
            z = np.where(e["mask"], e["logits"], -np.inf)
            ls = z - np.logaddexp.reduce(z, -1, keepdims=True)
            mean = np.where(present, np.where(e["mask"], ls, 0.0), 0.0).sum(1) / present.sum(1)
            return e | {"logits": mean[:, None], "mask": np.any(e["mask"], 1)[:, None]}
        return e

    def fit(d: Data) -> Predictor | None:
        if d["logits"].shape[1] < 2:
            return None
        f = grouped("type")(view(d))

        def predict(e: Data) -> np.ndarray:
            return f(view(e))

        predict.params = f.params
        return predict

    return fit


def second_alone(d: Data) -> Predictor | None:
    if "second" not in d:
        return None
    f = grouped("type")(d["second"])
    return lambda e: f(e["second"])


def checkpoint_ensemble(d: Data) -> Predictor | None:
    """Mean of the two checkpoints' per-type-calibrated probabilities (each its own T)."""
    if "second" not in d:
        return None
    f1, f2 = grouped("type")(d), grouped("type")(d["second"])
    return lambda e: (f1(e) + f2(e["second"])) / 2


def pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic (non-decreasing) regression of y on x: block right edges and values."""
    order = np.argsort(x)
    xs, ys = x[order], y[order].astype(float)
    vals, wts, ends = [], [], []
    for xi, yi in zip(xs, ys, strict=True):
        vals.append(yi), wts.append(1.0), ends.append(xi)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals[-2:], wts[-2:], ends[-2:] = [v], [w], [ends[-1]]
    return np.asarray(ends), np.asarray(vals)


def isotonic(d: Data) -> Predictor:
    """Diagnostic: per-type T, then the top-label probability mapped by isotonic
    regression on correctness (Zadrozny & Elkan 2002), the rest rescaled. Unclipped, so it
    can emit 0 / 1 (flagged)."""
    base = grouped("type")(d)
    probs = base(d)
    maps = {}
    for k in sorted(set(d["type"].tolist())):
        sel = d["type"] == k
        p = probs[sel]
        correct = np.take_along_axis(d["target"][sel], p.argmax(-1)[:, None], -1)[:, 0]
        maps[k] = pav(p.max(-1), correct)

    def predict(e: Data) -> np.ndarray:
        out = base(e).copy()
        m = label_mask(e["mask"])
        for k, (ends, vals) in maps.items():
            sel = np.flatnonzero(e["type"] == k)
            p = out[sel]
            conf, top = p.max(-1), p.argmax(-1)
            new = vals[np.clip(np.searchsorted(ends, conf), 0, len(vals) - 1)]
            rest = np.where(conf < 1, (1 - new) / np.maximum(1 - conf, 1e-12), 0.0)
            q = np.where(m[sel], p * rest[:, None], 0.0)
            q[np.arange(len(sel)), top] = new
            out[sel] = q / np.maximum(q.sum(-1, keepdims=True), 1e-300)
        return out

    return predict


METHODS.update({
    "feature T (thermometer-lite)": feature_t,
    "batch calibration (eval batch), choice+noul": batch_calibration("batch"),
    "batch calibration (calibration split), choice+noul": batch_calibration("calibration"),
    "per type T + Platt with yes-bias (noul)": platt_noul,
    "orderings: first only, per type T": orderings("first"),
    "orderings: mean of probabilities, per type T": orderings("mean"),
    "orderings: log-space mean, per type T": orderings("logmean"),
    "checkpoint 2 alone, per type T": second_alone,
    "checkpoints 1 + 2, mean of calibrated probabilities": checkpoint_ensemble,
    "[diagnostic] per type T + isotonic (top label)": isotonic,
})  # fmt: skip


def cross_validate(fit, d: Data, folds: int, seed: int = 1) -> float:
    fold = np.random.default_rng(seed).permutation(len(d["type"])) % folds
    total = 0.0
    for f in range(folds):
        predict = fit(take(d, fold != f))
        test = take(d, fold == f)
        total += sum_nll(test, predict(test))
    return total / len(d["type"])


def oracle(d: Data, grouping: str) -> dict:
    """Temperatures fitted on the evaluated set itself (in-sample): the floor."""
    return metrics(d, grouped(grouping, tau=0.0 if grouping == "type_lang" else None)(d)(d))


def lang_guess_table(d: Data) -> dict:
    out = {}
    for g in sorted(set(d["lang"].tolist())):
        sel = d["lang"] == g
        guesses = d["lang_guess"][sel]
        out[g] = {x: int((guesses == x).sum()) for x in sorted(set(guesses.tolist()))}
    return out


def table(rows: dict[str, dict], cols: list[tuple[str, str]], fmt: str = "{:.4f}") -> str:
    head = "| method | " + " | ".join(c[0] for c in cols) + " |\n|---|" + "---|" * len(cols)
    lines = [head]
    for name, r in rows.items():
        cells = []
        for _, path in cols:
            v = r
            for part in path.split("/"):
                v = v.get(part, {}) if isinstance(v, dict) else {}
            cells.append(fmt.format(v) if isinstance(v, float) else "–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--methods", default="", help="semicolon-separated subset (default: all)")
    ap.add_argument("--second", type=Path, help="another checkpoint's predictions directory")
    args = ap.parse_args()
    calib = load(args.predictions, "calibration", args.tag)
    evals = {s: load(args.predictions, s, args.tag) for s in SPLITS
             if (args.predictions / f"{s}{args.tag}.jsonl.gz").exists()}  # fmt: skip
    if args.second:  # both checkpoints on the items both scored
        calib = align(calib, load(args.second, "calibration", args.tag))
        evals = {s: align(e, load(args.second, s, args.tag)) for s, e in evals.items()}
    base = {s: view_probs(e["logits"], e["mask"]) for s, e in evals.items()}
    chosen = [m for m in METHODS if not args.methods or m in args.methods.split(";")]
    results: dict = {"n": {"calibration": len(calib["type"])} | {s: len(e["type"])
                     for s, e in evals.items()}, "methods": {}}  # fmt: skip
    for name in chosen:
        fit = METHODS[name]
        print(f"[study] {name}", flush=True)
        predict = fit(calib)
        if predict is None:
            print(f"[study]   n/a (needs {'K > 1 views' if 'ordering' in name else '--second'})")
            continue
        entry = {"cv_nll": cross_validate(fit, calib, args.folds),
                 "params": json.loads(json.dumps(getattr(predict, "params", {}), default=str))}  # fmt: skip
        for s, e in evals.items():
            probs = predict(e)
            entry[s] = metrics(e, probs) | {"flags": flags(e, probs, base[s])}
        results["methods"][name] = entry
    results["oracle"] = {f"{s} {g}": oracle(e, g) for s, e in evals.items()
                         for g in ("type", "type_lang")}  # fmt: skip
    results["lang_guess"] = {s: lang_guess_table(e) for s, e in
                             {"calibration": calib, **evals}.items()}  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "study.json").write_text(json.dumps(results, indent=1, default=str))
    md = [f"n: {results['n']}\n"]
    cols = [("CV NLL (cal)", "cv_nll")]
    for s in evals:
        cols += [(f"{s} NLL", f"{s}/all/nll"), (f"{s} ECE", f"{s}/all/ece"),
                 (f"{s} Brier", f"{s}/all/brier"), (f"{s} acc", f"{s}/all/accuracy")]  # fmt: skip
    md.append("## All slots\n\n" + table(results["methods"], cols))
    for s in evals:
        cols = [(f"{c} {m}", f"{s}/{c}/{m}") for c in ("lang=en", "lang=fr") for m in
                ("nll", "ece")] + [(f"{c} NLL", f"{s}/{c}/nll") for c in
                ("type=noul", "type=choice", "type=score")]  # fmt: skip
        md.append(f"## {s}: by language and type\n\n" + table(results["methods"], cols))
        cols = [("noul abstain", f"{s}/noul_band/abstain"),
                ("noul v1.5 acc", f"{s}/noul_band/v15_accuracy"),
                ("noul committed acc", f"{s}/noul_band/committed_accuracy"),
                ("argmax moved", f"{s}/flags/argmax_changed"),
                ("Δ accuracy", f"{s}/flags/accuracy_delta")]  # fmt: skip
        blown = ", ".join(f"{k}: {v[s]['flags']['p0_on_gold']}/{v[s]['flags']['p1']}"
                          for k, v in results["methods"].items()
                          if v[s]["flags"]["p0_on_gold"] or v[s]["flags"]["p1"])  # fmt: skip
        md.append(f"## {s}: noul in the v1.5 band [0.2, 0.8], and flags\n\n"
                  + table(results["methods"], cols)
                  + "\n\nItems with p < 1e-6 on a gold label (NLL blow-up) / p > 1 - 1e-9: "
                  + (blown or "none"))  # fmt: skip
    orc = {k: {"x": v} for k, v in results["oracle"].items()}
    md.append("## Oracle (fitted on the evaluated set)\n\n" + table(
        orc, [("NLL", "x/all/nll"), ("ECE", "x/all/ece"), ("en NLL", "x/lang=en/nll"),
              ("fr NLL", "x/lang=fr/nll")]))  # fmt: skip
    md.append("## Guessed language (rows: tagged)\n\n```\n"
              + json.dumps(results["lang_guess"], indent=1) + "\n```")  # fmt: skip
    md.append("## Fitted parameters\n\n```\n" + json.dumps(
        {k: v["params"] for k, v in results["methods"].items()}, indent=1, default=str)[:6000]
        + "\n```")  # fmt: skip
    (args.out / "study.md").write_text("\n\n".join(md) + "\n")
    print("\n\n".join(md[:4]))


if __name__ == "__main__":
    main()
