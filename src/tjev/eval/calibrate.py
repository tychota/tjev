"""Temperature scaling fitted by NLL on held-out in-distribution slots, per type or per
(type, language) shrunk to the type's temperature.

T=1 is always a candidate of a type's fit, so it never makes the fit set's NLL worse than
uncalibrated.

Groupings (``grouping`` in the artifact; artifacts without one are ``"type"``):

- ``"type"``: one temperature per question type (``{"noul": T, ...}``).
- ``"type_lang"``: also one per (type, language) cell, keyed ``"noul/fr"``. A cell's log T
  is a ridge towards its type's: it minimises ``sum NLL(T) + tau * (log T - log T_type)^2``
  (``tau`` in slots: a cell of n slots moves halfway to its own optimum around n ~ tau / c,
  c its NLL curvature), so a small cell (fr score: ~40 slots) cannot overfit. ``tau`` is
  chosen by K-fold cross-validation on the fit set over :data:`TAU_GRID` (``inf`` = the
  per-type temperatures, so the CV choice is never worse than per-type there). The type
  keys stay in the table: :func:`lookup` falls back to them for a language the fit never
  saw or a request whose language is unknown (the server detects it, :mod:`tjev.data.lang`).

The artifact records what it was fitted on (checkpoint identity, data fingerprint,
template version, backend, and the readout: how many option orderings were averaged and
the noul order) and refuses to be applied to anything else. A temperature fitted on single
orderings is wrong for order-averaged distributions (averaging already softens them), so a
mismatched ``orders`` is refused like a mismatched checkpoint. Artifacts written before
the readout was recorded read as ``orders=1, noul_order="shuffled"`` (what they were).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from tjev.eval.metrics import cell, label_mask, lookup, slot_temperatures, view_probs

__all__ = ["cell", "lookup", "slot_temperatures"]  # re-exported: the grouping's keys
from tjev.utils import fingerprint


def nll(logits, mask, target, temperature: float) -> float:
    """Mean NLL; [N,V,K] logits / mask are order-averaged (the mean of the views'
    softmaxes at ``temperature``, :func:`tjev.eval.metrics.view_probs`), so a temperature
    is fitted for the very distribution it will be served with."""
    probs = view_probs(logits, mask, temperature)
    mask = label_mask(mask)
    return float(-np.mean(np.sum(target * np.log(np.where(mask, probs, 1.0) + 1e-12), -1)))


def fit_temperature(logits, mask, target) -> tuple[float, float, float]:
    """Returns (T, nll_before, nll_after)."""
    before = nll(logits, mask, target, 1.0)
    result = minimize_scalar(
        lambda log_t: nll(logits, mask, target, float(np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-4},
    )
    t = float(np.exp(result.x))
    after = nll(logits, mask, target, t)
    if after > before:
        t, after = 1.0, before
    return t, before, after


def fit_per_type(logits, mask, target, types: list[str], *, min_slots: int = 50) -> dict:
    """One temperature per question type (the global one where a type has few slots)."""
    kinds = np.asarray(types)
    temps, detail = {}, {}
    global_t, *_ = fit_temperature(logits, mask, target)
    for kind in sorted(set(types)):
        sel = kinds == kind
        slots = int(np.sum(sel))
        if slots < min_slots:
            temps[kind] = global_t  # too few slots for a stable per-type fit
            detail[kind] = {"slots": slots, "fallback": "global"}
            continue
        t, before, after = fit_temperature(logits[sel], mask[sel], target[sel])
        temps[kind] = t
        detail[kind] = {"slots": slots, "nll_before": before, "nll_after": after}
    return {"temperatures": temps, "global": global_t, "detail": detail}


GROUPINGS = ("type", "type_lang")  # servable (artifacts, eval, serve)
FIT_GROUPINGS = (*GROUPINGS, "features")  # + the feature-conditioned T (the study)
# per type until the per-item study (docs/calibration.md) shows type_lang wins on held-out
DEFAULT_GROUPING = "type"
TAU_GRID = (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, float("inf"))
LOG_T_BOUNDS = (float(np.log(0.05)), float(np.log(20.0)))


def _ridge_fit(logits, mask, target, prior_t: float, tau: float) -> float:
    """argmin_T  n * NLL(T) + tau * (log T - log prior_t)^2 (tau = inf: the prior)."""
    if np.isinf(tau):
        return prior_t
    n, prior = len(logits), np.log(prior_t)
    result = minimize_scalar(
        lambda u: n * nll(logits, mask, target, float(np.exp(u))) + tau * (u - prior) ** 2,
        bounds=LOG_T_BOUNDS,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(np.exp(result.x))


def _cells(types, langs) -> dict[tuple[str, str], np.ndarray]:
    kinds, langs = np.asarray(types), np.asarray(langs)
    return {
        (k, g): (kinds == k) & (langs == g)
        for k, g in sorted(set(zip(kinds.tolist(), langs.tolist(), strict=True)))
    }


def _cell_fit(logits, mask, target, types, langs, tau: float, min_slots: int) -> dict:
    """Per-type temperatures, then each (type, lang) cell's ridge towards its type's."""
    base = fit_per_type(logits, mask, target, list(types), min_slots=min_slots)
    temps = dict(base["temperatures"])
    detail = {}
    for (kind, lang), sel in _cells(types, langs).items():
        prior = temps[kind]
        if sel.sum() < min_slots:
            # too few slots: no cell of its own. A small all-correct cell (fr score, 38
            # slots, every fold) drives its own T to the bound (0.05) and CV cannot see it
            detail[cell(kind, lang)] = {"slots": int(sel.sum()), "fallback": "type"}
            continue
        t = _ridge_fit(logits[sel], mask[sel], target[sel], prior, tau)
        temps[cell(kind, lang)] = t
        detail[cell(kind, lang)] = {"slots": int(sel.sum()), "prior": prior}
    return {"temperatures": temps, "global": base["global"], "detail": base["detail"] | detail}


def _sum_nll(logits, mask, target, temps: np.ndarray) -> float:
    probs = view_probs(logits, mask, temps)
    m = label_mask(mask)
    return float(-np.sum(target * np.log(np.where(m, probs, 1.0) + 1e-12)))


def fit_grouped(
    logits,
    mask,
    target,
    types: list[str],
    langs: list[str] | None = None,
    *,
    grouping: str = DEFAULT_GROUPING,
    taus: tuple[float, ...] = TAU_GRID,
    folds: int = 5,
    seed: int = 0,
    min_slots: int = 50,
    features: np.ndarray | None = None,
) -> dict:
    """A :data:`FIT_GROUPINGS` fit. ``features``: :func:`fit_features` on ``features``
    (:func:`temperature_features`; study only, not servable yet). ``type_lang``: ``tau`` by ``folds``-fold CV (mean slot NLL
    of the held-out folds, ties to the larger tau; a single ``taus`` entry is used as is),
    then a refit on everything with it."""
    if grouping not in FIT_GROUPINGS:
        raise ValueError(f"grouping must be one of {FIT_GROUPINGS}, got {grouping!r}")
    if grouping == "features":
        if features is None:
            raise ValueError("grouping 'features' needs the slots' features")
        return fit_features(logits, mask, target, types, features, folds=folds, seed=seed)
    if grouping == "type":
        return fit_per_type(logits, mask, target, list(types), min_slots=min_slots) | {
            "grouping": "type"
        }
    if langs is None:
        raise ValueError("grouping 'type_lang' needs the slots' languages")
    types, langs = np.asarray(types), np.asarray(langs)
    if len(taus) == 1:  # a given tau: no cross-validation
        fit = _cell_fit(logits, mask, target, types, langs, taus[0], min_slots)
        return fit | {"grouping": "type_lang", "tau": taus[0] if np.isfinite(taus[0]) else "inf"}
    fold = np.random.default_rng(seed).permutation(len(types)) % folds
    cv = {}
    for tau in taus:
        total = 0.0
        for f in range(folds):
            fit_sel, test = fold != f, fold == f
            fit = _cell_fit(logits[fit_sel], mask[fit_sel], target[fit_sel], types[fit_sel],
                            langs[fit_sel], tau, min_slots)  # fmt: skip
            temps = slot_temperatures(fit["temperatures"], types[test], langs[test])
            total += _sum_nll(logits[test], mask[test], target[test], temps)
        cv[str(tau)] = total / len(types)
    best = min(reversed(taus), key=lambda t: cv[str(t)])  # ties: more shrinkage
    fit = _cell_fit(logits, mask, target, types, langs, best, min_slots)
    return fit | {"grouping": "type_lang", "tau": best if np.isfinite(best) else "inf",
                  "cv_nll": cv, "folds": folds}  # fmt: skip


# ------------------------------------------------------------ feature-conditioned temperature

FEATURES = ("ln_options", "lang_fr", "lang_unknown", "ln_tokens", "entropy", "max_prob")
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, float("inf"))


def temperature_features(logits, mask, n_options, lang_guess, tokens) -> np.ndarray:
    """[N, len(FEATURES)] raw features of a slot, all known when serving: ln(option
    count), the guessed language (fr / unsure, en the reference), ln(prompt tokens), and
    the uncalibrated (T = 1, order-averaged) distribution's normalised entropy and top
    probability."""
    p = view_probs(logits, mask, 1.0)
    m = label_mask(mask)
    n = np.maximum(np.asarray(n_options, np.float64), 2.0)
    ent = -np.sum(np.where(m & (p > 0), p * np.log(np.where(p > 0, p, 1.0)), 0.0), -1)
    guess = np.asarray([g if g in ("en", "fr") else "?" for g in lang_guess])
    return np.stack([np.log(n), (guess == "fr").astype(float), (guess == "?").astype(float),
                     np.log(np.maximum(np.asarray(tokens, np.float64), 1.0)),
                     ent / np.log(n), p.max(-1)], 1)  # fmt: skip


def nll_and_grad(logits, mask, target, log_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per slot: NLL of the order-averaged softmax at T = exp(log_t), and its derivative
    in log_t (d softmax(z / T)_y / d ln T = s_y (E_s[z] - z_y) / T)."""
    logits, mask = np.asarray(logits, np.float64), np.asarray(mask)
    if logits.ndim == 2:
        logits, mask = logits[:, None], mask[:, None]
    t = np.exp(np.clip(log_t, *LOG_T_BOUNDS))[:, None, None]
    present = np.any(mask, -1)  # [N,V]
    zt = np.where(mask, logits / t, 0.0)
    a = np.where(mask, zt, -np.inf)
    a = a - np.where(present[..., None], a.max(-1, keepdims=True), 0.0)
    e = np.where(mask, np.exp(a), 0.0)
    s = e / np.maximum(e.sum(-1, keepdims=True), 1e-300)
    ds = s * (np.sum(s * zt, -1, keepdims=True) - zt)
    nv = present.sum(1)[:, None]
    p, dp = s.sum(1) / nv, ds.sum(1) / nv
    m = np.any(mask, 1)
    safe = np.where(m, p, 1.0) + 1e-12
    loss = -np.sum(target * np.log(safe), -1)
    grad = -np.sum(np.where(m, target * dp / safe, 0.0), -1)
    return loss, grad


def _feature_fit(logits, mask, target, types, x, lam: float, kinds: list[str]) -> dict:
    """log T = u_type + w · x (x standardised); minimises sum NLL + lam |w|^2 (lam = inf:
    w = 0, i.e. one free temperature per type)."""
    mean, std = x.mean(0), x.std(0)
    std = np.where(std > 1e-9, std, 1.0)
    xs = (x - mean) / std
    onehot = (np.asarray(types)[:, None] == np.asarray(kinds)[None]).astype(float)
    d = 0 if np.isinf(lam) else xs.shape[1]

    def f(theta):
        u, w = theta[: len(kinds)], theta[len(kinds) :]
        log_t = onehot @ u + (xs @ w if d else 0.0)
        loss, g = nll_and_grad(logits, mask, target, log_t)
        grad = np.r_[onehot.T @ g, xs.T @ g + 2 * lam * w if d else np.zeros(0)]
        return float(loss.sum() + (lam * w @ w if d else 0.0)), grad

    res = minimize(f, np.full(len(kinds) + d, np.log(1.2)) * np.r_[np.ones(len(kinds)),
                   np.zeros(d)], jac=True, method="L-BFGS-B")  # fmt: skip
    u, w = res.x[: len(kinds)], res.x[len(kinds) :]
    return {"type_log_t": dict(zip(kinds, u.tolist(), strict=True)),
            "w": (w if d else np.zeros(xs.shape[1])).tolist(), "mean": mean.tolist(),
            "std": std.tolist(), "features": list(FEATURES)}  # fmt: skip


def feature_log_t(fit: dict, types, x) -> np.ndarray:
    """The slots' log T under a ``features`` fit (an unseen type: log T = 0 + w · x)."""
    xs = (np.asarray(x) - np.asarray(fit["mean"])) / np.asarray(fit["std"])
    u = np.asarray([fit["type_log_t"].get(k, 0.0) for k in types])
    return np.clip(u + xs @ np.asarray(fit["w"]), *LOG_T_BOUNDS)


def fit_features(logits, mask, target, types, x, *, lambdas=LAMBDA_GRID, folds=5,
                 seed=0) -> dict:  # fmt: skip
    """Feature-conditioned temperature ("thermometer-lite"): ridge strength by K-fold CV
    (ties: more shrinkage; inf = per type), then a refit on everything."""
    types, x = np.asarray(types), np.asarray(x, np.float64)
    kinds = sorted(set(types.tolist()))
    if len(lambdas) == 1:
        lam = lambdas[0]
        return _feature_fit(logits, mask, target, types, x, lam, kinds) | {
            "grouping": "features", "lambda": lam if np.isfinite(lam) else "inf"}  # fmt: skip
    fold = np.random.default_rng(seed).permutation(len(types)) % folds
    cv = {}
    for lam in lambdas:
        total = 0.0
        for k in range(folds):
            tr, te = fold != k, fold == k
            fit = _feature_fit(logits[tr], mask[tr], target[tr], types[tr], x[tr], lam, kinds)
            loss, _ = nll_and_grad(logits[te], mask[te], target[te],
                                   feature_log_t(fit, types[te], x[te]))  # fmt: skip
            total += float(loss.sum())
        cv[str(lam)] = total / len(types)
    best = min(reversed(lambdas), key=lambda v: cv[str(v)])
    fit = _feature_fit(logits, mask, target, types, x, best, kinds)
    return fit | {"grouping": "features", "lambda": best if np.isfinite(best) else "inf",
                  "cv_nll": cv, "folds": folds}  # fmt: skip


READOUT_DEFAULTS = {"orders": 1, "noul_order": "shuffled", "grouping": "type"}


def artifact(
    fit: dict,
    *,
    checkpoint: str,
    data: str,
    template: int,
    backend: str,
    orders: int = 1,
    noul_order: str = "shuffled",
) -> dict:
    if fit.get("grouping", "type") not in GROUPINGS:
        raise ValueError(f"grouping {fit['grouping']!r} is not servable (one of {GROUPINGS})")
    body = {
        "grouping": fit.get("grouping", "type"),
        "temperatures": fit["temperatures"],
        "fit": fit,
        "checkpoint": checkpoint,
        "data": data,
        "template_version": template,
        "backend": backend,
        "orders": orders,
        "noul_order": noul_order,
    }
    return {**body, "identity": fingerprint(body)}


def validate(
    art: dict,
    *,
    checkpoint: str,
    template: int,
    backend: str,
    orders: int = 1,
    noul_order: str = "shuffled",
) -> dict[str, float]:
    body = {k: v for k, v in art.items() if k != "identity"}
    if fingerprint(body) != art.get("identity"):
        raise ValueError("calibration artifact is corrupt")
    for key, want in (
        ("checkpoint", checkpoint),
        ("template_version", template),
        ("backend", backend),
        ("orders", orders),
        ("noul_order", noul_order),
    ):
        if art.get(key, READOUT_DEFAULTS.get(key)) != want:
            got = art.get(key, READOUT_DEFAULTS.get(key))
            raise ValueError(f"calibration was fitted for {key}={got!r}, not {want!r}")
    if art.get("grouping", "type") not in GROUPINGS:
        raise ValueError(f"unknown calibration grouping {art['grouping']!r}")
    return art["temperatures"]


def fit_from_predictions(
    records: list[dict], *, grouping: str = DEFAULT_GROUPING, min_slots: int = 50
) -> dict:
    """A :func:`fit_grouped` fit on prediction dumps' raw logits (:mod:`tjev.eval.predictions`;
    offline, no model): e.g. a *calibration pool* of held-out sources (the held-out
    generators, sources the run never trained on) instead of the in-distribution
    calibration split. ``fit.pool`` records the pool's sources."""
    from collections import Counter

    from tjev.eval.predictions import arrays

    a = arrays(records)
    fit = fit_grouped(a["logits"], a["mask"], a["target"], list(a["type"]), list(a["lang"]),
                      grouping=grouping, min_slots=min_slots)  # fmt: skip
    return fit | {"pool": {"n": len(records), "sources": dict(Counter(a["source"]))}}
