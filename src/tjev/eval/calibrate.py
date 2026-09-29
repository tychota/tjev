"""Per-type temperature scaling fitted by NLL on held-out in-distribution slots.

T=1 is always a candidate, so a fit never makes held-out NLL worse than uncalibrated.
The artifact records what it was fitted on (checkpoint identity, data fingerprint,
template version, backend) and refuses to be applied to anything else.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from tjev.eval.metrics import softmax
from tjev.utils import fingerprint


def nll(logits, mask, target, temperature: float) -> float:
    probs = softmax(logits, mask, temperature)
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


def artifact(fit: dict, *, checkpoint: str, data: str, template: int, backend: str) -> dict:
    body = {
        "temperatures": fit["temperatures"],
        "fit": fit,
        "checkpoint": checkpoint,
        "data": data,
        "template_version": template,
        "backend": backend,
    }
    return {**body, "identity": fingerprint(body)}


def validate(art: dict, *, checkpoint: str, template: int, backend: str) -> dict[str, float]:
    body = {k: v for k, v in art.items() if k != "identity"}
    if fingerprint(body) != art.get("identity"):
        raise ValueError("calibration artifact is corrupt")
    for key, want in (
        ("checkpoint", checkpoint),
        ("template_version", template),
        ("backend", backend),
    ):
        if art[key] != want:
            raise ValueError(f"calibration was fitted for {key}={art[key]!r}, not {want!r}")
    return art["temperatures"]
