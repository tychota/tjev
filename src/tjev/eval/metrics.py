"""Decision metrics on host arrays: accuracy, NLL, KL, Brier, top-label ECE, by group.

Inputs are flat per-slot arrays: logits [N,K] (masked entries ignored), mask [N,K],
target [N,K] (soft allowed) and group labels. Temperatures, when given, are applied
per question type before the softmax.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

ECE_BINS = 15


def softmax(
    logits: np.ndarray, mask: np.ndarray, temperature: np.ndarray | float = 1.0
) -> np.ndarray:
    z = np.where(mask, logits / np.asarray(temperature).reshape(-1, 1), -np.inf)
    z = z - z.max(-1, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    return e / e.sum(-1, keepdims=True)


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float:
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = index == b
        if sel.any():
            total += sel.sum() * abs(confidence[sel].mean() - correct[sel].mean())
    return float(total / len(confidence))


def summarize(probs: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    """Soft targets (base rates, interpolated similarity, annotator distributions) count as
    the *expected* correctness ``target[argmax(probs)]``: a model that outputs the target
    has accuracy = ECE-correctness = its own confidence, so ECE is 0 (with an argmax gold
    it would be ``1 - max(target)``). Identical for one-hot targets.
    ``kl`` = NLL − H(target), 0 for a perfect model whatever the target entropy."""
    n = len(probs)
    if n == 0:
        return {"n": 0}
    eps = 1e-12
    nll = -np.sum(target * np.log(np.where(mask, probs, 1.0) + eps), -1)
    entropy = -np.sum(target * np.log(np.where(target > 0, target, 1.0)), -1)
    brier = np.sum((probs - target) ** 2, -1)
    pred = probs.argmax(-1)
    correct = np.take_along_axis(target, pred[:, None], -1)[:, 0].astype(np.float64)
    confidence = probs.max(-1)
    return {
        "n": int(n),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "kl": float((nll - entropy).mean()),
        "brier": float(brier.mean()),
        "ece": ece(confidence, correct),
        "confidence": float(confidence.mean()),
    }


def grouped_report(
    logits: np.ndarray,
    mask: np.ndarray,
    target: np.ndarray,
    groups: dict[str, list[str]],
    temperatures: dict[str, float] | None = None,
    type_names: list[str] | None = None,
) -> dict:
    """``groups``: name -> per-slot labels (e.g. {'type': [...], 'source': [...]})."""
    type_names = type_names if type_names is not None else groups.get("type")
    temps = np.ones(len(logits))
    if temperatures and type_names is not None:
        temps = np.asarray([temperatures.get(t, 1.0) for t in type_names])
    probs = softmax(logits, mask, temps)
    report = {"all": summarize(probs, target, mask)}
    for name, labels in groups.items():
        labels = np.asarray(labels)
        by = defaultdict(dict)
        for value in sorted(set(labels.tolist())):
            sel = labels == value
            by[value] = summarize(probs[sel], target[sel], mask[sel])
        report[name] = dict(by)
    return report
