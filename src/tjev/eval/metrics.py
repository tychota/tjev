"""Decision metrics on host arrays: accuracy, NLL, KL, Brier, top-label ECE, by group.

Inputs are flat per-slot arrays: logits [N,K] (masked entries ignored), mask [N,K],
target [N,K] (soft allowed) and group labels. Temperatures, when given, are applied
per question type (or per (type, language) cell) before the softmax.

Order averaging: logits and mask may also be [N,V,K], V displayed orderings of one
question each un-permuted to the canonical label order (a view whose mask row is all
False is absent: an item with fewer distinct orderings). The distribution is then the
arithmetic mean of the views' (temperature-scaled) softmaxes, :func:`view_probs`.
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


def label_mask(mask: np.ndarray) -> np.ndarray:
    """The [N,K] label mask of a [N,K] or [N,V,K] (order-averaged) mask."""
    mask = np.asarray(mask)
    return np.any(mask, axis=1) if mask.ndim == 3 else mask


def view_probs(
    logits: np.ndarray, mask: np.ndarray, temperature: np.ndarray | float = 1.0
) -> np.ndarray:
    """[N,K] probabilities: the softmax of [N,K] logits, or for [N,V,K] the mean over the
    present views of each view's softmax (one temperature per item, shared by its views).

    The mean is of probabilities (a mixture), not of log-probabilities. A mixture's NLL
    is at most the views' mean NLL (Jensen), so averaging never makes the expected log
    loss worse than a single ordering on average, and where the orderings disagree the
    result is less confident, which is what position-bias debiasing should do. The
    log-space mean (renormalised geometric mean, softmax of the mean logit) is a product
    of experts: it sharpens where the views agree and, under temperature scaling, is just
    one more logit vector; it is not used.
    """
    logits, mask = np.asarray(logits), np.asarray(mask)
    if logits.ndim == 2:
        return softmax(logits, mask, temperature)
    n, v, k = logits.shape
    present = np.any(mask, axis=-1)  # [N,V]
    if not np.all(np.any(present, axis=1)):
        raise ValueError("every item needs at least one view")
    temps = np.broadcast_to(np.asarray(temperature, np.float64).reshape(-1, 1), (n, v))
    flat_mask = np.where(present[..., None], mask, True)  # absent views: any valid row
    p = softmax(logits.reshape(n * v, k), flat_mask.reshape(n * v, k), temps.reshape(-1))
    p = np.where(present[..., None], p.reshape(n, v, k), 0.0)
    return p.sum(1) / present.sum(1, keepdims=True)


def cell(kind: str, lang: str) -> str:
    return f"{kind}/{lang}"


def lookup(temperatures: dict[str, float] | None, kind: str, lang: str | None = None) -> float:
    """The temperature of a (type, language) slot: its cell's, else its type's, else 1."""
    temps = temperatures or {}
    if lang is not None and cell(kind, lang) in temps:
        return float(temps[cell(kind, lang)])
    return float(temps.get(kind, 1.0))


def slot_temperatures(temperatures, types, langs=None) -> np.ndarray:
    """[N] temperatures of the slots (``langs`` None: per type only)."""
    langs = [None] * len(types) if langs is None else langs
    return np.asarray([lookup(temperatures, k, g) for k, g in zip(types, langs, strict=True)])


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
    """``groups``: name -> per-slot labels (e.g. {'type': [...], 'source': [...]}).
    ``temperatures``: per type, or per (type, language) cell (``"noul/fr"``, applied with
    the slots' ``groups["lang"]``; :func:`lookup`)."""
    type_names = type_names if type_names is not None else groups.get("type")
    temps = np.ones(len(logits))
    if temperatures and type_names is not None:
        temps = slot_temperatures(temperatures, list(type_names), groups.get("lang"))
    probs = view_probs(logits, mask, temps)
    mask = label_mask(mask)
    report = {"all": summarize(probs, target, mask)}
    for name, labels in groups.items():
        labels = np.asarray(labels)
        by = defaultdict(dict)
        for value in sorted(set(labels.tolist())):
            sel = labels == value
            by[value] = summarize(probs[sel], target[sel], mask[sel])
        report[name] = dict(by)
    return report
