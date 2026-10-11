"""Loss: soft-target cross-entropy + λ·Brier over the answer letters, summed per slot.

Sums are returned un-normalised: the step divides them once, after gradient accumulation,
by the expected number of slots per step (see :mod:`tjev.train.step`).

Regularisers (``loss.*``, :class:`tjev.config.LossSpec`; all off by default, and then the
loss is today's expression, bit for bit):

* **Label smoothing** ε: the *training* target is (1 − ε)·t + ε·u, u uniform over the
  displayed letters, on the slots of the selected sources. It is a soft CE (and Brier)
  against that smoothed target, nothing else.
* **Hard targets**: the training target is the one-hot argmax of t (the control arm of the
  soft-target CE).
* **KL to the base** w: + w·KL(p_base ‖ p) over the displayed letters, ``base_logits`` the
  frozen model's (adapters scaled to 0), on the slots of the selected sources.
* **Gated teacher targets** λ, g: on a slot whose teacher (``batch.teacher``) puts ≥ g on a
  letter other than the gold one (the argmax of t), the training target is
  (1 − λ)·t + λ·teacher; elsewhere t (:func:`teacher_mixed`).
* **CogCalib** w, t (arXiv 2505.20903, its CoECP variant): + w·𝟙[nll ≤ t]·(−H(p)), H the
  entropy of the predicted letters, nll the slot's CE against the data's target: the
  slots the model already knows are pushed away from p → 1 (:func:`cogcalib_gate`).

The logged ``nll`` / ``brier`` / ``accuracy`` are always against the data's target t, so
the paired online loss of the campaign planner compares arms on the same quantity.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from tjev.config import LossSpec
from tjev.data.pack import Batch


def source_table(patterns: tuple[str, ...], names: list[str]) -> np.ndarray | None:
    """[n_sources] float32 0/1: the sources (vocab order) a ``loss.*_sources`` selector
    picks (fnmatch patterns, or ``block:<block>``); None for an empty selector (all)."""
    if not patterns:
        return None
    from tjev.data.mix import block_of_source

    def picked(name: str) -> bool:
        return any(block_of_source(name) == p[6:] if p.startswith("block:") else fnmatchcase(name, p)
                   for p in patterns)  # fmt: skip

    table = np.asarray([picked(n) for n in names], np.float32)
    if not table.any():
        raise ValueError(f"source selector {list(patterns)} matches no source")
    return table


def masked_log_softmax(logits: jax.Array, mask: ArrayLike) -> jax.Array:
    logits = jnp.where(mask, logits, -jnp.inf)
    out = jax.nn.log_softmax(logits, axis=-1)
    return jnp.where(mask, out, 0.0)


def smoothed(target: jax.Array, mask: ArrayLike, eps: float | jax.Array) -> jax.Array:
    """(1 − ε)·target + ε·uniform over the displayed letters (``mask``)."""
    m = jnp.asarray(mask, jnp.float32)
    uniform = m / jnp.maximum(m.sum(-1, keepdims=True), 1.0)
    return (1.0 - eps) * target + eps * uniform


def hard(target: jax.Array, mask: ArrayLike) -> jax.Array:
    """The one-hot argmax of a (soft) target over the displayed letters."""
    pick = jnp.argmax(jnp.where(mask, target, -1.0), -1)
    return jax.nn.one_hot(pick, target.shape[-1], dtype=target.dtype) * jnp.asarray(
        mask, target.dtype
    )


def teacher_gate(target: jax.Array, teacher: jax.Array, mask: ArrayLike,
                 gate: float) -> jax.Array:  # fmt: skip
    """[...] 0/1: the teacher puts ≥ ``gate`` on a displayed letter other than the gold one
    (the argmax of ``target``). A slot without a teacher (all 0) never passes."""
    gold = jnp.argmax(jnp.where(mask, target, -1.0), -1)
    other = jnp.where(mask, teacher, 0.0) * (1.0 - jax.nn.one_hot(gold, target.shape[-1]))
    return (jnp.max(other, -1) >= gate).astype(jnp.float32)


def teacher_mixed(target: jax.Array, teacher: jax.Array, mask: ArrayLike, *, mix: float,
                  gate: float) -> tuple[jax.Array, jax.Array]:  # fmt: skip
    """((1 − λ)·target + λ·teacher where the gate passes, else target; the gate [...])."""
    on = teacher_gate(target, teacher, mask, gate)
    lam = mix * on[..., None]
    return (1.0 - lam) * target + lam * jnp.where(mask, teacher, 0.0), on


def entropy(logp: jax.Array, probs: jax.Array) -> jax.Array:
    """H(p) per slot over the displayed letters (masked letters have p = 0, log p = 0)."""
    return -jnp.sum(probs * logp, axis=-1)


def cogcalib_gate(nll: jax.Array, threshold: float | jax.Array) -> jax.Array:
    """CogCalib's indicator 𝟙[nll ≤ t] (Eq. 2): a step function, no gradient flows through it."""
    return (nll <= threshold).astype(jnp.float32)


def gate_nll(spec: LossSpec, nll: jax.Array, base_logits: jax.Array | None,
             batch: Batch) -> jax.Array:  # fmt: skip
    """The per-slot NLL CogCalib gates on: the trained model's (``cogcalib_gate=running``),
    or the base model's against the same target (``base``; constant, no gradient)."""
    if spec.cogcalib_gate != "base":
        return nll
    if base_logits is None:
        raise ValueError("loss.cogcalib_gate=base: the step passes no base logits")
    base = jax.lax.stop_gradient(masked_log_softmax(base_logits, batch.label_mask))
    p_base = jnp.where(batch.label_mask, jnp.exp(base), 0.0)
    return soft_ce_brier(base, p_base, batch.target)[0]


def soft_ce_brier(
    logp: jax.Array, probs: jax.Array, target: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Per slot: CE against a soft ``target``, and the Brier score against it."""
    return -jnp.sum(target * logp, axis=-1), jnp.sum((probs - target) ** 2, axis=-1)


def kl_to(logp: jax.Array, base_logits: jax.Array, mask: ArrayLike) -> jax.Array:
    """KL(p_base ‖ p) per slot over the displayed letters (p_base held constant)."""
    base = jax.lax.stop_gradient(masked_log_softmax(base_logits, mask))
    p_base = jnp.where(mask, jnp.exp(base), 0.0)
    return jnp.sum(p_base * (base - logp), axis=-1)


def slot_losses(
    logits: jax.Array, batch: Batch, *, brier_weight: float, loss: LossSpec | None = None,
    smooth_on: jax.Array | None = None, kl_on: jax.Array | None = None,
    base_logits: jax.Array | None = None, cogcalib_threshold: float | jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:  # fmt: skip
    """Per-slot terms [R,S]: (loss, nll, brier, correct, extras). ``smooth_on`` /
    ``kl_on``: [R,S] 0/1, the slots the label smoothing / the KL apply to (None: all).
    ``extras``: the per-slot ``kl``, ``teacher_on``, ``cogcalib_on`` and ``cogcalib``
    (the penalty w·gate·(−H)) of the options that are on. ``cogcalib_threshold``: t (the
    running quantile of the step); None: ``loss.cogcalib_threshold``."""
    mask = batch.label_mask
    logp = masked_log_softmax(logits, mask)
    probs = jnp.where(mask, jnp.exp(logp), 0.0)
    target = batch.target
    nll, brier = soft_ce_brier(logp, probs, target)
    spec = loss or LossSpec()
    train_target = target
    extras: dict[str, jax.Array] = {}
    if spec.hard_targets:
        train_target = hard(train_target, mask)
    if spec.teacher_mix:
        if batch.teacher is None:
            raise ValueError("loss.teacher_mix: the batch carries no teacher (Batch.teacher)")
        train_target, extras["teacher_on"] = teacher_mixed(
            train_target, batch.teacher, mask, mix=spec.teacher_mix, gate=spec.teacher_gate
        )
    if spec.label_smoothing:
        eps = spec.label_smoothing * (1.0 if smooth_on is None else smooth_on[..., None])
        train_target = smoothed(train_target, mask, eps)
    if train_target is target:
        out = nll + brier_weight * brier  # today's loss, bit for bit
    else:
        t_nll, t_brier = soft_ce_brier(logp, probs, train_target)
        out = t_nll + brier_weight * t_brier
    if spec.kl_base_weight and base_logits is not None:
        kl = extras["kl"] = kl_to(logp, base_logits, mask)
        out = out + spec.kl_base_weight * (kl if kl_on is None else kl_on * kl)
    if spec.cogcalib_weight:
        t = spec.cogcalib_threshold if cogcalib_threshold is None else cogcalib_threshold
        on = extras["cogcalib_on"] = cogcalib_gate(gate_nll(spec, nll, base_logits, batch), t)
        extras["cogcalib"] = spec.cogcalib_weight * on * -entropy(logp, probs)
        out = out + extras["cogcalib"]
    # expected correctness under a soft target (== argmax match for one-hot): eval.metrics
    pred = jnp.argmax(jnp.where(mask, logits, -jnp.inf), -1)
    correct = jnp.take_along_axis(target, pred[..., None], -1)[..., 0].astype(jnp.float32)
    return out, nll, brier, correct, extras


def loss_sums(
    logits: jax.Array, batch: Batch, *, brier_weight: float, loss: LossSpec | None = None,
    smooth_on: jax.Array | None = None, kl_on: jax.Array | None = None,
    base_logits: jax.Array | None = None, cogcalib_threshold: float | jax.Array | None = None,
    slot_nll: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array]]:  # fmt: skip
    """(Σ w·loss, the weighted sums). ``slot_nll``: also the per-slot NLL [R,S] (NaN on
    empty slots) as ``sums["slot_nll"]``, for the running CogCalib quantile (not a sum)."""
    total, nll, brier, correct, extras = slot_losses(
        logits, batch, brier_weight=brier_weight, loss=loss, smooth_on=smooth_on, kl_on=kl_on,
        base_logits=base_logits, cogcalib_threshold=cogcalib_threshold,
    )  # fmt: skip
    w = batch.weight
    sums = {
        "loss": jnp.sum(w * total),
        "nll": jnp.sum(w * nll),
        "brier": jnp.sum(w * brier),
        "correct": jnp.sum(w * correct),
        "weight": jnp.sum(w),
        "tokens": jnp.sum(batch.segment_ids > 0).astype(jnp.float32),
    }
    if "kl" in extras:  # KL to the base on every slot (the penalty applies to kl_on's)
        sums["kl_base"] = jnp.sum(w * extras["kl"])
    for key in ("teacher_on", "cogcalib_on", "cogcalib"):  # slots gated, the penalty added
        if key in extras:
            sums[key] = jnp.sum(w * extras[key])
    if slot_nll:  # the NLL the gate compares (the base model's with cogcalib_gate=base)
        g = gate_nll(loss or LossSpec(), nll, base_logits, batch)
        sums["slot_nll"] = jnp.where(w > 0, g, jnp.nan)
    return sums["loss"], sums
