"""Loss: soft-target cross-entropy + λ·Brier over the answer letters, summed per slot.

Sums are returned un-normalised: the step divides them once, after gradient accumulation,
by the expected number of slots per step (see :mod:`tjev.train.step`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from tjev.data.pack import Batch


def masked_log_softmax(logits: jax.Array, mask: ArrayLike) -> jax.Array:
    logits = jnp.where(mask, logits, -jnp.inf)
    out = jax.nn.log_softmax(logits, axis=-1)
    return jnp.where(mask, out, 0.0)


def slot_losses(
    logits: jax.Array, batch: Batch, *, brier_weight: float
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-slot terms [R,S]: (loss, nll, brier, correct)."""
    mask = batch.label_mask
    logp = masked_log_softmax(logits, mask)
    probs = jnp.where(mask, jnp.exp(logp), 0.0)
    target = batch.target
    nll = -jnp.sum(target * logp, axis=-1)
    brier = jnp.sum((probs - target) ** 2, axis=-1)
    loss = nll + brier_weight * brier
    # expected correctness under a soft target (== argmax match for one-hot): eval.metrics
    pred = jnp.argmax(jnp.where(mask, logits, -jnp.inf), -1)
    correct = jnp.take_along_axis(target, pred[..., None], -1)[..., 0].astype(jnp.float32)
    return loss, nll, brier, correct


def loss_sums(
    logits: jax.Array, batch: Batch, *, brier_weight: float
) -> tuple[jax.Array, dict[str, jax.Array]]:
    loss, nll, brier, correct = slot_losses(logits, batch, brier_weight=brier_weight)
    w = batch.weight
    sums = {
        "loss": jnp.sum(w * loss),
        "nll": jnp.sum(w * nll),
        "brier": jnp.sum(w * brier),
        "correct": jnp.sum(w * correct),
        "weight": jnp.sum(w),
        "tokens": jnp.sum(batch.segment_ids > 0).astype(jnp.float32),
    }
    return sums["loss"], sums
