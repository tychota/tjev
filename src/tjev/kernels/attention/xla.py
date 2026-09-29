"""Causal, segment-masked GQA attention in XLA, one query block at a time.

Each block only reads keys up to its own end (causal) and is rematerialised on its own, so
peak memory is O(block × T) instead of O(T²). Padding (segment 0) attends to earlier
padding and to itself: no softmax row is ever empty.

Shapes: q [B,T,H,D], k/v [B,T,KV,D], segment_ids [B,T] → [B,T,H,D] in q's dtype.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def blocked_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: jax.Array,
    *,
    block: int = 512,
    remat: bool = True,
) -> jax.Array:
    """``block`` = 0 (or ≥ T): one block. ``remat``: recompute each block in the backward."""
    length = q.shape[1]
    block = min(block, length) if block else length
    outs = []
    for start in range(0, length, block):
        end = min(start + block, length)

        def one(q_blk, k_ctx, v_ctx, seg, start=start, end=end):
            same = seg[:, start:end, None] == seg[:, None, :end]
            qi = jnp.arange(start, end)[:, None]
            ki = jnp.arange(end)[None, :]
            mask = (same & (ki <= qi)) | (ki == qi)  # diagonal: no empty rows on padding
            return jax.nn.dot_product_attention(q_blk, k_ctx, v_ctx, mask=mask[:, None])

        fn = jax.checkpoint(one, prevent_cse=False) if remat and block < length else one
        outs.append(fn(q[:, start:end], k[:, :end], v[:, :end], segment_ids))
    return outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=1)
