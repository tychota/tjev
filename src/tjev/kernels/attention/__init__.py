"""Causal, segment-masked GQA attention: one entry point, one module per implementation.

========  ======================  ==========================================================
impl      module                  notes
========  ======================  ==========================================================
xla       :mod:`.xla`             query-blocked ``jax.nn.dot_product_attention`` (reference)
splash    :mod:`.splash_tpu`      JAX's Pallas TPU splash kernels, under shard_map
========  ======================  ==========================================================
"""

from __future__ import annotations

from typing import Literal

import jax

from .xla import blocked_attention

AttentionImpl = Literal["xla", "splash"]
IMPLEMENTATIONS: tuple[AttentionImpl, ...] = ("xla", "splash")


def attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: jax.Array,
    *,
    impl: AttentionImpl = "xla",
    block: int = 512,
    remat: bool = True,
    splash_block: int = 512,
    splash_compute: int = 0,
    splash_fused_bwd: bool = False,
) -> jax.Array:
    """q [B,T,H,D], k/v [B,T,KV,D], segment_ids [B,T] (0 = padding) → [B,T,H,D].

    Token t attends key s iff s ≤ t and both share a segment id. ``block`` and ``remat``
    apply to the XLA path (splash never materialises [T, T]); ``splash_*`` to splash
    (splash_tpu.block_sizes)."""
    match impl:
        case "xla":
            return blocked_attention(q, k, v, segment_ids, block=block, remat=remat)
        case "splash":
            from tjev.kernels.sharding import batch_parallel

            from .splash_tpu import splash_attention

            def run(q: jax.Array, k: jax.Array, v: jax.Array, seg: jax.Array) -> jax.Array:
                return splash_attention(
                    q, k, v, seg, block=splash_block, compute=splash_compute,
                    fused_bwd=splash_fused_bwd,
                )  # fmt: skip

            return batch_parallel(run, q, k, v, segment_ids)
    raise ValueError(f"attention impl must be one of {IMPLEMENTATIONS}, not {impl!r}")


__all__ = ["IMPLEMENTATIONS", "AttentionImpl", "attention", "blocked_attention"]
