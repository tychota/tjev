"""Gated DeltaNet delta rule: one entry point, one module per implementation.

================  =================  ======================================================
impl              module             notes
================  =================  ======================================================
chunked           :mod:`.xla`        chunked WY/UT form, the reference (any backend)
recurrent         :mod:`.xla`        token-by-token scan (tests; decode)
pallas_tpu        :mod:`.pallas_tpu` fused Pallas forward, Pallas reverse-recurrence bwd
pallas_tpu_split  :mod:`.pallas_tpu` chunk-local terms in XLA around the Pallas recurrence
================  =================  ======================================================

Inputs arrive as the model prepares them: q and k L2-normalised (q scaled by Dk^-1/2),
g ≤ 0 the per-token log decay, beta the step size; the state restarts at every segment.
"""

from __future__ import annotations

from typing import Literal

import jax

from .xla import PRECISIONS, chunk_gated_delta_rule, l2norm, recurrent_gated_delta_rule

GDNImpl = Literal["chunked", "recurrent", "pallas_tpu", "pallas_tpu_split"]
IMPLEMENTATIONS: tuple[GDNImpl, ...] = ("chunked", "recurrent", "pallas_tpu", "pallas_tpu_split")


def gated_delta_rule(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    segment_ids: jax.Array,
    *,
    impl: GDNImpl = "chunked",
    chunk: int = 64,
    precision: str = "high",
) -> jax.Array:
    """q,k [B,T,H,Dk], v [B,T,H,Dv], g,beta [B,T,H], segment_ids [B,T] → out [B,T,H,Dv] fp32."""
    match impl:
        case "chunked":
            out, _ = chunk_gated_delta_rule(
                q, k, v, g, beta, segment_ids, chunk=chunk, precision=precision
            )
            return out
        case "recurrent":
            out, _ = recurrent_gated_delta_rule(q, k, v, g, beta, segment_ids)
            return out
        case "pallas_tpu" | "pallas_tpu_split":
            from tjev.kernels.sharding import batch_parallel

            from .pallas_tpu import chunk_gated_delta_rule_tpu

            def rule(*xs: jax.Array) -> jax.Array:
                return chunk_gated_delta_rule_tpu(
                    *xs, chunk=chunk, precision=precision, fused=impl == "pallas_tpu"
                )

            return batch_parallel(rule, q, k, v, g, beta, segment_ids)
    raise ValueError(f"gated delta rule impl must be one of {IMPLEMENTATIONS}, not {impl!r}")


__all__ = [
    "IMPLEMENTATIONS",
    "PRECISIONS",
    "GDNImpl",
    "chunk_gated_delta_rule",
    "gated_delta_rule",
    "l2norm",
    "recurrent_gated_delta_rule",
]
