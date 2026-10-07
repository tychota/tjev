"""Gated DeltaNet delta rule: one entry point, one module per implementation.

================  =================  ======================================================
impl              module             notes
================  =================  ======================================================
chunked           :mod:`.xla`        chunked WY/UT form, the reference (any backend)
recurrent         :mod:`.xla`        token-by-token scan (tests; decode)
pallas_tpu        :mod:`.pallas_tpu` fused Pallas forward, Pallas reverse-recurrence bwd
pallas_tpu_split  :mod:`.pallas_tpu` chunk-local terms in XLA around the Pallas recurrence
pallas_tpu_v2     :mod:`.pallas_tpu_v2` fused Pallas forward *and* backward, heads batched,
                                     GQA through the index maps
pallas_tpu_mega1  :mod:`.pallas_tpu_mega` v2 plus the gates, l2norm(q, k), q scale and GQA in
                                     the kernels (:func:`gdn_core_mega`, from post-conv qkv)
pallas_tpu_mega2  :mod:`.pallas_tpu_mega` mega1 plus the causal conv + SiLU (from the raw
                                     in_proj_qkv output)
================  =================  ======================================================

Inputs arrive as the model prepares them: q and k L2-normalised (q scaled by Dk^-1/2),
g ≤ 0 the per-token log decay, beta the step size; the state restarts at every segment.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from .xla import PRECISIONS, chunk_gated_delta_rule, l2norm, recurrent_gated_delta_rule

GDNImpl = Literal["chunked", "recurrent", "pallas_tpu", "pallas_tpu_split", "pallas_tpu_v2"]
IMPLEMENTATIONS: tuple[GDNImpl, ...] = (
    "chunked", "recurrent", "pallas_tpu", "pallas_tpu_split", "pallas_tpu_v2",
)  # fmt: skip
MEGA = ("pallas_tpu_mega1", "pallas_tpu_mega2")  # the core from the projections: gdn_core_mega


def kernel_inverse(inverse: str) -> str:
    """The in-kernel (I+L)⁻¹ of pallas_tpu_v2 / the megakernels for ``compute.gdn_inverse``:
    doubling (the default) or blocked (exact 16×16 diagonal blocks + doubling levels)."""
    if inverse in ("default", "doubling"):
        return "doubling"
    if inverse == "blocked":
        return "blocked"
    raise ValueError(f"gdn_inverse={inverse} has no in-kernel form: default, doubling, blocked")


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
    inverse: str = "default",
    heads_per_step: int = 0,
    vmem_fraction: float = 0.0,
) -> jax.Array:
    """q,k [B,T,Hk,Dk], v [B,T,H,Dv], g,beta [B,T,H], segment_ids [B,T] → out [B,T,H,Dv]
    (fp32; pallas_tpu_v2: v's dtype). GQA: Hk divides H, value head h reads key head
    h // (H/Hk); pallas_tpu_v2 does it in its index maps, the others repeat q and k here.

    ``inverse``: how the chunked form solves its UT transform (I+L)⁻¹ [vβ, kβe^γ]: ``solve``,
    ``doubling``, ``blocked`` (xla.solve_unit_lower), or ``default``: each implementation's
    own (chunked: solve; the Pallas paths' XLA terms: doubling; pallas_tpu_v2: its in-kernel
    inverse, :func:`kernel_inverse`). ``heads_per_step`` and ``vmem_fraction``: pallas_tpu_v2's
    heads per grid step (0: 8) and VMEM limit (0: the libtpu flag's)."""
    if q.shape[2] != v.shape[2] and impl != "pallas_tpu_v2":
        ratio = v.shape[2] // q.shape[2]
        q, k = jnp.repeat(q, ratio, axis=2), jnp.repeat(k, ratio, axis=2)
    match impl:
        case "chunked":
            out, _ = chunk_gated_delta_rule(
                q, k, v, g, beta, segment_ids, chunk=chunk, precision=precision,
                inverse="solve" if inverse == "default" else inverse,
            )  # fmt: skip
            return out
        case "recurrent":
            out, _ = recurrent_gated_delta_rule(q, k, v, g, beta, segment_ids)
            return out
        case "pallas_tpu" | "pallas_tpu_split":
            from tjev.kernels.sharding import batch_parallel

            from .pallas_tpu import chunk_gated_delta_rule_tpu

            def rule(*xs: jax.Array) -> jax.Array:
                return chunk_gated_delta_rule_tpu(
                    *xs, chunk=chunk, precision=precision, fused=impl == "pallas_tpu",
                    inverse="doubling" if inverse == "default" else inverse,
                )  # fmt: skip

            return batch_parallel(rule, q, k, v, g, beta, segment_ids)
        case "pallas_tpu_v2":
            from tjev.kernels.sharding import batch_parallel

            from .pallas_tpu_v2 import chunk_gated_delta_rule_v2

            def rule_v2(*xs: jax.Array) -> jax.Array:
                return chunk_gated_delta_rule_v2(
                    *xs, chunk=chunk, precision=precision, inverse=kernel_inverse(inverse),
                    heads_per_step=heads_per_step, vmem_fraction=vmem_fraction,
                )  # fmt: skip

            return batch_parallel(rule_v2, q, k, v, g, beta, segment_ids)
    raise ValueError(f"gated delta rule impl must be one of {IMPLEMENTATIONS}, not {impl!r}")


def gdn_core_mega(
    qkv: jax.Array,
    a: jax.Array,
    b: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_weight: jax.Array | None,
    segment_ids: jax.Array,
    *,
    key_heads: int,
    key_dim: int,
    value_heads: int,
    value_dim: int,
    chunk: int = 64,
    precision: str = "high",
    inverse: str = "default",
    heads_per_step: int = 0,
    vmem_fraction: float = 0.0,
) -> jax.Array:
    """The GDN core in one forward and one backward kernel (pallas_tpu_mega1 / mega2): qkv
    [B,T,2·Hk·Dk+H·Dv] (conv_weight [K, ·]: the raw projection, conv + SiLU in the kernel;
    None: post-conv activations), a, b [B,T,H], a_log, dt_bias [H] → [B,T,H,Dv] in qkv's
    dtype. Under data parallelism each device runs its rows (the parameters replicated)."""
    from tjev.kernels.sharding import batch_parallel_with

    from .pallas_tpu_mega import gdn_megakernel

    def rule(qkv, a, b, segment_ids, a_log, dt_bias, *weight):
        return gdn_megakernel(
            qkv, a, b, a_log, dt_bias, weight[0] if weight else None, segment_ids,
            key_heads=key_heads, key_dim=key_dim, value_heads=value_heads,
            value_dim=value_dim, chunk=chunk, precision=precision,
            inverse=kernel_inverse(inverse), heads_per_step=heads_per_step,
            vmem_fraction=vmem_fraction,
        )  # fmt: skip

    replicated = (a_log, dt_bias) + (() if conv_weight is None else (conv_weight,))
    return batch_parallel_with(rule, (qkv, a, b, segment_ids), replicated)


__all__ = [
    "IMPLEMENTATIONS",
    "MEGA",
    "PRECISIONS",
    "GDNImpl",
    "chunk_gated_delta_rule",
    "gated_delta_rule",
    "gdn_core_mega",
    "kernel_inverse",
    "l2norm",
    "recurrent_gated_delta_rule",
]
