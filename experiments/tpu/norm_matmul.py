"""Experimental: RMSNorm fused into the projection that reads it, as one Pallas TPU kernel.

Every decoder layer computes ``RMSNorm(x) @ W`` twice (the mixer's in-projections after the
input norm, the MLP's gate/up after the post-attention norm). XLA usually fuses the norm's
elementwise work into its neighbours, but the normalised activations can still make a round
trip through HBM before the matmul: [M, K] in bf16 written and read back once per layer.
This kernel keeps them in VMEM: each program loads a [block_m, K] row block of x (the full
contraction axis), normalises it in fp32, and multiplies it with a [K, block_n] block of W.
The norm is recomputed for every column block (cheap vector work next to the MXU matmul).

The backward is XLA autodiff of the reference (a custom VJP), so the kernel can be dropped
into training to measure end-to-end speed; only the forward is fused.

Status: correct in interpret mode and lowers for TPU (tests/experiments/); not yet timed
on a TPU. Measure with ``experiments/tpu/bench_norm_matmul.py`` on the VM before using it.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def reference(x: jax.Array, gamma: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Qwen3.5's zero-centred RMSNorm then the projection: (x̂ · (1 + γ)) @ W, as the model
    computes it (norm in fp32, cast to W's dtype, fp32 accumulation, output in x's dtype)."""
    xf = x.astype(jnp.float32)
    h = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    h = (h * (1.0 + gamma.astype(jnp.float32))).astype(w.dtype)
    return jnp.dot(h, w, preferred_element_type=jnp.float32).astype(x.dtype)


def _kernel(x_ref, g_ref, w_ref, o_ref, *, eps: float):
    x = x_ref[...].astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    h = (x * inv * (1.0 + g_ref[...].astype(jnp.float32))).astype(w_ref.dtype)
    o_ref[...] = jnp.dot(h, w_ref[...], preferred_element_type=jnp.float32).astype(o_ref.dtype)


def _block(n: int, preferred: int) -> int:
    size = preferred
    while n % size:
        size //= 2
    if size < 8:
        raise ValueError(f"dimension {n} has no power-of-two block ≥ 8")
    return size


def _forward(x, gamma, w, eps, block_m, block_n):
    m, k = x.shape
    n = w.shape[1]
    bm, bn = _block(m, block_m), _block(n, block_n)
    return pl.pallas_call(
        functools.partial(_kernel, eps=eps),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(m // bm, n // bn),
        in_specs=[
            pl.BlockSpec((bm, k), lambda i, j: (i, 0)),
            pl.BlockSpec((1, k), lambda i, j: (0, 0)),
            pl.BlockSpec((k, bn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=jax.default_backend() != "tpu",
        name="rmsnorm_matmul",
    )(x, gamma.reshape(1, k), w)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def fused_rmsnorm_matmul(x: jax.Array, gamma: jax.Array, w: jax.Array, eps: float = 1e-6,
                         block_m: int = 256, block_n: int = 512) -> jax.Array:  # fmt: skip
    """x [M, K], γ [K], W [K, N] → [M, N] in x's dtype; equals :func:`reference`."""
    return _forward(x, gamma, w, eps, block_m, block_n)


def _fwd(x, gamma, w, eps, block_m, block_n):
    return _forward(x, gamma, w, eps, block_m, block_n), (x, gamma, w)


def _bwd(eps, block_m, block_n, res, dy):
    del block_m, block_n
    x, gamma, w = res
    _, vjp = jax.vjp(lambda a, b, c: reference(a, b, c, eps), x, gamma, w)
    return vjp(dy)


fused_rmsnorm_matmul.defvjp(_fwd, _bwd)
