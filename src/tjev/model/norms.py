"""Qwen3.5 RMSNorms: zero-centred (decoder, q/k) and gated (GDN output), in fp32."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tjev.model.params import Frozen


def rms_norm(x: jax.Array, eps: float) -> jax.Array:
    x = x.astype(jnp.float32)
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


class RMSNorm(nnx.Module):
    """Qwen3.5 zero-centred RMSNorm: (x̂ · (1 + w)) computed in fp32, cast back."""

    def __init__(self, weight: np.ndarray | jax.Array, eps: float):
        self.weight = Frozen(jnp.asarray(weight, jnp.float32))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        return (rms_norm(x, self.eps) * (1.0 + self.weight[...])).astype(x.dtype)


class GatedRMSNorm(nnx.Module):
    """GDN output norm (plain w, HF ``Qwen3_5RMSNormGated``): w · x̂ · silu(z)."""

    def __init__(self, weight: np.ndarray | jax.Array, eps: float):
        self.weight = Frozen(jnp.asarray(weight, jnp.float32))
        self.eps = eps

    def __call__(self, x: jax.Array, gate: jax.Array) -> jax.Array:
        dtype = x.dtype
        normed = rms_norm(x, self.eps).astype(dtype).astype(jnp.float32)
        return (self.weight[...] * normed * jax.nn.silu(gate.astype(jnp.float32))).astype(dtype)
