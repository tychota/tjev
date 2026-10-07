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
        """x, gate [..., H, D]: per-head norm over D."""
        dtype = x.dtype
        normed = rms_norm(x, self.eps).astype(dtype).astype(jnp.float32)
        return (self.weight[...] * normed * jax.nn.silu(gate.astype(jnp.float32))).astype(dtype)

    def flat(self, x: jax.Array, gate: jax.Array, heads: int) -> jax.Array:
        """The same norm on the heads side by side: x, gate [..., H·D] (``compute.gdn_norm=
        flat``). The per-head mean square and its broadcast back to the H·D lanes are two
        fp32 matmuls with the 0/1 head-membership matrix E [H·D, H] (exact at HIGHEST: E's
        entries are 1), so no op ever views the activations as [..., H, D]: on TPU that view
        is a relayout of the (8, 128) tiles, f32 copies in the forward, remat and backward.
        """
        dtype = x.dtype
        width = x.shape[-1]
        dim = width // heads
        hi = jax.lax.Precision.HIGHEST
        member = (np.arange(width)[:, None] // dim == np.arange(heads)[None, :]).astype(np.float32)
        member = jnp.asarray(member)
        xf = x.astype(jnp.float32)
        mean_sq = jnp.matmul(xf * xf, member, precision=hi) * (1.0 / dim)  # [..., H]
        # the broadcast back too: jnp.repeat (a [.., H, 1] → [.., H, D] view) brings the
        # relayout copies back (AOT v6e: 3 f32 copies per GDN layer)
        scale = jnp.matmul(jax.lax.rsqrt(mean_sq + self.eps), member.T, precision=hi)
        normed = (xf * scale).astype(dtype).astype(jnp.float32)
        weight = jnp.tile(self.weight[...], heads)  # [D] per head → [H·D]
        return (weight * normed * jax.nn.silu(gate.astype(jnp.float32))).astype(dtype)
