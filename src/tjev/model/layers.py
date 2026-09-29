"""Leaf layers. Frozen base weights are a non-Param Variable, adapters are ``nnx.LoRAParam``.

Every layer accepts arrays with extra *leading* axes (stacked layers): the model scans over
those axes, so inside the forward pass each layer sees single-layer arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.ad_checkpoint import checkpoint_name


class Frozen(nnx.Variable):
    """Pretrained weight: never differentiated, never touched by the optimizer."""


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


class Linear(nnx.Module):
    """y = x W (+ s · x A B). W is [..., in, out] (NNX layout), frozen; A and B train.

    B starts at zero, so an adapted model is initially identical to the base. The adapters
    are fp32 and cast to the compute dtype for the matmuls.
    """

    def __init__(
        self,
        kernel: np.ndarray | jax.Array,
        *,
        dtype: jnp.dtype,
        rank: int = 0,
        scale: float = 1.0,
        rngs: nnx.Rngs | None = None,
    ):
        self.kernel = Frozen(jnp.asarray(kernel, dtype))
        self.dtype = dtype
        self.rank = rank
        self.scale = scale
        if rank:
            if rngs is None:
                raise ValueError("LoRA needs rngs")
            *lead, fan_in, fan_out = kernel.shape
            a = jax.random.normal(rngs.params(), (*lead, fan_in, rank), jnp.float32)
            self.lora_a = nnx.LoRAParam(a / np.sqrt(fan_in))
            self.lora_b = nnx.LoRAParam(jnp.zeros((*lead, rank, fan_out), jnp.float32))

    def __call__(self, x: jax.Array, name: str | None = None) -> jax.Array:
        """``name``: a checkpoint_name for the output (remat policies may save it)."""
        x = x.astype(self.dtype)
        y = self.base(x)
        if self.rank:
            with jax.named_scope("lora"):
                a = self.lora_a[...].astype(self.dtype)
                b = self.lora_b[...].astype(self.dtype)
                xa = checkpoint_name(x @ a, "lora_xa")
                y = y + (xa @ b) * jnp.asarray(self.scale, self.dtype)
        return checkpoint_name(y, name) if name else y

    def base(self, x: jax.Array) -> jax.Array:
        return x.astype(self.dtype) @ self.kernel[...]

    def delta(self) -> jax.Array:
        """s·AB in fp32: what export adds to the original HF weight."""
        return self.scale * (self.lora_a[...] @ self.lora_b[...])


def grouped(x: jax.Array, *linears: Linear, names: tuple[str | None, ...] = ()) -> list[jax.Array]:
    """``[l(x) for l in linears]`` for projections that share their input.

    The base matmuls stay separate (each already runs at the MXU roofline); the LoRA ``A``
    matrices are concatenated so x is read once by one wider low-rank matmul (r·n columns
    instead of n narrow r-column ones), in the forward and the backward. Parameters are
    unchanged: one ``lora_a`` / ``lora_b`` per projection. ``names``: optional
    checkpoint_names for the outputs (remat policies may save them).
    """
    dtype = linears[0].dtype
    x = x.astype(dtype)
    ys = [layer.base(x) for layer in linears]
    adapted = [i for i, layer in enumerate(linears) if layer.rank]
    if adapted:
        with jax.named_scope("lora"):
            a = jnp.concatenate([linears[i].lora_a[...].astype(dtype) for i in adapted], axis=-1)
            xa = checkpoint_name(x @ a, "lora_xa")
            splits = np.cumsum([linears[i].rank for i in adapted])[:-1].tolist()
            for i, hi in zip(adapted, jnp.split(xa, splits, axis=-1), strict=True):
                layer = linears[i]
                b = layer.lora_b[...].astype(dtype)
                ys[i] = ys[i] + (hi @ b) * jnp.asarray(layer.scale, dtype)
    return [
        checkpoint_name(y, n) if n else y
        for y, n in zip(ys, names or (None,) * len(ys), strict=True)
    ]


def rotary(x: jax.Array, positions: jax.Array, rotary_dim: int, theta: float) -> jax.Array:
    """Partial RoPE on the first ``rotary_dim`` dims (rotate-half layout, fp32 angles).

    Qwen3.5's interleaved MRoPE reduces to this for text: all three position grids equal.
    x [B,T,H,D]; positions [B,T].
    """
    inv = 1.0 / (theta ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
    angles = positions.astype(jnp.float32)[..., None] * inv  # [B,T,rot/2]
    angles = jnp.concatenate([angles, angles], axis=-1)[:, :, None, :]
    cos = jnp.cos(angles).astype(x.dtype)
    sin = jnp.sin(angles).astype(x.dtype)
    rot, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    rotated = jnp.concatenate([-rot[..., half:], rot[..., :half]], axis=-1)
    return jnp.concatenate([rot * cos + rotated * sin, rest], axis=-1)
