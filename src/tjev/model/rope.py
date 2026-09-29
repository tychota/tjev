"""Partial rotary position embedding (Qwen3.5 text: MRoPE with equal position grids)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


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
