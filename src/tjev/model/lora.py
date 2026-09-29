"""Frozen projections with low-rank adapters (LoRA, rsLoRA scaling), and their factory."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.ad_checkpoint import checkpoint_name

from tjev.config import LoRASpec
from tjev.model.params import Frozen


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


class LinearFactory:
    """Creates (possibly LoRA-adapted) projections from stacked HF-orientation weights."""

    def __init__(self, lora: LoRASpec | None, dtype: jnp.dtype, rngs: nnx.Rngs):
        self.lora, self.dtype, self.rngs = lora, dtype, rngs

    def __call__(self, name: str, weight: jax.Array) -> Linear:
        # HF stores [out, in]; NNX layout is [in, out]. Leading (stack) axes are kept.
        kernel = jnp.swapaxes(jnp.asarray(weight), -1, -2)
        lora = self.lora
        adapted = lora is not None and lora.rank > 0 and name in lora.targets
        return Linear(
            kernel,
            dtype=self.dtype,
            rank=lora.rank if lora is not None and adapted else 0,
            scale=lora.scale if lora is not None and adapted else 1.0,
            rngs=self.rngs,
        )
