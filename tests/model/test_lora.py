"""Leaf layers: grouped LoRA projections equal the per-projection calls."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tjev.model.lora import Linear, grouped


def _layers():
    rngs = nnx.Rngs(0)
    key = jax.random.key(1)
    layers = []
    for i, (fan_out, rank) in enumerate([(24, 4), (8, 0), (16, 4), (4, 4)]):
        kernel = jax.random.normal(jax.random.fold_in(key, i), (12, fan_out))
        layer = Linear(kernel, dtype=jnp.float32, rank=rank, scale=0.5 + i, rngs=rngs)
        if rank:  # non-zero B, so the adapter path is exercised
            layer.lora_b[...] = jax.random.normal(jax.random.fold_in(key, 10 + i), (rank, fan_out))
        layers.append(layer)
    return layers


def test_grouped_matches_separate_calls_and_gradients():
    layers = _layers()
    x = jax.random.normal(jax.random.key(2), (3, 5, 12))
    cot = [jax.random.normal(jax.random.key(3 + i), (3, 5, layer.kernel.shape[-1]))
           for i, layer in enumerate(layers)]  # fmt: skip

    graphdef, lora, rest = nnx.split(nnx.List(layers), nnx.LoRAParam, ...)

    def run(fn):
        def f(lora):
            model = nnx.merge(graphdef, lora, rest)
            ys = fn(x, *model)
            return sum(jnp.sum(y * c) for y, c in zip(ys, cot, strict=True)), ys

        (_, ys), grads = jax.value_and_grad(f, has_aux=True)(lora)
        return ys, grads

    got, got_g = run(grouped)
    want, want_g = run(lambda x, *ls: [layer(x) for layer in ls])
    for a, b in zip(got, want, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5), got_g, want_g)
