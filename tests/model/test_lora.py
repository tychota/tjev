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


def _merged_copies(layers):
    return [
        Linear(layer.kernel[...], dtype=layer.dtype, rank=layer.rank, scale=layer.scale,
               rngs=nnx.Rngs(0), merged=True)
        for layer in layers
    ]  # fmt: skip


def test_merged_lora_matches_unmerged_values_and_gradients():
    """compute.lora_impl=merged: x·(W + s·AB) under the custom VJP equals xW + s·(xA)B in
    fp32, outputs and the gradients of x, every A and B and (when asked) W."""
    layers = _layers()
    merged = _merged_copies(layers)
    for m, layer in zip(merged, layers, strict=True):
        if layer.rank:
            m.lora_a[...], m.lora_b[...] = layer.lora_a[...], layer.lora_b[...]
    x = jax.random.normal(jax.random.key(2), (3, 5, 12))
    cot = [jax.random.normal(jax.random.key(3 + i), (3, 5, layer.kernel.shape[-1]))
           for i, layer in enumerate(layers)]  # fmt: skip

    def run(ls, fn):
        graphdef, params = nnx.split(nnx.List(ls))  # every variable: W's gradient too

        def f(params, x):
            ys = fn(x, *nnx.merge(graphdef, params))
            return sum(jnp.sum(y * c) for y, c in zip(ys, cot, strict=True)), ys

        (_, ys), grads = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(params, x)
        return ys, grads

    for fn in (grouped, lambda x, *ls: [layer(x) for layer in ls]):
        want, want_g = run(layers, fn)
        got, got_g = run(merged, fn)
        for a, b in zip(got, want, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
        jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5),
                     got_g, want_g)  # fmt: skip


def test_merged_lora_never_forms_the_full_weight_gradient():
    """Training differentiates the adapters only: no [in, out] product of x and dy."""
    (layer,) = _merged_copies(_layers()[:1])
    graphdef, lora, rest = nnx.split(layer, nnx.LoRAParam, ...)
    x = jnp.ones((7, 12))

    def loss(lora):
        return jnp.sum(nnx.merge(graphdef, lora, rest)(x) ** 2)

    jaxpr = jax.make_jaxpr(jax.grad(loss))(lora)
    shape = lambda v: tuple(getattr(v.aval, "shape", ()))  # noqa: E731
    dots = [(shape(e.outvars[0]), {d for v in e.invars for d in shape(v)})
            for e in jaxpr.eqns if e.primitive.name == "dot_general"]  # fmt: skip
    # dW = xᵀdy would be a [12, 24] product over the 7 tokens (the merge AB is one over r)
    assert not [shape for shape, dims in dots if shape == (12, 24) and 7 in dims]
    assert {(12, 4), (4, 24)} <= {shape for shape, _ in dots}  # dA and dB
