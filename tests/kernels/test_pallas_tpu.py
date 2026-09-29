"""Pallas TPU kernels (GDN recurrence, splash attention) against the pure-JAX references.

On CPU (the default suite) the kernels run in Pallas interpret mode, and a second check
lowers them for TPU (``jax.export``, platforms=["tpu"]): Pallas → Mosaic lowering errors
show up here, before any TPU time is paid for. The ``tpu`` tests compile and run the real
kernels: ``JAX_PLATFORMS=tpu pytest -m tpu`` on a TPU VM (deselected elsewhere)."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tjev.kernels.attention import splash_tpu as attention_tpu
from tjev.kernels.gated_delta_rule import pallas_tpu as gdn_tpu
from tjev.kernels.gated_delta_rule.xla import chunk_gated_delta_rule, l2norm


def _gdn_inputs(batch, length, heads, seed=0, dim=128):
    keys = jax.random.split(jax.random.key(seed), 5)
    q = l2norm(jax.random.normal(keys[0], (batch, length, heads, dim))) * dim**-0.5
    k = l2norm(jax.random.normal(keys[1], (batch, length, heads, dim)))
    v = jax.random.normal(keys[2], (batch, length, heads, dim))
    g = -jax.nn.softplus(jax.random.normal(keys[3], (batch, length, heads)) - 1.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, heads)) + 1.0)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(batch):  # boundaries anywhere, not only on chunk edges
        cuts = np.sort(rng.choice(np.arange(1, length), size=4, replace=False))
        rows.append(np.searchsorted(cuts, np.arange(length), side="right") + 1)
    return q, k, v, g, beta, jnp.asarray(np.stack(rows), jnp.int32)


def _attn_inputs(batch, length, heads, kv_heads, dim, dtype=jnp.float32, seed=0):
    keys = jax.random.split(jax.random.key(seed), 3)
    q, k, v = (
        jax.random.normal(keys[i], (batch, length, h, dim), dtype)
        for i, h in ((0, heads), (1, kv_heads), (2, kv_heads))
    )
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(batch):  # segments, then trailing padding (segment 0)
        cuts = np.sort(rng.choice(np.arange(1, length - 8), size=3, replace=False))
        row = np.searchsorted(cuts, np.arange(length), side="right") + 1
        row[length - rng.integers(1, 8) :] = 0
        rows.append(row)
    return q, k, v, jnp.asarray(np.stack(rows), jnp.int32)


def _attn_reference(q, k, v, seg):
    length = q.shape[1]
    idx = jnp.arange(length)
    same = seg[:, :, None] == seg[:, None, :]
    mask = (same & (idx[None, :] <= idx[:, None])) | (idx[None, :] == idx[:, None])
    return jax.nn.dot_product_attention(q, k, v, mask=mask[:, None])


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


def _gdn_grads(fn, inputs, cot):
    *xs, seg = inputs

    def loss(*a):
        return jnp.sum(fn(*a, seg)[0] * cot)

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(*xs)


def _tpu_rule(chunk, precision, fused=True):
    def fn(*a):
        out = gdn_tpu.chunk_gated_delta_rule_tpu(*a, chunk=chunk, precision=precision, fused=fused)
        return out, None

    return fn


# ------------------------------------------------------------------------------------
# CPU: interpret mode and TPU lowering


@pytest.mark.parametrize(
    ("shape", "chunk"), [((1, 128, 2), 64), ((2, 200, 4), 64), ((1, 256, 8), 128)]
)
@pytest.mark.parametrize("fused", [True, False])
def test_gdn_matches_reference(shape, chunk, fused):
    inputs = _gdn_inputs(*shape)
    fn = _tpu_rule(chunk, "highest", fused)
    got, _ = fn(*inputs)
    want, _ = chunk_gated_delta_rule(*inputs)
    assert _rel(got, want) < 1e-5
    cot = jax.random.normal(jax.random.key(9), want.shape)
    grads = zip(
        _gdn_grads(fn, inputs, cot), _gdn_grads(chunk_gated_delta_rule, inputs, cot), strict=True
    )
    for name, (a, b) in zip("qkvgb", grads, strict=True):
        assert _rel(a, b) < 1e-5, (name, _rel(a, b))


def test_gdn_heads_per_block_divides_the_rows():
    assert [gdn_tpu.heads_per_block(n) for n in (16, 32, 12, 6, 3)] == [8, 8, 4, 2, 1]


def test_splash_matches_reference_gqa_segments_padding():
    q, k, v, seg = _attn_inputs(2, 200, 4, 2, 256)  # T padded to 256 inside
    got = attention_tpu.splash_attention(q, k, v, seg)
    want = _attn_reference(q, k, v, seg)
    valid = (seg > 0)[..., None, None]
    assert float(jnp.max(jnp.abs(jnp.where(valid, got - want, 0.0)))) < 1e-4
    cot = jax.random.normal(jax.random.key(3), got.shape) * valid

    def loss(fn):
        return jax.grad(lambda *a: jnp.sum(fn(*a, seg) * cot), (0, 1, 2))

    got_grads = loss(attention_tpu.splash_attention)(q, k, v)
    for a, b in zip(got_grads, loss(_attn_reference)(q, k, v), strict=True):
        assert _rel(a, b) < 1e-4


def _lower_for_tpu(fn, *shapes):
    from jax import export

    return export.export(jax.jit(fn), platforms=["tpu"])(*shapes).mlir_module()


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("precision", ["high", "bf16", "highest"])
def test_gdn_lowers_for_tpu(monkeypatch, precision, fused):
    monkeypatch.setattr(gdn_tpu, "_interpret", lambda: False)  # the real Mosaic path
    x = jax.ShapeDtypeStruct((2, 256, 16, 128), jnp.float32)
    gate = jax.ShapeDtypeStruct((2, 256, 16), jnp.float32)
    seg = jax.ShapeDtypeStruct((2, 256), jnp.int32)

    def loss(q, k, v, g, b, s):
        return jnp.sum(_tpu_rule(64, precision, fused)(q, k, v, g, b, s)[0])

    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))
    mlir = _lower_for_tpu(grads, x, x, x, gate, gate, seg)
    assert mlir.count("tpu_custom_call") == 2  # the forward and backward kernels


def test_splash_lowers_for_tpu(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")  # the real Mosaic path
    attention_tpu._kernel.cache_clear()
    try:
        q = jax.ShapeDtypeStruct((2, 1024, 8, 256), jnp.bfloat16)
        kv = jax.ShapeDtypeStruct((2, 1024, 2, 256), jnp.bfloat16)
        seg = jax.ShapeDtypeStruct((2, 1024), jnp.int32)

        def loss(q, k, v, s):
            return jnp.sum(attention_tpu.splash_attention(q, k, v, s).astype(jnp.float32))

        mlir = _lower_for_tpu(jax.grad(loss, argnums=(0, 1, 2)), q, kv, kv, seg)
        assert mlir.count("tpu_custom_call") == 3  # forward, dq, dkv
    finally:
        attention_tpu._kernel.cache_clear()


# ------------------------------------------------------------------------------------
# TPU: the compiled kernels


@pytest.fixture(scope="module")
def on_tpu():
    if os.environ.get("JAX_PLATFORMS") == "cpu" or jax.default_backend() != "tpu":
        raise AssertionError("run the tpu tests on a TPU VM: JAX_PLATFORMS=tpu pytest -m tpu")


@pytest.mark.tpu
@pytest.mark.parametrize(("precision", "tol"), [("highest", 1e-4), ("high", 1e-3), ("bf16", 3e-2)])
@pytest.mark.parametrize(
    ("shape", "chunk"), [((2, 1024, 16), 64), ((1, 4096, 32), 64), ((2, 1024, 16), 128)]
)
@pytest.mark.parametrize("fused", [True, False])
def test_tpu_gdn_matches_reference(on_tpu, shape, chunk, precision, tol, fused):
    jax.config.update("jax_default_matmul_precision", "highest")  # the reference in fp32
    inputs = _gdn_inputs(*shape, seed=3)
    fn = _tpu_rule(chunk, precision, fused)
    got, _ = jax.jit(fn)(*inputs)
    want, _ = jax.jit(chunk_gated_delta_rule)(*inputs)
    print(f"gdn {shape} C{chunk} {precision} fused={fused}: fwd rel {_rel(got, want):.2e}")
    assert _rel(got, want) < tol
    cot = jax.random.normal(jax.random.key(9), want.shape)
    grads = zip(
        _gdn_grads(fn, inputs, cot), _gdn_grads(chunk_gated_delta_rule, inputs, cot), strict=True
    )
    for name, (a, b) in zip("qkvgb", grads, strict=True):
        assert bool(jnp.all(jnp.isfinite(a))), name
        print(f"  grad {name}: rel {_rel(a, b):.2e}")
        assert _rel(a, b) < 2 * tol, (name, _rel(a, b))


@pytest.mark.tpu
@pytest.mark.parametrize(("length", "heads", "kv_heads"), [(1024, 8, 2), (4096, 16, 4)])
def test_tpu_splash_matches_reference(on_tpu, length, heads, kv_heads):
    q, k, v, seg = _attn_inputs(1, length, heads, kv_heads, 256, jnp.bfloat16, seed=1)
    got = jax.jit(attention_tpu.splash_attention)(q, k, v, seg).astype(jnp.float32)
    want = jax.jit(_attn_reference)(*(x.astype(jnp.float32) for x in (q, k, v)), seg)
    valid = (seg > 0)[..., None, None]
    err = float(jnp.max(jnp.abs(jnp.where(valid, got - want, 0.0))))
    print(f"splash T{length} H{heads}/{kv_heads}: max abs err {err:.2e}")
    assert err < 3e-2
