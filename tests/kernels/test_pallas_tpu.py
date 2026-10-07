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
from tjev.kernels.gated_delta_rule.pallas_tpu_v2 import chunk_gated_delta_rule_v2
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


def _tpu_rule(chunk, precision, fused=True, inverse="doubling"):
    """fused: True (pallas_tpu), False (pallas_tpu_split) or "v2" (pallas_tpu_v2)."""

    def fn(*a):
        if fused == "v2":
            return chunk_gated_delta_rule_v2(*a, chunk=chunk, precision=precision), None
        out = gdn_tpu.chunk_gated_delta_rule_tpu(
            *a, chunk=chunk, precision=precision, fused=fused, inverse=inverse
        )
        return out, None

    return fn


# ------------------------------------------------------------------------------------
# CPU: interpret mode and TPU lowering


@pytest.mark.parametrize("fused", [True, False, "v2"])
def test_gdn_tpu_interpreter_no_uninitialised_reads_or_races(monkeypatch, fused):
    """The TPU interpreter (pltpu.InterpretParams): every uninitialised VMEM / scratch read
    is NaN, which would reach the outputs or gradients, and races between grid steps are
    reported. Forward and every input gradient, segments packed mid-chunk."""
    monkeypatch.setenv("TJEV_PALLAS_TPU_INTERPRET", "1")
    jax.clear_caches()  # the interpreter choice is made at trace time
    inputs = _gdn_inputs(1, 256, 2, seed=5)
    fn = _tpu_rule(64, "high", fused)
    got, _ = jax.jit(fn)(*inputs)
    want, _ = jax.jit(chunk_gated_delta_rule)(*inputs)
    assert bool(jnp.all(jnp.isfinite(got)))
    assert _rel(got, want) < 1e-3
    cot = jax.random.normal(jax.random.key(4), want.shape)
    grads = zip(
        _gdn_grads(fn, inputs, cot), _gdn_grads(chunk_gated_delta_rule, inputs, cot), strict=True
    )
    for name, (a, b) in zip("qkvgb", grads, strict=True):
        assert bool(jnp.all(jnp.isfinite(a))), name
        assert _rel(a, b) < 2e-3, (name, _rel(a, b))
    jax.clear_caches()


@pytest.mark.parametrize(
    ("shape", "chunk"), [((1, 128, 2), 64), ((2, 200, 4), 64), ((1, 256, 8), 128)]
)
@pytest.mark.parametrize("fused", [True, False, "v2"])
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


@pytest.mark.parametrize("inverse", ["solve", "blocked"])
@pytest.mark.parametrize("fused", [True, False])
def test_gdn_inverse_options_match_reference(inverse, fused):
    """The XLA terms' UT solve (split forward and backward; the fused backward's recompute)."""
    inputs = _gdn_inputs(2, 200, 4)
    fn = _tpu_rule(64, "highest", fused, inverse)
    got, _ = fn(*inputs)
    want, _ = chunk_gated_delta_rule(*inputs)
    assert _rel(got, want) < 1e-5
    cot = jax.random.normal(jax.random.key(9), want.shape)
    grads = zip(
        _gdn_grads(fn, inputs, cot), _gdn_grads(chunk_gated_delta_rule, inputs, cot), strict=True
    )
    for name, (a, b) in zip("qkvgb", grads, strict=True):
        assert _rel(a, b) < 1e-5, (name, _rel(a, b))


def _cosine(a, b):
    a, b = np.ravel(np.asarray(a)), np.ravel(np.asarray(b))
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("fused", [True, False, "v2"])
def test_gdn_mixed_precision(fused):
    """precision="mixed": one bf16 pass on the state path (on CPU "high" is exact fp32, so
    this measures the state-path rounding alone): bf16-level errors, and the gate gradient
    reduced per head (what A_log / dt_bias see) keeps its direction."""
    inputs = _gdn_inputs(2, 256, 4)
    fn = _tpu_rule(64, "mixed", fused)
    got, _ = fn(*inputs)
    want, _ = chunk_gated_delta_rule(*inputs)
    assert _rel(got, want) < 3e-2
    cot = jax.random.normal(jax.random.key(9), want.shape)
    grads = _gdn_grads(fn, inputs, cot)
    ref = _gdn_grads(chunk_gated_delta_rule, inputs, cot)
    for name, a, b in zip("qkvgb", grads, ref, strict=True):
        assert _rel(a, b) < 6e-2, (name, _rel(a, b))
    assert _cosine(grads[3].sum((0, 1)), ref[3].sum((0, 1))) > 0.9999


@pytest.mark.parametrize("precision", ["highest", "mixed"])
def test_gdn_v2_gqa_reads_key_heads_through_the_index_map(precision):
    """pallas_tpu_v2 with 2 value heads per key head (4B/9B's ratio) against the reference
    on repeated q/k: forward and gradients (dq, dk summed over each group)."""
    q, k, v, g, beta, seg = _gdn_inputs(2, 200, 4)
    q, k = q[:, :, :2], k[:, :, :2]
    fn = _tpu_rule(64, precision, "v2")

    def ref(q, k, *rest):
        return chunk_gated_delta_rule(jnp.repeat(q, 2, 2), jnp.repeat(k, 2, 2), *rest)

    tol = 1e-5 if precision == "highest" else 3e-2
    got, _ = fn(q, k, v, g, beta, seg)
    want, _ = ref(q, k, v, g, beta, seg)
    assert _rel(got, want) < tol
    cot = jax.random.normal(jax.random.key(9), want.shape)
    grads = zip(_gdn_grads(fn, (q, k, v, g, beta, seg), cot),
                _gdn_grads(ref, (q, k, v, g, beta, seg), cot), strict=True)  # fmt: skip
    for name, (a, b) in zip("qkvgb", grads, strict=True):
        assert _rel(a, b) < 2 * tol, (name, _rel(a, b))


@pytest.mark.parametrize("chunk", [16, 64, 128])
def test_gdn_v2_blocked_inverse_is_the_inverse(chunk):
    """The in-kernel inverse (exact 16×16 diagonal blocks by substitution, then doubling)
    against the fp64 inverse, with correlated keys (L ≈ β everywhere: the case where power
    series overflow), next to the pure doubling."""
    from tjev.kernels.gated_delta_rule.pallas_tpu_v2 import _unit_lower_inverse

    k = jax.random.normal(jax.random.key(0), (3, chunk, 32)) + 2.0
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    lower = jnp.tril(0.9 * k @ jnp.swapaxes(k, -1, -2), -1)
    exact = np.linalg.inv(np.eye(chunk) + np.asarray(lower, np.float64))
    for block in (0, 16):
        got = np.asarray(_unit_lower_inverse(lower, "highest", block))
        assert np.max(np.abs(got - exact)) < 1e-5 * np.max(np.abs(exact)), block


@pytest.mark.parametrize(
    ("shape", "chunk", "kw"),
    [((2, 200, 4), 64, {"inverse": "blocked"}), ((1, 256, 4), 128, {"inverse": "blocked"}),
     ((1, 100, 4), 16, {"inverse": "blocked"}), ((2, 200, 8), 64, {"heads_per_step": 4}),
     ((2, 200, 8), 64, {"heads_per_step": 16, "inverse": "blocked"})],
)  # fmt: skip
def test_gdn_v2_options_match_reference(shape, chunk, kw):
    """compute.gdn_inverse=blocked and gdn_heads_per_step: the same function."""
    inputs = _gdn_inputs(*shape)

    def fn(*a):
        return chunk_gated_delta_rule_v2(*a, chunk=chunk, precision="highest", **kw), None

    got, _ = fn(*inputs)
    want, _ = chunk_gated_delta_rule(*inputs)
    assert _rel(got, want) < 1e-5
    cot = jax.random.normal(jax.random.key(9), want.shape)
    grads = zip(
        _gdn_grads(fn, inputs, cot), _gdn_grads(chunk_gated_delta_rule, inputs, cot), strict=True
    )
    for name, (a, b) in zip("qkvgb", grads, strict=True):
        assert _rel(a, b) < 1e-5, (name, _rel(a, b))


def test_gdn_v2_blocked_inverse_tpu_interpreter(monkeypatch):
    """The blocked inverse under the TPU interpreter (NaN on uninitialised reads, races)."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as tpu_interpret

    monkeypatch.setenv("TJEV_PALLAS_TPU_INTERPRET", "1")
    jax.clear_caches()
    inputs = _gdn_inputs(1, 128, 2, seed=5)

    def fn(*a):
        return chunk_gated_delta_rule_v2(*a, chunk=64, precision="high", inverse="blocked"), None

    got, _ = jax.jit(fn)(*inputs)
    want, _ = jax.jit(chunk_gated_delta_rule)(*inputs)
    assert bool(jnp.all(jnp.isfinite(got)))
    assert _rel(got, want) < 1e-3
    cot = jax.random.normal(jax.random.key(4), want.shape)
    for a, b in zip(_gdn_grads(fn, inputs, cot), _gdn_grads(chunk_gated_delta_rule, inputs, cot),
                    strict=True):  # fmt: skip
        assert bool(jnp.all(jnp.isfinite(a)))
        assert _rel(a, b) < 2e-3
    races = tpu_interpret.races
    assert races is None or not races.races_found
    jax.clear_caches()


def test_gdn_v2_vmem_limit_stays_under_the_scoped_flag(monkeypatch):
    from tjev.kernels.gated_delta_rule import pallas_tpu_v2

    monkeypatch.setattr(gdn_tpu, "_interpret", lambda: False)
    monkeypatch.setattr(
        pallas_tpu_v2.pltpu, "get_tpu_info",
        lambda: type("Info", (), {"vmem_capacity_bytes": 128 << 20})(),
    )  # fmt: skip
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--xla_tpu_scoped_vmem_limit_kib=98304")
    assert pallas_tpu_v2._vmem_limit() == int(0.9 * 96 * 2**20)
    assert pallas_tpu_v2._vmem_limit(0.5) == 64 << 20
    assert pallas_tpu_v2._vmem_limit(0.85) == int(0.9 * 96 * 2**20)  # clamped to the flag's
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "")
    assert pallas_tpu_v2._vmem_limit() is None
    assert pallas_tpu_v2._vmem_limit(0.85) == int(0.85 * (128 << 20))


def test_gdn_v2_head_blocks_hold_whole_groups():
    from tjev.kernels.gated_delta_rule.pallas_tpu_v2 import heads_per_block

    assert [heads_per_block(h, r) for h, r in ((16, 1), (32, 2), (4, 2), (3, 1), (48, 3))] == [
        8, 8, 4, 3, 6,
    ]  # fmt: skip


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


@pytest.mark.parametrize(
    ("block", "compute", "fused_bwd", "length"),
    [(512, 0, True, 500), (256, 128, False, 500), (1024, 512, True, 500),
     (1024, 0, True, 2100)],  # the v6e1 tune pick: 1024 tiles, fused backward, 3 tiles
)  # fmt: skip
def test_splash_block_options_match_reference(block, compute, fused_bwd, length):
    """Tile sizes and the fused dkv+dq backward (T = 500 padded to 512: a 1024 tile falls
    back to the whole row)."""
    q, k, v, seg = _attn_inputs(1, length, 4, 2, 256)
    run = lambda *a: attention_tpu.splash_attention(  # noqa: E731
        *a, block=block, compute=compute, fused_bwd=fused_bwd
    )
    valid = (seg > 0)[..., None, None]
    got, want = run(q, k, v, seg), _attn_reference(q, k, v, seg)
    assert float(jnp.max(jnp.abs(jnp.where(valid, got - want, 0.0)))) < 1e-4
    cot = jax.random.normal(jax.random.key(3), got.shape) * valid

    def grads(fn):
        return jax.grad(lambda *a: jnp.sum(fn(*a, seg) * cot), (0, 1, 2))(q, k, v)

    for a, b in zip(grads(run), grads(_attn_reference), strict=True):
        assert _rel(a, b) < 1e-4


def test_splash_block_sizes():
    sizes = attention_tpu.block_sizes(4096, 1024, 512, fused_bwd=True)
    assert (sizes.block_q, sizes.block_kv_compute, sizes.block_q_dq) == (1024, 512, None)
    assert attention_tpu.block_sizes(1536, 1024).block_q == 512  # 1024 does not divide
    with pytest.raises(ValueError, match="divide"):
        attention_tpu.block_sizes(4096, 512, 384)


def _lower_for_tpu(fn, *shapes):
    from jax import export

    return export.export(jax.jit(fn), platforms=["tpu"])(*shapes).mlir_module()


@pytest.mark.parametrize("fused", [True, False, "v2"])
@pytest.mark.parametrize(
    ("precision", "inverse"),
    [("high", "doubling"), ("bf16", "doubling"), ("highest", "doubling"), ("mixed", "doubling"),
     ("high", "solve"), ("high", "blocked")],
)  # fmt: skip
def test_gdn_lowers_for_tpu(monkeypatch, precision, inverse, fused):
    monkeypatch.setattr(gdn_tpu, "_interpret", lambda: False)  # the real Mosaic path
    x = jax.ShapeDtypeStruct((2, 256, 16, 128), jnp.float32)
    gate = jax.ShapeDtypeStruct((2, 256, 16), jnp.float32)
    seg = jax.ShapeDtypeStruct((2, 256), jnp.int32)

    def loss(q, k, v, g, b, s):
        return jnp.sum(_tpu_rule(64, precision, fused, inverse)(q, k, v, g, b, s)[0])

    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))
    mlir = _lower_for_tpu(grads, x, x, x, gate, gate, seg)
    assert mlir.count("tpu_custom_call") == 2  # the forward and backward kernels


@pytest.mark.parametrize(
    ("chunk", "kw"),
    [(64, {"inverse": "blocked"}), (128, {"inverse": "blocked"}), (128, {}),
     (64, {"heads_per_step": 16}), (64, {"heads_per_step": 32, "inverse": "blocked"})],
)  # fmt: skip
def test_gdn_v2_options_lower_for_tpu(monkeypatch, chunk, kw):
    """The blocked inverse, chunk 128 and 16 / 32 heads per step at 9B's layout."""
    monkeypatch.setattr(gdn_tpu, "_interpret", lambda: False)
    qk = jax.ShapeDtypeStruct((2, 256, 16, 128), jnp.float32)
    v = jax.ShapeDtypeStruct((2, 256, 32, 128), jnp.bfloat16)
    gate = jax.ShapeDtypeStruct((2, 256, 32), jnp.float32)
    seg = jax.ShapeDtypeStruct((2, 256), jnp.int32)

    def loss(q, k, v, g, b, s):
        out = chunk_gated_delta_rule_v2(q, k, v, g, b, s, chunk=chunk, precision="mixed", **kw)
        return jnp.sum(out.astype(jnp.float32))

    mlir = _lower_for_tpu(jax.grad(loss, argnums=(0, 1, 2, 3, 4)), qk, qk, v, gate, gate, seg)
    assert mlir.count("tpu_custom_call") == 2


@pytest.mark.parametrize("precision", ["high", "mixed", "bf16"])
def test_gdn_v2_lowers_for_tpu_with_gqa_and_bf16_values(monkeypatch, precision):
    """The model's layout: 16 key heads for 32 value heads (9B), v in bf16."""
    monkeypatch.setattr(gdn_tpu, "_interpret", lambda: False)
    qk = jax.ShapeDtypeStruct((2, 256, 16, 128), jnp.float32)
    v = jax.ShapeDtypeStruct((2, 256, 32, 128), jnp.bfloat16)
    gate = jax.ShapeDtypeStruct((2, 256, 32), jnp.float32)
    seg = jax.ShapeDtypeStruct((2, 256), jnp.int32)

    def loss(q, k, v, g, b, s):
        out = chunk_gated_delta_rule_v2(q, k, v, g, b, s, precision=precision)
        return jnp.sum(out.astype(jnp.float32))

    mlir = _lower_for_tpu(jax.grad(loss, argnums=(0, 1, 2, 3, 4)), qk, qk, v, gate, gate, seg)
    assert mlir.count("tpu_custom_call") == 2


@pytest.mark.parametrize(
    ("dim", "chunk", "fused"),
    [(16, 64, True), (64, 64, True), (128, 4, False), (16, 64, "v2"), (128, 4, "v2")],
)
def test_gdn_rejects_shapes_mosaic_cannot_tile(monkeypatch, dim, chunk, fused):
    """On TPU: a clear error for head dims the fused kernel's lane slices cannot take (the
    tiny test model's 16) and chunks below a sublane tile; interpret mode runs them."""
    x = jax.ShapeDtypeStruct((1, 128, 4, dim), jnp.float32)
    gate = jax.ShapeDtypeStruct((1, 128, 4), jnp.float32)
    seg = jax.ShapeDtypeStruct((1, 128), jnp.int32)
    jax.eval_shape(_tpu_rule(chunk, "high", fused), x, x, x, gate, gate, seg)  # interpret: any
    monkeypatch.setattr(gdn_tpu, "_interpret", lambda: False)
    with pytest.raises(ValueError, match="multiple"):
        jax.eval_shape(_tpu_rule(chunk, "high", fused), x, x, x, gate, gate, seg)


@pytest.mark.parametrize(
    ("block", "compute", "fused_bwd", "kernels"),
    [(512, 0, False, 3), (512, 0, True, 2), (1024, 512, True, 2), (512, 256, False, 3)],
)
def test_splash_lowers_for_tpu(monkeypatch, block, compute, fused_bwd, kernels):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")  # the real Mosaic path
    attention_tpu._kernel.cache_clear()
    try:
        q = jax.ShapeDtypeStruct((2, 2048, 8, 256), jnp.bfloat16)
        kv = jax.ShapeDtypeStruct((2, 2048, 2, 256), jnp.bfloat16)
        seg = jax.ShapeDtypeStruct((2, 2048), jnp.int32)

        def loss(q, k, v, s):
            out = attention_tpu.splash_attention(
                q, k, v, s, block=block, compute=compute, fused_bwd=fused_bwd
            )
            return jnp.sum(out.astype(jnp.float32))

        mlir = _lower_for_tpu(jax.grad(loss, argnums=(0, 1, 2)), q, kv, kv, seg)
        assert mlir.count("tpu_custom_call") == kernels  # forward, dq, dkv (fused: dkv + dq)
    finally:
        attention_tpu._kernel.cache_clear()


# ------------------------------------------------------------------------------------
# TPU: the compiled kernels


@pytest.fixture(scope="module")
def on_tpu():
    if os.environ.get("JAX_PLATFORMS") == "cpu" or jax.default_backend() != "tpu":
        raise AssertionError("run the tpu tests on a TPU VM: JAX_PLATFORMS=tpu pytest -m tpu")


@pytest.mark.tpu
@pytest.mark.parametrize(
    ("precision", "tol"), [("highest", 1e-4), ("high", 1e-3), ("bf16", 3e-2), ("mixed", 3e-2)]
)
@pytest.mark.parametrize(
    ("shape", "chunk"), [((2, 1024, 16), 64), ((1, 4096, 32), 64), ((2, 1024, 16), 128)]
)
@pytest.mark.parametrize("fused", [True, False, "v2"])
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
    """Forward and gradients (the dq / dkv kernels training uses) against fp32 attention.
    Splash runs at the default precision: the suite's global "highest" would make its bf16
    dots fp32-precision contractions, which Mosaic does not compile ("Bad lhs type")."""
    q, k, v, seg = _attn_inputs(1, length, heads, kv_heads, 256, jnp.bfloat16, seed=1)
    cot = jax.random.normal(jax.random.key(7), q.shape, jnp.float32)

    def loss(fn, *qkv):
        return jnp.sum(fn(*qkv, seg).astype(jnp.float32) * cot)

    with jax.default_matmul_precision("default"):
        got = jax.jit(attention_tpu.splash_attention)(q, k, v, seg).astype(jnp.float32)
        got_grads = jax.jit(jax.grad(lambda *x: loss(attention_tpu.splash_attention, *x),
                                     argnums=(0, 1, 2)))(q, k, v)  # fmt: skip
    q32, k32, v32 = (x.astype(jnp.float32) for x in (q, k, v))
    want = jax.jit(_attn_reference)(q32, k32, v32, seg)
    want_grads = jax.jit(jax.grad(lambda *x: loss(_attn_reference, *x), argnums=(0, 1, 2)))(
        q32, k32, v32
    )
    valid = (seg > 0)[..., None, None]
    err = float(jnp.max(jnp.abs(jnp.where(valid, got - want, 0.0))))
    print(f"splash T{length} H{heads}/{kv_heads}: max abs err {err:.2e}")
    assert err < 3e-2
    for name, a, b in zip("qkv", got_grads, want_grads, strict=True):
        a = a.astype(jnp.float32)
        assert bool(jnp.all(jnp.isfinite(a))), name
        print(f"  grad d{name}: rel {_rel(a, b):.2e}")
        assert _rel(a, b) < 3e-2, (name, _rel(a, b))
