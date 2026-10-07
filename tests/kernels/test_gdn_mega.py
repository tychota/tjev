"""The GDN megakernels (pallas_tpu_mega1 / mega2) against the XLA path they replace.

The reference is the model's own XLA code: segmented causal conv + SiLU + padding mask
(mega2 only), l2norm(q, k), q · Dk^-½, β = σ(b), g = −e^{A_log}·softplus(a + dt_bias),
repeated key heads, and the chunked delta rule at fp32 (``highest``, triangular solve).
Forward and the gradient of every input (qkv, a, b, A_log, dt_bias, the conv weight), in
Pallas interpret mode on CPU, then the TPU interpreter and the TPU lowering."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tjev.kernels.causal_conv1d import segmented_causal_conv1d
from tjev.kernels.gated_delta_rule import gated_delta_rule
from tjev.kernels.gated_delta_rule import pallas_tpu as gdn_tpu
from tjev.kernels.gated_delta_rule.pallas_tpu_mega import gdn_megakernel
from tjev.kernels.gated_delta_rule.xla import chunk_gated_delta_rule, l2norm

NAMES = ("qkv", "a", "b", "A_log", "dt_bias", "w")


def _inputs(batch, length, hk, h, d, seed=0, cuts=None, taps=4):
    keys = jax.random.split(jax.random.key(seed), 6)
    width = 2 * hk * d + h * d
    qkv = jax.random.normal(keys[0], (batch, length, width))
    a = jax.random.normal(keys[1], (batch, length, h))
    b = jax.random.normal(keys[2], (batch, length, h))
    a_log = jax.random.normal(keys[3], (h,)) * 0.5
    dt_bias = jax.random.normal(keys[4], (h,)) * 0.5
    w = jax.random.normal(keys[5], (taps, width)) * 0.5
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(batch):
        c = cuts if cuts is not None else np.sort(rng.choice(np.arange(1, length), 4, False))
        row = np.searchsorted(np.asarray(c), np.arange(length), side="right") + 1
        row[length - 5 :] = 0  # trailing padding (segment 0)
        rows.append(row)
    return qkv, a, b, a_log, dt_bias, w, jnp.asarray(np.stack(rows), jnp.int32)


def _reference(qkv, a, b, a_log, dt_bias, w, seg, *, hk, h, d, conv):
    batch, length, _ = qkv.shape
    if conv:
        x = segmented_causal_conv1d(qkv, w, seg)
        x = jnp.where((seg > 0)[..., None], jax.nn.silu(x), 0.0)
    else:
        x = qkv
    q, k, v = jnp.split(x, [hk * d, 2 * hk * d], axis=-1)
    q = l2norm(q.reshape(batch, length, hk, d)) * d**-0.5
    k = l2norm(k.reshape(batch, length, hk, d))
    v = v.reshape(batch, length, h, d)
    beta = jax.nn.sigmoid(b)
    g = -jnp.exp(a_log) * jax.nn.softplus(a + dt_bias)
    q, k = jnp.repeat(q, h // hk, 2), jnp.repeat(k, h // hk, 2)
    return chunk_gated_delta_rule(q, k, v, g, beta, seg, precision="highest")[0]


def _mega(*, hk, h, d, conv, chunk, **kw):
    def fn(qkv, a, b, a_log, dt_bias, w, seg):
        return gdn_megakernel(
            qkv, a, b, a_log, dt_bias, w if conv else None, seg, key_heads=hk, key_dim=d,
            value_heads=h, value_dim=d, chunk=chunk, **kw,
        )  # fmt: skip

    return fn


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


def _errors(fn, ref, xs, conv):
    got, want = jax.jit(fn)(*xs), jax.jit(ref)(*xs)
    cot = jax.random.normal(jax.random.key(9), want.shape)
    argnums = (0, 1, 2, 3, 4, 5) if conv else (0, 1, 2, 3, 4)
    grads = [jax.jit(jax.grad(lambda *x, f=f: jnp.sum(f(*x) * cot), argnums))(*xs)
             for f in (fn, ref)]  # fmt: skip
    finite = bool(jnp.all(jnp.isfinite(got))) and all(
        bool(jnp.all(jnp.isfinite(g))) for g in grads[0]
    )
    errors = {"out": _rel(got, want)}
    errors |= {n: _rel(p, q) for n, p, q in zip(NAMES, *grads, strict=False)}
    return errors, finite


# the per-token terms at the reference's fp32 level; A_log, dt_bias and the conv weight are
# sums over every token (their fp32 reduction error is larger, as on the XLA path)
TOL = {"out": 1e-5, "qkv": 1e-5, "a": 1e-5, "b": 1e-5, "A_log": 5e-5, "dt_bias": 5e-5, "w": 1e-5}


@pytest.mark.parametrize(
    ("shape", "chunk", "cuts", "kw"),
    [
        # segment starts at chunk offsets 0, 1, 2, 3 and C−1 (the conv halo paths)
        ((1, 120, 2, 4), 16, [16, 33, 50, 67, 95], {}),
        ((1, 120, 2, 2), 16, [16, 33, 50, 67, 95], {}),  # GQA ratio 1
        # T = 1100: not a multiple of 512 nor of C, segments across 512-row offsets
        ((1, 1100, 2, 4), 64, [510, 513, 1024, 1027], {}),
        ((2, 200, 2, 4), 64, None, {"inverse": "blocked"}),
        ((1, 256, 2, 4), 128, None, {"inverse": "blocked"}),
        ((1, 200, 4, 8), 64, None, {"heads_per_step": 4}),  # two head groups per row
        # the v6e1 tune picks (chunk 128, blocked): 16 heads per step over 32 value heads
        # (two groups, 4B's GQA 16 / 32), 32 heads per step (one group), 16 over 16 (2B)
        ((1, 256, 16, 32), 128, [100, 177], {"heads_per_step": 16, "inverse": "blocked"}),
        ((1, 256, 16, 32), 128, [100, 177], {"heads_per_step": 32, "inverse": "blocked"}),
        ((1, 256, 8, 16), 128, [100, 177], {"heads_per_step": 16, "inverse": "blocked"}),
        ((2, 200, 2, 4), 64, None, {"precision": "high"}),
    ],
)
@pytest.mark.parametrize("conv", [True, False], ids=["mega2", "mega1"])
def test_mega_matches_reference(shape, chunk, cuts, kw, conv):
    batch, length, hk, h = shape
    xs = _inputs(batch, length, hk, h, 16, cuts=cuts)
    fn = _mega(hk=hk, h=h, d=16, conv=conv, chunk=chunk, **({"precision": "highest"} | kw))
    ref = jax.tree_util.Partial(_reference, hk=hk, h=h, d=16, conv=conv)
    errors, finite = _errors(fn, ref, xs, conv)
    assert finite
    for name, err in errors.items():
        assert err < TOL[name], (name, errors)


def test_mega_mixed_precision():
    """precision="mixed" (one bf16 pass on the state path): bf16-level errors."""
    xs = _inputs(2, 256, 2, 4, 16)
    fn = _mega(hk=2, h=4, d=16, conv=True, chunk=64, precision="mixed")
    ref = jax.tree_util.Partial(_reference, hk=2, h=4, d=16, conv=True)
    errors, finite = _errors(fn, ref, xs, True)
    assert finite
    assert max(errors.values()) < 1e-2, errors


def test_mega_packed_rows_equal_segments_alone():
    """A packed row computes what each segment computes alone (conv taps and the state
    never cross a boundary), boundaries at chunk offsets 0 and 1 and mid-chunk."""
    qkv, a, b, a_log, dt_bias, w, _ = _inputs(1, 96, 2, 4, 16)
    cuts = [0, 16, 33, 71, 96]
    spans = list(itertools.pairwise(cuts))
    seg = np.concatenate([np.full(e - s, i + 1) for i, (s, e) in enumerate(spans)])
    fn = jax.jit(_mega(hk=2, h=4, d=16, conv=True, chunk=16, precision="highest"))
    packed = fn(qkv, a, b, a_log, dt_bias, w, jnp.asarray(seg[None], jnp.int32))
    for s, e in spans:
        alone = fn(qkv[:, s:e], a[:, s:e], b[:, s:e], a_log, dt_bias, w,
                   jnp.ones((1, e - s), jnp.int32))  # fmt: skip
        assert _rel(packed[:, s:e], alone) < 1e-5, (s, e)


def test_mega_conv_weight_gradient_only_when_differentiated():
    """Training never differentiates the frozen conv weight: its cotangent is not computed
    (no dz output), but it is exact when asked for (ground truth)."""
    xs = _inputs(1, 64, 2, 4, 16)
    fn = _mega(hk=2, h=4, d=16, conv=True, chunk=16, precision="highest")
    loss = lambda *x: jnp.sum(fn(*x) ** 2)  # noqa: E731
    jaxpr = str(jax.make_jaxpr(jax.grad(loss, argnums=0))(*xs))
    assert "gdn_mega2_bwd" in jaxpr and "gdn_mega2_bwd_dw" not in jaxpr
    assert "gdn_mega2_bwd_dw" in str(jax.make_jaxpr(jax.grad(loss, argnums=(0, 5)))(*xs))


@pytest.mark.parametrize("conv", [True, False], ids=["mega2", "mega1"])
def test_mega_tpu_interpreter_no_uninitialised_reads_or_races(monkeypatch, conv):
    """The TPU interpreter: uninitialised VMEM / scratch reads are NaN (they would reach the
    outputs or gradients), and races between grid steps are detected. The halos, the dz
    carries and the gate scratch tile are all read."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as tpu_interpret

    monkeypatch.setenv("TJEV_PALLAS_TPU_INTERPRET", "1")
    jax.clear_caches()
    xs = _inputs(1, 100, 2, 4, 16, cuts=[16, 33, 50, 67])
    fn = _mega(hk=2, h=4, d=16, conv=conv, chunk=16, precision="high", inverse="blocked")
    ref = jax.tree_util.Partial(_reference, hk=2, h=4, d=16, conv=conv)
    errors, finite = _errors(fn, ref, xs, conv)
    races = tpu_interpret.races
    assert races is None or not races.races_found
    assert finite
    assert max(errors.values()) < 1e-4, errors
    jax.clear_caches()


def _lower_for_tpu(fn, *shapes):
    from jax import export

    return export.export(jax.jit(fn), platforms=["tpu"])(*shapes).mlir_module()


@pytest.mark.parametrize(("hk", "h"), [(16, 16), (16, 32)], ids=["0.8B", "9B"])
@pytest.mark.parametrize(
    ("precision", "inverse", "chunk", "hb"),
    [("high", "doubling", 64, 0), ("mixed", "doubling", 64, 0), ("bf16", "doubling", 64, 0),
     ("high", "blocked", 64, 0), ("mixed", "blocked", 128, 0), ("high", "doubling", 64, 16)],
)  # fmt: skip
@pytest.mark.parametrize("conv", [True, False], ids=["mega2", "mega1"])
def test_mega_lowers_for_tpu(monkeypatch, hk, h, precision, inverse, chunk, hb, conv):
    """The model's layouts (Dk = Dv = 128, bf16 qkv), forward and backward kernels."""
    monkeypatch.setattr(gdn_tpu, "_interpret", lambda: False)
    width = 2 * hk * 128 + h * 128
    qkv = jax.ShapeDtypeStruct((2, 256, width), jnp.bfloat16)
    gate = jax.ShapeDtypeStruct((2, 256, h), jnp.bfloat16)
    head = jax.ShapeDtypeStruct((h,), jnp.float32)
    w = jax.ShapeDtypeStruct((4, width), jnp.bfloat16)
    seg = jax.ShapeDtypeStruct((2, 256), jnp.int32)

    def loss(qkv, a, b, a_log, dt_bias, w, seg):
        out = gdn_megakernel(
            qkv, a, b, a_log, dt_bias, w if conv else None, seg, key_heads=hk, key_dim=128,
            value_heads=h, value_dim=128, chunk=chunk, precision=precision, inverse=inverse,
            heads_per_step=hb,
        )  # fmt: skip
        return jnp.sum(out.astype(jnp.float32))

    grads = jax.grad(loss, argnums=(0, 1, 2))
    mlir = _lower_for_tpu(grads, qkv, gate, gate, head, head, w, seg)
    assert mlir.count("tpu_custom_call") == 2  # the forward and backward kernels


def test_mega_rejects_shapes_mosaic_cannot_tile(monkeypatch):
    x = jax.ShapeDtypeStruct((1, 128, 2 * 2 * 16 + 4 * 16), jnp.float32)
    gate = jax.ShapeDtypeStruct((1, 128, 4), jnp.float32)
    head = jax.ShapeDtypeStruct((4,), jnp.float32)
    seg = jax.ShapeDtypeStruct((1, 128), jnp.int32)
    fn = _mega(hk=2, h=4, d=16, conv=False, chunk=64)
    jax.eval_shape(fn, x, gate, gate, head, head, None, seg)  # interpret: any head dim
    monkeypatch.setattr(gdn_tpu, "_interpret", lambda: False)
    fn = _mega(hk=2, h=4, d=16, conv=False, chunk=64)  # a fresh trace
    with pytest.raises(ValueError, match="multiple"):
        jax.eval_shape(fn, x, gate, gate, head, head, None, seg)


def _v2_path(*, hk, h, d, conv, chunk, precision):
    """The model's current path (gdn.GatedDeltaNet.core with pallas_tpu_v2), its casts
    included: the XLA conv + SiLU rounded to the projection dtype, l2norm(q, k) and the gates
    in fp32, v in the projection dtype, the v2 rule, the output in the projection dtype."""

    def fn(qkv, a, b, a_log, dt_bias, w, seg):
        batch, length, _ = qkv.shape
        x = qkv
        if conv:
            x = segmented_causal_conv1d(qkv, w, seg)
            x = jnp.where((seg > 0)[..., None], jax.nn.silu(x), 0).astype(qkv.dtype)
        q, k, v = jnp.split(x, [hk * d, 2 * hk * d], axis=-1)
        q = l2norm(q.reshape(batch, length, hk, d).astype(jnp.float32)) * d**-0.5
        k = l2norm(k.reshape(batch, length, hk, d).astype(jnp.float32))
        v = v.reshape(batch, length, h, d)
        beta = jax.nn.sigmoid(b.astype(jnp.float32))
        g = -jnp.exp(a_log) * jax.nn.softplus(a.astype(jnp.float32) + dt_bias)
        out = gated_delta_rule(q, k, v, g, beta, seg, impl="pallas_tpu_v2", chunk=chunk,
                               precision=precision)  # fmt: skip
        return out.astype(qkv.dtype)

    return fn


TPU_CASES = [(16, 16, 64), (16, 32, 64), (16, 16, 128)]


def _tpu_inputs(hk, h):
    if jax.default_backend() != "tpu":
        raise AssertionError("run the tpu tests on a TPU VM: JAX_PLATFORMS=tpu pytest -m tpu")
    jax.config.update("jax_default_matmul_precision", "highest")
    return _inputs(1, 1100, hk, h, 128, seed=3, cuts=[64, 129, 510, 513, 1027])


@pytest.mark.tpu
@pytest.mark.parametrize(("hk", "h", "chunk"), TPU_CASES)
@pytest.mark.parametrize("conv", [True, False], ids=["mega2", "mega1"])
def test_tpu_mega_fp32_matches_reference(hk, h, chunk, conv):
    """On a TPU VM, fp32 (``highest``: 6-pass dots): the compiled kernels (Mosaic's roll
    direction, layouts) against the fp32 XLA reference, at the model's head dims and packed
    segments, every output and gradient (the frozen A_log / dt_bias too) at fp32 level."""
    xs = _tpu_inputs(hk, h)
    fn = _mega(hk=hk, h=h, d=128, conv=conv, chunk=chunk, precision="highest")
    ref = jax.tree_util.Partial(_reference, hk=hk, h=h, d=128, conv=conv)
    errors, finite = _errors(fn, ref, xs, conv)
    print(f"mega fp32 conv={conv} {hk}/{h} C{chunk}:", {k: f"{v:.1e}" for k, v in errors.items()})
    assert finite
    bad = {k: v for k, v in errors.items() if v > 10 * TOL[k]}
    assert not bad, errors


def _against_fp32(fn, xs_fn, ref, xs_ref, conv):
    """fn on xs_fn (e.g. bf16 projections) against ref on the same values in fp32: per
    output and input gradient, (max-abs relative error, relative L2 error)."""
    want = jax.jit(ref)(*xs_ref)
    cot = jax.random.normal(jax.random.key(9), want.shape)
    got = jax.jit(fn)(*xs_fn)
    argnums = (0, 1, 2, 3, 4, 5) if conv else (0, 1, 2, 3, 4)
    g_fn = jax.jit(jax.grad(lambda *x: jnp.sum(fn(*x).astype(jnp.float32) * cot), argnums))(*xs_fn)
    g_ref = jax.jit(jax.grad(lambda *x: jnp.sum(ref(*x) * cot), argnums))(*xs_ref)
    pairs = {"out": (got, want)} | dict(zip(NAMES, zip(g_fn, g_ref, strict=True), strict=False))
    out, finite = {}, True
    for name, (p, q) in pairs.items():
        p, q = p.astype(jnp.float32), q.astype(jnp.float32)
        finite &= bool(jnp.all(jnp.isfinite(p)))
        l2 = float(jnp.linalg.norm(p - q) / (jnp.linalg.norm(q) + 1e-30))
        out[name] = (_rel(p, q), l2)
    return out, finite


@pytest.mark.tpu
@pytest.mark.parametrize(("hk", "h", "chunk"), TPU_CASES)
@pytest.mark.parametrize("conv", [True, False], ids=["mega2", "mega1"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_tpu_mega_high_no_worse_than_v2(hk, h, chunk, conv, dtype):
    """On a TPU VM, the training precision (``high``: bf16x3 dots), fp32 or bf16 projections
    (the model's dtype): the megakernel's error against the fp32 reference (on the same
    projection values) is no larger than that of the path it replaces (the model's v2 path,
    its casts included) on the same inputs, output by output and gradient by gradient."""
    xs32 = _tpu_inputs(hk, h)
    lo = jnp.dtype(dtype)
    xs = tuple(x.astype(lo) if i < 3 else x for i, x in enumerate(xs32))  # qkv, a, b
    xs_ref = tuple(x.astype(jnp.float32) if i < 3 else x for i, x in enumerate(xs))
    ref = jax.tree_util.Partial(_reference, hk=hk, h=h, d=128, conv=conv)
    kw = {"hk": hk, "h": h, "d": 128, "conv": conv, "chunk": chunk, "precision": "high"}
    mega, ok_mega = _against_fp32(_mega(**kw), xs, ref, xs_ref, conv)
    v2, ok_v2 = _against_fp32(_v2_path(**kw), xs, ref, xs_ref, conv)
    fmt = lambda e: {k: f"{m:.1e}/{r:.1e}" for k, (m, r) in e.items()}  # noqa: E731
    print(f"high {dtype} conv={conv} {hk}/{h} C{chunk} (max/l2): mega {fmt(mega)} v2 {fmt(v2)}")
    assert ok_mega and ok_v2
    floor = 1e-6 if dtype == "float32" else 1e-4
    worse = {k: (mega[k], v2[k]) for k in mega
             if mega[k][1] > 1.25 * v2[k][1] + floor or mega[k][0] > 2 * v2[k][0] + floor}  # fmt: skip
    assert not worse, worse
