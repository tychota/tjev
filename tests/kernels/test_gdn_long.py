"""The GDN kernels at the long buckets' lengths (T = 8192, 16384): packed segments across
many chunk boundaries, the gates at the trained 0.8B's spread (long-memory heads included).

Pallas interpret mode on CPU at tiny head dims but the real T. The CPU ignores a dot's
``precision``, so ``high`` (bf16_3x on the MXU) is emulated: :func:`emulated_high` splits
each fp32 operand into bf16 hi + lo parts and sums hi·hi + hi·lo + lo·hi with fp32
accumulation, as the TPU does; ``bf16`` (one pass) casts its operands in the kernels
themselves. Forward and every input gradient against the fp32 XLA reference
(``test_gdn_mega._reference``), fp32 and bf16 projections; the gate gradients d(a),
d(A_log), d(dt_bias) are the MaxText concern (one-pass bf16 gate gradients collapsing at
8k tokens). :func:`fp64_truth` is a float64 token-by-token reference, for telling the
kernels' error from the fp32 reference's own (docs/results.md, long-context numerics).
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_gdn_mega import NAMES, _inputs, _mega, _reference, _v2_path

from tjev.kernels.gated_delta_rule import pallas_tpu, pallas_tpu_v2, xla

HK, H, D = 4, 8, 32
# per value head, spanning the trained Qwen3.5-0.8B's A_log / dt_bias (min, 10%, 50%, 90%,
# max over every layer): per-token decays g from ~-5 to ~-1e-5 (memories of ~10^5 tokens)
A_LOG = (-6.5, -4.0, -2.2, -1.3, -1.3, -0.5, 0.3, 2.3)
DT_BIAS = (-13.0, -8.0, -5.2, -1.7, 0.0, 1.9, -3.0, -10.0)
CHUNKS = {"mega2": 128, "v2": 64}  # the campaign's mega2 (blocked inverse) and v2's default


def long_inputs(length: int, seed: int = 0):
    """``test_gdn_mega._inputs`` at ``length`` with ~length/1024 + 3 packed segments (cuts at
    random offsets, so boundaries fall inside chunks) and the gates above."""
    rng = np.random.default_rng(seed + length)
    cuts = np.sort(rng.choice(np.arange(1, length - 8), length // 1024 + 3, replace=False))
    qkv, a, b, _, _, w, seg = _inputs(1, length, HK, H, D, seed=seed, cuts=cuts)
    return qkv, a, b, jnp.asarray(A_LOG), jnp.asarray(DT_BIAS), w, seg


def reference_fn():
    return jax.tree_util.Partial(_reference, hk=HK, h=H, d=D, conv=True)


def path_fn(path: str, precision: str):
    kw = {"hk": HK, "h": H, "d": D, "conv": True, "chunk": CHUNKS[path], "precision": precision}
    if path == "mega2":
        return _mega(inverse="blocked", **kw)
    return _v2_path(**kw)


def _bf16x3(a, b, dims):
    """fp32 dot as the MXU's bf16_3x: hi·hi + hi·lo + lo·hi, fp32 accumulation."""
    f32, bf = jnp.float32, jnp.bfloat16
    a, b = a.astype(f32), b.astype(f32)
    ah, bh = a.astype(bf), b.astype(bf)
    al, bl = (a - ah.astype(f32)).astype(bf), (b - bh.astype(f32)).astype(bf)

    def dot(x, y):
        return jax.lax.dot_general(x, y, dims, preferred_element_type=f32)

    return dot(ah, bh) + (dot(ah, bl) + dot(al, bh))


@contextlib.contextmanager
def emulated_high():
    """``high`` as bf16_3x in the v2 / mega kernels and their XLA terms (CPU dots are fp32
    whatever the precision)."""
    v2_dot, tpu_dot, mm = pallas_tpu_v2._dot, pallas_tpu._dot, xla.mm

    def v2_high(a, b, mode, contract=pallas_tpu_v2._NN):
        if mode == "high":
            return _bf16x3(a, b, (contract, ((0,), (0,))))
        return v2_dot(a, b, mode, contract)

    def tpu_high(a, b, precision, contract=((1,), (0,))):
        if precision == "high":
            return _bf16x3(a, b, (contract, ((), ())))
        return tpu_dot(a, b, precision, contract)

    def mm_high(a, b, mode="highest"):
        if mode == "high":
            batch = tuple(range(a.ndim - 2))
            return _bf16x3(a, b, (((a.ndim - 1,), (b.ndim - 2,)), (batch, batch)))
        return mm(a, b, mode)

    pallas_tpu_v2._dot, pallas_tpu._dot, xla.mm = v2_high, tpu_high, mm_high
    jax.clear_caches()
    try:
        yield
    finally:
        pallas_tpu_v2._dot, pallas_tpu._dot, xla.mm = v2_dot, tpu_dot, mm
        jax.clear_caches()


def _cotangent(shape):
    return jax.random.normal(jax.random.key(9), shape, jnp.float32)


def outputs(fn, xs):
    """(out, gradients of qkv, a, b, A_log, dt_bias, w) under a fixed random cotangent, fp32."""
    out = jax.jit(fn)(*xs)
    cot = _cotangent(out.shape)
    loss = lambda *x: jnp.sum(fn(*x).astype(jnp.float32) * cot)  # noqa: E731
    grads = jax.jit(jax.grad(loss, argnums=tuple(range(6))))(*xs)
    return [np.asarray(x, np.float64) for x in (out, *grads)]


def errors_vs(fn, xs, want) -> dict:
    """Per output / gradient: max-abs relative error, relative L2 error and cosine."""
    got = outputs(fn, xs) if callable(fn) else fn
    report = {}
    for name, p, q in zip(("out", *NAMES), got, want, strict=True):
        p, q = p.ravel(), q.ravel()
        report[name] = {
            "max": float(np.max(np.abs(p - q)) / (np.max(np.abs(q)) + 1e-30)),
            "l2": float(np.linalg.norm(p - q) / (np.linalg.norm(q) + 1e-30)),
            "cos": float(p @ q / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-30)),
            "finite": bool(np.all(np.isfinite(p))),
        }
    return report


def fp64_truth(xs):
    """The same math in float64, token by token (conv, SiLU, l2norm, gates, delta rule)."""
    with jax.enable_x64(True):
        f64 = jnp.float64
        qkv, a, b, a_log, dt_bias, w, seg = (
            x.astype(f64) if i < 6 else x for i, x in enumerate(xs)
        )

        def fn(qkv, a, b, a_log, dt_bias, w):
            length, taps = qkv.shape[1], w.shape[0]
            x = jnp.zeros_like(qkv)
            for lag in range(taps):
                same = jnp.pad(seg, ((0, 0), (lag, 0)), constant_values=-1)[:, :length] == seg
                shifted = jnp.pad(qkv, ((0, 0), (lag, 0), (0, 0)))[:, :length]
                x = x + jnp.where(same[..., None], shifted, 0.0) * w[taps - 1 - lag]
            x = jnp.where((seg > 0)[..., None], jax.nn.silu(x), 0.0)
            q, k, v = jnp.split(x, [HK * D, 2 * HK * D], axis=-1)

            def unit(y):
                return y / jnp.sqrt(jnp.sum(y * y, -1, keepdims=True) + 1e-6)

            q = unit(q.reshape(1, length, HK, D)) * D**-0.5
            k = unit(k.reshape(1, length, HK, D))
            q, k = jnp.repeat(q, H // HK, 2), jnp.repeat(k, H // HK, 2)
            v = v.reshape(1, length, H, D)
            beta = jax.nn.sigmoid(b)
            g = -jnp.exp(a_log) * jax.nn.softplus(a + dt_bias)
            previous = jnp.concatenate([seg[:, :1], seg[:, :-1]], axis=1)
            restart = (seg != previous).at[:, 0].set(True)

            def step(s, t):
                q_t, k_t, v_t, g_t, b_t, r_t = t
                s = jnp.where(r_t[:, None, None, None], 0.0, s) * jnp.exp(g_t)[..., None, None]
                delta = (v_t - jnp.einsum("bhkv,bhk->bhv", s, k_t)) * b_t[..., None]
                s = s + k_t[..., :, None] * delta[..., None, :]
                return s, jnp.einsum("bhkv,bhk->bhv", s, q_t)

            ts = tuple(jnp.moveaxis(y, 1, 0) for y in (q, k, v, g, beta, restart))
            _, out = jax.lax.scan(step, jnp.zeros((1, H, D, D), f64), ts)
            return jnp.moveaxis(out, 0, 1)

        out = jax.jit(fn)(qkv, a, b, a_log, dt_bias, w)
        cot = _cotangent(out.shape).astype(f64)
        loss = lambda *x: jnp.sum(fn(*x) * cot)  # noqa: E731
        grads = jax.jit(jax.grad(loss, argnums=tuple(range(6))))(qkv, a, b, a_log, dt_bias, w)
        return [np.asarray(x, np.float64) for x in (out, *grads)]


# relative L2 error against the fp32 reference at precision high (bf16_3x). Measured (tiny
# heads, T = 1k-16k, flat in T): fp32 inputs out / qkv / b / w ~1e-5, d(a) 3-4e-5,
# d(A_log) / d(dt_bias) 1-5e-4 (sums over T); bf16 inputs (the reference on the same
# rounded values) 2e-3 (mega2) and 6e-3-1e-2 (v2: v and its output rounded to bf16)
TOL = {
    "float32": {"out": 5e-5, "qkv": 5e-5, "a": 1.5e-4, "b": 5e-5, "A_log": 2e-3,
                "dt_bias": 2e-3, "w": 5e-5},
    "bfloat16": dict.fromkeys(("out", "qkv", "a", "b", "A_log", "dt_bias", "w"), 2e-2),
}  # fmt: skip


def test_emulated_high_is_bf16x3():
    """The emulation is between one bf16 pass and fp32: ~2^-16 relative, not fp32's 1e-7."""
    ka, kb = jax.random.split(jax.random.key(1))
    a = jax.random.normal(ka, (256, 256), jnp.float32)
    b = jax.random.normal(kb, (256, 256), jnp.float32)
    exact = np.asarray(a, np.float64) @ np.asarray(b, np.float64)
    dims = (((1,), (0,)), ((), ()))
    one = jax.lax.dot_general(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), dims,
                              preferred_element_type=jnp.float32)  # fmt: skip

    def rel(x):
        return float(np.linalg.norm(np.asarray(x, np.float64) - exact) / np.linalg.norm(exact))

    three, fp32 = rel(_bf16x3(a, b, dims)), rel(a @ b)
    assert fp32 < three < rel(one) / 30, (fp32, three, rel(one))


def _cast(xs, dtype):
    lo = jnp.dtype(dtype)
    xs = tuple(x.astype(lo) if i < 3 else x for i, x in enumerate(xs))  # qkv, a, b
    return xs, tuple(x.astype(jnp.float32) if i < 3 else x for i, x in enumerate(xs))


@pytest.mark.slow
@pytest.mark.parametrize("length", [8192, 16384])
@pytest.mark.parametrize("path", ["mega2", "v2"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_gdn_long_high_matches_fp32_reference(length, path, dtype):
    xs, xs_ref = _cast(long_inputs(length), dtype)
    want = outputs(reference_fn(), xs_ref)
    with emulated_high():
        errors = errors_vs(path_fn(path, "high"), xs, want)
    print(f"T={length} {path} {dtype} high:",
          {k: f"{e['l2']:.1e} cos {e['cos']:.6f}" for k, e in errors.items()})  # fmt: skip
    assert all(e["finite"] for e in errors.values())
    bad = {k: e["l2"] for k, e in errors.items() if e["l2"] > TOL[dtype][k]}
    assert not bad, errors


@pytest.mark.slow
@pytest.mark.parametrize("path", ["mega2", "v2"])
def test_gdn_gate_gradients_do_not_degrade_with_length(path):
    """At ``high``, fp32 inputs: the gate gradients' error against the fp64 truth grows at
    most like the fp32 reference's own from T = 1024 to 16384 (the sums over T of d(A_log)
    / d(dt_bias) accumulate fp32 rounding in both), and their cosine stays ≥ 0.9999."""
    rows = {}
    for length in (1024, 16384):
        xs = long_inputs(length)
        truth = fp64_truth(xs)
        ref = errors_vs(reference_fn(), xs, truth)
        with emulated_high():
            got = errors_vs(path_fn(path, "high"), xs, truth)
        rows[length] = (got, ref)
    for name in ("a", "A_log", "dt_bias"):
        (short, short_ref), (long, long_ref) = rows[1024], rows[16384]
        assert long[name]["cos"] > 0.9999, (name, long[name])
        growth = long[name]["l2"] / max(short[name]["l2"], 1e-12)
        ref_growth = long_ref[name]["l2"] / max(short_ref[name]["l2"], 1e-12)
        assert growth < 4 * max(ref_growth, 1.0), (name, growth, ref_growth, rows)
