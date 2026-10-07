"""What Mosaic (Pallas TPU) computes, against XLA:TPU, for the ops the GDN kernels use.

The megakernels move the gates, l2norm and the bf16 casts of the GDN core from XLA into
the kernel. If Mosaic rounds or approximates differently from XLA, the megakernel is less
exact than the XLA + v2 path with the same math. Each test measures one such op on the
TPU and prints the numbers (run with -s):

  * f32 → bf16 casts: round to nearest even, as XLA (or truncation: a biased error that
    accumulates over layers instead of averaging out);
  * fp32 dots at DEFAULT / HIGH / HIGHEST: the precision is honoured (HIGH ≈ bf16x3);
  * exp, log1p, softplus, sigmoid, rsqrt: the in-kernel accuracy against XLA's.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

pytestmark = pytest.mark.tpu


def _tpu():
    if jax.default_backend() != "tpu":
        raise AssertionError("run the tpu tests on a TPU VM: JAX_PLATFORMS=tpu pytest -m tpu")


def _call(body, out_dtype, *xs):
    def kernel(*refs):
        *ins, out = refs
        out[...] = body(*(r[...] for r in ins)).astype(out.dtype)

    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(xs[0].shape, out_dtype))(*xs)


def test_tpu_mosaic_bf16_cast_rounds_to_nearest_even():
    _tpu()
    x = jax.random.normal(jax.random.key(0), (512, 1024), jnp.float32)
    got = _call(lambda v: v, jnp.bfloat16, x).astype(jnp.float32)
    want = x.astype(jnp.bfloat16).astype(jnp.float32)  # XLA: round to nearest even
    trunc = jax.lax.bitcast_convert_type(
        jax.lax.bitcast_convert_type(x, jnp.uint32) & jnp.uint32(0xFFFF0000), jnp.float32
    )
    differ = float(jnp.mean(got != want))
    as_trunc = float(jnp.mean(got == trunc))
    bias = float(jnp.mean((got - x) / jnp.abs(x)))
    print(f"mosaic f32->bf16: differs from XLA on {differ:.2%}, equals truncation on "
          f"{as_trunc:.2%}, mean relative bias {bias:.2e}")  # fmt: skip
    assert differ == 0.0, f"Mosaic's cast differs from XLA's on {differ:.2%} (bias {bias:.1e})"


@pytest.mark.parametrize("precision", ["DEFAULT", "HIGH", "HIGHEST"])
def test_tpu_mosaic_dot_precision_is_honoured(precision):
    _tpu()
    p = getattr(jax.lax.Precision, precision)
    ka, kb = jax.random.split(jax.random.key(1))
    a = jax.random.normal(ka, (256, 256), jnp.float32)
    b = jax.random.normal(kb, (256, 256), jnp.float32)
    exact = np.asarray(a, np.float64) @ np.asarray(b, np.float64)

    def dot(x, y):
        return jax.lax.dot_general(x, y, (((1,), (0,)), ((), ())), precision=p,
                                   preferred_element_type=jnp.float32)  # fmt: skip

    def err(z):
        return float(np.linalg.norm(np.asarray(z, np.float64) - exact) / np.linalg.norm(exact))

    mosaic, xla = err(_call(dot, jnp.float32, a, b)), err(jax.jit(dot)(a, b))
    print(f"fp32 dot {precision}: mosaic rel L2 {mosaic:.2e}, XLA {xla:.2e}")
    assert mosaic <= 2 * xla + 1e-7, (mosaic, xla)


def _sig(v):
    return 1 / (1 + np.exp(-v))


TRANSCENDENTALS = {  # name: (the jax op, its float64 numpy reference)
    "exp": (jnp.exp, np.exp),
    "log1p": (lambda v: jnp.log1p(jnp.abs(v)), lambda v: np.log1p(np.abs(v))),
    "softplus": (jax.nn.softplus, lambda v: np.logaddexp(0, v)),
    "sigmoid": (jax.nn.sigmoid, _sig),
    "silu": (jax.nn.silu, lambda v: v * _sig(v)),
    "rsqrt": (lambda v: jax.lax.rsqrt(jnp.abs(v) + 1e-3), lambda v: 1 / np.sqrt(np.abs(v) + 1e-3)),
}


@pytest.mark.parametrize("name", list(TRANSCENDENTALS))
def test_tpu_mosaic_transcendentals_match_xla(name):
    _tpu()
    fn, ref = TRANSCENDENTALS[name]
    x = 3 * jax.random.normal(jax.random.key(2), (512, 1024), jnp.float32)
    exact = ref(np.asarray(x, np.float64))

    def err(z):
        z = np.asarray(z, np.float64)
        return float(np.max(np.abs(z - exact) / (np.abs(exact) + 1e-30)))

    mosaic, xla = err(_call(fn, jnp.float32, x)), err(jax.jit(fn)(x))
    print(f"{name}: mosaic max rel {mosaic:.2e}, XLA:TPU {xla:.2e}")
    assert mosaic <= 4 * xla + 1e-6, (mosaic, xla)


def test_tpu_mosaic_dot_precision_passes():
    """HIGH should cost ~3 bf16 passes and HIGHEST ~6: a HIGH that times like DEFAULT is
    mapped down (the Mosaic bf16x3 path is new: jax-ml/jax#39558, jax 0.11.1)."""
    import time

    _tpu()
    ka, kb = jax.random.split(jax.random.key(3))
    a = jax.random.normal(ka, (1024, 1024), jnp.float32)
    b = jax.random.normal(kb, (1024, 1024), jnp.float32)
    times = {}
    for name in ("DEFAULT", "HIGH", "HIGHEST"):
        p = getattr(jax.lax.Precision, name)

        def dot(x, y, p=p):
            return jax.lax.dot_general(x, y, (((1,), (0,)), ((), ())), precision=p,
                                       preferred_element_type=jnp.float32)  # fmt: skip

        f = jax.jit(lambda x, y, dot=dot: _call(dot, jnp.float32, x, y))
        f(a, b).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(20):
            out = f(a, b)
        out.block_until_ready()
        times[name] = (time.perf_counter() - t0) / 20
    ratio = {k: round(v / times["DEFAULT"], 2) for k, v in times.items()}
    print(f"mosaic fp32 dot 1024^3 (one VMEM block) time / DEFAULT: {ratio}")
    assert ratio["HIGH"] > 1.5, ratio
