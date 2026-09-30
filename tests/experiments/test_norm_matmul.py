"""The experimental fused RMSNorm → projection kernel: equals the reference (forward and
gradients) in interpret mode, and lowers to one Mosaic TPU call."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "norm_matmul", ROOT / "experiments" / "tpu" / "norm_matmul.py"
)
assert _spec is not None and _spec.loader is not None
norm_matmul = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(norm_matmul)


def _inputs(m=64, k=256, n=384, dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(keys[0], (m, k), dtype)
    gamma = 0.1 * jax.random.normal(keys[1], (k,), jnp.float32)
    w = (jax.random.normal(keys[2], (k, n), jnp.float32) / np.sqrt(k)).astype(dtype)
    return x, gamma, w


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_forward_and_gradients_equal_the_reference(dtype):
    x, gamma, w = _inputs(dtype=dtype)
    got = norm_matmul.fused_rmsnorm_matmul(x, gamma, w, 1e-6, 32, 128)
    want = norm_matmul.reference(x, gamma, w)
    tol = 1e-5 if dtype == jnp.float32 else 2e-2
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32),
                               atol=tol, rtol=tol)  # fmt: skip
    cot = jax.random.normal(jax.random.key(1), want.shape, jnp.float32)

    def loss(fn):
        return jax.grad(lambda *a: jnp.sum(fn(*a).astype(jnp.float32) * cot), argnums=(0, 1, 2))

    fused = loss(lambda a, g, b: norm_matmul.fused_rmsnorm_matmul(a, g, b, 1e-6, 32, 128))(
        x, gamma, w
    )
    ref = loss(norm_matmul.reference)(x, gamma, w)
    for a, b in zip(fused, ref, strict=True):
        np.testing.assert_allclose(np.asarray(a, np.float32), np.asarray(b, np.float32),
                                   atol=tol, rtol=tol)  # fmt: skip


def test_lowers_to_one_tpu_kernel(monkeypatch):
    from jax import export

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")  # the real Mosaic path
    x = jax.ShapeDtypeStruct((8192, 2048), jnp.bfloat16)
    gamma = jax.ShapeDtypeStruct((2048,), jnp.float32)
    w = jax.ShapeDtypeStruct((2048, 6144), jnp.bfloat16)
    mlir = export.export(jax.jit(norm_matmul.fused_rmsnorm_matmul),
                         platforms=["tpu"])(x, gamma, w).mlir_module()  # fmt: skip
    assert mlir.count("tpu_custom_call") == 1
