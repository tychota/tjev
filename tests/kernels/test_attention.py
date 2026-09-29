"""XLA attention: query blocking and remat never change the result; the dispatcher."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tjev.kernels import attention


def _inputs(batch=2, length=40, heads=4, kv_heads=2, dim=16):
    keys = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(keys[0], (batch, length, heads, dim))
    k = jax.random.normal(keys[1], (batch, length, kv_heads, dim))
    v = jax.random.normal(keys[2], (batch, length, kv_heads, dim))
    seg = np.ones((batch, length), np.int32)
    seg[0, 15:] = 2
    seg[1, 30:] = 0  # trailing padding
    return q, k, v, jnp.asarray(seg)


def _full(q, k, v, seg):
    idx = jnp.arange(q.shape[1])
    mask = ((seg[:, :, None] == seg[:, None, :]) & (idx[None, :] <= idx[:, None])) | (
        idx[None, :] == idx[:, None]
    )
    return jax.nn.dot_product_attention(q, k, v, mask=mask[:, None])


@pytest.mark.parametrize(("block", "remat"), [(0, True), (8, True), (16, False), (512, True)])
def test_blocked_equals_full(block, remat):
    q, k, v, seg = _inputs()
    want = _full(q, k, v, seg)
    got = attention(q, k, v, seg, impl="xla", block=block, remat=remat)
    np.testing.assert_allclose(got, want, atol=1e-5)
    grad = jax.grad(lambda q: jnp.sum(attention(q, k, v, seg, block=block, remat=remat) ** 2))
    np.testing.assert_allclose(grad(q), jax.grad(lambda q: jnp.sum(_full(q, k, v, seg) ** 2))(q),
                               atol=1e-4)  # fmt: skip


def test_segments_are_isolated():
    q, k, v, seg = _inputs()
    out = attention(q, k, v, seg)
    alone = attention(q[:1, 15:], k[:1, 15:], v[:1, 15:], seg[:1, 15:])
    np.testing.assert_allclose(out[:1, 15:], alone, atol=1e-5)


def test_unknown_impl_fails():
    q, k, v, seg = _inputs()
    with pytest.raises(ValueError, match="impl"):
        attention(q, k, v, seg, impl="cudnn")
