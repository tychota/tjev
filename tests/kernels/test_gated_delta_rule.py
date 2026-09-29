"""The XLA delta rule: chunked form vs the recurrence, segments, states, gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tjev.kernels.gated_delta_rule import gated_delta_rule
from tjev.kernels.gated_delta_rule.xla import (
    chunk_gated_delta_rule,
    l2norm,
    recurrent_gated_delta_rule,
    unit_lower_inverse,
)


def _inputs(seed, batch=2, length=45, heads=3, dk=8, dv=6):
    keys = jax.random.split(jax.random.key(seed), 5)
    q = l2norm(jax.random.normal(keys[0], (batch, length, heads, dk))) * dk**-0.5
    k = l2norm(jax.random.normal(keys[1], (batch, length, heads, dk)))
    v = jax.random.normal(keys[2], (batch, length, heads, dv))
    g = -jax.nn.softplus(jax.random.normal(keys[3], (batch, length, heads)))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, heads)))
    return q, k, v, g, beta


def _segments(batch, length, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(batch):
        cuts = np.sort(rng.choice(np.arange(1, length), size=4, replace=False))
        row = np.searchsorted(cuts, np.arange(length), side="right") + 1
        rows.append(row)
    return jnp.asarray(np.stack(rows), jnp.int32)


@pytest.mark.parametrize("chunk", [4, 8, 16, 64])
def test_chunked_matches_recurrence(chunk):
    q, k, v, g, beta = _inputs(0)
    ref, ref_state = recurrent_gated_delta_rule(q, k, v, g, beta)
    out, state = chunk_gated_delta_rule(q, k, v, g, beta, chunk=chunk)
    np.testing.assert_allclose(out, ref, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(state, ref_state, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("chunk", [4, 16])
def test_segments_match_independent_runs(chunk):
    q, k, v, g, beta = _inputs(1)
    seg = _segments(q.shape[0], q.shape[1], 1)
    out, _ = chunk_gated_delta_rule(q, k, v, g, beta, seg, chunk=chunk)
    rec, _ = recurrent_gated_delta_rule(q, k, v, g, beta, seg)
    np.testing.assert_allclose(out, rec, atol=2e-5, rtol=2e-5)
    for b in range(q.shape[0]):
        for s in np.unique(np.asarray(seg[b])):
            idx = np.flatnonzero(np.asarray(seg[b]) == s)
            alone, _ = recurrent_gated_delta_rule(*(x[b : b + 1, idx] for x in (q, k, v, g, beta)))
            np.testing.assert_allclose(out[b, idx], alone[0], atol=2e-5, rtol=2e-5)


def test_initial_state_continuation():
    q, k, v, g, beta = _inputs(2, length=40)
    full, full_state = chunk_gated_delta_rule(q, k, v, g, beta, chunk=8)
    first, state = chunk_gated_delta_rule(*(x[:, :24] for x in (q, k, v, g, beta)), chunk=8)
    second, end = chunk_gated_delta_rule(
        *(x[:, 24:] for x in (q, k, v, g, beta)), initial_state=state, chunk=8
    )
    np.testing.assert_allclose(jnp.concatenate([first, second], 1), full, atol=2e-5)
    np.testing.assert_allclose(end, full_state, atol=2e-5)
    step, _ = recurrent_gated_delta_rule(
        *(x[:, 24:25] for x in (q, k, v, g, beta)), initial_state=state
    )
    np.testing.assert_allclose(step[:, 0], full[:, 24], atol=2e-5)


def test_gradients_match():
    q, k, v, g, beta = _inputs(3, length=33)
    seg = _segments(q.shape[0], q.shape[1], 3)
    weights = jax.random.normal(jax.random.key(9), v.shape)

    def loss(fn, *args):
        return jnp.sum(fn(*args, seg)[0] * weights)

    chunked = jax.grad(
        lambda *a: loss(lambda *b: chunk_gated_delta_rule(*b, chunk=8), *a), argnums=(0, 1, 2, 3, 4)
    )(q, k, v, g, beta)
    recurrent = jax.grad(lambda *a: loss(recurrent_gated_delta_rule, *a), argnums=(0, 1, 2, 3, 4))(
        q, k, v, g, beta
    )
    for a, b in zip(chunked, recurrent, strict=True):
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, atol=5e-5, rtol=5e-4)


def test_doubling_inverse_matches_triangular_solve():
    raw = jax.random.normal(jax.random.key(4), (3, 64, 64)) * 0.3
    lower = jnp.tril(raw, -1)
    want = jnp.linalg.inv(lower + jnp.eye(64))
    np.testing.assert_allclose(unit_lower_inverse(lower, "highest"), want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("precision", "inverse", "tol"),
    [
        ("highest", "doubling", 5e-5),
        ("high", "solve", 3e-3),
        ("bf16", "solve", 3e-2),
        ("bf16", "doubling", 3e-2),
    ],
)
def test_fast_variants_match_recurrence(precision, inverse, tol):
    q, k, v, g, beta = _inputs(5, length=130)
    seg = _segments(q.shape[0], q.shape[1], 5)
    ref, ref_state = recurrent_gated_delta_rule(q, k, v, g, beta, seg)
    out, state = chunk_gated_delta_rule(
        q, k, v, g, beta, seg, chunk=16, precision=precision, inverse=inverse
    )
    scale = float(jnp.max(jnp.abs(ref)))
    assert float(jnp.max(jnp.abs(out - ref))) / scale < tol
    assert float(jnp.max(jnp.abs(state - ref_state))) / float(jnp.max(jnp.abs(ref_state))) < tol


def test_matmul_inverse_is_stable_for_correlated_keys():
    """Real keys are correlated: L ≈ β on the whole strict lower triangle. The inverse
    must stay accurate (the power series (I-L)(I+L^2)... overflows here)."""
    base = jax.random.normal(jax.random.key(11), (1, 1, 128))
    keys = l2norm(base + 0.05 * jax.random.normal(jax.random.key(12), (4, 64, 128)))
    beta = 0.95
    lower = jnp.tril(beta * keys @ jnp.swapaxes(keys, -1, -2), -1)
    want = jnp.linalg.inv(lower + jnp.eye(64))
    got = unit_lower_inverse(lower, "highest")
    assert bool(jnp.all(jnp.isfinite(got)))
    np.testing.assert_allclose(got, want, atol=1e-3, rtol=1e-3)


def test_doubling_inverse_variant_on_correlated_keys_matches_recurrence():
    base = jax.random.normal(jax.random.key(13), (1, 1, 1, 8))
    k = l2norm(base + 0.05 * jax.random.normal(jax.random.key(14), (2, 64, 3, 8)))
    q, v, g, beta = _inputs(15, length=64)[0][..., :8], *_inputs(15, length=64)[2:]
    g = g * 0.01  # slow decay: the hard case
    beta = jnp.clip(beta, 0.9, 1.0)
    ref, _ = recurrent_gated_delta_rule(q, k, v[..., :6], g, beta)
    out, _ = chunk_gated_delta_rule(q, k, v[..., :6], g, beta, chunk=16, inverse="doubling")
    np.testing.assert_allclose(out, ref, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("impl", ["chunked", "recurrent"])
def test_dispatch_matches_the_reference(impl):
    q, k, v, g, beta = _inputs(6, length=40)
    seg = _segments(q.shape[0], q.shape[1], 6)
    want, _ = recurrent_gated_delta_rule(q, k, v, g, beta, seg)
    got = gated_delta_rule(q, k, v, g, beta, seg, impl=impl, chunk=8, precision="highest")
    np.testing.assert_allclose(got, want, atol=2e-5, rtol=2e-5)
    with pytest.raises(ValueError, match="impl"):
        gated_delta_rule(q, k, v, g, beta, seg, impl="mosaic")
