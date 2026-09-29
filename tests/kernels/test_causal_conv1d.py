"""The segmented causal conv1d and its custom VJP."""

import jax
import jax.numpy as jnp
import numpy as np

from tjev.kernels import segmented_causal_conv1d


def _segments(batch, length, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(batch):
        cuts = np.sort(rng.choice(np.arange(1, length), size=4, replace=False))
        row = np.searchsorted(cuts, np.arange(length), side="right") + 1
        rows.append(row)
    return jnp.asarray(np.stack(rows), jnp.int32)


def test_segmented_conv_never_mixes():
    x = jax.random.normal(jax.random.key(0), (2, 20, 5))
    w = jax.random.normal(jax.random.key(1), (4, 5))
    seg = _segments(2, 20, 7)
    out = segmented_causal_conv1d(x, w, seg)
    for b in range(2):
        for s in np.unique(np.asarray(seg[b])):
            idx = np.flatnonzero(np.asarray(seg[b]) == s)
            alone = segmented_causal_conv1d(
                x[b : b + 1, idx], w, jnp.ones((1, len(idx)), jnp.int32)
            )
            np.testing.assert_allclose(out[b, idx], alone[0], atol=1e-6)


def _conv_reference(x, w, seg):
    taps = w.shape[0]
    length = x.shape[1]
    xp = jnp.pad(x, ((0, 0), (taps - 1, 0), (0, 0)))
    sp = jnp.pad(seg, ((0, 0), (taps - 1, 0)), constant_values=-1)
    out = jnp.zeros(x.shape, jnp.float32)
    for i in range(taps):
        keep = (sp[:, i : i + length] == seg)[..., None]
        out = out + jnp.where(keep, xp[:, i : i + length], 0.0) * w[i]
    return out


def test_conv_custom_vjp_matches_autodiff():
    x = jax.random.normal(jax.random.key(0), (2, 30, 7))
    w = jax.random.normal(jax.random.key(1), (4, 7))
    seg = _segments(2, 30, 3)
    cot = jax.random.normal(jax.random.key(2), (2, 30, 7))
    np.testing.assert_allclose(
        segmented_causal_conv1d(x, w, seg), _conv_reference(x, w, seg), atol=1e-5
    )
    got = jax.grad(lambda a, b: jnp.sum(segmented_causal_conv1d(a, b, seg) * cot), (0, 1))(x, w)
    want = jax.grad(lambda a, b: jnp.sum(_conv_reference(a, b, seg) * cot), (0, 1))(x, w)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)
