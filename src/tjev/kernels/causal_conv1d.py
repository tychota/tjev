"""Depthwise causal conv1d that never mixes packed segments (GDN input conv, XLA)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _tap_masks(segment_ids, taps):
    """keep[L] [B,T,1]: token t may read token t-L (same segment). L = lag 0..K-1."""
    sp = jnp.pad(segment_ids, ((0, 0), (taps - 1, 0)), constant_values=-1)
    length = segment_ids.shape[1]
    return [
        (sp[:, taps - 1 - lag : taps - 1 - lag + length] == segment_ids)[..., None]
        for lag in range(taps)
    ]


def _shift(x, lag):  # y[t] = x[t-lag], zeros before the start
    if lag == 0:
        return x
    return jnp.pad(x, ((0, 0), (lag, 0), (0, 0)))[:, : x.shape[1]]


def _unshift(g, lag):  # adjoint of _shift: y[t] = g[t+lag]
    if lag == 0:
        return g
    return jnp.pad(g, ((0, 0), (0, lag), (0, 0)))[:, lag:]


@jax.custom_vjp
def segmented_causal_conv1d(x, weight, segment_ids):
    """Depthwise causal conv (HF ``conv1d``, padding K-1) that never mixes segments.

    x [B,T,C]; weight [K,C] (weight[K-1] multiplies the current token). fp32
    accumulation, output in x.dtype like HF. Custom VJP: the backward pass keeps only
    x, weight and segment ids (no per-tap copies), which dominated GDN activation memory.
    """
    taps = weight.shape[0]
    keeps = _tap_masks(segment_ids, taps)
    out = jnp.zeros(x.shape, jnp.float32)
    for lag in range(taps):
        w = weight[taps - 1 - lag].astype(jnp.float32)
        out = out + jnp.where(keeps[lag], _shift(x, lag).astype(jnp.float32), 0.0) * w
    return out.astype(x.dtype)


def _conv_fwd(x, weight, segment_ids):
    return segmented_causal_conv1d(x, weight, segment_ids), (x, weight, segment_ids)


def _conv_bwd(res, g):
    x, weight, segment_ids = res
    taps = weight.shape[0]
    keeps = _tap_masks(segment_ids, taps)
    g = g.astype(jnp.float32)
    dx = jnp.zeros(x.shape, jnp.float32)
    dw = []
    for lag in range(taps):
        gk = jnp.where(keeps[lag], g, 0.0)
        dx = dx + _unshift(gk * weight[taps - 1 - lag].astype(jnp.float32), lag)
        dw.append(jnp.sum(gk * _shift(x, lag).astype(jnp.float32), axis=(0, 1)))
    dweight = jnp.stack(dw[::-1]).astype(weight.dtype)
    seg_ct = np.zeros(segment_ids.shape, dtype=jax.dtypes.float0)
    return dx.astype(x.dtype), dweight, seg_ct


segmented_causal_conv1d.defvjp(_conv_fwd, _conv_bwd)
