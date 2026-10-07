# Copyright 2026 The tjev Authors.
# SPDX-License-Identifier: Apache-2.0
#
# The design follows MaxText's Pallas GDN kernels (https://github.com/AI-Hypercomputer/maxtext,
# branch atwigg/mlperf: src/maxtext/kernels/gdn/{wrapper,compute_conv1d,compute_gdn}.py, the
# forward's conv halo carried in VMEM scratch and its in-kernel gates / l2norm / GQA, and
# gdn_bwd/pallas_mosaic_tpu_bwd.py, the in-kernel l2norm and gate backward; Copyright 2026
# Google LLC, Apache License 2.0), the flash-linear-attention conv backward
# (https://github.com/fla-org/flash-linear-attention, fla/modules/conv/triton/kernels.py;
# MIT License, Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li) and Dao-AILab
# causal-conv1d (https://github.com/Dao-AILab/causal-conv1d, csrc/causal_conv1d_bwd.cu;
# BSD 3-Clause License, Copyright (c) 2024, Tri Dao): the anti-causal d(input) with the
# pre-activation recomputed and per-token sequence masks on every tap. No code was copied:
# the kernels are written for tjev's layouts and segment masks (see NOTICE).
"""Gated DeltaNet megakernels: the GDN core from the projection outputs, forward and
backward, around the pallas_tpu_v2 chunk step (``compute.gdn_impl=pallas_tpu_mega1`` and
``pallas_tpu_mega2``).

What each kernel does per (row, head group, chunk), everything in VMEM, fp32:

=====================  =================================================================
stage                  work in the kernel
=====================  =================================================================
mega2 only             causal depthwise conv (K taps, frozen weight) on the raw qkv, the
                       K−1 previous rows from a halo (forward: VMEM scratch carried across
                       the chunk axis; backward: the previous 8 rows from XLA), each tap
                       masked to the token's own segment; SiLU; padding rows → 0
mega1 and mega2        q, k, v split per head (lane slices of the token-major block);
                       l2norm(q), l2norm(k) with the zero-row guard, q · Dk^-½; β = σ(b),
                       g = −e^{A_log}·softplus(a + dt_bias), γ = causal · g (one HIGHEST
                       dot for every head of the group), γ as rows by an identity dot;
                       GQA through the index maps; then v2's chunk step (fwd_step /
                       bwd_step)
=====================  =================================================================

The backward emits d(qkv) (token-major, the caller's dtype: in mega2 the pre-conv qkv, so it
feeds the in_proj_qkv backward directly), d(a) and d(b) in closed form (d(a) from the
reverse in-chunk cumsum, one TN dot), and per-token d(A_log) terms; XLA sums those and d(a)
for A_log / dt_bias (dead code in training: they are frozen). The conv backward is the
anti-causal conv: d(x)[t] = Σ_L w[K−1−L]·keep_L[t+L]·dz[t+L], dz = d(act)·SiLU'(z)·valid,
the next chunk's first rows of keep_L·dz (visited just before, chunks run in reverse)
carried in VMEM scratch. The conv weight's gradient is computed (in XLA, from dz, which the
kernel then also writes) only when the weight is differentiated (ground truth; never in
training): the custom VJP uses symbolic zeros to know.

Layouts (no transposes of qkv in XLA): qkv [B, T, 2·Hk·Dk + H·Dv] token-major, q, k and v
read through three BlockSpecs of the same array (lane blocks of hb/r·Dk and hb·Dv; split in
XLA when the offsets are not whole blocks); a and b packed per head group into one
lane-dense [B, G, T, 128] fp32 tile (lanes [0, hb) a, [hb, 2hb) b), A_log and dt_bias
likewise [G, 2, 128]; token masks aux [B, T, 8] (segment id, reads / writes the carried
state, keep_1..keep_{K−1}, valid); the output o token-major [B, T, H·Dv]. Residuals: v2's
T = (I+L)⁻¹ and the states before each chunk (``gdn_tinv``, ``gdn_states``), plus the inputs.
"""

from __future__ import annotations

import contextlib
import functools

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tjev.kernels.gated_delta_rule import pallas_tpu_v2 as v2
from tjev.kernels.gated_delta_rule.xla import PRECISIONS

F32 = jnp.float32
BF16 = jnp.bfloat16
HALO = 8  # rows of halo: one sublane tile, room for K − 1 ≤ 8 taps
LANES = 128
EPS = 1e-6  # xla.l2norm's
_HI = jax.lax.Precision.HIGHEST


def _dot2(a, b, contract=((1,), (0,))):
    """A 2-D fp32 dot at HIGHEST (the gate cumsum and the transposes through the MXU)."""
    return jax.lax.dot_general(a, b, (contract, ((), ())), precision=_HI,
                               preferred_element_type=F32)  # fmt: skip


def _heads(x, n: int, d: int):
    """[C, n·d] → [n, C, d]: whole lane tiles per head (Qwen3.5: d = 128), no data motion."""
    return jnp.stack([x[:, h * d : (h + 1) * d] for h in range(n)], axis=0)


def _unheads(x):
    """[n, C, d] → [C, n·d]."""
    return jnp.concatenate([x[h] for h in range(x.shape[0])], axis=-1)


def _l2norm(x):
    """xla.l2norm on [n, C, D]: (x·r, r), r = rsqrt(Σx² + eps), r = 0 on all-zero rows."""
    sq = jnp.sum(x * x, axis=-1, keepdims=True)
    zero = sq == 0
    r = jnp.where(zero, 0.0, jax.lax.rsqrt(jnp.where(zero, 1.0, sq) + EPS))
    return x * r, r


def _l2norm_bwd(unit, r, dy):
    """d/dx of y = x·r(x) (u = x·r): r·(dy − u·Σ(u·dy)); 0 on zero rows (r = 0)."""
    return r * (dy - unit * jnp.sum(unit * dy, axis=-1, keepdims=True))


class _Gates:
    """β, g and γ for the hb heads of the group from the lane-packed tile [C, 128]: lanes
    [0, hb) a, [hb, 2hb) b; const [2, 128]: A_log and dt_bias in lanes [0, hb)."""

    def __init__(self, tile, const, hb: int):
        c = tile.shape[0]
        self.hb = hb
        self.lane = jax.lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
        row = jax.lax.broadcasted_iota(jnp.int32, (c, c), 0)
        col = jax.lax.broadcasted_iota(jnp.int32, (c, c), 1)
        self.causal = (row >= col).astype(F32)
        eye = (row == col).astype(F32)
        self.a_log = const[0:1, :]
        self.x = tile + const[1:2, :]  # a + dt_bias (lanes < hb)
        g = -jnp.exp(self.a_log) * jax.nn.softplus(self.x)
        self.g = jnp.where(self.lane < hb, g, 0.0)
        self.beta_all = jax.nn.sigmoid(tile)  # lanes [hb, 2hb): β
        cum = _dot2(self.causal, self.g)  # [C, 128]: γ, the in-chunk cumsum of every head
        rows = _dot2(cum, eye, ((0,), (0,)))  # [128, C]: γ transposed through the MXU
        self.cum = jnp.stack([cum[:, h : h + 1] for h in range(hb)], axis=0)  # [hb, C, 1]
        self.cumr = jnp.stack([rows[h : h + 1, :] for h in range(hb)], axis=0)  # [hb, 1, C]
        self.beta = jnp.stack([self.beta_all[:, hb + h : hb + h + 1] for h in range(hb)], axis=0)

    def backward(self, d_cum, d_beta, d_scr):
        """dγ, dβ columns [hb, C, 1] → the gradient tile [C, 128]: lanes [0, hb) d(a),
        [hb, 2hb) d(b), [2hb, 3hb) the per-token d(A_log) terms dg·g."""
        hb = self.hb
        d_scr[...] = jnp.zeros(d_scr.shape, F32)
        for h in range(hb):  # one-lane stores (v2's pattern), not a lane concat
            d_scr[:, h : h + 1] = d_cum[h]
            d_scr[:, hb + h : hb + h + 1] = d_beta[h]
        tile = d_scr[...]
        dg = _dot2(self.causal, tile, ((0,), (0,)))  # reverse in-chunk cumsum: Σ_{i≥j} dγ_i
        da = dg * (-jnp.exp(self.a_log) * jax.nn.sigmoid(self.x))
        db = tile * self.beta_all * (1.0 - self.beta_all)
        d_alog = pltpu.roll(dg * self.g, 2 * hb, 1)  # lanes [0, hb) → [2hb, 3hb)
        lane = self.lane
        return jnp.where(
            lane < hb, da, jnp.where(lane < 2 * hb, db, jnp.where(lane < 3 * hb, d_alog, 0.0))
        )


def _conv(x, halo, w, keeps):
    """Causal depthwise conv of one chunk: x [C, W] fp32, halo [8, W] the 8 rows before it,
    w [K, W] (w[K−1] multiplies the current token), keeps[L−1] [C, 1]: token t may read
    t − L. The sum order of causal_conv1d.segmented_causal_conv1d."""
    taps = w.shape[0]
    ext = jnp.concatenate([halo, x], axis=0)  # [C + 8, W]
    z = x * w[taps - 1 : taps, :]
    for lag in range(1, taps):
        shifted = pltpu.roll(ext, lag, 0)[HALO:, :]  # row t: ext[8 + t − lag] = x[t − lag]
        z = z + jnp.where(keeps[lag - 1], shifted, 0.0) * w[taps - 1 - lag : taps - lag, :]
    return z


def _conv_bwd(dz, carried, w, keeps):
    """Adjoint of _conv in x: d(x)[t] = Σ_L w[K−1−L]·e_L[t + L], e_L = keep_L ⊙ dz, the rows
    past the chunk end from ``carried[L−1]`` (the next chunk's first 8 rows of e_L).
    Returns d(x) and this chunk's first rows of each e_L (for the previous chunk)."""
    taps, c = w.shape[0], dz.shape[0]
    dx = dz * w[taps - 1 : taps, :]
    firsts = []
    for lag in range(1, taps):
        e = jnp.where(keeps[lag - 1], dz, 0.0)
        ext = jnp.concatenate([e, carried[lag - 1]], axis=0)  # [C + 8, W]
        shifted = pltpu.roll(ext, c + HALO - lag, 0)[:c, :]  # row t: ext[t + lag]
        dx = dx + shifted * w[taps - 1 - lag : taps - lag, :]
        firsts.append(e[:HALO, :])
    return dx, firsts


def _silu_grad(z):
    s = jax.nn.sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


class _Dims:
    def __init__(self, hbk, hb, dk, dv, taps, conv, ratio, mode, inverse_block, emit_dz=False):
        self.hbk, self.hb, self.dk, self.dv, self.taps = hbk, hb, dk, dv, taps
        self.conv, self.ratio, self.mode, self.inverse_block = conv, ratio, mode, inverse_block
        self.emit_dz = emit_dz

    def keeps(self, aux):
        return [aux[:, 3 + i : 4 + i] > 0.5 for i in range(self.taps - 1)]

    def valid(self, aux):
        return aux[:, 2 + self.taps : 3 + self.taps] > 0.5


def _prologue(d: _Dims, raws, aux, halos, weights):
    """raw q, k, v blocks → (post-conv SiLU activations, pre-activations z or None)."""
    if not d.conv:
        return raws, None
    keeps, valid = d.keeps(aux), d.valid(aux)
    zs = [_conv(x, h, w, keeps) for x, h, w in zip(raws, halos, weights, strict=True)]
    return [jnp.where(valid, jax.nn.silu(z), 0.0) for z in zs], zs


def _chunk(d: _Dims, acts, gates: _Gates, aux, segr):
    q_unit, q_r = _l2norm(_heads(acts[0], d.hbk, d.dk))
    k_unit, k_r = _l2norm(_heads(acts[1], d.hbk, d.dk))
    v = _heads(acts[2], d.hb, d.dv)
    q = q_unit * d.dk**-0.5
    ch = v2._Chunk(q, k_unit, v, gates.cum, gates.beta, gates.cumr, aux, segr, d.ratio, d.mode)
    return ch, (q_unit, q_r, k_unit, k_r)


_region = v2.region  # the kernel's named regions (TJEV_KERNEL_REGIONS=1)


def _fwd_kernel(*refs, d: _Dims):
    n_in = 10 if d.conv else 7
    xq, xk, xv, gate_ref, const_ref, aux_ref, segr_ref = refs[:7]
    weights_refs = refs[7:10] if d.conv else ()
    o_ref, t_ref, st_ref, s_scr = refs[n_in : n_in + 4]
    halo_refs = refs[n_in + 4 :]

    @pl.when(pl.program_id(2) == 0)
    def _():
        s_scr[...] = jnp.zeros(s_scr.shape, F32)
        for h_ref in halo_refs:
            h_ref[...] = jnp.zeros(h_ref.shape, F32)

    with _region("prologue"):  # loads, the conv + SiLU
        aux = aux_ref[...]
        raws = [x[...].astype(F32) for x in (xq, xk, xv)]
        halos = [h[...] for h in halo_refs]
        weights = [w[...].astype(F32) for w in weights_refs]
        acts, _ = _prologue(d, raws, aux, halos, weights)
        c = aux.shape[0]
        for h_ref, x in zip(halo_refs, raws if d.conv else (), strict=True):  # next halo
            h_ref[...] = x[c - HALO :, :]
    with _region("chunk"):  # gates, l2norm, the chunk's terms
        gates = _Gates(gate_ref[...], const_ref[...], d.hb)
        ch, _ = _chunk(d, acts, gates, aux, segr_ref[...])
    with _region("state_update"):  # the inverse, the output, the next state
        s = s_scr[...]
        st_ref[...] = s.astype(st_ref.dtype)
        out, tinv, s_next = v2.fwd_step(ch, s, d.mode, d.inverse_block)
    with _region("store"):
        t_ref[...] = tinv.astype(t_ref.dtype)
        for h in range(d.hb):  # token-major output: head h in lanes [h·Dv, (h+1)·Dv)
            o_ref[:, h * d.dv : (h + 1) * d.dv] = out[h].astype(o_ref.dtype)
        s_scr[...] = s_next


def _bwd_kernel(*refs, d: _Dims):
    xq, xk, xv, gate_ref, const_ref, aux_ref, segr_ref = refs[:7]
    i = 7
    weights_refs = prev_refs = ()
    if d.conv:
        weights_refs, prev_refs = refs[7:10], refs[10:13]
        i = 13
    t_ref, st_ref, do_ref = refs[i : i + 3]
    i += 3
    dx_refs, dgate_ref = refs[i : i + 3], refs[i + 3]
    i += 4
    dz_refs = ()
    if d.emit_dz:
        dz_refs = refs[i : i + 3]
        i += 3
    g_scr, d_scr = refs[i], refs[i + 1]
    carry_refs = refs[i + 2 :]  # conv: [K−1, 8, W] per part

    @pl.when(pl.program_id(2) == 0)
    def _():
        g_scr[...] = jnp.zeros(g_scr.shape, F32)
        for e_ref in carry_refs:
            e_ref[...] = jnp.zeros(e_ref.shape, F32)

    with _region("prologue"):  # loads, the conv + SiLU recomputed
        aux = aux_ref[...]
        raws = [x[...].astype(F32) for x in (xq, xk, xv)]
        halos = [p[...].astype(F32) for p in prev_refs]
        weights = [w[...].astype(F32) for w in weights_refs]
        acts, zs = _prologue(d, raws, aux, halos, weights)
    with _region("chunk"):
        gates = _Gates(gate_ref[...], const_ref[...], d.hb)
        ch, (q_unit, q_r, k_unit, k_r) = _chunk(d, acts, gates, aux, segr_ref[...])
    with _region("state_bwd"):  # the chunk's backward and the state gradient
        tinv = t_ref[...].astype(F32)
        s = st_ref[...].astype(F32)
        d_out = _heads(do_ref[...].astype(F32), d.hb, d.dv)
        dq, dk, dv, d_cum, d_beta, grad = v2.bwd_step(ch, tinv, s, g_scr[...], d_out, d.ratio,
                                                      d.mode)  # fmt: skip
        g_scr[...] = grad
    with _region("qk_norm_bwd"):
        d_acts = [
            _unheads(_l2norm_bwd(q_unit, q_r, dq * d.dk**-0.5)),
            _unheads(_l2norm_bwd(k_unit, k_r, dk)),
            _unheads(dv),
        ]
    with _region("conv_bwd") if d.conv else contextlib.nullcontext():
        _conv_backward(d, d_acts, zs, weights, carry_refs, dz_refs, aux)
    with _region("store"):
        for ref, dx in zip(dx_refs, d_acts, strict=True):
            ref[...] = dx.astype(ref.dtype)
        dgate_ref[...] = gates.backward(d_cum, d_beta, d_scr)


def _conv_backward(d: _Dims, d_acts, zs, weights, carry_refs, dz_refs, aux) -> None:
    """The conv + SiLU backward of the q, k, v activation gradients, in place in ``d_acts``."""
    if d.conv:
        assert zs is not None
        keeps, valid = d.keeps(aux), d.valid(aux)
        for n, (da, z, w, e_ref) in enumerate(zip(d_acts, zs, weights, carry_refs, strict=True)):
            dz = jnp.where(valid, da * _silu_grad(z), 0.0)
            if d.emit_dz:
                dz_refs[n][...] = dz
            carried = [e_ref[lag] for lag in range(d.taps - 1)]
            dx, firsts = _conv_bwd(dz, carried, w, keeps)
            for lag, first in enumerate(firsts):
                e_ref[lag] = first
            d_acts[n] = dx


class _Layout:
    """Where q, k and v of head group g sit: arrays and lane-block indices."""

    def __init__(self, parts, widths, bases):
        self.parts, self.widths, self.bases = parts, widths, bases

    def specs(self, rows: int, index):
        """BlockSpecs (rows × lane block) for q, k, v; ``index(b, g, i)`` → (b, row block)."""
        out = []
        for width, base in zip(self.widths, self.bases, strict=True):

            def imap(b, g, i, base=base):
                return (*index(b, g, i), base + g)

            out.append(pl.BlockSpec((None, rows, width), imap))
        return out


def _split(x, key_dim: int, value_width: int, key_width: int, hb_dv: int):
    """q, k, v of [..., 2·key_dim + H·Dv]: the same array thrice with lane-block bases when
    the offsets are whole blocks, else three slices (an XLA copy)."""
    if key_dim % key_width == 0 and (2 * key_dim) % hb_dv == 0:
        return [x, x, x], [0, key_dim // key_width, 2 * key_dim // hb_dv]
    q, k, v = jnp.split(x, [key_dim, 2 * key_dim], axis=-1)
    return [q, k, v], [0, 0, 0]


def _masks(segment_ids, chunk: int, taps: int):
    """aux [B, T, 8]: v2.token_masks' three lanes, keep_1..keep_{K−1} (token t may read
    t − L: same segment, t − L ≥ 0), valid (segment > 0); segr [B, N, 1, C]."""
    aux3, segr = v2.token_masks(segment_ids, chunk)
    length = segment_ids.shape[1]
    padded = jnp.pad(segment_ids, ((0, 0), (taps - 1, 0)), constant_values=-1)
    keeps = [
        padded[:, taps - 1 - lag : taps - 1 - lag + length] == segment_ids for lag in range(1, taps)
    ]
    lanes = [aux3, *(k[..., None].astype(F32) for k in keeps)]
    lanes.append((segment_ids > 0)[..., None].astype(F32))
    aux = jnp.concatenate(lanes, axis=-1)
    return jnp.pad(aux, ((0, 0), (0, 0), (0, 8 - aux.shape[-1]))), segr


def _prev_rows(x, chunk: int):
    """[B, T, W] → [B, N, 8, W] fp32: the 8 rows before each chunk (zeros before the first;
    fp32, so the 8-row blocks fill whole (8, 128) tiles, which bf16's (16, 128) would not)."""
    batch, length, width = x.shape
    tail = x.reshape(batch, length // chunk, chunk, width)[:, :, chunk - HALO :, :].astype(F32)
    return jnp.concatenate([jnp.zeros_like(tail[:, :1]), tail[:, :-1]], axis=1)


class _Statics:
    """Everything static about one call (hashable: a custom_vjp nondiff argument)."""

    def __init__(self, *, key_heads, key_dim, value_heads, value_dim, taps, chunk, hb, mode,
                 inverse_block, vmem_fraction, conv, out_dtype):  # fmt: skip
        self.key_heads, self.key_dim = key_heads, key_dim
        self.value_heads, self.value_dim = value_heads, value_dim
        self.taps, self.chunk, self.hb, self.mode = taps, chunk, hb, mode
        self.inverse_block, self.vmem_fraction, self.conv = inverse_block, vmem_fraction, conv
        self.out_dtype = out_dtype
        self.ratio = value_heads // key_heads
        self.hbk = hb // self.ratio
        self.groups = value_heads // hb

    def _key(self):
        return tuple(sorted((k, str(v)) for k, v in vars(self).items()))

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return isinstance(other, _Statics) and self._key() == other._key()

    def dims(self, emit_dz=False):
        return _Dims(self.hbk, self.hb, self.key_dim, self.value_dim, self.taps, self.conv,
                     self.ratio, self.mode, self.inverse_block, emit_dz)  # fmt: skip

    def layout(self, qkv):
        wq, wv = self.hbk * self.key_dim, self.hb * self.value_dim
        parts, bases = _split(qkv, self.key_heads * self.key_dim, wv, wq, wv)
        return _Layout(parts, [wq, wq, wv], bases)


def _common_specs(st: _Statics, n: int, reverse: bool):
    """(chunk index map, specs of gate tile, const, aux, segr)."""
    c = st.chunk
    j = (lambda i: n - 1 - i) if reverse else (lambda i: i)
    specs = [
        pl.BlockSpec((None, None, c, LANES), lambda b, g, i: (b, g, j(i), 0)),
        pl.BlockSpec((None, 2, LANES), lambda b, g, i: (g, 0, 0)),
        pl.BlockSpec((None, c, 8), lambda b, g, i: (b, j(i), 0)),
        pl.BlockSpec((None, None, 1, c), lambda b, g, i: (b, j(i), 0, 0)),
    ]
    return j, specs


def _weight_specs(st: _Statics, layout: _Layout):
    return [
        pl.BlockSpec((st.taps, width), lambda b, g, i, base=base: (0, base + g))
        for width, base in zip(layout.widths, layout.bases, strict=True)
    ]


def _state_dtypes(mode: str):
    return (BF16 if mode == "bf16" else F32), (BF16 if mode in ("mixed", "bf16") else F32)


def _forward(st: _Statics, qkv, gates, const, aux, segr, weights):
    batch, length, _ = qkv.shape
    c, hb, dk, dv = st.chunk, st.hb, st.key_dim, st.value_dim
    n = length // c
    layout = st.layout(qkv)
    j, common = _common_specs(st, n, reverse=False)
    in_specs = [*layout.specs(c, lambda b, g, i: (b, j(i))), *common]
    inputs = [*layout.parts, gates, const, aux, segr]
    scratch = [pltpu.VMEM((hb, dk, dv), F32)]
    if st.conv:
        wl = st.layout(weights)
        in_specs += _weight_specs(st, wl)
        inputs += wl.parts
        scratch += [pltpu.VMEM((HALO, w), F32) for w in layout.widths]
    t_dtype, s_dtype = _state_dtypes(st.mode)
    heads = st.value_heads
    shape = jax.ShapeDtypeStruct
    outputs = (
        shape((batch, length, heads * dv), st.out_dtype),
        shape((batch, heads, n, c, c), t_dtype),
        shape((batch, heads, n, dk, dv), s_dtype),
    )
    blocks = v2._Blocks(c, n, hb, st.ratio, reverse=False)
    out_specs = (
        pl.BlockSpec((None, c, hb * dv), lambda b, g, i: (b, i, g)),
        blocks.per_chunk(c, c),
        blocks.per_chunk(dk, dv),
    )
    return pl.pallas_call(
        functools.partial(_fwd_kernel, d=st.dims()),
        out_shape=outputs,
        cost_estimate=v2._cost(inputs, outputs, batch * heads * n, c, dk, dv, backward=False),
        grid=(batch, st.groups, n),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch,
        compiler_params=v2._params(st.vmem_fraction),
        interpret=v2._interpret(),
        name="gdn_mega2_fwd" if st.conv else "gdn_mega1_fwd",
    )(*inputs)


def _backward(st: _Statics, qkv, gates, const, aux, segr, weights, tinv, states, d_out, emit_dz):
    batch, length, _ = qkv.shape
    c, hb, dk, dv = st.chunk, st.hb, st.key_dim, st.value_dim
    n = length // c
    layout = st.layout(qkv)
    j, common = _common_specs(st, n, reverse=True)
    rows = lambda b, g, i: (b, j(i))  # noqa: E731
    in_specs = [*layout.specs(c, rows), *common]
    inputs = [*layout.parts, gates, const, aux, segr]
    if st.conv:
        wl = st.layout(weights)
        prev = st.layout(_prev_rows(qkv, c))
        in_specs += _weight_specs(st, wl)
        in_specs += [
            pl.BlockSpec((None, None, HALO, w), lambda b, g, i, base=base: (b, j(i), 0, base + g))
            for w, base in zip(prev.widths, prev.bases, strict=True)
        ]
        inputs += [*wl.parts, *prev.parts]
    blocks = v2._Blocks(c, n, hb, st.ratio, reverse=True)
    in_specs += [
        blocks.per_chunk(c, c),
        blocks.per_chunk(dk, dv),
        pl.BlockSpec((None, c, hb * dv), lambda b, g, i: (b, j(i), g)),
    ]
    inputs += [tinv, states, d_out]
    shape = jax.ShapeDtypeStruct
    widths = layout.widths
    totals = [st.key_heads * dk, st.key_heads * dk, st.value_heads * dv]
    out_spec = lambda w: pl.BlockSpec((None, c, w), lambda b, g, i: (b, j(i), g))  # noqa: E731
    outputs = [shape((batch, length, t), qkv.dtype) for t in totals]
    out_specs = [out_spec(w) for w in widths]
    outputs.append(shape((batch, st.groups, length, LANES), F32))
    out_specs.append(pl.BlockSpec((None, None, c, LANES), lambda b, g, i: (b, g, j(i), 0)))
    if emit_dz:
        outputs += [shape((batch, length, t), F32) for t in totals]
        out_specs += [out_spec(w) for w in widths]
    scratch = [pltpu.VMEM((hb, dk, dv), F32), pltpu.VMEM((c, LANES), F32)]
    if st.conv:
        scratch += [pltpu.VMEM((st.taps - 1, HALO, w), F32) for w in widths]
    result = pl.pallas_call(
        functools.partial(_bwd_kernel, d=st.dims(emit_dz)),
        out_shape=tuple(outputs),
        cost_estimate=v2._cost(inputs, outputs, batch * st.value_heads * n, c, dk, dv, True),
        grid=(batch, st.groups, n),
        in_specs=in_specs,
        out_specs=tuple(out_specs),
        scratch_shapes=scratch,
        compiler_params=v2._params(st.vmem_fraction),
        interpret=v2._interpret(),
        name=("gdn_mega2_bwd" if st.conv else "gdn_mega1_bwd") + ("_dw" if emit_dz else ""),
    )(*inputs)
    d_qkv = jnp.concatenate(result[:3], axis=-1)
    dz = jnp.concatenate(result[4:7], axis=-1) if emit_dz else None
    return d_qkv, result[3], dz


def _weight_grad(x, dz, segment_ids, taps: int):
    """d(conv weight) [K, W] = Σ_t dz[t]·keep_L[t]·x[t − L] (causal_conv1d's backward)."""
    from tjev.kernels.causal_conv1d import _shift, _tap_masks

    keeps = _tap_masks(segment_ids, taps)
    x = x.astype(F32)
    dw = [jnp.sum(jnp.where(keeps[lag], dz, 0.0) * _shift(x, lag), axis=(0, 1))
          for lag in range(taps)]  # fmt: skip
    return jnp.stack(dw[::-1])


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _core(st: _Statics, qkv, gates, const, weights, aux, segr, segment_ids):
    return _forward(st, qkv, gates, const, aux, segr, weights)[0]


def _core_fwd(st, qkv, gates, const, weights, aux, segr, segment_ids):
    values = [x.value for x in (qkv, gates, const, weights, aux, segr, segment_ids)]
    qkv_, gates_, const_, weights_, aux_, segr_, seg_ = values
    out, tinv, states = _forward(st, qkv_, gates_, const_, aux_, segr_, weights_)
    # named for remat policies: core_states saves them, and the forward kernel never reruns
    tinv, states = checkpoint_name(tinv, "gdn_tinv"), checkpoint_name(states, "gdn_states")
    emit_dz = bool(st.conv and weights.perturbed)
    res = (qkv_, gates_, const_, weights_, aux_, segr_, seg_, tinv, states)
    return out, (res, () if emit_dz else None)  # the flag as the residuals' structure


def _core_bwd(st, saved, d_out):
    res, flag = saved
    emit_dz = flag is not None
    qkv, gates, const, weights, aux, segr, seg, tinv, states = res
    if isinstance(d_out, jax.custom_derivatives.SymbolicZero):
        d_out = jnp.zeros(d_out.shape, d_out.dtype)
    d_qkv, d_gates, dz = _backward(
        st, qkv, gates, const, aux, segr, weights, tinv, states, d_out, emit_dz
    )
    # A_log (row 0) and dt_bias (row 1) of each group: Σ over rows and tokens (dead code
    # unless they are differentiated)
    hb = st.hb
    d_const = jnp.zeros_like(const)
    d_const = d_const.at[:, 0, :hb].set(d_gates[..., 2 * hb : 3 * hb].sum(axis=(0, 2)))
    d_const = d_const.at[:, 1, :hb].set(d_gates[..., :hb].sum(axis=(0, 2)))
    d_weights = None
    if dz is not None:
        d_weights = _weight_grad(qkv, dz, seg, st.taps).astype(weights.dtype)
    return d_qkv, d_gates, d_const, d_weights, None, None, None


_core.defvjp(_core_fwd, _core_bwd, symbolic_zeros=True)


def check_tpu_shapes(dk: int, dv: int, chunk: int, hb: int) -> None:
    v2.check_tpu_shapes(dk, dv, chunk)
    if 3 * hb > LANES:
        raise ValueError(f"the GDN megakernel packs 3 lanes per head: at most 42 heads per "
                         f"step, not {hb}")  # fmt: skip


def gdn_megakernel(
    qkv, a, b, a_log, dt_bias, conv_weight, segment_ids, *, key_heads: int, key_dim: int,
    value_heads: int, value_dim: int, chunk: int = 64, precision: str = "high",
    inverse: str = "doubling", heads_per_step: int = 0, vmem_fraction: float = 0.0,
) -> jax.Array:  # fmt: skip
    """The GDN core from the projection outputs.

    qkv [B, T, 2·Hk·Dk + H·Dv]: with ``conv_weight`` [K, 2·Hk·Dk + H·Dv] (mega2) the raw
    in_proj_qkv output, the conv + SiLU + padding mask in the kernel; with ``conv_weight``
    None (mega1) the post-conv, post-SiLU, masked activations. a, b [B, T, H] raw gate
    projections, a_log, dt_bias [H]. Returns the delta rule's output [B, T, H, Dv] in qkv's
    dtype (T padded to a multiple of ``chunk`` inside, with segment-0 tokens)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision in {PRECISIONS}")
    if inverse not in v2.INVERSE_BLOCKS:
        raise ValueError(f"the GDN megakernel's inverse in {tuple(v2.INVERSE_BLOCKS)}")
    if chunk & (chunk - 1) or chunk < HALO:
        raise ValueError(f"the GDN megakernel needs a power-of-two chunk ≥ {HALO}, not {chunk}")
    batch, length, width = qkv.shape
    if width != 2 * key_heads * key_dim + value_heads * value_dim:
        raise ValueError(f"qkv width {width} is not 2·{key_heads}·{key_dim} + "
                         f"{value_heads}·{value_dim}")  # fmt: skip
    ratio = value_heads // key_heads
    hb = v2.heads_per_block(value_heads, ratio, heads_per_step or 8)
    if not v2._interpret():
        check_tpu_shapes(key_dim, value_dim, chunk, hb)
    elif 3 * hb > LANES:
        raise ValueError(f"at most 42 heads per step, not {hb}")
    taps = 1 if conv_weight is None else conv_weight.shape[0]
    if not 1 <= taps <= 5:
        raise ValueError(f"conv kernel of {taps} taps: the megakernel supports 1..5")
    pad = (-length) % chunk
    if pad:
        width_ = lambda x: ((0, 0), (0, pad)) + ((0, 0),) * (x.ndim - 2)  # noqa: E731
        qkv, a, b, segment_ids = (jnp.pad(x, width_(x)) for x in (qkv, a, b, segment_ids))
    groups = value_heads // hb
    full = length + pad

    def by_group(x):  # [B, T, H] → [B, G, T, hb]
        return jnp.moveaxis(x.astype(F32).reshape(batch, full, groups, hb), 2, 1)

    gates = jnp.concatenate([by_group(a), by_group(b)], axis=-1)
    gates = jnp.pad(gates, ((0, 0), (0, 0), (0, 0), (0, LANES - 2 * hb)))
    const = jnp.stack([a_log.astype(F32), dt_bias.astype(F32)]).reshape(2, groups, hb)
    const = jnp.pad(jnp.swapaxes(const, 0, 1), ((0, 0), (0, 0), (0, LANES - hb)))
    aux, segr = _masks(segment_ids, chunk, max(taps, 1))
    st = _Statics(
        key_heads=key_heads, key_dim=key_dim, value_heads=value_heads, value_dim=value_dim,
        taps=taps, chunk=chunk, hb=hb, mode=precision, inverse_block=v2.INVERSE_BLOCKS[inverse],
        vmem_fraction=float(vmem_fraction), conv=conv_weight is not None, out_dtype=qkv.dtype,
    )  # fmt: skip
    weights = (jnp.zeros((1, width), F32) if conv_weight is None
               else conv_weight.astype(F32))  # fmt: skip
    out = _core(st, qkv, gates, const, weights, aux, segr, segment_ids)
    return out.reshape(batch, full, value_heads, value_dim)[:, :length]


__all__ = ["gdn_megakernel"]
