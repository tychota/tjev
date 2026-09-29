# Copyright 2026 The tjev Authors.
# SPDX-License-Identifier: Apache-2.0
#
# The kernel design is adapted from MaxText (https://github.com/AI-Hypercomputer/maxtext,
# PR #5364, commit 61ef5db17a9fbefc8e4cca5e0caa0b506ae893e7, and the atwigg/mlperf branch
# @a51bcf5 with PR #5399):
#   src/maxtext/kernels/gdn/gdn_bwd/pallas_mosaic_tpu_bwd.py (reverse chunk order through
#     the index maps, dState carried in VMEM scratch initialised at chunk 0, the saved
#     per-chunk states instead of a recompute)
#   src/maxtext/kernels/gdn/compute_gdn.py and PR #5399 (bf16_3x "precise" matmuls for the
#     intra-chunk products)
# Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0.
# Modifications (2026, The tjev Authors): written for tjev's chunk layout (the core inputs of
# the chunked XLA reference), the chunk-local WY terms in XLA with its doubling inverse, a
# forward kernel with the state in scratch across an "arbitrary" chunk axis, several heads
# per program; see NOTICE.
"""Gated DeltaNet chunked delta rule with a Pallas TPU recurrence (forward and backward).

Drop-in for :func:`tjev.kernels.gated_delta_rule.xla.chunk_gated_delta_rule` (no initial
state), selected with ``compute.gdn_impl=pallas_tpu``. Layout of the computation:

forward
  prepare (XLA, differentiable)    cumsum/exp gates, segment masks, β scaling
  local (XLA, differentiable)      L = strict((kβ kᵀ)⊙D), T = (I+L)⁻¹ by recursive doubling
                                   (matmuls only: Mosaic has no triangular solve),
                                   u = T vβ, w = T kbe, intra = (q kᵀ)⊙D
  recurrence (Pallas, custom_vjp)  per group of heads, chunks in order on an "arbitrary"
                                   grid axis, the state S [Dk, Dv] fp32 in VMEM scratch:
                                     v_new = u − w S;  o = q_read S + intra v_new
                                     S ← carry·S + k_writeᵀ v_new
                                   saves S before each chunk and v_new
backward
  recurrence' (Pallas)             chunks in reverse, G = ∂L/∂S (after the chunk) in scratch:
                                     d_vnew = intraᵀ do + k_write G
                                     d_u = d_vnew, d_w = −d_vnew Sᵀ, d_intra = do v_newᵀ
                                     d_qread = do Sᵀ, d_kwrite = v_new Gᵀ, d_carry = ΣS⊙G
                                     G ← carry·G + q_readᵀ do − wᵀ d_vnew
  local and prepare                JAX autodiff (they are plain XLA)

Matmul precision (``precision``): "highest" = fp32 (6 bf16 passes on the MXU, the parity
reference), "high" = bf16_3x (MaxText's choice for the intra-chunk products: 1-pass bf16
there broke the gate gradients), "bf16" = bf16 operands with fp32 accumulation. The state
and every gate stay fp32 in all modes. On a non-TPU backend the kernels run in Pallas
interpret mode (tests).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .xla import PRECISIONS, doubling_masks, mm, unit_lower_inverse

F32 = jnp.float32
_LAX_PRECISION = {
    "highest": jax.lax.Precision.HIGHEST,
    "high": jax.lax.Precision.HIGH,
    "bf16": jax.lax.Precision.DEFAULT,
}


# ------------------------------------------------------------------------------------
# XLA side: differentiable preparation and chunk-local terms


def prepare(q, k, v, g, beta, segment_ids, chunk: int):
    """[B,T,H,*] -> chunked fp32 core inputs [B·H, N, C, *] (+ carry [B·H, N]).

    The chunked form's quantities (xla.chunk_gated_delta_rule), for any chunk size."""
    batch, length, heads, _ = k.shape
    n = length // chunk

    def chunked(x):
        x = x.reshape(batch, n, chunk, heads, *x.shape[3:])
        return jnp.moveaxis(x, 3, 1).reshape(batch * heads, n, chunk, *x.shape[4:])

    q, k, v, g, beta = (chunked(x.astype(F32)) for x in (q, k, v, g, beta))
    seg = jnp.repeat(segment_ids.reshape(batch, 1, n, chunk), heads, axis=1).reshape(
        batch * heads, n, chunk
    )
    before = jnp.concatenate([jnp.full((batch * heads, 1), -2, seg.dtype), seg[:, :-1, -1]], axis=1)
    continues = seg == before[..., None]
    in_last = seg == seg[..., -1:]
    causal = jnp.tril(jnp.ones((chunk, chunk), bool))
    same = (seg[..., :, None] == seg[..., None, :]) & causal
    cum = jnp.cumsum(g, axis=-1)
    decay = jnp.where(
        same, jnp.exp(jnp.where(same, cum[..., :, None] - cum[..., None, :], 0.0)), 0.0
    )
    kb = k * beta[..., None]
    return {
        "k": k,
        "kb": kb,
        "q": q,
        "vb": v * beta[..., None],
        "kbe": kb * (jnp.exp(cum) * continues)[..., None],
        "decay": decay,
        "q_read": q * (jnp.exp(cum) * continues)[..., None],
        "k_write": k * (jnp.exp(cum[..., -1:] - cum) * in_last)[..., None],
        "carry": jnp.exp(cum[..., -1]) * continues[..., -1],
    }


def local(p, precision: str):
    """Chunk-local WY terms (u, w, intra), batched over [B·H, N]."""
    chunk = p["decay"].shape[-1]
    dv = p["vb"].shape[-1]
    strict = jnp.tril(jnp.ones((chunk, chunk), bool), -1)
    kt = jnp.swapaxes(p["k"], -1, -2)
    lower = jnp.where(strict, mm(p["kb"], kt, precision) * p["decay"], 0.0)
    tinv = unit_lower_inverse(lower, precision)
    uw = mm(tinv, jnp.concatenate([p["vb"], p["kbe"]], axis=-1), precision)
    intra = mm(p["q"], kt, precision) * p["decay"]
    return uw[..., :dv], uw[..., dv:], intra


# ------------------------------------------------------------------------------------
# Pallas side: the chunk recurrence


def _dot(a, b, precision, contract=((1,), (0,))):
    if precision == "bf16":
        a, b = a.astype(jnp.bfloat16), b.astype(jnp.bfloat16)
    return jax.lax.dot_general(
        a, b, (contract, ((), ())), precision=_LAX_PRECISION[precision], preferred_element_type=F32
    )


def _dot_nt(a, b, precision):  # a @ bᵀ
    return _dot(a, b, precision, ((1,), (1,)))


def _dot_tn(a, b, precision):  # aᵀ @ b
    return _dot(a, b, precision, ((0,), (0,)))


def _fwd_kernel(
    w_ref, u_ref, qr_ref, kw_ref, intra_ref, carry_ref, out_ref, st_ref, vn_ref, s_scr,
    *, heads: int, precision: str,
):  # fmt: skip
    @pl.when(pl.program_id(1) == 0)
    def _():
        s_scr[...] = jnp.zeros(s_scr.shape, F32)

    for h in range(heads):
        s = s_scr[h]
        st_ref[h] = s
        v_new = u_ref[h] - _dot(w_ref[h], s, precision)
        out_ref[h] = _dot(qr_ref[h], s, precision) + _dot(intra_ref[h], v_new, precision)
        vn_ref[h] = v_new
        # carry_ref[h] is [1, Dv] (the chunk's scalar decay broadcast along lanes)
        s_scr[h] = s * carry_ref[h] + _dot_tn(kw_ref[h], v_new, precision)


def _bwd_kernel(
    st_ref, w_ref, qr_ref, kw_ref, intra_ref, carry_ref, vn_ref, do_ref,
    dw_ref, du_ref, dqr_ref, dkw_ref, dintra_ref, dcarry_ref, g_scr,
    *, heads: int, precision: str,
):  # fmt: skip
    @pl.when(pl.program_id(1) == 0)
    def _():
        g_scr[...] = jnp.zeros(g_scr.shape, F32)

    for h in range(heads):
        grad, s, d_out, v_new = g_scr[h], st_ref[h], do_ref[h], vn_ref[h]
        d_vnew = _dot_tn(intra_ref[h], d_out, precision) + _dot(kw_ref[h], grad, precision)
        du_ref[h] = d_vnew
        dw_ref[h] = -_dot_nt(d_vnew, s, precision)
        dintra_ref[h] = _dot_nt(d_out, v_new, precision)
        dqr_ref[h] = _dot_nt(d_out, s, precision)
        dkw_ref[h] = _dot_nt(v_new, grad, precision)
        total = jnp.sum(jnp.sum(s * grad, axis=0, keepdims=True), axis=1, keepdims=True)
        dcarry_ref[h] = jnp.broadcast_to(total, dcarry_ref.shape[1:])
        g_scr[h] = (
            grad * carry_ref[h]
            + _dot_tn(qr_ref[h], d_out, precision)
            - _dot_tn(w_ref[h], d_vnew, precision)
        )


def _interpret():
    # the generic interpreter (no effects, unlike pltpu.InterpretParams: remat can't AD those)
    return jax.default_backend() != "tpu"


def _specs(heads: int, n: int, reverse: bool, *tails):
    """Blocks of ``heads`` rows × one chunk × a full [*tail] tile; chunks forward or reversed."""

    def index(i, j):  # every tail is 2-D: the chunk's full tile
        return (i, n - 1 - j if reverse else j, 0, 0)

    return [pl.BlockSpec((heads, None, *tail), index) for tail in tails]


def _params():
    return pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"))


def _forward(w, u, q_read, k_write, intra, carry, *, heads: int, precision: str):
    lead, n, c, dk = w.shape
    dv = u.shape[-1]
    carry = jnp.broadcast_to(carry[..., None, None], (lead, n, 1, dv))
    shape = jax.ShapeDtypeStruct
    return pl.pallas_call(
        functools.partial(_fwd_kernel, heads=heads, precision=precision),
        out_shape=(
            shape((lead, n, c, dv), F32),  # out
            shape((lead, n, dk, dv), F32),  # state before each chunk
            shape((lead, n, c, dv), F32),  # v_new
        ),
        grid=(lead // heads, n),
        in_specs=_specs(heads, n, False, (c, dk), (c, dv), (c, dk), (c, dk), (c, c), (1, dv)),
        out_specs=tuple(_specs(heads, n, False, (c, dv), (dk, dv), (c, dv))),
        scratch_shapes=[pltpu.VMEM((heads, dk, dv), F32)],
        compiler_params=_params(),
        interpret=_interpret(),
        name="gdn_recurrence_fwd",
    )(w, u, q_read, k_write, intra, carry)


def _backward(states, w, q_read, k_write, intra, carry, v_new, d_out, *, heads, precision):
    lead, n, c, dk = w.shape
    dv = v_new.shape[-1]
    carry = jnp.broadcast_to(carry[..., None, None], (lead, n, 1, dv))
    shape = jax.ShapeDtypeStruct
    dw, du, dqr, dkw, dintra, dcarry = pl.pallas_call(
        functools.partial(_bwd_kernel, heads=heads, precision=precision),
        out_shape=(
            shape((lead, n, c, dk), F32),
            shape((lead, n, c, dv), F32),
            shape((lead, n, c, dk), F32),
            shape((lead, n, c, dk), F32),
            shape((lead, n, c, c), F32),
            shape((lead, n, 1, dv), F32),
        ),
        grid=(lead // heads, n),
        in_specs=_specs(
            heads, n, True, (dk, dv), (c, dk), (c, dk), (c, dk), (c, c), (1, dv), (c, dv), (c, dv)
        ),
        out_specs=tuple(
            _specs(heads, n, True, (c, dk), (c, dv), (c, dk), (c, dk), (c, c), (1, dv))
        ),
        scratch_shapes=[pltpu.VMEM((heads, dk, dv), F32)],
        compiler_params=_params(),
        interpret=_interpret(),
        name="gdn_recurrence_bwd",
    )(states, w, q_read, k_write, intra, carry, v_new, d_out)
    return dw, du, dqr, dkw, dintra, dcarry[..., 0, 0]


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def recurrence(w, u, q_read, k_write, intra, carry, heads: int, precision: str):
    """Chunk recurrence over [B·H, N, C, *] core inputs; returns out [B·H, N, C, Dv]."""
    return _forward(w, u, q_read, k_write, intra, carry, heads=heads, precision=precision)[0]


def _recurrence_fwd(w, u, q_read, k_write, intra, carry, heads, precision):
    out, states, v_new = _forward(
        w, u, q_read, k_write, intra, carry, heads=heads, precision=precision
    )
    return out, (w, q_read, k_write, intra, carry, states, v_new)


def _recurrence_bwd(heads, precision, res, d_out):
    w, q_read, k_write, intra, carry, states, v_new = res
    dw, du, dqr, dkw, dintra, dcarry = _backward(
        states, w, q_read, k_write, intra, carry, v_new, d_out.astype(F32),
        heads=heads, precision=precision,
    )  # fmt: skip
    return dw, du, dqr, dkw, dintra, dcarry


recurrence.defvjp(_recurrence_fwd, _recurrence_bwd)


# ------------------------------------------------------------------------------------
# Fused forward: prepare + chunk-local terms + recurrence in one kernel


def _fused_fwd_kernel(
    q_ref, k_ref, v_ref, g_ref, b_ref, seg_ref, prev_ref, m_ref,
    o_ref, st_ref, vn_ref, s_scr, *, heads: int, dk: int, dv: int, precision: str,
):  # fmt: skip
    """One (row, head group, chunk) program. Everything chunk-local lives in VMEM: the
    cumulative gates, segment masks, decay, (I+L)⁻¹, u, w and intra never reach HBM (they
    are ~80% of the unfused GDN core's HBM traffic)."""

    @pl.when(pl.program_id(2) == 0)
    def _():
        s_scr[...] = jnp.zeros(s_scr.shape, F32)

    c = seg_ref.shape[0]
    row = jax.lax.broadcasted_iota(jnp.int32, (c, c), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (c, c), 1)
    eye = (row == col).astype(F32)
    causal = (row >= col).astype(F32)
    seg_col = seg_ref[...]  # [C, 1] segment ids (float: exact for small integers)
    seg_row = _dot_tn(seg_col, eye, "highest")  # [1, C]: a transpose through the MXU
    cont = (seg_col == prev_ref[:, 0:1]).astype(F32)  # reads the incoming state
    in_last = (seg_col == seg_col[c - 1 : c, :]).astype(F32)  # writes the outgoing state
    same = (seg_col == seg_row) & (row >= col)
    strict = row > col
    for h in range(heads):
        q = q_ref[:, h * dk : (h + 1) * dk]
        k = k_ref[:, h * dk : (h + 1) * dk]
        v = v_ref[:, h * dv : (h + 1) * dv]
        beta = b_ref[:, h : h + 1]
        cum = _dot(causal, g_ref[:, h : h + 1], "highest")  # [C, 1] in-chunk cumsum
        cum_row = _dot_tn(cum, eye, "highest")
        decay = jnp.where(same, jnp.exp(jnp.where(same, cum - cum_row, 0.0)), 0.0)
        kb = k * beta
        lower = jnp.where(strict, _dot_nt(kb, k, precision) * decay, 0.0)
        tinv = eye  # (I + L)⁻¹ by recursive doubling (xla.unit_lower_inverse)
        for level in range(m_ref.shape[0]):
            off = lower * m_ref[level]
            tinv = tinv - _dot(_dot(tinv, off, precision), tinv, precision)
        grow = jnp.exp(cum) * cont
        u = _dot(tinv, v * beta, precision)
        w = _dot(tinv, kb * grow, precision)
        intra = _dot_nt(q, k, precision) * decay
        cum_last = cum[c - 1 : c, :]
        k_write = k * (jnp.exp(cum_last - cum) * in_last)
        carry = jnp.exp(cum_last) * cont[c - 1 : c, :]  # [1, 1]
        s = s_scr[h]
        st_ref[h] = s
        v_new = u - _dot(w, s, precision)
        o_ref[:, h * dv : (h + 1) * dv] = _dot(q * grow, s, precision) + _dot(
            intra, v_new, precision
        )
        vn_ref[h] = v_new
        s_scr[h] = s * carry + _dot_tn(k_write, v_new, precision)


def _fused_forward(q, k, v, g, beta, segment_ids, *, chunk: int, heads: int, precision: str):
    """q,k [B,T,H,Dk], v [B,T,H,Dv], g,beta [B,T,H], T % chunk == 0. Returns out
    [B,T,H·Dv], states [B,H,N,Dk,Dv] (before each chunk) and v_new [B,H,N,C,Dv]."""
    batch, length, n_heads, dk = k.shape
    dv = v.shape[-1]
    n, groups = length // chunk, n_heads // heads

    def by_group(x):  # [B,T,H] -> [B, H/heads, T, heads]
        return jnp.moveaxis(x.astype(F32).reshape(batch, length, groups, heads), 2, 1)

    seg = segment_ids.astype(F32)
    before = jnp.concatenate([jnp.full((batch, 1), -2.0), seg[:, chunk - 1 :: chunk][:, :-1]], 1)
    masks = jnp.asarray(np.stack(doubling_masks(chunk)), F32)
    shape = jax.ShapeDtypeStruct
    tok = lambda width: pl.BlockSpec((None, chunk, width), lambda b, h, j: (b, j, h))  # noqa: E731
    per_head = lambda *tail: pl.BlockSpec(  # noqa: E731
        (None, heads, None, *tail), lambda b, h, j: (b, h, j, 0, 0)
    )
    out, states, v_new = pl.pallas_call(
        functools.partial(_fused_fwd_kernel, heads=heads, dk=dk, dv=dv, precision=precision),
        out_shape=(
            shape((batch, length, n_heads * dv), F32),
            shape((batch, n_heads, n, dk, dv), F32),
            shape((batch, n_heads, n, chunk, dv), F32),
        ),
        grid=(batch, groups, n),
        in_specs=[
            tok(heads * dk),
            tok(heads * dk),
            tok(heads * dv),
            pl.BlockSpec((None, None, chunk, heads), lambda b, h, j: (b, h, j, 0)),
            pl.BlockSpec((None, None, chunk, heads), lambda b, h, j: (b, h, j, 0)),
            pl.BlockSpec((None, chunk, 1), lambda b, h, j: (b, j, 0)),
            pl.BlockSpec((None, None, 1, 128), lambda b, h, j: (b, j, 0, 0)),
            pl.BlockSpec(masks.shape, lambda b, h, j: (0, 0, 0)),
        ],
        out_specs=(tok(heads * dv), per_head(dk, dv), per_head(chunk, dv)),
        scratch_shapes=[pltpu.VMEM((heads, dk, dv), F32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=_interpret(),
        name="gdn_fused_fwd",
    )(
        q.astype(F32).reshape(batch, length, n_heads * dk),
        k.astype(F32).reshape(batch, length, n_heads * dk),
        v.astype(F32).reshape(batch, length, n_heads * dv),
        by_group(g),
        by_group(beta),
        seg[..., None],
        jnp.broadcast_to(before[..., None, None], (batch, n, 1, 128)),
        masks,
    )
    return out, states, v_new


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def fused_rule(q, k, v, g, beta, segment_ids, chunk: int, heads: int, precision: str):
    """Fused-forward GDN: out [B,T,H·Dv] fp32. The backward recomputes the chunk-local
    terms in XLA (their VJP by autodiff) around the Pallas reverse-recurrence kernel."""
    return _fused_forward(q, k, v, g, beta, segment_ids, chunk=chunk, heads=heads,
                          precision=precision)[0]  # fmt: skip


def _fused_rule_fwd(q, k, v, g, beta, segment_ids, chunk, heads, precision):
    out, states, v_new = _fused_forward(
        q, k, v, g, beta, segment_ids, chunk=chunk, heads=heads, precision=precision
    )
    return out, ((q, k, v, g, beta, segment_ids), states, v_new)


def _fused_rule_bwd(chunk, heads, precision, res, d_out):
    (q, k, v, g, beta, segment_ids), states, v_new = res
    batch, length, n_heads, dk = k.shape
    dv, n = v.shape[-1], length // chunk

    def terms(q, k, v, g, beta):
        p = prepare(q, k, v, g, beta, segment_ids, chunk)
        u, w, intra = local(p, precision)
        return w, u, p["q_read"], p["k_write"], intra, p["carry"]

    (w, u, q_read, k_write, intra, carry), back = jax.vjp(terms, q, k, v, g, beta)
    del u
    lead = batch * n_heads
    d_out = d_out.astype(F32).reshape(batch, n, chunk, n_heads, dv)
    d_out = jnp.moveaxis(d_out, 3, 1).reshape(lead, n, chunk, dv)
    grads = _backward(
        states.reshape(lead, n, dk, dv), w, q_read, k_write, intra, carry,
        v_new.reshape(lead, n, chunk, dv), d_out, heads=heads, precision=precision,
    )  # fmt: skip
    return (*back(grads), None)


fused_rule.defvjp(_fused_rule_fwd, _fused_rule_bwd)


def heads_per_block(lead: int, preferred: int = 8) -> int:
    """Heads per kernel program: fewer, larger grid steps (8 × 64 rows per MXU pass)."""
    size = preferred
    while lead % size:
        size //= 2
    return size


def chunk_gated_delta_rule_tpu(
    q, k, v, g, beta, segment_ids=None, *, chunk: int = 64, precision: str = "high",
    fused: bool = True,
) -> jax.Array:  # fmt: skip
    """The chunked XLA reference's contract, without an initial or final state.

    ``fused``: one forward kernel for prepare + chunk-local terms + recurrence (default);
    False: those in XLA around the recurrence kernel (``compute.gdn_impl=pallas_tpu_split``).
    Returns out [B,T,H,Dv] fp32. T is padded to a multiple of ``chunk`` with segment-0
    tokens (an isolated segment with q = k = v = β = 0, g = 0)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision in {PRECISIONS}")
    batch, length, heads, _ = k.shape
    dv = v.shape[-1]
    if segment_ids is None:
        segment_ids = jnp.ones((batch, length), jnp.int32)
    pad = (-length) % chunk
    if pad:
        width = lambda x: ((0, 0), (0, pad)) + ((0, 0),) * (x.ndim - 2)  # noqa: E731
        q, k, v, g, beta, segment_ids = (
            jnp.pad(x, width(x)) for x in (q, k, v, g, beta, segment_ids)
        )
    if fused:
        out = fused_rule(q, k, v, g, beta, segment_ids, chunk, heads_per_block(heads), precision)
        return out.reshape(batch, length + pad, heads, dv)[:, :length]
    p = prepare(q, k, v, g, beta, segment_ids, chunk)
    u, w, intra = local(p, precision)
    lead = batch * heads
    out = recurrence(
        w, u, p["q_read"], p["k_write"], intra, p["carry"], heads_per_block(lead), precision
    )
    out = out.reshape(batch, heads, length + pad, dv)
    return jnp.swapaxes(out, 1, 2)[:, :length]
