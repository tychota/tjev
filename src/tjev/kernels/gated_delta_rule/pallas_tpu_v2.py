# Copyright 2026 The tjev Authors.
# SPDX-License-Identifier: Apache-2.0
#
# The design follows MaxText's Pallas GDN kernels (https://github.com/AI-Hypercomputer/maxtext,
# branch atwigg/mlperf: src/maxtext/kernels/gdn/ and gdn_bwd/, PRs #5153, #5273, #5274,
# #5399; Copyright 2026 Google LLC, Apache License 2.0) and the flash-linear-attention
# Triton kernels (https://github.com/fla-org/flash-linear-attention, fla/ops/gated_delta_rule,
# MIT License, Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li): the forward saves
# (I+L)⁻¹ and the state before each chunk; one reverse backward kernel recomputes the
# chunk-local terms in VMEM, gets the inverse's gradient in closed form and the gate gradient
# from row sums; heads are the batch dimension of every dot; the RHS [vβ | kβe^γ] and the
# LHS [w ; q e^γ] share one matmul each. No code was copied: the kernels are written for
# tjev's layouts and segment masks (see NOTICE).
"""Gated DeltaNet delta rule, second-generation Pallas TPU kernels: fused forward *and*
fused backward (``compute.gdn_impl=pallas_tpu_v2``).

Same contract as :func:`tjev.kernels.gated_delta_rule.xla.chunk_gated_delta_rule` without
initial or final state, plus GQA: q and k may have fewer (key) heads than v; value head h
reads key head h // (H / Hk) through the BlockSpec index map, never a repeated copy.

XLA side (differentiable, cheap): the in-chunk cumulative log decay γ (one [C,C] matmul at
HIGHEST: exact, and no dependence on how XLA lowers a cumsum), the per-token segment masks,
the head-major layouts [B, H, T, D].

Forward kernel ``gdn_v2_fwd``, grid (B, H/hb, N), hb heads per step as the batch dimension
of every dot, the state S (hb, Dk, Dv) fp32 in VMEM scratch across the "arbitrary" chunk axis:
    D = e^{γ_i−γ_j}·[same segment, j ≤ i]      L = strict((kβ kᵀ) ⊙ D)
    T = (I+L)⁻¹ (recursive doubling, batched)  [u | w] = T [vβ | kβ e^γ·cont]
    [w S ; q_read S] (one matmul)              v_new = u − w S
    o = q_read S + ((q kᵀ) ⊙ D) v_new           S ← e^{γ_C}·cont_C·S + k_writeᵀ v_new
  outputs o and the residuals T [B,H,N,C,C] and S before each chunk [B,H,N,Dk,Dv].

Backward kernel ``gdn_v2_bwd``, chunks in reverse (index map N−1−j), G = ∂L/∂S after the
chunk in scratch. With X = Tᵀ d_vnew (no u, w, dT or differentiation through the inverse):
    v_new = T (vβ − kbe S)              d_vnew = intraᵀ do + k_write G
    dL = strict(−X v_newᵀ)               [d_qread ; −d_kbe] = [do ; X] Sᵀ   (one matmul)
    d_intra = do v_newᵀ                  d_kwrite = v_new Gᵀ,  d_carry = Σ S ⊙ G
    G ← carry·G + [q_read ; −kbe]ᵀ [do ; X]                                 (one matmul)
    dq = d_qread e^γ + (d_intra⊙D) k     dk = (dL⊙D)ᵀ kβ + (d_intra⊙D)ᵀ q + …
  and the gate gradient in closed form: ∂/∂γ_i of every e^{±γ} factor, the decay's row sums
  minus column sums (rowsum((dL⊙D)⊙(kβkᵀ)) = Σ kβ ⊙ ((dL⊙D) k), …), plus the last token's
  carry terms; XLA's matmul cumsum takes it back to g.

Precision (``precision``, xla.part): "highest", "high" (bf16_3x), "bf16", "mixed" (bf16_3x
intra-chunk, one bf16 pass for dots with S or G as an operand). The states are saved in
bf16 under "mixed" and "bf16" (their only consumers are those one-pass dots and Σ S⊙G),
T in bf16 under "bf16". v and o keep the caller's dtype (bf16 in the model).
"""

from __future__ import annotations

import contextlib
import functools
import os
import re

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tjev.kernels.gated_delta_rule.xla import PRECISIONS, part

F32 = jnp.float32
BF16 = jnp.bfloat16
_LAX = {
    "highest": jax.lax.Precision.HIGHEST,
    "high": jax.lax.Precision.HIGH,
    "bf16": jax.lax.Precision.DEFAULT,
}
# batched (heads = dim 0) dot_general contractions
_NN = ((2,), (1,))  # a @ b
_NT = ((2,), (2,))  # a @ bᵀ
_TN = ((1,), (1,))  # aᵀ @ b


def region(name: str):
    """A named region inside a Pallas kernel (``jax.named_scope`` lowers to
    ``tpu.trace_start`` / ``trace_stop`` in a Pallas TPU kernel: the kernel's phases in a
    detail capture's timeline) when TJEV_KERNEL_REGIONS=1; otherwise nothing, and the
    kernel is the timed one, op for op."""
    if os.environ.get("TJEV_KERNEL_REGIONS") == "1":
        return jax.named_scope(name)
    return contextlib.nullcontext()


def _dot(a, b, mode: str, contract=_NN):
    if mode == "bf16":
        a, b = a.astype(BF16), b.astype(BF16)
    return jax.lax.dot_general(
        a, b, (contract, ((0,), (0,))), precision=_LAX[mode], preferred_element_type=F32
    )


def _interpret():
    from tjev.kernels.gated_delta_rule import pallas_tpu  # one switch for every GDN kernel

    return pallas_tpu._interpret()


def _vmem_limit(fraction: float = 0.0) -> int | None:
    """The kernels' VMEM limit: 90% of ``--xla_tpu_scoped_vmem_limit_kib`` when
    LIBTPU_INIT_ARGS sets it, else Mosaic's default. ``fraction`` > 0
    (``compute.gdn_vmem_fraction``): that fraction of the chip's VMEM capacity
    (pltpu.get_tpu_info; MaxText and Tokamax use 0.8-0.9), but never above the flag's
    (vmem_limit_bytes must stay below XLA's scoped limit). None in interpret mode."""
    if _interpret():
        return None
    found = re.search(
        r"--xla_tpu_scoped_vmem_limit_kib=(\d+)", os.environ.get("LIBTPU_INIT_ARGS", "")
    )
    flag = int(0.9 * int(found.group(1)) * 1024) if found else None
    if fraction > 0:
        wanted = int(fraction * pltpu.get_tpu_info().vmem_capacity_bytes)
        return min(wanted, flag) if flag else wanted
    return flag


def _params(vmem_fraction: float = 0.0, semantics=("parallel", "parallel", "arbitrary")):
    return pltpu.CompilerParams(
        dimension_semantics=semantics, vmem_limit_bytes=_vmem_limit(vmem_fraction)
    )


def _cost(arrays, outputs, chunks_heads: int, chunk: int, dk: int, dv: int, backward: bool):
    """pl.CostEstimate per call (one MXU pass per product): XLA's scheduler otherwise models
    the kernel as free (MaxText #5427: exposed collectives under FSDP)."""
    c = chunk
    levels = max(c.bit_length() - 1, 0)
    forward = 2 * c * c * (2 * dk + dv + dk + dv) + 4 * levels * c**3 + 2 * c * dk * dv * 3
    per_chunk = (3 * forward if backward else forward) - (4 * levels * c**3 if backward else 0)
    size = lambda x: x.size * jnp.dtype(x.dtype).itemsize  # noqa: E731
    return pl.CostEstimate(
        flops=int(chunks_heads * per_chunk),
        transcendentals=int(chunks_heads * (c * c + 2 * c)),
        bytes_accessed=int(sum(map(size, arrays)) + sum(map(size, outputs))),
    )


INVERSE_BLOCKS = {"doubling": 0, "blocked": 16}  # the in-kernel inverse: its diagonal blocks


def _diagonal_block_inverse(lower, block: int):
    """(I + L_d)⁻¹ for L_d the ``block``×``block`` diagonal blocks of the strictly lower L,
    by exact forward substitution in fp32 on the VPU (no MXU pass), every block and head at
    once. Column form: at step r, row ``b·block + r`` of X is final and is eliminated from
    the rows below it in its block, X_i −= L_{i, b·block+r} · X_{b·block+r} (MaxText's
    ``invert_triangular_matrix`` / FLA's ``solve_tril`` 16×16 diagonal solve)."""
    c = lower.shape[-1]
    bits = block.bit_length() - 1
    row = jax.lax.broadcasted_iota(jnp.int32, (c, c), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (c, c), 1)
    shift = lambda x, n: jax.lax.shift_right_logical(x, jnp.int32(n))  # noqa: E731
    unshift = lambda x, n: jax.lax.shift_left(x, jnp.int32(n))  # noqa: E731
    row_start, col_start = unshift(shift(row, bits), bits), unshift(shift(col, bits), bits)
    same_block = row_start == col_start
    ld = jnp.where(same_block, lower, 0.0)
    x = jnp.broadcast_to((row == col).astype(F32), lower.shape)
    for r in range(block - 1):
        # per row i: L_d[i, start(i) + r] as a column; per column k: X[start(k) + r, k] as a row
        l_col = jnp.sum(jnp.where(col == row_start + r, ld, 0.0), axis=-1, keepdims=True)
        x_row = jnp.sum(jnp.where(row == col_start + r, x, 0.0), axis=-2, keepdims=True)
        x = x - jnp.where(same_block, l_col * x_row, 0.0)
    return x


def _unit_lower_inverse(lower, mode: str, block: int = 0):
    """(I + L)⁻¹ by recursive doubling (xla.unit_lower_inverse), heads batched, the level
    masks from iotas (C a power of two). ``block`` > 1: the doubling starts from the exact
    ``block``×``block`` diagonal-block inverse (:func:`_diagonal_block_inverse`), so only
    log2(C/block) levels (2 dependent dot pairs at C = 64, block 16) remain."""
    c = lower.shape[-1]
    row = jax.lax.broadcasted_iota(jnp.int32, (c, c), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (c, c), 1)
    block = min(block, c)
    if block > 1:
        tinv = _diagonal_block_inverse(lower, block)
        level = block.bit_length() - 1
    else:
        tinv = jnp.broadcast_to((row == col).astype(F32), lower.shape)
        level = 0  # s = 2**level; shifts, not floor division (no sign op in Mosaic)
    shift = lambda x, bits: jax.lax.shift_right_logical(x, jnp.int32(bits))  # noqa: E731
    while (1 << level) < c:
        mask = (
            (shift(row, level + 1) == shift(col, level + 1))
            & ((shift(row, level) & 1) == 1)
            & ((shift(col, level) & 1) == 0)
        )
        off = jnp.where(mask, lower, 0.0)
        tinv = tinv - _dot(_dot(tinv, off, mode), tinv, mode)
        level += 1
    return tinv


def _repeat_heads(x, ratio: int):
    """[Hk, C, D] -> [Hk·ratio, C, D], key head of value head h = h // ratio."""
    if ratio == 1:
        return x
    hk, c, d = x.shape
    return jnp.broadcast_to(x[:, None], (hk, ratio, c, d)).reshape(hk * ratio, c, d)


def _group_sum(x, ratio: int):
    """Adjoint of _repeat_heads."""
    if ratio == 1:
        return x
    h, c, d = x.shape
    return x.reshape(h // ratio, ratio, c, d).sum(axis=1)


class _Chunk:
    """The chunk-local quantities both kernels build from the same inputs: q, k [hbk, C, Dk]
    (key heads), v [hb, C, Dv], γ and β as columns [hb, C, 1], γ as rows [hb, 1, C], aux
    [C, ≥3] token masks and the segment ids as a row [1, C] (all fp32 values)."""

    def __init__(self, q, k, v, cum, beta, cumr, aux, segr, ratio, mode):
        hi = part(mode, "intra")
        c = aux.shape[0]
        self.c = c
        self.q = _repeat_heads(q, ratio)  # [hb, C, Dk]
        self.k = _repeat_heads(k, ratio)
        self.v = v  # [hb, C, Dv]
        self.cum, self.beta = cum, beta  # [hb, C, 1]
        # aux: segment id, reads the incoming state, writes the outgoing
        seg, cont, in_last = aux[:, 0:1], aux[:, 1:2], aux[:, 2:3]
        with region("decay"):
            row = jax.lax.broadcasted_iota(jnp.int32, (c, c), 0)
            col = jax.lax.broadcasted_iota(jnp.int32, (c, c), 1)
            self.strict = row > col
            same = (seg == segr) & (row >= col)  # [C, C]
            diff = self.cum - cumr  # [hb, C, 1] - [hb, 1, C]
            self.decay = jnp.where(same, jnp.exp(jnp.where(same, diff, 0.0)), 0.0)
        self.kb = self.k * self.beta
        self.skk = _dot(self.kb, self.k, hi, _NT)  # kβ kᵀ
        self.grow = jnp.exp(self.cum) * cont  # [hb, C, 1]
        self.kbe = self.kb * self.grow
        cum_last = self.cum[:, c - 1 : c, :]  # [hb, 1, 1]
        self.kws = jnp.exp(cum_last - self.cum) * in_last  # [hb, C, 1]
        self.k_write = self.k * self.kws
        self.carry = jnp.exp(cum_last) * cont[c - 1 : c, :]  # [hb, 1, 1]
        self.q_read = self.q * self.grow
        self.intra = _dot(self.q, self.k, hi, _NT) * self.decay

    @classmethod
    def from_refs(cls, q_ref, k_ref, v_ref, gb_ref, cumr_ref, aux_ref, segr_ref, ratio, mode):
        gb = gb_ref[...]  # [hb, C, 2]: γ, β
        return cls(
            q_ref[...].astype(F32), k_ref[...].astype(F32), v_ref[...].astype(F32),
            gb[..., 0:1], gb[..., 1:2], cumr_ref[...], aux_ref[...], segr_ref[...], ratio, mode,
        )  # fmt: skip


def fwd_step(ch: _Chunk, s, mode: str, inverse_block: int = 0):
    """One chunk forward from the state before it: (o, T, S after the chunk), all fp32."""
    hi, st = part(mode, "intra"), part(mode, "state")
    c, dv = ch.c, ch.v.shape[-1]
    with region("inverse"):
        lower = jnp.where(ch.strict, ch.skk * ch.decay, 0.0)
        tinv = _unit_lower_inverse(lower, hi, inverse_block)
    with region("state_read"):
        uw = _dot(tinv, jnp.concatenate([ch.v * ch.beta, ch.kbe], axis=-1), hi)  # 256 lanes
        u, w = uw[..., :dv], uw[..., dv:]
        reads = _dot(jnp.concatenate([w, ch.q_read], axis=1), s, st)  # [w ; q_read] S
        v_new = u - reads[:, :c]
    with region("output"):
        out = reads[:, c:] + _dot(ch.intra, v_new, hi)
    with region("state_write"):
        return out, tinv, s * ch.carry + _dot(ch.k_write, v_new, hi, _TN)


def bwd_step(ch: _Chunk, tinv, s, grad, d_out, ratio: int, mode: str):
    """One chunk backward: G = ∂L/∂S after the chunk → (dq, dk at the key heads, dv, dγ,
    dβ as [hb, C, 1] columns, G before the chunk), all fp32."""
    hi, st = part(mode, "intra"), part(mode, "state")
    c = ch.c
    with region("v_new"):  # recomputed from T and the state
        v_new = _dot(tinv, ch.v * ch.beta - _dot(ch.kbe, s, st), hi)
    with region("d_vnew"):
        d_vnew = _dot(ch.intra, d_out, hi, _TN) + _dot(ch.k_write, grad, st)
        x = _dot(tinv, d_vnew, hi, _TN)  # Tᵀ d_vnew = ∂/∂(vβ)
    with region("d_decay"):
        m1 = jnp.where(ch.strict, -_dot(x, v_new, hi, _NT), 0.0) * ch.decay  # dL ⊙ D
        m2 = _dot(d_out, v_new, hi, _NT) * ch.decay  # d_intra ⊙ D
    with region("state_grads"):
        reads = _dot(jnp.concatenate([d_out, x], axis=1), s, st, _NT)  # [do ; X] Sᵀ
        d_qread, d_kbe = reads[:, :c], -reads[:, c:]
        d_kwrite = _dot(v_new, grad, st, _NT)
        d_carry = jnp.sum(jnp.sum(s * grad, axis=2, keepdims=True), axis=1, keepdims=True)
        lhs = jnp.concatenate([ch.q_read, -ch.kbe], axis=1)  # [2C, Dk]
        grad = grad * ch.carry + _dot(lhs, jnp.concatenate([d_out, x], axis=1), hi, _TN)
    with region("qk_grads"):
        m1k = _dot(m1, ch.k, hi)
        m1t_kb = _dot(m1, ch.kb, hi, _TN)
        m2k = _dot(m2, ch.k, hi)
        m2t_q = _dot(m2, ch.q, hi, _TN)
        dkb = m1k + d_kbe * ch.grow
        dq = d_qread * ch.grow + m2k
        dk = m1t_kb + m2t_q + d_kwrite * ch.kws + dkb * ch.beta

    def rows(a, b):  # Σ over the feature lanes, per token: [hb, C, 1]
        return jnp.sum(a * b, axis=-1, keepdims=True)

    with region("gate_grads"):
        d_beta = rows(dkb, ch.k) + rows(x, ch.v)
        d_kws = rows(d_kwrite, ch.k)
        d_cum = (
            (rows(d_kbe, ch.kb) + rows(d_qread, ch.q)) * ch.grow  # e^γ in kbe and q_read
            - d_kws * ch.kws  # e^{γ_C − γ} in k_write
            + rows(ch.kb, m1k)
            - rows(ch.k, m1t_kb)  # decay in L: row sums − column sums
            + rows(ch.q, m2k)
            - rows(ch.k, m2t_q)  # decay in intra
        )
        last = jnp.sum(d_kws * ch.kws, axis=1, keepdims=True) + d_carry * ch.carry  # γ_C
        at_last = jax.lax.broadcasted_iota(jnp.int32, (c, 1), 0) == c - 1
        d_cum = d_cum + jnp.where(at_last, last, 0.0)
    return _group_sum(dq, ratio), _group_sum(dk, ratio), x * ch.beta, d_cum, d_beta, grad


def _fwd_kernel(
    q_ref, k_ref, v_ref, gb_ref, cumr_ref, aux_ref, segr_ref,
    o_ref, t_ref, st_ref, s_scr, *, ratio: int, mode: str, inverse_block: int,
):  # fmt: skip
    @pl.when(pl.program_id(2) == 0)
    def _():
        s_scr[...] = jnp.zeros(s_scr.shape, F32)

    ch = _Chunk.from_refs(q_ref, k_ref, v_ref, gb_ref, cumr_ref, aux_ref, segr_ref, ratio, mode)
    s = s_scr[...]
    st_ref[...] = s.astype(st_ref.dtype)
    out, tinv, s_next = fwd_step(ch, s, mode, inverse_block)
    t_ref[...] = tinv.astype(t_ref.dtype)
    o_ref[...] = out.astype(o_ref.dtype)
    s_scr[...] = s_next


def _bwd_kernel(
    q_ref, k_ref, v_ref, gb_ref, cumr_ref, aux_ref, segr_ref, t_ref, st_ref, do_ref,
    dq_ref, dk_ref, dv_ref, dgb_ref, g_scr, *, ratio: int, mode: str,
):  # fmt: skip
    @pl.when(pl.program_id(2) == 0)
    def _():
        g_scr[...] = jnp.zeros(g_scr.shape, F32)

    ch = _Chunk.from_refs(q_ref, k_ref, v_ref, gb_ref, cumr_ref, aux_ref, segr_ref, ratio, mode)
    tinv = t_ref[...].astype(F32)
    s = st_ref[...].astype(F32)
    dq, dk, dv, d_cum, d_beta, grad = bwd_step(
        ch, tinv, s, g_scr[...], do_ref[...].astype(F32), ratio, mode
    )
    g_scr[...] = grad
    dq_ref[...] = dq.astype(dq_ref.dtype)
    dk_ref[...] = dk.astype(dk_ref.dtype)
    dv_ref[...] = dv.astype(dv_ref.dtype)
    dgb_ref[:, :, 0:1] = d_cum  # two one-lane stores, not a lane concat of 1-wide values
    dgb_ref[:, :, 1:2] = d_beta


class _Blocks:
    """BlockSpecs on the grid (B, H/hb, N); chunks in order or reversed (the backward)."""

    def __init__(self, chunk: int, n: int, hb: int, ratio: int, reverse: bool):
        self.chunk, self.hb, self.ratio = chunk, hb, ratio
        self.j = (lambda i: n - 1 - i) if reverse else (lambda i: i)

    def key(self, dim: int):  # q, k [B, Hk, T, D]: value heads h·hb… read key heads h·hb/ratio…
        j = self.j
        return pl.BlockSpec((None, self.hb // self.ratio, self.chunk, dim),
                            lambda b, h, i: (b, h, j(i), 0))  # fmt: skip

    def head(self, dim: int):  # [B, H, T, dim]
        j = self.j
        return pl.BlockSpec((None, self.hb, self.chunk, dim), lambda b, h, i: (b, h, j(i), 0))

    def per_chunk(self, rows: int, cols: int):  # [B, H, N, rows, cols]
        j = self.j
        return pl.BlockSpec((None, self.hb, None, rows, cols),
                            lambda b, h, i: (b, h, j(i), 0, 0))  # fmt: skip

    def inputs(self, dk: int, dv: int):
        """q, k, v, [γ, β], γ as rows [B,N,H,1,C], token masks [B,T,3], segment rows."""
        j, c = self.j, self.chunk
        return [
            self.key(dk),
            self.key(dk),
            self.head(dv),
            self.head(2),
            pl.BlockSpec((None, None, self.hb, 1, c), lambda b, h, i: (b, j(i), h, 0, 0)),
            pl.BlockSpec((None, c, 3), lambda b, h, i: (b, j(i), 0)),
            pl.BlockSpec((None, None, 1, c), lambda b, h, i: (b, j(i), 0, 0)),
        ]


def _layouts(cum, beta, chunk: int):
    batch, heads, length = cum.shape
    n = length // chunk
    gb = jnp.stack([cum, beta.astype(F32)], axis=-1)  # [B, H, T, 2]
    cumr = jnp.swapaxes(cum.reshape(batch, heads, n, chunk), 1, 2)[..., None, :]  # [B,N,H,1,C]
    return gb, cumr, n


def _forward(q, k, v, cum, beta, aux, segr, *, chunk: int, hb: int, mode: str, opts: tuple):
    batch, heads, length, dv = v.shape
    dk = q.shape[-1]
    ratio = heads // q.shape[1]
    gb, cumr, n = _layouts(cum, beta, chunk)
    blocks = _Blocks(chunk, n, hb, ratio, reverse=False)
    state_dtype = BF16 if mode in ("mixed", "bf16") else F32
    t_dtype = BF16 if mode == "bf16" else F32
    shape = jax.ShapeDtypeStruct
    outputs = (
        shape((batch, heads, length, dv), v.dtype),
        shape((batch, heads, n, chunk, chunk), t_dtype),
        shape((batch, heads, n, dk, dv), state_dtype),
    )
    inputs = (q, k, v, gb, cumr, aux, segr)
    return pl.pallas_call(
        functools.partial(_fwd_kernel, ratio=ratio, mode=mode, inverse_block=opts[0]),
        out_shape=outputs,
        cost_estimate=_cost(inputs, outputs, batch * heads * n, chunk, dk, dv, backward=False),
        grid=(batch, heads // hb, n),
        in_specs=blocks.inputs(dk, dv),
        out_specs=(blocks.head(dv), blocks.per_chunk(chunk, chunk), blocks.per_chunk(dk, dv)),
        scratch_shapes=[pltpu.VMEM((hb, dk, dv), F32)],
        compiler_params=_params(opts[1]),
        interpret=_interpret(),
        name="gdn_v2_fwd",
    )(q, k, v, gb, cumr, aux, segr)


def _backward(q, k, v, cum, beta, aux, segr, tinv, states, d_out, *, chunk, hb, mode, opts):
    batch, heads, length, dv = v.shape
    dk = q.shape[-1]
    ratio = heads // q.shape[1]
    gb, cumr, n = _layouts(cum, beta, chunk)
    blocks = _Blocks(chunk, n, hb, ratio, reverse=True)
    shape = jax.ShapeDtypeStruct
    outputs = (
        shape(q.shape, q.dtype),
        shape(k.shape, k.dtype),
        shape(v.shape, v.dtype),
        shape((batch, heads, length, 2), F32),
    )
    inputs = (q, k, v, gb, cumr, aux, segr, tinv, states, d_out)
    dq, dk_, dv_, dgb = pl.pallas_call(
        functools.partial(_bwd_kernel, ratio=ratio, mode=mode),
        out_shape=outputs,
        cost_estimate=_cost(inputs, outputs, batch * heads * n, chunk, dk, dv, backward=True),
        grid=(batch, heads // hb, n),
        in_specs=[
            *blocks.inputs(dk, dv),
            blocks.per_chunk(chunk, chunk),
            blocks.per_chunk(dk, dv),
            blocks.head(dv),
        ],
        out_specs=(blocks.key(dk), blocks.key(dk), blocks.head(dv), blocks.head(2)),
        scratch_shapes=[pltpu.VMEM((hb, dk, dv), F32)],
        compiler_params=_params(opts[1]),
        interpret=_interpret(),
        name="gdn_v2_bwd",
    )(q, k, v, gb, cumr, aux, segr, tinv, states, d_out)
    return dq, dk_, dv_, dgb[..., 0], dgb[..., 1].astype(beta.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10))
def core(q, k, v, cum, beta, aux, segr, chunk: int, hb: int, mode: str, opts: tuple = (0, 0.0)):
    """q,k [B,Hk,T,Dk], v [B,H,T,Dv], γ (in-chunk cumulative log decay) and β [B,H,T],
    aux [B,T,3] token masks, segr [B,N,1,C] segment ids → o [B,H,T,Dv] in v's dtype.
    ``opts``: (the inverse's diagonal block, 0 = pure doubling; the VMEM fraction)."""
    return _forward(q, k, v, cum, beta, aux, segr, chunk=chunk, hb=hb, mode=mode, opts=opts)[0]


def _core_fwd(q, k, v, cum, beta, aux, segr, chunk, hb, mode, opts):
    out, tinv, states = _forward(
        q, k, v, cum, beta, aux, segr, chunk=chunk, hb=hb, mode=mode, opts=opts
    )
    # named for remat policies: core_states saves them, and the forward kernel never reruns
    tinv, states = checkpoint_name(tinv, "gdn_tinv"), checkpoint_name(states, "gdn_states")
    return out, (q, k, v, cum, beta, aux, segr, tinv, states)


def _core_bwd(chunk, hb, mode, opts, res, d_out):
    q, k, v, cum, beta, aux, segr, tinv, states = res
    dq, dk, dv, dcum, dbeta = _backward(
        q, k, v, cum, beta, aux, segr, tinv, states, d_out, chunk=chunk, hb=hb, mode=mode,
        opts=opts,
    )  # fmt: skip
    return dq, dk, dv, dcum, dbeta, jnp.zeros_like(aux), jnp.zeros_like(segr)


core.defvjp(_core_fwd, _core_bwd)


def inchunk_cumsum(g, chunk: int):
    """γ: the cumulative sum of g [B,T,H] within each chunk, by an explicit HIGHEST matmul
    with the lower-triangular ones (exact in fp32 terms; its VJP, the reverse cumsum, too)."""
    batch, length, heads = g.shape
    tri = jnp.tril(jnp.ones((chunk, chunk), F32))
    blocks = g.astype(F32).reshape(batch, length // chunk, chunk, heads)
    cum = jnp.einsum("ij,bnjh->bnih", tri, blocks, precision=jax.lax.Precision.HIGHEST)
    return cum.reshape(batch, length, heads)


def token_masks(segment_ids, chunk: int):
    """aux [B,T,3] (segment id, reads the incoming state, writes the outgoing state) and the
    segment ids as rows [B,N,1,C]; chunk 0 reads nothing (no initial state)."""
    batch, length = segment_ids.shape
    n = length // chunk
    seg = segment_ids.astype(F32).reshape(batch, n, chunk)  # exact for small integers
    before = jnp.concatenate([jnp.full((batch, 1), -2.0, F32), seg[:, :-1, -1]], axis=1)
    cont = seg == before[..., None]
    in_last = seg == seg[..., -1:]
    aux = jnp.stack([seg, cont.astype(F32), in_last.astype(F32)], axis=-1)
    return aux.reshape(batch, length, 3), seg[:, :, None, :]


def heads_per_block(heads: int, ratio: int, preferred: int = 8) -> int:
    """Value heads per program: the largest multiple of the GQA ratio that divides H, up to
    ``preferred`` (the batch dimension of every dot: more heads, more independent MXU work
    per step, more VMEM)."""
    if heads % ratio:
        raise ValueError(f"{heads} value heads are not groups of {ratio}")
    sizes = range(ratio, max(preferred, ratio) + 1, ratio)
    return max(size for size in sizes if heads % size == 0)


def check_tpu_shapes(dk: int, dv: int, chunk: int) -> None:
    """Mosaic layout rules for these kernels (lane-dense D, sublane-aligned chunks)."""
    if chunk % 8:
        raise ValueError(
            f"gdn_impl=pallas_tpu_v2 needs a chunk that is a multiple of 8, not {chunk}"
        )
    if dk % 128 or dv % 128:
        raise ValueError(
            f"gdn_impl=pallas_tpu_v2 needs head dims that are multiples of 128 on TPU, not "
            f"Dk={dk}, Dv={dv}: use gdn_impl=pallas_tpu_split or chunked"
        )


def chunk_gated_delta_rule_v2(
    q, k, v, g, beta, segment_ids=None, *, chunk: int = 64, precision: str = "high",
    inverse: str = "doubling", heads_per_step: int = 0, vmem_fraction: float = 0.0,
) -> jax.Array:  # fmt: skip
    """q,k [B,T,Hk,Dk] (Hk divides H), v [B,T,H,Dv], g,β [B,T,H] → out [B,T,H,Dv] in v's
    dtype. T is padded to a multiple of ``chunk`` with segment-0 tokens.

    ``inverse``: the forward kernel's (I+L)⁻¹, "doubling" (log2 C levels) or "blocked"
    (exact 16×16 diagonal blocks, then log2(C/16) doubling levels); ``heads_per_step``: the
    preferred value heads per grid step (0: 8); ``vmem_fraction``: the VMEM limit as a
    fraction of the chip's capacity (0: the libtpu flag's, see _vmem_limit)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision in {PRECISIONS}")
    if inverse not in INVERSE_BLOCKS:
        raise ValueError(f"gdn_impl=pallas_tpu_v2 inverse in {tuple(INVERSE_BLOCKS)}")
    batch, length, kheads, dk = k.shape
    heads, dv = v.shape[2], v.shape[-1]
    if heads % kheads:
        raise ValueError(f"{heads} value heads are not groups of {kheads} key heads")
    if chunk & (chunk - 1):
        raise ValueError("gdn_impl=pallas_tpu_v2 needs a power-of-two chunk")
    if not _interpret():
        check_tpu_shapes(dk, dv, chunk)
    if segment_ids is None:
        segment_ids = jnp.ones((batch, length), jnp.int32)
    pad = (-length) % chunk
    if pad:
        width = lambda x: ((0, 0), (0, pad)) + ((0, 0),) * (x.ndim - 2)  # noqa: E731
        q, k, v, g, beta, segment_ids = (
            jnp.pad(x, width(x)) for x in (q, k, v, g, beta, segment_ids)
        )
    if precision == "bf16":  # every dot rounds them anyway: half the DMA
        q, k = q.astype(BF16), k.astype(BF16)
    cum = inchunk_cumsum(g, chunk)
    aux, segr = token_masks(segment_ids, chunk)
    head_major = lambda x: jnp.swapaxes(x, 1, 2)  # noqa: E731  [B,T,H,*] -> [B,H,T,*]
    hb = heads_per_block(heads, heads // kheads, heads_per_step or 8)
    out = core(
        head_major(q), head_major(k), head_major(v), head_major(cum), head_major(beta),
        aux, segr, chunk, hb, precision, (INVERSE_BLOCKS[inverse], float(vmem_fraction)),
    )  # fmt: skip
    return head_major(out)[:, :length]
