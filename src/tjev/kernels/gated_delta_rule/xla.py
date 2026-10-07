"""Gated DeltaNet (GDN) delta rule: segment-aware chunked (WY/UT) form and a recurrence.

Math follows HF ``torch_chunk_gated_delta_rule`` / MaxText ``jax_chunk_gated_delta_rule``.
Per head, with state S ∈ R^{Dk×Dv} and per-token decay g ≤ 0 and step size β:

    S_t = exp(g_t) S_{t-1};   S_t += k_t ⊗ β_t (v_t − S_tᵀ k_t);   o_t = S_tᵀ q_t

Packed sequences carry ``segment_ids``: the state restarts at every segment boundary,
exactly as if each segment were run alone (MaxText PR #5351). Everything here is fp32;
q and k arrive L2-normalised and q already scaled by Dk^-1/2.

Shapes: q,k [B,T,H,Dk]; v [B,T,H,Dv]; g,beta [B,T,H]; segment_ids [B,T] int.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

_HI = jax.lax.Precision.HIGHEST
PRECISIONS = ("highest", "high", "bf16", "mixed")
# "mixed" (MaxText PR #5154/#5386/#5399): the intra-chunk products, whose errors reach the
# gate gradients through the decay (q kᵀ, kβ kᵀ, the UT solve, T·[vβ, kbe], intra·v_new, and
# the products that form a state update or its gradient) at bf16_3x; the dots with the
# state S or its gradient G as an operand (w·S, q_read·S and, backward, k_write·G, ·Sᵀ, ·Gᵀ)
# as one bf16 pass with fp32 accumulation.
_MIXED = {"intra": "high", "state": "bf16"}


def part(precision: str, kind: str) -> str:
    """The matmul mode of an ``intra``-chunk or ``state``-path product under ``precision``."""
    if kind not in _MIXED:
        raise ValueError(f"kind in {tuple(_MIXED)}")
    return _MIXED[kind] if precision == "mixed" else precision


INVERSES = ("solve", "doubling", "blocked")


def l2norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """x / sqrt(||x||² + eps) over the last axis (the HF / FLA definition).

    An all-zero row maps to 0 with a zero gradient: through the plain formula its Jacobian
    is 1/sqrt(eps) = 1000·I, and the delta-rule state does send gradient into such rows (a
    1000× gain per layer; MaxText #5295 / #5363). The double ``where`` keeps the masked
    branch's gradient out of the result."""
    sq = jnp.sum(x * x, axis=-1, keepdims=True)
    zero = sq == 0
    safe = jnp.where(zero, jnp.ones_like(x), x)
    out = safe * jax.lax.rsqrt(jnp.where(zero, jnp.ones_like(sq), sq) + eps)
    return jnp.where(zero, jnp.zeros_like(x), out)


def mm(a: jax.Array, b: jax.Array, mode: str = "highest") -> jax.Array:
    """fp32 matmul at a chosen cost/accuracy point (see chunk_gated_delta_rule)."""
    if mode == "bf16":  # bf16 operands, fp32 accumulation: tensor cores at full bf16 rate
        return jnp.matmul(
            a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), preferred_element_type=jnp.float32
        )
    precision = {"highest": jax.lax.Precision.HIGHEST, "high": jax.lax.Precision.HIGH}[mode]
    return jnp.matmul(a, b, precision=precision)


def doubling_masks(size: int) -> list[np.ndarray]:
    """For each level s = 1, 2, 4, …: mask of the lower-left s×s block inside every 2s×2s
    diagonal block (the entries recursive doubling eliminates at that level)."""
    row = np.arange(size)[:, None]
    col = np.arange(size)[None, :]
    masks, s = [], 1
    while s < size:
        masks.append(
            (row // (2 * s) == col // (2 * s))
            & (row // s == 2 * (row // (2 * s)) + 1)
            & (col // s == 2 * (col // (2 * s)))
        )
        s *= 2
    return masks


def unit_lower_inverse(lower: jax.Array, mode: str) -> jax.Array:
    """(I + L)^-1 for strictly lower-triangular L, by matmuls only (recursive doubling).

    With T_s = inverse of the s×s diagonal blocks of (I + L) (T_1 = I), one level is
        T_2s = T_s − T_s · (L ⊙ M_s) · T_s,
    M_s selecting the lower-left s×s block of each 2s×2s block: exact block inversion
    [[A,0],[B,D]]⁻¹ = [[A⁻¹,0],[−D⁻¹BA⁻¹,D⁻¹]]. Every intermediate is a sub-block of the
    true inverse, so it stays bounded (unlike the power series (I−L)(I+L²)…, whose
    partial products overflow when keys are correlated and L ≈ β everywhere).
    2·log2(C) matmuls, no triangular solve.
    """
    size = lower.shape[-1]
    if size & (size - 1):
        raise ValueError("chunk must be a power of two for the doubling inverse")
    inverse = jnp.broadcast_to(jnp.eye(size, dtype=lower.dtype), lower.shape)
    for mask in doubling_masks(size):
        off = jnp.where(mask, lower, 0.0)
        inverse = inverse - mm(mm(inverse, off, mode), inverse, mode)
    return inverse


def blocked_solve(lower: jax.Array, rhs: jax.Array, mode: str, block: int = 16) -> jax.Array:
    """(I + L)⁻¹ rhs by blocked forward substitution (L strictly lower, [..., C, C]).

    The ``block``×``block`` diagonal blocks are inverted by doubling (2·log2(block) tiny
    matmuls, all blocks batched), then the block rows are solved in order:
        X_i = (I + L_ii)⁻¹ (rhs_i − L_i,<i X_<i)
    C/block sequential steps of one [block, i·block] × [i·block, W] matmul each: far fewer
    FLOPs and [C,C] HBM round trips than the dense doubling (MaxText / tpu-inference
    ``invert_triangular_matrix``, FLA ``solve_tril``, applied to the right-hand side)."""
    size = lower.shape[-1]
    block = min(block, size)
    if size % block:
        raise ValueError(f"chunk {size} is not a multiple of the block {block}")
    starts = range(0, size, block)
    diagonal = jnp.stack([lower[..., i : i + block, i : i + block] for i in starts], axis=-3)
    diagonal_inverse = unit_lower_inverse(diagonal, mode)  # [..., C/block, block, block]
    rows: list[jax.Array] = []
    for n, i in enumerate(starts):
        part = rhs[..., i : i + block, :]
        if rows:
            part = part - mm(lower[..., i : i + block, :i], jnp.concatenate(rows, axis=-2), mode)
        rows.append(mm(diagonal_inverse[..., n, :, :], part, mode))
    return jnp.concatenate(rows, axis=-2)


def solve_unit_lower(lower: jax.Array, rhs: jax.Array, inverse: str, mode: str) -> jax.Array:
    """(I + L)⁻¹ rhs for strictly lower-triangular L: ``solve`` (XLA's triangular solve),
    ``doubling`` (the dense inverse by matmuls, as in the fused kernel) or ``blocked``."""
    if inverse == "solve":
        eye = jnp.eye(lower.shape[-1], dtype=lower.dtype)
        return solve_triangular(lower + eye, rhs, lower=True, unit_diagonal=True)
    if inverse == "doubling":
        return mm(unit_lower_inverse(lower, mode), rhs, mode)
    if inverse == "blocked":
        return blocked_solve(lower, rhs, mode)
    raise ValueError(f"inverse in {INVERSES}, not {inverse!r}")


def recurrent_gated_delta_rule(q, k, v, g, beta, segment_ids=None, initial_state=None):
    """Token-by-token reference (and T=1 decode step). Returns (out, final_state)."""
    batch, length, heads, dk = k.shape
    dv = v.shape[-1]
    q, k, v, g, beta = (x.astype(jnp.float32) for x in (q, k, v, g, beta))
    if segment_ids is None:
        segment_ids = jnp.ones((batch, length), jnp.int32)
    previous = jnp.concatenate([segment_ids[:, :1], segment_ids[:, :-1]], axis=1)
    restart = (segment_ids != previous).at[:, 0].set(initial_state is None)
    state = (
        jnp.zeros((batch, heads, dk, dv), jnp.float32)
        if initial_state is None
        else initial_state.astype(jnp.float32)
    )

    def step(s, xs):
        q_t, k_t, v_t, g_t, b_t, r_t = xs
        s = jnp.where(r_t[:, None, None, None], 0.0, s)
        s = s * jnp.exp(g_t)[..., None, None]
        memory = jnp.einsum("bhkv,bhk->bhv", s, k_t, precision=_HI)
        delta = (v_t - memory) * b_t[..., None]
        s = s + k_t[..., :, None] * delta[..., None, :]
        return s, jnp.einsum("bhkv,bhk->bhv", s, q_t, precision=_HI)

    xs = tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, g, beta, restart))
    state, out = jax.lax.scan(step, state, xs)
    return jnp.moveaxis(out, 0, 1), state


def chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    segment_ids=None,
    initial_state=None,
    chunk=64,
    precision: str = "highest",
    inverse: str = "solve",
):
    """Chunked WY form: O(T·C) intra-chunk work plus a scan over T/C chunks.

    ``precision`` sets the matmuls (``highest`` = true fp32, the parity reference;
    ``high`` = TF32 on GPU / bf16_3x on TPU; ``bf16`` = bf16 operands with fp32
    accumulation). ``inverse`` picks the UT transform: ``solve`` (triangular solve, the
    reference), ``doubling`` (matmuls only: what the Pallas TPU kernels use) or ``blocked``
    (blocked forward substitution, :func:`blocked_solve`).
    Gates, exponentials and the carried state are fp32 in every mode.
    """
    if precision not in PRECISIONS or inverse not in INVERSES:
        raise ValueError(f"precision in {PRECISIONS}, inverse in {INVERSES}")
    batch, length, heads, dk = k.shape
    dv = v.shape[-1]
    q, k, v, g, beta = (x.astype(jnp.float32) for x in (q, k, v, g, beta))
    if segment_ids is None:
        segment_ids = jnp.ones((batch, length), jnp.int32)
    pad = (-length) % chunk
    if pad:
        q, k, v = (jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0))) for x in (q, k, v))
        g, beta = (jnp.pad(x, ((0, 0), (0, pad), (0, 0))) for x in (g, beta))
        # Zero q/k/v/β and g=0 make pad tokens exact no-ops on the state; keeping the
        # last segment id means the final state still belongs to the last real segment.
        segment_ids = jnp.pad(segment_ids, ((0, 0), (0, pad)), mode="edge")
    n = (length + pad) // chunk

    def chunked(x):  # [B,T,H,...] -> [B,H,N,C,...]
        x = x.reshape(batch, n, chunk, heads, *x.shape[3:])
        return jnp.moveaxis(x, 3, 1)

    q, k, v, g, beta = (chunked(x) for x in (q, k, v, g, beta))
    seg = segment_ids.reshape(batch, n, chunk)
    # Segment of the token just before each chunk (chunk 0: "before" = initial state).
    before = jnp.concatenate(
        [
            (seg[:, :1, 0] if initial_state is not None else jnp.full((batch, 1), -2, seg.dtype)),
            seg[:, :-1, -1],
        ],
        axis=1,
    )
    continues = (seg == before[..., None])[:, None]  # [B,1,N,C] reads the incoming state
    in_last = (seg == seg[..., -1:])[:, None]  # [B,1,N,C] writes the outgoing state
    carry_keep = continues[..., -1]  # [B,1,N]: incoming state survives the chunk
    causal = jnp.tril(jnp.ones((chunk, chunk), bool))
    same = (seg[..., :, None] == seg[..., None, :]) & causal  # [B,N,C,C]
    same = same[:, None]

    cum = jnp.cumsum(g, axis=-1)  # [B,H,N,C] log-decay since chunk start
    decay = jnp.exp(jnp.where(same, cum[..., :, None] - cum[..., None, :], -jnp.inf))
    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]
    intra_mode, state_mode = part(precision, "intra"), part(precision, "state")
    lower = mm(k_beta, jnp.swapaxes(k, -1, -2), intra_mode) * decay
    lower = jnp.where(jnp.tril(jnp.ones((chunk, chunk), bool), -1), lower, 0.0)
    # (I + L) [u, w] = [v β, k β e^cum]: the UT transform of the in-chunk delta updates.
    rhs = jnp.concatenate([v_beta, k_beta * (jnp.exp(cum) * continues)[..., None]], axis=-1)
    with jax.named_scope("gdn_inverse"):
        solved = solve_unit_lower(lower, rhs, inverse, intra_mode)
    u, w = solved[..., :dv], solved[..., dv:]
    intra = mm(q, jnp.swapaxes(k, -1, -2), intra_mode) * decay
    q_read = q * (jnp.exp(cum) * continues)[..., None]
    k_write = k * (jnp.exp(cum[..., -1:] - cum) * in_last)[..., None]
    carry = jnp.exp(cum[..., -1]) * carry_keep  # [B,H,N]

    state = (
        jnp.zeros((batch, heads, dk, dv), jnp.float32)
        if initial_state is None
        else initial_state.astype(jnp.float32)
    )

    def step(s, xs):
        u_c, w_c, q_c, k_c, intra_c, carry_c = xs
        v_new = u_c - mm(w_c, s, state_mode)
        out = mm(q_c, s, state_mode) + mm(intra_c, v_new, intra_mode)
        s = s * carry_c[..., None, None] + mm(jnp.swapaxes(k_c, -1, -2), v_new, intra_mode)
        return s, out

    xs = tuple(jnp.moveaxis(x, 2, 0) for x in (u, w, q_read, k_write, intra, carry))
    with jax.named_scope("gdn_scan"):
        state, out = jax.lax.scan(step, state, xs)  # out [N,B,H,C,Dv]
    out = jnp.moveaxis(out, 0, 2).reshape(batch, heads, n * chunk, dv)
    return jnp.swapaxes(out, 1, 2)[:, :length], state
