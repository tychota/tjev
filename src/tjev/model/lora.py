"""Frozen projections with low-rank adapters (LoRA, rsLoRA scaling), and their factory.

Two equal forms (``compute.lora_impl``): ``unmerged`` y = xW + s·(xA)B (the adapter as two
narrow matmuls, plain autodiff); ``merged`` y = x·W' with W' = W + s·AB formed per call
in the compute dtype (one matmul at the base cost) under a custom VJP: dx = dy·W'ᵀ, and the
adapter gradients by the low-rank formulas dA = s·xᵀ(dy Bᵀ), dB = s·(xA)ᵀdy, never the full
dW = xᵀdy (which would cost a whole extra projection matmul). The merged forward rounds W'
to the compute dtype: in bf16 the small delta s·AB is quantised at W's scale (measured in
docs/kernels.md, "Merged LoRA").

Inits that start with s·A0B0 ≠ 0 (``lora.init`` pissa, lora-ga) keep the frozen kernel W
and the frozen copies A0, B0 (fp32, :class:`~tjev.model.params.Frozen`): the layer computes
y = xW + s·x(AB − A0B0), the adapter as the rank-2r product [A0 | A − A0]·[B − B0 ; B] of
fp32 differences (PEFT's PiSSA / LoRA-GA "offset"). At init the adapter is exactly zero and
the output bitwise the base model's; a residual kernel W − s·A0B0 rounded to bf16 would add
noise at the scale of s·A0B0 (≈ 3·‖W‖ for LoRA-GA, docs/training.md "LoRA initialisations").

LoRA dropout (``lora.dropout``) drops the adapter's *input*: y = xW + s·drop(x)·AB, the
base path untouched. It is active only inside :func:`dropout_scope` (the train step's
layers, one key per decoder layer, one fresh mask per projection group sharing an input:
the grouped A matmul reads one dropped x); eval, calibration and export never open one.
Merged LoRA computes x·(W + s·AB) and cannot drop the adapter's input alone: a merged
projection refuses to run inside an active scope (Qwen35 builds unmerged ones instead).
"""

from __future__ import annotations

import contextlib
import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.ad_checkpoint import checkpoint_name

from tjev.config import LoRASpec
from tjev.model.params import Frozen, Probe

# the active dropout scope (trace time): [key, rate, masks drawn so far]
_DROPOUT: list[list] = []


@contextlib.contextmanager
def dropout_scope(key: jax.Array | None, rate: float):
    """Adapter-input dropout at ``rate`` for the projections called inside (nothing when
    ``key`` is None or ``rate`` is 0). Each call draws its mask from fold_in(key, n), n
    counting the calls in trace order: a remat recompute replays the same masks."""
    if key is None or not rate:
        yield
        return
    _DROPOUT.append([key, float(rate), 0])
    try:
        yield
    finally:
        _DROPOUT.pop()


def dropout_active() -> bool:
    return bool(_DROPOUT)


def drop(x: jax.Array) -> jax.Array:
    """x with inverted dropout under the active scope (x itself outside one)."""
    if not _DROPOUT:
        return x
    scope = _DROPOUT[-1]
    key, rate, n = scope
    scope[2] = n + 1
    keep = jax.random.bernoulli(jax.random.fold_in(key, n), 1.0 - rate, x.shape)
    return jnp.where(keep, x / jnp.asarray(1.0 - rate, x.dtype), 0).astype(x.dtype)


class Linear(nnx.Module):
    """y = x W (+ s · x A B). W is [..., in, out] (NNX layout), frozen; A and B train.

    B starts at zero, so an adapted model is initially identical to the base. The adapters
    are fp32 and cast to the compute dtype for the matmuls.
    """

    # EVA's activation probe (tjev.model.lora_init.eva), set only while it runs
    probe_q: Probe
    probe_acc: Probe
    probe_sum: Probe  # Σx (centring) and Σ‖x‖², row count (EVA's gain statistic)
    probe_sq: Probe
    probe_n: Probe
    # the init adapters of an offset init (tjev.model.lora_init: pissa, lora-ga), fp32
    lora_a0: Frozen
    lora_b0: Frozen

    def __init__(
        self,
        kernel: np.ndarray | jax.Array,
        *,
        dtype: jnp.dtype,
        rank: int = 0,
        scale: float = 1.0,
        rngs: nnx.Rngs | None = None,
        merged: bool = False,
        full: bool = False,
        label: str = "proj",
        flat_rows: bool = False,
        out_in: bool = False,
    ):
        # flat_rows (compute.flat_rows): the matmuls on [B·T, D] rows, reshaped back after.
        # out_in (compute.kernel_out_in): the frozen kernel stored [..., out, in] (HF's
        # orientation), the matmuls reading its transpose (weight()): the layout XLA:TPU may
        # otherwise copy a stacked kernel into once per step
        self.flat_rows = flat_rows
        # full: the kernel itself trains (fp32, cast to the compute dtype per call): a LoRA
        # parameter, so the optimizer, checkpoints and the train / frozen split take it
        self.full = full
        self.out_in = out_in and not full
        # the projection's name (q_proj, in_proj_qkv, up_proj, …): its ops' named scope, so
        # xprof's op names read …/mlp/up_proj/base, …/mlp/up_proj/lora/B (metadata only)
        self.label = label
        self.kernel = (nnx.LoRAParam(jnp.asarray(kernel, jnp.float32)) if full
                       else Frozen(jnp.asarray(jnp.swapaxes(jnp.asarray(kernel), -1, -2)
                                               if self.out_in else kernel, dtype)))  # fmt: skip
        self.dtype = dtype
        self.rank = rank
        self.scale = scale
        self.merged = merged
        if rank:
            if rngs is None:
                raise ValueError("LoRA needs rngs")
            *lead, fan_in, fan_out = kernel.shape
            a = jax.random.normal(rngs.params(), (*lead, fan_in, rank), jnp.float32)
            self.lora_a = nnx.LoRAParam(a / np.sqrt(fan_in))
            self.lora_b = nnx.LoRAParam(jnp.zeros((*lead, rank, fan_out), jnp.float32))

    def __call__(self, x: jax.Array, name: str | None = None) -> jax.Array:
        """``name``: a checkpoint_name for the output (remat policies may save it)."""
        x = x.astype(self.dtype)
        lead = x.shape[:-1]
        if self.flat_rows:
            x = x.reshape(-1, x.shape[-1])
        observe(self, x)
        if self.rank and self.merged:
            (y,) = _merged_group(x, self)
        else:
            y = self.base(x)
            if self.rank:
                with jax.named_scope(self.label), jax.named_scope("lora"):
                    a, b = self.factors(self.dtype)
                    with jax.named_scope("A"):
                        xa = checkpoint_name(drop(x) @ a, "lora_xa")
                    with jax.named_scope("B"):
                        y = y + (xa @ b) * jnp.asarray(self.scale, self.dtype)
        y = y.reshape(*lead, y.shape[-1])
        return checkpoint_name(y, name) if name else y

    @property
    def offset(self) -> bool:
        """True when the adapter is s·(AB − A0B0) (an offset init: pissa, lora-ga)."""
        return hasattr(self, "lora_a0")

    def factors(self, dtype) -> tuple[jax.Array, jax.Array]:
        """(A', B') in ``dtype`` with the adapter's output s·x·A'B': (A, B), or with an init
        offset ([A0 | A − A0], [B − B0 ; B]), whose product is AB − A0B0 as two products of
        small fp32 differences, never the difference of two large products."""
        a, b = self.lora_a[...], self.lora_b[...]
        if not self.offset:
            return a.astype(dtype), b.astype(dtype)
        f32 = jnp.float32
        a0, b0 = self.lora_a0[...].astype(f32), self.lora_b0[...].astype(f32)
        da, db = a.astype(f32) - a0, b.astype(f32) - b0
        return (
            jnp.concatenate([a0, da], axis=-1).astype(dtype),
            jnp.concatenate([db, b.astype(f32)], axis=-2).astype(dtype),
        )

    def weight(self) -> jax.Array:
        """This layer's kernel W [in, out] (a transposed view of an ``out_in`` one)."""
        w = self.kernel[...]
        return jnp.swapaxes(w, -1, -2) if getattr(self, "out_in", False) else w

    def base(self, x: jax.Array) -> jax.Array:
        """The frozen projection alone (named scope ``<label>/base``: xprof separates it from
        the adapters, ``<label>/lora/{A,B}``)."""
        with jax.named_scope(self.label), jax.named_scope("base"):
            return x.astype(self.dtype) @ self.weight().astype(self.dtype)

    def delta(self) -> jax.Array:
        """s·AB (s·(AB − A0B0) with an init offset) in fp32: what export adds to the
        kernel (the original HF weight)."""
        a, b = self.factors(jnp.float32)
        return self.scale * jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def grouped(x: jax.Array, *linears: Linear, names: tuple[str | None, ...] = ()) -> list[jax.Array]:
    """``[l(x) for l in linears]`` for projections that share their input.

    The base matmuls stay separate (each already runs at the MXU roofline); the LoRA ``A``
    matrices are concatenated so x is read once by one wider low-rank matmul (r·n columns
    instead of n narrow r-column ones), in the forward and the backward. Parameters are
    unchanged: one ``lora_a`` / ``lora_b`` per projection. ``names``: optional
    checkpoint_names for the outputs (remat policies may save them).
    """
    dtype = linears[0].dtype
    x = x.astype(dtype)
    lead = x.shape[:-1]
    if getattr(linears[0], "flat_rows", False):
        x = x.reshape(-1, x.shape[-1])
    for layer in linears:
        observe(layer, x)
    ys: list = [None] * len(linears)
    adapted = [i for i, layer in enumerate(linears) if layer.rank]
    merged = [i for i in adapted if linears[i].merged]
    if merged:  # x·(W + s·AB) for each, the adapter gradients shared (_merged_group)
        for i, y in zip(merged, _merged_group(x, *(linears[i] for i in merged)), strict=True):
            ys[i] = y
    adapted = [i for i in adapted if ys[i] is None]
    ys = [layer.base(x) if y is None else y for layer, y in zip(linears, ys, strict=True)]
    if adapted:
        for i, y in zip(adapted, _adapters(x, [linears[i] for i in adapted], [ys[i] for i in adapted]),
                        strict=True):  # fmt: skip
            ys[i] = y
    ys = [y.reshape(*lead, y.shape[-1]) for y in ys]
    return [
        checkpoint_name(y, n) if n else y
        for y, n in zip(ys, names or (None,) * len(ys), strict=True)
    ]


def _adapters(x: jax.Array, linears: list[Linear], ys: list[jax.Array]) -> list[jax.Array]:
    """y_i + s_i·(drop(x)·A_i)B_i, the A_i concatenated: one pass over x (unmerged LoRA).
    The shared x·[A_1 | A_2 | …] is named after the group (gate_proj+up_proj/lora/A), each
    B after its projection (up_proj/lora/B)."""
    dtype = x.dtype
    group = "+".join(layer.label for layer in linears)
    with jax.named_scope(group), jax.named_scope("lora"):
        factors = [layer.factors(dtype) for layer in linears]
        with jax.named_scope("A"):
            a = jnp.concatenate([fa for fa, _ in factors], axis=-1)
            xa = checkpoint_name(drop(x) @ a, "lora_xa")
        splits = np.cumsum([fa.shape[-1] for fa, _ in factors])[:-1].tolist()
        parts = jnp.split(xa, splits, axis=-1)
    out = []
    for layer, y, hi, (_, b) in zip(linears, ys, parts, factors, strict=True):
        with jax.named_scope(layer.label), jax.named_scope("lora"), jax.named_scope("B"):
            out.append(y + (hi @ b) * jnp.asarray(layer.scale, dtype))
    return out


def observe(layer: Linear, x: jax.Array) -> None:
    """EVA's activation probe (tjev.model.lora_init.eva): when ``layer`` carries one, add
    XᵀX·Q of this call's input rows to it (fp32), one subspace-iteration product, and the
    rows' Σx, Σ‖x‖² and count."""
    acc = getattr(layer, "probe_acc", None)
    if acc is None:
        return
    rows = x.reshape(-1, x.shape[-1]).astype(jnp.float32)
    hi = jax.lax.Precision.HIGHEST
    xq = jnp.matmul(rows, layer.probe_q[...], precision=hi)
    acc[...] = acc[...] + jnp.matmul(rows.T, xq, precision=hi)
    layer.probe_sum[...] = layer.probe_sum[...] + rows.sum(0)
    layer.probe_sq[...] = layer.probe_sq[...] + jnp.sum(rows * rows)
    layer.probe_n[...] = layer.probe_n[...] + rows.shape[0]


def _merged_group(x: jax.Array, *linears: Linear) -> list[jax.Array]:
    """[x·W'_i] for adapted projections sharing x, W'_i = W_i + s_i·A_i B_i (merged LoRA)."""
    if dropout_active():
        raise ValueError("merged LoRA cannot drop the adapter input: build unmerged layers")
    dtype = x.dtype
    kernels = tuple(layer.weight() for layer in linears)
    a = tuple(layer.lora_a[...] for layer in linears)
    b = tuple(layer.lora_b[...] for layer in linears)
    # an offset init's frozen A0, B0 (None without): W' = W + s·(AB − A0B0)
    init = tuple((layer.lora_a0[...], layer.lora_b0[...]) if layer.offset else None
                 for layer in linears)  # fmt: skip
    scales = tuple(float(layer.scale) for layer in linears)
    labels = tuple(layer.label for layer in linears)
    # one matmul per projection at the base cost (<label>/base), the merge W + s·AB
    # (<label>/lora/merge), the adapter gradients (<group>/lora/…)
    return list(_merged_matmuls(x, kernels, a, b, init, scales, jnp.dtype(dtype).name, labels))


def _value(x):
    return getattr(x, "value", x)  # a symbolic-zeros CustomVJPPrimal, or the array itself


def _merge(w, a, b, init, scale: float, dtype, label: str = "proj"):
    """W' = W + s·AB rounded once to the compute dtype (A, B rounded to it first, as the
    unmerged path's matmuls do; the delta accumulated in fp32). With an init offset
    (A0, B0) the delta is AB − A0B0 = A0(B − B0) + (A − A0)B, from fp32 differences."""
    with jax.named_scope(label), jax.named_scope("lora"), jax.named_scope("merge"):
        if init is not None:
            f32 = jnp.float32
            a0, b0 = (_value(t).astype(f32) for t in init)
            a = jnp.concatenate([a0, a.astype(f32) - a0], axis=-1)
            b = jnp.concatenate([b.astype(f32) - b0, b.astype(f32)], axis=-2)
        delta = jnp.matmul(a.astype(dtype), b.astype(dtype), preferred_element_type=jnp.float32)
        return (w.astype(jnp.float32) + scale * delta).astype(dtype)


def _base_matmul(x, w, label: str):
    with jax.named_scope(label), jax.named_scope("base"):
        return x @ w


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def _merged_matmuls(x, kernels, a, b, init, scales, dtype, labels):
    return tuple(
        _base_matmul(x, _merge(w, ai, bi, oi, s, dtype, n), n)
        for w, ai, bi, oi, s, n in zip(kernels, a, b, init, scales, labels, strict=True)
    )


def _merged_fwd(x, kernels, a, b, init, scales, dtype, labels):
    merged = tuple(
        _merge(w.value, ai.value, bi.value, oi, s, dtype, n)
        for w, ai, bi, oi, s, n in zip(kernels, a, b, init, scales, labels, strict=True)
    )
    ys = tuple(_base_matmul(x.value, w, n) for w, n in zip(merged, labels, strict=True))
    # dW only when the frozen base is differentiated (the ground truth; never training)
    wants_dw = tuple(w.perturbed for w in kernels)
    return ys, (x.value, merged, tuple(ai.value for ai in a), tuple(bi.value for bi in b),
                tuple(() if p else None for p in wants_dw))  # fmt: skip


def _merged_bwd(scales, dtype, labels, res, dys):
    x, merged, a, b, wants_dw = res
    f32 = jnp.float32
    dys = tuple(
        jnp.zeros((*x.shape[:-1], w.shape[-1]), x.dtype)
        if isinstance(dy, jax.custom_derivatives.SymbolicZero) else dy.astype(x.dtype)
        for dy, w in zip(dys, merged, strict=True)
    )  # fmt: skip
    lead = tuple(range(x.ndim - 1))
    parts = []
    for dy, w, n in zip(dys, merged, labels, strict=True):
        with jax.named_scope(n), jax.named_scope("base"):
            parts.append(jnp.matmul(dy, w.T, preferred_element_type=f32))
    with jax.named_scope("+".join(labels)), jax.named_scope("base"):
        dx = functools.reduce(jnp.add, parts)
    with jax.named_scope("+".join(labels)), jax.named_scope("lora"):
        # one pass over x for every adapter: x·[A_1 | A_2 | …] and xᵀ·[s·t_1 | s·t_2 | …]
        a_cat = jnp.concatenate([ai.astype(dtype) for ai in a], axis=-1)
        xa = jnp.matmul(x, a_cat, preferred_element_type=f32).astype(dtype)
        t = [
            (s * jnp.matmul(dy, bi.astype(dtype).T, preferred_element_type=f32)).astype(dtype)
            for dy, bi, s in zip(dys, b, scales, strict=True)
        ]
        da_cat = jnp.tensordot(x, jnp.concatenate(t, axis=-1), axes=(lead, lead),
                               preferred_element_type=f32)  # fmt: skip
        splits = np.cumsum([ai.shape[-1] for ai in a])[:-1].tolist()
        da = jnp.split(da_cat, splits, axis=-1)
        xas = jnp.split(xa, splits, axis=-1)
        db = [
            s * jnp.tensordot(xai, dy, axes=(lead, lead), preferred_element_type=f32)
            for xai, dy, s in zip(xas, dys, scales, strict=True)
        ]
    dkernels = tuple(
        None if flag is None
        else jnp.tensordot(x, dy, axes=(lead, lead), preferred_element_type=f32).astype(w.dtype)
        for flag, dy, w in zip(wants_dw, dys, merged, strict=True)
    )  # fmt: skip
    return (
        dx.astype(x.dtype),
        dkernels,
        tuple(d.astype(ai.dtype) for d, ai in zip(da, a, strict=True)),
        tuple(d.astype(bi.dtype) for d, bi in zip(db, b, strict=True)),
        None,  # the frozen A0, B0: no cotangent
    )


_merged_matmuls.defvjp(_merged_fwd, _merged_bwd, symbolic_zeros=True)


class LinearFactory:
    """Creates (possibly LoRA-adapted) projections from stacked HF-orientation weights.
    ``merged``: adapted projections compute x·(W + s·AB) (``compute.lora_impl=merged``)."""

    def __init__(
        self, lora: LoRASpec | None, dtype: jnp.dtype, rngs: nnx.Rngs, merged: bool = False,
        *, flat_rows: bool = False, out_in: tuple[str, ...] = (),
    ):  # fmt: skip
        self.lora, self.dtype, self.rngs, self.merged = lora, dtype, rngs, merged
        self.flat_rows, self.out_in = flat_rows, out_in

    def __call__(self, name: str, weight: jax.Array) -> Linear:
        # HF stores [out, in]; NNX layout is [in, out]. Leading (stack) axes are kept.
        kernel = jnp.swapaxes(jnp.asarray(weight), -1, -2)
        lora = self.lora
        rank = lora.rank_of(name) if lora is not None and lora.rank > 0 else 0
        return Linear(
            kernel,
            dtype=self.dtype,
            rank=rank,
            scale=lora.scale_of(name) if lora is not None and rank else 1.0,
            rngs=self.rngs,
            merged=self.merged,
            full=lora is not None and lora.full(name),
            label=name,
            flat_rows=self.flat_rows,
            out_in=name in self.out_in,
        )
