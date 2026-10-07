"""LoRA initialisations (``lora.init``): default, PiSSA, EVA, LoRA-GA.

Every adapted :class:`~tjev.model.lora.Linear` computes y = x·W + s·x·A·B (NNX layout:
W [..., in, out], A [..., in, r], B [..., r, out], leading axes stacked over super-blocks).

* ``default``: A ~ N(0, 1/in), B = 0 (built by :class:`~tjev.model.lora.Linear` itself).
* ``pissa`` (Meng et al. 2024, arXiv 2404.02948): W ≈ U S Vᵀ (top r, randomized "fast
  SVD"), A = U_r·√S_r/√s, B = √S_r·V_rᵀ/√s, so s·AB is W's principal part, and the model
  is W + s·(AB − A0B0) (the init offset below). Weights only.
* ``eva`` (Paischer et al. 2024, arXiv 2410.07170): A = the top-r principal directions
  of each projection's input activations X over a few microbatches (orthonormal columns
  here, rows in the HF / PEFT layout), B = 0: the model is unchanged at init. Fixed rank
  (EVA's rank redistribution is off). Centred, as the authors' PEFT implementation
  (IncrementalPCA; no whitening, its default); padding rows included (GDN inputs are
  zeroed at padding; packed rows carry little). The init prints each projection's gain,
  rms(x·A) / rms(x·A_default) = √(mean_j a_jᵀXᵀXa_j / (tr XᵀX / in)): how much more
  activation energy EVA's A collects than the default's random one (an Adam step on B
  moves the output in proportion; docs/training.md "LoRA initialisations").
* ``lora-ga`` (Wang et al. 2024, arXiv 2407.05000): G = ∂L/∂W over a few microbatches,
  G ≈ P Σ Rᵀ (P in-dim, R out-dim singular vectors), A = c·P[:, :r], B = c·R[:, r:2r]ᵀ with
  the paper's stable scale c = d_out^(1/4)/√γ (as the reference code's ``scale: stable``,
  direction ``ArB2r``), and the model W + s·(AB − A0B0).

All SVDs are randomized subspace iterations (Halko et al. 2011) in fp32 at HIGHEST matmul
precision, on the accelerator. Data-driven inits never materialise X or G:

* EVA: each probe pass adds XᵀX·Q to an [in, k] sketch inside the forward (one more
  [in, k] fp32 pair per adapted projection); ``niter + 1`` forward passes.
* LoRA-GA: the low-rank trick. With a rank-k adapter, A = 0 and B = Yᵀ leave the output at
  the base model's and dA = s·G·Y; A = Q and B = 0 give dB = s·Qᵀ·G. So G·Y and Qᵀ·G are
  ordinary LoRA gradients (at rank k = 4r, as the paper's code), and the randomized SVD
  of G costs ``2·niter + 2`` forward + backward passes of the init microbatches, with
  adapter-sized memory (no full-weight gradient).

Inits whose s·A0B0 ≠ 0 (``RESIDUAL``: pissa, lora-ga) keep the kernel W and frozen fp32
copies A0, B0; the adapter computes s·x(AB − A0B0) from the fp32 differences A − A0, B − B0
(:meth:`tjev.model.lora.Linear.factors`), so the model at init is bitwise the base. The
papers' form, a frozen residual kernel W − s·A0B0, rounds in bf16 at the scale of s·A0B0:
LoRA-GA's stable scale makes s·A0B0's singular values s·c² ≈ 28 for d_out = 6144, ~12× W's
largest, and the residual's bf16 rounding plus the cancelling bf16 adapter product put a
~1.4e-2 relative error on each projection's output (the base's own bf16 noise: 1.7e-3).
They must be reproduced exactly when a run is resumed, evaluated or exported:
:func:`save_init` writes A0, B0 next to the run's checkpoints, and :func:`restore_init`
re-applies them (a file of before the offset form, ``form`` absent, rebuilds that run's
residual kernels, bitwise as it trained).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tjev.config import LoRASpec
from tjev.model.lora import Linear
from tjev.model.params import Frozen, Probe
from tjev.model.qwen35 import Qwen35

INITS = ("default", "pissa", "eva", "lora-ga")
DATA_DRIVEN = ("eva", "lora-ga")
RESIDUAL = ("pissa", "lora-ga")  # the frozen kernel becomes W − s·A0B0
INIT_FILE = "lora_init.safetensors"
HI = jax.lax.Precision.HIGHEST

Batches = Callable[[], Iterable[Any]]  # a few microbatches (tjev.data.pack.Batch), re-iterable
Loss = Callable[[Any, Any], jax.Array]  # (model, microbatch) -> scalar training loss


def adapted_modules(model: nnx.Module) -> dict[str, Linear]:
    """Path ("blocks/mlps/0/gate_proj") → every adapted (stacked) Linear, in graph order."""
    return {
        "/".join(map(str, path)): node
        for path, node in nnx.iter_graph(model)
        if isinstance(node, Linear) and node.rank
    }


def _mm(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.matmul(a, b, precision=HI)


def _orth(y: jax.Array) -> jax.Array:
    return jnp.linalg.qr(y)[0]


def randomized_svd(m: jax.Array, k: int, niter: int, key: jax.Array):
    """Top-k (u [r, k], s [k], vt [k, c]) of a 2-D fp32 matrix by subspace iteration."""
    omega = jax.random.normal(key, (m.shape[1], k), jnp.float32)
    q = _orth(_mm(m, omega))
    for _ in range(niter):
        q = _orth(_mm(m, _orth(_mm(m.T, q))))
    ub, s, vt = jnp.linalg.svd(_mm(q.T, m), full_matrices=False)
    return _mm(q, ub), s, vt


def _pad(x: jax.Array, axis: int, size: int) -> jax.Array:
    """Zero columns up to ``size`` (a rank above min(in, out) has nothing to hold)."""
    widths = [(0, 0)] * x.ndim
    widths[axis] = (0, size - x.shape[axis])
    return jnp.pad(x, widths)


@functools.partial(jax.jit, static_argnames=("dtype",))
def residual_kernel(kernel: jax.Array, a: jax.Array, b: jax.Array, scale: float, dtype: Any):
    """W − s·A·B in fp32, stored in the model dtype (stacked layers one at a time: one
    layer's fp32 copy is live, not the stack's)."""

    def one(w: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
        delta = _mm(a.astype(jnp.float32), b.astype(jnp.float32))
        return (w.astype(jnp.float32) - scale * delta).astype(dtype)

    lead = kernel.shape[:-2]
    if not lead:
        return one(kernel, a, b)
    flat = [x.reshape(-1, *x.shape[-2:]) for x in (kernel, a, b)]
    return jax.lax.map(lambda xs: one(*xs), tuple(flat)).reshape(kernel.shape)


@functools.partial(jax.jit, static_argnames=("rank", "niter"))
def pissa_factors(kernel: jax.Array, key: jax.Array, scale: float, rank: int, niter: int):
    """PiSSA's A [..., in, r], B [..., r, out] of a (stacked) kernel: s·AB = top-r of W.
    Stacked layers one at a time (lax.map): one layer's fp32 copy is live, not the stack's."""
    fan_in, fan_out = kernel.shape[-2:]
    k = min(2 * rank, fan_in, fan_out)  # oversampled: the top r converge in a few niter
    r = min(rank, k)

    def one(w: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        u, s, vt = randomized_svd(w.astype(jnp.float32), k, niter, key)
        root = jnp.sqrt(s[:r] / scale)
        return _pad(u[:, :r] * root, 1, rank), _pad(root[:, None] * vt[:r], 0, rank)

    lead = kernel.shape[:-2]
    if not lead:
        return one(kernel, key)
    n = int(np.prod(lead))
    flat = kernel.reshape(n, fan_in, fan_out)
    a, b = jax.lax.map(lambda xs: one(*xs), (flat, jax.random.split(key, n)))
    return a.reshape(*lead, fan_in, rank), b.reshape(*lead, rank, fan_out)


def pissa(model: nnx.Module, key: jax.Array, niter: int = 4) -> None:
    """PiSSA in place: principal adapters, residual kernels (weights only, no data)."""
    for i, layer in enumerate(adapted_modules(model).values()):
        a, b = pissa_factors(
            layer.kernel[...], jax.random.fold_in(key, i), float(layer.scale), layer.rank, niter
        )
        _set(layer, a, b, offset=True)


def _set(layer: Linear, a: jax.Array | np.ndarray, b: jax.Array | np.ndarray, *,
         offset: bool, legacy_residual: bool = False) -> None:  # fmt: skip
    """A, B ← (a, b); ``offset``: frozen copies A0, B0 = (a, b), the kernel untouched (the
    model is W + s·(AB − A0B0)). ``legacy_residual``: the kernel W − s·AB in the model
    dtype instead (init files of before the offset form)."""
    a, b = jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)
    if legacy_residual:
        layer.kernel[...] = residual_kernel(
            layer.kernel[...], a, b, float(layer.scale), jnp.dtype(layer.dtype)
        )
    layer.lora_a[...] = a
    layer.lora_b[...] = b
    if offset:  # own buffers: the trained A, B are donated by the step, A0, B0 never
        layer.lora_a0 = Frozen(jnp.array(a, copy=True))
        layer.lora_b0 = Frozen(jnp.array(b, copy=True))


# ------------------------------------------------------------------------------------------
# EVA


def _probe_pass(model: nnx.Module, batches: Batches, forward: Callable) -> None:
    graphdef, probes, rest = nnx.split(model, Probe, ...)
    for batch in batches():
        probes = forward(graphdef, probes, rest, batch)
    nnx.update(model, probes)


@functools.partial(jax.jit, static_argnums=0)
def _eva_forward(graphdef, probes, rest, batch):
    model = nnx.merge(graphdef, probes, rest)
    model.hidden(batch.tokens, batch.segment_ids, batch.positions)
    return nnx.state(model, Probe)


def eva(model: Qwen35, batches: Batches, key: jax.Array, niter: int = 4) -> None:
    """EVA in place: A = top-r right singular vectors of each projection's inputs, B = 0.

    ``niter + 1`` forward passes over ``batches()`` (remat off: no backward runs)."""
    layers = adapted_modules(model)
    blocks = model.blocks
    remat, blocks.remat = blocks.remat, "none"  # a remat trace would leak the probe updates
    model.probing = True
    try:
        for i, layer in enumerate(layers.values()):
            *lead, fan_in, _ = layer.kernel.shape
            k = min(2 * layer.rank, fan_in)
            omega = jax.random.normal(jax.random.fold_in(key, i), (*lead, fan_in, k))
            layer.probe_q = Probe(_orth(omega))
            layer.probe_acc = Probe(jnp.zeros((*lead, fan_in, k), jnp.float32))
            layer.probe_sum = Probe(jnp.zeros((*lead, fan_in), jnp.float32))
            layer.probe_sq = Probe(jnp.zeros(tuple(lead), jnp.float32))
            layer.probe_n = Probe(jnp.zeros(tuple(lead), jnp.float32))
        gains = {}
        for it in range(niter + 1):
            _probe_pass(model, batches, _eva_forward)
            for path, layer in layers.items():
                stats = (layer.probe_q[...], layer.probe_acc[...], layer.probe_sum[...],
                         layer.probe_n[...])  # fmt: skip
                if it < niter:  # Q ← orth(C·Q), C = XᵀX − n·μμᵀ
                    layer.probe_q[...] = _orth(_centred(*stats))
                else:  # Rayleigh-Ritz on span(Q): QᵀCQ = V Λ Vᵀ, A = Q·V (top r)
                    a, gain = _eva_ritz(*stats, layer.probe_sq[...], layer.rank)
                    _set(layer, a, jnp.zeros_like(layer.lora_b[...]), offset=False)
                    gains[path] = np.asarray(gain)
                for name in ("probe_acc", "probe_sum", "probe_sq", "probe_n"):
                    var = getattr(layer, name)
                    var[...] = jnp.zeros_like(var[...])
        print(json.dumps({"event": "eva_gain", **_gain_summary(gains)}), flush=True)
    finally:
        for layer in layers.values():
            for name in ("probe_q", "probe_acc", "probe_sum", "probe_sq", "probe_n"):
                if hasattr(layer, name):
                    delattr(layer, name)
        blocks.remat = remat
        del model.probing  # back to the class default: the graph is as built


@jax.jit
def _centred(q: jax.Array, acc: jax.Array, total: jax.Array, n: jax.Array) -> jax.Array:
    """C·Q with C = XᵀX − (Σx)(Σx)ᵀ/n the centred scatter, from acc = XᵀX·Q."""
    proj = jnp.einsum("...i,...ik->...k", total, q, precision=HI)
    return acc - total[..., :, None] * proj[..., None, :] / jnp.maximum(n, 1.0)[..., None, None]


def _sym(m: jax.Array) -> jax.Array:
    return 0.5 * (m + jnp.swapaxes(m, -1, -2))


@functools.partial(jax.jit, static_argnames=("rank",))
def _eva_ritz(q, acc, total, n, sq, rank: int) -> tuple[jax.Array, jax.Array]:
    """(A, gain): A = Q·V, the top-r Ritz vectors of the centred scatter on span(Q); gain =
    √(mean_j a_jᵀXᵀXa_j / (tr XᵀX / in)), A's uncentred energy against a random A's."""
    qt = jnp.swapaxes(q, -1, -2)
    _, vecs = jnp.linalg.eigh(_sym(jnp.matmul(qt, _centred(q, acc, total, n), precision=HI)))
    r = min(rank, q.shape[-1])
    v = vecs[..., ::-1][..., :r]  # eigh is ascending
    a = jnp.matmul(q, v, precision=HI)
    raw = jnp.matmul(jnp.swapaxes(v, -1, -2), jnp.matmul(qt, acc, precision=HI), precision=HI)
    energy = jnp.trace(jnp.matmul(raw, v, precision=HI), axis1=-2, axis2=-1) / r
    gain = jnp.sqrt(energy / jnp.maximum(sq / q.shape[-2], 1e-30))
    return _pad(a, a.ndim - 1, rank), gain


def _gain_summary(gains: dict[str, np.ndarray]) -> dict[str, float]:
    """The median and max gain over every adapted (stacked) projection, and per kind."""
    every = np.concatenate([g.reshape(-1) for g in gains.values()])
    out = {"median": round(float(np.median(every)), 2), "max": round(float(every.max()), 2)}
    kinds: dict[str, list[np.ndarray]] = {}
    for path, g in gains.items():
        kinds.setdefault(path.rsplit("/", 1)[-1], []).append(g.reshape(-1))
    for kind, gs in kinds.items():
        out[kind] = round(float(np.median(np.concatenate(gs))), 2)
    return out


# ------------------------------------------------------------------------------------------
# LoRA-GA


def lora_ga(
    model: nnx.Module,
    batches: Batches,
    loss: Loss,
    key: jax.Array,
    niter: int = 4,
    gamma: float = 16.0,
) -> None:
    """LoRA-GA in place, by the low-rank trick (module docstring): ``2·niter + 2`` gradient
    passes over ``batches()`` at adapter rank 4r, never the full-weight gradient. The
    sketches G·Y and Qᵀ·G carry the compute dtype's rounding (the model's backward); the
    subspace algebra is fp32."""
    layers = list(adapted_modules(model).values())
    ranks = [layer.rank for layer in layers]
    ks = [min(4 * r, *layer.kernel.shape[-2:]) for r, layer in zip(ranks, layers, strict=True)]
    for layer, k in zip(layers, ks, strict=True):  # temporary rank-k adapters
        *lead, fan_in, fan_out = layer.kernel.shape
        layer.rank = k
        layer.lora_a = nnx.LoRAParam(jnp.zeros((*lead, fan_in, k), jnp.float32))
        layer.lora_b = nnx.LoRAParam(jnp.zeros((*lead, k, fan_out), jnp.float32))
    graphdef, _, rest = nnx.split(model, nnx.LoRAParam, ...)
    # the frozen base is an argument, never a closure constant of the jitted gradient
    grad = jax.jit(jax.grad(lambda lora, rest, batch: loss(nnx.merge(graphdef, lora, rest), batch)))

    def sketch(values: list[tuple[jax.Array, jax.Array]]) -> list[tuple[jax.Array, jax.Array]]:
        for layer, (a, b) in zip(layers, values, strict=True):
            layer.lora_a[...], layer.lora_b[...] = a, b
        lora = nnx.state(model, nnx.LoRAParam)
        total = None
        for batch in batches():
            g = grad(lora, rest, batch)
            total = g if total is None else jax.tree.map(jnp.add, total, g)
        if total is None:
            raise ValueError("lora-ga needs at least one init batch")
        nnx.update(model, total)  # read the gradients back through the layers
        return [(layer.lora_a[...] / layer.scale, layer.lora_b[...] / layer.scale)
                for layer in layers]  # fmt: skip

    def right(ys: list[jax.Array]) -> list[jax.Array]:  # G·Y: A = 0, B = Yᵀ → dA = s·G·Y
        vals = [(jnp.zeros_like(layer.lora_a[...]), jnp.swapaxes(y, -1, -2))
                for layer, y in zip(layers, ys, strict=True)]  # fmt: skip
        return [da for da, _ in sketch(vals)]

    def left(qs: list[jax.Array]) -> list[jax.Array]:  # Qᵀ·G: A = Q, B = 0 → dB = s·QᵀG
        vals = [(q, jnp.zeros_like(layer.lora_b[...])) for layer, q in zip(layers, qs, strict=True)]
        return [db for _, db in sketch(vals)]

    omegas = [
        jax.random.normal(jax.random.fold_in(key, i), (*layer.kernel.shape[:-2], layer.kernel.shape[-1], k))
        for i, (layer, k) in enumerate(zip(layers, ks, strict=True))
    ]  # fmt: skip
    qs = [_orth(y) for y in right(omegas)]
    for _ in range(niter):
        zs = [_orth(jnp.swapaxes(z, -1, -2)) for z in left(qs)]
        qs = [_orth(y) for y in right(zs)]
    projected = left(qs)  # Qᵀ·G [..., k, out]
    for layer, r, q, bm in zip(layers, ranks, qs, projected, strict=True):
        a, b = _lora_ga_factors(q, bm, r, gamma)
        layer.rank = r
        layer.lora_a = nnx.LoRAParam(a)
        layer.lora_b = nnx.LoRAParam(b)
        _set(layer, a, b, offset=True)


@functools.partial(jax.jit, static_argnames=("rank",))
def _lora_ga_factors(q: jax.Array, bm: jax.Array, rank: int, gamma: float):
    """A = c·P[:, :r], B = c·R[:, r:2r]ᵀ from G ≈ (Q·Ub) S Vt, c = d_out^(1/4)/√γ."""
    ub, _, vt = jnp.linalg.svd(bm, full_matrices=False)
    p = jnp.matmul(q, ub, precision=HI)  # in-dim singular vectors [..., in, k]
    k, fan_out = vt.shape[-2:]
    c = fan_out**0.25 / gamma**0.5
    a = _pad(c * p[..., :, : min(rank, k)], p.ndim - 1, rank)
    b = c * vt[..., rank : min(2 * rank, k), :]  # rows r..2r of Vt: R[:, r:2r]ᵀ
    return a, _pad(b, b.ndim - 2, rank)


# ------------------------------------------------------------------------------------------
# Entry point, persistence


def initialize(
    model: Qwen35,
    spec: LoRASpec,
    *,
    seed: int = 0,
    batches: Batches | None = None,
    loss: Loss | None = None,
) -> bool:
    """Apply ``spec.init`` to a freshly built model in place; True if it changed anything.
    ``batches`` (eva, lora-ga) and ``loss`` (lora-ga) come from the training stream."""
    if spec.init not in INITS:
        raise ValueError(f"lora.init must be one of {INITS}")
    if spec.init == "default" or not adapted_modules(model):
        return False
    key = jax.random.fold_in(jax.random.key(seed), 0x1A1)
    if spec.init == "pissa":
        pissa(model, key, spec.init_niter)
        return True
    if batches is None:
        raise ValueError(f"lora.init={spec.init} needs init batches")
    if spec.init == "eva":
        eva(model, batches, key, spec.init_niter)
    else:
        if loss is None:
            raise ValueError("lora.init=lora-ga needs the training loss")
        lora_ga(model, batches, loss, key, spec.init_niter, spec.init_gamma)
    return True


def init_adapters(model: nnx.Module) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The current (A, B) of every adapted Linear, by path (host fp32 copies)."""
    return {
        path: (np.asarray(jax.device_get(layer.lora_a[...])),
               np.asarray(jax.device_get(layer.lora_b[...])))
        for path, layer in adapted_modules(model).items()
    }  # fmt: skip


def save_init(path: str | Path, model: nnx.Module, init: str) -> None:
    """A0, B0 of every adapted Linear (written right after :func:`initialize`)."""
    from safetensors.numpy import save_file

    tensors = {}
    for name, (a, b) in init_adapters(model).items():
        tensors[name + "/lora_a"], tensors[name + "/lora_b"] = a, b
    residual = init in RESIDUAL
    meta = {"init": init, "residual": str(residual)} | ({"form": "offset"} if residual else {})
    save_file(tensors, str(path), metadata=meta)


def load_init(path: str | Path) -> tuple[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    init, saved, _ = _read_init(path)
    return init, saved


def _read_init(path: str | Path):
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata() or {}
        init = meta.get("init", "default")
        form = meta.get("form", "residual kernel")
        names = sorted({k.rsplit("/", 1)[0] for k in f.keys()})  # noqa: SIM118
        saved = {n: (f.get_tensor(n + "/lora_a"), f.get_tensor(n + "/lora_b")) for n in names}
        return init, saved, form


def restore_init(model: nnx.Module, path: str | Path) -> str:
    """Re-apply a saved init: A0, B0 (and, residual inits, the frozen offset A0, B0, or a
    legacy file's kernels W − s·A0B0), bitwise as in training. Returns the init's name."""
    init, saved, form = _read_init(path)
    layers = adapted_modules(model)
    if set(saved) != set(layers):
        raise ValueError(f"{path}: adapted projections differ from the model's")
    for name, layer in layers.items():
        a, b = saved[name]
        if a.shape != layer.lora_a.shape or b.shape != layer.lora_b.shape:
            raise ValueError(f"{path}: {name} has rank {a.shape[-1]}, not {layer.rank}")
        residual = init in RESIDUAL
        _set(layer, a, b, offset=residual and form == "offset",
             legacy_residual=residual and form != "offset")  # fmt: skip
    return init
