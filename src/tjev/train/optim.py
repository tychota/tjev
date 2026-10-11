"""Optimizers and the learning-rate schedule for adapter training.

AdamW is the default (Qwen3.5 was pretrained with AdamW, and Muon's measured edge on the
LoRA factors was within seed noise: docs/research/muon.md). Muon is available for the LoRA
factors as an option: stacked [S, in, out] factors are orthogonalised per layer (leading
axes are batch axes) with the Polar Express Newton-Schulz polynomials, at the update RMS
of AdamW, so learning rates carry over; everything that is not a matrix uses Adam.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.typing import ArrayLike

from tjev.config import LossSpec, OptimSpec


def schedule(spec: OptimSpec, steps: int) -> optax.Schedule:
    """Warmup-stable-decay: linear warmup over ``warmup_steps``, a constant peak, then a
    cooldown over the last ``decay_fraction`` of the steps to ``final_lr_fraction`` of the
    peak, linear or 1 − √t (Hägele et al. 2024, arXiv 2405.18392). The warmup does not depend
    on ``steps``, so a shorter run branched off a longer one shares its schedule."""
    warmup = min(spec.warmup_steps, max(steps - 1, 1))
    peak, floor = spec.lr, spec.lr * spec.final_lr_fraction
    decay = max(1, int(spec.decay_fraction * steps))
    stable = cooldown_start(spec, steps) - warmup
    cooldown: optax.Schedule
    if spec.decay_shape == "sqrt":

        def sqrt_cooldown(t: ArrayLike) -> jax.Array:
            frac = jnp.clip(t / decay, 0.0, 1.0)
            return floor + (peak - floor) * (1.0 - jnp.sqrt(frac))

        cooldown = sqrt_cooldown
    else:
        cooldown = optax.linear_schedule(peak, floor, decay)
    return optax.join_schedules(
        [optax.linear_schedule(0.0, peak, warmup), optax.constant_schedule(peak), cooldown],
        [warmup, warmup + stable],
    )


def cooldown_start(spec: OptimSpec, steps: int) -> int:
    """The first (0-based) step of the WSD cooldown: the schedule's last ``decay_fraction``."""
    warmup = min(spec.warmup_steps, max(steps - 1, 1))
    decay = max(1, int(spec.decay_fraction * steps))
    return warmup + max(0, steps - warmup - decay)


def scale_by_adam_cooldown_b2(b1: float, b2: float, cooldown_b2: float, start: int,
                              eps: float) -> optax.GradientTransformation:  # fmt: skip
    """optax.scale_by_adam with β2 = ``b2`` on the first ``start`` updates and
    ``cooldown_b2`` after (β2 raised in the WSD cooldown: a longer second-moment average
    while the rate decays). Its state is optax's ScaleByAdamState, so a checkpoint of either
    restores into the other (a resume, or a cooldown branch off a stable phase). The second
    moment's bias correction is 1 − Π β2 over the updates so far (b2^min(t, start) ·
    cooldown_b2^max(t − start, 0)), not 1 − β2^t: switching β2 does not rescale it."""
    base = optax.scale_by_adam(b1=b1, b2=b2, eps=eps)

    def cooled(updates, state):
        t = optax.safe_increment(state.count)
        mu = optax.tree.update_moment(updates, state.mu, b1, 1)
        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, cooldown_b2, 2)
        mu_hat = optax.tree.bias_correction(mu, b1, t)
        after = (t - start).astype(jnp.float32)
        prod = jnp.float32(b2) ** start * jnp.float32(cooldown_b2) ** after
        nu_hat = jax.tree.map(lambda v: v / (1 - prod).astype(v.dtype), nu)
        updates = jax.tree.map(lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat)
        nu = optax.tree.cast_like(nu, state.nu)
        return updates, optax.ScaleByAdamState(count=t, mu=mu, nu=nu)

    def update(updates, state, params=None):
        # before the cooldown: optax's own update, op for op (a branch off a run without
        # cooldown_b2 trains bitwise what the run trained up to its branch step)
        del params
        inside = optax.safe_increment(state.count) > start  # 0-based step t − 1 ≥ start
        return jax.lax.cond(inside, cooled, lambda u, s: base.update(u, s), updates, state)

    return optax.GradientTransformation(base.init, update)


# Polar Express (Amsel et al. 2025, arXiv 2505.16932): per-step minimax degree-5 polynomials,
# 8 steps, safety factor 1.01 on all but the last step (github.com/NoahAmsel/PolarExpress).
# Its update has the intended RMS whatever the rank and spectrum (Keller Jordan's 5-step
# quintic under-delivers it by 2-3x on rank-32/64 factors).
_PE_RAW = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
)
POLAR_EXPRESS_8 = (
    *((a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _PE_RAW[:-1]),
    _PE_RAW[-1],
)


def _muon(spec: OptimSpec, lr: optax.Schedule) -> optax.GradientTransformation:
    from optax.contrib import MuonDimensionNumbers, muon

    def dims(params):
        return jax.tree.map(
            lambda p: (
                MuonDimensionNumbers(reduction_axis=p.ndim - 2, output_axis=p.ndim - 1)
                if p.ndim >= 2
                else None
            ),
            params,
        )

    return muon(
        lr,
        ns_coeffs=POLAR_EXPRESS_8,
        ns_steps=len(POLAR_EXPRESS_8),  # optax raises if the table is longer than ns_steps
        beta=spec.muon_beta,
        weight_decay=spec.weight_decay,
        adam_b1=spec.b1,
        adam_b2=spec.b2,
        muon_weight_dimension_numbers=dims,
        consistent_rms=spec.muon_rms,  # 0.2: AdamW's update RMS
    )


class ZClipState(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


ZCLIP_WARMUP = 25


def zscore_limit(state: ZClipState, z: float, warmup: int = ZCLIP_WARMUP) -> jax.Array:
    """The norm above which ZClip scales the gradient down (inf during its warmup)."""
    std = jnp.sqrt(jnp.maximum(state.var, 0.0))
    return jnp.where(state.count < warmup, jnp.inf, state.mean + z * std)


def zscore_clip(z: float, decay: float, warmup: int = ZCLIP_WARMUP) -> optax.GradientTransformation:
    """Clip only outliers: scale the gradient down to mean + z·std of an EMA of past global
    norms (ZClip, arXiv 2504.02507). Unlike a fixed threshold it is inactive on ordinary
    steps whatever the model size (raw LoRA gradient norms are 5–15, so clip_norm=1 clips
    every step and reweights steps by 1/‖g‖). The first ``warmup`` steps only build the
    statistics."""

    def init(params):
        del params  # separate buffers: the optimizer state is donated leaf by leaf
        return ZClipState(
            jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32)
        )

    def update(updates, state, params=None):
        del params
        norm = optax.tree.norm(updates)
        limit = zscore_limit(state, z, warmup)
        scale = jnp.minimum(1.0, limit / jnp.maximum(norm, 1e-12))
        clipped = norm * scale
        first = state.count == 0
        mean = jnp.where(first, clipped, decay * state.mean + (1 - decay) * clipped)
        var = jnp.where(first, 0.0, decay * state.var + (1 - decay) * (clipped - mean) ** 2)
        updates = jax.tree.map(lambda g: g * scale.astype(g.dtype), updates)
        return updates, ZClipState(mean, var, state.count + 1)

    return optax.GradientTransformation(init, update)


class PeakState(NamedTuple):
    peak: jax.Array


def unit_schedule(spec: OptimSpec, steps: int) -> optax.Schedule:
    """The schedule's shape at a peak of 1: the learning rate is ``peak · unit(t)``."""
    import dataclasses

    return schedule(dataclasses.replace(spec, lr=1.0), steps)


def _named(path, *names: str) -> bool:
    return any(getattr(k, "key", getattr(k, "name", None)) in names for k in path)


def scale_by_peak(peak: float, b_ratio: float = 1.0) -> optax.GradientTransformation:
    """Multiply the update by the peak learning rate, held as an array in the optimizer
    state (not a compile-time constant): runs that differ only in their rate (a sweep's
    grid) compile to the same program and share the persistent compilation cache.
    ``b_ratio`` ≠ 1 (LoRA+, Hayou et al. 2024, arXiv 2402.12354): the B factors' rate is
    ``b_ratio`` times the A factors' (a compile-time constant; the state is unchanged)."""

    def init(params):
        del params
        return PeakState(jnp.asarray(peak, jnp.float32))

    def update(updates, state, params=None):
        del params
        if b_ratio == 1.0:
            return jax.tree.map(lambda u: u * state.peak.astype(u.dtype), updates), state
        b_peak = state.peak * jnp.float32(b_ratio)

        def scale(path, u):
            return u * (b_peak if _named(path, "lora_b") else state.peak).astype(u.dtype)

        return jax.tree_util.tree_map_with_path(scale, updates), state

    return optax.GradientTransformation(init, update)


class EmaState(NamedTuple):
    ema: Any  # the adapters' average, fp32, the params' tree
    count: jax.Array


def ema_decay_at(decay: float, count: jax.Array) -> jax.Array:
    """min(decay, (1 + t)/(10 + t)) at the t-th accepted update (t = 0, 1, …)."""
    t = count.astype(jnp.float32)
    return jnp.minimum(jnp.float32(decay), (1.0 + t) / (10.0 + t))


def track_ema(decay: float) -> optax.GradientTransformation:
    """The last stage of the chain: the updates pass unchanged; the state keeps an fp32 EMA
    of the parameters *after* the update (params + updates). Its init is the parameters
    (the run's start). Being optimizer state, it is checkpointed, restored and donated with
    it, and a rejected (non-finite) update leaves it unchanged with the rest."""

    def init(params):
        return EmaState(jax.tree.map(lambda p: jnp.array(p, jnp.float32), params),
                        jnp.zeros((), jnp.int32))  # fmt: skip

    def update(updates, state, params=None):
        if params is None:
            raise ValueError("track_ema needs the parameters")
        d = ema_decay_at(decay, state.count)
        ema = jax.tree.map(
            lambda e, p, u: d * e + (1.0 - d) * (p.astype(jnp.float32) + u.astype(jnp.float32)),
            state.ema, params, updates,
        )  # fmt: skip
        return updates, EmaState(ema, state.count + 1)

    return optax.GradientTransformation(init, update)


def ema_params(opt_state) -> Any | None:
    """The EMA adapters held in an optimizer state (None without ``optim.ema_decay``)."""
    found = [s for s in jax.tree.leaves(opt_state, is_leaf=lambda x: isinstance(x, EmaState))
             if isinstance(s, EmaState)]  # fmt: skip
    return found[0].ema if found else None


class CogCalibState(NamedTuple):
    threshold: jax.Array  # the EMA of the steps' NLL quantiles (nats)
    count: jax.Array  # steps folded in (0: no threshold yet, nothing is gated)


def track_cogcalib() -> optax.GradientTransformation:
    """The running CogCalib threshold (loss.cogcalib_quantile) as optimizer state: the
    updates pass unchanged, the train step writes the state (:func:`with_cogcalib`). Being
    optimizer state it is checkpointed, restored and donated with it, and a rejected
    (non-finite) step leaves it unchanged with the rest."""

    def init(params):
        del params
        return CogCalibState(jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32))

    def update(updates, state, params=None):
        del params
        return updates, state

    return optax.GradientTransformation(init, update)


def _is_cogcalib(x) -> bool:
    return isinstance(x, CogCalibState)


def cogcalib_state(opt_state) -> CogCalibState | None:
    found = [s for s in jax.tree.leaves(opt_state, is_leaf=_is_cogcalib) if _is_cogcalib(s)]
    return found[0] if found else None


def with_cogcalib(opt_state, state: CogCalibState):
    """``opt_state`` with its CogCalibState replaced by ``state``."""
    return jax.tree.map(lambda x: state if _is_cogcalib(x) else x, opt_state, is_leaf=_is_cogcalib)


def cogcalib_update(state: CogCalibState, quantile: jax.Array, decay: float) -> CogCalibState:
    """Fold one step's NLL quantile into the EMA (the first one sets it; a NaN quantile, a
    step without slots, leaves the state unchanged)."""
    ok = jnp.isfinite(quantile)
    q = jnp.where(ok, quantile, state.threshold).astype(jnp.float32)
    new = jnp.where(state.count == 0, q, decay * state.threshold + (1.0 - decay) * q)
    return CogCalibState(jnp.where(ok, new, state.threshold), state.count + ok.astype(jnp.int32))


def decay_mask(params) -> Any:
    """Weight decay on the LoRA factors only: a ``lora.gate_full`` kernel is the base weight
    itself, and decaying it pulls it to 0, not to the base."""

    return jax.tree_util.tree_map_with_path(lambda path, _: _named(path, "lora_a", "lora_b"),
                                            params)  # fmt: skip


def make_optimizer(spec: OptimSpec, steps: int,
                   loss: LossSpec | None = None) -> optax.GradientTransformation:  # fmt: skip
    """Clipping, then AdamW or Muon at the unit schedule (peak 1), then the peak rate from
    the state. Every term of either optimizer's update (decoupled weight decay included) is
    linear in the rate, so this is the optimizer at ``schedule(spec, steps)``. AdamW's
    decoupled weight decay applies to the LoRA factors only (:func:`decay_mask`; Muon's
    applies to its matrix parameters, a gate_full kernel included); ``ema_decay`` adds
    :func:`track_ema` last (it sees the final update); a running CogCalib threshold
    (``loss.cogcalib_quantile``) adds :func:`track_cogcalib` after it. ``cooldown_b2``
    (AdamW only) raises β2 in the cooldown (:func:`scale_by_adam_cooldown_b2`, the same
    state); ``loraplus_ratio`` scales the B factors' rate (:func:`scale_by_peak`)."""
    lr = unit_schedule(spec, steps)
    clip = {
        "global": [optax.clip_by_global_norm(spec.clip_norm)],
        "zscore": [zscore_clip(spec.clip_z, spec.clip_ema)],
        "none": [],
    }[spec.clip_mode]
    if spec.name == "adamw":
        mask = {"mask": decay_mask} if spec.weight_decay else {}
        if spec.cooldown_b2:  # optax.adamw's chain, its Adam stage replaced: the same state
            adam = scale_by_adam_cooldown_b2(spec.b1, spec.b2, spec.cooldown_b2,
                                             cooldown_start(spec, steps), spec.eps)  # fmt: skip
            base = optax.chain(adam, optax.add_decayed_weights(spec.weight_decay, mask.get("mask")),
                               optax.scale_by_learning_rate(lr))  # fmt: skip
        else:
            base = optax.adamw(lr, b1=spec.b1, b2=spec.b2, eps=spec.eps,
                               weight_decay=spec.weight_decay, **mask)  # fmt: skip
    else:
        if spec.cooldown_b2:
            raise ValueError("optim.cooldown_b2 is implemented for AdamW only")
        base = _muon(spec, lr)
    ema = [track_ema(spec.ema_decay)] if spec.ema_decay else []
    calib = [track_cogcalib()] if loss is not None and loss.cogcalib_adaptive else []
    return optax.chain(*clip, base, scale_by_peak(spec.lr, spec.loraplus_ratio), *ema, *calib)


def clip_stats(spec: OptimSpec, norm: jax.Array, opt_state) -> dict[str, jax.Array]:
    """What the clipping stage does to a gradient of global norm ``norm`` given the optimizer
    state *before* the update: the scale it applies (1: inactive), the threshold, and for
    ZClip its EMA mean and std. Recomputed from the state, so the optimizer is unchanged."""
    if spec.clip_mode == "none":
        return {"clip_scale": jnp.ones((), jnp.float32)}
    if spec.clip_mode == "global":
        limit = jnp.asarray(spec.clip_norm, jnp.float32)
        out = {}
    else:
        state = next(
            s
            for s in jax.tree.leaves(opt_state, is_leaf=lambda x: isinstance(x, ZClipState))
            if isinstance(s, ZClipState)
        )
        limit = zscore_limit(state, spec.clip_z)
        std = jnp.sqrt(jnp.maximum(state.var, 0.0))
        # the threshold once the warmup is over (finite: logged every step)
        out = {
            "zclip_mean": state.mean,
            "zclip_std": std,
            "zclip_limit": state.mean + spec.clip_z * std,
        }
    scale = jnp.minimum(1.0, limit / jnp.maximum(norm, 1e-12))
    return out | {"clip_scale": scale}
