"""Optimizers and the learning-rate schedule for adapter training.

AdamW is the default (Qwen3.5 was pretrained with AdamW, and Muon's measured edge on the
LoRA factors was within seed noise: docs/research/muon.md). Muon is available for the LoRA
factors as an option: stacked [S, in, out] factors are orthogonalised per layer (leading
axes are batch axes) with the Polar Express Newton-Schulz polynomials, at the update RMS
of AdamW, so learning rates carry over; everything that is not a matrix uses Adam.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.typing import ArrayLike

from tjev.config import OptimSpec


def schedule(spec: OptimSpec, steps: int) -> optax.Schedule:
    """Warmup-stable-decay: linear warmup over ``warmup_steps``, a constant peak, then a
    cooldown over the last ``decay_fraction`` of the steps to ``final_lr_fraction`` of the
    peak, linear or 1 − √t (Hägele et al. 2024, arXiv 2405.18392). The warmup does not depend
    on ``steps``, so a shorter run branched off a longer one shares its schedule."""
    warmup = min(spec.warmup_steps, max(steps - 1, 1))
    peak, floor = spec.lr, spec.lr * spec.final_lr_fraction
    decay = max(1, int(spec.decay_fraction * steps))
    stable = max(0, steps - warmup - decay)
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


def zscore_clip(z: float, decay: float, warmup: int = 25) -> optax.GradientTransformation:
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
        std = jnp.sqrt(jnp.maximum(state.var, 0.0))
        limit = jnp.where(state.count < warmup, jnp.inf, state.mean + z * std)
        scale = jnp.minimum(1.0, limit / jnp.maximum(norm, 1e-12))
        clipped = norm * scale
        first = state.count == 0
        mean = jnp.where(first, clipped, decay * state.mean + (1 - decay) * clipped)
        var = jnp.where(first, 0.0, decay * state.var + (1 - decay) * (clipped - mean) ** 2)
        updates = jax.tree.map(lambda g: g * scale.astype(g.dtype), updates)
        return updates, ZClipState(mean, var, state.count + 1)

    return optax.GradientTransformation(init, update)


def make_optimizer(spec: OptimSpec, steps: int) -> optax.GradientTransformation:
    lr = schedule(spec, steps)
    clip = {
        "global": [optax.clip_by_global_norm(spec.clip_norm)],
        "zscore": [zscore_clip(spec.clip_z, spec.clip_ema)],
        "none": [],
    }[spec.clip_mode]
    if spec.name == "adamw":
        base = optax.adamw(lr, b1=spec.b1, b2=spec.b2, eps=spec.eps, weight_decay=spec.weight_decay)
    else:
        base = _muon(spec, lr)
    return optax.chain(*clip, base)
