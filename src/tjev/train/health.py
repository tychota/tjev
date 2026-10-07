"""Run health: cheap on-device optimizer diagnostics, a host-side loss-spike detector and
the kill rules that turn them into a per-run verdict (``health.json``, docs/sweep-plan.md).

On device (inside the jitted step, reduced to scalars: a few reductions over the LoRA
factors, which are ~1% of the model, against a forward and backward over 65k tokens):

* the global gradient norm per module group (gdn, attention, mlp) and per factor (A, B);
* the clipping stage: scale applied (1: inactive), threshold, ZClip's EMA mean and std;
* per factor, the update norm, the update and parameter RMS and their ratio (the relative
  step actually applied), the parameter norms;
* the first moment (AdamW's ``mu``, Muon's momentum): its norm and its cosine with the
  incoming gradient (negative for long stretches: oscillation or a stale momentum);
* AdamW: the mean of √v̂ and the fraction of entries whose √v̂ is below ε (where ε, not the
  gradient, sets the step); Muon: the orthogonalised update's norm (update / lr);
* the learning rate the optimizer actually used (its schedule at its own step count; a
  rejected update does not advance it).

On the host, :class:`HealthMonitor` consumes the per-step values (fetched one step late
with the other metrics), counts loss spikes against a running median/MAD and applies the
rules of :class:`HealthRules`. The thresholds are rules of thumb (Wortsman et al. 2023,
arXiv 2309.14322: update/param ratio and logit growth; ZClip, arXiv 2504.02507) to be
recalibrated on the first sweep runs.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tjev.config import OptimSpec
from tjev.train.optim import clip_stats

GROUPS = ("gdn", "attention", "mlp")


def _group(path: str) -> str | None:
    for name in GROUPS:  # SuperBlock: .gdn[i], .attention, .mlps[i]
        if f"'{name}" in path or f".{name}" in path:
            return name
    return None


def _factor(path: str) -> str | None:
    return "a" if "lora_a" in path else "b" if "lora_b" in path else None


def _sq(x: jax.Array) -> jax.Array:
    return jnp.sum(jnp.square(x.astype(jnp.float32)))


def _fields(tree: Any, cls_name: str) -> list[Any]:
    """Optimizer sub-states of the NamedTuple class ``cls_name`` (depth first)."""
    out: list[Any] = []

    def visit(node: Any) -> None:
        if type(node).__name__ == cls_name:
            out.append(node)
            return
        if isinstance(node, tuple):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            for child in node.values():
                visit(child)
        elif hasattr(node, "inner_state"):
            visit(node.inner_state)
        elif hasattr(node, "inner_states"):
            visit(node.inner_states)

    visit(tree)
    return out


def _momentum(opt_state: Any) -> Any | None:
    for name in ("MuonState", "ScaleByAdamState"):
        for state in _fields(opt_state, name):
            if jax.tree.leaves(state.mu):  # Muon's masked Adam partition has no leaves
                return state
    return None


def device_metrics(
    grads: Any,
    updates: Any,
    lora: Any,
    new_lora: Any,
    opt_state: Any,
    new_opt: Any,
    *,
    grad_norm: jax.Array,
    optim: OptimSpec | None = None,
    lr_schedule: optax.Schedule | None = None,
) -> dict[str, jax.Array]:
    """The ``health/*`` scalars of one step (jit-traceable). ``grads`` are before clipping,
    ``opt_state`` / ``lora`` before the update, ``new_*`` after it."""
    paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(grads)[0]]
    g_sq = [_sq(g) for g in jax.tree.leaves(grads)]
    u_leaves = jax.tree.leaves(updates)
    p_leaves = jax.tree.leaves(new_lora)
    out: dict[str, jax.Array] = {}
    zero = jnp.zeros((), jnp.float32)
    for name in GROUPS:
        out[f"grad_norm_{name}"] = jnp.sqrt(
            sum((s for p, s in zip(paths, g_sq, strict=True) if _group(p) == name), zero)
        )
    for f in ("a", "b"):
        idx = [i for i, p in enumerate(paths) if _factor(p) == f]
        size = float(sum(p_leaves[i].size for i in idx)) or 1.0
        g2 = sum((g_sq[i] for i in idx), zero)
        u2 = sum((_sq(u_leaves[i]) for i in idx), zero)
        p2 = sum((_sq(p_leaves[i]) for i in idx), zero)
        out[f"grad_norm_{f}"] = jnp.sqrt(g2)
        out[f"update_norm_{f}"] = jnp.sqrt(u2)
        out[f"param_norm_{f}"] = jnp.sqrt(p2)
        out[f"update_rms_{f}"] = jnp.sqrt(u2 / size)
        out[f"param_rms_{f}"] = jnp.sqrt(p2 / size)
        # against the parameters after the update: defined at step 1 for B (B₀ = 0)
        out[f"update_ratio_{f}"] = jnp.sqrt(u2 / jnp.maximum(p2, 1e-30))
    update_norm = jnp.sqrt(out["update_norm_a"] ** 2 + out["update_norm_b"] ** 2)
    out["update_norm"] = update_norm
    if optim is not None:
        clip = clip_stats(optim, grad_norm, opt_state)
        out |= clip
        out["clip_active"] = (clip["clip_scale"] < 1.0).astype(jnp.float32)
        out["grad_norm_clipped"] = grad_norm * clip["clip_scale"]
    old, new = _momentum(opt_state), _momentum(new_opt)
    if old is not None and new is not None:
        mu_old, mu_new = jax.tree.leaves(old.mu), jax.tree.leaves(new.mu)
        dot = sum((jnp.sum(m.astype(jnp.float32) * g.astype(jnp.float32))
                   for m, g in zip(mu_old, jax.tree.leaves(grads), strict=True)), zero)  # fmt: skip
        m_norm = jnp.sqrt(sum((_sq(m) for m in mu_old), zero))
        g_norm = jnp.sqrt(sum(g_sq, zero))
        out["momentum_cos"] = dot / jnp.maximum(m_norm * g_norm, 1e-30)  # 0 at step 1
        out["momentum_norm"] = jnp.sqrt(sum((_sq(m) for m in mu_new), zero))
        if type(new).__name__ == "ScaleByAdamState" and optim is not None:
            t = new.count.astype(jnp.float32)
            nu = jax.tree.leaves(new.nu)
            corr = 1.0 - jnp.asarray(optim.b2, jnp.float32) ** jnp.maximum(t, 1.0)
            roots = [jnp.sqrt(v.astype(jnp.float32) / corr) for v in nu]
            n = float(sum(v.size for v in nu))
            out["adam_sqrt_v_mean"] = sum((jnp.sum(r) for r in roots), zero) / n
            out["adam_eps_fraction"] = (
                sum((jnp.sum((r < optim.eps).astype(jnp.float32)) for r in roots), zero) / n
            )
    if lr_schedule is not None:
        counts = _fields(opt_state, "ScaleByScheduleState")
        if counts:
            lr = jnp.asarray(lr_schedule(counts[0].count), jnp.float32)
            peaks = _fields(opt_state, "PeakState")  # the unit schedule × the traced peak
            if peaks:
                lr = lr * peaks[0].peak
            out["lr"] = lr
            if optim is not None and optim.name == "muon":
                out["muon_ortho_norm"] = update_norm / jnp.maximum(lr, 1e-30)
    return out


# ------------------------------------------------------------------------------------------
# Host side


@dataclass(frozen=True)
class HealthRules:
    """Kill (``kill``) and warning (``warn``) thresholds; ``window`` steps per rolling median."""

    window: int = 20
    spike_k: float = 5.0  # loss > median + k·MAD (σ-scaled, ×1.4826) of the last spike_history
    spike_history: int = 50
    spike_min_history: int = 10
    spikes_warn: int = 3
    spikes_kill_fraction: float = 0.05  # spikes over > 5% of the steps (≥ 100 steps)
    rejected_kill: int = 3  # non-finite loss or gradient (update rejected on device)
    grad_blowup_warn: float = 3.0  # recent median grad norm / early median
    grad_blowup_kill: float = 10.0
    clip_fraction_warn: float = 0.2  # zscore clipping active on > 20% of the last 50 steps
    clip_history: int = 50
    ratio_a_warn: float = 1e-2  # median update RMS / param RMS of A (after the warmup)
    ratio_a_kill: float = 5e-2
    ratio_a_dead: float = 1e-5  # below: the rate does nothing
    ratio_b_warn: float = 0.2  # B grows from zero: only after b_after steps
    b_after: int = 100
    cos_negative_warn: int = 50  # consecutive steps with momentum·gradient cosine < 0
    regress_warn: float = 0.5  # gave back this fraction of the loss progress
    regress_kill: float = 0.75


class SpikeDetector:
    """Running median / MAD of the last ``history`` losses; a loss above median + k·σ̂
    (σ̂ = 1.4826·MAD) is a spike. Spikes do not enter the history (they would inflate it)."""

    def __init__(self, k: float = 5.0, history: int = 50, min_history: int = 10):
        self.k, self.min_history = k, min_history
        self.values: deque[float] = deque(maxlen=history)
        self.count = 0
        self.median = math.nan
        self.mad = math.nan

    def add(self, loss: float) -> bool:
        spike = False
        if len(self.values) >= self.min_history:
            arr = np.asarray(self.values)
            self.median = float(np.median(arr))
            self.mad = 1.4826 * float(np.median(np.abs(arr - self.median)))
            spike = not math.isfinite(loss) or loss > self.median + self.k * max(self.mad, 1e-6)
        if spike:
            self.count += 1
        elif math.isfinite(loss):
            self.values.append(loss)
        return spike


class BucketSpikes:
    """One SpikeDetector per sequence bucket. A step holds a single bucket, and a long-text
    step (16–32 answer slots) has a higher loss than a short one (~200 slots): against one
    pooled median every long-text step would count as a spike (measured: 14 of 300 steps
    of a healthy 0.8B run, 4.7%, at the 5% kill threshold)."""

    def __init__(self, k: float = 5.0, history: int = 50, min_history: int = 10):
        self.args = (k, history, min_history)
        self.by: dict[object, SpikeDetector] = {}

    def add(self, loss: float, key: object = None) -> bool:
        detector = self.by.setdefault(key, SpikeDetector(*self.args))
        return detector.add(loss)

    @property
    def count(self) -> int:
        return sum(d.count for d in self.by.values())

    def _main(self) -> SpikeDetector | None:  # the bucket with the most steps
        return max(self.by.values(), key=lambda d: len(d.values) + d.count, default=None)

    @property
    def median(self) -> float:
        d = self._main()
        return d.median if d else math.nan

    @property
    def mad(self) -> float:
        d = self._main()
        return d.mad if d else math.nan


def _median(xs) -> float:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.median(xs)) if xs else math.nan


class HealthMonitor:
    """Per-step values in, a verdict out: ``ok`` / ``warn`` / ``kill`` with the rules that
    fired. ``warmup``: the optimizer warmup (ratio and gradient rules start after it)."""

    def __init__(self, rules: HealthRules | None = None, *, warmup: int = 0,
                 clip_mode: str = "global", optimizer: str = "adamw"):  # fmt: skip
        self.rules = rules or HealthRules()
        self.warmup, self.clip_mode, self.optimizer = warmup, clip_mode, optimizer
        r = self.rules
        self.spikes = BucketSpikes(r.spike_k, r.spike_history, r.spike_min_history)
        self.steps = 0
        self.last_step = 0
        self.rejected = 0
        self.losses: list[float] = []
        self.grad: list[float] = []
        self.clip: deque[float] = deque(maxlen=r.clip_history)
        self.ratio_a: deque[float] = deque(maxlen=r.window)
        self.ratio_b: deque[float] = deque(maxlen=r.window)
        self.cos_run = 0
        self.cos_run_max = 0
        self.lr, self.lr_peak = math.nan, 0.0
        self.best_rolling = math.inf
        self.kill_step: int | None = None
        self.fired: dict[str, dict] = {}

    def add(self, step: int, m: dict[str, float], bucket: object = None) -> bool:
        """One step's values (``loss``, ``grad_norm``, ``update_rejected`` and ``health/*``
        without the prefix) and its sequence bucket (spikes are judged within a bucket);
        returns whether the loss spiked."""
        self.steps += 1
        self.last_step = step
        self.rejected += round(m.get("update_rejected", 0.0))
        loss = float(m["loss"])
        spike = self.spikes.add(loss, bucket)
        if math.isfinite(loss):
            self.losses.append(loss)
        if step > self.warmup:  # the reference is the first 2·window steps after warmup
            self.grad.append(float(m.get("grad_norm", math.nan)))
        if "clip_active" in m:
            self.clip.append(float(m["clip_active"]))
        if "lr" in m:
            self.lr, self.lr_peak = float(m["lr"]), max(self.lr_peak, float(m["lr"]))
        if step > self.warmup:
            for key, buf in (("update_ratio_a", self.ratio_a), ("update_ratio_b", self.ratio_b)):
                if key in m and math.isfinite(m[key]):
                    buf.append(float(m[key]))
        if "momentum_cos" in m and step > 1:
            self.cos_run = self.cos_run + 1 if m["momentum_cos"] < 0 else 0
            self.cos_run_max = max(self.cos_run_max, self.cos_run)
        w = self.rules.window
        if len(self.losses) >= w:
            self.best_rolling = min(self.best_rolling, _median(self.losses[-w:]))
        self._check(step)
        return spike

    def _fire(self, rule: str, level: str, step: int, value: float, threshold: float) -> None:
        previous = self.fired.get(rule)
        if previous is None or (previous["level"] == "warn" and level == "kill"):
            self.fired[rule] = {"level": level, "step": step, "value": value,
                                "threshold": threshold}  # fmt: skip
            if level == "kill" and self.kill_step is None:
                self.kill_step = step

    def _check(self, step: int) -> None:
        r, w = self.rules, self.rules.window
        if self.rejected >= r.rejected_kill:
            self._fire("non_finite", "kill", step, self.rejected, r.rejected_kill)
        if self.spikes.count >= r.spikes_warn:
            self._fire("loss_spikes", "warn", step, self.spikes.count, r.spikes_warn)
        if self.steps >= 100 and self.spikes.count > r.spikes_kill_fraction * self.steps:
            self._fire("loss_spikes", "kill", step, self.spikes.count / self.steps,
                       r.spikes_kill_fraction)  # fmt: skip
        if len(self.losses) >= 2 * w and step > self.warmup + w:
            first, recent = _median(self.losses[:w]), _median(self.losses[-w:])
            if recent > first:
                self._fire("loss_above_start", "kill", step, recent, first)
            progress, lost = first - self.best_rolling, recent - self.best_rolling
            # significant: more than one per-step σ̂ (~3.5 SE of a 20-step median)
            if progress > 0 and lost > max(self.spikes.mad, 0.0):
                frac = lost / progress
                if frac > r.regress_kill:
                    self._fire("loss_regression", "kill", step, frac, r.regress_kill)
                elif frac > r.regress_warn:
                    self._fire("loss_regression", "warn", step, frac, r.regress_warn)
        if len(self.grad) >= 3 * w:
            ratio = _median(self.grad[-w:]) / max(_median(self.grad[: 2 * w]), 1e-12)
            if ratio > r.grad_blowup_kill:
                self._fire("grad_blowup", "kill", step, ratio, r.grad_blowup_kill)
            elif ratio > r.grad_blowup_warn:
                self._fire("grad_blowup", "warn", step, ratio, r.grad_blowup_warn)
        if self.clip_mode == "zscore" and len(self.clip) == self.clip.maxlen:
            frac = float(np.mean(self.clip))
            if frac > r.clip_fraction_warn:
                self._fire("clip_saturated", "warn", step, frac, r.clip_fraction_warn)
        if len(self.ratio_a) == w:
            ra = _median(self.ratio_a)
            if ra > r.ratio_a_kill:
                self._fire("update_ratio_a", "kill", step, ra, r.ratio_a_kill)
            elif ra > r.ratio_a_warn:
                self._fire("update_ratio_a", "warn", step, ra, r.ratio_a_warn)
            elif ra < r.ratio_a_dead and not self.lr < 0.5 * self.lr_peak:  # not in the cooldown
                self._fire("update_ratio_a_dead", "warn", step, ra, r.ratio_a_dead)
        if len(self.ratio_b) == w and step > r.b_after:
            rb = _median(self.ratio_b)
            if rb > r.ratio_b_warn:
                self._fire("update_ratio_b", "warn", step, rb, r.ratio_b_warn)
        if self.cos_run >= r.cos_negative_warn:
            self._fire("momentum_against_gradient", "warn", step, self.cos_run,
                       r.cos_negative_warn)  # fmt: skip

    @property
    def status(self) -> str:
        levels = {f["level"] for f in self.fired.values()}
        return "kill" if "kill" in levels else "warn" if levels else "ok"

    def row(self) -> dict[str, float]:
        """Host-side scalars for the metrics row (``learning/spikes`` and ``health/*``)."""
        out = {"spikes": float(self.spikes.count),
               "status": {"ok": 0.0, "warn": 1.0, "kill": 2.0}[self.status]}  # fmt: skip
        if math.isfinite(self.spikes.median):
            out |= {"loss_median": self.spikes.median, "loss_mad": self.spikes.mad}
        return out

    def verdict(self, *, killed: bool = False) -> dict:
        w = self.rules.window
        return _finite(
            {
                "status": "killed" if killed else self.status,
                "step": self.last_step,
                "kill_step": self.kill_step,
                "rules": self.fired,
                "stats": {
                    "steps": self.steps,
                    "rejected": self.rejected,
                    "spikes": self.spikes.count,
                    "loss_first": _median(self.losses[:w]),
                    "loss_best_rolling": self.best_rolling,
                    "loss_recent": _median(self.losses[-w:]),
                    "grad_norm_early": _median(self.grad[: 2 * w]),
                    "grad_norm_recent": _median(self.grad[-w:]),
                    "clip_fraction_recent": float(np.mean(self.clip)) if self.clip else None,
                    "update_ratio_a_recent": _median(self.ratio_a),
                    "update_ratio_b_recent": _median(self.ratio_b),
                    "momentum_cos_negative_run_max": self.cos_run_max,
                },
                "thresholds": asdict(self.rules),
            }
        )


def _finite(value: Any) -> Any:
    """NaN / inf → None (health.json is strict JSON)."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
