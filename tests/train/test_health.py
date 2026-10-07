"""Run health: the on-device diagnostics against direct computations, clip statistics against
the clipping transforms, the spike detector and the kill rules."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from tjev.config import OptimSpec
from tjev.train.health import HealthMonitor, HealthRules, SpikeDetector, device_metrics
from tjev.train.optim import clip_stats, make_optimizer, schedule, unit_schedule, zscore_clip


def _params(key):
    """A LoRA-shaped tree with the model's module names (gdn, attention, mlps)."""
    ks = iter(jax.random.split(key, 12))

    def pair(lead, fan_in, r, fan_out):
        return {"lora_a": jax.random.normal(next(ks), (*lead, fan_in, r)) / math.sqrt(fan_in),
                "lora_b": 0.01 * jax.random.normal(next(ks), (*lead, r, fan_out))}  # fmt: skip

    return {
        "blocks": {
            "gdn": {
                0: {"in_proj_qkv": pair((2,), 16, 4, 24)},
                1: {"out_proj": pair((2,), 8, 4, 16)},
            },
            "attention": {"q_proj": pair((2,), 16, 4, 32)},
            "mlps": {0: {"up_proj": pair((2,), 16, 4, 40)}},
        }
    }


def _sq(tree, pred=lambda p: True):
    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    return sum(float(jnp.sum(x**2)) for p, x in leaves if pred(jax.tree_util.keystr(p)))


def _run(spec, steps=3):
    params = _params(jax.random.key(0))
    tx = make_optimizer(spec, 100)
    state = tx.init(params)
    out = []
    for i in range(steps):
        grads = jax.tree.map(
            lambda p, i=i: jax.random.normal(jax.random.key(i + 1), p.shape) * 3, params
        )
        norm = optax.tree.norm(grads)
        updates, new_state = tx.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        m = device_metrics(grads, updates, params, new_params, state, new_state, grad_norm=norm,
                           optim=spec, lr_schedule=unit_schedule(spec, 100))  # fmt: skip
        out.append((params, grads, updates, new_params, state, m))
        params, state = new_params, new_state
    return out


@pytest.mark.parametrize("name", ["adamw", "muon"])
@pytest.mark.parametrize("clip", ["global", "zscore", "none"])
def test_device_metrics_match_direct_computations(name, clip):
    spec = OptimSpec(name=name, clip_mode=clip, lr=1e-3, warmup_steps=2, clip_norm=5.0)
    with jax.default_matmul_precision("highest"):
        steps = _run(spec)
    for i, (_, grads, updates, new_params, _, m) in enumerate(steps):
        total = _sq(grads)
        groups = sum(float(m[f"grad_norm_{g}"]) ** 2 for g in ("gdn", "attention", "mlp"))
        assert groups == pytest.approx(total, rel=1e-5)
        assert float(m["grad_norm_a"]) ** 2 + float(m["grad_norm_b"]) ** 2 == pytest.approx(
            total, rel=1e-5
        )
        assert float(m["grad_norm_attention"]) ** 2 == pytest.approx(
            _sq(grads, lambda p: "'attention'" in p), rel=1e-5)  # fmt: skip
        for f in ("a", "b"):
            is_f = lambda p, f=f: f"lora_{f}" in p  # noqa: E731
            u2, p2 = _sq(updates, is_f), _sq(new_params, is_f)
            assert float(m[f"update_norm_{f}"]) == pytest.approx(math.sqrt(u2), rel=1e-5)
            assert float(m[f"update_ratio_{f}"]) == pytest.approx(math.sqrt(u2 / p2), rel=1e-5)
            assert float(m[f"param_norm_{f}"]) == pytest.approx(math.sqrt(p2), rel=1e-5)
        assert float(m["lr"]) == pytest.approx(float(schedule(spec, 100)(i)), rel=1e-6)
        if clip == "global":
            assert float(m["clip_scale"]) == pytest.approx(
                min(1.0, 5.0 / math.sqrt(total)), rel=1e-5
            )
            assert float(m["clip_active"]) == 1.0
        if clip == "zscore":
            assert float(m["clip_scale"]) == 1.0  # inside ZClip's warmup
            assert {"zclip_mean", "zclip_std", "zclip_limit"} <= set(m)
        if clip == "none":
            assert "grad_norm_clipped" in m and float(m["clip_scale"]) == 1.0
        if name == "adamw":
            assert 0.0 <= float(m["adam_eps_fraction"]) <= 1.0
            assert float(m["adam_sqrt_v_mean"]) > 0
        else:
            lr = float(m["lr"])
            if lr > 0:
                assert float(m["muon_ortho_norm"]) == pytest.approx(
                    float(m["update_norm"]) / lr, rel=1e-5)  # fmt: skip
        assert math.isfinite(float(m["momentum_cos"]))
    assert float(steps[0][-1]["momentum_cos"]) == 0.0  # no momentum before step 1


def test_momentum_cosine_is_the_cosine_of_the_old_momentum_and_the_gradient():
    spec = OptimSpec(name="adamw", clip_mode="none", lr=1e-3, warmup_steps=0)
    steps = _run(spec, 2)
    _, grads, _, _, state, m = steps[1]
    mu = state[0][0].mu  # chain(adamw): (ScaleByAdamState, ...) after step 1
    dot = sum(float(jnp.sum(a * b)) for a, b in zip(jax.tree.leaves(mu), jax.tree.leaves(grads),
                                                    strict=True))  # fmt: skip
    cos = dot / math.sqrt(_sq(mu) * _sq(grads))
    assert float(m["momentum_cos"]) == pytest.approx(cos, rel=1e-4, abs=1e-6)


def test_clip_stats_match_the_scale_zclip_applies():
    spec = OptimSpec(clip_mode="zscore", clip_z=0.5, clip_ema=0.9)
    clip = zscore_clip(spec.clip_z, spec.clip_ema)
    params = _params(jax.random.key(0))
    state = clip.init(params)
    rng = np.random.default_rng(0)
    active = 0
    for i in range(60):
        g = jax.tree.map(lambda p, i=i: jax.random.normal(jax.random.key(i), p.shape), params)
        factor = 1 + 3 * rng.random()
        g = jax.tree.map(lambda x, f=factor: x * f, g)
        norm = optax.tree.norm(g)
        stats = clip_stats(spec, norm, (state,))
        out, state = clip.update(g, state)
        applied = float(optax.tree.norm(out)) / float(norm)
        assert float(stats["clip_scale"]) == pytest.approx(applied, rel=1e-5)
        if i < 25:
            assert applied == 1.0  # ZClip's warmup
        active += applied < 1.0
    assert active > 0


def test_spike_detector_flags_outliers_only():
    det = SpikeDetector(k=5.0, history=50, min_history=10)
    rng = np.random.default_rng(0)
    flags = [det.add(1.0 + 0.01 * rng.standard_normal()) for _ in range(100)]
    assert sum(flags) == 0
    assert det.add(1.5)  # 50 σ
    assert det.count == 1
    assert det.add(float("nan"))
    assert not det.add(1.0)


def _healthy(step, rng, **extra):
    return {"loss": 1.0 / (1 + 0.01 * step) + 0.01 * rng.standard_normal(), "grad_norm": 8.0,
            "update_rejected": 0.0, "update_ratio_a": 2e-3, "update_ratio_b": 3e-2,
            "momentum_cos": 0.3, "clip_active": 0.0} | extra  # fmt: skip


def test_monitor_healthy_run_is_ok():
    mon, rng = HealthMonitor(warmup=10, clip_mode="zscore"), np.random.default_rng(0)
    for step in range(1, 301):
        mon.add(step, _healthy(step, rng))
    v = mon.verdict()
    assert v["status"] == "ok", v["rules"]
    assert v["stats"]["spikes"] == 0
    assert v["stats"]["loss_recent"] < v["stats"]["loss_first"]


@pytest.mark.parametrize(
    ("rule", "level", "bad"),
    [
        (
            "non_finite",
            "kill",
            lambda s: {"update_rejected": 1.0, "loss": float("nan")} if s > 100 else {},
        ),
        ("grad_blowup", "kill", lambda s: {"grad_norm": 200.0} if s > 150 else {}),
        ("grad_blowup", "warn", lambda s: {"grad_norm": 40.0} if s > 150 else {}),
        ("update_ratio_a", "kill", lambda s: {"update_ratio_a": 0.1}),
        ("update_ratio_a_dead", "warn", lambda s: {"update_ratio_a": 1e-7}),
        ("update_ratio_b", "warn", lambda s: {"update_ratio_b": 0.5}),
        ("momentum_against_gradient", "warn", lambda s: {"momentum_cos": -0.2} if s > 100 else {}),
        ("clip_saturated", "warn", lambda s: {"clip_active": 1.0} if s % 2 else {}),
        ("loss_above_start", "kill", lambda s: {"loss": 3.0} if s > 150 else {}),
        ("loss_spikes", "warn", lambda s: {"loss": 5.0} if s in (120, 160, 200) else {}),
        ("loss_spikes", "kill", lambda s: {"loss": 5.0} if s % 10 == 0 and s > 100 else {}),
    ],
)
def test_monitor_rules_fire(rule, level, bad):
    mon, rng = HealthMonitor(warmup=10, clip_mode="zscore"), np.random.default_rng(0)
    for step in range(1, 301):
        mon.add(step, _healthy(step, rng, **bad(step)))
    v = mon.verdict()
    assert rule in v["rules"], v["rules"]
    assert v["rules"][rule]["level"] == level
    assert mon.status == ("kill" if level == "kill" else mon.status)
    if level == "kill":
        assert v["kill_step"] is not None


def test_monitor_regression_against_best_progress():
    mon = HealthMonitor(HealthRules(), warmup=0)
    rng = np.random.default_rng(1)
    for step in range(1, 301):
        loss = 1.0 - 0.5 * min(step, 150) / 150 + (0.45 if step > 200 else 0.0)
        mon.add(step, {"loss": loss + 0.005 * rng.standard_normal(), "grad_norm": 5.0})
    assert mon.fired["loss_regression"]["level"] == "kill"
    assert "loss_above_start" not in mon.fired


def test_clip_rule_ignores_global_clipping():
    mon, rng = HealthMonitor(warmup=0, clip_mode="global"), np.random.default_rng(0)
    for step in range(1, 201):
        mon.add(step, _healthy(step, rng, clip_active=1.0))  # clip_norm=1 clips every step
    assert "clip_saturated" not in mon.fired


def test_verdict_is_strict_json():
    import json

    mon = HealthMonitor()
    mon.add(1, {"loss": 1.0})
    json.dumps(mon.verdict(), allow_nan=False)


def test_spikes_are_judged_within_a_sequence_bucket():
    """Long-text steps (few slots) have a higher loss than short ones: mixed steps of two
    buckets, each steady, are no spikes; a real jump within one bucket is."""
    rng = np.random.default_rng(0)
    m = HealthMonitor()
    spikes = 0
    for step in range(1, 301):
        long = step % 12 == 0
        loss = (1.1 if long else 0.6) + 0.02 * rng.normal()
        spikes += m.add(step, {"loss": loss, "grad_norm": 1.0}, bucket=4096 if long else 1024)
    assert spikes == 0 and m.spikes.count == 0
    assert m.add(301, {"loss": 0.95, "grad_norm": 1.0}, bucket=1024)  # +0.35 in its bucket
