"""Optimizers: Polar Express orthogonalisation at LoRA factor shapes, Muon's update RMS and
first step (B = 0), the WSD schedule, z-score clipping."""

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optax.contrib import MuonDimensionNumbers
from optax.contrib._muon import orthogonalize_via_newton_schulz

from tjev.config import OptimSpec
from tjev.train.checkpoint import Checkpoints
from tjev.train.optim import POLAR_EXPRESS_8, ZClipState, make_optimizer, schedule, zscore_clip

DIMS = MuonDimensionNumbers(reduction_axis=1, output_axis=2)


def _spectrum(key, s, m, n, power):
    """[s, m, n] matrices with singular values i^-power (power 0: flat)."""
    r = min(m, n)
    k1, k2 = jax.random.split(key)
    u, _ = jnp.linalg.qr(jax.random.normal(k1, (s, m, r)))
    v, _ = jnp.linalg.qr(jax.random.normal(k2, (s, n, r)))
    return (u * jnp.arange(1, r + 1.0) ** -power) @ jnp.swapaxes(v, -1, -2)


# LoRA A (in x r) and B (r x out) shapes of the 2B (r 32) and 4B (r 64)
@pytest.mark.parametrize(("m", "n"), [(2048, 32), (32, 6144), (32, 16), (2560, 64), (64, 9216)])
@pytest.mark.parametrize("power", [0.0, 1.0, 1.5])
def test_polar_express_matches_svd_polar(m, n, power):
    with jax.default_matmul_precision("highest"):
        x = _spectrum(jax.random.key(0), 2, m, n, power)
        u, _, vt = jnp.linalg.svd(x, full_matrices=False)
        o = orthogonalize_via_newton_schulz(
            x, jnp.asarray(POLAR_EXPRESS_8), len(POLAR_EXPRESS_8), "frobenius", 1e-8, DIMS
        )
        sv = jnp.linalg.svd(o, compute_uv=False)
    assert float(jnp.abs(sv - 1).max()) < 2e-2
    assert float(jnp.linalg.norm(o - u @ vt) / jnp.linalg.norm(u @ vt)) < 2e-2


def test_muon_update_has_the_target_rms_and_zero_a_step():
    spec = OptimSpec(name="muon", lr=1.0, warmup_steps=0, decay_fraction=0.0, clip_mode="none")
    params = {
        "lora_a": jax.random.normal(jax.random.key(1), (2, 256, 16)) / 16.0,  # [S, in, r]
        "lora_b": jnp.zeros((2, 16, 512)),  # [S, r, out]: zero at init
    }
    tx = make_optimizer(spec, 10)
    # step 1 of LoRA: B = 0 so dL/dA = 0; dL/dB is dense
    grads = {
        "lora_a": jnp.zeros_like(params["lora_a"]),
        "lora_b": jax.random.normal(jax.random.key(2), params["lora_b"].shape),
    }
    with jax.default_matmul_precision("highest"):
        out, _ = tx.update(grads, tx.init(params), params)
    updates = cast("dict[str, jax.Array]", out)
    assert all(bool(jnp.all(jnp.isfinite(u))) for u in jax.tree.leaves(updates))
    np.testing.assert_array_equal(updates["lora_a"], 0.0)
    rms = float(jnp.sqrt(jnp.mean(updates["lora_b"] ** 2)))
    assert abs(rms - 0.2) < 2e-3, rms  # consistent_rms=0.2 at lr 1


def _lr(spec: OptimSpec, steps: int):
    s = schedule(spec, steps)
    return lambda t: float(jnp.asarray(s(t)))


def test_wsd_schedule_warmup_stable_and_cooldowns():
    spec = OptimSpec(lr=1.0, warmup_steps=10, final_lr_fraction=0.02)
    lin = _lr(replace(spec, decay_shape="linear"), 100)
    sq = _lr(spec, 100)
    assert sq(0) == 0.0
    assert abs(sq(10) - 1.0) < 1e-6
    assert abs(sq(79) - 1.0) < 1e-6
    assert abs(sq(100) - 0.02) < 1e-6
    assert all(sq(t) <= lin(t) + 1e-6 for t in range(81, 100))  # 1-sqrt drops faster
    # the warmup does not depend on the horizon: a branch shares its parent's schedule
    short = _lr(spec, 50)
    assert all(abs(short(t) - sq(t)) < 1e-7 for t in range(40))


def test_zscore_clip_leaves_ordinary_steps_and_clips_outliers():
    tx = zscore_clip(z=2.5, decay=0.97, warmup=25)  # the OptimSpec defaults
    state = tx.init({"w": jnp.zeros(3)})
    rng = np.random.default_rng(0)
    clipped = 0
    for _ in range(200):  # norms around 10 (like raw LoRA gradient norms)
        g = {"w": jnp.asarray([10.0 + rng.normal(), 0.0, 0.0])}
        out, state = tx.update(g, state)
        clipped += float(out["w"][0]) != float(g["w"][0])
    assert clipped <= 10  # only the > 2.5 σ tail of ordinary steps (~1%, some slack)
    zstate = cast("ZClipState", state)
    limit = float(zstate.mean + 2.5 * jnp.sqrt(zstate.var))
    out, _ = tx.update({"w": jnp.asarray([100.0, 0.0, 0.0])}, state)
    assert float(out["w"][0]) == pytest.approx(limit, rel=1e-5)
    assert limit < 20


def test_best_step_survives_checkpoint_rotation(tmp_path):
    ckpt = Checkpoints(tmp_path / "ckpt", keep=2)
    lora, opt = {"a": jnp.ones(3)}, {"m": jnp.zeros(3)}
    for step in (100, 200, 300, 400, 500):
        ckpt.save(step, {"a": jnp.full(3, step)}, opt, {"step": step}, keep=200)
    ckpt.wait()
    assert sorted(ckpt.manager.all_steps()) == [200, 400, 500]
    restored, _, meta = ckpt.restore(200, lora, opt)
    assert meta["step"] == 200
    assert float(restored["a"][0]) == 200
    ckpt.close()
    ckpt = Checkpoints(tmp_path / "plain", keep=3)
    for step in range(1, 7):
        ckpt.save(step, {"a": jnp.ones(2)}, {"m": jnp.zeros(2)}, {})
    ckpt.wait()
    assert sorted(ckpt.manager.all_steps()) == [4, 5, 6]
    ckpt.close()


@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_peak_rate_in_state_matches_the_scheduled_optimizer(name):
    """The optimizer at the unit schedule × the state's peak equals optax's at the full
    schedule (weight decay included), step by step."""
    import optax

    from tjev.train.optim import _muon

    spec = OptimSpec(name=name, lr=3e-4, warmup_steps=2, weight_decay=0.1, clip_mode="none")
    steps = 8
    full = (optax.adamw(schedule(spec, steps), b1=spec.b1, b2=spec.b2, eps=spec.eps,
                        weight_decay=spec.weight_decay)
            if name == "adamw" else _muon(spec, schedule(spec, steps)))  # fmt: skip
    ours = make_optimizer(spec, steps)
    keys = jax.random.split(jax.random.key(0), steps + 1)
    # LoRA factor names: AdamW's weight decay applies to them only (optim.decay_mask)
    params = {"lora_a": jax.random.normal(keys[0], (2, 16, 4)),
              "lora_b": jax.random.normal(keys[0], (3,))}  # fmt: skip
    p1, p2, s1, s2 = params, params, full.init(params), ours.init(params)
    for k in keys[1:]:
        g = jax.tree.map(lambda x, k=k: jax.random.normal(k, x.shape), params)
        u1, s1 = full.update(g, s1, p1)
        u2, s2 = ours.update(g, s2, p2)
        p1, p2 = optax.apply_updates(p1, u1), optax.apply_updates(p2, u2)
    for x, y in zip(jax.tree.leaves(p1), jax.tree.leaves(p2), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-7)


def test_rate_is_not_compiled_in():
    """Two runs that differ only in their rate lower to the same program (one compile, the
    persistent cache shared across a sweep's grid)."""
    import optax

    params = {"a": jnp.ones((2, 16, 4)), "b": jnp.ones((3,))}

    def text(lr):
        tx = make_optimizer(OptimSpec(lr=lr), 100)

        def step(p, s, g):
            u, s = tx.update(g, s, p)
            return optax.apply_updates(p, u), s

        return jax.jit(step).lower(params, tx.init(params), params).as_text()

    assert text(1e-4) == text(3.5e-5)


def _grads(params, key):
    return jax.tree.map(lambda x: jax.random.normal(key, x.shape), params)


def _today(spec: OptimSpec, steps: int):
    """make_optimizer as it was before cooldown_b2 / loraplus_ratio (AdamW, global clip)."""
    import optax

    from tjev.train.optim import decay_mask, scale_by_peak, unit_schedule

    mask = {"mask": decay_mask} if spec.weight_decay else {}
    return optax.chain(
        optax.clip_by_global_norm(spec.clip_norm),
        optax.adamw(unit_schedule(spec, steps), b1=spec.b1, b2=spec.b2, eps=spec.eps,
                    weight_decay=spec.weight_decay, **mask),
        scale_by_peak(spec.lr),
    )  # fmt: skip


LORA = {"lora_a": jnp.full((2, 16, 4), 0.1), "lora_b": jnp.full((2, 4, 8), 0.05)}


@pytest.mark.parametrize("wd", [0.0, 0.1])
def test_cooldown_b2_and_loraplus_off_are_todays_optimizer_bitwise(wd):
    import optax

    spec = OptimSpec(lr=3e-4, warmup_steps=2, weight_decay=wd)
    ours, today = make_optimizer(spec, 10), _today(spec, 10)
    p1 = p2 = LORA
    s1, s2 = ours.init(LORA), today.init(LORA)
    for k in jax.random.split(jax.random.key(0), 10):
        g = _grads(LORA, k)
        u1, s1 = ours.update(g, s1, p1)
        u2, s2 = today.update(g, s2, p2)
        p1, p2 = optax.apply_updates(p1, u1), optax.apply_updates(p2, u2)
    for x, y in zip(jax.tree.leaves((p1, s1)), jax.tree.leaves((p2, s2)), strict=True):
        np.testing.assert_array_equal(x, y)


def test_cooldown_b2_state_has_the_same_structure():
    """A checkpoint of a run without cooldown_b2 restores into one with it (a cooldown
    branch off the stable phase), and the other way round."""
    for wd in (0.0, 0.1):
        spec = OptimSpec(weight_decay=wd, ema_decay=0.99)
        plain = make_optimizer(spec, 10).init(LORA)
        cool = make_optimizer(replace(spec, cooldown_b2=0.9999), 10).init(LORA)
        assert jax.tree.structure(plain) == jax.tree.structure(cool)
        assert [x.shape for x in jax.tree.leaves(plain)] == [x.shape for x in jax.tree.leaves(cool)]
        lora8 = make_optimizer(replace(spec, loraplus_ratio=8.0), 10).init(LORA)
        assert jax.tree.structure(plain) == jax.tree.structure(lora8)


def test_cooldown_b2_is_b2_before_the_cooldown_and_cooldown_b2_inside():
    import optax

    from tjev.train.optim import cooldown_start

    steps = 10
    spec = OptimSpec(lr=1e-3, warmup_steps=2, decay_fraction=0.3, clip_mode="none",
                     cooldown_b2=0.9999)  # fmt: skip
    start = cooldown_start(spec, steps)
    assert start == 7  # the last 3 steps, where the rate decays
    sched = schedule(spec, steps)
    assert float(sched(start - 1)) == pytest.approx(spec.lr)
    assert float(sched(start + 1)) < spec.lr
    tx = make_optimizer(spec, steps)
    state = tx.init(LORA)
    nu = jax.tree.map(jnp.zeros_like, LORA)
    weight = 0.0  # Σ of the second moment's weights: 1 − Π β2
    for t, k in enumerate(jax.random.split(jax.random.key(1), steps)):
        g = _grads(LORA, k)
        beta2 = 0.9999 if t >= start else 0.999
        nu = jax.tree.map(lambda v, x, b=beta2: b * v + (1 - b) * x * x, nu, g)
        weight = beta2 * weight + (1 - beta2)
        _, state = tx.update(g, state, LORA)
        adam = next(s for s in jax.tree.leaves(state, is_leaf=lambda x: isinstance(
            x, optax.ScaleByAdamState)) if isinstance(s, optax.ScaleByAdamState))  # fmt: skip
        for x, y in zip(jax.tree.leaves(adam.nu), jax.tree.leaves(nu), strict=True):
            np.testing.assert_allclose(x, y, rtol=1e-4)
    # bias correction by Π β2: a constant gradient's normalised update stays 1 across the switch
    from tjev.train.optim import scale_by_adam_cooldown_b2

    adam_tx = scale_by_adam_cooldown_b2(0.9, 0.999, 0.9999, 3, eps=0.0)
    s = adam_tx.init(LORA)
    g = jax.tree.map(jnp.ones_like, LORA)
    for _ in range(8):
        u, s = adam_tx.update(g, s)
        for x in jax.tree.leaves(u):
            np.testing.assert_allclose(x, 1.0, rtol=1e-4)


def test_cooldown_b2_equal_to_b2_matches_optax_adam():
    import optax

    from tjev.train.optim import scale_by_adam_cooldown_b2

    ours, theirs = scale_by_adam_cooldown_b2(0.9, 0.999, 0.999, 3, 1e-8), optax.scale_by_adam()
    s1, s2 = ours.init(LORA), theirs.init(LORA)
    for k in jax.random.split(jax.random.key(2), 6):
        g = _grads(LORA, k)
        u1, s1 = ours.update(g, s1)
        u2, s2 = theirs.update(g, s2)
        for x, y in zip(jax.tree.leaves((u1, s1)), jax.tree.leaves((u2, s2)), strict=True):
            np.testing.assert_allclose(x, y, rtol=1e-4)


def test_cooldown_b2_is_adamw_only():
    with pytest.raises(ValueError, match="AdamW only"):
        make_optimizer(OptimSpec(name="muon", cooldown_b2=0.9999), 10)


@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_loraplus_scales_the_b_factors_rate_only(name):
    spec = OptimSpec(name=name, lr=3e-4, warmup_steps=0, clip_mode="none", weight_decay=0.1)
    one, eight = make_optimizer(spec, 10), make_optimizer(replace(spec, loraplus_ratio=8.0), 10)
    s1, s8 = one.init(LORA), eight.init(LORA)
    for k in jax.random.split(jax.random.key(3), 3):
        g = _grads(LORA, k)
        u1, s1 = one.update(g, s1, LORA)
        u8, s8 = eight.update(g, s8, LORA)
        np.testing.assert_array_equal(u8["lora_a"], u1["lora_a"])
        np.testing.assert_allclose(u8["lora_b"], 8.0 * u1["lora_b"], rtol=1e-6)


def test_cooldown_b2_before_the_cooldown_is_optax_adam_bitwise():
    """Up to the cooldown, the same update op for op as optax.scale_by_adam: a branch off a
    run without cooldown_b2 continues it bitwise."""
    import optax

    from tjev.train.optim import scale_by_adam_cooldown_b2

    ours, theirs = scale_by_adam_cooldown_b2(0.9, 0.999, 0.9999, 4, 1e-8), optax.scale_by_adam()
    s1, s2 = ours.init(LORA), theirs.init(LORA)
    step1, step2 = jax.jit(ours.update), jax.jit(theirs.update)
    for k in jax.random.split(jax.random.key(4), 4):
        g = _grads(LORA, k)
        u1, s1 = step1(g, s1)
        u2, s2 = step2(g, s2)
        for x, y in zip(jax.tree.leaves((u1, s1)), jax.tree.leaves((u2, s2)), strict=True):
            np.testing.assert_array_equal(x, y)
