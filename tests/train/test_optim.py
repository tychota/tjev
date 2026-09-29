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
