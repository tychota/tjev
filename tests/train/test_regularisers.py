"""The regularisers (loss.*, optim.ema_decay, lora.dropout): at 0 they are today's training
bit for bit; label smoothing is a soft CE against the smoothed target; the KL to the base is
0 while B = 0; the EMA's update math; dropout is a training-only, remat-stable mask; every
gradient finite."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from tjev.config import LossSpec, OptimSpec
from tjev.data.pack import Batch
from tjev.train.objective import (
    hard,
    loss_sums,
    masked_log_softmax,
    smoothed,
    source_table,
)
from tjev.train.optim import EmaState, decay_mask, ema_decay_at, ema_params, make_optimizer


def _batch(seed=0, rows=3, slots=4, labels=5, soft=True) -> Batch:
    rng = np.random.default_rng(seed)
    mask = np.zeros((rows, slots, labels), bool)
    n = rng.integers(2, labels + 1, (rows, slots))
    for r in range(rows):
        for s in range(slots):
            mask[r, s, : n[r, s]] = True
    raw = rng.random((rows, slots, labels)) ** (1 if soft else 8) * mask
    target = (raw / raw.sum(-1, keepdims=True)).astype(np.float32)
    weight = (rng.random((rows, slots)) > 0.2).astype(np.float32)
    zeros = np.zeros((rows, slots), np.int32)
    return Batch(
        tokens=np.zeros((rows, 8), np.int32), segment_ids=np.ones((rows, 8), np.int32),
        positions=np.zeros((rows, 8), np.int32), slots=zeros, label_ids=np.zeros(mask.shape, np.int32),
        label_mask=mask, target=target, weight=weight, type_id=zeros,
        source_id=rng.integers(0, 3, (rows, slots)).astype(np.int32), lang_id=zeros,
        family_id=zeros, index=zeros,
    )  # fmt: skip


def _logits(seed=1, shape=(3, 4, 5)):
    return jnp.asarray(np.random.default_rng(seed).normal(0, 2, shape), jnp.float32)


def _today(logits, batch, brier_weight):
    """The objective before the regularisers (tjev 89a4ee1), verbatim."""
    mask = batch.label_mask
    logp = masked_log_softmax(logits, mask)
    probs = jnp.where(mask, jnp.exp(logp), 0.0)
    target = batch.target
    nll = -jnp.sum(target * logp, axis=-1)
    brier = jnp.sum((probs - target) ** 2, axis=-1)
    loss = nll + brier_weight * brier
    return jnp.sum(batch.weight * loss)


@pytest.mark.parametrize(
    "spec", [None, LossSpec(), LossSpec(smooth_sources=("x*",), kl_sources=("y",))]
)
def test_the_loss_with_every_option_at_zero_is_todays_loss_exactly(spec):
    batch, logits = _batch(), _logits()
    total, sums = loss_sums(logits, batch, brier_weight=0.1, loss=spec,
                            base_logits=_logits(7))  # fmt: skip
    assert np.asarray(total).tobytes() == np.asarray(_today(logits, batch, 0.1)).tobytes()
    assert "kl_base" not in sums


@pytest.mark.parametrize("eps", [0.05, 0.1])
def test_label_smoothing_is_soft_ce_against_the_smoothed_target(eps):
    batch, logits = _batch(soft=False), _logits()
    total, sums = loss_sums(logits, batch, brier_weight=0.1, loss=LossSpec(label_smoothing=eps))
    target = smoothed(jnp.asarray(batch.target), batch.label_mask, eps)
    smooth = dataclasses.replace(batch, target=np.asarray(target))
    np.testing.assert_allclose(total, _today(logits, smooth, 0.1), rtol=1e-6)
    # the logged nll / brier stay against the data's target
    np.testing.assert_array_equal(sums["nll"], loss_sums(logits, batch, brier_weight=0.1)[1]["nll"])
    # a smoothed target is a distribution over the displayed letters only
    t = np.asarray(target)
    np.testing.assert_allclose(t.sum(-1), 1.0, rtol=1e-6)
    assert (t[~batch.label_mask] == 0).all()


def test_label_smoothing_on_selected_sources_only():
    batch, logits = _batch(), _logits()
    on = jnp.asarray(batch.source_id == 1, jnp.float32)
    spec = LossSpec(label_smoothing=0.1)
    _, picked = loss_sums(logits, batch, brier_weight=0.1, loss=spec, smooth_on=on)
    _, every = loss_sums(logits, batch, brier_weight=0.1, loss=spec)
    _, none = loss_sums(logits, batch, brier_weight=0.1)
    only = dataclasses.replace(batch, weight=batch.weight * np.asarray(on))
    rest = dataclasses.replace(batch, weight=batch.weight * (1 - np.asarray(on)))
    want = (loss_sums(logits, only, brier_weight=0.1, loss=spec)[1]["loss"]
            + loss_sums(logits, rest, brier_weight=0.1)[1]["loss"])  # fmt: skip
    np.testing.assert_allclose(picked["loss"], want, rtol=1e-6)
    assert none["loss"] != every["loss"]


def test_hard_targets_train_on_the_argmax():
    batch, logits = _batch(), _logits()
    total, _ = loss_sums(logits, batch, brier_weight=0.1, loss=LossSpec(hard_targets=True))
    onehot = dataclasses.replace(
        batch, target=np.asarray(hard(jnp.asarray(batch.target), batch.label_mask))
    )
    np.testing.assert_allclose(total, _today(logits, onehot, 0.1), rtol=1e-6)
    assert np.asarray(onehot.target).sum(-1).max() == 1.0


def test_soft_ce_and_brier_against_a_soft_target():
    """The objective is the CE against a soft target (minimised at p = target, where it is
    the target's entropy) and the Brier score against it (0 there)."""
    batch = _batch()
    exact = jnp.where(batch.label_mask, jnp.log(jnp.maximum(batch.target, 1e-30)), -1e9)
    _, sums = loss_sums(exact, batch, brier_weight=0.1)
    t, w = batch.target, batch.weight
    entropy = -np.sum(np.where(t > 0, t * np.log(np.maximum(t, 1e-30)), 0), -1)
    np.testing.assert_allclose(sums["nll"], np.sum(w * entropy), rtol=1e-5)
    assert float(sums["brier"]) < 1e-10


def test_kl_to_the_base_is_zero_at_the_base_and_its_gradient_is_finite():
    batch, logits = _batch(), _logits()
    spec = LossSpec(kl_base_weight=0.5)
    _, same = loss_sums(logits, batch, brier_weight=0.1, loss=spec, base_logits=logits)
    assert abs(float(same["kl_base"])) < 1e-6
    base = _logits(9)
    total, sums = loss_sums(logits, batch, brier_weight=0.1, loss=spec, base_logits=base)
    p_b = np.asarray(jax.nn.softmax(jnp.where(batch.label_mask, base, -jnp.inf)))
    p = np.asarray(jax.nn.softmax(jnp.where(batch.label_mask, logits, -jnp.inf)))
    kl = np.sum(np.where(batch.label_mask, p_b * (np.log(p_b + 1e-30) - np.log(p + 1e-30)), 0), -1)
    np.testing.assert_allclose(sums["kl_base"], np.sum(batch.weight * kl), rtol=1e-5)
    np.testing.assert_allclose(total, _today(logits, batch, 0.1) + 0.5 * sums["kl_base"], rtol=1e-6)
    grad = jax.grad(lambda x: loss_sums(x, batch, brier_weight=0.1, loss=spec, base_logits=base)[0])
    g = grad(logits)
    assert bool(jnp.isfinite(g).all())
    # the base side is held constant: no gradient flows into it
    gb = jax.grad(
        lambda b: loss_sums(logits, batch, brier_weight=0.1, loss=spec, base_logits=b)[0]
    )(base)
    assert float(jnp.abs(gb).max()) == 0.0


def test_every_option_has_a_finite_gradient():
    batch, logits = _batch(), _logits()
    spec = LossSpec(label_smoothing=0.1, kl_base_weight=0.1, hard_targets=True)
    on = jnp.asarray(batch.source_id != 0, jnp.float32)
    g = jax.grad(lambda x: loss_sums(x, batch, brier_weight=0.1, loss=spec, smooth_on=on, kl_on=on,
                                     base_logits=_logits(3))[0])(logits)  # fmt: skip
    assert bool(jnp.isfinite(g).all()) and float(jnp.abs(g).max()) > 0


def test_source_selectors_take_patterns_and_mix_blocks():
    names = ["boolq", "gen_weekday", "gen_spelling_error", "xnli_fr", "plumb"]
    assert source_table((), names) is None
    assert source_table(("block:generators",), names).tolist() == [0, 1, 0, 0, 0]
    assert source_table(("block:replay", "*_fr"), names).tolist() == [1, 0, 0, 1, 0]
    assert source_table(("gen_*",), names).tolist() == [0, 1, 1, 0, 0]
    with pytest.raises(ValueError, match="matches no source"):
        source_table(("nothing",), names)


def test_ema_update_math():
    params = {"lora_a": jnp.ones((2, 3)), "lora_b": jnp.zeros((3, 4))}
    spec = OptimSpec(lr=0.1, warmup_steps=0, decay_fraction=0.0, clip_mode="none", ema_decay=0.999)
    tx = make_optimizer(spec, 100)
    state = tx.init(params)
    ema = ema_params(state)
    np.testing.assert_array_equal(ema["lora_a"], params["lora_a"])  # starts at the params
    p, e = params, {k: np.asarray(v, np.float64) for k, v in params.items()}
    rng = np.random.default_rng(0)
    for t in range(30):
        g = {k: jnp.asarray(rng.normal(size=v.shape), jnp.float32) for k, v in p.items()}
        updates, state = tx.update(g, state, p)
        p = optax.apply_updates(p, updates)
        d = min(0.999, (1 + t) / (10 + t))
        e = {k: d * e[k] + (1 - d) * np.asarray(p[k], np.float64) for k in e}
        assert float(ema_decay_at(0.999, jnp.asarray(t))) == pytest.approx(d, rel=1e-6)
    for k in e:
        np.testing.assert_allclose(ema_params(state)[k], e[k], rtol=1e-5, atol=1e-6)
        assert ema_params(state)[k].dtype == jnp.float32
    # off by default: today's chain, no EMA state
    plain = make_optimizer(dataclasses.replace(spec, ema_decay=0.0), 100).init(params)
    assert ema_params(plain) is None
    leaves = jax.tree.leaves(plain, is_leaf=lambda x: isinstance(x, EmaState))
    assert not any(isinstance(x, EmaState) for x in leaves)


def test_weight_decay_applies_to_the_lora_factors_only():
    params = {"proj": {"lora_a": jnp.ones((2, 3)), "lora_b": jnp.ones((3, 2)),
                       "kernel": jnp.ones((2, 2))}}  # fmt: skip
    assert decay_mask(params) == {"proj": {"lora_a": True, "lora_b": True, "kernel": False}}
    spec = OptimSpec(lr=1.0, warmup_steps=0, decay_fraction=0.0, clip_mode="none", weight_decay=0.1)
    tx = make_optimizer(spec, 10)
    zero = jax.tree.map(jnp.zeros_like, params)
    updates, _ = tx.update(zero, tx.init(params), params)
    assert float(updates["proj"]["kernel"].sum()) == 0.0  # a gate_full kernel is not decayed
    np.testing.assert_allclose(updates["proj"]["lora_a"], -0.1)  # −lr·wd·p
