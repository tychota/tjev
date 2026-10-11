"""The calibration options of the loss (loss.cogcalib_*, loss.teacher_*): at weight 0 they
are today's loss bit for bit; CogCalib's gate picks the slots with NLL ≤ t and its entropy
penalty pushes them away from p → 1; the teacher mixture applies only where the teacher puts
≥ the gate on a non-gold letter, and is a distribution; the running quantile threshold's
state; every gradient finite."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tjev.config import LossSpec, OptimSpec
from tjev.data.pack import Batch
from tjev.train.objective import (
    cogcalib_gate,
    entropy,
    loss_sums,
    masked_log_softmax,
    teacher_gate,
    teacher_mixed,
)
from tjev.train.optim import (
    CogCalibState,
    cogcalib_state,
    cogcalib_update,
    make_optimizer,
    with_cogcalib,
)


def _batch(seed=0, rows=3, slots=4, labels=5, soft=False, teacher=True) -> Batch:
    rng = np.random.default_rng(seed)
    mask = np.zeros((rows, slots, labels), bool)
    n = rng.integers(2, labels + 1, (rows, slots))
    for r in range(rows):
        for s in range(slots):
            mask[r, s, : n[r, s]] = True
    raw = rng.random((rows, slots, labels)) ** (1 if soft else 8) * mask
    if not soft:  # one-hot gold
        raw = (raw == raw.max(-1, keepdims=True)) * mask
    target = (raw / raw.sum(-1, keepdims=True)).astype(np.float32)
    traw = rng.random((rows, slots, labels)) ** 4 * mask
    tdist = (traw / traw.sum(-1, keepdims=True)).astype(np.float32)
    tdist[0, 0] = 0.0  # a slot without a teacher
    weight = (rng.random((rows, slots)) > 0.2).astype(np.float32)
    zeros = np.zeros((rows, slots), np.int32)
    return Batch(
        tokens=np.zeros((rows, 8), np.int32), segment_ids=np.ones((rows, 8), np.int32),
        positions=np.zeros((rows, 8), np.int32), slots=zeros, label_ids=np.zeros(mask.shape, np.int32),
        label_mask=mask, target=target, weight=weight, type_id=zeros, source_id=zeros,
        lang_id=zeros, family_id=zeros, index=zeros, teacher=tdist if teacher else None,
    )  # fmt: skip


def _logits(seed=1, shape=(3, 4, 5), scale=2.0):
    return jnp.asarray(np.random.default_rng(seed).normal(0, scale, shape), jnp.float32)


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


def _slot_nll(logits, batch):
    logp = masked_log_softmax(logits, batch.label_mask)
    return -jnp.sum(batch.target * logp, -1)


@pytest.mark.parametrize("spec", [
    LossSpec(cogcalib_threshold=0.5, teacher_gate=0.3),
    LossSpec(cogcalib_quantile=0.3, cogcalib_ema=0.5),
    LossSpec(cogcalib_weight=0.0, cogcalib_threshold=100.0, teacher_mix=0.0),
])  # fmt: skip
@pytest.mark.parametrize("teacher", [True, False])
def test_weight_zero_is_todays_loss_bit_for_bit(spec, teacher):
    batch, logits = _batch(teacher=teacher), _logits()
    total, sums = loss_sums(logits, batch, brier_weight=0.1, loss=spec,
                            cogcalib_threshold=jnp.float32(10.0))  # fmt: skip
    assert np.asarray(total).tobytes() == np.asarray(_today(logits, batch, 0.1)).tobytes()
    assert not {"cogcalib", "cogcalib_on", "teacher_on"} & set(sums)
    grads = jax.grad(lambda x: loss_sums(x, batch, brier_weight=0.1, loss=spec)[0])(logits)
    want = jax.grad(lambda x: _today(x, batch, 0.1))(logits)
    assert np.asarray(grads).tobytes() == np.asarray(want).tobytes()


def test_the_options_need_a_threshold_and_valid_ranges():
    with pytest.raises(ValueError, match="cogcalib_threshold"):
        LossSpec(cogcalib_weight=0.1)
    with pytest.raises(ValueError):
        LossSpec(cogcalib_quantile=1.0)
    with pytest.raises(ValueError):
        LossSpec(teacher_mix=1.5)
    assert LossSpec(cogcalib_weight=0.1, cogcalib_quantile=0.3).cogcalib_adaptive
    assert not LossSpec(cogcalib_weight=0.1, cogcalib_threshold=0.2).cogcalib_adaptive
    assert not LossSpec(cogcalib_quantile=0.3).cogcalib_adaptive  # weight 0: off


@pytest.mark.parametrize("t", [0.05, 0.3, 1.0])
def test_cogcalib_gates_the_slots_with_nll_at_most_t(t):
    batch, logits = _batch(), _logits(scale=4.0)
    nll = np.asarray(_slot_nll(logits, batch))
    on = np.asarray(cogcalib_gate(jnp.asarray(nll), t))
    np.testing.assert_array_equal(on, (nll <= t).astype(np.float32))
    spec = LossSpec(cogcalib_weight=0.1, cogcalib_threshold=t)
    total, sums = loss_sums(logits, batch, brier_weight=0.1, loss=spec)
    w = batch.weight
    np.testing.assert_allclose(sums["cogcalib_on"], np.sum(w * on), rtol=0, atol=0)
    logp = masked_log_softmax(logits, batch.label_mask)
    probs = jnp.where(batch.label_mask, jnp.exp(logp), 0.0)
    h = np.asarray(entropy(logp, probs))
    penalty = np.sum(w * 0.1 * on * -h)
    np.testing.assert_allclose(sums["cogcalib"], penalty, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(total, _today(logits, batch, 0.1) + penalty, rtol=1e-6)
    # the logged nll stays the data target's
    np.testing.assert_array_equal(sums["nll"], loss_sums(logits, batch, brier_weight=0.1)[1]["nll"])


def test_a_traced_threshold_overrides_the_configured_one():
    batch, logits = _batch(), _logits(scale=4.0)
    spec = LossSpec(cogcalib_weight=0.1, cogcalib_quantile=0.3)
    _, none = loss_sums(logits, batch, brier_weight=0.1, loss=spec,
                        cogcalib_threshold=jnp.float32(-jnp.inf))  # fmt: skip
    _, every = loss_sums(logits, batch, brier_weight=0.1, loss=spec,
                         cogcalib_threshold=jnp.float32(jnp.inf))  # fmt: skip
    assert float(none["cogcalib_on"]) == 0.0 and float(none["cogcalib"]) == 0.0
    assert float(every["cogcalib_on"]) == float(np.sum(batch.weight))


def test_the_entropy_penalty_pushes_a_known_slot_away_from_certainty():
    """−H(p) decreases along its negative gradient: the gold logit goes down, the others up,
    and the entropy rises. Ungated slots get no gradient from it."""
    mask = np.ones((1, 2, 4), bool)
    target = np.zeros((1, 2, 4), np.float32)
    target[..., 0] = 1.0
    zeros = np.zeros((1, 2), np.int32)
    batch = Batch(tokens=np.zeros((1, 4), np.int32), segment_ids=np.ones((1, 4), np.int32),
                  positions=np.zeros((1, 4), np.int32), slots=zeros,
                  label_ids=np.zeros(mask.shape, np.int32), label_mask=mask, target=target,
                  weight=np.ones((1, 2), np.float32), type_id=zeros, source_id=zeros,
                  lang_id=zeros, family_id=zeros, index=zeros)  # fmt: skip
    # slot 0 confident and right (nll ~0.07: known), slot 1 unsure (nll ~1.2: not known)
    logits = jnp.asarray([[[4.0, 1.0, 0.5, 0.0], [0.5, 0.0, 0.8, 0.2]]], jnp.float32)
    spec = LossSpec(cogcalib_weight=1.0, cogcalib_threshold=0.5)

    def penalty(x):
        return loss_sums(x, batch, brier_weight=0.1, loss=spec)[1]["cogcalib"]

    g = np.asarray(jax.grad(penalty)(logits))
    assert g[0, 0, 0] > 0 and (g[0, 0, 1:] < 0).all()  # descent: gold down, others up
    np.testing.assert_array_equal(g[0, 1], 0.0)  # not gated: no gradient
    # with the CE: the penalty only reduces the push toward certainty on the known slot
    full = np.asarray(
        jax.grad(lambda x: loss_sums(x, batch, brier_weight=0.1, loss=spec)[0])(logits)
    )
    plain = np.asarray(jax.grad(lambda x: _today(x, batch, 0.1))(logits))
    assert full[0, 0, 0] > plain[0, 0, 0]
    np.testing.assert_allclose(full[0, 1], plain[0, 1], rtol=1e-6)
    step = logits - 0.1 * g
    h = lambda x: entropy(masked_log_softmax(x, mask), jax.nn.softmax(x))[0, 0]  # noqa: E731
    assert float(h(step)) > float(h(logits))


def test_the_teacher_gate_fires_only_on_confident_disagreement():
    mask = np.ones((5, 3), bool)
    gold = np.zeros((5, 3), np.float32)
    gold[:, 0] = 1.0
    teacher = np.asarray([
        [0.2, 0.7, 0.1],  # disagrees at 0.7: fires
        [0.4, 0.6, 0.0],  # exactly the gate: fires
        [0.45, 0.55, 0.0],  # disagrees below the gate
        [0.9, 0.05, 0.05],  # agrees
        [0.0, 0.0, 0.0],  # no teacher
    ], np.float32)  # fmt: skip
    on = np.asarray(teacher_gate(jnp.asarray(gold), jnp.asarray(teacher), mask, 0.6))
    np.testing.assert_array_equal(on, [1, 1, 0, 0, 0])
    mixed, on2 = teacher_mixed(jnp.asarray(gold), jnp.asarray(teacher), mask, mix=0.5, gate=0.6)
    mixed = np.asarray(mixed)
    np.testing.assert_array_equal(on2, on)
    np.testing.assert_allclose(mixed[0], [0.6, 0.35, 0.05], rtol=1e-6)
    np.testing.assert_allclose(mixed[1], [0.7, 0.3, 0.0], rtol=1e-6)
    np.testing.assert_array_equal(mixed[2:], gold[2:])  # the gold target unchanged
    # a teacher mass on a letter that is not displayed never counts
    hidden = np.asarray([[True, True, False]])
    off = teacher_gate(jnp.asarray(gold[:1]), jnp.asarray([[0.3, 0.0, 0.7]]), hidden, 0.6)
    assert float(off[0]) == 0.0


@pytest.mark.parametrize("soft", [False, True])
def test_teacher_targets_are_distributions_and_the_loss_is_the_soft_ce_against_them(soft):
    batch, logits = _batch(soft=soft), _logits()
    spec = LossSpec(teacher_mix=0.5, teacher_gate=0.4)
    total, sums = loss_sums(logits, batch, brier_weight=0.1, loss=spec)
    mixed, on = teacher_mixed(jnp.asarray(batch.target), jnp.asarray(batch.teacher),
                              batch.label_mask, mix=0.5, gate=0.4)  # fmt: skip
    mixed, on = np.asarray(mixed), np.asarray(on)
    assert 0 < on.sum() < on.size and on[0, 0] == 0  # some fire; the teacher-less slot never
    np.testing.assert_allclose(mixed.sum(-1), 1.0, rtol=1e-6)
    assert (mixed >= 0).all() and (mixed[~batch.label_mask] == 0).all()
    np.testing.assert_array_equal(mixed[on == 0], batch.target[on == 0])
    np.testing.assert_allclose(sums["teacher_on"], np.sum(batch.weight * on))
    want = _today(logits, dataclasses.replace(batch, target=mixed), 0.1)
    np.testing.assert_allclose(total, want, rtol=1e-6)
    np.testing.assert_array_equal(sums["nll"], loss_sums(logits, batch, brier_weight=0.1)[1]["nll"])


def test_teacher_mix_without_a_teacher_in_the_batch_is_refused():
    with pytest.raises(ValueError, match="teacher"):
        loss_sums(_logits(), _batch(teacher=False), brier_weight=0.1,
                  loss=LossSpec(teacher_mix=0.5))  # fmt: skip


def test_every_option_together_has_finite_gradients_even_at_extreme_logits():
    batch = _batch()
    spec = LossSpec(cogcalib_weight=0.1, cogcalib_threshold=0.3, teacher_mix=0.5,
                    teacher_gate=0.3, label_smoothing=0.1, hard_targets=True)  # fmt: skip
    for scale in (1.0, 30.0, 300.0):
        logits = _logits(scale=scale)
        total, sums = loss_sums(logits, batch, brier_weight=0.1, loss=spec)
        g = jax.grad(lambda x: loss_sums(x, batch, brier_weight=0.1, loss=spec)[0])(logits)
        assert bool(jnp.isfinite(total)) and bool(jnp.isfinite(g).all())
        assert all(bool(jnp.isfinite(v)) for v in sums.values())


def test_the_slot_nlls_for_the_running_quantile():
    batch, logits = _batch(), _logits()
    _, sums = loss_sums(logits, batch, brier_weight=0.1, slot_nll=True)
    got = np.asarray(sums["slot_nll"])
    want = np.asarray(_slot_nll(logits, batch))
    real = batch.weight > 0
    np.testing.assert_array_equal(got[real], want[real])
    assert np.isnan(got[~real]).all()


def test_the_running_threshold_lives_in_the_optimizer_state():
    params = {"a": jnp.ones((2, 2))}
    plain = make_optimizer(OptimSpec(), 10)
    assert cogcalib_state(plain.init(params)) is None
    assert cogcalib_state(make_optimizer(OptimSpec(), 10, LossSpec()).init(params)) is None
    tx = make_optimizer(OptimSpec(), 10, LossSpec(cogcalib_weight=0.1, cogcalib_quantile=0.3))
    opt = tx.init(params)
    state = cogcalib_state(opt)
    assert isinstance(state, CogCalibState) and int(state.count) == 0
    # the stage passes the updates through unchanged
    grads = {"a": jnp.full((2, 2), 0.5)}
    want, _ = plain.update(grads, plain.init(params), params)
    got, new = tx.update(grads, opt, params)
    np.testing.assert_array_equal(got["a"], want["a"])
    # the EMA: the first quantile sets it, later ones fold in; a NaN one changes nothing
    s1 = cogcalib_update(state, jnp.float32(0.4), 0.9)
    assert float(s1.threshold) == pytest.approx(0.4) and int(s1.count) == 1
    s2 = cogcalib_update(s1, jnp.float32(0.2), 0.9)
    assert float(s2.threshold) == pytest.approx(0.9 * 0.4 + 0.1 * 0.2) and int(s2.count) == 2
    s3 = cogcalib_update(s2, jnp.float32(jnp.nan), 0.9)
    assert float(s3.threshold) == float(s2.threshold) and int(s3.count) == 2
    replaced = with_cogcalib(new, s2)
    assert float(cogcalib_state(replaced).threshold) == float(s2.threshold)
    assert jax.tree.structure(replaced) == jax.tree.structure(opt)


def test_base_gate_uses_the_base_models_nll_and_keeps_the_penalty_on_the_trained_model():
    """cogcalib_gate=base: the gate compares the BASE model's slot NLL (the paper's fixed
    familiarity) with t; the penalty is still the trained model's −H; without base logits
    it refuses; the running quantile then reads the base NLLs."""
    batch = _batch(seed=3)
    logits, base = _logits(seed=4), _logits(seed=5)
    t = float(np.median(np.asarray(_slot_nll(base, batch))))
    spec = LossSpec(cogcalib_weight=0.3, cogcalib_threshold=t, cogcalib_gate="base")
    _, sums = loss_sums(logits, batch, brier_weight=0.0, loss=spec, base_logits=base,
                        slot_nll=True)  # fmt: skip
    on = np.asarray(cogcalib_gate(_slot_nll(base, batch), t))
    w = np.asarray(batch.weight)
    assert np.isclose(float(sums["cogcalib_on"]), float((w * on).sum()))
    logp = masked_log_softmax(logits, batch.label_mask)
    probs = jnp.where(batch.label_mask, jnp.exp(logp), 0.0)
    pen = 0.3 * on * -np.asarray(entropy(logp, probs))
    assert np.isclose(float(sums["cogcalib"]), float((w * pen).sum()), atol=1e-5)
    base_nll = np.where(w > 0, np.asarray(_slot_nll(base, batch)), np.nan)
    assert np.allclose(np.asarray(sums["slot_nll"]), base_nll, equal_nan=True)
    # the running gate on the same inputs differs (it reads the trained model's NLL)
    running = dataclasses.replace(spec, cogcalib_gate="running")
    _, sums_r = loss_sums(logits, batch, brier_weight=0.0, loss=running, base_logits=base)
    assert not np.isclose(
        float(sums_r["cogcalib_on"]), float(sums["cogcalib_on"])
    ) or not np.allclose(np.asarray(cogcalib_gate(_slot_nll(logits, batch), t)), on)
    with pytest.raises(ValueError, match="base logits"):
        loss_sums(logits, batch, brier_weight=0.0, loss=spec)
    with pytest.raises(ValueError, match="cogcalib_gate"):
        LossSpec(cogcalib_weight=0.3, cogcalib_quantile=0.6, cogcalib_gate="other")
