"""The train step's regularisers on the tiny model: the KL to the base is 0 while B = 0 and
its base ignores the adapters; LoRA dropout is a training-only mask, the same in the remat
recompute, refused by merged layers (the model builds unmerged ones); every option together
trains with finite gradients; at 0 the step is today's."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from tjev.config import ComputeSpec, LoRASpec, LossSpec, OptimSpec
from tjev.data.pack import Batch, stack_batches
from tjev.model import Qwen35, load_hf, stack_layers
from tjev.model.lora import Linear, dropout_scope
from tjev.testing import make_tiny_snapshot
from tjev.train.optim import make_optimizer
from tjev.train.step import forward_logits, make_train_step, split_model

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tiny-step") / "model"
    make_tiny_snapshot(folder)
    return load_hf(folder, "float32")


def _model(snapshot, *, dropout=0.0, **compute):
    config, tensors = snapshot
    lora = LoRASpec(rank=4, alpha=8.0, dropout=dropout)
    return Qwen35(stack_layers(dict(tensors), config), config,
                  ComputeSpec(gdn_chunk=16, **compute), lora, dtype=jnp.float32,
                  rngs=nnx.Rngs(0))  # fmt: skip


def _batch(accum=2, rows=2, length=48, seed=0) -> Batch:
    rng = np.random.default_rng(seed)
    half = length // 2

    def one():
        raw = rng.random((rows, 2, 4)).astype(np.float32)
        return Batch(
            tokens=rng.integers(0, 512, (rows, length)).astype(np.int32),
            segment_ids=np.repeat([[1, 2]], half, 1).repeat(rows, 0).astype(np.int32),
            positions=np.tile(np.arange(half), 2)[None].repeat(rows, 0).astype(np.int32),
            slots=np.asarray([[half - 1, length - 1]] * rows, np.int32),
            label_ids=np.broadcast_to(np.arange(10, 14), (rows, 2, 4)).astype(np.int32).copy(),
            label_mask=np.ones((rows, 2, 4), bool),
            target=raw / raw.sum(-1, keepdims=True),
            weight=np.ones((rows, 2), np.float32),
            type_id=np.zeros((rows, 2), np.int32),
            source_id=np.asarray([[0, 1]] * rows, np.int32),
            lang_id=np.zeros((rows, 2), np.int32),
            family_id=np.zeros((rows, 2), np.int32),
            index=np.full((rows, 2), -1, np.int32),
        )

    parts = [one() for _ in range(accum)]
    return stack_batches(parts)


def _first(batch):
    return jax.tree.map(lambda x: x[0], batch)


def _perturb_b(lora, scale=0.05):
    leaves, treedef = jax.tree_util.tree_flatten_with_path(lora)
    key = jax.random.key(3)
    out = []
    for i, (path, leaf) in enumerate(leaves):
        if "lora_b" in jax.tree_util.keystr(path):
            leaf = scale * jax.random.normal(jax.random.fold_in(key, i), leaf.shape, leaf.dtype)
        out.append(leaf)
    return jax.tree.unflatten(treedef, out)


def _step(graphdef, lora, **kw):
    tx = make_optimizer(OptimSpec(lr=1e-3, warmup_steps=0), 10)
    return tx, jax.jit(make_train_step(graphdef, tx, brier_weight=0.1, slots_per_step=8.0,
                                       optim=OptimSpec(lr=1e-3, warmup_steps=0), **kw))  # fmt: skip


def test_the_kl_to_the_base_is_zero_while_b_is_zero(snapshot):
    graphdef, lora, frozen = split_model(_model(snapshot))
    batch = _batch()
    mb = _first(batch)
    np.testing.assert_array_equal(forward_logits(graphdef, lora, frozen, mb, base=True),
                                  forward_logits(graphdef, lora, frozen, mb))  # fmt: skip
    tx, step = _step(graphdef, lora, loss=LossSpec(kl_base_weight=0.1))
    _, _, metrics = step(lora, tx.init(lora), frozen, batch)
    assert float(metrics["kl_base"]) == pytest.approx(0.0, abs=1e-7)
    trained = _perturb_b(lora)
    base = forward_logits(graphdef, trained, frozen, mb, base=True)
    np.testing.assert_allclose(
        base, forward_logits(graphdef, lora, frozen, mb), rtol=1e-6, atol=1e-6
    )
    _, _, metrics = step(trained, tx.init(trained), frozen, batch)
    assert float(metrics["kl_base"]) > 0


def test_kl_on_selected_sources_only(snapshot):
    graphdef, lora, frozen = split_model(_model(snapshot))
    trained, batch = _perturb_b(lora), _batch()
    spec = LossSpec(kl_base_weight=1.0)
    losses = {}
    for name, table in (("all", None), ("none", np.zeros(2, np.float32)),
                        ("one", np.asarray([0, 1], np.float32))):  # fmt: skip
        tx, step = _step(graphdef, trained, loss=spec, kl_table=table)
        _, _, m = step(trained, tx.init(trained), frozen, batch)
        losses[name] = float(m["loss"])
    tx, step = _step(graphdef, trained)
    _, _, plain = step(trained, tx.init(trained), frozen, batch)
    assert losses["none"] == pytest.approx(float(plain["loss"]), rel=1e-6)
    assert losses["none"] < losses["one"] < losses["all"]


def test_options_at_zero_are_todays_step(snapshot):
    graphdef, lora, frozen = split_model(_model(snapshot))
    trained, batch = _perturb_b(lora), _batch()
    tx, today = _step(graphdef, trained)
    _, explicit = _step(graphdef, trained, loss=LossSpec(), dropout=0.0)
    a = today(trained, tx.init(trained), frozen, batch)
    b = explicit(trained, tx.init(trained), frozen, batch)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


def test_lora_dropout_is_a_training_only_mask(snapshot):
    graphdef, lora, frozen = split_model(_model(snapshot, dropout=0.3))
    trained, mb = _perturb_b(lora), _first(_batch())
    plain = forward_logits(graphdef, trained, frozen, mb)
    key = jax.random.key(5)
    dropped = forward_logits(graphdef, trained, frozen, mb, dropout=(key, 0.3))
    assert float(jnp.abs(dropped - plain).max()) > 1e-4
    np.testing.assert_array_equal(dropped, forward_logits(graphdef, trained, frozen, mb,
                                                          dropout=(key, 0.3)))  # fmt: skip
    # rate 0 or no key: exactly the eval forward
    np.testing.assert_array_equal(plain, forward_logits(graphdef, trained, frozen, mb,
                                                        dropout=(key, 0.0)))  # fmt: skip
    # B = 0: the adapter adds nothing, dropped or not
    np.testing.assert_array_equal(forward_logits(graphdef, lora, frozen, mb, dropout=(key, 0.3)),
                                  forward_logits(graphdef, lora, frozen, mb))  # fmt: skip


def test_lora_dropout_masks_are_the_same_in_the_remat_recompute(snapshot):
    """The gradient under full remat equals the one without remat for the same key: the
    recompute replays the forward's masks."""
    batch = _batch()
    grads = {}
    for remat in ("none", "full"):
        graphdef, lora, frozen = split_model(_model(snapshot, dropout=0.2, remat=remat))
        trained = _perturb_b(lora)
        tx, step = _step(graphdef, trained, dropout=0.2)
        new, _, metrics = step(trained, tx.init(trained), frozen, batch, jax.random.key(1))
        assert bool(jnp.isfinite(metrics["grad_norm"]))
        grads[remat] = new
    # the same masks: equal up to the recompute's float reordering (~1e-6); another key's
    # masks move the update by orders of magnitude more
    other, _, _ = step(trained, tx.init(trained), frozen, batch, jax.random.key(2))
    gap = max(float(jnp.abs(x - y).max())
              for x, y in zip(jax.tree.leaves(grads["full"]), jax.tree.leaves(other), strict=True))  # fmt: skip
    assert gap > 1e-4
    for x, y in zip(jax.tree.leaves(grads["none"]), jax.tree.leaves(grads["full"]), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-5)


def test_merged_lora_falls_back_to_unmerged_with_dropout(snapshot):
    merged = _model(snapshot, lora_impl="merged")
    fallback = _model(snapshot, dropout=0.1, lora_impl="merged")
    flags = lambda m: {x.merged for _, x in nnx.iter_modules(m) if isinstance(x, Linear) and x.rank}  # noqa: E731
    assert flags(merged) == {True}
    assert flags(fallback) == {False}
    mb = _first(_batch())
    with dropout_scope(jax.random.key(0), 0.1), pytest.raises(ValueError, match="merged"):
        merged.hidden(mb.tokens, mb.segment_ids, mb.positions)


def test_every_option_together_trains_with_finite_gradients(snapshot):
    graphdef, lora, frozen = split_model(_model(snapshot, dropout=0.1))
    batch = _batch()
    spec = LossSpec(label_smoothing=0.1, kl_base_weight=0.1, hard_targets=False)
    optim = OptimSpec(lr=1e-3, warmup_steps=0, ema_decay=0.99, weight_decay=0.05)
    tx = make_optimizer(optim, 10)
    step = jax.jit(make_train_step(graphdef, tx, brier_weight=0.1, slots_per_step=8.0, optim=optim,
                                   loss=spec, smooth_table=np.asarray([1, 0], np.float32),
                                   dropout=0.1), donate_argnums=(0, 1))  # fmt: skip
    opt = tx.init(lora)
    for i in range(3):
        lora, opt, metrics = step(
            lora, opt, frozen, batch, jax.random.fold_in(jax.random.key(0), i)
        )
        assert float(metrics["update_rejected"]) == 0.0
        assert all(bool(jnp.isfinite(v).all()) for v in jax.tree.leaves(metrics))
    assert float(metrics["kl_base"]) > 0


def test_weight_decay_reaches_every_adapter_of_the_real_model(snapshot):
    """decay_mask on the NNX adapter state: every LoRA factor is decayed, a gate_full
    kernel is not."""
    from tjev.train.optim import decay_mask

    _, lora, _ = split_model(_model(snapshot))
    assert all(jax.tree.leaves(decay_mask(lora)))
    config, tensors = snapshot
    full = Qwen35(stack_layers(dict(tensors), config), config, ComputeSpec(gdn_chunk=16),
                  LoRASpec(rank=4, alpha=8.0, gate_full=True), dtype=jnp.float32,
                  rngs=nnx.Rngs(0))  # fmt: skip
    _, lora, _ = split_model(full)
    flags = jax.tree.leaves(decay_mask(lora))
    assert any(flags) and not all(flags)


def test_cogcalib_running_threshold_and_teacher_mix_in_the_step(snapshot):
    """Step 1 gates nothing and sets the threshold to its slots' NLL quantile (from the
    optimizer state, no extra pass); step 2 gates on it. The teacher mixture trains with
    finite gradients; at weight 0 the step with a teacher-carrying batch is today's."""
    from tjev.train.objective import loss_sums
    from tjev.train.optim import cogcalib_state

    graphdef, lora, frozen = split_model(_model(snapshot))
    trained = _perturb_b(lora, 0.5)
    batch = _batch(accum=3)
    # slot 0 of every row: the teacher agrees with the gold letter; slot 1: it is sure of another
    gold = batch.target.argmax(-1)
    pick = np.where(np.arange(2) == 0, gold, (gold + 1) % 4)
    teacher = (np.arange(4) == pick[..., None]).astype(np.float32)
    batch = dataclasses.replace(batch, teacher=teacher)
    spec = LossSpec(cogcalib_weight=0.1, cogcalib_quantile=0.3, cogcalib_ema=0.5,
                    teacher_mix=0.5, teacher_gate=0.4)  # fmt: skip
    optim = OptimSpec(lr=1e-3, warmup_steps=0)
    tx = make_optimizer(optim, 10, spec)
    step = jax.jit(make_train_step(graphdef, tx, brier_weight=0.1, slots_per_step=8.0,
                                   optim=optim, loss=spec))  # fmt: skip
    opt = tx.init(trained)
    nll = []
    for i in range(3):
        mb = jax.tree.map(lambda x, i=i: x[i], batch)
        logits = forward_logits(graphdef, trained, frozen, mb)
        nll.append(loss_sums(logits, mb, brier_weight=0.1, slot_nll=True)[1]["slot_nll"])
    want = float(jnp.nanquantile(jnp.stack(nll), 0.3))
    new, opt1, m1 = step(trained, opt, frozen, batch)
    assert float(m1["cogcalib_frac"]) == 0.0 and float(m1["cogcalib_term"]) == 0.0
    state = cogcalib_state(opt1)
    assert int(state.count) == 1
    assert float(state.threshold) == pytest.approx(want, rel=1e-5)
    assert float(m1["teacher_frac"]) == 0.5
    _, opt2, m2 = step(new, opt1, frozen, batch)
    assert 0.0 < float(m2["cogcalib_frac"]) < 1.0 and float(m2["cogcalib_term"]) < 0.0
    assert float(m2["cogcalib_threshold"]) == pytest.approx(float(state.threshold))
    assert int(cogcalib_state(opt2).count) == 2
    for m in (m1, m2):
        assert float(m["update_rejected"]) == 0.0
        assert all(bool(jnp.isfinite(v).all()) for v in jax.tree.leaves(m))
    # weight 0 with the teacher-carrying batch: today's step, bit for bit
    tx0, today = _step(graphdef, trained)
    _, off = _step(graphdef, trained, loss=LossSpec(teacher_gate=0.4, cogcalib_quantile=0.3))
    a = today(trained, tx0.init(trained), frozen, dataclasses.replace(batch, teacher=None))
    b = off(trained, tx0.init(trained), frozen, batch)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


def test_cogcalib_base_gate_in_the_step(snapshot):
    """cogcalib_gate=base: the step runs the base forward (adapters at 0, no backward) and the
    running threshold is the quantile of the BASE model's slot NLLs, whatever the adapters."""
    from tjev.train.objective import loss_sums
    from tjev.train.optim import cogcalib_state

    graphdef, lora, frozen = split_model(_model(snapshot))
    trained = _perturb_b(lora, 0.5)
    batch = _batch(accum=3)
    spec = LossSpec(cogcalib_weight=0.3, cogcalib_quantile=0.6, cogcalib_ema=0.5,
                    cogcalib_gate="base")  # fmt: skip
    optim = OptimSpec(lr=1e-3, warmup_steps=0)
    tx = make_optimizer(optim, 10, spec)
    step = jax.jit(make_train_step(graphdef, tx, brier_weight=0.1, slots_per_step=8.0,
                                   optim=optim, loss=spec))  # fmt: skip
    nll = []
    for i in range(3):
        mb = jax.tree.map(lambda x, i=i: x[i], batch)
        base = forward_logits(graphdef, trained, frozen, mb, base=True)
        nll.append(loss_sums(base, mb, brier_weight=0.1, slot_nll=True)[1]["slot_nll"])
    want = float(jnp.nanquantile(jnp.stack(nll), 0.6))
    new, opt1, _ = step(trained, tx.init(trained), frozen, batch)
    assert float(cogcalib_state(opt1).threshold) == pytest.approx(want, rel=1e-5)
    _, opt2, m2 = step(new, opt1, frozen, batch)
    assert 0.0 < float(m2["cogcalib_frac"]) < 1.0 and float(m2["cogcalib_term"]) < 0.0
    # the base NLLs do not move with the adapters: the quantile of step 2 is step 1's
    assert float(cogcalib_state(opt2).threshold) == pytest.approx(want, rel=1e-5)
    assert all(bool(jnp.isfinite(v).all()) for v in jax.tree.leaves(m2))
