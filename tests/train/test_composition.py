"""Where the perf features meet: merged LoRA with the XLA-side flags (gdn_norm=flat,
flat_rows, head_rows=letters), the per-bucket remat (minimal_states,
compute.remat_by_bucket), best-fit packing and the named per-bucket programs. The all-in
configuration computes what the reference one does (unmerged LoRA, the per-head norm, the
full head, core_states): the same loss and LoRA gradients, the same run, within fp32
rounding."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from tjev.config import ComputeSpec, LoRASpec, load_config
from tjev.data.item import parse_item, write_jsonl
from tjev.model import Qwen35, load_hf, stack_layers
from tjev.testing import make_tiny_snapshot, synthetic_batch, tiny_items
from tjev.train.checkpoint import Checkpoints
from tjev.train.loop import train
from tjev.train.optim import make_optimizer
from tjev.train.step import jit_train_step, make_train_step, split_model

pytestmark = pytest.mark.slow
LETTER_IDS = (40, 41, 42, 43, 77)  # stand-ins for the answer letters' token ids
REFERENCE = {"lora_impl": "unmerged", "gdn_norm": "heads", "remat": "core_states"}
ALL_IN = {"lora_impl": "merged", "gdn_norm": "flat", "head_rows": "letters",
          "remat": "minimal_states"}  # fmt: skip


@pytest.fixture(scope="module")
def aligned(tmp_path_factory):
    """Lane-aligned widths and an untied head (the 9B layout): head_rows=letters slices
    the head."""
    root = tmp_path_factory.mktemp("compose")
    make_tiny_snapshot(root / "model", hidden_size=128, intermediate_size=256,
                       tie_word_embeddings=False)  # fmt: skip
    write_jsonl(root / "train.jsonl", [parse_item(r) for r in tiny_items(300, seed=1)])
    write_jsonl(root / "val.jsonl", [parse_item(r) for r in tiny_items(60, seed=2)])
    return root


def _model(root, **compute):
    config, tensors = load_hf(root / "model", "float32")
    return Qwen35(stack_layers(dict(tensors), config), config,
                  ComputeSpec(gdn_chunk=16, **compute), LoRASpec(rank=4, alpha=8.0),
                  dtype=jnp.float32, rngs=nnx.Rngs(0), head_rows=LETTER_IDS)  # fmt: skip


def _packed(seed=0):
    """Two 64-token rows: segments of 23 + 30 tokens and 11 padding tokens; one segment."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(1, 512, (2, 64))
    seg = np.zeros((2, 64), np.int32)
    seg[0, :23], seg[0, 23:53], seg[1] = 1, 2, 1
    pos = np.zeros((2, 64), np.int32)
    pos[0, :23], pos[0, 23:53], pos[1] = np.arange(23), np.arange(30), np.arange(64)
    return jnp.asarray(tokens), jnp.asarray(seg), jnp.asarray(pos)


def _loss_and_grads(root, **compute):
    model = _model(root, **compute)
    g, lora, frozen = nnx.split(model, nnx.LoRAParam, ...)
    leaves, treedef = jax.tree.flatten(lora)
    lora = jax.tree.unflatten(treedef, [  # non-zero B: gradients reach A too
        x + 0.01 * jax.random.normal(jax.random.fold_in(jax.random.key(7), i), x.shape)
        for i, x in enumerate(leaves)
    ])  # fmt: skip
    tokens, seg, pos = _packed()
    slots = jnp.asarray([[5, 40], [0, 63]])
    labels = jnp.broadcast_to(jnp.asarray(LETTER_IDS), (2, 2, len(LETTER_IDS)))

    def loss(lo):
        m = nnx.merge(g, lo, frozen)
        h = m.hidden(tokens, seg, pos)
        letters = jax.nn.log_softmax(m.label_logits(h, slots, labels), axis=-1)
        return -jnp.sum(letters[..., 0]) + 1e-3 * jnp.sum(
            jnp.where((seg > 0)[..., None], h, 0) ** 2
        )

    value, grads = jax.jit(jax.value_and_grad(loss))(lora)
    return float(value), jax.tree.leaves(grads)


@pytest.mark.parametrize(
    "compute",
    [ALL_IN, {**ALL_IN, "flat_rows": True, "gdn_input_mask": False},
     {**ALL_IN, "remat": "core_states", "remat_by_bucket": {"64": "minimal_states"}}],
    ids=["all-in", "all-in+flat_rows+nomask", "all-in-by-bucket"],
)  # fmt: skip
def test_all_in_equals_the_reference(aligned, compute):
    """Merged LoRA + flat GDN norm + letters head + minimal_states (directly or as this
    bucket's policy): the reference's loss and LoRA gradients in fp32."""
    want_loss, want = _loss_and_grads(aligned, **REFERENCE)
    got_loss, got = _loss_and_grads(aligned, **compute)
    assert abs(got_loss - want_loss) / abs(want_loss) < 1e-6
    for a, b in zip(got, want, strict=True):
        err = np.max(np.abs(np.asarray(a) - np.asarray(b))) / (np.max(np.abs(b)) + 1e-12)
        assert err < 1e-4, err


def _run_config(root, name, steps, compute: dict):
    return load_config(overrides=[
        f"name={name}", f"output={root / 'runs'}", f"model.path={root / 'model'}",
        "model.dtype=float32", f"data.train=[{root / 'train.jsonl'}]",
        f"data.validation={root / 'val.jsonl'}", f"train.steps={steps}",
        "train.seq_buckets=[384,1024]", "train.microbatch_tokens=3072",
        "train.tokens_per_step=6144", "train.packing=best_fit", "train.packing_bins=2",
        "train.max_segments=4", "train.log_every=1", "train.quick_eval_every=0",
        f"train.checkpoint_every={steps}", "train.checkpoint_secs=0", "compute.gdn_chunk=16",
        "lora.rank=4", "optim.lr=3e-3", "optim.warmup_steps=1", "log.tensorboard=false",
        *(f"compute.{k}={json.dumps(v) if isinstance(v, dict) else v}"
          for k, v in compute.items()),
    ])  # fmt: skip


def test_all_in_run_matches_the_reference_run(aligned, capsys):
    """The training loop with best-fit packing over two buckets: merged + flat + letters head
    + minimal_states in both buckets (compute.remat_by_bucket) trains the reference run's
    adapters (the same packed steps: packing is a function of the data and the seed), and
    each bucket's program is named after it."""
    steps = 3
    reference = {**REFERENCE}
    all_in = {**ALL_IN, "remat": "core_states",
              "remat_by_bucket": {"384": "minimal_states", "1024": "minimal_states"}}  # fmt: skip
    train(_run_config(aligned, "reference", steps, reference))
    capsys.readouterr()
    train(_run_config(aligned, "all-in", steps, all_in))
    events = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    programs = {e["program"] for e in events if e.get("event") == "compiled"}
    assert programs and all(p.startswith("train_step_T") for p in programs), programs

    def adapters(name):
        ckpt = Checkpoints(aligned / "runs" / name / "checkpoints")
        out = ckpt.manager.restore(steps)
        ckpt.close()
        return out["lora"]

    want, got = adapters("reference"), adapters("all-in")
    assert set(want) == set(got)
    # Adam turns fp32 rounding of near-zero gradients into update-sized steps on a few
    # entries: measured ≤ 1.9e-3 of the leaf's scale after 3 steps (merged LoRA forms
    # W + s·AB per call: its own fp32 reordering; a wrong function moves every entry by its
    # whole update, ~3e-2 of the scale, B from zero by O(1))
    for key in want:
        a, b = np.asarray(got[key]), np.asarray(want[key])
        assert np.max(np.abs(a - b)) <= 5e-3 * np.max(np.abs(b)) + 1e-6, key
    lengths = set()
    for name in ("reference", "all-in"):
        path = aligned / "runs" / name / "metrics.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        lengths.add(tuple(r["data/seq_len"] for r in rows if "data/seq_len" in r))
    assert len(lengths) == 1  # the same packed steps, in the same order


def test_named_programs_carry_their_bucket_remat(aligned):
    """NamedJit compiles one program per bucket (train_step_T…_mb…), each traced with its
    bucket's policy: remat_by_bucket changes the 64-token program only."""

    def lowered(compute, length):
        model = _model(aligned, **compute)
        graphdef, lora, frozen = split_model(model)
        tx = make_optimizer(load_config(overrides=["model.path=/nowhere"]).optim, 10)
        step = jit_train_step(make_train_step(graphdef, tx, brier_weight=0.1,
                                              slots_per_step=1.0), donate_argnums=(0, 1))  # fmt: skip
        # lowered only; 4 rows over the 4 test devices
        batch = synthetic_batch(4, length, 1, seg_len=32, slots=2, labels=4)
        return step.lower(lora, tx.init(lora), frozen, batch).as_text()

    base = {**ALL_IN, "remat": "core_states"}
    by_bucket = {**base, "remat_by_bucket": {"64": "minimal_states"}}
    short_base, short = lowered(base, 64), lowered(by_bucket, 64)
    long_base, long = lowered(base, 128), lowered(by_bucket, 128)
    assert "module @jit_train_step_T64_mb1" in short
    assert "module @jit_train_step_T128_mb1" in long
    assert short != short_base  # the 64-token bucket runs minimal_states
    assert long == long_base  # the other keeps compute.remat
