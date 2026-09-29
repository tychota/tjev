"""The training loop end to end on CPU (4 virtual devices), on the tiny fixture: resume and
branches equal uninterrupted runs, metrics never overlap, selection, the TPU kernels."""

import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from tjev.config import RunConfig, load_config
from tjev.data.item import parse_item, write_jsonl
from tjev.testing import make_tiny_snapshot, tiny_items
from tjev.train.checkpoint import Checkpoints
from tjev.train.loop import calibrated_nll, expected_slots_per_step, train

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("train")
    make_tiny_snapshot(root / "model")
    write_jsonl(root / "train.jsonl", [parse_item(r) for r in tiny_items(300, seed=1)])
    write_jsonl(root / "val.jsonl", [parse_item(r) for r in tiny_items(60, seed=2)])
    write_jsonl(root / "heldout.jsonl", [parse_item(r) for r in tiny_items(24, seed=5)])
    return root


def tiny_config(root, name, steps, **extra) -> RunConfig:
    overrides = [
        f"name={name}",
        f"output={root / 'runs'}",
        f"model.path={root / 'model'}",
        "model.dtype=float32",
        f"data.train=[{root / 'train.jsonl'}]",
        f"data.validation={root / 'val.jsonl'}",
        f"train.steps={steps}",
        "train.seq_buckets=[1024]",
        "train.microbatch_tokens=1024",
        "train.tokens_per_step=8192",
        "train.max_segments=4",
        "train.log_every=2",
        "train.eval_every=4",
        "train.quick_eval_every=0",
        "train.checkpoint_every=2",
        "train.checkpoint_secs=0",
        "compute.gdn_chunk=16",
        "lora.rank=4",
        "optim.lr=3e-3",
        "optim.warmup_steps=1",
        "log.tensorboard=false",
        *(f"{k}={v}" for k, v in extra.items()),
    ]
    return load_config(overrides=overrides)


def lora_at(root, name, step):
    ckpt = Checkpoints(root / "runs" / name / "checkpoints")
    out = ckpt.manager.restore(step)
    ckpt.close()
    return out["lora"]


def assert_same_adapters(a, b, rtol=1e-6, atol=1e-7):
    for key in a:
        np.testing.assert_allclose(np.asarray(a[key]), np.asarray(b[key]), rtol=rtol, atol=atol)


def rows(root, name):
    path = root / "runs" / name / "metrics.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_train_and_resume_equal_an_uninterrupted_run(setup):
    full = train(tiny_config(setup, "full", 6))
    assert full["best"]["step"] in (4, 6)
    learning = [x for x in rows(setup, "full") if "learning/loss" in x]
    assert len(learning) == 3
    assert all(np.isfinite(x["learning/loss"]) for x in learning)
    assert all(x["learning/update_rejected"] == 0 for x in learning)
    assert "perf/tflops_per_device" in learning[-1]
    assert any("eval/all/ece" in x for x in rows(setup, "full"))
    train(tiny_config(setup, "split", 4))  # interrupted after step 4 …
    train(tiny_config(setup, "split", 6))  # … resumed to 6
    assert_same_adapters(lora_at(setup, "full", 6), lora_at(setup, "split", 6))


def test_identity_mismatch_refuses_resume(setup):
    train(tiny_config(setup, "ident", 2))
    with pytest.raises(ValueError, match="identity"):
        train(tiny_config(setup, "ident", 4, **{"lora.rank": 8}))


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("adamw", {}),
        ("muon", {"optim.name": "muon", "optim.lr": 1e-3}),
        ("zclip", {"optim.clip_mode": "zscore", "optim.lr": 1e-3}),
    ],
)
def test_optimizers_train(setup, name, extra):
    train(tiny_config(setup, f"opt-{name}", 16, **{"train.eval_every": 16, "train.log_every": 1},
                      **extra))  # fmt: skip
    learning = [r for r in rows(setup, f"opt-{name}") if "learning/loss" in r]
    losses = np.asarray([r["learning/loss"] for r in learning])
    assert all(r["learning/update_rejected"] == 0.0 for r in learning)
    assert np.isfinite(losses).all()
    assert losses[-4:].mean() < losses[:4].mean(), losses.round(3).tolist()


def test_expected_slots_per_step_is_the_mean_over_a_fresh_stream():
    class Fake:
        closed = False

        def __init__(self):
            self.i = 0

        def __next__(self):
            self.i += 1
            w = np.zeros((2, 3, 4))
            w.flat[: 16 if self.i % 2 else 18] = 1.0  # alternating small / large steps
            return SimpleNamespace(weight=w)

        def close(self):
            Fake.closed = True

    assert expected_slots_per_step(Fake(), steps=4) == 17.0
    assert Fake.closed


def test_quick_eval_logs_between_full_evals(setup):
    extra = {"train.quick_eval_every": 2, "train.eval_every": 6,
             "data.quick_validation_per_source": 4, "train.log_every": 1}  # fmt: skip
    train(tiny_config(setup, "quick-eval", 6, **extra))
    logged = rows(setup, "quick-eval")
    assert [r["step"] for r in logged if "eval_quick/all/nll" in r] == [2, 4]
    assert [r["step"] for r in logged if "eval/all/nll" in r] == [6]
    assert all("learning/loss" in r for r in logged if r["step"] in (1, 3, 5))


def test_resume_drops_metrics_logged_past_the_checkpoint(setup):
    """An interrupted process may log steps after its last checkpoint; the resumed run must
    not duplicate them (curves would overlap)."""
    train(tiny_config(setup, "resume-log", 4, **{"train.log_every": 1}))
    path = setup / "runs" / "resume-log" / "metrics.jsonl"
    with path.open("a") as f:  # what a process killed at step 6 would have left behind
        for step in (5, 6):
            f.write(json.dumps({"step": step, "learning/loss": 99.0}) + "\n")
    train(tiny_config(setup, "resume-log", 6, **{"train.log_every": 1}))
    learning = [r for r in rows(setup, "resume-log") if "learning/loss" in r]
    assert [r["step"] for r in learning] == [1, 2, 3, 4, 5, 6]
    assert all(r["learning/loss"] != 99.0 for r in learning)


def test_restart_before_the_first_checkpoint_starts_the_curves_over(setup):
    run = setup / "runs" / "restart-log"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(json.dumps({"step": 1, "learning/loss": 99.0}) + "\n")
    train(tiny_config(setup, "restart-log", 2, **{"train.log_every": 1}))
    learning = [r for r in rows(setup, "restart-log") if "learning/loss" in r]
    assert [r["step"] for r in learning] == [1, 2]
    assert all(r["learning/loss"] != 99.0 for r in learning)


def test_branch_from_equals_the_standalone_shorter_run(setup):
    """WSD cooldown branch: resuming a longer run's stable phase with a shorter horizon
    trains exactly what that shorter run trains on its own (same data stream)."""
    wsd = {"optim.decay_fraction": 0.25, "train.eval_every": 99}
    train(tiny_config(setup, "br-long", 8, **wsd, **{"train.keep_checkpoints": 10}))
    train(tiny_config(setup, "br-short", 4, **wsd))
    branch = f"{setup / 'runs' / 'br-long'}@2"
    train(tiny_config(setup, "br-branch", 4, **wsd, **{"train.branch_from": branch}))
    assert_same_adapters(lora_at(setup, "br-short", 4), lora_at(setup, "br-branch", 4))
    late = f"{setup / 'runs' / 'br-long'}@4"  # step 4 is stable in the long run, decayed here
    with pytest.raises(ValueError, match="learning rates differ"):
        train(tiny_config(setup, "br-late", 4, **wsd, **{"train.branch_from": late}))


def test_selection_is_the_calibrated_nll_with_the_heldout_set(setup):
    train(tiny_config(setup, "select", 4, **{"data.heldout": setup / "heldout.jsonl"}))
    run = setup / "runs" / "select"
    selection = json.loads((run / "selection.json").read_text())
    assert selection["heldout"]["n"] == 24
    assert selection["score"] == pytest.approx(selection["calibrated"]["score"])
    assert (run / "eval" / "step-4-heldout.json").exists()
    assert any("eval_heldout/all/nll" in r for r in rows(setup, "select"))
    rng = np.random.default_rng(0)  # sharpened logits: a temperature > 1 fixes them
    target = np.eye(4)[rng.integers(0, 4, 200)]
    logits = 8.0 * (target + rng.normal(0, 0.7, target.shape))
    out = calibrated_nll((logits, np.ones_like(target, bool), target))
    assert out["temperature"] > 1.5
    assert np.isfinite(out["score"])


@pytest.mark.parametrize("fsdp", [1, 2])
def test_tpu_kernels_train_like_the_xla_path(setup, fsdp):
    """Splash attention + the Pallas GDN (interpret mode on CPU) under shard_map on the
    4-device mesh give the XLA path's adapters; fsdp=2 is v5e's 4B layout (data 2 × fsdp 2)."""
    exact = {"compute.gdn_precision": "highest", "train.eval_every": 99, "mesh.fsdp": fsdp}
    train(tiny_config(setup, f"k-xla-{fsdp}", 2, **exact))
    kernels = {"compute.attention": "splash", "compute.gdn_impl": "pallas_tpu"}
    try:
        train(tiny_config(setup, f"k-tpu-{fsdp}", 2, **exact, **kernels))
    finally:
        jax.set_mesh(None)
    a, b = lora_at(setup, f"k-xla-{fsdp}", 2), lora_at(setup, f"k-tpu-{fsdp}", 2)
    assert_same_adapters(a, b, rtol=2e-4, atol=2e-6)


def test_cli_train_and_compile_check(setup):
    from typer.testing import CliRunner

    from tjev.cli import app

    cfg = tiny_config(setup, "cli", 2, **{"train.eval_every": 99})
    overrides = [
        "name=cli", f"output={cfg.output}", f"model.path={cfg.model.path}",
        "model.dtype=float32", f"data.train=[{cfg.data.train[0]}]", "train.steps=2",
        "train.seq_buckets=[1024]", "train.microbatch_tokens=1024", "train.tokens_per_step=8192",
        "train.max_segments=4", "compute.gdn_chunk=16", "lora.rank=4", "log.tensorboard=false",
    ]  # fmt: skip
    runner = CliRunner()
    result = runner.invoke(app, ["compile-check", *overrides])
    assert result.exit_code == 0, result.output
    assert '"1024"' in result.output
    result = runner.invoke(app, ["train", *overrides])
    assert result.exit_code == 0, result.output
    assert (setup / "runs" / "cli" / "checkpoints" / "2").is_dir()
