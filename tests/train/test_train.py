"""The training loop end to end on CPU (4 virtual devices), on the tiny fixture: resume and
branches equal uninterrupted runs, metrics never overlap, selection, the TPU kernels."""

import json
import logging
import re
import shutil
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
    assert full["best"]["step"] in (2, 4, 6)
    learning = [x for x in rows(setup, "full") if "learning/loss" in x]
    assert len(learning) == 3
    assert all(np.isfinite(x["learning/loss"]) for x in learning)
    assert all(x["learning/update_rejected"] == 0 for x in learning)
    assert "perf/tflops_per_device" in learning[-1]
    assert any("eval/all/ece" in x for x in rows(setup, "full"))
    train(tiny_config(setup, "split", 4))  # interrupted after step 4 …
    train(tiny_config(setup, "split", 6))  # … resumed to 6
    assert_same_adapters(lora_at(setup, "full", 6), lora_at(setup, "split", 6))


def test_best_fit_and_remat_by_bucket_train_and_resume_exactly(setup):
    """train.packing=best_fit (two buckets: the short segments fill the 1024 rows' tails)
    and a per-bucket remat: a run resumed from its checkpoint equals the uninterrupted one."""
    extra = {"train.seq_buckets": "[384,1024]", "train.microbatch_tokens": 3072,
             "train.tokens_per_step": 6144, "train.packing": "best_fit", "train.packing_bins": 2,
             "compute.remat": "core_states", "train.log_every": 1,
             "compute.remat_by_bucket": "{384: minimal_states}"}  # fmt: skip
    train(tiny_config(setup, "bf-full", 12, **extra))
    train(tiny_config(setup, "bf-split", 6, **extra))
    train(tiny_config(setup, "bf-split", 12, **extra))
    assert_same_adapters(lora_at(setup, "bf-full", 12), lora_at(setup, "bf-split", 12))
    lengths = {x["data/seq_len"] for x in rows(setup, "bf-full") if "data/seq_len" in x}
    assert lengths == {384.0, 1024.0}


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
    train(tiny_config(setup, f"opt-{name}", 16, **{"train.checkpoint_every": 16, "train.log_every": 1},
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
    extra = {"train.quick_eval_every": 2, "train.checkpoint_every": 6,
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
    wsd = {"optim.decay_fraction": 0.25}
    train(tiny_config(setup, "br-long", 8, **wsd, **{"train.keep_checkpoints": 10}))
    train(tiny_config(setup, "br-short", 4, **wsd))
    branch = f"{setup / 'runs' / 'br-long'}@2"
    train(tiny_config(setup, "br-branch", 4, **wsd, **{"train.branch_from": branch}))
    assert_same_adapters(lora_at(setup, "br-short", 4), lora_at(setup, "br-branch", 4))
    assert min(r["step"] for r in rows(setup, "br-branch")) > 2  # its own curve, from its step
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
    exact = {"compute.gdn_precision": "highest", "mesh.fsdp": fsdp}
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

    cfg = tiny_config(setup, "cli", 2)
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


def test_health_metrics_and_verdict_are_logged(setup):
    train(tiny_config(setup, "health", 6, **{"train.log_every": 1, "train.checkpoint_every": 6}))
    learning = [r for r in rows(setup, "health") if "learning/loss" in r]
    for key in ("health/grad_norm_gdn", "health/grad_norm_attention", "health/grad_norm_mlp",
                "health/update_ratio_a", "health/update_ratio_b", "health/clip_fraction",
                "health/grad_norm_clipped", "health/adam_sqrt_v_mean", "health/momentum_cos",
                "learning/spikes", "health/status"):  # fmt: skip
        assert key in learning[-1], key
    # the rate the optimizer used: step 1 runs at schedule(0) = 0 (linear warmup from zero)
    assert learning[0]["learning/lr"] == 0.0
    verdict = json.loads((setup / "runs" / "health" / "health.json").read_text())
    assert verdict["status"] in ("ok", "warn")
    assert verdict["step"] == 6


def test_health_kill_stops_the_run(setup, monkeypatch):
    import functools

    from tjev.train import loop
    from tjev.train.health import HealthMonitor, HealthRules

    rules = HealthRules(window=2, ratio_a_kill=1e-12, ratio_a_warn=1e-12)
    monkeypatch.setattr(loop, "HealthMonitor", functools.partial(HealthMonitor, rules))
    out = train(tiny_config(setup, "killed", 12, **{"train.log_every": 1, "train.health_kill": True,
                                                   "train.checkpoint_every": 12}))  # fmt: skip
    assert out["health"] == "killed"
    verdict = json.loads((setup / "runs" / "killed" / "health.json").read_text())
    assert verdict["status"] == "killed"
    assert verdict["rules"]["update_ratio_a"]["level"] == "kill"
    assert max(r["step"] for r in rows(setup, "killed")) < 12


def test_a_stop_file_ends_the_run_at_the_current_step(setup):
    """<run>/STOP: the run finishes at the step it is seen (a checkpoint and the full eval
    there) as completed (exit 0, the best checkpoint selected), not interrupted."""
    from tjev.train.loop import STOP_FILE

    run = setup / "runs" / "stopped"
    run.mkdir(parents=True, exist_ok=True)
    (run / STOP_FILE).touch()
    out = train(tiny_config(setup, "stopped", 8))  # checkpoints every 2 steps
    assert out["best"]["step"] == 1
    learning = [x for x in rows(setup, "stopped") if "learning/loss" in x]
    assert max(x["step"] for x in learning) == 1
    assert (run / "selection.json").exists() and (run / "eval" / "step-1.json").exists()
    assert (run / "checkpoints" / "1").is_dir()


def _at_step(monkeypatch, step, action):
    """Run ``action`` once the loop has dispatched ``step`` (MetricWindow.add)."""
    from tjev.train import loop

    add = loop.MetricWindow.add

    def hooked(self, s, *args, **kw):
        add(self, s, *args, **kw)
        if s == step:
            action()

    monkeypatch.setattr(loop.MetricWindow, "add", hooked)


def _losses(root, name):
    return {r["step"]: r["learning/loss"] for r in rows(root, name) if "learning/loss" in r}


@pytest.mark.parametrize("how", ["stop-file", "sigterm", "sigint"])
def test_stopping_at_any_step_resumes_exactly(setup, monkeypatch, how):
    """A STOP file, SIGTERM or SIGINT at step 3 (between the checkpoints of 2 and 4) saves
    step 3 (waiting for the async save) with the eval state; running again resumes from it:
    the same loss at every step, the same adapters, as an uninterrupted run."""
    import os
    import signal

    from tjev.train.loop import STOP_FILE

    extra = {"train.log_every": 1}
    train(tiny_config(setup, "whole", 6, **extra))  # (trained once: then already done)
    name = f"cut-{how}"
    run = setup / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    action = {
        "stop-file": lambda: (run / STOP_FILE).touch(),
        "sigterm": lambda: os.kill(os.getpid(), signal.SIGTERM),
        "sigint": lambda: os.kill(os.getpid(), signal.SIGINT),
    }[how]
    with monkeypatch.context() as m:
        _at_step(m, 3, action)
        if how == "stop-file":
            assert train(tiny_config(setup, name, 6, **extra))["best"]["step"] in (2, 3)
        else:
            with pytest.raises(SystemExit) as stopped:
                train(tiny_config(setup, name, 6, **extra))
            assert stopped.value.code == (143 if how == "sigterm" else 130)
    ckpt = Checkpoints(run / "checkpoints")
    assert ckpt.latest() == 3
    meta = ckpt.manager.restore(3)["meta"]
    ckpt.close()
    # the eval state (the selection so far) is saved; a STOP step is itself evaluated
    assert meta["best"]["step"] in ((2, 3) if how == "stop-file" else (2,))
    (run / STOP_FILE).unlink(missing_ok=True)
    train(tiny_config(setup, name, 6, **extra))
    whole, cut = _losses(setup, "whole"), _losses(setup, name)
    assert sorted(cut) == [1, 2, 3, 4, 5, 6]
    for step in whole:
        assert cut[step] == pytest.approx(whole[step], rel=1e-6, abs=1e-7), step
    assert_same_adapters(lora_at(setup, "whole", 6), lora_at(setup, name, 6))


def test_retention_keeps_branch_points_and_drops_superseded_resume_saves(setup, monkeypatch):
    """keep_checkpoints=1: the latest planned step, the best one and train.keep_steps stay;
    a resume-only save (here SIGTERM at step 5) is deleted by the next save."""
    import os
    import signal

    extra = {"train.keep_checkpoints": 1, "train.keep_steps": "[4]"}
    with monkeypatch.context() as m:
        _at_step(m, 5, lambda: os.kill(os.getpid(), signal.SIGTERM))
        with pytest.raises(SystemExit):
            train(tiny_config(setup, "retain", 8, **extra))
    ckpt = Checkpoints(setup / "runs" / "retain" / "checkpoints")
    assert 5 in ckpt.all_steps() and 4 in ckpt.all_steps()
    ckpt.close()
    out = train(tiny_config(setup, "retain", 8, **extra))
    ckpt = Checkpoints(setup / "runs" / "retain" / "checkpoints")
    assert ckpt.all_steps() == sorted({4, 8, out["best"]["step"]})
    ckpt.close()


def test_cadence_keep_every_from_a_step():
    from tjev.train.loop import Cadence

    cfg = load_config(overrides=["train.steps=1000", "train.checkpoint_every=100",
                                 "train.keep_every=200", "train.keep_from=500",
                                 "train.keep_steps=[300]"])  # fmt: skip
    c = Cadence(cfg)
    assert [s for s in range(1, 1001) if c.kept(s)] == [300, 600, 800, 1000]
    assert all(c.checkpoint(s) for s in (300, 600, 800, 1000))


def test_cadence_keep_tail_keeps_every_cooldown_checkpoint():
    """keep_tail: every planned checkpoint of the tail (the cooldown) is kept, whatever
    keep_from says: a cooldown branch (its tail is its whole run) keeps its decay."""
    from tjev.train.loop import Cadence

    c = Cadence(load_config(overrides=["train.steps=1000", "train.checkpoint_every=200",
                                       "train.checkpoint_tail_every=50", "train.keep_every=200",
                                       "train.keep_from=500", "train.keep_tail=true"]))  # fmt: skip
    assert [s for s in range(1, 1001) if c.kept(s)] == [600, 800, 850, 900, 950, 1000]
    assert not c.resume_only(850) and c.resume_only(777)
    branch = Cadence(load_config(overrides=["train.steps=500", "train.checkpoint_every=200",
                                            "train.checkpoint_tail_every=50", "train.keep_from=2500",
                                            "train.keep_every=200", "train.keep_tail=true",
                                            "train.branch_from=/x/long@400"]))  # fmt: skip
    assert [s for s in range(1, 501) if branch.kept(s)] == [450, 500]
    off = Cadence(load_config(overrides=["train.steps=1000", "train.checkpoint_tail_every=50"]))
    assert [s for s in range(801, 1001) if off.kept(s)] == []


def test_cadence_tail_and_full_eval_cadences():
    from tjev.train.loop import Cadence

    cfg = load_config(overrides=["train.steps=1000", "train.checkpoint_every=200",
                                 "train.checkpoint_tail_every=50", "train.quick_eval_every=100",
                                 "train.quick_eval_tail_every=25"])  # fmt: skip
    c = Cadence(cfg)  # the tail: the cooldown, steps 801-1000
    assert [s for s in range(1, 1001) if c.checkpoint(s)] == [200, 400, 600, 800, 850, 900, 950,
                                                              1000]  # fmt: skip
    assert [s for s in range(1, 1001) if c.full_eval(s)] == [200, 400, 600, 800, 850, 900, 950,
                                                             1000]  # fmt: skip
    assert [s for s in range(801, 1001) if c.quick_eval(s)] == [825, 875, 925, 975]
    own = Cadence(load_config(overrides=["train.steps=1000", "train.checkpoint_every=500",
                                         "train.full_eval_every=250", "train.full_eval_tail_every=100",
                                         "train.tail_fraction=0.3", "train.keep_steps=[123]"]))  # fmt: skip
    assert [s for s in range(1, 1001) if own.full_eval(s)] == [250, 500, 800, 900, 1000]
    # every full eval is a checkpoint (selectable); keep_steps are planned saves too
    assert [s for s in range(1, 1001) if own.checkpoint(s)] == [123, 250, 500, 800, 900, 1000]
    assert own.resume_only(777) and not own.resume_only(123)
    today = Cadence(load_config(overrides=["train.steps=120"]))  # the defaults: unchanged
    assert [s for s in range(1, 121) if today.checkpoint(s)] == [50, 100, 120]
    assert [s for s in range(1, 121) if today.full_eval(s)] == [50, 100, 120]
    assert [s for s in range(1, 121) if today.quick_eval(s)] == [10, 20, 30, 40, 60, 70, 80, 90,
                                                                  110]  # fmt: skip


def test_default_identity_is_unchanged_by_the_new_fields():
    """Every field added with the regularisers and cadences is, at its default, left out of
    the identity: runs from before them resume (the identity of 89a4ee1's algorithm)."""
    from tjev.train import loop
    from tjev.utils import fingerprint

    cfg = load_config(overrides=["model.path=/nowhere"])
    raw = cfg.to_dict()
    del raw["loss"], raw["lora"]["dropout"], raw["optim"]["ema_decay"]
    for key in ("tail_fraction", "checkpoint_tail_every", "full_eval_every",
                "full_eval_tail_every", "quick_eval_tail_every", "keep_steps", "keep_every",
                "keep_from", "keep_tail", "select_on", "select_ema"):  # fmt: skip
        del raw["train"][key]
    old_not_identity = {
        "train": ("steps", "log_every", "quick_eval_every", "checkpoint_every",
                  "checkpoint_secs", "keep_checkpoints", "profile_start", "profile_steps",
                  "branch_from", "health_metrics", "health_report", "health_kill",
                  "precompile_buckets"),
        "compute": ("libtpu_flags",),
        "data": ("workers", "worker_buffer", "validation", "validation_per_source", "heldout",
                 "quick_validation_per_source"),
    }  # fmt: skip
    for section, keys in old_not_identity.items():
        for key in keys:
            raw[section].pop(key)
    for section in ("compute", "lora", "train"):
        for key, default in loop.IDENTITY_NEW_DEFAULTS[section].items():
            if loop._same(raw[section].get(key), default):
                raw[section].pop(key)
    for key in ("name", "output", "log"):
        raw.pop(key)
    want = fingerprint(raw)
    got = loop.run_identity(cfg, "m")["config"]
    assert got == want
    for override in (
        "loss.label_smoothing=0.1",
        "optim.ema_decay=0.999",
        "lora.dropout=0.1",
        "loss.kl_base_weight=0.1",
        "loss.hard_targets=true",
        "train.packing=best_fit",
        "compute.remat_by_bucket={1024: minimal_states}",
    ):
        changed = load_config(overrides=["model.path=/nowhere", override])
        assert loop.run_identity(changed, "m")["config"] != want, override
    cadence = load_config(overrides=["model.path=/nowhere", "train.checkpoint_tail_every=100",
                                     "train.select_on=lang_mean", "train.keep_steps=[600]"])  # fmt: skip
    assert loop.run_identity(cadence, "m")["config"] == want


def test_identity_of_the_runs_before_the_perf_fields_is_kept():
    """The identities a config had before the perf fields (gdn_norm, …,
    remat_by_bucket, train.packing; computed at f8afbf3): unchanged at their defaults, so
    the sweep's runs resume. (The test above builds its expectation from the current
    fields: it cannot see a new field missing from IDENTITY_NEW_DEFAULTS.)"""
    from tjev.train import loop

    pinned = {
        (): "eaa0a8f196ec22cd5b488bf29b15a159ba341aef8e696fe8c01ac97e757be691",
        ("compute.remat=core_states", "compute.gdn_impl=pallas_tpu_mega2"):
            "43fa08781715ea8d0fd3140239fa6c5afb503bd428c80eb6fdad86172d486360",
    }  # fmt: skip
    for overrides, want in pinned.items():
        cfg = load_config(overrides=["model.path=/nowhere", *overrides])
        assert loop.run_identity(cfg, "m")["config"] == want, overrides


def test_calibrated_nll_per_language():
    rng = np.random.default_rng(0)
    target = np.eye(4)[rng.integers(0, 4, 200)]
    logits = 3.0 * (target + rng.normal(0, 0.7, target.shape))
    logits[100:] = rng.normal(0, 1, (100, 4))  # the second half (fr): uninformative
    mask = np.ones_like(target, bool)
    langs = ["en"] * 100 + ["fr"] * 100
    held = (logits[:40], mask[:40], target[:40])
    out = calibrated_nll((logits, mask, target), held, (langs, ["en"] * 40))
    assert set(out["lang"]) == {"en", "fr"}
    assert out["lang"]["fr"] > out["lang"]["en"]
    assert out["lang_max"] == out["lang"]["fr"]
    assert out["lang_mean"] == pytest.approx((out["lang"]["en"] + out["lang"]["fr"]) / 2)
    plain = calibrated_nll((logits, mask, target), held)
    assert plain["score"] == out["score"] and "lang" not in plain


def test_selection_on_the_worst_language(setup):
    root = setup
    val = root / "val-langs.jsonl"
    if not val.exists():
        write_jsonl(val, [parse_item(r) for r in tiny_items(30, seed=2)]
                    + [parse_item(r) for r in tiny_items(30, seed=3, lang="fr")])  # fmt: skip
    train(tiny_config(setup, "by-lang", 4, **{"data.validation": val,
                                              "train.select_on": "lang_max"}))  # fmt: skip
    sel = json.loads((setup / "runs" / "by-lang" / "selection.json").read_text())
    assert sel["select_on"] == "lang_max"
    assert set(sel["calibrated"]["lang"]) == {"en", "fr"}
    assert sel["score"] == pytest.approx(max(sel["calibrated"]["lang"].values()))
    logged = [r for r in rows(setup, "by-lang") if "eval_calibrated/lang/fr" in r]
    assert [r["step"] for r in logged] == [2, 4]


def test_ema_is_evaluated_checkpointed_and_selectable(setup):
    from tjev.eval.runs import LoadedRun as Run

    train(tiny_config(setup, "ema", 4, **{"optim.ema_decay": 0.9, "train.select_ema": "true"}))
    run = setup / "runs" / "ema"
    logged = rows(setup, "ema")
    assert any("eval_ema/all/nll" in r for r in logged)
    assert any("eval_calibrated_ema/score" in r for r in logged)
    assert (run / "eval" / "step-4-ema.json").exists()
    sel = json.loads((run / "selection.json").read_text())
    assert sel["weights"] == "ema"
    loaded = Run(run)
    assert loaded.weights == "ema" and loaded.step == sel["step"]
    raw = lora_at(setup, "ema", sel["step"])
    ema = jax.tree.leaves(loaded.lora)
    assert any(
        not np.allclose(np.asarray(e), np.asarray(r))
        for e, r in zip(ema, raw.values(), strict=True)
    )
    assert Run(run, step=sel["step"]).weights == "raw"  # an explicit step: the raw adapters
    # the EMA is its own checkpoint item (ema/, the adapters' tree and dtype): what the
    # loader reads, and what a backup without the optimizer state keeps
    ckpt = Checkpoints(run / "checkpoints")
    assert ckpt.has_ema(4) and (run / "checkpoints" / "4" / "ema").is_dir()
    for a, b in zip(jax.tree.leaves(ckpt.restore_ema(4, loaded.lora)), ema, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    ckpt.close()
    plain = Checkpoints(setup / "runs" / "br-short" / "checkpoints")  # no optim.ema_decay
    assert not plain.has_ema(4) and plain.restore_ema(4, loaded.lora) is None
    plain.close()
    # probe_calib's RUN:STEP:ema / RUN:STEP entries (tjev.campaign.calib): the run's checkpoint
    # as the selection, scored on its EMA (from the ema/ item, even once the backup dropped
    # the optimizer state) or on its adapters
    from tjev.campaign import calib

    for entry, weights in (("c1/ema:4:ema", "ema"), ("c1/ema:4", "raw")):
        parsed = calib.parse_entry(entry)
        dest = calib.prepare(run, setup / "calib" / parsed.local, parsed)
        shutil.rmtree(dest / "checkpoints" / "4" / "opt")
        scored = Run(dest)
        assert scored.weights == weights and scored.step == 4
        want = jax.tree.leaves(Run(run, step=4).lora) if weights == "raw" else None
        if want is not None:
            for a, b in zip(jax.tree.leaves(scored.lora), want, strict=True):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        else:
            assert any(not np.allclose(np.asarray(e), np.asarray(r)) for e, r in
                       zip(jax.tree.leaves(scored.lora), lora_at(setup, "ema", 4).values(),
                           strict=True))  # fmt: skip


def test_regularised_run_trains(setup):
    """Every regulariser at once on the loop (label smoothing on a source pattern, the KL,
    dropout under merged LoRA, weight decay, EMA): finite losses, the KL logged."""
    extra = {"loss.label_smoothing": 0.1, "loss.smooth_sources": "[colors,d*]",
             "loss.kl_base_weight": 0.1, "lora.dropout": 0.1, "compute.lora_impl": "merged",
             "optim.weight_decay": 0.05, "optim.ema_decay": 0.9, "train.log_every": 1}  # fmt: skip
    train(tiny_config(setup, "regularised", 4, **extra))
    learning = [r for r in rows(setup, "regularised") if "learning/loss" in r]
    assert all(
        np.isfinite(r["learning/loss"]) and r["learning/update_rejected"] == 0 for r in learning
    )
    assert all("learning/kl_base" in r for r in learning)


def test_benchmark_run_without_checkpoints_or_evals(setup, capsys, monkeypatch):
    """checkpoint_every=0 (a kernel-journey run): no checkpoint, no full eval, the bucket and
    the padded throughput in every row, one compile event per new batch shape; the libtpu
    flags it ran with and their scoped VMEM limit in stdout, config.json and summary.json."""
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--xla_tpu_scoped_vmem_limit_kib=65536 --x=1")
    out = train(tiny_config(setup, "bench", 4, **{"train.checkpoint_every": 0,
                                                  "train.log_every": 1}))  # fmt: skip
    assert out["steps"] == 4
    run = setup / "runs" / "bench"
    assert not (run / "checkpoints").exists() or not any((run / "checkpoints").iterdir())
    assert not (run / "eval").exists() and not (run / "selection.json").exists()
    perf = [r for r in rows(setup, "bench") if "perf/step_seconds" in r]
    assert [r["step"] for r in perf] == [1, 2, 3, 4]
    assert all(r["data/seq_len"] == 1024 for r in perf)
    for r in perf:
        real, padded = (
            r["perf/tokens_per_second_per_device"],
            r["perf/padded_tokens_per_second_per_device"],
        )
        assert 0 < real <= padded
        assert real / padded == pytest.approx(1 - r["data/pad_fraction"], rel=1e-6)
    events = [json.loads(x) for x in capsys.readouterr().out.splitlines()
              if x.startswith('{"event": ')]  # fmt: skip
    compiled = [e for e in events if e["event"] == "compiled"]
    assert [c["step"] for c in compiled] == [1]
    assert compiled[0]["shape"][-1] == 1024 and compiled[0]["seconds"] > 0
    want = {"libtpu_init_args": "--xla_tpu_scoped_vmem_limit_kib=65536 --x=1",
            "libtpu_source": "env", "scoped_vmem_limit_kib": 65536,
            "kernel_vmem_limit_bytes": int(0.9 * 65536 * 1024)}  # fmt: skip
    assert [{k: e[k] for k in want} for e in events if e.get("event") == "libtpu"] == [want]
    assert json.loads((run / "config.json").read_text())["libtpu"] == want
    summary = json.loads((run / "summary.json").read_text())
    assert summary["libtpu/scoped_vmem_limit_kib"] == 65536 and "health/status" in summary


def test_precompile_buckets_leaves_training_unchanged(setup):
    """train.precompile_buckets runs each bucket's step once on a discarded all-padding batch:
    the trained adapters equal a run without it."""
    train(tiny_config(setup, "plain", 2))
    train(tiny_config(setup, "pre", 2, **{"train.precompile_buckets": "true"}))
    assert_same_adapters(lora_at(setup, "plain", 2), lora_at(setup, "pre", 2))


def test_precompile_buckets_compiles_every_bucket_before_step_1(setup, capsys):
    """Each bucket's train step compiles once before step 1 (step-0 events), and the real
    steps hit the jit cache (same shapes, dtypes and shardings: no further compile of the
    step), so every row is steady-state timing."""
    messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    pxla = logging.getLogger("jax._src.interpreters.pxla")
    pxla.addHandler(handler)
    try:
        with jax.log_compiles(True):
            train(tiny_config(setup, "pre-bench", 4, **{
                "train.checkpoint_every": 0, "train.log_every": 1,
                "train.seq_buckets": "[512,1024]", "train.precompile_buckets": "true"}))  # fmt: skip
    finally:
        pxla.removeHandler(handler)
    # one program per bucket, named after its shape (train.step.NamedJit)
    step_compiles = [
        m for m in messages if re.match(r"Compiling jit\(train_step_T\d+_mb\d+\) with", m)
    ]
    assert len(step_compiles) == 2, step_compiles
    assert {re.match(r"Compiling jit\((\w+)\)", m)[1] for m in step_compiles} == {
        "train_step_T512_mb2", "train_step_T1024_mb2"}  # fmt: skip
    compiled = [json.loads(x) for x in capsys.readouterr().out.splitlines()
                if x.startswith('{"event": "compiled"')]  # fmt: skip
    assert [(c["step"], c["shape"][-1], c["precompiled"]) for c in compiled] == [
        (0, 512, True), (0, 1024, True)]  # fmt: skip
    assert all(0 < c["seconds"] <= c["run_seconds"] for c in compiled)
    perf = [r for r in rows(setup, "pre-bench") if "perf/step_seconds" in r]
    assert [r["step"] for r in perf] == [1, 2, 3, 4]
    assert all(r["perf/includes_compile"] == 0 and r["perf/compiled_shapes"] == 2 for r in perf)
    # the first step's clock starts after the compiles
    assert perf[0]["perf/step_seconds"] < sum(c["run_seconds"] for c in compiled)


def test_long_buckets_train_and_evaluate_end_to_end(setup):
    """Buckets (1024, 16384) at microbatch_tokens=1024: the ~9k-token items train and are
    evaluated in one-row-per-device microbatches of 16384 tokens (no drop)."""
    filler = "the door is shut. " * 500  # ~9k tokens of the tiny char-level tokenizer
    long = [dict(r, state=r["state"] + " " + filler, id=f"long-{r['id']}")
            for r in tiny_items(60, seed=7)]  # fmt: skip
    write_jsonl(setup / "train-long.jsonl",
                [parse_item(r) for r in tiny_items(200, seed=1) + long])  # fmt: skip
    write_jsonl(setup / "val-long.jsonl",
                [parse_item(r) for r in tiny_items(20, seed=2) + long[:8]])  # fmt: skip
    cfg = tiny_config(setup, "long", 6, **{
        "data.train": f"[{setup / 'train-long.jsonl'}]",
        "data.validation": str(setup / "val-long.jsonl"),
        "train.seq_buckets": "[1024,16384]",
        "train.checkpoint_every": 6,
    })  # fmt: skip
    out = train(cfg)
    assert out["best"]["step"] == 6
    learning = [x for x in rows(setup, "long") if "learning/loss" in x]
    assert all(np.isfinite(x["learning/loss"]) for x in learning)
    assert max(x.get("perf/compiled_shapes", 0) for x in rows(setup, "long")) >= 2
    evals = list((setup / "runs" / "long").rglob("*.json"))
    reports = [json.loads(p.read_text()) for p in evals if "eval" in p.parts]
    assert any(r.get("dropped_too_long") == 0 for r in reports), [p.name for p in evals]
