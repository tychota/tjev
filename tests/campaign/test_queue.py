"""The queue runner on CPU (1 chip): a cooldown branch starts at its parent's checkpoint, and
a session deadline pauses every run (exit 3) for the next session to resume."""

import pytest

from tjev.campaign import queue
from tjev.data.item import parse_item, write_jsonl
from tjev.testing import make_tiny_snapshot, tiny_items

pytestmark = pytest.mark.slow


def _common(tmp_path) -> str:
    make_tiny_snapshot(tmp_path / "model")
    write_jsonl(tmp_path / "train.jsonl", [parse_item(r) for r in tiny_items(200, seed=1)])
    return (
        f"model.path={tmp_path / 'model'} model.dtype=float32 "
        f"data.train=[{tmp_path / 'train.jsonl'}] train.seq_buckets=[1024] "
        "train.microbatch_tokens=1024 train.tokens_per_step=4096 train.max_segments=4 "
        "train.quick_eval_every=0 train.checkpoint_secs=0 "
        "compute.gdn_chunk=16 lora.rank=4 optim.lr=3e-3 optim.warmup_steps=1 "
        "optim.decay_fraction=0.25 log.tensorboard=false"
    )


def test_queue_runs_a_branch_after_its_checkpoint(tmp_path):
    common = _common(tmp_path)
    path = tmp_path / "q"
    path.write_text(
        f"long chips=1 post=none {common} train.steps=8 train.checkpoint_every=2 "
        "train.keep_checkpoints=10\n"
        f"short chips=1 post=none after=long@2 {common} train.steps=4 "
        "train.branch_from=$RUNS/long@2\n"
    )
    runs = tmp_path / "runs"
    assert queue.main([str(path), "--chips", "1", "--runs", str(runs), "--mix", "unused"]) == 0
    assert (runs / "long" / "queue.done").exists()
    assert (runs / "short" / "queue.done").exists()
    assert '"event": "branched"' in (runs / "short" / "stdout.log").read_text()


def test_queue_pauses_at_the_deadline_and_resumes(tmp_path):
    common = _common(tmp_path)
    path = tmp_path / "q"
    path.write_text(f"job chips=1 post=none {common} train.steps=4 train.checkpoint_every=1\n")
    runs = tmp_path / "runs"
    args = [str(path), "--chips", "1", "--runs", str(runs), "--mix", "unused"]
    assert queue.main([*args, "--deadline-hours", "0.0003"]) == 3  # ~1 s: paused
    assert not (runs / "job" / "queue.done").exists()
    assert queue.main(args) == 0  # the next session finishes it
    assert (runs / "job" / "queue.done").exists()


def test_post_jobs_take_probe_post_args(tmp_path, monkeypatch):
    """PROBE_POST_ARGS (the campaign's env) is appended to every `tjev post` command."""
    seen = []
    monkeypatch.setattr(queue.subprocess, "Popen", lambda cmd, **kw: seen.append(cmd))
    monkeypatch.setenv("PROBE_POST_ARGS", "--orders 2 --extra-validation validation-v7.jsonl")
    job = queue.Job("run", 1, ["tpu-v6e", "qwen35-4b"], post="full", phase="post", group=[0])
    queue.start(job, tmp_path, queue.PostInputs("mix", "public.jsonl"), 1, None)
    assert seen[0][-4:] == ["--orders", "2", "--extra-validation", "validation-v7.jsonl"]
    assert "--quick" not in seen[0]
    monkeypatch.delenv("PROBE_POST_ARGS")
    queue.start(job, tmp_path, queue.PostInputs("mix", "public.jsonl"), 1, None)
    assert seen[1][-2:] == ["--jevbench", "public.jsonl"]
