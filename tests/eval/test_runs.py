"""Finished runs: evaluation, calibration bound to the checkpoint, the zero-shot control and
the post-training pipeline through the CLI."""

import json

import pytest
from typer.testing import CliRunner

from tjev.cli import app
from tjev.config import load_config
from tjev.data.item import parse_item, write_jsonl
from tjev.eval import runs
from tjev.testing import make_tiny_snapshot, tiny_items
from tjev.train.loop import train

pytestmark = pytest.mark.slow

TINY = [
    "model.dtype=float32", "train.seq_buckets=[1024,4096]", "train.microbatch_tokens=4096",
    "train.tokens_per_step=16384", "train.max_segments=4", "compute.gdn_chunk=16",
    "lora.rank=4", "log.tensorboard=false",
]  # fmt: skip


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    make_tiny_snapshot(root / "model")
    mix = root / "mix"
    write_jsonl(mix / "train" / "tiny.jsonl", [parse_item(r) for r in tiny_items(300, seed=1)])
    write_jsonl(mix / "validation.jsonl", [parse_item(r) for r in tiny_items(60, seed=2)])
    write_jsonl(mix / "calibration.jsonl", [parse_item(r) for r in tiny_items(90, seed=3)])
    overrides = [
        "name=run", f"output={root / 'runs'}", f"model.path={root / 'model'}",
        f"data.train=[{mix / 'train' / 'tiny.jsonl'}]", f"data.validation={mix / 'validation.jsonl'}",
        "train.steps=4", "train.checkpoint_every=2", "train.quick_eval_every=0",
        "optim.lr=3e-3", "optim.warmup_steps=1", *TINY,
    ]  # fmt: skip
    train(load_config(overrides=overrides))
    return root


def test_evaluate_and_calibrate_bind_to_the_checkpoint(trained):
    run, data = trained / "runs" / "run", trained / "mix" / "validation.jsonl"
    report = runs.evaluate(run, data, step=4)
    assert report["all"]["n"] == 60
    assert set(report["type"]) == {"noul", "choice", "score"}
    art = runs.calibrate(run, trained / "mix" / "calibration.jsonl", step=4)
    assert set(art["temperatures"]) == {"noul", "choice", "score"}
    path = run / "calibration-step-4.json"
    calibrated = runs.evaluate(run, data, step=4, calibration=path)
    assert calibrated["temperatures"] == art["temperatures"]
    with pytest.raises(ValueError, match="checkpoint"):
        runs.evaluate(run, data, step=2, calibration=path)


def test_zero_shot_control_calibrates_on_another_split(trained):
    report = runs.evaluate_base(
        trained / "model",
        trained / "mix" / "validation.jsonl",
        calibrate_on=trained / "mix" / "calibration.jsonl",
        cfg=load_config(overrides=TINY),
    )
    assert report["raw"]["all"]["n"] == 60
    assert report["calibrated"]["all"]["nll"] <= report["raw"]["all"]["nll"] + 0.05
    assert report["calibration_artifact"]["checkpoint"].startswith("base:")


def test_post_training_pipeline_through_the_cli(trained):
    runner = CliRunner()
    zeroshot = trained / "zeroshot.json"
    args = ["eval-base", str(trained / "model"), str(trained / "mix" / "validation.jsonl"),
            *TINY, "--calibrate-on", str(trained / "mix" / "calibration.jsonl"),
            "--out", str(zeroshot)]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    args = ["post", str(trained / "runs" / "run"), "--mix", str(trained / "mix"),
            "--jevbench", str(trained / "mix" / "validation.jsonl"), "--zeroshot", str(zeroshot)]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    post = trained / "runs" / "run" / "post"
    for name in (
        "calibration-jax",
        "jevbench-raw",
        "jevbench-cal",
        "generators-cal",
        "validation-cal",
    ):
        assert json.loads((post / f"{name}.json").read_text())
    text = (post / "summary.md").read_text()
    assert "zero-shot, calibrated" in text
    assert "held-out generators" in text
