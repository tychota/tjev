"""Finished runs: evaluation, calibration bound to the checkpoint, the zero-shot control and
the post-training pipeline through the CLI."""

import json

import pytest
from typer.testing import CliRunner

from tjev.cli import app
from tjev.config import load_config
from tjev.data.item import parse_item, write_jsonl
from tjev.eval import predictions as dump
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
    assert art["grouping"] == "type" and set(art["temperatures"]) == {"noul", "choice", "score"}
    cells = runs.calibrate(run, trained / "mix" / "calibration.jsonl", step=4,
                           grouping="type_lang", out=run / "calibration-cells.json")  # fmt: skip
    assert cells["grouping"] == "type_lang"  # type keys (the fallback) plus (type, lang) cells
    assert {"noul", "choice", "score"} <= set(cells["temperatures"])
    # 90 calibration items: every (type, lang) cell is under 50 slots, so keeps its type's T
    assert cells["fit"]["detail"]["noul/en"]["fallback"] == "type"
    report = runs.evaluate(run, data, step=4, calibration=run / "calibration-cells.json")
    assert report["temperatures"] == cells["temperatures"]
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
    v15 = json.loads((post / "v15.json").read_text())  # v1.5 scoring of the dumps
    assert set(v15) == {"jevbench", "generators", "validation"}
    assert v15["jevbench"]["calibrated"]["intelligence"]["intelligence_equal"] is not None
    assert set(v15["validation"]["sweep"]) == {"commit_margin", "noul_temperature",
                                               "score_temperature"}  # fmt: skip
    for split, n in (("calibration", 90), ("jevbench", 60), ("validation", 60),
                     ("generators", None)):  # fmt: skip
        rows = dump.read(post / "predictions" / f"{split}.jsonl.gz")
        assert n is None or len(rows) == n
        assert {"id", "type", "lang", "logits", "orders", "target", "probs", "tokens"} <= set(
            rows[0]
        )
        assert rows[0]["tokens"] > 0
    # an older post dir (no predictions yet) gets them on the next run, nothing else redone
    (post / "predictions" / "validation.jsonl.gz").unlink()
    before = (post / "validation-cal.json").stat().st_mtime_ns
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert (post / "predictions" / "validation.jsonl.gz").exists()
    assert (post / "validation-cal.json").stat().st_mtime_ns == before
    assert "held-out generators" in text


def test_post_extra_validation_reports_per_source_and_dumps(trained):
    runner = CliRunner()
    mix = trained / "mix"
    rows = [{**r, "source": f"src{i % 2}"} for i, r in enumerate(tiny_items(40, seed=7))]
    write_jsonl(mix / "validation-v7.jsonl", [parse_item(r) for r in rows])
    args = ["post", str(trained / "runs" / "run"), "--mix", str(mix), "--quick",
            "--jevbench", str(mix / "validation.jsonl"), "--extra-validation",
            "validation-v7.jsonl"]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    post = trained / "runs" / "run" / "post"
    rep = json.loads((post / "validation-v7-cal.json").read_text())
    assert rep["all"]["n"] == 40 and set(rep["source"]) == {"src0", "src1"}
    assert len(dump.read(post / "predictions" / "validation-v7.jsonl.gz")) == 40
    text = (post / "summary.md").read_text()
    assert "validation-v7 (per source)" in text and "source=src0" in text
    assert "validation-v7" in json.loads((post / "v15.json").read_text())
    bad = [*args[:-1], "missing.jsonl"]
    assert runner.invoke(app, bad).exit_code != 0


def test_order_averaged_calibration_is_bound_to_its_readout(trained):
    run, data = trained / "runs" / "run", trained / "mix" / "validation.jsonl"
    art = runs.calibrate(run, trained / "mix" / "calibration.jsonl", step=4, orders=2)
    assert art["orders"] == 2 and art["noul_order"] == "shuffled"
    path = run / "calibration-step-4-k2.json"
    report = runs.evaluate(run, data, step=4, calibration=path, orders=2)
    assert report["orders"] == 2 and report["all"]["n"] == 60
    scores = report["type"]["score"]["n"]
    assert report["cost"]["segments"] == 2 * 60 - scores  # score levels: one pass
    with pytest.raises(ValueError, match="orders=2, not 1"):
        runs.evaluate(run, data, step=4, calibration=path)


def test_post_with_two_orders_writes_its_own_files(trained):
    runner = CliRunner()
    args = ["post", str(trained / "runs" / "run"), "--mix", str(trained / "mix"),
            "--jevbench", str(trained / "mix" / "validation.jsonl"), "--quick",
            "--orders", "2", "--noul-order", "fixed"]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    post = trained / "runs" / "run" / "post"
    art = json.loads((post / "calibration-jax-k2-noul-fixed.json").read_text())
    assert (art["orders"], art["noul_order"]) == (2, "fixed")
    cal = json.loads((post / "jevbench-cal-k2-noul-fixed.json").read_text())
    assert (cal["orders"], cal["noul_order"]) == (2, "fixed")
    assert (post / "summary-k2-noul-fixed.md").exists()


def test_post_with_a_calibration_pool(trained):
    runner = CliRunner()
    mix = trained / "mix"
    base = ["post", str(trained / "runs" / "run"), "--mix", str(mix), "--jevbench",
            str(mix / "validation.jsonl"), "--quick"]  # fmt: skip
    refused = runner.invoke(app, [*base, "--calibration-pool", str(mix / "validation.jsonl"),
                                  "--calibration-pool", str(mix / "calibration.jsonl")])  # fmt: skip
    assert refused.exit_code != 0 and "reporting set" in str(refused.exception)
    pool = ["--calibration-pool", str(mix / "calibration.jsonl"),
            "--calibration-pool", str(mix / "train" / "tiny.jsonl")]  # fmt: skip
    result = runner.invoke(app, [*base, *pool, "--abstention", "noul"])
    assert result.exit_code == 0, result.output
    post = trained / "runs" / "run" / "post"
    art = json.loads((post / "calibration-jax-pool.json").read_text())
    assert art["fit"]["pool"]["files"] == ["calibration.jsonl", "tiny.jsonl"]
    assert (post / "jevbench-cal-pool.json").exists()
    assert (post / "predictions" / "pool-pool.jsonl.gz").exists()
    v15 = json.loads((post / "v15-pool.json").read_text())
    assert v15["jevbench-pool"]["policy"]["noul_abstention_rate"] == 0.0
