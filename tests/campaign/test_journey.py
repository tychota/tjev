"""The kernel journey's pieces (the probe itself needs a TPU): the queue lines from the planner's
final run, the per-run statistics, the merge into journey.json, the estimate, the report."""

import gzip
import json
import os
from pathlib import Path

import pytest

from tjev.campaign import bench, journey
from tjev.campaign import plan as planner
from tjev.campaign.queue import parse_lines
from tjev.config import load_config


def _final_line(tmp_path, size="0.8B") -> str:
    mix = tmp_path / "mix"
    mix.mkdir(parents=True, exist_ok=True)
    (mix / "mix.yaml").write_text("data:\n  validation: v.jsonl\n")
    (tmp_path / "runs").mkdir(exist_ok=True)
    c = planner.Campaign(
        runs=tmp_path / "runs",
        mix=str(mix),
        models="/models",
        hardware="v6e1",
        final_sizes=(size,),
        horizon=3000,
        wandb={"project": "tjev-sweep", "campaign": "c1"},
    )
    return planner.plan("final", c)[0]


def test_every_journey_variant_is_a_bench_and_parity_variant():
    from tjev.campaign import parity

    assert set(journey.JOURNEY) <= set(bench.VARIANTS) <= set(parity.VARIANTS)
    assert set(journey.IR_VARIANTS) <= set(bench.VARIANTS)
    assert set(journey.LABELS) == set(journey.JOURNEY)


@pytest.mark.parametrize("variant", journey.JOURNEY)
def test_journey_line_replaces_the_compute_of_the_final_run(tmp_path, variant):
    line = journey.journey_line(_final_line(tmp_path), "0.8B", variant, 200)
    (job,) = parse_lines([line], tmp_path / "runs")
    assert job.name == f"journey-0.8b-{variant}"
    assert (job.chips, job.post, job.after) == (1, "none", None)
    files = [a for a in job.args if "=" not in a]
    cfg = load_config(*files, overrides=[a for a in job.args if "=" in a])
    # the variant's compute alone: what bench.VARIANTS sets, the preset's defaults otherwise
    expected = load_config("tpu-v6e", overrides=list(bench.VARIANTS[variant])).compute
    for key in ("gdn_impl", "gdn_chunk", "gdn_inverse", "remat", "lora_impl", "attention"):
        assert getattr(cfg.compute, key) == getattr(expected, key), key
    # the final run's training otherwise: model, rank, lr, gates, mix
    assert cfg.model.path == "/models/Qwen3.5-0.8B"
    assert cfg.train.microbatch_tokens == 8192
    assert any(a.startswith("lora.targets=") for a in job.args)  # FINAL_OVERRIDES kept
    assert (cfg.train.steps, cfg.train.checkpoint_every, cfg.train.quick_eval_every) == (200, 0, 0)
    assert (cfg.train.profile_start, cfg.train.profile_steps) == (192, 3)
    assert not cfg.train.health_kill
    assert cfg.log.wandb_project == "tjev-benchmarks"
    assert cfg.log.wandb_group == "journey"
    assert cfg.log.wandb_tags == ("journey", "0.8B", f"variant:{variant}")
    assert job.env["JAX_ENABLE_COMPILATION_CACHE"] == "false"
    assert job.env["TJEV_PROFILE_DETAIL"] == "1"
    assert job.env.get("TJEV_GDN_CONV_BARRIER") == ("0" if variant == "xla-nobarrier" else None)


def test_journey_line_options(tmp_path):
    line = journey.journey_line(_final_line(tmp_path, "4B"), "4B", "mega2", 6, cold=False)
    (job,) = parse_lines([line], tmp_path / "runs")
    assert job.env == {}  # too short for a trace, warm compiles
    assert "train.profile_start=0" in job.args
    with pytest.raises(ValueError):
        journey.journey_line(_final_line(tmp_path, "4B"), "4B", "no-such-variant", 10)


def _cli_plan(tmp_path, sizes: str) -> list[str]:
    """``tjev campaign plan final`` as cloud/tpu_campaign.sh runs it: through the CLI."""
    from typer.testing import CliRunner

    from tjev.cli.campaign import app

    mix = tmp_path / "mix"
    mix.mkdir(parents=True, exist_ok=True)
    (mix / "mix.yaml").write_text("data:\n  validation: v.jsonl\n")
    (tmp_path / "runs").mkdir(exist_ok=True)
    result = CliRunner().invoke(app, [
        "plan", "final", "--runs", str(tmp_path / "runs"), "--mix", str(mix),
        "--models", "/models", "--hardware", "v6e1", "--sizes", sizes, "--horizon", "3000"])  # fmt: skip
    assert result.exit_code == 0, result.output
    return result.output.splitlines()


@pytest.mark.parametrize("sizes", ["4B", "0.8B,4B", "4B,2B,0.8B"])
def test_the_4b_journey_line_is_a_4b_config(tmp_path, sizes):
    """tjev-journey2 trained 0.8B under the 4B names: probe_journey took the first line of a
    plan whose --sizes the wrapper had overridden. The line is picked by its run name, and
    the 4B journey runs are 4B: model, the planner's rank and microbatch."""
    lines = _cli_plan(tmp_path, sizes)
    final = journey.final_line_for(lines, "4B", 3000)
    assert final.split()[0] == "final-4b-h3000"
    assert journey.final_line_for(lines, "4B") == final  # the size's only root line
    for variant in journey.KEY_VARIANTS:
        (job,) = parse_lines([journey.journey_line(final, "4B", variant, 20)], tmp_path / "runs")
        assert job.name == f"journey-4b-{variant}"
        files = [a for a in job.args if "=" not in a]
        cfg = load_config(*files, overrides=[a for a in job.args if "=" in a])
        assert cfg.model.path == "/models/Qwen3.5-4B"
        assert cfg.lora.rank == planner.RANK["4B"] == 64
        assert cfg.train.microbatch_tokens == planner.HARDWARE["v6e1"]["microbatch"]["4B"] == 4096
        assert cfg.train.tokens_per_step == 65536
    if "0.8B" in sizes:
        small = journey.final_line_for(lines, "0.8B", 3000)
        assert small.split()[0] == "final-0.8b-h3000" and "Qwen3.5-0.8B" in small


def test_a_final_line_of_another_size_is_refused(tmp_path):
    small = _final_line(tmp_path, "0.8B")
    with pytest.raises(ValueError, match="final-4b"):
        journey.journey_line(small, "4B", "mega2", 20)  # the tjev-journey2 bug
    renamed = small.replace("final-0.8b-", "final-4b-", 1)
    with pytest.raises(ValueError, match=r"Qwen3\.5-0\.8B"):
        journey.journey_line(renamed, "4B", "mega2", 20)
    with pytest.raises(ValueError, match="0 lines for final-4b-h3000"):
        journey.final_line_for([small], "4B", 3000)


def test_final_line_cli_and_queue(tmp_path, capsys):
    plan = tmp_path / "plan.txt"
    plan.write_text("\n".join(_cli_plan(tmp_path, "0.8B,4B")) + "\n")  # 0.8B lines first
    assert (
        journey.main(["final-line", "--plan", str(plan), "--size", "4B", "--horizon", "3000"]) == 0
    )
    line = capsys.readouterr().out
    assert line.startswith("final-4b-h3000 ") and line.count("\n") == 1
    (tmp_path / "final-4B.line").write_text(line)
    assert journey.main(["queue", "--final-line", str(tmp_path / "final-4B.line"), "--size", "4B",
                         "--steps", "20", "--variants", "xla,mega2"]) == 0  # fmt: skip
    queue = capsys.readouterr().out.splitlines()
    assert [q.split()[0] for q in queue] == ["journey-4b-xla", "journey-4b-mega2"]
    assert all("model.path=/models/Qwen3.5-4B" in q for q in queue)
    assert (
        journey.main(["final-line", "--plan", str(plan), "--size", "2B", "--horizon", "3000"]) == 1
    )


def test_queue_env_tokens():
    (job,) = parse_lines(
        ["a chips=1 env=X=1 env=Y=a=b env=Z=$RUNS/x tpu-v6e train.steps=3"],
        Path("/r"),
    )
    assert job.env == {"X": "1", "Y": "a=b", "Z": "/r/x"}
    assert job.args == ["tpu-v6e", "train.steps=3"]


def test_variant_env_sets_and_restores(monkeypatch):
    monkeypatch.delenv("TJEV_GDN_CONV_BARRIER", raising=False)
    with bench.variant_env("xla-nobarrier") as env:
        assert env == {"TJEV_GDN_CONV_BARRIER": "0"}
        assert os.environ["TJEV_GDN_CONV_BARRIER"] == "0"
    assert "TJEV_GDN_CONV_BARRIER" not in os.environ
    monkeypatch.setenv("TJEV_GDN_CONV_BARRIER", "1")
    with bench.variant_env("xla-nobarrier"):
        pass
    assert os.environ["TJEV_GDN_CONV_BARRIER"] == "1"
    with bench.variant_env("xla") as env:
        assert env == {}


def test_parse_steps_and_profile_window():
    assert journey.parse_steps("0.8B:200, 4B:100") == {"0.8B": 200, "4B": 100}
    assert journey.profile_window(100) == (92, 3)
    assert journey.profile_window(10) == (0, 0)
    assert journey.profile_window(100, 0) == (0, 0)


def _row(step, seconds, seq_len, compile_=False, tokens=60000.0, padded=65536.0):
    return {
        "step": step,
        "perf/step_seconds": seconds,
        "perf/tokens_per_second_per_device": tokens / seconds,
        "perf/padded_tokens_per_second_per_device": padded / seconds,
        "perf/mfu": 0.2,
        "perf/tflops_per_device": 100.0,
        "perf/includes_compile": float(compile_),
        "perf/hbm_peak_gb": 10.0 + step / 100,
        "data/seq_len": seq_len,
        "learning/loss": 1.0 - step / 100,
    }


def test_step_stats_leave_out_compiles_and_the_profile():
    rows = [_row(1, 60.0, 1024, compile_=True)]
    rows += [_row(s, 1.0 if s % 2 else 2.0, 1024 if s % 2 else 4096) for s in range(2, 42)]
    rows[20]["perf/step_seconds"] = 99.0  # step 21: in the excluded profile window
    stats = journey.step_stats(rows, excluded={20, 21, 22})
    assert stats["steps_logged"] == 41
    assert stats["steps_timed"] == 36  # without the compile row, the window and the last row
    assert stats["step_s"]["p90"] == 2.0 and stats["step_s"]["p10"] == 1.0
    timed = [r for r in rows[1:-1] if r["step"] not in (20, 21, 22)]
    seconds = sum(r["perf/step_seconds"] for r in timed)
    assert stats["tokens_per_second"] == pytest.approx(60000 * len(timed) / seconds)
    assert stats["padded_tokens_per_second"] == pytest.approx(65536 * len(timed) / seconds)
    assert stats["pad_fraction"] == pytest.approx(1 - 60000 / 65536)
    assert stats["mfu"] == pytest.approx(0.2)
    assert stats["hbm_peak_gb"] == pytest.approx(10.41)
    assert set(stats["by_bucket"]) == {"1024", "4096"}
    assert stats["by_bucket"]["4096"]["step_s"]["p50"] == 2.0
    assert stats["by_bucket"]["1024"]["tokens_per_second"] == pytest.approx(60000)
    assert stats["loss_first"] > stats["loss_last"]


def _fake_run(runs, size, variant, *, seconds=1.0, ok=True, oom=False, profile_start=30):
    run = runs / journey.run_name(size, variant)
    run.mkdir(parents=True)
    rows = [_row(1, 50.0, 1024, compile_=True)]
    rows += [_row(s, seconds, 1024 if s % 3 else 4096, compile_=(s == 3)) for s in range(2, 41)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    log = [
        json.dumps({"event": "expected_slots_per_step", "value": 3.0}),
        json.dumps({"event": "compiled", "step": 1, "shape": [8, 8, 1024], "seconds": 48.5}),
        json.dumps({"event": "compiled", "step": 3, "shape": [8, 2, 4096], "seconds": 70.0}),
    ]
    if oom:
        log.append("jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Ran out of memory in hbm")
    (run / "stdout.log").write_text("\n".join(log) + "\n")
    cfg = {
        "train": {"profile_start": profile_start, "profile_steps": 3, "tokens_per_step": 65536},
        "compute": {"gdn_impl": "chunked"},
        "lora": {"rank": 32},
    }
    (run / "config.json").write_text(json.dumps({"config": cfg, "identity": {}}))
    if ok:
        (run / "queue.done").touch()
    return run


def _trace(path, ops, steps=3, step_us=1000.0):
    """A minimal Chrome trace of the TPU's "XLA Ops" and "Steps" lines."""
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "/device:TPU:0"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1, "args": {"name": "XLA Ops"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 2, "args": {"name": "Steps"}},
    ]
    events += [
        {"ph": "X", "pid": 1, "tid": 2, "name": str(i), "ts": i * step_us, "dur": step_us}
        for i in range(steps)
    ]
    for i, (name, scope, dur, nbytes) in enumerate(ops):
        args = {"hlo_category": "convolution fusion", "tf_op": f"jit(step)/blocks/{scope}/dot"}
        if nbytes is not None:
            args["bytes_accessed"] = nbytes
            args["raw_bytes_accessed"] = str(nbytes)
        events.append(
            {"ph": "X", "pid": 1, "tid": 1, "name": name, "ts": i, "dur": dur, "args": args}
        )
    path.mkdir(parents=True, exist_ok=True)
    with gzip.open(path / "host.trace.json.gz", "wt") as f:
        json.dump({"traceEvents": events}, f)


def test_steady_steps_leave_out_the_first_and_last_rows():
    """tjev-journey2, journey-0.8b-mega2-c128-merged: precompiled, yet the first row holds a
    24 s recompile (its includes_compile 0), the stop of the trace lands on step 14 and the
    final fetch on step 20. Counting the first row made 52k tok/s look like 20k."""
    secs = [25.19, *[1.25] * 12, 34.89, 0.22, 1.25, 1.25, 1.25, 1.25, 1.30]
    rows = [_row(s, t, 1024, tokens=63000.0) for s, t in enumerate(secs, start=1)]
    start, count = journey.profile_window(20)
    excluded = set(range(start - 1, start + count + 2))
    stats = journey.step_stats(rows, excluded)
    assert [r["step"] for r in journey.steady_steps(rows, excluded)] == [*range(2, 11), 17, 18, 19]
    assert stats["steps_timed"] == 12
    assert stats["padded_tokens_per_second"] == pytest.approx(65536 / 1.25)
    assert stats["tokens_per_second"] == pytest.approx(63000 / 1.25)
    assert stats["warmup_s"] == 25.19 and stats["warmup_excess_s"] == pytest.approx(23.94)
    assert journey.steady_steps(rows[:2], set()) == []


def test_collect_merges_runs_correctness_and_speedups(tmp_path):
    runs = tmp_path / "runs"
    _fake_run(runs, "0.8B", "xla", seconds=4.0)
    fast = _fake_run(runs, "0.8B", "mega2", seconds=1.0)
    _fake_run(runs, "0.8B", "kernels", ok=False, oom=True)
    _trace(
        fast / "profile" / "plugins",
        [
            ("fusion.1", "gdn_core", 600.0, "2.0GB"),
            ("fusion.2", "mlp", 300.0, 1e9),
            ("copy.3", "lora", 100.0, None),
        ],
    )
    parity = {
        "sizes": {
            "0.8B": {
                "variants": {
                    "xla": {
                        "pass": True,
                        "loss_rel": 1e-4,
                        "logit_max_abs": 0.1,
                        "grad_cosine": 0.9999,
                    },
                    "mega2": {"error": "boom"},
                }
            }
        }
    }
    out = journey.collect(runs, ["0.8B"], ["xla", "kernels", "mega2", "split"], parity)
    by = {r["variant"]: r for r in out["runs"]}
    assert by["split"]["status"] == "missing"
    assert by["kernels"]["status"] == "failed" and by["kernels"]["error"] == "oom"
    xla = by["xla"]
    assert xla["status"] == "ok"
    # minus the step-1 and step-3 compiles, the window 29..34 and the last row
    assert xla["steps_timed"] == 40 - 2 - 6 - 1
    assert xla["compile_s_total"] == pytest.approx(118.5)
    assert xla["by_bucket"]["4096"]["compile_s"] == 70.0
    assert xla["correctness"] == {
        "pass": True,
        "loss_rel": 1e-4,
        "logit_max_abs": 0.1,
        "grad_cosine": 0.9999,
    }
    assert by["mega2"]["correctness"] == {"pass": False, "error": "boom"}
    assert by["mega2"]["speedup_vs_xla"] == pytest.approx(4.0)
    prof = by["mega2"]["profile"]
    assert prof["steps"] == 3
    assert prof["ms_by_scope"] == {"gdn_core": 0.2, "mlp": 0.1, "lora": 0.033}
    assert prof["hbm_gb_by_scope"] == {
        "gdn_core": pytest.approx(0.6667, abs=1e-4),
        "mlp": pytest.approx(0.3333, abs=1e-4),
    }
    assert prof["hbm_gb_per_step"] == pytest.approx(1.0)
    assert prof["top_ops"][0] == {"op": "fusion", "percent": 90.0, "gb_per_step": 1.0}
    rows = [journey.flat_row(r) for r in out["runs"]]
    assert rows[0]["size"] == "0.8B" and rows[0]["correct"] is True


def test_collect_cli_writes_the_reports(tmp_path):
    runs = tmp_path / "runs"
    _fake_run(runs, "0.8B", "xla")
    out = tmp_path / "out"
    assert (
        journey.main(
            [
                "collect",
                "--runs",
                str(runs),
                "--out",
                str(out),
                "--sizes",
                "0.8B",
                "--variants",
                "xla,mega2",
                "--no-wandb",
            ]
        )
        == 0
    )
    combined = json.loads((out / "journey.json").read_text())
    assert [r["variant"] for r in combined["runs"]] == ["xla", "mega2"]
    assert json.loads((out / "0.8B-xla.json").read_text())["status"] == "ok"


def test_bytes_of():
    from tjev.campaign.xprof_summary import bytes_of

    assert bytes_of(12) == 12.0
    assert bytes_of("1048576") == 1048576.0
    assert bytes_of("1.5MiB") == 1.5 * 2**20
    assert bytes_of("3.2 GB") == pytest.approx(3.2e9)
    assert bytes_of("1,024") == 1024.0
    assert bytes_of("n/a") is None and bytes_of(None) is None and bytes_of(True) is None


def test_estimate_at_the_measured_speeds():
    est = journey.estimate(
        {"0.8B": 200, "4B": 100},
        list(journey.JOURNEY),
        parity_sizes=["0.8B"],
        ir_sizes=["0.8B"],
        ir_variants=list(journey.IR_VARIANTS),
    )
    assert len(est["runs"]) == 20
    xla = next(r for r in est["runs"] if r["size"] == "0.8B" and r["variant"] == "xla")
    assert xla["train_s"] == pytest.approx((200 + journey.BUCKETS) * 65536 / 13000)
    assert est["cu"] == pytest.approx(est["total_h"] * 4.08)
    assert 4 < est["total_h"] < 9
    assert "CU" in journey.estimate_markdown(est)


def test_ir_compile_on_cpu_dumps_the_hlo(tmp_path, monkeypatch):
    """The XLA variant's GDN core on a tiny layout: compiles, dumps, and the HLO parses."""
    from tjev.campaign.ir_stats import summarize_dumps
    from tjev.model.config import ModelConfig

    monkeypatch.setenv("LIBTPU_INIT_ARGS", "")
    report = journey.ir_compile(
        "xla", "0.8B", tmp_path / "xla", rows=1, length=64, model=ModelConfig.tiny()
    )
    assert report["shape"] == [1, 64, ModelConfig.tiny().conv_dim]
    assert "flops" in report["xla_cost"] or "error" in report["xla_cost"]
    dumps = summarize_dumps(tmp_path / "xla")
    assert dumps["hlo"] and max(v["fusions"] for v in dumps["hlo"].values()) > 0
    assert json.loads((tmp_path / "xla" / "compile.json").read_text())["variant"] == "xla"
    entry = {"exit": 0, **report, "dumps": dumps}
    assert journey.brief_ir(entry)["hlo_fusions"] > 0


def test_report_tables_and_charts(tmp_path, monkeypatch):
    import importlib.util
    import sys

    runs = tmp_path / "runs"
    _fake_run(runs, "0.8B", "xla-nobarrier", seconds=3.9)
    _fake_run(runs, "0.8B", "xla", seconds=4.0)
    fast = _fake_run(runs, "0.8B", "mega2", seconds=1.0)
    _trace(fast / "profile", [("fusion.1", "gdn_core", 600.0, 1e9), ("f.2", "mlp", 300.0, 1e9)])
    parity = {
        "sizes": {
            "0.8B": {
                "variants": {
                    "xla-nobarrier": {"pass": False, "loss_rel": 0.03, "logit_max_abs": 11.0}
                }
            }
        }
    }
    data = journey.collect(runs, ["0.8B"], ["xla-nobarrier", "xla", "mega2"], parity)
    data["ir"] = {
        "flags": ["--xla_jf_dump_to=<dumps>/llo"],
        "rejected": [],
        "runs": {
            "0.8B-mega2": {
                "exit": 0,
                "compile_s": 12.0,
                "dumps": {
                    "llo": {
                        "totals": {
                            "kernel_bundles": {"gdn_mega2_fwd": 1651, "gdn_mega2_bwd": 4302},
                            "fwd_bundles": 1651,
                            "bwd_bundles": 4302,
                            "xla_bundles": 100,
                            "mxu_ops": 4000,
                            "llo_passes": 79,
                            "kernel_vmem_bytes": {"gdn_mega2_fwd": 2**21},
                        }
                    }
                },
            }
        },
    }
    path = tmp_path / "journey.json"
    path.write_text(json.dumps(data))
    spec = importlib.util.spec_from_file_location(
        "journey_report", Path(__file__).parents[2] / "scripts" / "journey_report.py"
    )
    report = importlib.util.module_from_spec(spec)
    sys.modules["journey_report"] = report
    spec.loader.exec_module(report)
    text = report.markdown(data)
    assert "| XLA chunked, no conv barrier (miscompiled) |" in text
    assert "**no** (loss 3.0e-02, logit 11.00)" in text
    assert "4.00×" in text  # mega2 vs XLA
    assert "gdn_mega2_fwd 1,651" in text and "2.0 MiB" in text
    assert "gdn_core" in text
    monkeypatch.setenv("MPLBACKEND", "Agg")  # the script sets it too; Colab exports another
    assert report.main([str(path), "--out", str(tmp_path / "report")]) == 0
    made = sorted(p.name for p in (tmp_path / "report").glob("*.png"))
    other = dict(data, sizes=["4B"], runs=[dict(r, size="4B") for r in data["runs"]])
    del other["ir"]
    merged = report.merge([data, other])
    assert merged["sizes"] == ["0.8B", "4B"] and len(merged["runs"]) == 6
    assert merged["ir"] == data["ir"]
    assert made == ["hbm_peak.png", "mfu.png", "scopes.png", "tokens_per_second.png"]


def test_estimate_defaults_all_variants_at_0_8b_and_the_key_ones_at_4b(capsys):
    assert journey.main(["estimate"]) == 0
    text = capsys.readouterr().out
    assert text.count("| 0.8B |") == len(journey.JOURNEY)
    assert text.count("| 4B |") == len(journey.KEY_VARIANTS) == 5
    assert "| 20 |" in text and "| 4B | kernels |" not in text
    est = journey.estimate({"0.8B": 20, "4B": 20}, list(journey.JOURNEY),
                           size_variants={"4B": list(journey.KEY_VARIANTS)})  # fmt: skip
    assert [r["variant"] for r in est["runs"] if r["size"] == "4B"] == list(journey.KEY_VARIANTS)
    assert 1.0 < est["total_h"] < 2.5


def test_compile_table_uses_the_precompile_events(tmp_path):
    """train.precompile_buckets: the step-0 events give the compile time per shape, no row
    is a compile row, and a shape is counted once."""
    runs = tmp_path / "runs"
    run = _fake_run(runs, "4B", "mega2", profile_start=0)
    rows = [_row(s, 1.0, 1024 if s % 3 else 4096) for s in range(1, 21)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    log = [
        json.dumps({"event": "compiled", "step": 0, "shape": [8, 8, 1024], "seconds": 30.0,
                    "precompiled": True, "run_seconds": 31.0}),
        json.dumps({"event": "compiled", "step": 0, "shape": [8, 4, 2048], "seconds": 35.0,
                    "precompiled": True, "run_seconds": 36.5}),
        json.dumps({"event": "compiled", "step": 0, "shape": [8, 2, 4096], "seconds": 40.0,
                    "precompiled": True, "run_seconds": 42.0}),
    ]  # fmt: skip
    (run / "stdout.log").write_text("\n".join(log) + "\n")
    out = journey.collect(runs, ["4B"], list(journey.JOURNEY),
                          size_variants={"4B": ["mega2"]})  # fmt: skip
    (r,) = out["runs"]
    assert out["variants"] == ["mega2"]
    assert r["steps_timed"] == 18  # every step but the first and the last row
    assert [(c["step"], c["seq_len"], c["precompiled"]) for c in r["compiles"]] == [
        (0, 1024, True), (0, 2048, True), (0, 4096, True)]  # fmt: skip
    assert r["compile_s_total"] == pytest.approx(105.0)
    assert r["by_bucket"]["4096"]["compile_s"] == 40.0
    assert r["by_bucket"]["2048"] == {"compile_s": 35.0}  # compiled, never drawn
