"""The xplane summary's attribution (synthetic hlo_stats DataTables), the streamed trace
fallback, and the analyse probe's queue lines."""

from __future__ import annotations

import gzip
import json

import pytest

from tjev.campaign import analyse, xprof_summary, xprof_tools

TRAIN, EVAL = "111", "222"
FWD = "jit(step)/grad_accum/while/body/eval_jaxpr/jvp()/while/body/eval_jaxpr/layer_gdn"
BWD = (
    "jit(step)/grad_accum/while/body/eval_jaxpr/transpose(jvp())/while/body/eval_jaxpr/"
    "checkpoint/layer_gdn"
)
REMAT = BWD + "/rematted_computation"


def _datatable(rows: list[dict]) -> dict:
    """Google DataTable JSON as xprof writes it ({cols, rows: [{c: [{v}]}]})."""
    ids = list(dict.fromkeys(k for r in rows for k in r))
    return {
        "cols": [{"id": i, "label": i, "type": "string"} for i in ids],
        "rows": [{"c": [{"v": r.get(i)} for i in ids]} for r in rows],
    }


def _op(name, tf_op, us, *, program=TRAIN, flops=0.0, hbm=0.0, mem=0.0, bound="HBM", occ=2.0,
        remat="No", category="convolution fusion"):  # fmt: skip
    return {
        "program_id": program,
        "category": category,
        "hlo_op_name": name,
        "tf_op_name": tf_op,
        "occurrences": occ,
        "total_self_time": us,
        "model_flop_rate": flops,  # GFLOP/s
        "hbm_bw": hbm,  # GiB/s
        "measured_memory_bw": mem,  # GiB/s
        "operational_intensity": flops / mem if mem else 0.0,
        "bound_by": bound,
        "hlo_rematerialization": remat,
        "dma_stall_percent": 0.0,
    }


ROWS = [
    # a train program run twice: 2 s of base matmuls (fwd, bwd, remat), 1 s of LoRA, the
    # optimizer, and an eval program the step tables must leave out
    _op("fusion.1", f"{FWD}/mlp/base/dot_general:", 500_000, flops=700_000, bound="Compute"),
    _op("fusion.2", f"{BWD}/mlp/base/dot_general:", 1_000_000, flops=700_000, bound="Compute"),
    _op(
        "fusion.3",
        f"{REMAT}/mlp/base/dot_general:",
        500_000,
        flops=700_000,
        bound="Compute",
        remat="Yes",
    ),
    _op(
        "fusion.4.refs_0_",
        f"{BWD}/mlp/lora/dot_general:",
        1_000_000,
        flops=20_000,
        hbm=1000.0,
        mem=1000.0,
    ),
    _op("add.5", "jit(step)/optimizer/add:", 0.0, occ=2.0),  # no time: dropped
    _op("reduce.6", "jit(step)/health/reduce_sum:", 10_000, occ=2.0, category="reduce"),
    _op(
        "fusion.7",
        "jit(step)/while/body/mlp/base/dot_general:",
        4_000_000,
        program=EVAL,
        flops=700_000,
        bound="Compute",
        occ=10.0,
    ),
]


def test_phase_of_reads_the_jax_path():
    assert xprof_tools.phase_of(f"{FWD}/mlp/base/dot") == "fwd"
    assert xprof_tools.phase_of(f"{BWD}/mlp/base/dot") == "bwd"
    assert xprof_tools.phase_of(f"{REMAT}/mlp/base/dot") == "remat"
    assert xprof_tools.phase_of(f"{BWD}/mlp/base/dot", "Yes") == "remat"  # the HLO flag
    assert xprof_tools.phase_of("jit(step)/optimizer/add") == "other"
    assert xprof_tools.phase_of("") == "other"


def test_table_and_program_kinds():
    rows = xprof_tools.table(json.dumps(_datatable(ROWS)))
    assert rows[0]["hlo_op_name"] == "fusion.1" and rows[-1]["program_id"] == EVAL
    assert xprof_tools.table([_datatable(ROWS[:1]), {}], 1) == []
    assert xprof_tools.program_kinds(rows) == {TRAIN: "train", EVAL: "eval"}
    assert xprof_tools.op_kind("fusion.4.refs_0_") == "fusion"
    assert xprof_tools.op_kind("gdn_mega2_bwd.27") == "gdn_mega2_bwd"


def test_executions_from_modules_and_ops():
    modules = [{"name": f"jit_step({TRAIN})"}, {"name": f"jit_step({EVAL})"},
               {"name": f"jit_step({TRAIN})"}, {"name": "weird"}]  # fmt: skip
    assert xprof_tools.executions_from_modules(modules) == {TRAIN: 2, EVAL: 1}
    # top-level ops (outside every loop): their usual occurrence count
    assert xprof_tools.executions_from_ops(ROWS) == {TRAIN: 2, EVAL: 10}


def test_attribute_scope_phase_and_rates():
    rows = xprof_tools.table(_datatable(ROWS))
    out = xprof_tools.attribute(rows, {TRAIN: 2, EVAL: 10}, top=3)
    assert out is not None
    assert out["steps"] == 2 and set(out["programs"]) == {TRAIN}  # the eval left out
    assert out["device_ms_per_step"] == pytest.approx(1505.0)  # 3.01 s over 2 steps
    assert out["by_scope"] == {"base": pytest.approx(66.45, abs=0.01),
                               "lora": pytest.approx(33.22, abs=0.01),
                               "health": pytest.approx(0.33, abs=0.01)}  # fmt: skip
    assert out["ms_by_scope"]["base"] == pytest.approx(1000.0)
    assert out["scope_phase_ms"]["base"] == {"fwd": 250.0, "bwd": 500.0, "remat": 250.0}
    assert out["scope_phase_ms"]["lora"] == {"bwd": 500.0}
    assert out["scope_phase_ms"]["health"] == {"other": 5.0}
    assert out["ms_by_phase"]["bwd"] == pytest.approx(1000.0)
    base = out["scope_stats"]["base"]
    assert base["tflops"] == pytest.approx(700.0)  # time-weighted: every base op at 700
    assert base["bound_by"] == {"Compute": 100.0}
    lora = out["scope_stats"]["lora"]
    assert lora["hbm_gbps"] == pytest.approx(1000 * 2**30 / 1e9, abs=0.1)  # GiB/s → GB/s
    assert lora["intensity"] == pytest.approx(20_000 / 1000 * 1e9 / 2**30, abs=0.1)
    assert [o["op"] for o in out["top_ops"]] == ["fusion.2", "fusion.4.refs_0_", "fusion.1"]
    assert out["top_ops"][0]["phase"] == "bwd" and out["top_ops"][0]["ms"] == 500.0
    kinds = {(k["op"], k["scope"], k["phase"]): k["ms"] for k in out["top_kinds"]}
    assert kinds[("fusion", "base", "bwd")] == 500.0
    every = xprof_tools.attribute(rows, {TRAIN: 2, EVAL: 10}, programs="all")
    assert every["steps"] == 12 and every["by_phase"]["eval"] == pytest.approx(57.06, abs=0.01)
    text = xprof_tools.scope_phase_table(out)
    assert text.splitlines()[1].startswith("base") and "total" in text


def test_attribute_without_a_backward_keeps_every_program():
    rows = xprof_tools.table(_datatable([ROWS[-1]]))
    out = xprof_tools.attribute(rows)
    assert out["steps"] == 10 and out["by_phase"] == {"eval": 100.0}
    assert xprof_tools.attribute([]) is None


def test_device_peaks_counters_and_buffers():
    roof = [{"cols": [], "rows": [], "p": {"device_type": "TPU v6 Lite",
             "peak_flop_rate": "946700", "peak_hbm_bw": "1525.5", "hbm_ridge_point": "577.96"}}]  # fmt: skip
    peaks = xprof_tools.device_peaks(roof)
    assert peaks["peak_tflops"] == 946.7 and peaks["hbm_ridge_point"] == 578.0
    assert peaks["peak_hbm_gbps"] == pytest.approx(1638.0, abs=0.1)
    util = _datatable([
        {"node": 0.0, "name": "Avg MXU Busy", "achieved": 50.0, "peak": 100.0},
        {"node": 0.0, "name": "HBM Rd+Wr - core 0", "achieved": 4e12, "peak": 8e12},
        {"node": 0.0, "name": "HBM Read Ratio", "achieved": 3e12, "peak": 4e12},
        {"node": 1.0, "name": "Avg MXU Busy", "achieved": 0.0, "peak": 100.0},
    ])  # fmt: skip
    measured = xprof_tools.measured_counters(util)
    assert measured["mxu_busy_fraction"] == 0.5 and measured["hbm_bw_utilization"] == 0.5
    assert measured["hbm_tb"] == 4.0 and measured["hbm_read_fraction"] == 0.75
    viewer = {
        "moduleName": "jit_step", "peakHeapMib": 2048.0, "entryComputationParametersMib": 1024.0,
        "maybeLiveOutMib": 10.0, "totalBufferAllocationMib": 2100.0,
        "maxHeap": [{"groupName": "Parameter", "logicalBufferSizeMib": 1024.0},
                    {"groupName": "Temporary", "logicalBufferSizeMib": 512.0}],
        "maxHeapBySize": [{"label": "w", "logicalBufferSizeMib": 1024.0, "groupName": "Parameter"}],
    }  # fmt: skip
    mem = xprof_tools.peak_buffers(viewer)
    assert mem["peak_heap_gib"] == 2.0 and mem["at_peak_by_group_gib"]["Parameter"] == 1.0
    assert mem["top_buffers"][0]["gib"] == 1.0
    assert xprof_tools.peak_buffers({}) == {}


def test_summarize_xplane_from_tool_outputs(tmp_path):
    xplane = tmp_path / "plugins" / "profile" / "t" / "host.xplane.pb"
    xplane.parent.mkdir(parents=True)
    xplane.write_bytes(b"")
    raw = {
        "hlo_stats": _datatable(ROWS),
        "roofline_model": [{"cols": [], "rows": [], "p": {"peak_flop_rate": "946700"}}],
        "overview_page": [{"p": {"device_duty_cycle_percent": "98.0%"}},
                          {**_datatable([{"stepnum": "0"}]), "p": {"steptime_ms_average": "1600"}}],
        "memory_profile": {"hloModules": [{"name": f"jit_step({TRAIN})"}] * 2},
    }  # fmt: skip
    out = xprof_tools.summarize_xplane(tmp_path, raw=raw, memory=False)
    assert out["steps"] == 2 and out["step_source"] == "module executions"
    assert out["step_ms"] == 1600.0 and out["step_ms_source"] == "step annotations"
    assert out["busy_fraction"] == 0.98 and out["source"] == "xplane"
    assert "eval" in out["all_programs"]["by_phase"]
    assert set(xprof_tools.profile_tables(out)) == {"scope_phase", "top_ops", "top_kinds"}
    scalars = xprof_tools.wandb_scalars(out, "9B/x/")
    assert scalars["9B/x/scope/base"] == out["by_scope"]["base"]


def test_find_xplane_skips_the_detail_capture(tmp_path):
    timed = tmp_path / "plugins" / "profile" / "2026_01_01" / "h.xplane.pb"
    detail = tmp_path / "detail" / "plugins" / "profile" / "2026_02_02" / "h.xplane.pb"
    for p in (timed, detail):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
    assert xprof_tools.find_xplane(tmp_path) == timed
    assert xprof_tools.find_xplane(tmp_path / "detail") == detail
    assert xprof_tools.find_xplane(tmp_path / "missing") is None


def _trace(path, chunk_pad: int = 0):
    """A Chrome trace with a train and an eval program on "XLA Modules", host step markers."""
    meta = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "/device:TPU:0"}},
        {"ph": "M", "name": "process_name", "pid": 9, "args": {"name": "/host:CPU"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1, "args": {"name": "XLA Ops"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 2, "args": {"name": "Steps"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 3, "args": {"name": "XLA Modules"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 4, "args": {"name": "Other"}},
    ]
    events = list(meta)
    for i, (module, t0) in enumerate([("jit_step(1)", 0), ("jit_step(2)", 1000),
                                      ("jit_step(1)", 2000)]):  # fmt: skip
        events.append({"ph": "X", "pid": 1, "tid": 3, "name": module, "ts": t0, "dur": 900})
        events.append({"ph": "X", "pid": 1, "tid": 2, "name": str(i), "ts": t0, "dur": 900})
        tf = f"{BWD}/mlp/base/dot" if module == "jit_step(1)" else "jit(step)/while/mlp/base/dot"
        events.append({"ph": "X", "pid": 1, "tid": 1, "name": "fusion.1", "ts": t0 + 10,
                       "dur": 600, "args": {"hlo_category": "convolution fusion", "tf_op": tf,
                                            "bytes_accessed": "1GB", "pad": "x" * chunk_pad}})  # fmt: skip
        events.append({"ph": "X", "pid": 1, "tid": 4, "name": "noise", "ts": t0, "dur": 1})
    events.append({"ph": "X", "pid": 9, "tid": 1, "name": "train", "ts": 0, "dur": 5,
                   "args": {"step_num": "7"}})  # fmt: skip
    path.mkdir(parents=True, exist_ok=True)
    with gzip.open(path / "host.trace.json.gz", "wt") as f:
        json.dump({"displayTimeUnit": "ns", "traceEvents": events}, f)


def test_stream_events_across_chunks(tmp_path, monkeypatch):
    _trace(tmp_path, chunk_pad=300)
    monkeypatch.setattr(xprof_summary, "_CHUNK", 256)  # every event straddles a chunk
    kept = xprof_summary.stream_events(tmp_path / "host.trace.json.gz")
    names = [e.get("name") for e in kept if e["ph"] == "X"]
    assert "noise" not in names and names.count("fusion.1") == 3 and "train" in names
    op = next(e for e in kept if e.get("name") == "fusion.1")
    assert "pad" not in op["args"] and op["args"]["bytes_accessed"] == "1GB"


def test_summarize_trace_counts_the_train_programs(tmp_path):
    _trace(tmp_path)
    out = xprof_summary.summarize(tmp_path, steps=99)
    assert out["steps"] == 2 and out["step_source"] == "module executions"
    assert out["step_ms"] == 0.9 and out["ms_by_scope"] == {"base": 0.6}
    assert out["hbm_gb_per_step"] == 1.0 and out["source"] == "trace json"


def test_summarize_any_falls_back_to_the_trace(tmp_path):
    _trace(tmp_path)
    (tmp_path / "h.xplane.pb").write_bytes(b"")  # an xplane, but no xprof here (or no rows)
    out = xprof_tools.summarize_any(tmp_path, steps=4)
    assert out["source"] == "trace json" and out["steps"] == 2


FINAL = (
    "final-9b-h3000 chips=1 post=quick tpu-v6e qwen35-9b /m/mix.yaml model.path=/x/Qwen3.5-9B "
    "train.microbatch_tokens=2048 compute.gdn_impl=pallas_tpu_mega2 compute.remat=core_states "
    "train.steps=3000 lora.rank=64 train.checkpoint_every=300 train.keep_steps=[600,1200] "
    "log.wandb=true log.wandb_project=tjev-final"
)


def test_analyse_lines():
    line = analyse.analyse_line(FINAL, "9B", "aggregate", steps=25, traced=3,
                                extra=["compute.attention=xla"])  # fmt: skip
    tokens = line.split()
    assert tokens[:3] == ["analyse-9b-aggregate", "chips=1", "post=none"]
    assert (
        "compute.gdn_impl=pallas_tpu_mega2" in tokens and "train.microbatch_tokens=2048" in tokens
    )
    assert "lora.rank=16" in tokens and "lora.rank=64" not in tokens
    assert "train.steps=25" in tokens and "train.steps=3000" not in tokens
    # 3 traced steps: start_trace before 21, stop after 23 (profile_steps + 1 steps)
    assert "train.profile_start=21" in tokens and "train.profile_steps=2" in tokens
    assert "train.checkpoint_every=0" in tokens and "train.quick_eval_every=0" in tokens
    assert "log.wandb_project=tjev-benchmarks" in tokens and tokens[-1] == "compute.attention=xla"
    assert not any(t.startswith("env=") for t in tokens)
    detail = analyse.analyse_line(FINAL, "9B", "detail", rank=None).split()
    assert "env=TJEV_PROFILE_DETAIL=1" in detail
    assert "env=TJEV_KERNEL_REGIONS=1" in detail and "env=TJEV_UNROLL_LAYERS=1" not in detail
    unrolled = analyse.analyse_line(FINAL, "9B", "detail", unroll=True).split()
    assert "env=TJEV_UNROLL_LAYERS=1" in unrolled
    assert "lora.rank=64" in detail and "train.profile_steps=0" in detail
    assert analyse.window("detail", 25, 3) == (5, 4, 0)
    with pytest.raises(ValueError):
        analyse.window("aggregate", 5, 3)
    with pytest.raises(ValueError):
        analyse.analyse_line(FINAL, "4B", "aggregate")  # not the 4B line
    items = analyse.compile_items(line)
    assert items[:3] == ["tpu-v6e", "qwen35-9b", "/m/mix.yaml"]
    assert not any(t.startswith(("chips=", "post=", "log.")) for t in items)


def test_analyse_cli(tmp_path, capsys):
    plan = tmp_path / "final.plan"
    plan.write_text(FINAL + "\nfinal-9b-h750 chips=1 after=final-9b-h3000@600 x\n")
    assert analyse.main(["line", "--plan", str(plan), "--size", "9B", "--mode", "aggregate",
                         "--extra=a=1 b=2"]) == 0  # fmt: skip
    line = capsys.readouterr().out.strip()
    assert line.startswith("analyse-9b-aggregate ") and line.endswith(" a=1 b=2")
    (tmp_path / "q").write_text(line + "\n")
    assert analyse.main(["items", "--line-file", str(tmp_path / "q")]) == 0
    assert capsys.readouterr().out.startswith("tpu-v6e qwen35-9b")
