"""Readable profiles: op paths, the scan-buffer labels from the HLO, the saved residuals,
the report page."""

from __future__ import annotations

import json

import pytest

from tjev.campaign import hlo_labels, profile_report, residuals

HLO = """HloModule jit_train_step_T8_mb1, is_scheduled=true

%body (wide.arg_tuple.7: (s32[], f32[2,4], f32[3,2,4])) -> (s32[], f32[2,4], f32[3,2,4]) {
  %wide.arg_tuple.7 = (s32[], f32[2,4]{1,0}, f32[3,2,4]{2,1,0}) parameter(0)
  %get-tuple-element.1 = s32[] get-tuple-element(%wide.arg_tuple.7), index=0
  %get-tuple-element.2 = f32[3,2,4]{2,1,0} get-tuple-element(%wide.arg_tuple.7), index=2
  %gdn_mega2_fwd.5 = (f32[2,4]{1,0}, f32[2,4]{1,0}) custom-call(%get-tuple-element.1), custom_call_target="tpu_custom_call", metadata={}
  %pallas_call.9 = f32[2,4]{1,0} get-tuple-element(%gdn_mega2_fwd.5), index=1
  %dus_fusion.3 = f32[3,2,4]{2,1,0} fusion(%get-tuple-element.2, %pallas_call.9, %get-tuple-element.1), kind=kLoop, calls=%fc
  ROOT %tuple.4 = (s32[], f32[2,4]{1,0}, f32[3,2,4]{2,1,0}) tuple(%get-tuple-element.1, %pallas_call.9, %dus_fusion.3)
}

ENTRY %main.1 (p: f32[2,4]) -> f32[3,2,4] {
  %p = f32[2,4]{1,0} parameter(0), metadata={op_name="x"}
  %c = s32[] constant(0)
  %custom-call.7 = f32[3,2,4]{2,1,0} custom-call(), custom_call_target="AllocateBuffer"
  %custom-call.8 = bf16[3,16]{1,0} custom-call(), custom_call_target="AllocateBuffer"
  %tuple.1 = (s32[], f32[2,4]{1,0}, f32[3,2,4]{2,1,0}) tuple(%c, %p, %custom-call.7)
  %while.2 = (s32[], f32[2,4]{1,0}, f32[3,2,4]{2,1,0}) while(%tuple.1), condition=%cond, body=%body
  ROOT %get-tuple-element.9 = f32[3,2,4]{2,1,0} get-tuple-element(%while.2), index=2, metadata={op_name="jit(train_step_T8_mb1)/grad_accum/while"}
}
"""
RESIDUALS = {
    "remat": "core_states", "rows": 2, "length": 4, "super_blocks": 3, "stacked_gib": 0.0,
    "layers": [{"position": 0, "kind": "layer_gdn", "bytes": 32, "saved": [
        {"label": "gdn_states", "shape": [2, 4], "dtype": "float32", "bytes": 32,
         "stacked_bytes": 96, "description": "named 'gdn_states' from …"}]}],
}  # fmt: skip


def test_fine_path_drops_the_wrappers():
    f = hlo_labels.fine_path
    assert f("jit(train_step_T1024_mb32)/grad_accum/while/body/eval_jaxpr/transpose(jvp())/"
             "while/body/eval_jaxpr/checkpoint/rematted_computation/layer_gdn/mlp/up_proj/"
             "lora/B/dot_general:") == "layer_gdn/mlp/up_proj/lora/B/dot_general"  # fmt: skip
    assert f("jit(step)/grad_accum/while/body/eval_jaxpr/transpose(jvp(readout))/x") == (
        "readout/x"
    )
    assert f("jit(step)/grad_accum/while:") == "grad_accum"
    assert f("jit(step)/while/body/jit(dynamic_update_index_in_dim)/dynamic_update_slice") == (
        "scan stack write"
    )
    assert hlo_labels.phase_hint("a/transpose(jvp())/b") == "bwd"


def test_parse_and_shapes():
    module = hlo_labels.parse_hlo(HLO, {"dus_fusion.3": "jit(s)/grad_accum/while"})
    ins = module["instrs"]
    assert module["entry"] == "main.1" and ins["while.2"].opcode == "while"
    assert ins["tuple.1"].operands == ["c", "p", "custom-call.7"]
    assert ins["dus_fusion.3"].op_name == "jit(s)/grad_accum/while"  # from hlo_stats
    assert ins["p"].op_name == "x" and ins["get-tuple-element.9"].root
    assert hlo_labels.shape_key("bf16[8,2,1024]{2,1,0:T(8,128)(2,1)}") == ((8, 2, 1024), "bfloat16")
    assert hlo_labels.shape_bytes("f32[3,2,4]{2,1,0}") == 96
    assert hlo_labels.describe("bf16[1024,1024]") == "bfloat16 1024×1024, 2 MiB"


def test_label_allocations_follows_the_loop_to_the_writer():
    labels = hlo_labels.label_allocations(hlo_labels.parse_hlo(HLO), RESIDUALS)
    stack = labels["custom-call.7"]
    assert stack["residual"] == ["gdn_states"] and stack["producer"] == "gdn_mega2_fwd"
    assert "scan residual gdn_states" in stack["label"] and "gdn_mega2_fwd.5" in stack["label"]
    assert stack["bytes"] == 96 and stack["loop"] == "while.2"
    assert labels["custom-call.8"]["label"] == "loop buffer (XLA AllocateBuffer)"
    assert hlo_labels.label_allocations(hlo_labels.parse_hlo(HLO))["custom-call.7"][
        "label"
    ].startswith("loop buffer written by gdn_mega2_fwd")


def test_stack_labels():
    assert residuals.stack_labels(RESIDUALS) == {
        ((3, 2, 4), "float32"): ["layer_gdn[0] gdn_states"]
    }
    assert residuals._label("from the argument h") == "layer_input"
    assert residuals._label("named 'mlp_up' from x.py:1") == "mlp_up"
    assert residuals._label("output of jitted function '_splash_attention' …", 5) == "attn_context"


@pytest.fixture(scope="module")
def tiny_cfg(tmp_path_factory):
    from tjev.config import load_config
    from tjev.testing import make_tiny_snapshot

    root = tmp_path_factory.mktemp("res")
    make_tiny_snapshot(root / "model")
    return load_config(overrides=[f"model.path={root / 'model'}", "model.dtype=float32",
                                  "train.seq_buckets=[128]", "train.microbatch_tokens=256",
                                  "compute.gdn_chunk=16", "lora.rank=4",
                                  "compute.remat=core"])  # fmt: skip


def test_layer_residuals_of_a_tiny_model(tiny_cfg):
    report = residuals.layer_residuals(tiny_cfg, 128)
    assert report["rows"] == 2 and report["remat"] == "core"
    kinds = [ly["kind"] for ly in report["layers"]]
    assert kinds == ["layer0_gdn", "layer1_gdn", "layer2_gdn", "layer3_attn"]
    labels = {s["label"] for ly in report["layers"] for s in ly["saved"]}
    assert {"layer_input", "gdn_core_out"} <= labels  # the core policy's names
    gdn = report["layers"][0]["saved"]
    assert all(s["stacked_bytes"] == s["bytes"] * report["super_blocks"] for s in gdn)


def test_residuals_cli(tiny_cfg, tmp_path, capsys):
    out = tmp_path / "res.json"
    path = tiny_cfg.model.path
    assert residuals.main([f"model.path={path}", "model.dtype=float32", "train.seq_buckets=[128]",
                           "train.microbatch_tokens=256", "compute.gdn_chunk=16", "lora.rank=4",
                           "--policies", "full,core", "--out", str(out)]) == 0  # fmt: skip
    data = json.loads(out.read_text())
    assert set(data["buckets"]) == {"128"} and set(data["policies"]) == {"full", "core"}
    assert data["policies"]["full"]["128"]["saved"] == ["layer_input"]
    assert "layer_input" in capsys.readouterr().out


def test_report_page():
    summary = {
        "steps": 2, "step_source": "module executions", "step_ms": 10.0, "busy_fraction": 0.9,
        "device_ms_per_step": 9.0, "step_ms_source": "step annotations",
        "programs": {"1": {"kind": "train", "executions": 2, "name": "jit_train_step_T8_mb1"}},
        "ms_by_phase": {"fwd": 3.0, "bwd": 6.0},
        "scope_phase_ms": {"base": {"fwd": 3.0, "bwd": 6.0}},
        "scope_stats": {"base": {"ms": 9.0, "percent": 100.0, "tflops": 700.0, "hbm_gbps": 1.0,
                                 "bound_by": {"Compute": 100.0}}},
        "top_paths": [{"where": "layer_gdn/mlp/up_proj/base/dot_general", "phase": "bwd",
                       "ms": 6.0, "percent": 66.7, "tflops": 700.0, "hbm_gbps": 1.0,
                       "bound_by": {"Compute": 100.0}}],
        "memory": {"module": "jit_train_step_T8_mb1", "peak_heap_gib": 1.0, "parameters_gib": 0.5,
                   "composition": [{"what": "scan residual: gdn_states", "gib": 0.5, "buffers": 6}],
                   "top_buffers": []},
        "scan_buffers": {"custom-call.7": {"bytes": 96, "label": "scan residual gdn_states",
                                           "producer": "gdn_mega2_fwd"}},
        "residuals": RESIDUALS,
    }  # fmt: skip
    md = profile_report.markdown(summary, "9B")
    assert "jit_train_step_T8_mb1" in md and "| scan residual: gdn_states | 0.50 | 6 |" in md
    assert "layer_gdn/mlp/up_proj/base/dot_general" in md and "gdn_states" in md
    page = profile_report.page(summary, "9B")
    assert page.startswith("<!doctype html>") and "prefers-color-scheme" in page
    assert "custom-call.7" in page


RELAYOUT = """HloModule m

ENTRY %main (p: f32[4,8]) -> f32[8,4] {
  %p = f32[4,8]{1,0} parameter(0)
  %fusion.1 = f32[4,8]{1,0} fusion(%p), kind=kLoop, calls=%f, metadata={op_name="jit(s)/grad_accum/while/body/eval_jaxpr/transpose(jvp())/while/body/layer1_gdn/gdn_out/mul"}
  %bitcast.2 = f32[8,4]{0,1} bitcast(%fusion.1)
  ROOT %copy.3 = f32[8,4]{1,0} copy(%bitcast.2), metadata={op_name="jit(s)/grad_accum/while/body/eval_jaxpr/transpose(jvp())/while:"}
}
"""


def test_loop_named_copies_take_their_producers_path():
    module = hlo_labels.parse_hlo(RELAYOUT)
    new = hlo_labels.reattribute(module, "copy.3", "data formatting")
    assert new.endswith("layer1_gdn/gdn_out/mul/relayout") and "transpose(" in new
    assert hlo_labels.reattribute(module, "fusion.1") is None  # already named


def test_readable_kinds_and_partial_executions():
    from tjev.campaign import xprof_tools as xt

    assert xt.readable_kind("convolution fusion", "fusion.3") == "matmul (fused)"
    assert xt.readable_kind("custom-call", "gdn_mega2_fwd.2") == "Pallas kernel"
    assert xt.readable_kind("loop fusion", "bitcast_dynamic-update-slice_fusion.4").startswith(
        "scan stack write"
    )
    mods = [{"name": "jit_a(7)", "startTimePs": "0", "endTimePs": str(t)}
            for t in (100, 100, 100, 50)]  # fmt: skip
    assert xt.executions_from_modules(mods) == {"7": 3.5}  # the cut one counts for half
