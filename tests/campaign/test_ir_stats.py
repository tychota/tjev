"""The compiler-dump parsers on crafted samples of the Mosaic / LLO / HLO text formats."""

from tjev.campaign import ir_stats

BUNDLES = """\
// kernel: gdn_v2_fwd
#allocation0 [shape = 'u32[144,128]', space=vmem, size = 0x12000, scoped, tag = 'scratch']
#allocation1 [shape = 'u8[512]', space=smem, size = 0x200]
#allocation2 [shape = 'f32[64,128]', space=vmem, size = 32768]
  0x0   :  { %s1 = sld [smem:[#allocation1]] }
  0x1   : > { %p222 = pnand %p51265, %p221  ;;  %s202 = scalar_lea.vmem [#allocation0], %s51260 }
  0x2   :  { %v10 = vmatpush1.bf16.xpose.msra.mxu0 %v1  ;;  %v11 = vmatpush1.bf16.msra.mxu1 %v2
              ;;  %v12 = vmatprep.subr.mxu0 %v3 }
  0x1f  : > { %v20 = vmatmul.mubr.bf16.gmra.mxu0 %v10  ;;  %v21 = vmatmul.mubr.bf16.gmra.mxu1 %v11 }
  0x20  :  { %v30 = vpop.f32.mrf.mxu0  ;;  %v31 = vpop.f32.mrf.mxu1  ;;  %v32 = vadd.f32 %v30, %v31 }

#allocation0 [shape = 'u32[144,128]', space=vmem, size = 0x12000]
"""


def test_bundles_ops_and_mxu():
    assert ir_stats.count_bundles(BUNDLES) == 5
    # 0x0: 1, 0x1: 2, 0x2: 3 (a continuation line), 0x1f: 2, 0x20: 3
    assert ir_stats.bundle_ops(BUNDLES) == 11
    mxu = ir_stats.mxu_ops(BUNDLES)
    assert mxu["total"] == 7
    assert mxu["by_op"] == {"vmatpush": 2, "vmatmul": 2, "vpop": 2, "vmatprep": 1}
    assert mxu["by_unit"] == {"mxu0": 4, "mxu1": 3}


def test_allocations_count_each_id_once():
    alloc = ir_stats.allocations(BUNDLES)
    assert alloc == {"count": 3, "bytes": {"smem": 0x200, "vmem": 0x12000 + 32768}}


def test_pass_of_the_dump_layouts(tmp_path):
    for rel, expected in {
        "0001-original.txt": (".", 1, "original"),
        "gdn_v2_fwd/0012-post-finalize-llo.txt": ("gdn_v2_fwd", 12, "post-finalize-llo"),
        "module_0042.jit_step.79-final-bundles.txt": ("module_0042.jit_step", 79, "final-bundles"),
        "x/fusion.3-02-original.llo": ("x/fusion.3", 2, "original"),
        "notes.txt": ("notes", None, "notes"),
    }.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        assert ir_stats.pass_of(path, tmp_path) == expected, rel


def test_summarize_dumps_groups_passes_and_names_kernels(tmp_path):
    llo = tmp_path / "llo"
    k = llo / "pallas_kernel_7"
    k.mkdir(parents=True)
    (k / "02-original.txt").write_text("func @gdn_v2_fwd(...)\n")
    (k / "13-post-mxu-assigner.txt").write_text("...\n")
    (k / "79-final.txt").write_text(BUNDLES)
    (llo / "fusion.12.02-original.txt").write_text("fusion\n")
    (llo / "fusion.12.40-final.txt").write_text(
        "  0x0  : { %v1 = vadd.f32 %v0, %v0 }\n  0x1  : { %v2 = vmatmul.f32.mxu0 %v1 }\n"
    )
    bwd = llo / "gdn_mega2_bwd_dw"
    bwd.mkdir()
    (bwd / "0001-original.txt").write_text("x\n")
    (bwd / "0002-final.txt").write_text("  0x0  : { %s = sadd %s0 }\n" * 3)
    mosaic = tmp_path / "mosaic"
    mosaic.mkdir()
    for i, name in enumerate(["original", "infer-layout", "post-finalize-llo"], 1):
        (mosaic / f"gdn_v2_fwd-{i:04d}-{name}.txt").write_text(
            "#allocation3 [shape = 'f32[8,128]', space=vmem, size = 0x1000]\n" if i == 3 else "x\n"
        )
    hlo = tmp_path / "hlo"
    hlo.mkdir()
    (hlo / "module_0001.jit_step.tpu_after_optimizations.txt").write_text(
        "HloModule jit_step\n"
        "%fused_computation (p: f32[8]) -> f32[8] {\n  %p = f32[8] parameter(0)\n}\n"
        "ENTRY %main {\n"
        "  %a = f32[8] parameter(0)\n"
        "  %f = f32[8] fusion(f32[8] %a), kind=kLoop, calls=%fused_computation\n"
        "  %g = (f32[8], f32[8]) fusion(f32[8] %f), kind=kOutput\n"
        '  %c = f32[8] custom-call(f32[8] %g), custom_call_target="tpu_custom_call"\n'
        "}\n"
    )
    (hlo / "module_0001.jit_step.tpu_after_optimizations-buffer-assignment.txt").write_text("x")
    out = ir_stats.summarize_dumps(tmp_path)
    groups = out["llo"]["groups"]
    assert set(groups) == {"gdn_v2_fwd", "gdn_mega2_bwd_dw", "fusion.12"}
    fwd = groups["gdn_v2_fwd"]
    assert (fwd["passes"], fwd["bundles"], fwd["mxu"]["total"]) == (3, 5, 7)
    assert fwd["pass_names"] == ["original", "post-mxu-assigner", "final"]
    assert fwd["ops_per_bundle"] == 2.2
    totals = out["llo"]["totals"]
    assert totals["kernel_bundles"] == {"gdn_v2_fwd": 5, "gdn_mega2_bwd_dw": 3}
    assert (totals["fwd_bundles"], totals["bwd_bundles"], totals["xla_bundles"]) == (5, 3, 2)
    assert totals["all_bundles"] == 10 and totals["mxu_ops"] == 8
    assert totals["kernel_vmem_bytes"]["gdn_v2_fwd"] == 0x12000 + 32768
    assert totals["llo_passes"] == 3
    m = out["mosaic"]["groups"]["gdn_v2_fwd"]
    assert m["passes"] == 3 and m["allocations"]["bytes"] == {"vmem": 0x1000}
    ((name, stats),) = out["hlo"].items()
    assert name.endswith("tpu_after_optimizations.txt")
    assert stats["fusions"] == 2
    assert stats["custom_calls"] == {"tpu_custom_call": 1}


def test_hlo_and_empty_dirs(tmp_path):
    assert ir_stats.summarize_dumps(tmp_path) == {}
    assert ir_stats.totals({})["llo_passes"] == 0
