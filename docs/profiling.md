# Profiling

How we profile a train step on a TPU v6e-1, and how we read the profile without the xprof
UI or a big machine. The capture protocol has four parts, and each one answers a different
question. `PROBE_ONLY=analyse` runs all four for every model size in one Colab session. The
named scopes that the tables group by are listed in [kernels.md](kernels.md#profiling).

## The capture protocol

| Capture | What | Answers | Used for timing? |
|---|---|---|---|
| **Timing run** | the training run itself, its `perf/step_seconds` rows (the steady steps: [kernel-journey.md](kernel-journey.md#cost)) | tokens/s, MFU | yes, the only source of timing |
| **Aggregate capture** | 3 traced steps after the warm-up, trace mode `TRACE_ONLY_XLA` with the perf counters (`xprof_summary.profile_options`) | where the step time goes: by scope, phase and op; FLOP and HBM rates; the measured counters | per-op times (the trace adds ~nothing to the device time) |
| **Detail capture** | 1 step with `TRACE_COMPUTE_AND_SYNC`, the kernels' named regions (`TJEV_KERNEL_REGIONS=1`), and `LIBTPU_INIT_ARGS` += `--xla_xprof_register_llo_debug_info=true` and `--xla_enable_custom_call_region_trace=true` (each only if libtpu accepts it; never runtime custom-call tracing: one traced 0.8B step made a 16.5 GiB XSpace) | the timeline: sync-flag and semaphore waits (DMA stalls in a Pallas pipeline), the regions inside the kernels, the LLO debug info for the static per-bundle analysis | **never**: the extra tracing slows the step |
| **HLO dump** | a compile-only `tjev compile-check` of the same step with `XLA_FLAGS="--xla_dump_to=… --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*"` and the compile cache off (a cache hit dumps nothing) | the executed program: fusions, layout copies, remat, the buffer assignment (what is live at the HBM peak) | no (static) |

The bench (`tjev campaign bench --profile DIR`) makes the same two captures for every
train-step variant it times; `PROBE_ONLY=bench` sets the detail flags first. A training run makes the aggregate one with
`train.profile_start` / `train.profile_steps`. The window traces `profile_steps + 1`
steps: `start_trace` runs before step `profile_start` and `stop_trace` after step
`profile_start + profile_steps`.

## The analyse probe

```bash
PROBE_ONLY=analyse CAMPAIGN=tjev-analyse1 MAX_SESSIONS=1 SESSION_HOURS=2 MIX=mix-v7.5 \
  MODELS="0.8B 2B 4B 9B" HF_DATASET=tychot/tjev-data TPU=v6e1 bash cloud/colab.sh session 1 probe
```

On a TPU host, run `PROBE_ONLY=analyse bash cloud/tpu_campaign.sh probe`. For each size, the
probe takes the planner's `final` line from the `v6e1` table. That line sets the microbatch
and the compute overrides of `plan.HARDWARE["v6e1"]["extra"]`: the megakernel at chunk 128,
the blocked inverse, `core_states` remat, and merged LoRA up to 4B. The probe sets LoRA
rank 16 (the finals' rank) and writes two queue lines (`python -m tjev.campaign.analyse
line`): the aggregate run, 25 steps with the last 3 before the tail traced, and the detail
run, one traced step. Neither line checkpoints or evaluates, so no eval program enters the
window, and both precompile every bucket. For another rank, more overrides or the layers
traced one by one (`--unroll`, `TJEV_UNROLL_LAYERS=1`), run `tjev.campaign.analyse line`
by hand and queue its output.

| Variable | Default | |
|---|---|---|
| `PROBE_ANALYSE_SIZES` | `0.8B,2B,4B,9B` | |
| `PROBE_HF_REPO` | `tychot/tjev-4b` | private (checked before each upload): capture files over 150 MB go there under `analyse/<campaign>/<size>/{aggregate,detail}/` (a session save restores only files under ~150 MB); needs `HF_TOKEN` (set by `HF_DATASET`) |

The probe installs `xprof` into the venv if it is missing, and probes the two detail flags.
Then, per size:

1. The saved-residual listing of every bucket and remat policy (`<size>/residuals.json`,
   `python -m tjev.campaign.residuals`; tracing only, seconds). The summaries use it to
   name the scan residual buffers.
2. The aggregate run (`runs/analyse-<size>-aggregate`) and its summary in `<size>/aggregate/`.
3. The detail run (`runs/analyse-<size>-detail`) with the detail flags, and its summary in
   `<size>/detail/`.
4. The kernels' LLO capture: `xprof check_kernel_profiling` (its capture gates:
   `capture_valid_for_llo_analysis`, the Pallas Primitives line) and `xprof
   get_llo_analysis` (all kernels, and `gdn_mega2_fwd`, `gdn_mega2_bwd`, `splash` alone)
   on the detail capture. They come from xprof-nightly, installed in its own venv
   (`$ROOT/xprof-nightly-venv`, with tensorflow-cpu), and write
   `<size>/detail/{check_kernel_profiling,llo_analysis*}.json`. If the install fails, the
   xplane is kept for a later run.
5. `tjev compile-check` with the dump flags. Only `*after_optimizations*`,
   `*buffer-assignment*` and `*memory-usage-report*` are kept, in `<size>/hlo.tgz`.
   `<size>/compile-check.json` holds the XLA memory analysis per bucket.
6. The offload: every capture file over the limit (the xplane and trace JSON of a 9B
   capture) goes to the HF repo. A `<file>.hf` pointer stays in its place.

Each summary directory holds `report.md` / `report.html` (the readable page, see
[Readable names](#readable-names)), `summary.json`, `scope_phase.txt` (the table below),
`train_program.hlo.txt.gz` (the main train program's optimized HLO),
`hlo_stats.json.gz`, `roofline_model.json.gz`, `overview_page.json`,
`utilization_viewer.json`, `memory_profile.json`, `memory_viewer.json.gz`,
`framework_op_stats.json`, `perf_counters.json.gz`, and its W&B run (`tjev-benchmarks`,
job type `profile`) with the tables.

**Cost.** The estimate below is built from the measured speeds: 9B 7.1 s per 65k-token
step, 4B 4.6 s, 2B 1.8 s, 0.8B 1.25 s. A cold precompile of the three buckets takes
~2-2.5 min per process at 4B and 9B, ~1.5 min at 0.8B and 2B (tjev-journey3,
tjev-smoke9b). Each size compiles three times: the aggregate run, the detail run (other
libtpu flags) and the uncached dump. That gives about 8 / 10 / 13 / 16 min for
0.8B / 2B / 4B / 9B. With the bootstrap (venv, ~32 GB of model snapshots), the flag
probes and the save, the session takes ~1.0-1.1 h. Add ~3 min for the xprof-nightly venv
and an unmeasured few minutes per size for the residual listing and the LLO analyses
(each capped by a timeout). The total is **~1.2-1.4 h**, so **~5-6 CU** (v6e-1 at
4.08 CU/h). `SESSION_HOURS=2` caps it at 8.2 CU. At the deadline the queue pauses
(exit 3) and the probe stops after the current size.

## Readable names

What the xprof UI shows and where each name comes from (jax 0.11.2, xprof 2.23):

| What you see | Where it comes from | What we set |
|---|---|---|
| **Program** (`jit_step(<id>)`) | the HLO module name is `jit_<fn.__name__>` (`jax/_src/interpreters/pxla.py`: `module_name = util.wrap_name(api_name, fun_name)`, `fun_name` = `__name__`). xprof appends the program id | every jitted step gets a name per role and shape (`train.step.NamedJit`: one jitted copy per name): `jit_train_step_T1024_mb32`, `jit_eval_step_T4096_b16`. compile-check and the bench name theirs the same way |
| **Op path** (`tf_op_name`, "framework op" in hlo_stats and framework_op_stats) | the `op_name` metadata, which is the jax name stack: `jit(<program>)` + `jax.named_scope`s + autodiff and loop wrappers + the primitive (`jax/_src/interpreters/mlir.py`: a `NameLoc` per equation) | the scopes below |
| **HLO instruction** (`multiply_add_fusion.222`, `custom-call.133`) | XLA names an instruction after its opcode and its fusion's contents. Only parameters take their name from jax: the argument's pytree path (`frozen__head___value.1`). No jax API or XLA flag sets the others | none: our report puts the op path and a label next to each one |
| **Category** (`convolution fusion`, `loop fusion`) | XLA's HLO category; op_profile groups by program or by category (`raw_to_tool_data`, `group_by`) | none: group by scope in our tables, or in framework_op_stats |
| **Buffers XLA creates** (`custom-call.119 … 133` at the memory peak) | `AllocateBuffer` custom calls for the scan's stacked outputs, without metadata | labelled by our report (below) |

**Scopes.** Each decoder layer is named by its position in the super-block and its mixer
(`layer0_gdn` … `layer2_gdn`, `layer3_attn`), and every projection names its ops: `…/layer1_gdn/mlp/up_proj/base`,
`…/mlp/gate_proj+up_proj/lora/A` (the shared x·[A₁|A₂]), `…/mlp/up_proj/lora/B`, and with
merged LoRA `…/up_proj/lora/merge` (its own coarse scope, `lora_merge`). The other scopes are those of
[kernels.md](kernels.md#profiling) plus `residual` (the residual-stream adds). The coarse
tables still group by the innermost scope (`base`, `lora`, `mlp`, …), and `top_paths`
gives the full path. The kernels' names come from their Pallas calls
(`gdn_mega2_fwd/pallas_call`, `splash_mqa_fwd_segmented_residuals`). These are metadata
only: a tiny model's train step (2 steps, 4 remat / LoRA configurations, f32 and bf16)
gives bit-identical losses, gradients, updated adapters and eval logits before and after.

**Residual names.** The remat policies save values by `checkpoint_name` (`gdn_states`,
`gdn_tinv`, `gdn_core_out`, `attn_context`, `mlp_up`, `lora_xa`, …). The `name`
primitive lowers to nothing, so these names are absent from the HLO. The scan's stacking
ops read `…/while/body/jit(dynamic_update_index_in_dim)/dynamic_update_slice` whatever they
store, and the stacks themselves carry no metadata. Two things recover them:

* `python -m tjev.campaign.residuals ITEMS --policies all` lists each layer's saved residuals
  per bucket and remat policy, on an abstract model (`jax.ad_checkpoint.print_saved_residuals`'s
  function, `jax._src.ad_checkpoint.saved_residuals`). The layer input appears there as
  "from the argument h": `layer_input`.
* `hlo_labels.label_allocations` follows each `AllocateBuffer` into its `while` loop to the
  value the body writes (`gdn_mega2_fwd.25`, `reduce_precision` of `gdn_core_out`, …) and
  names the buffer by its stacked shape `[super-blocks, *residual]`.

On the 9B capture (core_states, T = 1024, 2 rows) that gives:

| Buffer | Label | Size |
|---|---|---|
| `custom-call.119`, `.120`, `.123`, `.124`, `.127`, `.128` | scan residual `gdn_states` / `gdn_tinv`, written by `gdn_mega2_fwd.24/.25/.26` (one pair per GDN layer) | 6 × 256 MiB |
| `custom-call.122`, `.126`, `.130`, `.133` | scan residual `layer_input` (one per layer of the super-block) | 4 × 128 MiB |
| `custom-call.121`, `.125`, `.129` | scan residual `gdn_core_out` | 3 × 128 MiB |
| `custom-call.131` | scan residual `attn_context` (splash's output), written by `splash_mqa_fwd_segmented_residuals.8` | 128 MiB |

The 2.5 GiB of scan residuals sits on top of 16.7 GiB of frozen weights in the 21.1 GiB
peak. The listing gives 1.0 GiB for `core`, 0.5 GiB for `full` and 6.0 GiB for `minimal`
at this bucket.

**Per layer.** Under the scan every layer shares its ops and buffers with the other
super-blocks (8 executions of one body). `TJEV_UNROLL_LAYERS=1` traces the super-blocks
one by one instead: each op gets an `L00` … `L31` scope, and each layer gets its
own buffers. The math is the same (equal to float32 rounding on the tiny model), but the
compile grows with depth and XLA fuses differently, so it is for a detail capture
(`tjev.campaign.analyse line --unroll`), never for timing.

**Ops named after their loop.** XLA's layout copies, transposes and loop plumbing inside
the layer scans inherit the enclosing `while`'s name (`…/transpose(jvp())/while`), which
puts them under `grad_accum`. `reattribute_rows` gives each one the path of its nearest
named producer, else consumer, in the program's HLO, with a suffix for what it does:
`…/layer1_gdn/gdn_out/mul/relayout`, `…/scan stack write`, `…/scan stack read`. The
original name stays in `tf_op_original`, and `reattributed_ms` reports the total.
`by_kind` names xprof's categories in our words: matmul, Pallas kernel, copy / relayout,
scan stack write (DUS) and read (DS), elementwise.

**Kernel regions.** With `TJEV_KERNEL_REGIONS=1` (the detail capture sets it), the GDN
kernels (the megakernel, and v2, which shares its chunk step) mark their phases with
`jax.named_scope` inside the kernel:

| Kernel | Phases | Inside |
|---|---|---|
| megakernel forward | `prologue` (loads, conv + SiLU), `chunk`, `state_update`, `store` | `chunk` → `decay`; `state_update` → `inverse`, `state_read`, `output`, `state_write` |
| megakernel backward | `prologue`, `chunk`, `state_bwd`, `qk_norm_bwd`, `conv_bwd`, `store` | `state_bwd` → `v_new`, `d_vnew`, `d_decay`, `state_grads`, `qk_grads`, `gate_grads` | In a
Pallas TPU kernel those lower to `tpu.trace_start` / `trace_stop`, and the detail flag
`--xla_enable_custom_call_region_trace` shows
them in the timeline (`jax/_src/pallas/mosaic/lowering.py`: `trace_start` /
`trace_stop` on the name-stack boundaries). Without the variable the kernels are
unchanged. Checked in interpret mode against the previous code: the forward + backward
jaxpr and the values are identical for the megakernel and for v2, and the values stay
identical with the regions on.

**The window.** The loop now blocks on the previous step before `start_trace`, so no
truncated tail opens the window. The evals run under a `TraceAnnotation("eval")`. The
`compiled` events log each shape's program name (`"program": "train_step_T1024_mb32"`).

**The report.** `xprof_tools --out DIR` writes `report.md` and `report.html`:
* the programs by name;
* the scope × phase table;
* the top model paths and the top HLO ops, each with its op path, phase, rates and bound-by;
* the HBM peak by labelled group: frozen weights, LoRA and optimizer state, each scan
  residual by name, the loop buffers by writer, the temporaries by scope;
* the largest buffers at the peak, labelled;
* the scan buffers with their writers;
* the per-layer residual listing (with `--residuals residuals.json`).

## The headless tools

The xplane (`*.xplane.pb`, 194 MB for a 9B capture) is the record. Our old summariser
loaded the trace JSON (3.7 GB of text for the same capture) with `json.loads`, which does
not fit in 18 GB of RAM. Stable `xprof` (`pip install 'tjev[profile]'`, xprof ≥ 2.23)
converts the xplane into the UI's tables in Python, offline: ~5-15 s and ~3 GB of RAM for
the 9B capture.

```bash
python -m tjev.campaign.xprof_tools PROFILE_DIR                 # the scope × phase table + key numbers
python -m tjev.campaign.xprof_tools PROFILE_DIR --out DIR       # + summary.json and the raw tables
python -m tjev.campaign.xprof_tools PROFILE_DIR --programs all  # every program (evals in the window too)
```

`PROFILE_DIR` can be a run's `profile/`, an xprof session directory or the xplane
itself. A `detail/` capture is skipped unless it is the only one. In Python:

```python
from xprof.convert import raw_to_tool_data as r
xs = [".../plugins/profile/<ts>/<host>.xplane.pb"]
r.xspace_to_tool_names(xs)        # also writes the per-module HLO protos memory_viewer needs
data, _ = r.xspace_to_tool_data(xs, "hlo_stats", {})   # Google DataTable JSON {cols, rows}
# roofline_model, overview_page, op_profile ({"group_by": "program"}), framework_op_stats,
# memory_profile, perf_counters, utilization_viewer,
# memory_viewer ({"module_name": "jit_train_step_T1024_mb32(<program id>)", "memory_space": "0"})
# graph_viewer ({"graph_viewer_options": {"node_name": "", "module_name": …, "type": "long_txt"}})
```

The converters cache their results next to the xplane (`ALL_HOSTS.op_stats_v2.pb`,
`<module>.hlo_proto.pb`).

xprof-nightly (≥ 2.24, with `tensorflow` for `hlo_pb2`) adds a CLI, `xprof <command>
<session_dir>`:

| Area | Commands |
|---|---|
| Overview | `get_overview`, `get_avg_step_time`, `get_device_information` (peaks, ridge points) |
| Per-op | `get_top_hlo_ops`, `get_hlo_op_profile`, `get_roofline_model` |
| Hardware | `compute_utilization` (measured MXU and HBM bandwidth utilisation from the counters), `get_utilization_viewer`, `get_perf_counters` |
| Memory | `get_peak_allocations`, `get_memory_profile` |
| HLO | `list_hlo_modules`, `get_hlo_text --path`, `get_hlo_neighborhood`, `detect_layout_mismatch_copies` |
| Kernels | `get_llo_analysis`, `get_kernel_stats`, `check_kernel_profiling` |

Without xprof, `xprof_summary.summarize` reads the trace JSON as a fallback (no phases, no
roofline, no counters). It streams the file in 16 MB chunks (`json.JSONDecoder.raw_decode`)
and keeps only the metadata, TPU:0's "XLA Ops", "Steps" and "XLA Modules" rows and the host
step markers, with the args a summary reads. On the 9B trace that is ~30 s and 2.3 GB. The
bench (`_profile_summary`) and the journey (`collect`) call `xprof_tools.summarize_any`,
which uses the xplane when xprof is installed and the trace otherwise. Their W&B runs get
the tables: scope × phase, the top ops with their roofline, and the buffers at the peak.

## How to read a summary

* **Programs.** A capture can hold several XLA programs: the train step, a second train step
  variant (the health statistics), an eval. A program counts as a train step when one of
  its ops has a backward path (`grad_accum`, `transpose(`). The step tables count only
  those programs. `all_programs` keeps the shares over all of them. The steps are the train
  programs' executions, from the memory profile's module timeline (`step_source`).
* **`step_ms`.** The wall time per annotated step (`StepTraceAnnotation`, from
  `overview_page`). `device_ms_per_step` is the train programs' device time per execution.
  `busy_fraction` is the duty cycle.
* **Scopes.** Each op goes to the innermost model scope in its `tf_op_name`
  ([kernels.md](kernels.md#profiling)), else `other` (loop plumbing, unnamed copies).
* **Phases**, from the jax path of a train program's op:
  * `bwd`: `…/transpose(jvp(…))/…`;
  * `remat`: `…/checkpoint/rematted_computation/…`, the forward recomputed for the
    backward (it matches hlo_stats' rematerialisation flag);
  * `fwd`: `…/jvp(…)/…`;
  * `other`: the optimizer, the health reductions, unnamed ops.
* **Rates.** `scope_stats[s].tflops` and `.hbm_gbps` are time-weighted, Σ(rate × self
  time) / Σ self time, i.e. the scope's model FLOPs (or HBM bytes) over its measured
  time. `bound_by` gives the share of the scope's time spent in ops xprof labels Compute,
  HBM, VMEM Read/Write or Unknown (a custom call without a `CostEstimate`).
* **`top_ops`** holds the top HLO ops with their roofline (FLOP rate, HBM rate, intensity,
  roofline efficiency, DMA-stall share). **`top_kinds`** holds the same op kind (its name
  without the numeric suffix) summed within one scope and phase.
* **`memory`.** The static HBM peak of the main train program (memory viewer): the
  parameters, the temporaries and the largest buffers live at the peak. `memory_dynamic`
  is the allocator's peak over the window (memory profile).
* **`device`.** The peaks and ridge points xprof uses for v6e: 946.7 TFLOP/s, 1638 GB/s
  HBM, an HBM ridge at 578 FLOP/byte.

### Estimated vs measured

`summary.json` lists which field is which under `fields`.

| Kind | Fields | Source |
|---|---|---|
| **measured** | op self times, hence every ms, share, phase and step time; occurrences; `measured.*` | the trace's device events; the session's hardware counters (utilization viewer: MXU busy cycles, HBM bytes). The counters cover the whole session, idle time and other programs included |
| **estimated** | `tflops`, `hbm_gbps`, `intensity`, `bound_by`, `roofline_efficiency`, `program.*`, the trace fallback's `hbm_gb_*` | XLA's cost model (FLOPs, bytes accessed) divided by the measured time |
| **static** | `memory.*` | the compiler's buffer assignment |

The cost-model bytes overcount. On the 9B capture `grad_accum` reads at 1.90 TB/s, above
the 1.64 TB/s peak. A dynamic-update-slice that is not the fusion root counts the whole
buffer, and the "memory BW" column includes VMEM traffic. Pallas kernels count only what
their `pl.CostEstimate` says. On v6e the per-op HBM bandwidth is always model-derived:
per-op counter sampling exists only from TPU v7 on. Judge HBM-boundness on the HBM
column against the ridge point. Use the measured session counters
(`measured.hbm_gbps`, `measured.mxu_busy_fraction`) for the whole step.

### The 9B capture (tjev-smoke9b, v6e-1, rank 16, 2048-token microbatches)

The capture holds 4.9 train-step executions and 191 executions of 3 eval programs (a
quick eval fell inside the window). The 0.9 comes from one execution that the trace's
stop cut off (`executions_from_modules`). Over every program, the shares match the old
trace-JSON summary: base 54.0%, gdn_rule 14.6%, lora 14.2%, gdn_out 5.2%, grad_accum
4.2%, with base at 701 TFLOP/s and compute-bound.

The train step alone, in ms per step, is below. It moves the 222 ms per step of ops that
XLA named after their loop (layout copies) to their neighbours' scopes: `grad_accum`
drops from 6.0% to 3.0%, and the rest goes to gdn_out, gdn_rule and base.

```
scope                 fwd      bwd    remat    other        ms      %  TFLOP/s  HBM GB/s  bound_by
base               1294.4   1337.8   1048.5      0.0    3680.7  53.27    681.8     784.7  Compute 92%, HBM 8%
lora                242.5    690.6    176.2      0.0    1109.3  16.05     20.7    1282.3  HBM 96%, VMEM Read 4%
gdn_rule            444.2    489.7      3.5      0.4     937.8  13.57    130.0     335.1  HBM 50%, Compute 49%
gdn_out             113.2    176.2    146.0      0.0     435.4   6.30      0.7    1459.7  HBM 100%, VMEM Write 0%
grad_accum          116.5     73.6      0.0     14.2     204.3   2.96      0.0    1421.0  HBM 99%, VMEM Write 0%
attention_core       55.0    105.4     19.4      0.2     180.0   2.61      0.3     245.3  Unknown 75%, HBM 24%
attn_proj            36.3     78.4     47.2      0.2     162.2   2.35      0.7     852.8  HBM 93%, VMEM Write 5%
norm                  4.1     76.5     14.6      0.0      95.3   1.38      2.5    1842.5  HBM 97%, VMEM Write 1%
attn_out             18.9     10.8     34.3      0.0      64.0   0.93      0.7    1244.1  HBM 100%, VMEM Read 0%
…
total              2357.9   3045.2   1489.8     16.6    6909.5 100.00    384.3     873.0
```

Measured over the session:
* the MXUs were busy 49.8% of the cycles;
* the HBM moved 44.7 TB at 55.7% of the peak (913 GB/s).

The static HBM peak of the train program is 21.1 GiB:
* 16.7 GiB of frozen weights;
* 0.45 GiB of adapters and Adam state;
* 2.5 GiB of scan residuals;
* the temporaries.

The remat recompute costs 22% of the step (1.5 s), 70% of it in the base matmuls. By kind
of op: matmuls 67.5%, Pallas kernels 14.5%, fused elementwise 7.4%, copies and relayouts
5.3%, scan stack writes and reads 3.5%.

## Sources

* xprof CLI and converters: the `openxla/xprof` repository, `plugin/xprof/cli/README.md`,
  `plugin/xprof/server.py` (CLI subcommands from xprof-nightly ≥ 2.24.0) and
  `xprof/convert/raw_to_tool_data.py` (the tool names and params). Tool pages:
  https://openxla.org/xprof and https://openxla.org/xprof/hlo_op_stats ("Model GFLOP/s …
  computed by the XLA compiler, time … measured by the profile"; HBM BW counts HBM only).
* Utilization viewer and perf counters: https://openxla.org/xprof/utilization_viewer.
* What the cost model counts: `xla/service/hlo_cost_analysis.cc` (in-place DUS, fusion
  parameters, custom calls without a cost estimate).
* Per-op counter sampling from TPU v7, the LLO debug info, custom-call tracing,
  `TRACE_COMPUTE_AND_SYNC` and the DMA-stall signature: the xprof deep-kernel-profiling
  post,
  https://opensource.googleblog.com/2026/06/unlocking-tpu-performance-deep-kernel-profiling-with-xprof.html,
  and its notebook, https://github.com/openxla/xprof/blob/master/docs/xprof_kernel_demo.ipynb.
* Program and op names: `jax/_src/interpreters/pxla.py` (`module_name = wrap_name(api_name,
  fun_name)`), `jax/_src/util.py` (`fun_name`: `__name__`), `jax/_src/interpreters/mlir.py`
  (a `NameLoc` from the name stack per equation: the `op_name` metadata),
  https://docs.jax.dev/en/latest/_autosummary/jax.named_scope.html; saved residuals:
  `jax/_src/ad_checkpoint.py` (`saved_residuals`, `print_saved_residuals`).
* `jax.named_scope` inside Pallas TPU kernels lowers to `tpu.trace_start` / `trace_stop`
  (`jax/_src/pallas/mosaic/lowering.py`).
* HLO dumps: https://openxla.org/xla/hlo_dumps; per-jit compiler options:
  https://docs.jax.dev/en/latest/201/controlling-xla.html.
* Hosted xprof for large profiles (cloud-diagnostics-xprof):
  https://cloud.google.com/blog/topics/developers-practitioners/supercharge-ml-performance-on-xpus-with-the-new-xprof-profiler-and-cloud-diagnostics-xprof-library.
* Pallas performance practice: https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html.
