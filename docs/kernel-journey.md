# The kernel journey

Every iteration of the Gated DeltaNet core, from the XLA chunked path to the megakernel
with merged LoRA, measured the same way: the real LoRA training of Qwen3.5 on a TPU v6e-1.
Each iteration gets a real `tjev train` run, a correctness check on one fixed batch, an
xprof capture, and the compiler's view of its kernels (VLIW bundles, MXU ops, VMEM). The
article style follows ["When XLA isn't enough"](https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas)
(Pallas → Mosaic → LLO → bundles, HBM traffic). Here the static numbers come with real
timings, profiles and correctness.

## The iterations

The iterations are `tjev campaign bench` variant names (`bench.VARIANTS`); the probe runs
them in this order (`JOURNEY_VARIANTS` in `cloud/tpu_campaign.sh`):

| Variant | What changes | Compute overrides |
|---|---|---|
| `xla-nobarrier` | XLA chunked, the GDN conv fused into the core: XLA:TPU miscompiles it ([results](results.md#tpu-correctness-v6e-1-measured)) | `attention=xla gdn_impl=chunked`, env `TJEV_GDN_CONV_BARRIER=0` |
| `xla` | the same with optimization barriers around the conv | `attention=xla gdn_impl=chunked` |
| `kernels` | Pallas v1: the fused forward kernel, the reverse recurrence backward, splash (the preset) | — |
| `split` | Pallas v1, split: the chunk-local terms in XLA around the Pallas recurrence | `gdn_impl=pallas_tpu_split` |
| `fused-bwd` | Pallas v2: fused forward and backward | `gdn_impl=pallas_tpu_v2` |
| `fused-bwd+states` | v2 + the `core_states` remat (saves the kernel residuals) | `+ remat=core_states` |
| `mega1` | megakernel 1: gates, l2norm, GQA in the kernel | `gdn_impl=pallas_tpu_mega1 remat=core_states` |
| `mega2` | megakernel 2: the conv and SiLU inside too | `gdn_impl=pallas_tpu_mega2 remat=core_states` |
| `mega2-c128-blocked` | chunk 128, blocked in-kernel inverse (the sweep's path) | `+ gdn_chunk=128 gdn_inverse=blocked` |
| `mega2-c128-merged` | + merged LoRA (`y = x(W + s·AB)`) | `+ lora_impl=merged` |

The integration runs also measured `mega2-c128-fused`, the sweep's path with the exact
fused LoRA kernels instead of merged LoRA ([kernels](kernels.md#fused-lora-removed)):
slower (−16% at 4B, −7% at 9B), so the kernel was removed. The speed-ups the planner adopted after
these iterations (best-fit packing, the flat GDN norm, the 9B letters head and per-bucket
remat, the tuned splash tiles) are in [training.md](training.md#integration-on-v6e1-what-was-adopted).

A variant may also set environment variables (`bench.VARIANT_ENV`; only `xla-nobarrier`
does). The bench and parity runners apply them in-process, and a training job gets them
as `env=KEY=VALUE` tokens on its queue line.

## Running it

```bash
PROBE_ONLY=journey CAMPAIGN=tjev-journey1 MAX_SESSIONS=1 MIX=mix-v4 MODELS="0.8B 4B" \
  TPU=v6e1 bash cloud/colab.sh session 1 probe
```

On a TPU host, run `PROBE_ONLY=journey bash cloud/tpu_campaign.sh probe`.

The one setting is `PROBE_JOURNEY_SIZES` (default `0.8B,4B`). The rest is fixed: 20
training steps per size (65,536 tokens each), every variant above at 0.8B and the five key
iterations at 4B (`xla`, `fused-bwd+states`, `mega2`, `mega2-c128-blocked`,
`mega2-c128-merged`), the parity check and the IR dumps (`xla`, `fused-bwd`, `mega1`,
`mega2`, `mega2-c128-blocked`) at 0.8B, a 3-step xprof capture per run, the final plan's
3000-step horizon, and the raw traces kept.

`probe` empties `reports/probe/` when it starts. To split the journey over two sessions,
run one size per session (`PROBE_JOURNEY_SIZES=0.8B`, then `4B`). Then merge both
`journey.json` files in the report:

```bash
python scripts/journey_report.py s1/reports/probe/journey/journey.json \
  s2/reports/probe/journey/journey.json --out article/
```

## What runs

1. **The queue.** For each size, the `final-<size>-h<horizon>` line of
   `tjev campaign plan final --sizes SIZE` is the long final run (the planner's recipe:
   rate, rank, frozen gates, the mix, `--hardware`). `python -m tjev.campaign.journey
   final-line` picks it by its run name, never by its position, and `queue` refuses a line
   of another size (its name or `model.path`). tjev-journey2 took the first line of a plan
   whose `--sizes` the `plan()` wrapper had overridden with `$SIZES`: its five "4B" runs
   trained 0.8B. `python -m tjev.campaign.journey queue` drops every
   `compute.gdn_* / attention* / lora_impl / remat` override and appends the variant's. It
   then sets the following:
   - `train.steps` = 20 (the journey step count);
   - `train.checkpoint_every=0`: no checkpoints and no full evals;
   - `train.quick_eval_every=0`, `train.health_kill=false`, `train.log_every=1`;
   - `train.precompile_buckets=true`: before step 1, each bucket's train step is compiled
     and run once on an all-padding batch (zero loss and gradient) on copies of the
     adapters and optimizer state, then discarded. The real steps should then hit the jit
     cache; the compile time per shape comes from these step-0
     `compiled` events (`seconds`: the dispatch, i.e. trace + compile; `run_seconds`: with
     the dummy step's run);
   - a profile window of 3 steps, 5 steps before the end, in detail mode
     (`TJEV_PROFILE_DETAIL=1`), when every bucket has compiled;
   - cold compiles (`JAX_ENABLE_COMPILATION_CACHE=false`), so every run's compile time is
     a fresh process's, and `JAX_EXPLAIN_CACHE_MISSES=true`, so a step that compiles
     despite the precompile says why;
   - W&B project `tjev-benchmarks`, group `journey`, tags `journey`, the size and
     `variant:<name>`;
   - `chips=1 post=none`.

   The real data comes from the mix stream with its three buckets (1024 / 2048 / 4096).
2. **Correctness first.** It is cheap, and it gives the article its "fast but wrong" row.
   `tjev campaign parity --lengths 4096 --batches 1` takes one real batch of the 4096-token
   bucket. On that batch, every variant's loss, label logits and LoRA gradients are compared
   against the XLA reference with the conv barriers (`gdn_precision=highest`). The pass rule
   is parity's: the gradient error is at most 2× the production XLA path's (floor 3e-2),
   the cosine is ≥ 0.9995, and the loss is within 5e-3. Each variant costs one compile:
   about a minute at 0.8B.
3. **The IR.** `python -m tjev.campaign.journey ir` compiles the GDN core once per variant,
   forward and backward, without running it. It is the model's own `GatedDeltaNet.core`
   under the variant's ComputeSpec, at the training shape (rows × 4096 tokens, 16 packed
   segments). Each compile runs in its own process, because libtpu reads its flags once.
   The dumps on:
   - XLA: `--xla_dump_to=…/hlo --xla_dump_hlo_as_text` (compile options);
   - libtpu (each candidate probed first, since an unknown flag aborts libtpu):
     `--xla_mosaic_dump_to`, `--xla_mosaic_enable_dump_debug_info`,
     `--xla_mosaic_enable_llo_source_annotations`, `--xla_jf_dump_to`,
     `--xla_jf_dump_llo_text`, `--xla_jf_dump_hlo_text`, `--xla_jf_dump_llo_html=false`,
     `--xla_jf_dump_llo_static_gaps`, `--xla_jf_emit_annotations`, `--xla_jf_debug_level=2`.
     The accepted and rejected ones are recorded in `ir/ir.json`.

   `tjev.campaign.ir_stats` parses the dumps. Per kernel (`gdn_v2_fwd`, `gdn_mega2_bwd`,
   … the `name=` of each `pallas_call`) it reports:
   - the LLO and Mosaic pass counts;
   - the VLIW bundles of the last pass (the lines that start with a bundle address,
     `0x1f  : > { … ;; … }`);
   - the ops per bundle;
   - the MXU ops by family and unit (`vmatmul`, `vmatpush`, `vmatprep`, `vpop`; `mxu0` /
     `mxu1`);
   - the allocations per memory space (`#allocationN [… space=vmem, size = 0x…]`).

   Everything outside the kernels counts as XLA bundles. From the HLO it reports the
   fusions and custom calls, and the XLA cost analysis gives the bytes accessed.
4. **The training runs.** `tjev campaign queue --retries 0` runs them one at a time. A
   failure (an OOM, a compile error) is a result: `collect` reports it.
5. **The reports.** `python -m tjev.campaign.journey collect` reads each run's
   `metrics.jsonl`, `stdout.log`, `config.json` and `profile/`:
   - **Step time:** p10 / p50 / p90 / mean, overall and per sequence bucket, over the
     steady steps only (`journey.steady_steps`). It leaves out the rows that include a
     compile, the profiled steps with their neighbours, the first row and the last row.
     The loop times a row as the wall interval between two metric fetches, each made after
     the next step's dispatch, so the first row holds the dispatch of steps 1 and 2 (and
     any compile there, whatever its `perf/includes_compile`), and the last row waits for
     its own step (no next step to overlap). `warmup_s` / `warmup_excess_s` keep the first
     row's time and its excess over the median.
   - **Throughput:** real tokens/s (non-padding) and padded tokens/s, as totals over the
     timed steps, the pad fraction, and the speed-up over `xla` at the same size.
   - **MFU** (model FLOPs of the real segment lengths), TFLOP/s and **peak HBM**.
   - **Compile time per bucket shape:** the dispatch time of the step that first saw the
     shape. The loop logs it as `{"event": "compiled", …}`.
   - **The xprof summary** (`xprof_tools.summarize_any`, [profiling.md](profiling.md)):
     device time by model scope and phase (share and ms per step), by HLO category, and
     the top ops with their roofline, from the xplane when xprof is installed. Without
     xprof it reads the trace JSON (`xprof_summary.summarize`): no phases, plus the
     cost-model bytes per step by scope and per top op when the trace args carry them.
   - **Correctness** from `parity.json`, and the IR from `ir/ir.json`.

   It writes `reports/probe/journey/<size>-<variant>.json` and `journey.json`.
   `scripts/journey_report.py` turns those into `report/journey.md` (the article's tables)
   and four PNGs: tokens/s per iteration, step time by scope stacked, peak HBM, MFU. The
   PNGs need matplotlib, the `report` extra; the probe installs it if missing. `journey.py
   log` sends the numbers, tables, bar charts and PNGs to W&B. The raw traces and dumps
   never go to W&B (the free plan has 5 GB). They come back with the session download
   under `reports/probe/journey/` (`runs/*/profile`, `ir/<size>-<variant>/`).

## Cost

The estimate uses the measured speeds below and 65,536 tokens per step
(`python -m tjev.campaign.journey estimate`, defaults `--steps 0.8B:20,4B:20`, all variants
at 0.8B and `--variants-4b` the five key ones). At 0.8B the speeds come from
docs/results.md (bench3 / bench4) and the s0 bench for Pallas v1. At 4B, the sweep path
was measured at 14.1k; the other variants are scaled by their 9B ratio to mega2-c128.
`fused-bwd` without `core_states` is assumed at 0.9× the v2 + states speed. `xla-nobarrier`
is assumed as fast as `xla`. Each run trains its steps plus one precompile step per bucket
(3), and adds a fixed overhead: start, model load, three cold compiles, trace and summary
(4 min at 0.8B, 6 min at 4B). The overhead is most of it.

| Variant | 0.8B tok/s | 0.8B, 20 steps | 4B tok/s | 4B, 20 steps |
|---|---|---|---|---|
| xla-nobarrier | 13,000 | 1.9 + 4 min | | |
| xla | 13,000 | 1.9 + 4 | 5,000 | 5.0 + 6 |
| kernels | 14,600 | 1.7 + 4 | | |
| split | 14,000 | 1.8 + 4 | | |
| fused-bwd | 25,300 | 1.0 + 4 | | |
| fused-bwd+states | 28,100 | 0.9 + 4 | 10,900 | 2.3 + 6 |
| mega1 | 34,200 | 0.7 + 4 | | |
| mega2 | 40,100 | 0.6 + 4 | 12,600 | 2.0 + 6 |
| mega2-c128-blocked | 41,800 | 0.6 + 4 | 12,900 | 1.9 + 6 |
| mega2-c128-merged | 52,700 | 0.5 + 4 | 14,100 | 1.8 + 6 |
| **Total** | | **0.86 h** | | **0.72 h** |

Parity at 0.8B adds ~11 min (11 variants), and the IR dumps add ~15 min (flag probe plus
5 compiles). **In total: ~2.0 h ≈ 8.2 CU** at 4.08 CU/h. Options:

- **All ten variants at 4B too** (`JOURNEY_KEY_VARIANTS=$JOURNEY_VARIANTS` in the script,
  `estimate --variants-4b ''`): +5 runs of ~9–11 min, ~+0.8 h.
- **No `xla-nobarrier` training run** (parity alone shows "wrong"): −6 min.
- **One size per session.** Each half fits a Colab session comfortably.

## What to check on a new libtpu

- **The LLO / Mosaic dump layout.** A new libtpu may name its files differently from
  the article's layout. `ir_stats` groups files by name without the pass index, and it names a
  group after the Pallas kernel the group mentions. Check `ir/ir.json` (`groups`,
  `last_pass`) once, and adjust `ir_stats.LEADING` / `INNER` / `KERNEL` if a kernel shows
  up as an anonymous group.
- **Which dump flags libtpu accepts.** They are probed and recorded.
  `--xla_jf_debug_level=2` may make the dumps large; drop it from `ir_flag_candidates` if
  so.
- **Whether the TPU trace args carry `bytes_accessed`.** Without them, the HBM-per-op
  fields are absent. The XLA cost analysis of the IR compile still gives the bytes of the
  GDN core.
- **Detail-mode trace size and summarising time at 4B.** Delete `runs/*/profile` from the
  session save if the download gets too big.

## How many steps

Timing statistics need the steady state, not many steps: the step time on a v6e-1 varies by
~1% between steps (the 4B smoke run: 4.50–4.64 s), so 20 steady steps give the median to
±0.3%. What a short run must not do is time a compile: each bucket shape compiles on its
first step (20–45 s). With `train.precompile_buckets` (the journey sets it) every shape is
compiled before step 1 (`perf/includes_compile` 0) and the default is **20 steps**: 12
timed after leaving out the first and last rows and the 3 profiled steps with their
neighbours (xprof recommends a short capture of a few steady steps after warm-up: long
captures blow up the trace viewer's event limit and the profiling overhead). Without the
precompile, 40 steps were needed for the 4096 bucket (~14% of steps) to appear with
probability > 99.7%.

tjev-journey2 showed why the first row goes: every splash / Pallas run recompiled the 1024
bucket at step 1 despite the precompile (its LLO warnings repeat in `stdout.log`; the XLA
runs did not), a 23–33 s first row against 1.2–5 s steps. Counted, it made mega2 c128 +
merged LoRA look like 19.9k tok/s (2.04× XLA) when its steady step is 1.25 s: 47.8k real,
52.5k padded tok/s (4.88× XLA), the planner's 52,650 (tjev-bench4, padded: synthetic full
rows) within 0.3%. The planner's rates are padded tokens/s, i.e. step time; real
training's tokens/s is that × (1 − pad), about 0.9 here (14.5% padding over the 4B final).

The real tokens/s depends on the bucket mix (padding 3–40% per step with bucket packing,
under 3% with `train.packing=best_fit`): report padded tokens/s and the pad fraction
separately, or compare variants on the same seed (they see the same batches). The
learning-rate warmup and cooldown do not change the step time. The cost at 20 steps is in
[Cost](#cost).
