# Training

## Configuration

A run config is a set of frozen dataclasses (`tjev.config.schema`) whose defaults are the
current recipe. It is built in layers:

```
dataclass defaults → presets / YAML files (in order, with `extends`) → key=value overrides
```

```bash
tjev config tpu-v6e qwen35-2b data/mix-v3/mix.yaml optim.lr=3e-5    # print the result
```

- **Presets are named by stem.** `configs/hardware/tpu-v6e.yaml` and `tpu-v5e.yaml` set
  the Pallas kernels and the per-chip microbatch. `configs/models/qwen35-{0.8b,2b,4b,9b}.yaml`
  set the rank and the learning-rate prior. A built mix's `mix.yaml` sets the data section.
- **Overrides are parsed as YAML** (`train.seq_buckets=[1024,2048]`, `log.wandb=true`).
- **Mappings merge key by key**, except `data.mixture`, which a layer replaces whole.
- **Loading is fail-closed.** Unknown keys, invalid literals and non-integral integers are
  errors.

The main sections:

| Section | What it sets |
|---|---|
| `model` | `path` (a HF snapshot), `dtype` (bfloat16) |
| `lora` | `rank` (32; 64 for 4B / 9B), `alpha` (32), `rslora`, `targets` (every projection), `init` (default / pissa / eva / lora-ga, with `init_niter`, `init_batches`, `init_gamma`), `dropout` |
| `compute` | `remat` (full / core / core_states / minimal / minimal_states / none), `remat_by_bucket` (a policy per bucket), `attention` (xla / splash) with `attention_splash_block`, `attention_fused_bwd`, `gdn_impl` (chunked / recurrent / pallas_tpu / pallas_tpu_v2 / pallas_tpu_mega / pallas_tpu_mega2), `gdn_chunk`, `gdn_inverse`, `gdn_precision`, `gdn_norm` (heads / flat), `lora_impl` (unmerged / merged), `head_rows` (full / letters) |
| `mesh` | `data` (−1: every device), `fsdp` (shards the frozen base) |
| `optim` | `name` (adamw / muon), `lr`, `warmup_steps`, `decay_fraction`, `final_lr_fraction`, `decay_shape`, `clip_mode` (global / zscore / none), `weight_decay`, `ema_decay`, … |
| `loss` | the regularisers ([below](#regularisers)): `label_smoothing`, `smooth_sources`, `kl_base_weight`, `kl_sources`, `hard_targets`, `cogcalib_*`, `teacher_mix`, `teacher_gate` |
| `train` | `steps`, `seed`, `tokens_per_step`, `seq_buckets`, `microbatch_tokens`, `packing_bins`, `packing` (bucket / best_fit, [data.md](data.md#packing-trainpacking)), eval / checkpoint cadence and retention (`keep_checkpoints`, `keep_steps`, `keep_every`, `keep_from`), `select_on`, `select_ema`, `branch_from`, `health_kill`, `precompile_buckets`, `profile_start` |
| `data` | `train`, `validation`, `heldout`, per-source caps, `mixture`, Grain `workers` |
| `log` | TensorBoard, W&B (project, entity, group, tags, offline mode) |

`tjev compile-check PRESETS… key=value…` compiles the train step ahead of time for every
bucket, from abstract shapes only (no weights, no data). It reports XLA's memory analysis
per bucket and warns above 90% of device memory. Run it before a long job.

## The recipe

```bash
tjev train tpu-v6e qwen35-2b data/mix-v3/mix.yaml model.path=models/Qwen3.5-2B \
    data.heldout=data/mix-v3/heldout-select.jsonl name=q2b output=runs
```

| Choice | Value | Why |
|---|---|---|
| Objective | soft-target CE + 0.1 · Brier over the displayed letters | Brier keeps probabilities honest; soft targets carry base rates and annotator spread |
| Optimizer | AdamW (β 0.9 / 0.999, eps 1e-8, no weight decay) | Qwen3.5 was pretrained with AdamW; Muon (Polar Express) tied it on 2B |
| Learning rate | 2B 3.6e-5, 0.8B 3.7e-5, 4B / 9B 2.5e-5 at 1200 × 65k tokens | Local mix-v3 paired runs put the 2B optimum at or below 4.5e-5; rank 64 takes 1/√2; horizon rule (see below) |
| Schedule | WSD: 50 warmup steps, stable, 20% cooldown with a 1 − √t shape to 2% | Cheap horizon exploration by cooldown branches; the √ shape is the better cooldown (Hägele et al. 2024) |
| Batch | 65,536 tokens per step, buckets 1024 / 2048 / 4096, 16 packing bins | ~150 answer slots per step on mix-v3 |
| Loss normalisation | divide by the expected slots per step | Steps hold one bucket; per-step normalisation made each long item weigh ~20× |
| Clipping | global norm 1.0 | The sweep's `zclip` / `noclip` arms were neutral (±0.0003), so the simplest mode stays; see *Clipping* below |
| Selection | NLL after one temperature (see below), evaluated at every checkpoint (every 50 steps) | ECE is not a proper score; the held-out generators catch overfitting to in-distribution templates |
| Precision | bf16 base, fp32 adapters and moments, fp32 GDN state and gates, `gdn_precision=high` (bf16_3x) | One-pass bf16 is 10–30% off on the gate gradients at every length (*Numerics at long T*, below); the sweep's `gdn-bf16` arm was neutral at 300 steps but is not adopted |
| GDN gate adapters | off in the final runs (`lora.targets` without `in_proj_a` / `in_proj_b`, `plan.FINAL_OVERRIDES`) | Frozen gates were the one stage-4 sweep win (−0.013 ± 0.006 online, [results.md](results.md)) |

On one v6e chip the planner's `--hardware v6e1` line (`HARDWARE["v6e1"]["extra"]`,
[tpu.md](tpu.md)) adds the measured speed defaults on top of this recipe: the mega2 GDN
kernel at chunk 128 with the blocked inverse, `core_states` remat, merged LoRA at ≤ 4B,
`train.packing=best_fit`, `compute.gdn_norm=flat`, the tuned splash tiles (block 1024,
fused backward) at ≤ 4B, and at 9B the letters-only head and `minimal_states` remat on
the 1024 / 2048 buckets ([Integration on v6e1](#integration-on-v6e1-what-was-adopted)).
These fields are part of a run's identity: a run started without them does not resume
under them.

### Long buckets (8192, 16384)

JevBench's hard long-policy items are 2–6k tokens, and Jev serves a single question up to
32k tokens. The long buckets train on 5–14.5k-token documents (`gen_longdoc_refund_policy`,
[data.md](data.md#long-items)):

```bash
tjev data build data/mix-long --max-prompt-tokens 16384   # keeps prompts ≤ 16384, adds the long generator
tjev train tpu-v6e qwen35-2b data/mix-long/mix.yaml model.path=… \
    train.seq_buckets=[1024,2048,4096,8192,16384] train.microbatch_tokens=16384
tjev campaign plan final … --hardware v6e1 --max-bucket 16384  # per size, below
```

- **A bucket longer than `microbatch_tokens` takes one row per device.** A microbatch of
  bucket T holds `max(microbatch_tokens, T)` tokens per device
  (`TrainSpec.microbatch_tokens_at`, `tjev.data.pack.rows_for`). The 9B plan already relied
  on this: microbatch 2048 with the 4096 bucket. So the largest bucket sets the activation
  memory, whatever `microbatch_tokens` is. A smaller `microbatch_tokens` only shrinks the
  short buckets' microbatches, which costs throughput and saves no peak memory.
- **Validation.** Buckets must be distinct, ascending and multiples of 128.
  `microbatch_tokens` must be a multiple of every bucket it holds, so rows fill it.
- **Steps.** Every step has `tokens_per_step / (microbatch_tokens × devices)` microbatches
  of one bucket. A long-bucket step therefore holds more tokens (and takes longer) but
  fewer answer slots. Gradients are normalised per expected answer slot, so its weight
  follows its slots. A 16384 row holds one 9–14k-token document: these rows are 55–90% full.
- **Eval and serving.** The eval sets pack into the run's buckets, so items over the
  largest bucket are counted in `dropped_too_long`. `tjev serve` (JAX) compiles
  1024…32768 at start; `--max-tokens` caps that. The MLX backend has no buckets.

**Memory on one v6e chip (32 GB).** `perf/hbm_peak_gb` (the process's `peak_bytes_in_use`)
does not count a step's temporaries on TPU. The 10.3 GB of the 4B smoke run are its
arguments only: the bf16 base, the adapters and Adam's state. The step's own footprint is
`compiled.memory_analysis()`: tjev-bench4 measured arguments + temporaries of 21.6 GB for
4B at 4096 tokens per microbatch.

`tjev.campaign.plan.step_hbm_gb` scales the measured temporaries linearly in tokens per
microbatch. That is an upper bound, since the gradient accumulators do not grow with T.
Under `remat=full` it also drops the `core_states` residuals. Those are fp32 states
[chunks × Dk × Dv] plus v_new: 1 KB per token and value head per GDN layer at chunk 128,
which is 288 KB per token for 0.8B / 2B and 768 KB for 4B / 9B.

| Size (bench point) | 4096 | 8192 | 16384 | 32768 | Proposed (v6e1) |
|---|---|---|---|---|---|
| 0.8B (9.1 GB at 8192, merged) | 5.4 | 9.1 | **16.4** | 31.0 (21.3 full) | buckets to 16384, microbatch 16384 |
| 2B (13.4 GB at 8192, merged) | 8.8 | 13.4 | **22.6** | 41.1 | buckets to 16384, microbatch 16384 |
| 4B (21.6 GB at 4096, merged) | 21.6 | 33.2 (26.7 full) | 56.4 (43.5 full) | — | buckets to 8192, microbatch 8192, `compute.remat=full` |
| 9B (28.2 GB at 2048) | 36.3 (33.1 full) | 52.7 | 85.3 | — | no long bucket on one chip |

All values in the table are estimated GB, except the bench points. The planner's
`HARDWARE["v6e1"]["long"]` encodes the proposals, and `--max-bucket` applies them, capped
per size.

**Not yet measured on a TPU.** The long buckets have only been sized by this estimate;
check these before a long-bucket run:

1. `tjev compile-check` (memory analysis per bucket) for each size at its proposed buckets.
   The 4B at 8192 under `remat=full` is the tightest: about 27 GB estimated, 0.84 of the
   chip.
2. The 9B's current 4096 bucket: the linear estimate is 36 GB, over the chip. Either XLA's
   rematerialisation pass fits it, or the 9B plan needs `seq_buckets=[1024,2048]`.
3. The step time of the long steps, and `remat=full`'s cost on 4B: the GDN forward runs
   three times per layer instead of once.
4. A per-bucket remat (core_states for the short buckets, full only for 8192):
   `compute.remat_by_bucket` (below).

### Remat per bucket (`compute.remat_by_bucket`)

`compute.remat` is the layer checkpoint policy of every bucket;
`compute.remat_by_bucket` overrides it per bucket, e.g.
`compute.remat_by_bucket='{1024: minimal_states, 2048: minimal_states}'` on top of
`compute.remat=core_states` (a bucket not named keeps `remat`). Each bucket's train step is
already its own jitted program (the row length is static), so the decoder reads the policy
from the row length at trace time (`SuperBlock.remat_at`); `tjev compile-check` reports
each bucket's policy and memory.

`minimal_states` saves `minimal`'s names (MLP gate / up, GDN qkv / z / b·a, mixer output,
attention qkv, LoRA x·A) and `core_states`' GDN kernel residuals (the per-chunk states,
(I+L)⁻¹ or v_new): the layer recompute is down to the norms, gates, conv and small
elementwise ops, and the GDN forward kernel runs once per microbatch
(`tests/model/test_remat.py`). It changes no math: on CPU (fp32, the tiny model, packed
multi-segment rows, mega2 + splash in interpret mode) a train step under it, or under
`core_states` with `remat_by_bucket`, gives bit-identical loss, gradient norm and updated
adapters to `core_states`. In bf16 the CPU's fusions round at different points: the loss
moves by ~0.5% on that tiny model, the size of bf16-vs-fp32 itself. On the TPU the
integration run measured it ([below](#integration-on-v6e1-what-was-adopted)): adopted at
9B on its 1024 / 2048 buckets, not at ≤ 4B.

**Memory per size and bucket** (`scripts/remat_memory.py`: `compiled.memory_analysis()`
of the full train step on CPU, abstract inputs: the v6e1 configuration, mega2 chunk 128 blocked, splash, rank 16
without the gate adapters, merged LoRA ≤ 4B and unmerged 9B, the v6e1 microbatch 8192 /
8192 / 4096 / 2048, 65,536 tokens per step). At the bench points the CPU analysis gives
the TPU's arguments exactly and its temporaries within −2 / +18% (4B / 9B: 11.43 / 8.41
GB on CPU against 11.60 / 8.17 measured; 0.8B / 2B 18% / 15% over), so the table scales
each size's temporaries by that ratio too; the table gives the larger of the two:

| GiB (args + temp) | 1024 | 2048 | 4096 |
|---|---|---|---|
| 0.8B: full / core_states / minimal / minimal_states | 7.0 / 9.5 / 13.3 / 14.9 | same | same |
| 2B | 10.5 / 13.4 / 19.4 / 21.2 | same | same |
| 4B | 14.3 / 18.2 / 23.7 / 26.3 | same | same |
| 9B | 22.1 / 23.9 / 27.2 / 28.6 | same | 25.3 / 29.0 / 35.4 / **38.1** |

Below 4096 tokens per row, every bucket of a size holds the same tokens per microbatch, so
its memory does not depend on the bucket; only 9B's 4096 bucket (one 4096 row against a
2048-token microbatch) is larger. Against ~29 GiB usable on a v6e chip:

- **0.8B, 2B, 4B:** `minimal_states` fits every bucket (14.9 / 21.2 / 26.3 GiB):
  `compute.remat=minimal_states` for the whole run, no per-bucket map needed. 4B leaves
  ~3 GiB.
- **9B:** `compute.remat=core_states compute.remat_by_bucket='{1024: minimal_states,
  2048: minimal_states}'`. Its 1024 / 2048 buckets are at 28.6 GiB, the tightest point:
  if the TPU compile-check puts them over 29 GiB, use `minimal` there (27.2). The 4096
  bucket stays `core_states` (29.0; `minimal_states` 38.1).
- Saved residuals per microbatch (the jaxpr's, all layers): 9B at 2048 tokens 2.95 GB
  under core_states, 8.28 under minimal_states (MLP gate / up 3.2, GDN qkv 1.6,
  mixer_out 1.1, states and (I+L)⁻¹ 1.6 each at 4096).

The speed side is the profile review's projection, not measured here: the remat share of
a step (21–23% at 9B) is cut to the recompute of the norms and elementwise ops, about
10–13% faster at 9B (≈0.7–0.9 s per 65k-token step) and ≈8 / 10 / 12% at 0.8B / 2B / 4B.

### Integration on v6e1: what was adopted

`train.packing=best_fit` and `compute.remat=minimal_states` are off in the schema's
defaults; the planner's v6e1 line turns on what the TPU measured as a gain. Measured on
one v6e chip (tjev-integ-4b / tjev-integ-9b, the v6e1 run: mega2 chunk 128 blocked, merged
LoRA ≤ 4B, `train.microbatch_tokens` 8192 / 8192 / 4096 / 2048, rank 16, mix-v6, seed 0,
200 steps with a profile at step 100; real tokens/s against the size's baseline arm):

| Lever | 4B | 9B | Adopted |
|---|---|---|---|
| `train.packing=best_fit` | +13.8% | +16% | every size |
| `compute.gdn_norm=flat` | +3.8% | (with letters) +5.2% | every size |
| `compute.head_rows=letters` | — | +5.2% with flat | 9B (untied head) |
| `compute.remat_by_bucket={"1024":"minimal_states","2048":"minimal_states"}` | — | +16.9% | 9B (4096 keeps `core_states`) |
| `compute.remat=minimal_states` | −0.6% | — | no |
| fused LoRA kernel (since removed) | −16% | −7% | no |
| `compute.gdn_precision=mixed` | not measured on its own | | no |

These fields are part of a run's identity: a run started without them does not resume
under them. The flat, remat and letters arms compute the baseline's function (loss within
bf16 noise at the same step); the packing arms train other batches, so they were compared
at equal real tokens.

**What the predictions were.** From the CPU memory analysis and the xprof review:
`minimal_states` at −8 / −10 / −12 / −10–13% step time at 0.8B / 2B / 4B / 9B (the remat
share, 21–23% at 9B, cut to the recompute of the norms and elementwise ops); `best_fit`
at `data/pad_fraction` 3.9 / 1.8 / 0.7% against 4.0 / 34.7 / 29.4% per bucket,
`data/slots_per_step` on the 2048 / 4096 buckets ~122 / 56 at 4B from 32 / 16, ×1.15 real
tokens per second over a run, and ×1.25–1.3 together at 4B / 9B. Packing came in as
predicted; the remat gain did not materialise at ≤ 4B (the saved recompute is small
against the larger live set) and did at 9B, where the GDN forward kernel ran once instead
of twice per microbatch.

**v6e memory per bucket** (`scripts/aot_arms.py`: libtpu's compile-only client on a CPU
host, the planner's v6e1 configuration, rank 64, without the gate adapters, 65,536 tokens
per step; GiB, temporaries / arguments / total, outputs aliased to the donated arguments):

| size, arm | 1024 | 2048 | 4096 |
|---|---|---|---|
| 4B baseline (merged, core_states) | 10.6 / 9.2 / 19.8 | 10.5 / 9.2 / 19.7 | 10.2 / 9.2 / 19.4 |
| 4B + flat | 10.5 / 9.2 / 19.7 | 10.5 / 9.2 / 19.7 | 10.6 / 9.2 / 19.8 |
| 4B + fused | 12.6 / 9.2 / 21.8 | 12.6 / 9.2 / 21.8 | 12.4 / 9.2 / 21.6 |
| 4B + remat (minimal_states) | 18.8 / 9.2 / 28.0 | 18.9 / 9.2 / 28.1 | 19.0 / 9.2 / 28.2 |
| 4B all-in candidate | 16.9 / 9.2 / 26.1 | 17.0 / 9.2 / 26.2 | 16.8 / 9.2 / 26.0 |
| 9B baseline (unmerged, core_states) | 7.3 / 18.5 / 25.8 | 7.3 / 18.5 / 25.7 | 11.8 / 18.5 / 30.2 |
| 9B + flat (+ letters) | 7.1 / 16.6 / 23.7 | 7.0 / 16.6 / 23.6 | 11.2 / 16.6 / 27.8 |
| 9B + fused | 9.7 / 18.5 / 28.1 | 9.5 / 18.5 / 27.9 | 14.8 / 18.5 / 33.3 (`4096: core`: 27.9) |
| 9B + remat (by bucket) | 12.1 / 18.5 / 30.5 | 12.0 / 18.5 / 30.4 | 11.8 / 18.5 / 30.2 |
| 9B all-in candidate | 12.3 / 16.6 / 28.9 | 12.0 / 16.6 / 28.6 | 12.6 / 16.6 / 29.2 (`4096: core`: 24.8) |

The 9B baseline's own 4096 bucket is at 30.2 GiB and trains, so 29 GiB is a margin, not
the limit. The fused LoRA kernel added ~2–2.4 GiB of temporaries at both sizes, and alone
at 9B did not fit its 4096 bucket under `core_states` (33.3), which with its speed loss
settled it: the kernel was removed ([kernels.md](kernels.md#fused-lora-removed)). Its
Pallas calls per program confirmed the remat composition: 50 forward kernels under
`core_states` (the forward and the layer recompute), 25 under `minimal_states` (XLA drops
the dead recompute), 16 backward kernels either way.

**The arms** (bench variants, `src/tjev/campaign/bench.py` `INTEGRATION`; the journey
probe puts each on the planner's final line of the size, [kernel-journey.md](kernel-journey.md)).
The fused and all-in arms went with the fused LoRA kernel and no longer exist:

| arm | 4B | 9B |
|---|---|---|
| baseline | `mega2-c128-merged` | `mega2-c128-blocked` |
| + flat | `c128-merged-flat` | `c128-flat-letters` |
| + fused (removed) | `mega2-c128-fused` | `c128-fused-4096core` |
| + packing | `c128-merged-packing` | `c128-packing` |
| + remat | `c128-merged-mstates` | `c128-mstates-bucket` |
| all-in (removed) | `allin` | `allin-9b` (fallback `allin-9b-4096core`) |

**Numerics at long T** (CPU, Pallas interpret mode, `tests/kernels/test_gdn_long.py`). These
runs check the MaxText report that one-pass bf16 gate gradients collapse at 8k tokens. The
heads are tiny (Hk 4, H 8, D 32) but T is real. ~T/1024 + 3 packed segments cut at random
offsets, the gates span the trained 0.8B's A_log / dt_bias, and `high` is emulated as
bf16_3x (CPU dots ignore `precision`). The table gives relative L2 error against a float64
token-by-token reference, with fp32 inputs:

| Path, precision | T | out | d(a) | d(A_log) | d(dt_bias) |
|---|---|---|---|---|---|
| fp32 XLA reference | 1k / 4k / 8k / 16k | 3.0e-7 / 3.1e-7 / 3.1e-7 / 3.1e-7 | 7.5e-7 / 7.6e-7 / 7.4e-7 / 7.6e-7 | 2.7e-6 / 7.2e-6 / 6.1e-6 / 3.9e-6 | 1.6e-6 / 3.9e-6 / 3.6e-6 / 2.5e-6 |
| mega2 (C128, blocked), high | 1k / 4k / 8k / 16k | 1.0e-5 / 1.1e-5 / 1.1e-5 / 1.1e-5 | 4.2e-5 / 4.0e-5 / 4.1e-5 / 4.0e-5 | 3.5e-4 / 5.1e-4 / 2.0e-4 / 1.6e-4 | 2.2e-4 / 2.4e-4 / 1.1e-4 / 1.0e-4 |
| v2 (C64), high | 1k / 4k / 8k / 16k | 8.8e-6 / 9.0e-6 / 9.1e-6 / 9.1e-6 | 2.7e-5 (all) | 1.3e-4 / 1.3e-4 / 8.9e-5 / 1.2e-4 | 8.5e-5 / 9.0e-5 / 6.8e-5 / 8.4e-5 |
| mega2, mixed | 1k / 4k / 8k / 16k | 1.4e-3 / 1.8e-3 / 1.9e-3 / 1.9e-3 | 5.6e-4 / 8.2e-4 / 8.0e-4 / 8.0e-4 | 2.0e-3 / 1.7e-3 / 1.6e-3 / 1.1e-3 | 2.1e-3 / 1.7e-3 / 1.8e-3 / 1.6e-3 |
| mega2, bf16 (one pass) | 1k / 4k / 8k / 16k | 5.0e-3 / 5.3e-3 / 5.3e-3 / 5.3e-3 | 1.7e-2 / 1.9e-2 / 1.8e-2 / 1.8e-2 | 9.6e-2 / 2.8e-1 / 1.2e-1 / 1.1e-1 (cos 0.995 / 0.967 / 0.994 / 0.994) | 7.9e-2 / 1.6e-1 / 1.0e-1 / 1.2e-1 |
| v2, bf16 (one pass) | 1k / 4k / 8k / 16k | 4.7e-3 / 4.8e-3 / 4.9e-3 / 4.8e-3 | 4.4e-3 (all) | 1.8e-2 / 1.5e-2 / 5.2e-3 / 5.1e-3 | 1.2e-2 / 1.2e-2 / 5.6e-3 / 6.5e-3 |

With bf16 projections (the model's dtype), `high` is flat in T: about 2e-3 for mega2 on
every term, and 6e-3–1e-2 for v2, which rounds v and its output to bf16.

- **At `high`, nothing degrades with length.** Every term is flat from 1k to 16k, and the
  gate-gradient cosine is ≥ 0.99999 with fp32 or bf16 inputs. The run's precision is safe
  for the long buckets.
- **One-pass bf16 is poor at every length, not only at 8k.** mega2's d(A_log) is 10–30%
  off (cosine down to 0.92 with bf16 inputs at 1k), and v2's is ~2%. Keep
  `gdn_precision=high`, the `gdn-bf16` arm stays report-only, and `mixed` stays ~1e-3.
- **What the CPU cannot settle.** These are Mosaic's own dot and exp accuracies at
  D = 128. `tests/kernels/test_gdn_mega.py -m tpu` covers T = 1100 on TPU; a T = 8192 run
  of `test_tpu_mega_high_no_worse_than_v2` is still to do.

The rationale, literature and measurements are in
[research/training_notes.md](research/training_notes.md) and
[research/muon.md](research/muon.md). The campaign planner (`tjev campaign fit`,
[tpu.md](tpu.md)) replaces the learning-rate priors with fitted ones.

### Learning-rate rules

These are applied by the planner where nothing was measured:

- **Rank.** rsLoRA scaling (α/√r) moves the optimum as 1/√r, so rank 64 takes lr/√2.
- **Width.** lr ∝ d^−0.05, the prior for Qwen LoRA across widths.
- **Batch.** lr × 0.85 per halving of the batch; √batch scaling is too steep.
- **Horizon.** lr ∝ steps^−0.15 from the 600-step sweep.

### Clipping

Raw LoRA gradient norms are 5–15, so `clip_norm=1.0` is active on *every* step and gives each
step unit norm. Combined with one-bucket steps, a 16-item 4096-token step then weighs about
15× per item, which undoes the loss normalisation. ZClip (`optim.clip_mode=zscore`) clips
only outliers, above mean + 2.5σ of an EMA of past norms. The sweep's paired arms (`zclip`,
`noclip`) were neutral at 300 steps (−0.0003 ± 0.006 and −0.0002 ± 0.004 online,
[results.md](results.md)): clipping does not matter at these rates, and global 1.0 stays.

### Muon

`optim.name=muon` runs Muon on the LoRA factors (`optax.contrib.muon`):

- **Orthogonalisation.** Polar Express Newton-Schulz polynomials, 8 steps. Each stacked
  [S, in, r] factor is orthogonalised per layer.
- **Update size.** `muon_rms=0.2` gives Muon AdamW's update RMS, so the learning rates
  transfer.
- **Everything else.** Non-matrix parameters use Adam.

It is an option, not the default. See [research/muon.md](research/muon.md).

### LoRA initialisations

`lora.init` chooses how the adapters start (`tjev.model.lora_init`). The sweep compared them
as paired arms (`pissa`, `eva`, `lora-ga`) at the centre rate and at fractions of it: none
beat the default at 300 steps (PiSSA at ½ of the rate ties it; LoRA-GA and EVA lose;
[results.md](results.md), stages 2–4). The default stays. In the NNX layout a projection
is y = xW + s·xAB, W [in, out], s = α/√r.

| `lora.init` | A, B at step 0 | Model | Output at init | Needs |
|---|---|---|---|---|
| `default` | A ~ N(0, 1/in), B = 0 | xW + s·xAB | the base | nothing |
| `pissa` (Meng et al. 2024, [arXiv 2404.02948](https://arxiv.org/abs/2404.02948)) | top-r SVD of W: A = U_r√S_r/√s, B = √S_r V_rᵀ/√s | xW + s·x(AB − A0B0) | exactly the base | the weights |
| `eva` (Paischer et al. 2024, [arXiv 2410.07170](https://arxiv.org/abs/2410.07170)) | A = top-r principal directions of the projection's inputs X (centred), B = 0 | xW + s·xAB | exactly the base | `init_batches` microbatches, forward |
| `lora-ga` (Wang et al. 2024, [arXiv 2407.05000](https://arxiv.org/abs/2407.05000)) | G = ∂L/∂W ≈ PΣRᵀ: A = c·P[:, :r], B = c·R[:, r:2r]ᵀ, c = d_out^¼/√γ | xW + s·x(AB − A0B0) | exactly the base | `init_batches` microbatches, forward + backward |

- **SVDs.** Randomized subspace iteration (PiSSA's "fast SVD"), `init_niter` power
  iterations (4), oversampled (2r columns; LoRA-GA 4r, as its code), fp32 at HIGHEST matmul
  precision on the accelerator. Stacked layers go one at a time (`lax.map`).
- **EVA.** Fixed rank: the paper's rank redistribution is off. The activations are never
  stored: each probe forward adds XᵀX·Q to an [in, 2r] sketch per projection, `init_niter` + 1
  passes. Centred (the sketch is of XᵀX − n·μμᵀ), as the authors' PEFT implementation
  (IncrementalPCA, no whitening); padding rows count (the GDN input is zeroed there). The
  init prints `eva_gain`: rms(x·A) over a random A's, √(mean_j a_jᵀXᵀXa_j / (tr XᵀX / in)),
  median, max and per projection kind.
- **LoRA-GA.** The full-weight gradient is never formed. With a temporary rank-k adapter,
  A = 0, B = Yᵀ keeps the base output and gives dA = s·G·Y; A = Q, B = 0 gives dB = s·QᵀG.
  The randomized SVD of G is then 2·`init_niter` + 2 ordinary LoRA gradient passes.
  `init_gamma` is the paper's stable-scale γ (16).
- **Offset, not residual kernel.** PiSSA and LoRA-GA keep the kernel W and frozen fp32
  copies A0, B0; the adapter is the rank-2r [A0 | A − A0]·[B − B0 ; B] of fp32 differences,
  so it is exactly zero at init and small while training (cost: a 2r- instead of r-wide
  adapter matmul pair, +r·(in+out) fp32 frozen values per projection). The papers' frozen
  residual kernel W − s·A0B0 in bf16 rounds at the scale of s·A0B0: LoRA-GA's s·A0B0 has
  singular values s·c² (27.7 at d_out = 6144, s = 5.66, γ = 16; the reference code's
  `scale: stable` with rsLoRA α = 16, r = 8 gives the same s), ~12× W's largest, ‖s·A0B0‖ ≈
  3·‖W‖, and each projection's bf16 output was ~1.4e-2 off (the base's own bf16 noise:
  1.7e-3; PiSSA 3.0e-3). Measured on a 256-wide model: the 0.8B sweep's LoRA-GA started
  above the base's loss (steps 1-20: 1.10 vs 1.02).
- **Data.** The init microbatches are the first ones of a fresh stream, the ones step 1
  trains on.
- **Persistence.** The init writes A0, B0 to `<run>/lora_init.safetensors`. Resume, branches,
  eval and export re-apply it, bitwise, and never recompute it (a file of before the offset
  form, no `form` metadata, rebuilds that run's residual kernels W − s·A0B0).
- **Export.** For offset inits (pissa, lora-ga) the merged snapshot is W + s·(AB − A0B0)
  (a legacy run's own W_res + s·AB). The PEFT adapter is rank 2r, [A | A0]·[B ; −B0] at the same scale, because
  PEFT adds it to the original W.
- **Rate.** Under Adam an update moves each factor entry by ~lr whatever the gradient's
  size, so the output moves in proportion to the factors it multiplies. Default: s·x·A·ΔB
  with ‖A's columns‖ = 1. EVA: the same, but x·A collects the top activation energy, ×gain
  (2.3 on a decaying spectrum, 4-5 with Qwen-like outlier channels, synthetic; the run
  prints its own `eva_gain`), and the loss's curvature in B grows with it: the 0.8B sweep's
  EVA at the centre rate rose from step ~30 (B's update ratio 0.046 against the default's
  0.002: B oscillates). LoRA-GA: both factors are c-scaled (c = 1.4-2.2 at d_out 1024-6144),
  s·(ΔA·B + A·ΔB) is 2.4-3.7× the default's per step, along the gradient by construction.
  PiSSA: √(S_j/s)-scaled factors, 0.7-1.6×. Rates relative to the default's: EVA ≈ 1/gain
  (~1/4), LoRA-GA ~1/3, PiSSA 1/2 to 1.

Cost (the defaults: `init_niter=4`, `init_batches=8`; 2B at rank 32, 9B at rank 64, every
projection adapted: 1.37B and 6.92B base parameters, 34M and 173M adapter parameters):

| | 2B | 9B |
|---|---|---|
| `pissa` | ~1.8 TFLOP in fp32 (seconds); one layer's fp32 copy live (≤ 50 MB) | ~18 TFLOP (well under a minute); ≤ 200 MB live |
| `eva` | 5 forwards of 8 microbatches (~2 train steps); probe state 245 MB | the same passes; probe state 1.3 GB |
| `lora-ga` | 10 forward + backward passes of 8 microbatches (~10 train steps at 8 microbatches per step), the LoRA matmuls at rank 4r; adapters, their gradient and its sum at rank 4r: ~1.6 GB | the same passes; ~8.3 GB per chip (replicated like the adapters, on top of the base and the activations: tight on 16 GB v5e chips; the sweep arms run on 2B) |
| `lora_init.safetensors` | 135 MB | 690 MB |

## Regularisers

All off by default: at their defaults the step is today's program and the identity of an
older run is unchanged (tested). Each is logged under `hp.*` in W&B.

| Option | What it does |
|---|---|
| `loss.label_smoothing` ε | trains on (1 − ε)·t + ε·uniform over the displayed letters, in the CE and Brier terms (a soft CE against the smoothed target) |
| `loss.smooth_sources` | restricts it to some sources: fnmatch patterns on the source name, or `block:<block>` (jev, generators, replay, analysis, writing; `tjev.data.mix.block_of_source`) |
| `loss.kl_base_weight` w | + w·KL(p_base ‖ p) over the letters, p_base the same model with the adapters off; logged as `learning/kl_base` |
| `loss.kl_sources` | the rows the KL applies to (same selectors, e.g. `[block:replay]`) |
| `loss.hard_targets` | trains on the argmax of a soft target: the control of the soft-target CE |
| `loss.cogcalib_weight` w, `cogcalib_threshold` t / `cogcalib_quantile` q, `cogcalib_ema` | CogCalib's gated confidence penalty: + w·𝟙[nll ≤ t]·(−H(p)) on each slot ([below](#calibration-losses)); logged as `learning/cogcalib_frac`, `cogcalib_term`, `cogcalib_threshold` |
| `loss.teacher_mix` λ, `loss.teacher_gate` g | gated teacher soft targets: (1 − λ)·t + λ·teacher on the rows whose teacher puts ≥ g on a non-gold letter ([below](#calibration-losses)); logged as `learning/teacher_frac` |
| `optim.ema_decay` d | an fp32 EMA of the adapters in the optimizer state, d_t = min(d, (1 + t)/(10 + t)); checkpointed with it, and every full eval scores it too (`eval_ema/*`, `eval_calibrated_ema/*`, `eval/step-N-ema.json`) |
| `train.select_ema` | selects on the EMA's score; `selection.json` then says `"weights": "ema"`, and `tjev eval` / `calibrate` / `post` / `export` load the EMA adapters (bound to their own calibration identity) |
| `lora.dropout` p | inverted dropout on the adapter's input, y = xW + s·drop(x)·AB, in training steps only (a key per step from the seed, one per microbatch and decoder layer; the remat recompute replays the masks). Merged LoRA cannot drop the adapter input alone: with p > 0 the model builds the (equal) unmerged layers, ~9% slower at 4B |
| `optim.weight_decay` | decoupled AdamW decay, on the LoRA factors only (a `gate_full` kernel is the base weight: decaying it would pull it to 0, not to the base). Under Muon it applies to the matrix parameters |

- **Logged metrics.** `learning/nll`, `brier` and `accuracy` stay against the data's target
  whatever the regularisers, so the planner's paired online loss compares the same quantity.
- **Soft targets.** The objective has always been the CE against the data's (soft) target
  plus λ·Brier against it. Soft rows: stsb, pavlick_formality (interpolated rater means),
  go_emotions, mlma_en / mlma_fr (annotator votes), gen_base_rate (exact posteriors), and
  part of plumb, openjev, certo, mghafiri_scenarios (teacher or exact distributions).
- **The KL's base.** The base is the same graph with every adapter switched off at trace
  time (the frozen kernel is the base's for every init), one forward per microbatch outside
  the gradient: ~1/3 of a train step (≤ 1/2 by FLOPs), ~+1.5 s on a 4.6 s 4B step on a
  v6e-1. Precomputing the base letters once per stream segment (segments are a function of
  (seed, index) and packed segments are isolated, so their logits do not depend on the
  packing) costs the same forward once per stream, and pays only when several KL runs share
  a seed or the KL is restricted to a few sources; it is not implemented.

### Calibration losses

Two options aimed at over-confidence on top of the soft CE + 0.1·Brier, both off by default
(weight 0: the loss, its gradient and the step are today's bit for bit, tested). The notation:
per slot, p the predicted distribution over the displayed letters, t the data's target,
nll = −Σ t·log p (the logged `learning/nll`), H(p) = −Σ p·log p.

**CogCalib** (Wang et al., arXiv 2505.20903, the CoECP variant). Over-confidence comes from
the rows the model already knows: their NLL drops fast and keeps pushing p → 1. The paper
gates a calibration term on the NLL (its Eq. 2–3, ℒ = ℒ_CE + 𝟙[ℒ_NLL ≤ t]·α·ℒ_cal) and,
with ECP as ℒ_cal (Pereyra et al. 2017), penalises low entropy on the gated rows only:

    loss_slot = nll + 0.1·brier + w · 𝟙[nll ≤ t] · (−H(p))

- The gate is the paper's hard indicator, evaluated on the current forward (no gradient
  flows through a step function; no other stop-gradient). The penalty's gradient raises
  H(p): on a known slot the gold logit is pushed down and the others up. Ungated slots
  get exactly today's gradient.
- **Threshold.** `loss.cogcalib_threshold` sets t in nats. The paper re-fits t each epoch on
  a calibration set (argmax over a grid of TPR + TNR at telling correct from wrong rows);
  here `loss.cogcalib_quantile=q` makes t the q-quantile of the slot NLLs instead, from an
  EMA (`loss.cogcalib_ema`, 0.9) of each step's quantile: every microbatch of a step gates
  on the threshold of the previous steps, then the step's own quantile (its slots' NLLs,
  collected by the accumulation scan: no extra pass) is folded in. It lives in the optimizer
  state (`CogCalibState`), so it is checkpointed, restored and rejected with it; the first
  step gates nothing. q = 0.3 gates ~30 % of the slots.
- **Values.** The paper's multiple-choice setting uses β = 0.1 (its ECP factor; 0.05–0.125
  robust, Table 18; 0.15 for open-ended QA); w here is α·β. The `rank16-cogcalib` arm
  uses w = 0.1, q = 0.3.
- Soft-target rows have nll ≥ H(t), so a low threshold mostly gates one-hot rows.

**Gated teacher targets** (Kev, github.com/jaredpalmer/kev: soft targets help only where
a teacher disagrees with the label at p ≥ 0.6; softening everything hurts). Rows carry an
optional teacher distribution (`Item.teacher`, the JSONL `teacher` field, carried in
displayed order into `Batch.teacher` [R,S,K] only when `loss.teacher_mix > 0`; every other
batch has no such leaf). With gold = argmax t:

    gate     = 𝟙[max over displayed letters k ≠ gold of teacher_k ≥ g]
    target'  = (1 − λ·gate)·t + λ·gate·teacher          (t one-hot: (1 − λ)·one-hot(gold) + λ·teacher)
    loss_slot = −Σ target'·log p + 0.1·Σ (p − target')²

target' is a distribution over the displayed letters (tested); rows without a teacher, or
whose teacher agrees or disagrees below g, keep t. It composes with `hard_targets` (applied
first) and label smoothing (applied after).

- **Sources.** plumb's `teacher_probs` (every row; the two-solver share, exact in the
  `probability` family) and decider's `teacher_p`, the teacher's probability of the gold
  answer: the rest goes to `teacher_pred` when it is another label (noul: exact), else
  uniformly to the other options (decider keeps only P(gold)). Mixes built before the field
  (mix-v6 and older) have no `teacher`: `scripts/teacher_annotate.py MIX/train --out …`
  recomputes it from the raw sources by row id and counts the gated rows.
- **mix-v6 (counted on Colab CPU, 2026-10-05).** 32,679 of its 33,730 plumb + decider_teacher
  train rows carry a teacher (plumb 4,920 / 4,920; decider 27,759 / 28,810), and **none passes
  the 0.6 gate** (16 noul ties at exactly 0.5 pass a 0.5 gate). The decider adapter already
  drops the disagreeing questions (`teacher_ok` false, or custom `teacher_p` < 0.5): 6,375 of
  them, 4,985 of which would pass the gate (1,345 noul, 3,640 choice). plumb's teachers
  never disagree with its key. On mix-v6 the `rank16-teacher` arm is therefore `rank16`
  (`learning/teacher_frac` = 0); it needs a mix that keeps those decider rows with their
  teacher.

## Selection

At every full-eval step (by default every checkpoint, `train.checkpoint_every`, 50 by
default, and the last step) the loop runs the full eval and scores:

- **`validation`**: the mix validation set, capped at 64 rows per source.
- **`heldout`**: the held-out generator selection set, if given.

It fits one temperature on the even validation slots, then scores NLL on the odd ones and
on the held-out set. `score` is the mean of those NLLs, and lower is better.

- **Per language.** `eval_calibrated/lang/<lang>` is the calibrated NLL of that language's
  odd validation and held-out slots (the global NLL is ~3/4 English); `lang_mean` and
  `lang_max` aggregate them. `train.select_on=lang_mean|lang_max` selects on one of them
  instead of `score`.
- Every evaluated step is a checkpoint, so the best is always restorable; it is never
  pruned. `selection.json` names it once its checkpoint is written.
- `tjev eval`, `tjev calibrate`, `tjev post` and `tjev export` use the selected step by
  default.
- JevBench is never used for selection.

A quick eval (`train.quick_eval_every`, every 10 steps, 24 items per source) is logged as `eval_quick/*`
for the curves only.

### Cadences

| Field | Default | When |
|---|---|---|
| `train.checkpoint_every` | 50 | a checkpoint (≤ 0: none, and no full evals: a benchmark) |
| `train.full_eval_every` | 0 | the full eval (0: at every checkpoint step) |
| `train.quick_eval_every` | 10 | the quick eval (not at full-eval steps) |
| `train.tail_fraction` | 0 | the tail: the last fraction of the steps (0: the cooldown, `optim.decay_fraction`) |
| `train.checkpoint_tail_every`, `full_eval_tail_every`, `quick_eval_tail_every` | 0 | the same, denser, in the tail (0: the main cadence) |

Every full-eval step is also a checkpoint, and the last step and a STOP step always get the
full eval. The planner's long final runs checkpoint every H/10 and every 100 steps in their
cooldown (`tjev.campaign.plan.tail_every`); its final-length reg runs (`--horizon` past
1500 steps) every 250 steps, every 100 in the cooldown, and get `post=full` (calibration,
JevBench and the per-item dumps) at the end ([sweep-plan.md](sweep-plan.md#stage-4-the-regularisers-the-reg-stage-4b-and-2b)).

## Checkpoints, resume and branches

- **What is saved.** Orbax saves are async: adapters, optimizer state (with the EMA), the
  data-stream state, and the selection so far.
- **Retention.** The latest `keep_checkpoints` planned steps are kept, plus the selected step,
  `train.keep_steps` (the planner sets the branch points of the cooldown branches it
  plans: a sequential queue starts them after the long run) and, with `train.keep_every=K
  train.keep_from=S`, every K-th step from S on (the final-length reg runs keep every 250
  steps from half the run, for JevBench on the late trajectory; K a multiple of
  `checkpoint_every`). A resume-only save (time-based, signal, deadline) is deleted once a
  newer save exists. The sweep backup (`PROBE_SWEEP_BACKUP`, [tpu.md](tpu.md)) uploads
  every kept checkpoint, the optimizer state of the last only; `tjev post` / JevBench can
  score a checkpoint without its optimizer state.
- **When.** Every checkpoint or full-eval step, and, for resuming only, every
  `checkpoint_secs` seconds (900 s by default, for preemptible or time-limited hosts).
- **Stopping at any step.** SIGTERM, SIGINT and `TJEV_DEADLINE` (unix seconds, checked every
  step) save the current step, wait for the save, and exit 143 (SIGINT 130). `<run>/STOP`,
  checked every step, gives the current step the full eval and a checkpoint, then ends the
  run as completed. Running the same command again (STOP removed) resumes exactly: the same
  loss at every step as an uninterrupted run (tested for all three). `tjev campaign queue
  --deadline-hours` sends SIGTERM at the session deadline.
- **Exact resume.** Running the same command again resumes exactly: same batches, same
  adapters on CPU. A changed training (the *identity*: config without logging and cadence
  fields, base snapshot hash, data file hashes, template version) is refused.
- **Cooldown branches.** `train.branch_from=RUN@STEP` starts from another run's checkpoint:
  adapters, optimizer and data stream. The source must be the same training with the same
  learning rates up to STEP. A WSD cooldown branch off a longer run's stable phase trains
  exactly what the shorter run trains alone (tested), so one long run prices several
  horizons.

## Metrics and logging

Every step's metrics are fetched one step late, while the next step runs, so logging every
step does not stall the device. The sinks are:

- `metrics.jsonl`, always.
- TensorBoard (`tb/`).
- W&B (`log.wandb=true`; `log.wandb_mode=offline` for hosts without a key or network).

A resumed run drops rows logged after its checkpoint in every sink (JSONL filter,
TensorBoard `purge_step`, W&B rewind), so resumed curves never overlap.

| Prefix | Keys |
|---|---|
| `learning/` | `loss`, `nll`, `brier`, `accuracy`, `kl_base` (with `loss.kl_base_weight`), `cogcalib_frac` (the fraction of slots gated), `cogcalib_term` (the penalty w·𝟙·(−H) per slot, ≤ 0) and `cogcalib_threshold` (t; 0 before the first quantile) with `loss.cogcalib_weight`, `teacher_frac` (the fraction of slots on the teacher mixture) with `loss.teacher_mix`, `grad_norm` (before clipping), `lr` (the rate the optimizer used), `update_rejected` (a non-finite update is rejected on device), `spikes` (loss > median + 5σ̂ of the last 50) |
| `health/` | on-device run health ([sweep-plan.md](sweep-plan.md) §6): `grad_norm_clipped`, `clip_scale`, `clip_fraction`, `zclip_*`; grad norms per group (`gdn`, `attention`, `mlp`) and factor (`a`, `b`); `update_norm`, `update_rms`, `param_norm`, `param_rms` and `update_ratio` per factor; `momentum_norm`, `momentum_cos`; `adam_sqrt_v_mean`, `adam_eps_fraction`; `muon_ortho_norm`; `status` |
| `perf/` | `step_seconds`, `tokens_per_second_per_device`, `tflops_per_device`, `mfu`, `hbm_peak_gb`, `includes_compile` |
| `data/` | `pad_fraction`, `slots_per_step`, `segments_consumed`, `wait_seconds` (host time blocked on the stream: add `data.workers` above ~5% of a step) |
| `eval/`, `eval_heldout/`, `eval_quick/` (and `eval_ema/`, `eval_heldout_ema/`) | `all/{accuracy,nll,kl,brier,ece,confidence}` and per `type/`, `lang/` |
| `eval_calibrated/` (and `eval_calibrated_ema/`) | `temperature`, `val_nll`, `heldout_nll`, `score`, `lang/<lang>`, `lang_mean`, `lang_max` |

`health.json` holds the run's health verdict (`ok`, `warn`, `kill`, or `killed` with
`train.health_kill=true`) and the rules that fired. `train.health_metrics=false` turns the
`health/` metrics off.

`train.profile_start=N` records a `jax.profiler` trace of `profile_steps` steps into
`RUN/profile/`.

## After training

```bash
tjev post RUN --mix data/mix-v3 --jevbench data/jevbench/public.jsonl --zeroshot zeroshot-2B.json
```

1. `calibrate`: per-type temperatures on the mix calibration split (`--grouping
   type_lang`: per type × language, shrunk to per type; [calibration.md](calibration.md)),
   bound to the checkpoint (never fitted on JevBench).
2. JevBench public, raw and calibrated.
3. The held-out generator *reporting* set, whose seed differs from the selection set's.
4. The mix validation set (skipped by `--quick`).
5. `predictions/`: per-item raw logits of the calibration, validation and held-out splits
   (`*.jsonl.gz`, for `scripts/calibration_study.py`); an older post dir gets them on the
   next `tjev post` without redoing anything else.
6. `summary.md`: overall, by JevBench tier, the weakest families / types / languages, and
   Δ against the calibrated zero-shot control from
   `tjev eval-base MODEL DATA --calibrate-on CAL` (optional: a missing report leaves the Δ
   columns out).

Each step is skipped when its output exists. The individual commands are `tjev eval RUN
DATA [--calibration ART]`, `tjev calibrate RUN DATA` and `tjev eval-base`. The official
harness's numbers come from `tjev eval-official RUN` ([official-eval.md](official-eval.md)).
Each takes `--orders K` (order averaging) and `--noul-order fixed`; the temperatures are then
fitted on, and bound to, that readout ([serve.md](serve.md#readout-options)).

Metrics (`tjev.eval.metrics`):

- **What is reported.** Accuracy, NLL, KL (NLL minus the target's entropy), Brier and
  top-label ECE with 15 bins, overall and by type, language, family and source.
- **Soft targets.** They count as their *expected* correctness `target[argmax p]`: a model
  that outputs the target has zero ECE.
