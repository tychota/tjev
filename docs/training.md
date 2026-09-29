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
| `lora` | `rank` (32; 64 for 4B / 9B), `alpha` (32), `rslora`, `targets` (every projection) |
| `compute` | `remat` (full / core / minimal / none), `attention` (xla / splash), `gdn_impl` (chunked / recurrent / pallas_tpu / pallas_tpu_split), `gdn_chunk`, `gdn_precision` |
| `mesh` | `data` (−1: every device), `fsdp` (shards the frozen base) |
| `optim` | `name` (adamw / muon), `lr`, `warmup_steps`, `decay_fraction`, `final_lr_fraction`, `decay_shape`, `clip_mode` (global / zscore / none), … |
| `train` | `steps`, `seed`, `tokens_per_step`, `seq_buckets`, `microbatch_tokens`, `packing_bins`, eval / checkpoint cadence, `branch_from` |
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
| Clipping | global norm 1.0 (z-score clipping is a sweep arm) | See *Clipping* below |
| Selection | NLL after one temperature (see below), evaluated at every checkpoint (every 50 steps) | ECE is not a proper score; the held-out generators catch overfitting to in-distribution templates |
| Precision | bf16 base, fp32 adapters and moments, fp32 GDN state and gates, `gdn_precision=high` (bf16_3x) | One-pass bf16 there broke the gate gradients (MaxText) |

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
only outliers, above mean + 2.5σ of an EMA of past norms. The sweep's paired arms decide
between global, z-score and no clipping.

### Muon

`optim.name=muon` runs Muon on the LoRA factors (`optax.contrib.muon`):

- **Orthogonalisation.** Polar Express Newton-Schulz polynomials, 8 steps. Each stacked
  [S, in, r] factor is orthogonalised per layer.
- **Update size.** `muon_rms=0.2` gives Muon AdamW's update RMS, so the learning rates
  transfer.
- **Everything else.** Non-matrix parameters use Adam.

It is an option, not the default. See [research/muon.md](research/muon.md).

## Selection

At every checkpoint step (`train.checkpoint_every`, 50 by default, and the last step) the
loop runs the full eval and scores:

- **`validation`**: the mix validation set, capped at 64 rows per source.
- **`heldout`**: the held-out generator selection set, if given.

It fits one temperature on the even validation slots, then scores NLL on the odd ones and
on the held-out set. `score` is the mean of those NLLs, and lower is better.

- Every evaluated step is a checkpoint, so the best is always restorable; it is never
  pruned. `selection.json` names it once its checkpoint is written.
- `tjev eval`, `tjev calibrate`, `tjev post` and `tjev export` use the selected step by
  default.
- JevBench is never used for selection.

A quick eval (`train.quick_eval_every`, every 10 steps, 24 items per source) is logged as `eval_quick/*`
for the curves only.

## Checkpoints, resume and branches

- **What is saved.** Orbax saves are async: adapters, optimizer state, and the data-stream
  state. The latest `keep_checkpoints` are kept, plus the selected step.
- **When.** Every `checkpoint_every` steps (with a full eval), and, for resuming only,
  every `checkpoint_secs` seconds (900 s by default, for preemptible or time-limited hosts)
  and on SIGTERM. SIGTERM saves, then exits
  with code 143.
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
| `learning/` | `loss`, `nll`, `brier`, `accuracy`, `grad_norm`, `lr`, `update_rejected` (a non-finite update is rejected on device) |
| `perf/` | `step_seconds`, `tokens_per_second_per_device`, `tflops_per_device`, `mfu`, `hbm_peak_gb`, `includes_compile` |
| `data/` | `pad_fraction`, `slots_per_step`, `segments_consumed`, `wait_seconds` (host time blocked on the stream: add `data.workers` above ~5% of a step) |
| `eval/`, `eval_heldout/`, `eval_quick/` | `all/{accuracy,nll,kl,brier,ece,confidence}` and per `type/`, `lang/` |
| `eval_calibrated/` | `temperature`, `val_nll`, `heldout_nll`, `score` |

`train.profile_start=N` records a `jax.profiler` trace of `profile_steps` steps into
`RUN/profile/`.

## After training

```bash
tjev post RUN --mix data/mix-v3 --jevbench data/jevbench/public.jsonl --zeroshot zeroshot-2B.json
```

1. `calibrate`: per-type temperatures on the mix calibration split, bound to the
   checkpoint (never fitted on JevBench).
2. JevBench public, raw and calibrated.
3. The held-out generator *reporting* set, whose seed differs from the selection set's.
4. The mix validation set (skipped by `--quick`).
5. `summary.md`: overall, by JevBench tier, the weakest families / types / languages, and
   Δ against the calibrated zero-shot control from
   `tjev eval-base MODEL DATA --calibrate-on CAL`.

Each step is skipped when its output exists. The individual commands are `tjev eval RUN
DATA [--calibration ART]`, `tjev calibrate RUN DATA` and `tjev eval-base`.

Metrics (`tjev.eval.metrics`):

- **What is reported.** Accuracy, NLL, KL (NLL minus the target's entropy), Brier and
  top-label ECE with 15 bins, overall and by type, language, family and source.
- **Soft targets.** They count as their *expected* correctness `target[argmax p]`: a model
  that outputs the target has zero ECE.
