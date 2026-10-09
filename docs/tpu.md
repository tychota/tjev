# The TPU campaign

Training runs on Cloud TPU v6e (Trillium) or v5e, with the Pallas kernels
([kernels.md](kernels.md)) and bf16 / fp32 only. The campaign has four parts:

- **Sweep.** It settles what the local runs could not.
- **Transfer.** It checks the proxy's horizon and the other sizes.
- **Final.** It trains every size at its fitted recipe.
- **Post-training.** It post-trains and exports the best final run of each size.

Every step is resumable, so the campaign runs as a chain of time-limited Colab sessions.

> **Status.** The kernels, the planner and the runners are tested on CPU (interpret mode,
> TPU lowering, the queue on one CPU "chip") and have run on Colab v6e-1: the measured
> correctness, throughput and sweep results are in [results.md](results.md), the
> per-size defaults they fixed in `plan.HARDWARE["v6e1"]`. Phase `s0` still checks the
> kernels on the hardware before any training and falls back to the XLA paths if they fail.

> **Colab v6e-1 budget.** With ~80 compute units (~19.6 TPU-hours) the default campaign
> (~34 h) does not fit. [sweep-plan.md](sweep-plan.md) has the reduced, staged sweep (0.8B
> proxy, 300 steps, `--hardware v6e1 --lean`, ~9.1 h), the run-health kill rules, the W&B
> layout (`tjev-sweep`) and the final-training budget per size.

## Hardware

| | v6e (Trillium) | v5e (v5litepod) |
|---|---|---|
| Per chip | 32 GB HBM, 1.6 TB/s, 918 TFLOP/s bf16, 256×256 MXU | 16 GB HBM, 0.82 TB/s, 197 TFLOP/s bf16, 128×128 MXUs |
| Critical intensity | ~575 FLOP/B | ~240 FLOP/B |
| Preset | `tpu-v6e`: microbatch 8192 tokens per chip | `tpu-v5e`: 4096 per chip; 4B `mesh.fsdp=2`, 9B `fsdp=8` |
| Where | Colab v6e-1 (≈ 4.6 compute units / h) | Colab v5e-1 (free tier) |

- **Data parallelism.** Only LoRA gradients are all-reduced (<1% of a step), so the frozen
  base stays replicated while it fits.
- **The kernels under `shard_map`.** They run over the batch axis.
- **Chips per run.** 0.8B and 2B take one chip, 4B two, 9B four (v6e) or eight (v5e).

## Phases

`tjev campaign plan PHASE` writes a queue. Each phase is planned from the finished runs of
the phases before it (`tjev campaign fit`).

| Phase | Jobs | What it settles |
|---|---|---|
| `s0` | — | libtpu flags probed one by one; the Pallas kernel tests on the TPU (on failure every job falls back to the XLA paths); kernel and train-step bench per size; zero-shot controls |
| `sweep` | 21 | The 2B proxy at 600 steps: AdamW at 2.5, 3.54, 5 and 7.07e-5 (seed 0; 3.54e-5 also seed 1); Muon (Polar Express) at 2.5, 3.54 and 5e-5; paired arms at 3.54e-5 (below) |
| `transfer` | 12 | The proxy's horizon at its fitted rate and at 0.64× it (each a 2400-step run with WSD cooldown branches at 600 and 1200): the best horizon and the horizon exponent γ; 0.8B and 4B at the predicted rate × {½, 1, 2} |
| `final` | 2 per size | Every size of `--sizes` (4B, 2B, 0.8B, 9B) at the decided recipe (below): the long run to the horizon H (`final-<size>-h<H>`) and its 1-epoch WSD cooldown branch to H/2 (`final-<size>-h<H>-b<H/2>`); `tjev campaign select` picks the better of the two per size |

**The final recipe** (`FINAL_RECIPE` in `tjev.campaign.plan`, decided from the reg stage,
[results.md](results.md)): fit's recipe at the horizon (`--horizon H`, else the fitted
one) with rank 16 at its rank-rule rate (lr/√2 per rank doubling, as the reg stage's rank
arms: 4B / 9B from rank 64 at lr × 2, 2B / 0.8B from 32 at lr × √2), CogCalib
(`loss.cogcalib_weight=0.1 loss.cogcalib_quantile=0.3`), the EMA of the adapters tracked
(`optim.ema_decay=0.999`: raw and EMA scored at every full eval, the selection on the raw
adapters), the frozen gates, the hardware's perf defaults (`HARDWARE["v6e1"]`: best-fit
packing, the flat GDN norm, …), `post=full` (calibration, JevBench, and the extra
validation files of `PROBE_POST_ARGS`), a checkpoint and full eval every 200 steps (50 in
the cooldown, a quick eval every 25) all kept from H/2 on and through the cooldown
(`train.keep_tail`: the branch keeps its whole decay), health kill on the long run. The
branch starts at 0.8·H/2 (kept by `train.keep_steps`) with `train.health_kill=false`. The
cadence costs a 4B 5,000-step run 40 full evals (20 + 20) against 26 before, about 15–20
min at 1–1.5 min each (the quick evals' cost is measured by the first final run), and
keeps 28 checkpoints of ~13 MB of adapters plus ~13 MB of EMA adapters (8 from H/2, 20 in
the cooldown), and the optimizer state (222 MB) of the selected and the last step.

Both candidates of a size, the long run and its branch, are kept with their full `post/`
outputs: the selection does not pick between them automatically. On mix-v7.5 the held-out
selection score favoured the two-pass run (.370 against .390, lower better) while JevBench
public favoured the 1-epoch branch (75.3 against 70.2), inside JevBench's own 64–72 swing
between neighbouring checkpoints ([results.md](results.md#the-reg-stage-and-the-final-length-4b-runs-v6e-1-measured)).
The `--final-mix rep1` final (every source at most ~1 epoch, fresh rows in place of the
second epoch: "one larger epoch") is the pair that decides. `--final-mix rep1`
trains on the sibling mix `<mix>-rep1` (the held-out stays the campaign's; runs
`final-<size>-h<H>-rep1`, `…-b<H/2>-rep1`). `--final-recipe generic` plans fit's recipe
as is (its rank, no CogCalib or EMA, `post=quick`, branches at H/4 and H/2 named
`final-<size>-h<H/4>`, `-h<H/2>`): the kernel journey's line. Its branch of one horizon has
the name of the long run of half that horizon: plan one generic horizon per campaign.

```bash
tjev campaign plan final --runs runs --mix MIX --models MODELS --hardware v6e1 \
    --sizes 4B,2B,0.8B,9B --horizon 5000 [--final-mix rep1]
```

The sweep arms, all at the centre rate and paired with the AdamW run of the same seed:

| Arm | Change | Decides |
|---|---|---|
| `zclip`, `noclip` | z-score clipping, no clipping | the clipping |
| `b32k`, `b16k` | half / quarter batch at lr × 0.85 per halving, for 2× / 4× the steps | the batch (compared at matching tokens: the same segments) |
| `r16`, `r64`, `r128` (+ `-samelr`) | ranks 16 / 64 / 128 at the 1/√r rate (lr × √2, /√2, /2) and at the same lr | the rank of 0.8B / 2B (4B / 9B keep ≥ 64), and whether the rate moves with the rank |
| `b2-0.99` | Adam β2 0.99 | β2 |
| `fp32-storage` | fp32 weights and activations (one-pass bf16 matmuls) | nothing: reported only (a true fp32 control needs 6-pass matmuls, ~4–6× the cost) |

Budget knobs, the same for `plan`, `fit` and `cost` (`BUDGET="…"` for the runners):
`--seeds` (AdamW grid seeds), `--arm-seeds` (seeds of the arms and the Muon grid),
`--sweep-steps`, `--arms zclip,b32k,…`, `--no-muon`, `--proxy 0.8B|2B`, `--centre RATE`
(the grid rate the arms pair with), `--horizon STEPS` (the final horizon), `--final-recipe
decided|generic` and `--final-mix NAME` (the final phase, above). `plan` and
`cost` also take `--stage lr|arms|all` (the rate brackets, the paired arms) and `--lean`
(no quick eval, an eval every 100 steps, no post-training, health kills; see
[sweep-plan.md](sweep-plan.md)). Further arms: `gdn-bf16` (report only), the LoRA inits
`pissa`, `eva` and `lora-ga` (`lora.init`), the gate arms and the CogCalib / teacher losses
of [sweep-plan.md](sweep-plan.md). Runs stopped by their health rules (`health.json`
status `killed`) are left out of the fit.

How the fit decides:

- **Paired online loss.** Runs with the same seed see the same batches, since segment i is a
  function of (seed, i). The *paired online loss* is the training NLL window by window over
  the last 20% of steps. It compares two runs with a standard error of ~0.01 from one seed,
  against a seed standard deviation of ~0.026 for the eval score.
- **The rate curve.** A quadratic in log2(lr) with seed fixed effects, fitted on the online
  loss.
- **Arms and Muon.** An arm, or Muon (its best grid run against AdamW's), is adopted when
  its paired difference is below −max(0.007, 2 SE) and its eval score is not worse by more
  than 0.007.
- **Horizon exponent.** At 600 and 2400 steps, the paired difference of the two transfer
  rates places each horizon's optimum on the sweep's quadratic; the optimum's shift per
  log2 of steps is γ, shrunk toward the prior 0.15 ± 0.15.
- **Horizon and sizes.** The horizon is the best of the transfer runs. Sizes that were not
  measured follow the rank, width, batch and horizon rules ([training.md](training.md)).

The queue (`tjev campaign queue`):

- **Chip groups.** It runs several jobs per VM, each on its own chip group, through libtpu's
  per-process chip variables. A preflight verifies each group size; if it fails, jobs run
  one at a time on all chips.
- **Branches.** A branch starts as soon as its parent's checkpoint exists.
- **Failures.** A failed or stalled run (no metrics for 30 min) restarts from its last
  checkpoint.
- **After each run.** It runs `tjev post` (quick for sweeps).
- **Session limits.** At a session deadline every run checkpoints and the queue exits 3; the
  next session resumes it.

## Cost

`tjev campaign cost` prices each phase from the measured tokens/s of `tjev campaign bench`
(`HARDWARE[…]["tokens"]`, or MFU priors where nothing was benched). It counts training
tokens, compile time, evals and post-training. The default campaign, priced before any run
with the priors (the v6e1 Colab budget is in [sweep-plan.md](sweep-plan.md)):

| Hardware | Chip-hours | Wall time (8-chip VM) | Cost |
|---|---|---|---|
| v6e-8 (GCP list price, the cost model's rate) | 23.7 | ~3.0 h | ~$32 |

Without Muon and with only the clipping and batch arms (`--arms zclip,noclip,b32k
--no-muon`), the v6e cost drops to ~$26.

The MFU priors were 18–19% for 2B / 4B on v6e with the first kernels (their per-op
roofline is in [kernels.md](kernels.md#performance-model-and-roadmap)). On v6e-1 the bench
measured 52,650 / 36,250 / 14,120 / 9,220 padded tokens/s at 0.8B / 2B / 4B / 9B with the
planner's configuration (28% MFU at 9B; [results.md](results.md#tpu-throughput-v6e-1-measured)),
and those numbers price `--hardware v6e1`.

## Running it

The runner is driven from the machine that holds the data (Linux, macOS or WSL), with the
data under `TJEV_ROOT` (default `~/tjev-work`):

```
$TJEV_ROOT/data/<mix>/            tjev data build (mix.yaml, calibration.jsonl, …; MIX=mix-v7.5)
$TJEV_ROOT/data/jevbench/         tjev data jevbench (public.jsonl)
```

or, with `HF_DATASET=USER/REPO`, in a private HF dataset repo with the same layout. The host
downloads the models from Hugging Face and creates the held-out selection set itself
(`cloud/tpu_bootstrap.sh`). Outputs come back under `runs/` (metrics, evals, post reports,
the selected checkpoint of each run), plus `reports/` and `logs/`.

### Colab

```bash
TPU=v6e1 bash cloud/colab.sh all s0 sweep transfer final     # or TPU=v5e1 (free tier)
PROBE_ONLY=analyse bash cloud/colab.sh session 1 probe       # one session, one probe
```

Driven headless by `google-colab-cli`: each session (`SESSION_HOURS`, 11) restores the
previous one's output, runs until 0.4 h before its limit and saves only the checkpoints a
resume or a pending cooldown branch still needs (`cloud/session_run.py`). `all` chains up to
`MAX_SESSIONS` sessions. Settings: `MIX`, `MODELS`, `SIZES`, `BUDGET`, `CAMPAIGN`,
`WANDB_ENTITY`, `HF_DATASET`, and every `PROBE_*` variable of the shell, passed to the host.
The finished runs' checkpoints go to private HF model repos before the save drops them,
under `<CAMPAIGN>/<run>` (same-named runs of two campaigns overwrote each other under the
bare run name; older backups stay there): every checkpoint retention kept, each with its
adapters and its EMA adapters (`ema/`), the optimizer state of the last and of the selected
step only. The
final models (`final-*`, and the `smoke` probe's run of their line) always go to
`PROBE_HF_FINAL_REPO` (`tychot/tjev-final`, the repo of [model-card.md](model-card.md)),
unless `PROBE_SWEEP_BACKUP=0`; the sweep and reg runs go to `PROBE_HF_REPO`
(`tychot/tjev-4b`) with `PROBE_SWEEP_BACKUP=1`. The final runs also log to their own W&B
project, `tjev-final` (`TJEV_WANDB_FINAL_PROJECT`), the rest to `tjev-sweep`.
`PROBE_FETCH_CKPTS="RUN@STEP …"` fetches branch points back, and `PROBE_CALIB_RUNS` /
`PROBE_FINISH_RUN` read the backups; the repo follows the run's name, and RUN is the backup
path, `<CAMPAIGN>/<run>` (or a bare `<run>` for the older backups).

### On the host

`cloud/tpu_campaign.sh PHASES…` runs the phases on any TPU host; the Colab runner calls it.

- **Queues.** It plans each phase's queue once (`queues/PHASE.queue`).
- **Resume.** It skips finished phases and resumes paused ones.
- **Output.** It writes `logs/campaign.{done,failed,paused}`.
- **Final phase.** It post-trains and exports the selected run of each size with
  `tjev post` and `tjev export`.
- **Environment.** `cloud/tpu_env.sh` sets the JAX / libtpu environment: the compile
  cache, W&B, and the libtpu flags that libtpu accepted on this host.
- **Probes.** `bash cloud/tpu_campaign.sh probe` runs one probe under `reports/probe/`,
  chosen by `PROBE_ONLY` (default `validate,bench`); every variable has a default and the
  script header lists them:

  | Mode | What | Variables |
  |---|---|---|
  | `validate` | the TPU kernel tests, then the fp32 and bf16 ground truth against the official model with per-layer traces ([results.md](results.md#tpu-correctness-v6e-1-measured)) | `PROBE_TESTS` |
  | `bench` | tokens/s per variant and size with the two xprof captures ([profiling.md](profiling.md)) | `PROBE_SIZES`, `PROBE_VARIANTS`, `PROBE_DEBUG` |
  | `analyse` | every profile of the final train step per size | `PROBE_ANALYSE_SIZES` |
  | `smoke` | a final run's exact job, shortened: throughput, HBM, one checkpoint and eval | `PROBE_SMOKE_SIZE`, `PROBE_SMOKE_STEPS`, `PROBE_SMOKE_EXTRA` |
  | `resume` | a run and its SIGTERMed + resumed twin, step for step | `PROBE_RESUME_SIZE` |
  | `journey` | every GDN-core iteration on real training ([kernel-journey.md](kernel-journey.md)) | `PROBE_JOURNEY_SIZES` |
  | `calib` | per-item calibration dumps of kept checkpoints ([calibration.md](calibration.md)); an entry `RUN:STEP:ema` scores that step's EMA adapters (every kept checkpoint's `ema/` item; the runs before it need the step's optimizer state, backed up for the selected and the last step) | `PROBE_CALIB_RUNS`, `PROBE_CALIB_EXTRA_VALIDATION` |
  | `official` | the official JevBench harness against tjev's server ([official-eval.md](official-eval.md)) | `PROBE_OFFICIAL_TARGETS`, `PROBE_OFFICIAL_DI_SUITE` |
  | `finish` | a final run's full post-training and export from its HF backup | `PROBE_FINISH_RUN`, `PROBE_FINISH_SIZE` |

  `PROBE_HF_REPO` (default `tychot/tjev-4b`) is the private HF repo the probes read
  checkpoints from and write backups, dumps and large captures to.
- **libtpu flags.** They live in the TPU presets (`compute.libtpu_flags`: MaxText's v6e
  dense-model set), so a plain `tjev train tpu-v6e …` gets them. Phase `s0` probes them one
  by one, and an exported `LIBTPU_INIT_ARGS` (the accepted subset) takes precedence. They
  change speed, never results, so they are not part of a run's identity. Every training
  run logs the flags it runs with at its start: a `{"event": "libtpu"}` line in
  `stdout.log` with `libtpu_init_args`, `libtpu_source` (`env`: exported before the run,
  e.g. s0's probed subset; `preset`: the preset's, set by the run; `none`),
  `scoped_vmem_limit_kib` (`--xla_tpu_scoped_vmem_limit_kib`, null: libtpu's default) and
  `kernel_vmem_limit_bytes` (the GDN kernels' 90% of it), the same under `libtpu` in
  `config.json` and `system` in the W&B config, and the two limits in `summary.json`
  (`libtpu/*`, 0: the default).
