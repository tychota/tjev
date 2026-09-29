# The TPU campaign

Training runs on Cloud TPU v6e (Trillium) or v5e, with the Pallas kernels
([kernels.md](kernels.md)) and bf16 / fp32 only. The campaign has four parts:

- **Sweep.** It settles what the local runs could not.
- **Transfer.** It checks the proxy's horizon and the other sizes.
- **Final.** It trains every size at its fitted recipe.
- **Post-training.** It post-trains and exports the best final run of each size.

Every step is resumable, so the campaign can run as a chain of time-limited sessions on
Kaggle's free v5e-8.

> **Status.** The kernels, the planner and the runners are tested on CPU (interpret mode,
> TPU lowering, the queue on one CPU "chip"). Nothing has run on a real TPU yet, so phase
> `s0` checks the kernels on the hardware before any training.

## Hardware

| | v6e (Trillium) | v5e (v5litepod) |
|---|---|---|
| Per chip | 32 GB HBM, 1.6 TB/s, 918 TFLOP/s bf16, 256×256 MXU | 16 GB HBM, 0.82 TB/s, 197 TFLOP/s bf16, 128×128 MXUs |
| Critical intensity | ~575 FLOP/B | ~240 FLOP/B |
| Preset | `tpu-v6e`: microbatch 8192 tokens per chip | `tpu-v5e`: 4096 per chip; 4B `mesh.fsdp=2`, 9B `fsdp=8` |
| Where | GCP (flex-start ≈ $1.35 / chip-hour), Colab v6e-1 | Kaggle v5e-8 (free, ~20 h / week, 9 h sessions), Colab v5e-1 |

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
| `sweep` | 8 | The 2B proxy at 600 steps: AdamW at 2.5, 3.54, 5 and 7.07e-5 (seed 0; 3.54e-5 also seed 1); paired arms at 3.54e-5: z-score clipping, no clipping, half the batch at lr × 0.85 for twice the steps |
| `transfer` | 9 | The proxy's horizon: a 2400-step run with WSD cooldown branches at 600 and 1200; 0.8B and 4B at the predicted rate × {½, 1, 2} |
| `final` | 9 | 4B, 2B, 0.8B at their recipe: one long run, and cooldown branches at ¼ and ½ of its horizon; `tjev campaign select` picks the best of the three per size |

How the fit decides:

- **Paired online loss.** Runs with the same seed see the same batches, since segment i is a
  function of (seed, i). The *paired online loss* is the training NLL window by window over
  the last 20% of steps. It compares two runs with a standard error of ~0.01 from one seed,
  against a seed standard deviation of ~0.026 for the eval score.
- **The rate curve.** A quadratic in log2(lr) with seed fixed effects, fitted on the online
  loss.
- **Arms.** An arm is adopted when its paired difference is below −max(0.007, 2 SE) and its
  eval score is not worse by more than 0.007.
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
- **Mirror.** `--bucket gs://…` mirrors the runs every 10 minutes (spot VMs).
- **Session limits.** At a session deadline every run checkpoints and the queue exits 3; the
  next session resumes it.

## Cost

`tjev campaign cost` prices each phase from the measured tokens/s of `tjev campaign bench`
(or MFU priors before any bench). It counts training tokens, compile time, evals and
post-training. Before any run, with the priors:

| Hardware | Chip-hours | Wall time (8-chip VM) | Cost |
|---|---|---|---|
| v6e-8, flex-start | 16.6 | ~2.1 h | ~$22 |
| v5e-8, Kaggle | 39.4 | ~4.9 h (one session) | free |

The MFU priors are 18–19% for 2B / 4B on v6e with the current kernels. Their per-op
roofline and the fused-backward roadmap are in [kernels.md](kernels.md). A bench on the real
hardware replaces them.

## Running it

All runners are driven from the machine that holds the data (Linux, macOS or WSL), with the
data under `TJEV_ROOT` (default `~/tjev-work`):

```
$TJEV_ROOT/data/mix-v3/           tjev data build (mix.yaml, calibration.jsonl, …)
$TJEV_ROOT/data/jevbench/         tjev data jevbench (public.jsonl)
```

The host downloads the models from Hugging Face and creates the held-out selection set
itself (`cloud/tpu_bootstrap.sh`). Outputs come back under `runs/` (metrics, evals, post
reports, the selected checkpoint of each run), plus `reports/` and `logs/`.

### Kaggle (free v5e-8)

```bash
bash cloud/kaggle.sh all s0 sweep transfer final
```

The Kaggle CLI must be set up (`~/.kaggle/kaggle.json`), on a phone-verified account.

1. The code and data are pushed as private datasets.
2. The script chains headless script kernels, one per 9-hour session. Each session restores
   the previous one's output and runs until 0.4 h before its limit.
3. Each session saves only the checkpoints a resume or a pending cooldown branch still
   needs (`cloud/session_run.py`).

W&B logs offline unless `WANDB_INLINE=1`. A pushed kernel can silently get a CPU image
(kaggle-cli #1197); the session then stops at once, and the TPU has to be picked on the
kernel page.

### Colab

```bash
TPU=v6e1 bash cloud/colab.sh all s0 sweep transfer final     # or TPU=v5e1 (free tier)
```

Driven headless by `google-colab-cli`, with the same session chaining (sessions end at
~12 h).

### GCP

```bash
bash cloud/tpu.sh all s0 sweep transfer final    # up push setup run wait pull, then always down
```

Settings are environment variables (`TYPE=v6e-8`, `ZONE`, `MODE=flex|spot|on-demand`,
`MAX_HOURS`, `BUCKET`, …; see the script header).

- **Queued resources.** VMs come from queued resources. Flex-start is never preempted and
  is deleted by GCP at `MAX_HOURS`.
- **Cleanup.** `all` deletes the VM whatever happens.
- **Manual steps.** `up`, `push`, `setup`, `run`, `status`, `wait`, `pull` and `down`
  exist separately.
- **W&B key.** It is read from `WANDB_API_KEY` or `~/.netrc` and uploaded as a file, never
  put on a command line.

### On the host

`cloud/tpu_campaign.sh PHASES…` runs the phases on any TPU host. It is also the entry point
of the three runners.

- **Queues.** It plans each phase's queue once (`queues/PHASE.queue`).
- **Resume.** It skips finished phases and resumes paused ones.
- **Output.** It writes `logs/campaign.{done,failed,paused}`.
- **Final phase.** It post-trains and exports the selected run of each size with
  `tjev post` and `tjev export`.
- **Environment.** `cloud/tpu_env.sh` sets the JAX / libtpu environment: the compile
  cache, W&B, and the libtpu flags that libtpu accepted on this host.
- **libtpu flags.** They live in the TPU presets (`compute.libtpu_flags`: MaxText's v6e
  dense-model set), so a plain `tjev train tpu-v6e …` gets them. Phase `s0` probes them one
  by one, and an exported `LIBTPU_INIT_ARGS` (the accepted subset) takes precedence. They
  change speed, never results, so they are not part of a run's identity.
