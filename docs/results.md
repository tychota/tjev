# Results

Everything measured so far, with where and how. The first measurements (model parity, the
optimizer sweeps on mix-v1 and mix-v3, the GPU profiles) come from the development machine,
a single RTX 5060 Ti (16 GB) running the XLA and the (since removed) Mosaic GPU kernels. The
TPU sections measure the kernels, the sweep and the finals on a Colab TPU v6e-1. Per-run
reports live next to the code: `reports/data/` (the data-quality report of every mix) and
`reports/calibration/` (the calibration study of the 4B finals).

## Model parity

Against HF transformers, on real checkpoints:

| Check | Result |
|---|---|
| 0.8B fp32, CPU, XLA paths (`scripts/check_parity.py`) | max \|Δlogit\| 2.6e-5, top-1 100%, top-10 100%, KL ≤ 4e-6 |
| 0.8B fp32, CPU, Pallas TPU kernels in interpret mode | max \|Δlogit\| 2.8e-5, top-1 100% |
| 0.8B / 2B / 4B bf16, GPU | top-1 100%, KL ≤ 1.2e-3 |
| Untied head (the 9B layout) | fp32 logit parity on a random untied fixture (`tests/model/test_parity.py`) |

Gradient parity of the GDN core on real 0.8B weights, 4096 real tokens: the worst cosine
against fp32 was 0.9988 with `gdn_precision=high`. MaxText's reported collapse of one-pass
bf16 gate gradients (cosine ≈ 0.3 at 8k tokens) did not reproduce at 4k.

At 8k and 16k (CPU, Pallas interpret mode, `high` emulated as bf16_3x, tiny heads, packed
segments; [training.md](training.md#long-buckets-8192-16384)), `high` stays flat in T on
mega2 and v2: d(a) is 3–4e-5 and d(A_log) / d(dt_bias) 1–5e-4 off the float64 truth, with
cosine ≥ 0.99999. One-pass bf16 is poor at every length rather than collapsing at 8k:
mega2's d(A_log) is 10–30% off. The on-TPU check at T = 8192 is still to do.

## Zero-shot controls

The untrained base models with tjev's prompt and readout, on the 231 public JevBench items.
Per-type temperatures are fitted on the mix calibration split.

| Qwen3.5 | Accuracy | ECE raw → calibrated | Easy | Original | Hard |
|---|---|---|---|---|---|
| 0.8B | 0.532 | 0.203 → 0.100 | 0.96 | 0.51 | 0.36 |
| 2B | 0.584 | 0.149 → 0.129 | 1.00 | 0.56 | 0.42 |
| 4B | 0.779 | 0.089 → 0.062 | 1.00 | 0.89 | 0.61 |

- **The weakest families** are tradeoff, multi-hop and temporal-numeric (4B: 0.27 on
  temporal-numeric).
- **French vs English** on the held-out generators: 2B 0.46 vs 0.51, 4B 0.56 vs 0.58.

## Training

These results are from mix-v1 (the first mix) and mix-v3, on GPU.

**2B optimizer sweep, mix-v1, 300 steps** (validation NLL + ECE; lower is better):

| Run | Validation | Validation ECE | JevBench accuracy |
|---|---|---|---|
| AdamW 3.75e-5 | 0.464 | | |
| AdamW 7.5e-5 | 0.425 | 0.041 | 0.701 (zero-shot 0.584) |
| AdamW 1.125e-4 | 0.443 | | |
| Muon 7.5e-5 | 0.419 | 0.024 | 0.667 |
| LoRA+ ×16 | 0.706 | | |

The seed spread (±0.02) is as large as most differences here. A 0.8B proxy sweep at 150
steps was monotone in the rate (0.532, 0.566, 0.674 at 0.5×, 1× and 2× of 1.5e-4).

**2B on mix-v3, 300 steps, paired online loss.** This is the training NLL over identical
batches, with an SE of ≈ 0.010 from one seed.

| Comparison | Stable phase | After cooldown |
|---|---|---|
| AdamW 4.5e-5 vs 6.4e-5 | −0.036 ± 0.010 | −0.038 |
| AdamW 9e-5 vs 6.4e-5 | +0.018 | |
| Muon-PE 3.5e-5 vs 5e-5 | −0.035 | |
| Muon-PE 7e-5 vs 5e-5 | | +0.040 |

- **Lower rates are better** for both optimizers, and the optimum is at or below the
  lowest rate tested. Eval scores looked flat only because of their noise (seed sd 0.026).
- **The priors follow.** This sets the priors (AdamW 4e-5 at 600 steps) and the TPU
  sweep's bracket.
- **Loss bumps come from clipping.** In the stable phase they grow with the rate. They
  line up with clipping being active on every step while steps hold a single bucket (see
  [training.md](training.md), *Clipping*).
- **Earlier runs are invalid.** Runs before the expected-slots loss normalisation weighed
  each long item ~20× and are not comparable.

## TPU sweep (v6e-1, 0.8B proxy, measured)

`tjev-sweep1` (W&B tychota/tjev-sweep), 300 steps × 65,536 tokens, `--lean`, the GDN
megakernel. The online loss is the training NLL over the last 20% of steps, paired on the
same batches (seed 0); negative is better.

**Stage 1, the rate.** AdamW: 2.5e-5 0.5875, **3.54e-5 0.5760**, 5e-5 0.6008, 7.07e-5
0.6279, a bracketed optimum at 3.24e-5. Muon (Polar Express): 2.5e-5 0.5768, as good as
AdamW's best; its optimum is at or below 2.5e-5 (unbracketed; the 3.54e-5 and 5e-5 runs were
stopped near their end by the pooled spike rule, since fixed: spikes are now judged per
sequence bucket).

**Stage 2, paired arms at AdamW 3.54e-5** (online Δ ± SE; one seed):

| Arm | Δ online | Reading |
|---|---|---|
| zclip / no clipping | −0.0003 ± 0.006 / −0.0002 ± 0.004 | clipping does not matter |
| batch 32k / 16k (same tokens) | +0.022 ± 0.011 / +0.054 ± 0.039 | smaller batches worse |
| rank 16 (lr·√2) / same rate | +0.004 ± 0.007 / −0.005 ± 0.009 | rank 16 as good |
| rank 64 (lr/√2) / same rate | −0.003 ± 0.005 / +0.019 ± 0.010 | neutral at the rsLoRA rate |
| rank 128 (lr/2) / same rate | +0.019 ± 0.010 / +0.069 ± 0.012 | worse; much worse unscaled |
| Adam β2 0.99 | +0.002 ± 0.006 | neutral |
| fp32 weights and activations | +0.008 ± 0.003 | bf16 storage costs nothing |
| merged LoRA (bf16 W + s·AB) | −0.005 ± 0.004 | no cost (and +20% speed at 0.8B) |
| GDN matmuls in one bf16 pass | +0.004 ± 0.005 | no cost at 300 steps |
| PiSSA | +0.032 ± 0.009 | worse at this rate (needs its own) |
| EVA, LoRA-GA | stopped at step 72 | diverging at this rate |

- **The rank rule holds.** The rate must move with the rank (the same-rate controls are
  worse by 0.02 at r64 and 0.07 at r128), and at the rsLoRA rate no rank beats 32 in 300 steps.
- **EVA and LoRA-GA did not train at the default rate** (health kill, `loss_above_start`):
  EVA learned for ~40 steps then rose (its A spans the activations' top directions, where
  the outliers are: a larger effective step); LoRA-GA started above the base model (1.10 vs
  1.02 for the first 20 steps, which an output-preserving init should not do).

**Stage 3, follow-ups** (online Δ ± SE against the centre run of the same seed):

| Arm | Δ online | Eval Δ | Reading |
|---|---|---|---|
| Muon 3.54e-5 / 2.5e-5 / 1.77e-5 | +0.019 / +0.001 / −0.007 | −0.002 / −0.030 / −0.003 | optimum ~2e-5, tied with AdamW's best |
| PiSSA ½ / ¼ of the rate | +0.004 / +0.001 | +0.014 / +0.037 | no gain (eval worse) |
| EVA ½ / ¼ (uncentred) | +0.124 / +0.019 | +0.067 / −0.012 | worse |
| merged LoRA, seed 0 / seed 1 | −0.005 / −0.009 | −0.005 / −0.008 | no cost on two seeds: adopted for 0.8B / 2B |
| one-pass bf16 GDN, seed 0 / 1 | +0.004 / −0.005 | −0.005 / −0.004 | neutral, but its gate gradient d(a) is 5% off (CPU interpret): not adopted as is |

- **The inits' bf16 residual.** PiSSA and LoRA-GA stored W − s·A0B0 as the bf16 kernel;
  LoRA-GA's s·A0B0 is ~3× ‖W‖, so the layer output error was 8× the base's own bf16 noise
  (the run started at 1.10 against 1.02). Fixed: the kernel stays W and the adapter computes
  s·(AB − A0B0) from fp32 differences (the model is the base bit for bit at init). EVA now
  centres its activations (as the authors' PEFT code). Stages 2–3 ran the old code.

**Stage 4, the GDN gates and the fixed inits** (seed 0, against the centre run):

| Arm | Δ online | Eval Δ | Reading |
|---|---|---|---|
| gates frozen (no adapter on in_proj_a / in_proj_b) | −0.013 ± 0.006 | −0.006 | better (2.2 SE) on seed 0; adopted in the planner's finals. Seed 1 has since run and the two-seed mean is about 1 SE, below the plan's adoption rule: the pair is reported with the reg stage |
| gate adapters at rank 8 | +0.008 ± 0.003 | −0.016 | mixed |
| gate kernels trained directly | +0.017 ± 0.007 | −0.008 | worse at the LoRA rate |
| PiSSA (fixed) at 1 / ½ of the rate | +0.037 / −0.003 | +0.021 / −0.013 | at ½ ties the default |
| LoRA-GA (fixed) at 0.3 | +0.140 ± 0.010 | +0.083 | much worse |
| EVA (fixed, centred) at ¼ | +0.026 ± 0.009 | −0.003 | worse |

No initialisation beats the default (A Gaussian, B = 0) at 300 steps.

## Throughput and memory (GPU, historical)

- **2B per-block profile** (8k tokens per step, full remat): GEMMs took 64% of the time at
  the tensor-core roofline. With custom kernels, the GDN delta rule was 3.9× faster than
  the chunked XLA form, which was launch-bound.
- **Remat policies, 2B at 8k tokens per step.**

  | Policy | Step | Peak memory |
  |---|---|---|
  | full | 2284 ms | 6.4 GB |
  | mlp | 2062 ms | 9.3 GB |
  | minimal | 1959 ms | 10.8 GB |

  With all three buckets, `minimal` spilled past 16 GB.
- **Multi-bin packing** (mix-v1): 1 bin 10.1% padding, 16 bins 5.8%, for about +5% answer
  slots per step at ~9 ms of host time.
- **Train-step temporaries per microbatch token** (full remat): 0.8B 0.55 MB, 2B 0.9 MB,
  4B 1.9 MB.
- **Host data work** was under 1% of a 2B step. Grain workers were not needed.

## TPU correctness (v6e-1, measured)

The ground truth is the official Qwen3.5 (HF transformers, fp32, on the host CPU):
`scripts/ground_truth.py` compares the loss, the logits and 115 per-layer gradients on long
real texts packed in one row (3 × 1365 tokens), with per-layer and per-op traces.

- **An XLA:TPU miscompile, fixed.** On jax 0.11.2 / libtpu 0.0.48, the GDN causal conv fused
  into the rest of the GDN core read corrupted rows near 512-row tiles. Its input was exact,
  its output 3% off, every layer after it 2–5% off, max |Δlogit| 11. This hit every path
  (XLA, Pallas v1/v2, splash, fp32 and bf16), while the conv compiled alone is exact. It fits
  the in-place aliasing hazard of openxla/xla#38597 and jax-ml/jax#35409. Optimization
  barriers around the conv (`gdn.py`) fix it; `TJEV_GDN_CONV_BARRIER=0` reproduces it.
- **Our code on TPU, fp32, against the official model.** Every layer ≤ 1e-6, loss 1.2e-5,
  max |Δlogit| 3.5e-5, top-1 100%, gradients worst 5.3e-4 (cosine 0.9999998). On CPU it is
  2.3e-5 / 1.3e-4 / 1e-3.
- **The bf16 paths on TPU, against the official fp32.** Max |Δlogit| 0.25–0.33, KL 2.9e-3,
  top-1 99.4–99.8%, gradient cosine 0.991–0.996. The v2 fused kernel is the closest of the
  bf16 paths. For scale, the official model in bf16 against itself in fp32 is 0.20 / 1.9e-3 /
  cosine 0.992 (tiny model).
- **A test artefact to avoid.** Parity with random LoRA `B ~ N(0, 0.01)` makes the adapters
  ~50% of each projection. The model becomes unstable: bf16 rounding grows ~1.2× per layer
  until correct paths decorrelate (62% in the loss). Parity now uses 2e-4 (~1%).

- **The GDN megakernels against the official model.** mega1 (the GDN core after the XLA
  conv) and mega2 (the conv inside the kernel) on TPU, fp32 (`highest`): loss 1.2e-5,
  max |Δlogit| 4.1e-5, top-1 100%, gradient cosine 0.9999998, as the XLA path. Their TPU
  unit tests (`tests/kernels/test_gdn_mega.py`, 18) check every output and gradient at fp32
  level, and at the training precision (`high`) no worse than the v2 path on the same
  inputs, fp32 and bf16 projections.
- **bf16, against the official fp32, per layer.** The relative L2 error of the hidden state
  after each of the 24 layers: mega1 equals v2 to 0.5% at every layer, mega2 is 3–8% lower
  (its conv stays fp32 instead of being rounded to bf16). The official model in bf16 is the
  noise floor:

  | bf16 path | Max \|Δlogit\| | KL max | Top-1 | Worst grad cosine | Last layer |
  |---|---|---|---|---|---|
  | official Qwen3.5 in bf16 | 0.43 | 5.0e-3 | 98.6% | 0.989 | |
  | v2 | 0.25 | 2.9e-3 | 99.6% | 0.9956 | 1.013% |
  | mega1 | 0.50 | 4.4e-3 | 99.6% | 0.9897 | 1.010% |
  | mega2 | 0.48 | 2.7e-3 | 100% | 0.9970 | 0.988% |
  | mega2, chunk 128, blocked inverse | 0.48 | 3.6e-3 | 99.6% | | 0.985% |

  The max |Δlogit| is one token's extreme over ~1,500 positions × the vocabulary: within
  the official model's own bf16 spread, and not a per-layer trend.

## TPU throughput (v6e-1, measured)

`tjev campaign bench`, 16,384 tokens per step, LoRA rank 32, `remat=core_states`
(tjev-bench3, W&B tychota/tjev-benchmarks):

| Variant | 0.8B tok/s | vs v2 | 9B tok/s | vs v2 | 9B MFU |
|---|---|---|---|---|---|
| XLA chunked (bench1) | 13,004 | 0.46× | 3,556 | 0.46× | |
| v2 fused forward + backward | 28,142 | 1 | 7,758 | 1 | 23.6% |
| v2, chunk 128, blocked inverse | 28,789 | +2.3% | 7,876 | +1.5% | 24.0% |
| merged LoRA (v2) | 31,513 | +12% | 7,737 | −0.3% | |
| mega1 | 34,192 | +21% | 8,455 | +9% | 25.7% |
| mega2 | 40,141 | +43% | 9,031 | +16% | 27.5% |
| **mega2, chunk 128, blocked inverse** | **41,791** | **+48%** | **9,254** | **+19%** | 28.2% |
| mega2 + merged LoRA | 50,103 | +78% | 8,698 | +12% | |
| mega2 blocked + merged LoRA + splash 1024 | 51,358 | +82% | 8,794 | +13% | |

- **The conv was the megakernel's target.** In the v2 step at 0.8B, the XLA conv and its
  barriers, the l2norm and the GDN glue were ~29% of device time; in mega2 they are gone.
- **What is left (0.8B, the fastest variant, 320 ms):** the GDN kernels 33%, the frozen base
  matmuls 28%, the saved residuals of `core_states` (the fp32 chunk states, ~0.8 GB per
  microbatch, written and read back) plus layout copies 12%, LoRA 10%.
- **Merged LoRA** pays at 0.8B (+20% on top of mega2) but not at 9B (+2.8 GB of HBM, slower).
  It rounds W + s·AB to bf16, so a small adapter is quantised at W's scale: the sweep
  measures its quality as a paired arm before any adoption.
- **The sweep runs on mega2, chunk 128, blocked inverse** (`--hardware v6e1`).

## 4B finals and their calibration (v6e-1, measured)

Two 4B finals (LoRA r64, WSD; W&B `tychota/tjev-sweep`), scored on the mix validation set and the 220 held-out generator items:
`final-4b-h3000` (selected at step 1200, run to 3000, about one epoch) and `final-4b-h1500`
(selected at step 1450). Their checkpoints were dumped per item (`PROBE_ONLY=calib`) and
studied by `scripts/calibration_study.py`; the per-checkpoint tables are in
`reports/calibration/<run>-<step>/study.md` and the reading in
[calibration.md](calibration.md). With the K = 2 readout and per-type temperatures:

| Checkpoint | Validation NLL | Validation ECE | Validation accuracy | Held-out generators NLL | Held-out ECE | Held-out accuracy |
|---|---|---|---|---|---|---|
| h3000 @ 1200 (selected) | 0.4465 | 0.0184 | 0.8076 | 0.3838 | 0.0545 | 0.8081 |
| h1500 @ 1450 (selected) | 0.4297 | 0.0140 | 0.8170 | 0.3892 | 0.0585 | 0.8024 |
| h3000 @ 3000 (last) | 0.4270 | 0.0076 | 0.8259 | 0.5159 | 0.1176 | 0.7981 |
| h3000 @ 1200 + h1500 @ 1450, mean of calibrated probabilities | 0.4328 | 0.0129 | 0.8135 | 0.3646 | 0.0449 | 0.8182 |

- **Step 3000 is over-fit out of distribution.** It has the best validation NLL and the worst
  held-out NLL (0.516 against 0.384 at step 1200): selection on the held-out generators was
  right not to pick it.
- **Per-type temperature stays.** `type_lang`, the feature-conditioned temperature,
  vector scaling, batch calibration, Platt and beta were all measured; none wins
  consistently at a well-selected checkpoint (the feature T gains 0.002–0.003 of held-out
  NLL). The per-source oracle (an upper bound) reaches 0.364 at step 1200.
- **Two runs ensembled is the one big lever**: held-out NLL 0.365 against 0.384 / 0.389
  alone, at twice the inference cost. The pair is complementary by language (h3000@1200
  better in English, h1500@1450 in French).

## The reg stage and the final-length 4B runs (v6e-1, measured)

Paired arms (same seed, same batches) against their size's centre, W&B `tychota/tjev-sweep`.
"Online" is the paired training loss over the last 20% of steps (10-step windows, mean ± SE,
negative better); "per-eval" the mean difference of the calibrated selection score over the
common full evals (negative better, wins out of evals); fit reports
`reports/fit-after-sweep-reg-200378.json` (campaign tjev-reg) unless stated. Seed noise of the
selection score: 4B centre s0 vs s1 best .4349 vs .4161, end .4682 vs .4503; 2B centre s0 vs
s1 best .4703 vs .4610, per-eval sd 0.025. One seed unless stated.

| Axis (size, mix, steps) | Arm vs control | Online Δ | Selection score | Decision |
|---|---|---|---|---|
| Rank (4B, mix-v5, 1500) | r32 at 3.39e-5 vs r64 at 2.4e-5 | +0.0014 ± 0.0024 | per-eval −0.004; end .4671 vs .4682 | — |
| | r16 at 4.8e-5 vs r64 | +0.0036 ± 0.0028 | per-eval −0.019 (11/15); end .4356 vs .4682 | r16 adopted |
| Rate (4B r16, mix-v6, 1500) | lr ×0.7 vs ×1 | −0.0066 ± 0.0032 | per-eval −0.0146 (11/15); best .3911 vs .3988 | open (JevBench below, 1 ckpt) |
| | lr ×1.4 vs ×1 | +0.0016 ± 0.0038 | end .4525 vs .4056 | rejected |
| Weight decay (4B) | wd 0.1, s0 / s1 | −0.0004 ± 0.0028 / +0.0020 ± 0.0033 | −0.034 (14/15) / −0.005 (8/15) | not adopted (seeds disagree) |
| Weight decay (2B) | wd 1 / 5 / 10 | +0.0122 ± 0.0040 / +0.0053 ± 0.0044 / +0.0049 ± 0.0027 | +0.004 / +0.028 / +0.007 | rejected |
| LR floor (4B) | final_lr_fraction 0.1 | −0.0003 ± 0.0007 | 0.0 | rejected |
| Label smoothing (2B) | 0.1 all / generators only | +0.0546 ± 0.0038 / +0.0178 ± 0.0033 | +0.0086 / +0.0178 | rejected |
| KL to base (2B) | weight 0.1 | +0.0344 ± 0.0035 | −0.0074 (+1/3 step cost) | rejected (neutral) |
| LoRA on GDN (2B) | nogdn | +0.0055 ± 0.0042 | per-eval +0.019 (3/15) | rejected |
| CogCalib (4B r16, mix-v6, 1500) | weight 0.1, q 0.3 vs off | −0.0113 ± 0.0026 | per-eval −0.0125 (11/15); best .3879 vs .3988 | adopted (0.1) |
| CogCalib (2B, mix-v7.5, 2500; tjev-cc2b) | weight 1.0 vs 0.1 | +0.0944 ± 0.0018 | per-eval +0.0335 (0/13); T 0.78 vs 1.19 | rejected |
| EMA weights (centres, decay 0.999) | EMA vs raw, 4B s0 / s1 | (same run) | per-eval −0.018 (13/15) / −0.016 (14/15) | open (not used since) |
| | EMA vs raw, 2B s0 / s1 | (same run) | per-eval −0.020 (13/15) / −0.021 (14/15) | |

EMA is the one lever consistent over two seeds and two sizes; no run since mix-v5b enabled it
and `train.select_ema` was never used.

**Data mixes** (4B r16 at 4.8e-5, 1500 steps, seed 0; one checkpoint each for JevBench; v1.5 public, 231 items,
per-type temperature and noul commit at margin 0.1, `tjev eval-v15`):

| Mix | Best score | Held-out NLL at best | End score | JevBench I / C |
|---|---|---|---|---|
| v5 | .4168 | .4099 | .4356 | not measured |
| v5b | .4036 | .4097 | .4091 | 66.9 / 84.2 |
| v6 | .3988 | .4076 | .4056 | 70.1 / 86.1 |
| v7 | .3983 | .4066 | .3983 | 64.2 / 87.2 |

Different mixes draw different batches: no paired loss. validation-v7 favours v7 (66.1 / 90.4
against 54.9 / 85.8 for v6) but is in-distribution for it.

**Final-length 4B on mix-v7.5** (r16, CogCalib 0.1, best-fit packing, 4e-5; tjev-v75b, tjev-jb-v75b):

| Checkpoint | Selection score | Temperature | JevBench I / C | validation-v7 I / C |
|---|---|---|---|---|
| 1-epoch branch, cooled 2000 → 2500 (core sources once) | .3902 | 1.21 | 75.3 / 85.8 | 65.3 / 89.3 |
| 5000 steps, selected at 4800 (core sources twice) | .3702 | 1.41 | 70.2 / 84.6 | 66.8 / 90.2 |
| 2750 / 3000 / 3500 / 4000, not cooled | — | — | 70.0 / 64.4 / 71.7 / 65.5 | 64.8 / 65.7 / 66.1 / 66.1 |

- **JevBench public is too small to rank checkpoints alone**: neighbouring checkpoints of one
  run swing 64–72, as wide as the 1-epoch / 2-epoch gap (75.3 / 70.2). The two metrics
  disagree (validation-v7, 12.8k rows, slightly favours two epochs). Open; the
  `-rep1` pair (fresh rows in place of the second epoch) decides.
- The second epoch fits the training distribution: lower selection score, but a calibration
  temperature of 1.41 against 1.21 (over-confident before calibration).
- For reference, the r64 finals on mix-v4 (same scoring): h3000 @ 1200 75.2 / 85.8,
  @ 3000 72.7 / 86.0, h1500 @ 1450 71.4 / 84.4.

## Still unmeasured

- The on-TPU gradient check of the GDN kernels at T = 8192 (the CPU interpret-mode check
  above is flat in T).
- A trained 9B run.
- The planner's `v6e1` line (mega2, merged LoRA, flat norm, best-fit packing, splash 1024
  with the fused backward) timed as one configuration: each gain was measured against its
  own baseline, and the +5–9% of the splash tiles on a setup with the fused LoRA kernel.
- The `-rep1` mix (running), CogCalib off against 0.1 on mix-v7.5, the teacher loss, a second
  seed of any final-length 4B arm.
