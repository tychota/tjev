# Results

Everything measured so far, with where and how. Most training measurements come from the
development machine, a single RTX 5060 Ti (16 GB) running the XLA and the (since removed)
Mosaic GPU kernels. They inform the TPU recipe, but TPU speed and quality remain to be
measured (phase `s0` of [the campaign](tpu.md)).

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
bf16 gate gradients (cosine ≈ 0.3 at 8k tokens) did not reproduce at 4k. Re-check at 8k
before training long buckets in bf16.

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

## TPU (predicted, unmeasured)

- **Current kernels.** The per-op roofline puts them at 9–28% model FLOP utilisation. On
  2B / v6e, about half of the step is the GDN core.
- **After the fused backward.** Its roadmap predicts ~40% for 2B and ~44% for 4B (see
  [kernels.md](kernels.md)).
- **Campaign cost.** With the current priors, the whole cheap campaign is about 17
  chip-hours on v6e (~$22 at flex-start rates) or ~5 h of a free Kaggle v5e-8 session.
