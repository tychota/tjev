# Training notes: the literature and measurements behind the recipe

The evidence behind the learning rate, schedule, batch, loss normalisation, checkpoint
selection, clipping and LoRA targets. The rules actually applied by the TPU campaign are in
the docstring of `src/tjev/campaign/plan.py`; the code is `src/tjev/train/optim.py`
(schedule, clipping, optimizers), `step.py` (loss normalisation) and `loop.py` (selection).
Muon is covered separately in [`muon.md`](muon.md).

Tags: **[V]** read in the primary source, **[S]** secondary summary, **[I]** our inference.
Local measurements were made on an RTX 5060 Ti during development (2B, 300 steps); they are
history and priors, not TPU results.

## 1. Learning rate

**Scale.** Every rule below is stated for the nominal `optim.lr`; the effective rate is
`lr × s`, with the rsLoRA scale `s = α/√r` (5.66 at r = 32, 4.0 at r = 64; α = 32).
- "LoRA Without Regret" (Thinking Machines 2025, thinkingmachines.ai/blog/lora): the optimal
  LoRA LR is ~10× full fine-tuning, ~15× for very short runs.
- O'Neill et al. (arXiv 2609.01244) [V]: Qwen3 0.6B–32B, LoRA r64, α 32, α/r scaling,
  all-linear, AdamW, cosine, 1 epoch. The LoRA optimum is flat at 1e-3 (effective 5e-4);
  α = 32 was best in every comparison; LoRA recovers ~98% of the full-FT gain.
- Converted to our scale that is ≈ 8.8e-5, consistent with the 7.5e-5 optimum measured on
  mix-v1. On mix-v3, with noisier steps (below), the optimum moved to ≤ 4.5e-5.

**Width.** lr*(d) ∝ d^−p with prior **p ~ N(0.05, 0.1²)** (`P_WIDTH = 0.05`).
- 2609.01244 fits p = 0.00, 95% CI [0, 0.03] for Qwen LoRA; the Tinker formula gives
  ≈ 0.08. The μA theory (arXiv 2602.06204) gives the n^−1/2 upper case.

**Rank (rsLoRA).** If the optimum is rank-invariant under α/r scaling, then under α/√r it
moves as 1/√r: **rank 64 (4B, 9B) takes lr/√2**. The rsLoRA paper claims rank-stable LRs;
the campaign tests r64 at lr/√2 and at lr. Rank itself matters little: in 2609.01244 r32 is
0.003–0.006 nats worse than r64, and r128 gains only 0.0004–0.0008 [V].

**Horizon.** lr* ∝ steps^−γ with prior **γ = 0.15 ± 0.15** (`GAMMA`).
- Bjorck et al. (arXiv 2409.19913) fit 0.32, but for *pretraining* (cosine, ≥ 760M), and
  the per-size exponent ranges 0.32–0.70.
- Fine-tuning evidence is near zero: 2609.01244 sees the LoRA optimum *rise* by +0.18 dex
  per decade of tokens (confounded with the dataset).
- Schaipp et al. (arXiv 2501.18965) [V], convex WSD theory: γ* ∝ 1/√T at a fixed cooldown
  fraction, with small measured gains (~0.01 loss). WSqD (arXiv 2607.10959) [V]: the WSD
  optimum drifts lower with the horizon.
- The campaign measures γ: the transfer phase trains the proxy for 2400 steps with
  cooldown branches at 600 and 1200.

**Local evidence (mix-v3, paired training loss against AdamW 6.4e-5; mean ± SE):**

| run | stable phase (70–240) | cooldown (250–300) |
|---|---|---|
| AdamW 4.5e-5 | −0.036 ± 0.010 | −0.038 ± 0.014 |
| AdamW 9e-5 | +0.018 ± 0.010 | +0.001 ± 0.011 |
| Muon-PE 3.5e-5 | −0.035 ± 0.012 | −0.023 ± 0.012 |

Lower is better for both optimizers, and the optimum is at or below the lowest rate tested.
The final eval scores of the same runs (0.564 / 0.591 / 0.574) looked flat only because of
their noise. Hence `PRIOR_LR = 4e-5` (AdamW, 2B, 600 steps) and a sweep grid of
2.5e-5 – 7.07e-5, √2 apart.

**Instabilities grow with the rate.** On identical batches the stable-phase loss bumps at
higher rates (AdamW 6.4e-5: 0.551 → 0.602 near step 210; 9e-5: 0.589 → 0.669 near 240)
while 4.5e-5 stays smooth. This is the WSD "river valley" picture (Wen et al. 2024): the
iterate bounces across the valley at a high constant rate and the decay descends into it.
Select after the cooldown.

## 2. Schedule

- **WSD**: linear warmup, constant peak, cooldown over the last 20% (`decay_fraction=0.2`)
  to 2% of peak (`final_lr_fraction=0.02`) with a **1 − √t** shape (`decay_shape=sqrt`).
- Hägele et al. (arXiv 2405.18392): WSD with a 20% cooldown matches cosine, gains plateau
  near 20%, and (1−√) beats a linear cooldown. Decaying to ~0 beats decaying to 10% on long
  runs (arXiv 2502.15938). Schaipp et al.: the cooldown fraction shifts the optimum (20% vs a
  full decay ≈ 2× in peak LR), so every branch keeps 20%.
- **Warmup is a fixed 50 steps**, independent of the horizon. A cooldown branch off a longer
  run's stable phase therefore trains exactly what a shorter run would, and one long run
  prices several horizons: `train.branch_from=<run dir>@<step>` resumes adapters, optimizer
  and data stream, and refuses a source whose identity or pre-branch rates differ.

## 3. Batch

- `train.tokens_per_step=65536`: ~150 examples per step on mix-v3. The published optima are
  8–64 sequences on 5k–100k examples, so we are probably above the optimum [I].
- 2602.09492 [V] (Llama-2 7B/13B, Qwen3-0.6B, Gemma3-1B; 3 seeds, 9 batch sizes, LR
  re-tuned per batch): the optimal LoRA batch is non-monotonic, stable across ranks 32–256
  and 7B→13B, and larger datasets tolerate larger batches. The claim that batch alone opens
  >10% gaps is this paper's only; 2601.22708 finds the LR the dominant sensitivity, and
  tuned vanilla LoRA matching most variants.
- LR vs batch: 2609.01244 finds +0.07 dex per doubling; Marek et al. (arXiv 2507.07101) [V]
  about 3× over a 1024× batch range. **Rule: lr × 0.85 per halving** (`BATCH_LR`), not √batch.
- The sweep's `b32k` arm tests `train.tokens_per_step=32768` at lr × 0.85 for twice the steps.

## 4. Loss normalisation: expected slots per step

**The bug.** Every step holds microbatches of one sequence bucket. On mix-v2 a 4096-token
step held 16 long items against ~330 in a 1024-token step. With a per-step mean, each long
item weighed ~20×; the gradient-norm median was 13 with spikes to 53–59 (5 on mix-v1).

**The fix.** The literature (the 2024 Hugging Face gradient-accumulation fix, Dr. GRPO)
normalises by a global count and prefers a fixed denominator when step contents vary.
`make_train_step` sums the per-slot losses over all microbatches and divides the gradient by
the **expected** number of slots per step, measured once per run from the first 64 steps of
a fresh stream (`expected_slots_per_step`; 151.6 on mix-v3). Every item weighs the same
whatever its bucket. After the fix the grad norm was ~8 and stable. Runs made with the old
per-step mean are not usable for LR choice. The metrics still report per-slot means.

## 5. Clipping

- Raw LoRA gradient norms are 5–15, so **`clip_norm=1.0` is active on every step**. Adam then
  sees g/‖g‖, which reweights steps by 1/‖g_t‖ (up to ~3×) and partly undoes the expected-
  slots normalisation [I].
- Measured mechanism (mix-v3): the norm scales as √(items per step) (norm/√slots 0.66 on long
  steps, 0.71 on short ones), so per-step gradients are noise-dominated. 42% of steps are
  long-bucket steps with 16–32 items, high loss (median 0.80) and small raw norm (median 3.5).
  Under always-on clipping a 16-item step moves the adapters as much as a 250-item step,
  ~15× the weight per item, which is a plausible cause of the loss bumps in §1.
- The largest raw norms (29–58) are rare (p99 ≈ 23) short, many-item steps: an outlier clip
  is the right tool for them.
- **`optim.clip_mode=zscore`** clips only above mean + 2.5 σ (`clip_z`) of an EMA
  (`clip_ema=0.97`) of past norms, after 25 warm-up steps (ZClip, arXiv 2504.02507 [V,
  abstract]; AdaGC, arXiv 2502.11034 [S]). It keeps the per-item weighting. The default is
  still `global`; the sweep's `zclip` and `noclip` arms decide, and the finals inherit the
  winner.
- Non-finite updates are rejected on device (parameters and moments unchanged).

## 6. Sweep method and selection

**Noise.** On mix-v1 the seed spread at a fixed config was ±0.02 in val NLL + ECE (Muon
7.5e-5: 0.419 vs 0.457), as large as the LR effects. Quick evals at 8 items per source
jitter ±0.02–0.04 per point; the TPU runs use 24 (`quick_validation_per_source`) and 64 for
the full eval (`validation_per_source`).

**Paired online loss.** Seed k fixes the LoRA init and the data order for every arm, so runs
on one seed see identical batches. Below one epoch every batch is new data, so the training
NLL is a held-out estimate; window-by-window differences between two runs cancel the batch
noise (SE ≈ 0.010 from one seed, against an eval-score seed sd of 0.026). `tjev campaign fit`:
- fits the LR curve as a quadratic in log₂ lr with seed fixed effects on the online loss
  (training NLL over the last 20% of steps, 10-step windows), within ±1.5 octaves of the best
  point (the curve is steep on the high side);
- adopts an arm when its paired online difference is below −max(0.007, 2 SE) and its eval
  score is not worse by more than 0.007 (the same-seed rerun floor);
- does no successive halving on the LR axis: truncated runs favour high rates.

**Selection by calibrated NLL.** NLL + ECE double-counts calibration, and binned ECE is not a
proper score (smECE, arXiv 2309.12236 [V]); calibration and refinement are optimal at
different points (arXiv 2501.19195 [V, abstract]). Temperatures are applied after training
anyway, so `loop.py` selects the checkpoint on **NLL after one temperature**, fitted on the
even validation slots and scored on the odd ones, averaged with the held-out generators
(`data.heldout`). ECE is reported, not selected on. Held-out generators are in the score
because locally a run kept improving in-distribution past 300 steps while held-out
generators and JevBench got worse. JevBench public is reported only.

## 7. Calibration objective

- Proper scoring over the answer letters only: soft-target cross-entropy + 0.1·Brier
  (`train.brier_weight`). This avoids the large-vocabulary label-smoothing problem
  (arXiv 2508.00264); label smoothing is redundant with soft targets.
- Soft targets: LLM soft-label distillation gave −43% ECE and −34% Brier (arXiv 2605.11954).
- Temperature scaling per question type × language after training, shrunk toward the global
  temperature for small groups [I].
- Optional: averaging the merged ΔW = s·AB over the cooldown (SWAG-LoRA and LoRA ensembles
  improve ECE); merging after a full cooldown adds little (WSM, arXiv 2507.17634).
- The `score` type is the weakest (NLL ~1.0, ECE ~0.12): a data issue (ordinal targets),
  not an optimizer one.

## 8. LoRA targets and other settings

- **Targets** (`LORA_TARGETS`): attention `q/k/v/o_proj`; Gated DeltaNet `in_proj_qkv`,
  `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`; MLP `gate/up/down_proj`. qkv and z are
  adapted together (vLLM is reported to fail on qkv without z [S]; one framework's "all"
  target once silently skipped the GDN layers [S]).
- **Frozen:** the causal conv1d, `A_log` and `dt_bias` (fp32).
- Rank 32 (0.8B, 2B) and 64 (4B, 9B), α = 32, rsLoRA; B = 0 at init. Adapters and moments
  are fp32 and cast to bf16 for the matmul.
- Dropout 0. Weight decay 0: decoupled decay `lr·λ` per step is a no-op at these rates.
- Variants: PiSSA, LoRA-GA and LoRA-One are within 1–2% once tuned (arXiv 2602.04998 [V,
  abstract]) and want lower LRs; DoRA gains are small; tuned vanilla LoRA ≈ most variants
  (2601.22708). LoRA+ ×16 lost clearly locally (0.706 vs 0.425). Muon (arXiv 2609.01244's
  flatter-minima result is full fine-tuning of Qwen3-8B, not LoRA) is covered in `muon.md`.
- **Epochs.** 2609.01244 sees overfitting past ~2 epochs. At 2400 steps on mix-v3 the
  generators reach ~2.6 epochs and some sources 3–3.6, so the final horizon comes from
  held-out selection over the cooldown branches, not from an assumption.
- **Precision.** The frozen base is bf16 on TPU (v6e's FP8 peak equals bf16). An MXFP8 base,
  measured on an RTX 5060 Ti during development, was neutral (+0.001 ± 0.010 after the
  cooldown against bf16).
