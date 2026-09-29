# Muon on the LoRA factors

Scope: per-factor Muon (`optax.contrib.muon`) on the stacked LoRA factors of Qwen3.5 2B
(r = 32) and 4B (r = 64), rsLoRA α = 32, frozen bf16 base. The survey of gauge-invariant
"product" optimizers (LoRA-Muon, sMuon, PoLoRA, Riemannion) is in
[`lora_optimizers.md`](lora_optimizers.md); this note only asks which settings per-factor
Muon should use, and whether to use it at all.

Code: `src/tjev/train/optim.py` (`_muon`, `POLAR_EXPRESS_8`). Config: `optim.name=muon`,
`optim.muon_beta`, `optim.muon_rms`. Tags: **[src]** read in a primary source,
**[measured]** CPU experiment or training run, **[inference]** reasoning, not measured.

## Summary

- **AdamW is the default. Muon is an option** (`optim.name=muon`). Qwen3.5 was pretrained
  with AdamW, and every measurement so far puts Muon within seed noise of AdamW.
- The Newton–Schulz (NS) polynomial is **Polar Express with 8 steps** (PE-8), the only
  variant implemented. Keller Jordan's fixed quintic under-delivers the step that
  `consistent_rms` assumes by 2–3× at our ranks, so its effective LR depends on rank and
  gradient spectrum.
- `consistent_rms=0.2` matches AdamW's update RMS in our noise-dominated regime, so AdamW
  learning rates and schedules carry over.
- **sMuon** (arXiv 2608.14492) is implemented only if Muon-PE ties or beats AdamW on 2
  seeds.

## What `optax.contrib.muon` does on stacked factors [src]

Muon branch: `scale_by_muon → scale_by_shape → add_decayed_weights(wd) →
scale_by_learning_rate`, so the update is `−lr_t · (c(shape)·O + wd·p)`.

- **Momentum.** `mu = β·mu + (1−β)·g`, bias-corrected. With Nesterov (the default) the
  orthogonalised vector is `β·m̂ + (1−β)·ĝ`. Momentum is stored in the param dtype (fp32).
- **Stacked layers.** `muon_weight_dimension_numbers` sets `reduction_axis=ndim−2`,
  `output_axis=ndim−1`. Every other axis is a batch axis and is `vmap`ped, so each layer's
  `[S, in, r]` / `[S, r, out]` factor is orthogonalised on its own.
- **Gram side.** Tall inputs are transposed, so NS always works on the small r×r Gram
  matrix, for A and for B. Frobenius preconditioning (the default) divides by `‖X‖_F`.
- **Coefficients.** `ns_coeffs` is one `(a, b, c)` triple or an `(n, 3)` table run with
  `lax.scan`. Gotcha: `init` raises `Not enough coeffs` when the table is *longer* than
  `ns_steps` (the message is backwards); `optim.py` sets `ns_steps=len(table)`.
- **Scale.** With `consistent_rms=ρ`, `c = ρ·√max(fan_in, fan_out)`. For an exact
  semi-orthogonal m×n factor RMS(O) = 1/√max(m, n), so the update RMS is exactly ρ·lr for
  every shape, including the 32×16 GDN gate B (r > out). This holds only if the singular
  values really are 1, which depends on the NS polynomial (next section).
- **Zero B.** At step 1, B = 0 gives `g_A = 0`; `0/(0+eps)` is a zero update, not a NaN
  [measured].
- **Weight decay** is `lr_t·wd` per step (PyTorch convention). At lr ~5e-5 with λ = 0.1
  the total shrink over 1000 steps is ≈ 0.5%: a no-op. `weight_decay` stays 0.
- **`adaptive=True`** rescales the update by `⟨m̂, O⟩` (≈ nuclear norm), which re-couples
  the step to the gradient scale and to clipping. Off.

**Why ρ = 0.2.** Moonlight (Liu et al., arXiv 2502.16982) scales the Muon update to
`0.2·O·√max(A, B)` because AdamW's update RMS is 0.2–0.4; 0.2 and 0.4 tied in their
ablation, and Muon then reuses AdamW's LR and weight decay [src]. Simulated Adam
(β₁ = 0.9, β₂ = 0.999) on gradients with per-coordinate SNR s reaches update RMS 0.228
(s = 0), 0.248 (0.1), 0.349 (0.3), 0.616 (1.0) [measured]; the noise-only limit is
√((1−β₁)/(1+β₁)) ≈ 0.229. Our gradients (a few answer-letter positions per sequence) are
noise-dominated, so 0.2 is the right match, and the AdamW and Muon LR optima coincide in
practice.

## Why Polar Express, 8 steps

Candidates:
- **Keller's quintic** (3.4445, −4.7750, 2.0315) × 5 (modded-nanogpt). Tuned for slope at
  0, deliberately non-convergent: singular values land in roughly [0.5, 1.5].
- **Polar Express** (Amsel, Persson, Musco, Gower, arXiv 2505.16932): a minimax-optimal
  degree-5 polynomial per step, converging to (1.875, −1.25, 0.375). The table in
  `optim.py` is the paper's 8-step list with the reference code's safety factor
  (a/1.01, b/1.01³, c/1.01⁵ on all but the last step).

Measured error (CPU, fp32, `highest` precision, optax's own
`orthogonalize_via_newton_schulz`). Cells are singular-value range / update RMS relative to
the exact polar factor, i.e. an **effective LR multiplier**. "Realistic" is an EMA
(β = 0.95, 50 steps) of `Aᵀ G` with G = rank-8 decaying signal × snr + fresh noise.

| input | Keller-5 | PE-5 | PE-6 | PE-8 |
|---|---|---|---|---|
| 2048×32 Gaussian | 0.75–1.04 / 0.98 | 0.86–1.14 / 1.00 | – | 1.000 / 1.000 |
| 2560×64 Gaussian | 0.68–0.86 / **0.73** | 0.86–1.14 / 1.02 | – | 1.000 / 1.000 |
| 32×6144, σᵢ ∝ i⁻² (cond 1024) | 0.44–1.20 / 0.87 | 0.82–1.14 / 1.01 | – | 1.000 / 1.000 |
| 64×9216, σᵢ ∝ i⁻² (cond 4096) | 0.11–1.20 / **0.63** | 0.23–1.14 / 0.79 | 0.42–1.00 / 0.86 | 0.94–1.00 / 0.99 |
| 32×6144 realistic, snr 5 (cond 1500) | 0.23–1.13 / **0.52** | 0.46–1.14 / 0.71 | 0.74–1.00 / 0.89 | 1.000 / 1.000 |
| 64×6144 realistic, snr 5 (cond 2200) | 0.16–1.13 / **0.38** | 0.33–1.13 / 0.54 | 0.57–1.00 / 0.75 | 0.99–1.00 / 1.00 |

- With a few dominant momentum directions (the usual case), Keller-5 delivers **0.52× the
  target step at r = 32 and 0.38× at r = 64**. Even on a flat spectrum it gives 0.73× at
  r = 64 against 0.98× at r = 32, because Frobenius normalisation starts a flat r-dim
  spectrum at 1/√r. The nominal LR therefore means different things on 2B and 4B, which
  breaks rank transfer and the Muon-vs-AdamW comparison.
- PE-8 is within 1e-3 of the exact polar factor up to cond ≈ 2000 and ≤ 6e-2 at cond 4096.
  Running Keller's quintic for 8 steps does not converge (it plateaus at [0.68, 1.13]).
- Keller-5 acts as a soft spectral filter on weak (mostly noise) directions. PE-8 removes
  it, so PE-8 is better *defined*, not necessarily better.
- bf16 momentum degrades PE-8 to σ ∈ [0.994, 1.006]; momentum stays fp32.

Rejected alternatives:
- **Exact polar via `eigh` of the r×r Gram matrix** squares the condition number and fails
  in fp32 above cond ~10³ (singular values in [0, 1.99]); batched SVD is exact but slow
  [measured].
- **Gram Newton–Schulz** (Dao-AILab) evaluates the *same* polynomial with fewer FLOPs, so
  it cannot fix accuracy; NS is already ~0.005% of step FLOPs here (8 × 4r × P_lora ≈
  34 GFLOP per 2B step at r = 32).
- `aol` preconditioning (arXiv 2512.04632) was worse than Frobenius in 10 of 15 cases.

## What per-factor Muon does to the adapters [inference]

Notation: `W_eff = W₀ + s·A B`, s = α/√r (5.66 at r = 32, 4.0 at r = 64); A is in×r,
N(0, 1/in), so `AᵀA ≈ I`; B starts at 0.

- **Early phase.** With `AᵀA ≈ I`, `A·polar(AᵀM) = polar(AAᵀM)`: the first B steps are
  exactly spectral steepest descent on W restricted to col(A), the gauge-correct step that
  LoRA-Muon/sMuon aim for.
- **Rank independence.** ‖ΔW‖_F per step is `lr·s·ρ·√(r·out) = lr·α·ρ·√out`, independent of
  r (s√r = α). AdamW in the noise regime gives nearly the same size. With PE-8 this makes
  "rank 64 at lr/√2" the same rule for both optimizers.
- **A vs B.** B dominates the change in W (A's W-effect is scaled by ‖B‖, small for
  hundreds of steps). A spectral/μP split would imply a B/A LR ratio ≈ 110, LoRA+
  territory, and LoRA+ ×16 was the worst run measured (below). A and B share one LR.
- **Clipping** matters to Muon only through the momentum mix (msign is scale-invariant).

## Measured results

History, measured on an RTX 5060 Ti during development (2B, 300 steps).

**mix-v1, Keller-5 coefficients**, val score = NLL + ECE:

| run | val score | val ECE | held-out gen. NLL / ECE |
|---|---|---|---|
| AdamW 7.5e-5 | 0.425 | 0.041 | 0.393 / 0.053 |
| AdamW 3.75e-5 | 0.464 | 0.042 | 0.450 / 0.055 |
| AdamW 1.125e-4 | 0.443 | 0.038 | 0.482 / 0.048 |
| Muon 7.5e-5 | 0.419 | 0.024 | 0.401 / 0.036 |
| Muon 1.5e-4 | 0.479 | 0.037 | 0.475 / 0.053 |
| LoRA+ ×16 at 3.75e-5 | 0.706 | 0.056 | 0.645 / 0.054 |

A second seed of Muon 7.5e-5 scored 0.457: the seed spread (±0.02) is larger than the
0.006 Muon/AdamW gap. **Tie.**

**mix-v3, PE-8, paired training loss.** Runs on one seed see identical batches, so
window-by-window training-NLL differences cancel batch noise (SE ≈ 0.010 from one seed,
against an eval-score seed sd of 0.026). Differences against AdamW 6.4e-5:

| run | stable phase (70–240) | cooldown (250–300) |
|---|---|---|
| AdamW 4.5e-5 | −0.036 ± 0.010 | −0.038 ± 0.014 |
| AdamW 9e-5 | +0.018 ± 0.010 | +0.001 ± 0.011 |
| Muon-PE 3.5e-5 | −0.035 ± 0.012 | −0.023 ± 0.012 |
| Muon-PE 5e-5 | −0.024 ± 0.011 | −0.017 ± 0.012 |
| Muon-PE 7e-5 | +0.015 ± 0.010 | +0.040 ± 0.007 |

- Muon-PE's best point ties AdamW's best point (−0.035 vs −0.036 stable; the cooldown gap
  is within 1 SE).
- Both optima are at or below the lowest rate tested. Lower is better for both
  optimizers, consistent with Qu et al.'s finding that Muon on an Adam-pretrained model
  prefers a smaller LR (arXiv 2605.10468).
- The literature agrees: Moonlight reports no SFT advantage when the fine-tuning optimizer
  differs from the pretraining one (arXiv 2502.16982); on AdamW-pretrained Qwen2.5-3B,
  sMuon's benchmark has AdamW 74.5 vs per-factor Muon 73.6 (arXiv 2608.14492); PoLoRA
  finds per-factor Muon "does not improve over Adam" (arXiv 2607.17620).

## Decision rule

1. AdamW is the default (`optim.name=adamw`). The TPU campaign sweeps AdamW only.
2. Muon (PE-8, ρ = 0.2, β = 0.95 Nesterov, wd 0, equal A/B LR) is an option at the AdamW
   rates or slightly below.
3. Muon replaces AdamW only if its best grid point beats AdamW's by 2 paired standard
   errors with ECE no worse.
4. **sMuon** is implemented only if Muon-PE ties or beats AdamW on 2 seeds. Costs: about a
   day of code (matmul-only, a 2r×2r core); it needs a semi-orthonormal init with B ≠ 0, so
   the frozen W needs a −B₀A₀ offset handled at export; no calibration results published.

## Open points

- Real momentum spectra were not measured; logging `rms(update)/lr` per module type
  (0.200 under PE-8) would confirm the synthetic "realistic" rows.
- Qwen3.5 QK-norm is unverified; it would matter only for QK-Clip (Kimi K2, arXiv
  2507.20534), which LoRA-scale deltas do not need.
