# Calibration: is a per-type temperature enough?

Runs studied: `final-4b-h3000` at its selected step 1200 and at its last step 3000 (4B,
LoRA r64, WSD ~1 epoch; W&B `tychota/tjev-sweep` 9025cfdc), and `final-4b-h1500` at step
1450. The per-item results use the K = 2 readout (`tjev post --orders 2`: two orderings
per item, averaged), shuffled noul order; the grouped evidence further down is K = 1.

## Verdict for the finals

1. **Ship per-type temperature scaling** (the current default, `--grouping type`). Do not
   switch to `type_lang`: at the selected checkpoints it changes held-out NLL by ≤ 0.001
   and makes held-out ECE *worse* at step 1200 (0.054 → 0.066); it only pays off on an
   over-trained checkpoint (step 3000: held-out NLL 0.516 → 0.498), which selection should
   not pick in the first place.
2. **The feature-conditioned temperature ("thermometer-lite") is the best single
   calibrator**: the lowest held-out NLL among temperature methods on all three
   checkpoints, the lowest CV and validation NLL at 1200 and 3000 (within 0.0002 of the
   best at 1450), and lower held-out ECE at 1200 (0.054 → 0.042) and 3000
   (0.118 → 0.113); at 1450 its ECE is level with per type (0.060 vs 0.059, 1 item's worth
   on 220). It passes the decision rule at 1200 and 3000 and ties at 1450. But at a
   well-selected checkpoint the gain is small (held-out NLL −0.002 at 1200, −0.003 at 1450),
   and it is not servable yet (the server would need the option count, the prompt length
   and the uncalibrated distribution before it picks T; about a day's work). Wire it for
   the finals only if a late checkpoint gets shipped; otherwise per type.
3. **Ensembling two checkpoints from different runs is the one big lever**: 1200 (h3000)
   + 1450 (h1500), each per-type calibrated, probabilities averaged: held-out NLL 0.365 vs
   0.384 / 0.389 alone, ECE 0.045, accuracy 0.818 vs 0.808 / 0.802. That is level with the
   per-source oracle bound (0.364), at twice the inference cost (two adapters). Two
   checkpoints of the *same* run (1200 + 3000) help validation (0.416) but not held-out
   (0.382): step 3000 is over-fit out of distribution. The pair is complementary by
   language: h3000@1200 is better in English (held-out en NLL 0.298 vs 0.366), h1500@1450
   in French (0.414 vs 0.476).
4. Nothing beyond temperature wins consistently. Position-space vector scaling has the
   best CV at 1200 but moves 6 % of held-out argmaxes and loses accuracy there (0.808 →
   0.804), and its biases flip sign between checkpoints. Batch calibration on the eval
   batch beats the feature T on held-out NLL and ECE at 1450 (0.385 / 0.055 vs 0.386 /
   0.060) but loses at 1200 and 3000, and it needs a batch of requests at serving; the
   servable variant (biases from the calibration split) does nothing. Platt / beta for
   noul improve ECE only at 3000 (0.099 vs 0.113) and lose NLL. None is shipped. Order averaging stays as is (mean of
   probabilities; the first ordering alone loses 0.8 points held-out accuracy at 1200).
5. **JevBench v1.5 noul**: every calibrator softens noul (T ≈ 1.3) and pushes more answers
   into the [0.2, 0.8] abstention band: held-out abstentions 35 % → 42 % (per type), v1.5
   noul accuracy 0.60 → 0.55. That is the calibration being honest, not a bug. To score v1.5
   accuracy, use the serve abstention policy (`committed`, `into_noul`), not a sharper
   calibrator.
6. Remaining gap: the per-source oracle (an upper bound, sources are unknown at serving)
   reaches held-out NLL 0.364 at 1200 against 0.382 for the feature T; per type fitted on
   held-out itself gives 0.374. The rest is domain shift, which calibration cannot fix
   (training side, below).

A defect found on the real data and fixed: with `type_lang`, the 38-slot fr score cell is
all-correct in every fold, so its own T ran to the 0.05 bound (5 held-out items at
p > 1 − 10⁻⁹) and CV could not see it. Cells under 50 slots now keep the type's T
(`fit_grouped`, test `test_a_small_all_correct_cell_does_not_run_to_the_temperature_bound`).

## Data

| set | items | en / fr | noul / choice / score | used for |
|---|---|---|---|---|
| mix calibration split (mix-v4) | 11,188 | 8,695 / 2,493 | 4,655 / 4,758 / 1,775 | fitting (CV) |
| mix validation, full (post) | 11,234 | 8,714 / 2,520 | 4,675 / 4,769 / 1,790 | in-distribution |
| mix validation, train-loop (64 per source) | 2,946 | 2,221 / 725 | 1,338 / 1,196 / 412 | selection |
| held-out generators, reporting seed | 220 | 114 / 106 | 100 / 100 / 20 | domain-shifted |

Calibration cells (type × lang): en choice 3,771, en noul 3,187, en score 1,737, fr choice
987, fr noul 1,468, **fr score 38**. The fr score cell is why per-cell temperatures must be
shrunk.

The per-item results below come from the prediction dumps `tjev post` writes
(`post/predictions/*.jsonl.gz`, [Implementation](#implementation)). Before those existed
the eval JSONs and W&B held grouped metrics only; that earlier evidence is kept further
down ([the grouped reports](#evidence-from-the-grouped-reports-step-1200-per-type-t-applied)),
and the two agree.

## Results (per-item, K = 2)

Calibration split 11,187 items (fit and 5-fold CV), validation 11,233, held-out generators
220 (en 113, fr 107; noul 100, choice 100, score 20). Columns: CV NLL on the calibration
split; validation NLL / ECE / Brier / accuracy; held-out NLL / ECE / Brier / accuracy;
noul abstention rate in the v1.5 band (validation / held-out); v1.5 noul accuracy
(abstentions wrong; validation / held-out); flags = items with p < 10⁻⁶ on a gold label /
items with p > 1 − 10⁻⁹ (validation + held-out); share of held-out argmaxes moved from T = 1.

### final-4b-h3000 @ 1200 (the selected checkpoint)

| method | CV | val NLL | val ECE | val Brier | val acc | HO NLL | HO ECE | HO Brier | HO acc | noul abst. | v1.5 noul acc | flags | argmax moved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none (T = 1) | 0.4547 | 0.4519 | 0.0344 | 0.2175 | 0.808 | 0.4044 | 0.0785 | 0.2059 | 0.808 | 0.17 / 0.35 | 0.782 / 0.600 | 0 / 0 | 0.000 |
| global T | 0.4504 | 0.4474 | 0.0167 | 0.2150 | 0.807 | 0.3906 | 0.0640 | 0.2011 | 0.808 | 0.21 / 0.39 | 0.753 / 0.570 | 0 / 0 | 0.000 |
| per type (old artifact) | 0.4494 | 0.4465 | 0.0184 | 0.2148 | 0.808 | 0.3838 | 0.0545 | 0.1987 | 0.808 | 0.24 / 0.42 | 0.729 / 0.550 | 0 / 0 | 0.000 |
| per lang | 0.4502 | 0.4471 | 0.0169 | 0.2150 | 0.808 | 0.3890 | 0.0699 | 0.2002 | 0.808 | 0.21 / 0.41 | 0.752 / 0.560 | 0 / 0 | 0.000 |
| per type × lang, unshrunk | 0.4491 | 0.4460 | 0.0174 | 0.2147 | 0.808 | 0.3830 | 0.0663 | 0.1985 | 0.808 | 0.24 / 0.42 | 0.729 / 0.550 | 0 / 0 | 0.000 |
| per type × lang, ridge (CV tau) (impl.) | 0.4491 | 0.4460 | 0.0174 | 0.2147 | 0.808 | 0.3830 | 0.0663 | 0.1985 | 0.808 | 0.24 / 0.42 | 0.728 / 0.550 | 0 / 0 | 0.000 |
| per type × lang, ridge, served (guessed lang) | 0.4491 | 0.4462 | 0.0174 | 0.2147 | 0.808 | 0.3829 | 0.0664 | 0.1985 | 0.808 | 0.24 / 0.42 | 0.729 / 0.550 | 0 / 0 | 0.000 |
| per type × lang, empirical Bayes | 0.4492 | 0.4462 | 0.0179 | 0.2147 | 0.808 | 0.3832 | 0.0541 | 0.1986 | 0.808 | 0.24 / 0.43 | 0.728 / 0.540 | 0 / 0 | 0.000 |
| T(n) = a + b ln n | 0.4484 | 0.4469 | 0.0160 | 0.2146 | 0.808 | 0.3846 | 0.0520 | 0.1987 | 0.808 | 0.24 / 0.43 | 0.727 / 0.540 | 0 / 0 | 0.000 |
| T(n) = a_type + b ln n | 0.4487 | 0.4467 | 0.0177 | 0.2146 | 0.808 | 0.3834 | 0.0502 | 0.1984 | 0.808 | 0.24 / 0.44 | 0.724 / 0.530 | 0 / 0 | 0.000 |
| feature T (thermometer-lite) | 0.4474 | 0.4452 | 0.0121 | 0.2140 | 0.808 | 0.3821 | 0.0415 | 0.1959 | 0.808 | 0.27 / 0.44 | 0.701 / 0.530 | 0 / 0 | 0.000 |
| vector scaling: T + position bias, choice | 0.4496 | 0.4469 | 0.0186 | 0.2150 | 0.806 | 0.3842 | 0.0595 | 0.1989 | 0.804 | 0.24 / 0.42 | 0.729 / 0.550 | 0 / 0 | 0.064 |
| vector scaling: T + position bias, every type | 0.4464 | 0.4443 | 0.0121 | 0.2140 | 0.807 | 0.3808 | 0.0476 | 0.1950 | 0.804 | 0.27 / 0.44 | 0.703 / 0.530 | 0 / 0 | 0.064 |
| batch calibration (eval batch), choice+noul | 0.4504 | 0.4469 | 0.0180 | 0.2150 | 0.807 | 0.3834 | 0.0568 | 0.1988 | 0.817 | 0.24 / 0.42 | 0.729 / 0.550 | 0 / 0 | 0.023 |
| batch calibration (calibration split), choice+noul | 0.4491 | 0.4466 | 0.0172 | 0.2148 | 0.807 | 0.3844 | 0.0546 | 0.1989 | 0.808 | 0.24 / 0.42 | 0.729 / 0.550 | 0 / 0 | 0.036 |
| per type T + Platt with yes-bias (noul) | 0.4494 | 0.4463 | 0.0177 | 0.2147 | 0.809 | 0.3851 | 0.0518 | 0.1994 | 0.808 | 0.23 / 0.42 | 0.734 / 0.540 | 0 / 0 | 0.009 |
| per type T + beta (noul) | 0.4494 | 0.4464 | 0.0179 | 0.2147 | 0.809 | 0.3850 | 0.0495 | 0.1995 | 0.808 | 0.23 / 0.42 | 0.734 / 0.540 | 0 / 0 | 0.009 |
| orderings: first only, per type T | 0.4515 | 0.4489 | 0.0151 | 0.2158 | 0.808 | 0.3823 | 0.0560 | 0.1981 | 0.800 | 0.24 / 0.43 | 0.728 / 0.530 | 0 / 0 | 0.050 |
| orderings: mean of probabilities, per type T | 0.4494 | 0.4465 | 0.0184 | 0.2148 | 0.808 | 0.3838 | 0.0545 | 0.1987 | 0.808 | 0.24 / 0.42 | 0.729 / 0.550 | 0 / 0 | 0.000 |
| orderings: log-space mean, per type T | 0.4495 | 0.4464 | 0.0176 | 0.2147 | 0.808 | 0.3839 | 0.0510 | 0.1986 | 0.808 | 0.24 / 0.43 | 0.728 / 0.540 | 0 / 0 | 0.000 |
| per source (upper bound) | 0.4434 | 0.4483 | 0.0151 | 0.2122 | 0.808 | 0.3643 | 0.0285 | 0.1890 | 0.808 | 0.23 / 0.55 | 0.739 / 0.440 | 5 / 529 | 0.000 |
| [diagnostic] per type T + histogram binning | 0.4527 | 0.4489 | 0.0112 | 0.2161 | 0.806 | 0.3924 | 0.0523 | 0.2006 | 0.808 | 0.30 / 0.44 | 0.676 / 0.530 | 0 / 0 | 0.000 |
| [diagnostic] per type T + isotonic (top label) | 0.4573 | 0.4481 | 0.0089 | 0.2140 | 0.808 | 0.3749 | 0.0570 | 0.1916 | 0.818 | 0.24 / 0.42 | 0.729 / 0.550 | 5 / 578 | 0.032 |

### final-4b-h3000 @ 3000 (end of the cooldown)

| method | CV | val NLL | val ECE | val Brier | val acc | HO NLL | HO ECE | HO Brier | HO acc | noul abst. | v1.5 noul acc | flags | argmax moved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none (T = 1) | 0.4467 | 0.4458 | 0.0359 | 0.2030 | 0.826 | 0.6507 | 0.1321 | 0.2581 | 0.798 | 0.14 / 0.28 | 0.809 / 0.630 | 0 / 8 | 0.000 |
| global T | 0.4308 | 0.4292 | 0.0105 | 0.1999 | 0.826 | 0.5281 | 0.1131 | 0.2415 | 0.798 | 0.20 / 0.36 | 0.768 / 0.590 | 0 / 0 | 0.000 |
| per type (old artifact) | 0.4278 | 0.4270 | 0.0076 | 0.1997 | 0.826 | 0.5159 | 0.1176 | 0.2371 | 0.798 | 0.24 / 0.40 | 0.735 / 0.550 | 0 / 0 | 0.000 |
| per lang | 0.4303 | 0.4280 | 0.0117 | 0.1996 | 0.826 | 0.5058 | 0.1157 | 0.2376 | 0.798 | 0.20 / 0.37 | 0.769 / 0.580 | 0 / 0 | 0.000 |
| per type × lang, unshrunk | 0.4276 | 0.4261 | 0.0076 | 0.1994 | 0.826 | 0.4961 | 0.1168 | 0.2346 | 0.798 | 0.24 / 0.40 | 0.734 / 0.550 | 0 / 0 | 0.000 |
| per type × lang, ridge (CV tau) (impl.) | 0.4276 | 0.4261 | 0.0076 | 0.1994 | 0.826 | 0.4980 | 0.1170 | 0.2349 | 0.798 | 0.24 / 0.40 | 0.734 / 0.550 | 0 / 0 | 0.000 |
| per type × lang, ridge, served (guessed lang) | 0.4277 | 0.4263 | 0.0081 | 0.1995 | 0.826 | 0.4979 | 0.1170 | 0.2349 | 0.798 | 0.24 / 0.40 | 0.734 / 0.550 | 0 / 0 | 0.000 |
| per type × lang, empirical Bayes | 0.4277 | 0.4264 | 0.0076 | 0.1995 | 0.826 | 0.5058 | 0.1219 | 0.2359 | 0.798 | 0.24 / 0.40 | 0.733 / 0.550 | 0 / 0 | 0.000 |
| T(n) = a + b ln n | 0.4302 | 0.4290 | 0.0115 | 0.1995 | 0.826 | 0.5370 | 0.1140 | 0.2405 | 0.798 | 0.21 / 0.39 | 0.756 / 0.560 | 0 / 0 | 0.000 |
| T(n) = a_type + b ln n | 0.4282 | 0.4269 | 0.0081 | 0.1997 | 0.826 | 0.5153 | 0.1176 | 0.2370 | 0.798 | 0.24 / 0.40 | 0.734 / 0.550 | 0 / 0 | 0.000 |
| feature T (thermometer-lite) | 0.4267 | 0.4254 | 0.0103 | 0.1995 | 0.826 | 0.4925 | 0.1128 | 0.2346 | 0.798 | 0.22 / 0.40 | 0.747 / 0.550 | 0 / 0 | 0.000 |
| vector scaling: T + position bias, choice | 0.4279 | 0.4273 | 0.0073 | 0.1998 | 0.825 | 0.5140 | 0.1174 | 0.2369 | 0.798 | 0.24 / 0.40 | 0.735 / 0.550 | 0 / 0 | 0.000 |
| vector scaling: T + position bias, every type | 0.4275 | 0.4273 | 0.0093 | 0.1999 | 0.826 | 0.5129 | 0.1170 | 0.2362 | 0.798 | 0.24 / 0.40 | 0.735 / 0.550 | 0 / 0 | 0.000 |
| batch calibration (eval batch), choice+noul | 0.4289 | 0.4272 | 0.0068 | 0.1998 | 0.826 | 0.5136 | 0.1126 | 0.2372 | 0.798 | 0.24 / 0.40 | 0.735 / 0.550 | 0 / 0 | 0.000 |
| batch calibration (calibration split), choice+noul | 0.4278 | 0.4271 | 0.0083 | 0.1997 | 0.826 | 0.5152 | 0.1176 | 0.2370 | 0.798 | 0.24 / 0.40 | 0.735 / 0.550 | 0 / 0 | 0.000 |
| per type T + Platt with yes-bias (noul) | 0.4271 | 0.4262 | 0.0084 | 0.1992 | 0.826 | 0.5160 | 0.0990 | 0.2361 | 0.809 | 0.22 / 0.40 | 0.744 / 0.550 | 0 / 0 | 0.027 |
| per type T + beta (noul) | 0.4274 | 0.4263 | 0.0083 | 0.1992 | 0.826 | 0.5158 | 0.0991 | 0.2362 | 0.809 | 0.23 / 0.40 | 0.743 / 0.550 | 0 / 0 | 0.027 |
| orderings: first only, per type T | 0.4298 | 0.4275 | 0.0089 | 0.2000 | 0.826 | 0.5103 | 0.1027 | 0.2356 | 0.803 | 0.24 / 0.39 | 0.736 / 0.560 | 0 / 0 | 0.018 |
| orderings: mean of probabilities, per type T | 0.4278 | 0.4270 | 0.0076 | 0.1997 | 0.826 | 0.5159 | 0.1176 | 0.2371 | 0.798 | 0.24 / 0.40 | 0.735 / 0.550 | 0 / 0 | 0.000 |
| orderings: log-space mean, per type T | 0.4274 | 0.4270 | 0.0085 | 0.1998 | 0.826 | 0.5150 | 0.1180 | 0.2373 | 0.798 | 0.24 / 0.40 | 0.734 / 0.550 | 0 / 0 | 0.000 |
| per source (upper bound) | 0.4111 | 0.4098 | 0.0083 | 0.1938 | 0.826 | 0.3982 | 0.0669 | 0.2071 | 0.798 | 0.23 / 0.57 | 0.748 / 0.420 | 2 / 485 | 0.000 |
| [diagnostic] per type T + histogram binning | 0.4302 | 0.4299 | 0.0115 | 0.2015 | 0.825 | 0.5215 | 0.1111 | 0.2397 | 0.798 | 0.20 / 0.35 | 0.769 / 0.600 | 0 / 0 | 0.000 |
| [diagnostic] per type T + isotonic (top label) | 0.4324 | 0.4296 | 0.0122 | 0.2004 | 0.823 | 0.5347 | 0.1099 | 0.2384 | 0.822 | 0.25 / 0.39 | 0.726 / 0.560 | 5 / 1232 | 0.041 |

### final-4b-h1500 @ 1450

| method | CV | val NLL | val ECE | val Brier | val acc | HO NLL | HO ECE | HO Brier | HO acc | noul abst. | v1.5 noul acc | flags | argmax moved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none (T = 1) | 0.4357 | 0.4338 | 0.0242 | 0.2078 | 0.817 | 0.4058 | 0.0696 | 0.1923 | 0.802 | 0.18 / 0.37 | 0.774 / 0.590 | 0 / 0 | 0.000 |
| global T | 0.4335 | 0.4315 | 0.0139 | 0.2067 | 0.817 | 0.3956 | 0.0533 | 0.1902 | 0.802 | 0.20 / 0.38 | 0.762 / 0.580 | 0 / 0 | 0.000 |
| per type (old artifact) | 0.4319 | 0.4297 | 0.0140 | 0.2064 | 0.817 | 0.3892 | 0.0585 | 0.1886 | 0.802 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.000 |
| per lang | 0.4334 | 0.4312 | 0.0132 | 0.2067 | 0.817 | 0.3961 | 0.0575 | 0.1897 | 0.802 | 0.20 / 0.40 | 0.760 / 0.560 | 0 / 0 | 0.000 |
| per type × lang, unshrunk | 0.4320 | 0.4295 | 0.0138 | 0.2064 | 0.817 | 0.3902 | 0.0556 | 0.1886 | 0.802 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.000 |
| per type × lang, ridge (CV tau) (impl.) | 0.4319 | 0.4296 | 0.0144 | 0.2064 | 0.817 | 0.3897 | 0.0559 | 0.1886 | 0.802 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.000 |
| per type × lang, ridge, served (guessed lang) | 0.4319 | 0.4296 | 0.0141 | 0.2064 | 0.817 | 0.3897 | 0.0560 | 0.1886 | 0.802 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.000 |
| per type × lang, empirical Bayes | 0.4319 | 0.4297 | 0.0140 | 0.2064 | 0.817 | 0.3892 | 0.0585 | 0.1886 | 0.802 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.000 |
| T(n) = a + b ln n | 0.4315 | 0.4304 | 0.0106 | 0.2063 | 0.817 | 0.3873 | 0.0646 | 0.1887 | 0.802 | 0.22 / 0.42 | 0.742 / 0.540 | 0 / 0 | 0.000 |
| T(n) = a_type + b ln n | 0.4315 | 0.4300 | 0.0130 | 0.2063 | 0.817 | 0.3867 | 0.0657 | 0.1885 | 0.802 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.000 |
| feature T (thermometer-lite) | 0.4317 | 0.4295 | 0.0090 | 0.2060 | 0.817 | 0.3862 | 0.0595 | 0.1877 | 0.802 | 0.24 / 0.44 | 0.727 / 0.520 | 0 / 0 | 0.000 |
| vector scaling: T + position bias, choice | 0.4322 | 0.4301 | 0.0135 | 0.2065 | 0.818 | 0.3874 | 0.0498 | 0.1880 | 0.812 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.027 |
| vector scaling: T + position bias, every type | 0.4322 | 0.4301 | 0.0106 | 0.2065 | 0.818 | 0.3879 | 0.0510 | 0.1873 | 0.812 | 0.23 / 0.42 | 0.733 / 0.540 | 0 / 0 | 0.027 |
| batch calibration (eval batch), choice+noul | 0.4331 | 0.4301 | 0.0141 | 0.2066 | 0.817 | 0.3851 | 0.0546 | 0.1874 | 0.812 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.055 |
| batch calibration (calibration split), choice+noul | 0.4319 | 0.4299 | 0.0151 | 0.2064 | 0.817 | 0.3892 | 0.0586 | 0.1885 | 0.802 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.005 |
| per type T + Platt with yes-bias (noul) | 0.4320 | 0.4297 | 0.0139 | 0.2063 | 0.817 | 0.3896 | 0.0576 | 0.1887 | 0.806 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.009 |
| per type T + beta (noul) | 0.4318 | 0.4296 | 0.0132 | 0.2063 | 0.818 | 0.3873 | 0.0611 | 0.1889 | 0.806 | 0.22 / 0.42 | 0.739 / 0.540 | 0 / 0 | 0.023 |
| orderings: first only, per type T | 0.4336 | 0.4314 | 0.0107 | 0.2070 | 0.817 | 0.3897 | 0.0576 | 0.1893 | 0.812 | 0.23 / 0.41 | 0.737 / 0.550 | 0 / 0 | 0.009 |
| orderings: mean of probabilities, per type T | 0.4319 | 0.4297 | 0.0140 | 0.2064 | 0.817 | 0.3892 | 0.0585 | 0.1886 | 0.802 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.000 |
| orderings: log-space mean, per type T | 0.4318 | 0.4297 | 0.0141 | 0.2063 | 0.817 | 0.3889 | 0.0578 | 0.1884 | 0.802 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.000 |
| per source (upper bound) | 0.4288 | 0.4306 | 0.0108 | 0.2042 | 0.817 | 0.3655 | 0.0382 | 0.1868 | 0.802 | 0.23 / 0.61 | 0.737 / 0.390 | 4 / 362 | 0.000 |
| [diagnostic] per type T + histogram binning | 0.4362 | 0.4346 | 0.0120 | 0.2093 | 0.814 | 0.4037 | 0.0742 | 0.1989 | 0.802 | 0.20 / 0.38 | 0.758 / 0.580 | 0 / 0 | 0.000 |
| [diagnostic] per type T + isotonic (top label) | 0.4417 | 0.4348 | 0.0086 | 0.2067 | 0.818 | 0.3844 | 0.0509 | 0.1877 | 0.810 | 0.21 / 0.40 | 0.754 / 0.560 | 3 / 823 | 0.073 |

### Checkpoint ensembles (first = h3000 @ 1200)

| pair | method | CV | val NLL | val ECE | val Brier | val acc | HO NLL | HO ECE | HO Brier | HO acc | noul abst. | v1.5 noul acc | flags | argmax moved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1200 + 3000 (same run) | per type (old artifact) | 0.4494 | 0.4465 | 0.0184 | 0.2148 | 0.808 | 0.3838 | 0.0545 | 0.1987 | 0.808 | 0.24 / 0.42 | 0.729 / 0.550 | 0 / 0 | 0.000 |
| 1200 + 3000 (same run) | checkpoint 2 alone, per type T | 0.4278 | 0.4270 | 0.0076 | 0.1997 | 0.826 | 0.5159 | 0.1176 | 0.2371 | 0.798 | 0.24 / 0.40 | 0.735 / 0.550 | 0 / 0 | 0.091 |
| 1200 + 3000 (same run) | checkpoints 1 + 2, mean of calibrated probabilities | 0.4175 | 0.4157 | 0.0129 | 0.1979 | 0.824 | 0.3824 | 0.0617 | 0.2002 | 0.818 | 0.25 / 0.43 | 0.726 / 0.540 | 0 / 0 | 0.068 |
| 1200 + h1500@1450 | per type (old artifact) | 0.4494 | 0.4465 | 0.0184 | 0.2148 | 0.808 | 0.3838 | 0.0545 | 0.1987 | 0.808 | 0.24 / 0.42 | 0.729 / 0.550 | 0 / 0 | 0.000 |
| 1200 + h1500@1450 | checkpoint 2 alone, per type T | 0.4319 | 0.4297 | 0.0140 | 0.2064 | 0.817 | 0.3892 | 0.0585 | 0.1886 | 0.802 | 0.23 / 0.42 | 0.737 / 0.540 | 0 / 0 | 0.127 |
| 1200 + h1500@1450 | checkpoints 1 + 2, mean of calibrated probabilities | 0.4353 | 0.4328 | 0.0129 | 0.2078 | 0.813 | 0.3646 | 0.0449 | 0.1849 | 0.818 | 0.23 / 0.42 | 0.735 / 0.540 | 0 / 0 | 0.077 |

"argmax moved" for the ensembles is against checkpoint 1 at T = 1.

### By language and type (held-out NLL / ECE; validation in the study files)

| checkpoint | method | en | fr | noul | choice | score |
|---|---|---|---|---|---|---|
| h3000 @ 1200 | none (T = 1) | 0.320 / 0.053 | 0.495 / 0.114 | 0.478 / 0.124 | 0.407 / 0.052 | 0.022 / 0.019 |
| h3000 @ 1200 | per type (old artifact) | 0.298 / 0.047 | 0.476 / 0.088 | 0.437 / 0.099 | 0.403 / 0.040 | 0.023 / 0.020 |
| h3000 @ 1200 | per type × lang, ridge, served (guessed lang) | 0.297 / 0.047 | 0.476 / 0.107 | 0.437 / 0.109 | 0.401 / 0.048 | 0.023 / 0.020 |
| h3000 @ 1200 | feature T (thermometer-lite) | 0.295 / 0.037 | 0.475 / 0.069 | 0.433 / 0.083 | 0.402 / 0.041 | 0.027 / 0.023 |
| h3000 @ 3000 | none (T = 1) | 0.405 / 0.095 | 0.914 / 0.173 | 0.582 / 0.156 | 0.838 / 0.129 | 0.060 / 0.040 |
| h3000 @ 3000 | per type (old artifact) | 0.325 / 0.074 | 0.721 / 0.165 | 0.468 / 0.142 | 0.655 / 0.114 | 0.060 / 0.041 |
| h3000 @ 3000 | per type × lang, ridge, served (guessed lang) | 0.326 / 0.074 | 0.683 / 0.172 | 0.469 / 0.134 | 0.614 / 0.119 | 0.060 / 0.041 |
| h3000 @ 3000 | feature T (thermometer-lite) | 0.318 / 0.074 | 0.680 / 0.172 | 0.464 / 0.125 | 0.607 / 0.120 | 0.063 / 0.044 |
| h1500 @ 1450 | none (T = 1) | 0.388 / 0.074 | 0.425 / 0.079 | 0.471 / 0.101 | 0.410 / 0.053 | 0.062 / 0.044 |
| h1500 @ 1450 | per type (old artifact) | 0.366 / 0.055 | 0.414 / 0.070 | 0.438 / 0.092 | 0.406 / 0.054 | 0.062 / 0.045 |
| h1500 @ 1450 | per type × lang, ridge, served (guessed lang) | 0.367 / 0.055 | 0.415 / 0.064 | 0.438 / 0.092 | 0.407 / 0.053 | 0.062 / 0.045 |
| h1500 @ 1450 | feature T (thermometer-lite) | 0.362 / 0.055 | 0.413 / 0.073 | 0.436 / 0.077 | 0.401 / 0.053 | 0.065 / 0.048 |
| 1200 + h1500@1450 | checkpoint ensemble | 0.303 / 0.040 | 0.431 / 0.068 | 0.427 / 0.069 | 0.367 / 0.043 | 0.041 / 0.032 |

Reading: French is where the late checkpoint breaks (3000: fr held-out NLL 0.91 raw,
0.72 per type, 0.68 with `type_lang` or the feature T), and where `type_lang` earns its
keep; at 1200 and 1450 the fr cells are already served by per type (fr NLL 0.476 vs
0.476, 0.414 vs 0.415). noul carries most of the held-out ECE (0.08–0.14 after any
temperature): its over-confidence is source-specific (gen_return_window,
gen_invoice_total), not a language or type effect.

Fitted parameters worth knowing (1200): per type T choice 1.13, noul 1.31, score 1.06;
`type_lang` cells choice/en 1.09, choice/fr 1.28, noul/en 1.33, noul/fr 1.27 (tau 10);
feature T: CV lambda 1, the weight is on the uncalibrated entropy (+0.26) and top
probability (+0.22), option count / length / language near 0. Platt's yes-bias c ≈ +0.09
(1200), +0.19 (3000), +0.04 (1450), i.e. no systematic yes-bias; its slope a ≈ 0.76 is the
per-type noul T. The position biases are not stable across checkpoints (noul position 1
vs 0: −1.78 at 1200, −0.42 at 3000, +0.57 at 1450), another reason not to ship them.

## Evidence from the grouped reports (step 1200, per-type T applied)

Local post and the TPU post uploaded to `tychot/tjev-4b` (`final-4b-h3000/post/`) agree to
±0.001 NLL. Artifact: T(noul) 1.324, T(choice) 1.155, T(score) 1.062 (global 1.185). CV gain on the
calibration split itself is small: NLL 0.285 → 0.274 noul, 0.433 → 0.428 choice.

| set | slice | n | acc | conf | conf − acc | NLL | ECE |
|---|---|---|---|---|---|---|---|
| validation | en | 8,713 | .789 | .801 | +.012 | .490 | .016 |
| validation | fr | 2,520 | .875 | .886 | +.011 | .307 | .018 |
| held-out | en | 114 | .854 | .898 | +.044 | .292 | .045 |
| held-out | fr | 106 | .752 | .811 | +.059 | .479 | .073 |
| held-out | noul | 100 | .751 | .841 | **+.090** | .441 | .101 |
| held-out | choice | 100 | .820 | .845 | +.025 | .394 | .040 |
| held-out | score | 20 | 1.000 | .980 | −.020 | .023 | .020 |

Validation, single-language sources aggregated by (type, lang) after per-type T (the
bilingual generators, certo, decider_teacher, xnli/mnli/snli etc. excluded):

| cell | n | conf − acc | NLL |
|---|---|---|---|
| choice en | 2,087 | +.005 | .398 |
| choice fr | 584 | **+.054** | .494 |
| noul en | 1,523 | −.007 | .335 |
| noul fr | 1,151 | **−.016** | .173 |
| score en | 1,500 | +.025 | 1.107 |

Per-source residual gaps (validation, after per-type T) have a standard deviation of .038
over the 26 single-language sources; the largest are source-specific, in both languages:
gen_return_window +.249, gen_invoice_total +.121, tweet_sentiment_fr +.114, gen_base_rate
+.077, pavlick_formality +.069; textdetox_fr −.060, mlma_en −.046, mage −.036. The held-out
over-confidence is the same generators: gen_invoice_total +.225, gen_return_window +.161,
gen_ticket_triage +.072.

Reading:

1. **Per language alone is the wrong axis.** In French, choice wants a *higher* T (+5.4
   points over-confident, mostly tweet_sentiment_fr) and noul a *lower* one (−1.6, the
   easy fr noul sources: hc3_fr, textdetox_fr, allocine_fr). A per-lang T averages them out.
2. **Per (type × lang) has a real in-distribution signal** at step 1200 for fr choice; the
   other cells are within ±2.5 points. At step 1200 the fr over-confidence of the cooldown
   (steps 2400 → 3000: val fr ECE .062 → .090, held-out fr NLL .640 → .710) has not
   happened yet; per-type T already brings both languages to ECE ≈ .017 on validation.
3. **The held-out gap is domain shift, not language**: +.044 en and +.059 fr after
   calibration, concentrated in noul and in two generators (invoice total, return window)
   that are already over-confident in distribution. A temperature fitted in distribution
   cannot remove it; only a per-source T could (an upper bound: sources are unknown at
   serving), or training-side fixes.
4. Risk for per (type × lang) on held-out: the noul/fr cell's lower T (driven by easy fr
   noul sources) would sharpen the held-out fr noul generators, which are over-confident.
   Whether choice/fr's gain outweighs it is exactly what the per-item study measures.

## The study (per-item)

`scripts/calibration_study.py RUN/post/predictions --out reports/calibration/RUN` fits each
method on the calibration split (5-fold CV NLL reported too) and scores the full
validation and the held-out generators (NLL / ECE / Brier, by lang and type):

| family | methods |
|---|---|
| temperature groupings | none; global; per type (the old artifact); per lang; per type × lang unshrunk; per type × lang ridge on log T towards the type's, tau by CV (**implemented**); the same served with the *guessed* language; empirical Bayes (normal-normal on log T, Laplace curvatures → tau); T(n) = a + b ln n and a_type + b ln n; per source (upper bound) |
| feature-conditioned T ("thermometer-lite") | log T = u_type + w · x, x = (ln option count, guessed fr, unsure language, ln prompt tokens, normalised entropy and top probability of the T = 1 distribution), all known when serving; ridge on w, strength by CV over {0.1 … 10⁴, ∞}; ∞ gives w = 0, i.e. per type. Analytic gradient (checked against finite differences). `tjev.eval.calibrate.fit_features`. The candidate for sources the calibration split never saw |
| position-space vector scaling (Dirichlet-lite) | per type, T plus a bias per *displayed letter position* (position 0 the reference, ≥ 7 shared): choice only, and every type. A label-space vector / matrix does not exist (options vary in number and meaning, and are shuffled); this is the part of Dirichlet calibration that is defined |
| batch calibration | per (type, option count) with ≥ 20 items, subtract log(mean T = 1 probability of each displayed position), from the evaluated batch itself (label-free, transductive, as the paper) or from the calibration split; then per-type T; choice and noul (score levels keep their order, so their "bias" is the real prior). Contextual calibration proper needs a forward pass on a content-free input ("N/A"): not in the dumps, not run |
| noul | Platt (logit p' = a logit p + c: a = 1/T, c the explicit yes-bias) and beta (a ln p − b ln(1−p) + c, the two-class Dirichlet), on the T = 1 order-averaged P(yes). Every method also reports the JevBench v1.5 band reading: abstention rate (0.2 < P(yes) < 0.8), v1.5 accuracy (abstentions wrong), accuracy of the committed answers |
| ensembles | over orderings (needs a K > 1 dump): first ordering only vs mean of probabilities (what tjev serves) vs log-space mean (product of experts), each then per-type T; over checkpoints (`--second DIR`, e.g. the 1500 dump next to 1200; items matched by id): each checkpoint per-type calibrated, probabilities averaged |
| diagnostics only | histogram binning (10 equal-mass bins, smoothed) and isotonic regression (PAV, unclipped) of the top-label confidence: they show how much is left beyond a monotone map, but are piecewise-constant, can emit 0 / 1 and do not transfer to new sources |
| oracles | per type / per type × lang fitted on the evaluated set itself: the floor for any temperature grouping, so "remaining gap" = method − oracle |

Every method row also carries flags: items with p < 10⁻⁶ on a gold label (NLL blow-up),
items with p > 1 − 10⁻⁹, the share of items whose argmax moved from T = 1, and the
accuracy change (temperature never moves the argmax; position biases, batch calibration,
Platt / beta and the ensembles can). The dump also records each item's prompt tokens.

### Methods people use

| method | what it fits | here | reference |
|---|---|---|---|
| temperature scaling | one T on the logits | per type, per type × lang (shrunk), feature T | Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 |
| Platt scaling | a, c on the binary logit | noul, with the yes-bias c reported | Platt, *Probabilistic Outputs for Support Vector Machines…*, 1999 |
| beta calibration | a, b, c on ln p, ln(1 − p) | noul | Kull, Silva Filho, Flach, AISTATS 2017 |
| Dirichlet / vector / matrix scaling | a linear map on the log-probabilities | only its position-bias part (variable, shuffled options) | Kull et al., *Beyond temperature scaling…*, NeurIPS 2019 ([arXiv 1910.12656](https://arxiv.org/abs/1910.12656)) |
| contextual calibration | the bias of a content-free input, divided out | not run (needs a forward pass) | Zhao et al., *Calibrate Before Use*, ICML 2021 ([arXiv 2102.09690](https://arxiv.org/abs/2102.09690)) |
| batch calibration | the batch-mean class probability, divided out | per (type, option count) and position | Zhou et al., *Batch Calibration…*, ICLR 2024 ([arXiv 2309.17249](https://arxiv.org/abs/2309.17249)) |
| option-ID debiasing | the prior over option letters, removed (permutations of a few items) | order averaging (K > 1), position biases | Zheng et al., *Large Language Models Are Not Robust Multiple Choice Selectors*, ICLR 2024 ([arXiv 2309.03882](https://arxiv.org/abs/2309.03882)) |
| an input-dependent temperature | an auxiliary model predicting T per task / input from features | feature T, a linear "lite" version on cheap features | Shen et al., *Thermometer: Towards Universal Calibration for LLMs*, ICML 2024 ([arXiv 2403.08819](https://arxiv.org/abs/2403.08819)) |
| histogram binning / isotonic | a monotone piecewise-constant map | diagnostics only | Zadrozny & Elkan, ICML 2001 / KDD 2002 |
| ensembles | average of several models' probabilities | orderings, two checkpoints | Lakshminarayanan et al., *Deep Ensembles*, NeurIPS 2017 |

Under dataset shift, post-hoc calibration fitted in distribution degrades (Ovadia et al.,
*Can You Trust Your Model's Uncertainty?*, NeurIPS 2019), which is what the held-out
generators show here; ensembles hold up best there, and an input-dependent T is the
cheapest thing that can follow a new source's difficulty.

The study ran on the three K = 2 dumps (the tables above); the full tables per checkpoint
are in `reports/calibration/*/study.md` (and `study.json`). The ordering rows compare the two
views of the K = 2 readout; the checkpoint rows use `--second`. The synthetic test
(`tests/eval/test_calibration_study.py`: fr choice true T = 2 vs 1) checks that the study
ranks a known grouping first and that the ridge does not lose to the unshrunk cells. Each
follow-up method has a synthetic case with a planted effect: T depending on the option
count (the feature T tracks it, corr > 0.8; with no effect w ≈ 0 and NLL ≈ per type), a
+1.5 logit on the first displayed option (vector scaling recovers +1.5 for positions
1–4; both batch-calibration variants beat per type), a +1 yes-bias (Platt's c moves by
−a), per-view and per-checkpoint noise (mean over orderings and over checkpoints beat a
single member), and a planted 0-probability (flagged). In the end-to-end synthetic run,
isotonic is the method that gets flagged for 0 / 1 outputs.

Decision rule for the extra methods: keep one only if, on held-out, it beats the best
temperature row in NLL *and* ECE, has no 0 / 1 flags, and loses no accuracy; methods that
move the argmax (position bias, batch calibration) must also be applied by the server,
so they are worth it only if the held-out gain is clear.

Decision rule for the default (`DEFAULT_GROUPING` in `tjev/eval/calibrate.py`, today
`"type"`): switch to `"type_lang"` only if its *served* row (guessed language) beats per
type on held-out NLL without losing on validation; keep any beyond-temperature method only
if it beats the best temperature row on held-out NLL *and* ECE.

## Implementation

- `tjev.eval.calibrate.fit_grouped(..., grouping="type" | "type_lang")`. `type_lang`: per
  type first (as before, T = 1 a candidate), then each (type, lang) cell minimises
  `n · NLL(T) + tau · (log T − log T_type)²`; tau from {0, 10, 30, 100, 300, 1000, 3000,
  inf} by 5-fold CV on the calibration split (ties → more shrinkage; inf = per type, so the
  CV choice never loses to per type in CV). A 38-slot cell moves at most a fraction of the
  way to its own optimum.
- Artifact: `grouping` recorded; `temperatures` keeps the type keys and adds `"noul/fr"`
  cells; `fit` keeps tau and the CV curve. Artifacts without `grouping` are per type
  (identity unchanged, still valid). `lookup(temps, type, lang)`: cell, else type, else 1.
- Applied by `tjev eval` / post reports (`grouped_report` uses the items' `lang`), `tjev
  serve` (JAX and MLX backends) and therefore `tjev eval-official`, and the MLX eval.
- **Language at serving**: the wire format has no language. The server guesses it with the
  data report's stop-word heuristic, moved to `tjev.data.lang.guess_lang` (state +
  instructions + option descriptions); `None` (unsure) → the type's T. Accuracy of the
  guess against the `lang` tags:

  | set | en → en / fr / None | fr → fr / en / None |
  |---|---|---|
  | calibration | 7,710 / 2 / 983 | 2,395 / 0 / 98 |
  | validation | 7,683 / 1 / 1,030 | 2,418 / 0 / 102 |
  | held-out generators | 84 / 0 / 30 | 97 / 0 / 9 |
  | JevBench public | 199 / 0 / 32 | – |

  3 wrong guesses in 22.4k; 11–12 % of en and 4 % of fr items fall back to per type.
  A per-type artifact needs no guess.
- `--grouping type|type_lang` on `tjev calibrate` and `tjev post`.
- `post/predictions/{calibration,jevbench,validation,generators}{tag}.jsonl.gz`
  (`tjev.eval.predictions`): per item id, source, lang, lang_guess, family, type, labels,
  gold target, raw logits per view in label order, the displayed orders, and the reported
  probabilities. Raw logits, so any temperature / grouping / position bias can be refitted
  offline. `tjev eval --predictions FILE` and `tjev calibrate --predictions FILE` also
  dump them.

## Calibration pool: temperatures fitted on held-out sources

```bash
tjev calibrate RUN MIX/heldout-select.jsonl HELDOUT_SOURCES_DIR/ --orders 2 --out CAL.json   # several files / dirs = a pool
tjev post RUN --mix MIX --jevbench PUBLIC --orders 2 --calibration-pool MIX/heldout-select.jsonl --calibration-pool DIR
tjev eval-v15 RUN/post/predictions/generators-k2.jsonl.gz --fit-on POOL.jsonl.gz      # offline, from dumps
```

The calibration split holds held-out *items* of the training sources: still in
distribution. Kev (github.com/jaredpalmer/kev, PLAN.md) found that one temperature fitted
on held-out *datasets* transfers to new sources where the in-distribution one does not
(their hard-v1 panel: ECE about 0.10 → 0.04–0.05). `tjev calibrate` and `tjev post --calibration-pool` take a
pool (files and directories, an item kept once across files; the artifact's `fit.pool`
lists the files and sources, its `data` fingerprint every file); `post` writes `-pool`
files and refuses a pool that contains a reporting set (JevBench public,
`generators-heldout.jsonl`). `tjev.eval.calibrate.fit_from_predictions` fits the same from
prediction dumps.

Measured on the K = 2 dumps, the only held-out-source data they have being the generators
(11 sources, 220 items): leave one generator out, each scored at the per-type T fitted on
the ten others (`scripts/v15_study.py`; a type with < 50 pool slots takes the pool's
global T), against the in-distribution T and T = 1. ECE is tjev's (15 bins, top label); C
and I are v1.5 (I without the noul commit, then with it).

| run | T | ECE | ECE choice | ECE noul | NLL | v1.5 C | I | C, commit | I, commit |
|---|---|---|---|---|---|---|---|---|---|
| h3000@1200 | T = 1 | 0.078 | 0.052 | 0.124 | 0.404 | 88.0 | 64.8 | | |
| | in-distribution | 0.054 | 0.040 | 0.099 | 0.384 | 88.9 | 61.4 | 86.2 | 75.4 |
| | pool (LOSO) | **0.031** | 0.039 | **0.073** | 0.381 | **91.2** | 58.7 | **87.5** | 75.4 |
| h3000@3000 | T = 1 | 0.132 | 0.129 | 0.156 | 0.651 | 79.7 | 66.8 | | |
| | in-distribution | 0.118 | 0.114 | 0.142 | 0.516 | 82.2 | 61.5 | 78.1 | 73.5 |
| | pool (LOSO) | **0.061** | **0.054** | **0.078** | **0.467** | **87.3** | 57.5 | **82.5** | 73.5 |
| h1500@1450 | T = 1 | 0.070 | 0.053 | 0.101 | 0.406 | 87.7 | 63.2 | | |
| | in-distribution | 0.058 | 0.054 | 0.092 | 0.389 | 88.4 | 59.8 | 85.3 | 74.5 |
| | pool (LOSO) | 0.057 | 0.062 | 0.088 | 0.406 | 87.8 | 55.8 | 86.2 | 74.5 |

Pool temperatures (all generators) are higher than the in-distribution ones (choice 1.22 /
1.30 / 2.54 vs 1.13 / 1.06 / 1.34; noul 2.04 / 1.91 / 3.01 vs 1.31 / 1.31 / 1.60, at
1200 / 1450 / 3000): new sources need softer distributions. In distribution they cost
(validation ECE 0.018 → 0.030 at 1200, 0.014 → 0.039 at 1450, 0.008 → 0.117 at 3000).

Reading: on new sources the pool T halves the ECE at 1200 and 3000 (Kev's effect), and is
level at 1450. Without the noul commit it costs v1.5 Intelligence (a softer noul P(yes)
abstains more: −2.7 to −4.0); with the commit it costs none and adds +0.9 to +4.4
Calibration. JevBench is new sources, so the pool T is the right candidate there, but 220
generator items (also our reporting set) are too few to ship it on: build a pool of held-out
sources the run never trained on (plus the generator *selection* seed set), fit with
`--calibration-pool`, and decide on the JevBench dump (`tjev post` writes it). Not done
yet: the pool is a candidate, not the default.

## Training-side recommendations

What the study asked of training, and where each item stands (the measurements of the
arms are the reg stage's, [sweep-plan.md](sweep-plan.md#stage-4-the-regularisers-the-reg-stage-4b-and-2b)).

Per-item evidence (K = 2): at step 3000 the uncalibrated held-out fr NLL is 0.914 against
0.405 en (step 1200: 0.495 / 0.320), and no calibrator brings fr below 0.68 there. The
h1500 run at 1450 is the better French model (held-out fr 0.414 vs 0.476 at h3000@1200)
and the worse English one (0.366 vs 0.298); whatever differs between those two runs is
worth isolating for French.

1. **Per-language early-stopping signal.** Selection uses one global calibrated NLL (one
   temperature, 76 % English); from ~step 1400 fr accuracy is flat at .84–.85 while fr
   confidence keeps rising, and in the cooldown fr NLL goes up while en goes down.
   Select on the mean of per-language calibrated NLLs (or max), and log
   `eval_calibrated/lang/*`; a run that only gets better in English should stop.
   *Implemented:* `eval_calibrated/lang/<lang>`, `lang_mean`, `lang_max` are logged and
   `train.select_on=lang_mean|lang_max` selects on them ([training.md](training.md#selection)).
2. **fr data balance in the cooldown.** The fr over-confidence grows in the WSD cooldown
   (quick-eval fr NLL +.10 vs en −.017). French is ~25 % of the eval mix; the French-only
   sources are few and easy (hc3_fr, textdetox_fr, allocine_fr reach .96–.99 accuracy) and
   get memorised. For the data side: more *hard* French (fr choice: tweet_sentiment_fr is
   the worst-calibrated fr source; translated versions of the hard en sources), cap the
   easy fr noul sources' epochs, and keep the fr share constant through the cooldown.
   *Taken up by the mixes since:* the French machine translation of the hard English
   sources and the per-source epoch caps ([fr-data.md](fr-data.md), [data.md](data.md));
   the `-rep1` sibling mixes cap the core sources at one epoch.
3. **Label smoothing on soft-target-free sources** (ε ≈ 0.05–0.1), at least for the easy
   memorised fr sources and the over-confident generators (gen_return_window,
   gen_invoice_total: over-confident in *and* out of distribution, so a generator /
   template defect or label noise is likely — check their gold labels first).
   *Arms:* `ls0.05`, `ls0.1`, `ls0.1-gen` (`loss.label_smoothing`, `loss.smooth_sources`),
   and CogCalib's gated entropy penalty (`rank16-cogcalib`, `rank16-cc*`) as the targeted
   alternative ([training.md](training.md#calibration-losses)).
4. **Shorter or no cooldown at 4B** (the calibrated step-1200 checkpoint beats calibrated
   step 3000 on held-out NLL, .403 vs .472), or a cooldown that holds the LR floor higher.
   *Arms:* `cool0.1`, `floor0.1`, and the 1-epoch cooldown branch of the final-length
   `rank16` / `rank16-cogcalib` runs.
5. Calibration cannot fix domain shift: after per-type T held-out noul is still +9 points
   over-confident. Held-out *calibrated* NLL by type and language is now a logged selection
   metric (item 1), not only validation.
