# LoRA, calibration and data: literature against our measurements (2026-10-04)

Three research passes (Semantic Scholar, arXiv, HF cards, competitor repos) set against what
our own sweeps measured. Tags: **[READ]** the paper/card was read, **[ABS]** abstract or TLDR
only. Our numbers: `docs/sweep-plan.md` (reg stage) and the W&B project tjev-sweep. Companion
notes: `jev-landscape-2026-10.md` (benchmarks and Jev-alikes, 2026-10-01).

## What our sweeps measured (mix-v5, 1500 steps, frozen gates)

| Lever | Result | Verdict |
|---|---|---|
| Seed noise (2B, two seeds) | score sd 0.025 per eval, ~0.01 on the end score | the yardstick for everything below |
| Rank, 4B (64 / 32 / 16) | end score .468 / .467 / **.436**; mean of the last 5 evals .470 / .472 / **.431**; health warn / warn / **ok** | **r16** (r64 and r32 over-fit late; rank matters in the cooldown) |
| Rank, 2B (32 / 16) | equal within noise | r16 for every size |
| LoRA on the GDN layers (2B, `nogdn` = attention + MLP, no GDN projections) | without it: end .508 vs .470, worse at 12/15 evals | keep LoRA on in_proj_qkv / in_proj_z / out_proj (gates frozen); the paper's attention-only (no MLP either) was not run |
| Weight decay 0.1 / 1 / 10 | 0.1 and 1 are near no-ops (‖B‖ −1…3 %); 10 shrinks A −15 %, B −6 % and hurts the cooldown (+0.04–0.05) | 0.1 at most (free); decoupled decay is lr·wd per step |
| LR floor 10 % (4B) | identical until the cooldown, then no effect | default floor |
| Label smoothing 0.1 (2B; all / generators only) | worse (.504) / slightly worse (.488) | no |
| KL to the base (2B) | neutral, +25 % compute | no |
| Determinism and resume (TPU) | a rerun is bit-identical; SIGTERM + resume is bit-exact (data, losses, adapters) | resume is safe for multi-session finals |
| 4B cooldown drift | concentrated in gen_return_window (conf − acc +0.42) and the long refund policy | a data problem: the calendar shortcut (`fr-data.md` §7) |

## The literature, by question

### Where LoRA goes on a hybrid (Gated DeltaNet) model
- **arXiv 2604.22127, Where Should LoRA Go?** [READ] Qwen3.5-0.8B: attention-only LoRA best, GDN-only destructive, on *generative* GSM8K (n = 128, one seed); its GDN arm also adapted the gates; logit-scored MMLU / ARC / HellaSwag were not harmed. **Our measurement contradicts it for logit readout**: dropping only the GDN projections (keeping attention + MLP) is worse.
- **LoRA Without Regret** (Thinking Machines, 2025) [READ]: adapt all layers, MLP included; with α/r scaling the best LR barely moves with rank; rank only matters once capacity binds, late. Matches our r16 result.
- **PEFT of State Space Models** (ICML 2025, 2410.09016) [READ]: adapt the linear projections, not the recurrence parameters. Supports frozen gates.

### Rank and learning rate
- **S0 Tuning** (2604.01168) [READ]: Qwen3.5-4B, 10 seeds: r24 +12.7 pp, r48 unstable, r64 negative for 8/10 seeds (over-fitting; tiny data). Matches 4B r16 > r32 ≈ r64.
- **μA, LR scaling across LoRA ranks** (2602.06204) [READ]: standard init gives η ∝ r^{−1/4} under rsLoRA; our 4B rate 2.4e-5 vs the predicted ~2.6e-5: the LR is not the cause of the drift.
- **Learning Rate Matters: Vanilla LoRA May Suffice** (2602.04998) [READ]: once the LR is tuned, LoRA variants land within 1–2 %. No CU on LoRA+, DoRA, PiSSA.

### Regularisation
- Decoupled weight decay needs lr·wd·T ≈ 0.2–0.4 to matter (our arithmetic, confirmed: wd 10 at lr 3.5e-5 shrinks A by ~15 % over 500 steps). 2502.09376 [ABS] argues zero-init + decay biases to low rank; at our rates the bias is negligible below wd ~5.
- **LoRA Learns Less and Forgets Less** (2405.09673) [ABS]: LoRA limits forgetting more than decay or dropout. Consistent with decay / KL doing little here.
- **LoRA dropout as a sparsity regulariser** (2404.09610) [ABS]: rank dropout is mergeable (a per-step mask on the r dims); untested here.

### Calibration beyond temperature
- **CogCalib** (2505.20903) [READ]: over-confidence comes from rows the model already knows (low NLL); a calibration term gated on NLL ≤ t gives −57 % ECE. Explains why uniform label smoothing did nothing; the gated version is the next calibration lever.
- **Kev PLAN.md** (github.com/jaredpalmer/kev, a Qwen3.5 LoRA Jev-alike) [READ]: smoothing, focal and Brier gave nothing against matched CE; per-type and per-(type, K) temperatures hurt; a temperature fitted on held-out *datasets* transfers (new-source ECE .103 → .041 at 9B); soft targets help only where a teacher disagrees with the label at p ≥ 0.6.
- **Temperature scaling is not enough under human label distributions** (2607.13423) [ABS]; **CIFAR-10H** (1908.07086) [READ]: soft targets from annotator disagreement beat temperature.
- **Multilingual calibration** (2510.03136 [ABS]; 2210.12265 [READ intro]): non-English is worse calibrated; in-language data, not more MT, fixes it.
- Ordinal (score) heads: SORD, EMD, CORAL/CORN, regression-aware inference [ABS]; Kev's ordinal loss did not help, so prefer soft ordinal targets from rater distributions.
- Our own study (`calibration.md`): per-type T; type × lang fails; a 2-checkpoint ensemble is the only large gain.

### The benchmark's scoring and noul
- **JevBench v1.5.7** [READ METHOD-v1.5 + addendum]: choice / noul / score each 1/3; tiers 10/20/30/40; a noul P(yes) in (0.2, 0.8) is an abstention scored worse than a guess; abstention rate vs noul score r = −0.89 across the board. 4B LoRAs lose on the sealed noul, not on choice or calibration.
- **Plumb v5.2** `--noul-commit` [READ]: noul CC 37.8 → 81.1 on the public set. Our measurement (`official-eval.md`): commit at margin 0 gives +10 Intelligence on validation for −1…−4 Calibration.
- **Beyond Calibration: probability axioms for typed decisions** (2609.33209) [READ]: a first-token noul readout misses P(X) + P(not X) = 1 by 0.29 for Qwen; Jev 0.06. Negation twins with target 1 − p (in mix-v6) and a coherence metric.
- **Evaluating and Benchmarking Jev** (2609.37647) [READ]: Jev's binary probabilities rank well but sit badly against 0.5; arithmetic is kept out of its scope.
- **JevAdvBench** (2609.31142) [READ]: one appended opinion flips 12 % of decisions; distractor-opinion rows are a cheap robustness addition.

### What makes fine-tuning data good
- **Synthetic shortcut learning**: our one-template / whole-month generator taught a calendar shortcut (`fr-data.md` §7). Kev [READ]: LoRA erodes date arithmetic; stating the day count restores it (0.65 → 1.00); more rows from one generator saturate fast (+26 pp then +5 pp); synthetic long documents do not transfer to real ones.
- **Bespoke-Nimble-9B** [READ]: 2,676 contrastive one-fact pairs (with evidence ablation) reach 90.1 %. Design pattern for generators.
- **Quality over quantity**: LIMA, AlpaGasus (9k filtered beat 52k) [READ]; IFD / Cherry, Superfiltering, Deita [READ/ABS]; LESS (gradient similarity, cosine not dot) [READ].
- **Dataset cartography** [READ]: the ambiguous third helps out of distribution; hard-to-learn rows flag label errors. **RHO-LOSS** [READ]: train on learnable, not-yet-learnt rows.
- **Decontamination and dedup** (Lee et al., SemDeDup, D4) [ABS]: mix-v6 removes 1,261 JevBench 8-gram hits and 2,198 train rows sharing a passage with our held-out sets.
- **Mixing and replay**: UniMax (uniform with an epoch cap, 1 epoch best) [READ]; Data Mixing Laws, RegMix [READ]; ~1 % replay for a mild shift, 25 % for a language shift (Ibrahim et al. 2024) [READ].

## Open questions worth a run
1. Does the generator fix (mix-v5b) remove the 4B cooldown drift? Does mix-v6 move the weak JevBench families? (4B r16 on each, paired with v5.)
2. Is the 4B wd 0.1 gain (better at 14/15 evals, −0.026) real? A seed replica.
3. CogCalib-style gated calibration loss on 4B.
4. The temperature fitted on a pool of held-out *sources* (implemented: `post --calibration-pool`), on a pool we never trained on.
