---
license: cc-by-nc-sa-4.0
base_model:
  - Qwen/Qwen3.5-0.8B
  - Qwen/Qwen3.5-2B
  - Qwen/Qwen3.5-4B
  - Qwen/Qwen3.5-9B
library_name: peft
language:
  - fr
  - en
tags:
  - lora
  - calibration
  - decision
  - jevbench
  - tpu
---

# tjev: calibrated decision adapters for Qwen3.5

This repository holds the final LoRA adapters of **tjev**, a fine-tune of the Qwen3.5 base
models for JevBench-style decisions: a question with context, answered yes / no / unclear
with a calibrated probability. One set of adapters per base size (0.8B, 2B, 4B, 9B), each
with its calibration (per-type temperatures) bound to the checkpoint it was fitted on.

The same text lives in the code repository as `docs/model-card.md`; the training code,
kernels, data pipeline and every measurement are documented there (`docs/`).

## Repository layout

```
<campaign>/<run>/                  e.g. final-v75/final-4b-h5000
  config.json, selection.json      the run's configuration and its selected step
  metrics.jsonl, eval/             the training curve and every full evaluation
  checkpoints/<step>/lora/         LoRA adapters (bf16, ~13 MB at rank 16 for 4B)
  checkpoints/<step>/ema/          their EMA (decay 0.999), same format
  checkpoints/<step>/opt/          optimizer state, the selected and the last step only
  post-calib-<step>[-ema]/         calibration, JevBench per-item dumps and predictions of that step
<campaign>/<run>-b<H/2>/           the run's 1-epoch cooldown branch, same layout
```

`selection.json` names the step the training selected on its held-out score. The
`post-calib-*` folders hold the post-training of a step: the calibration (`calibration-jax.json`),
the JevBench v1.5 scores and the per-item predictions.

## Recipe

| | |
|---|---|
| Base | Qwen3.5 0.8B / 2B / 4B / 9B (hybrid Gated DeltaNet + attention), frozen, bf16 |
| Adapters | LoRA rank 16, α 32, rsLoRA scaling, on every projection except the GDN gate projections (`in_proj_a`, `in_proj_b`, frozen); B = 0 at init |
| Optimizer | AdamW (β₁ 0.9, β₂ 0.999, no weight decay), global-norm clipping; the learning rate follows the rank rule (the sweep's rank-32 rate × √2): 4e-5 at 4B for a 5,000-step horizon |
| Schedule | 50 warmup steps, stable, cooldown over the last 20% of the steps (WSD); the long run to the horizon H and a cooldown branch to H/2 off the same stable phase (the "1-epoch" model) |
| Batch | 65,536 tokens per step, best-fit packing into 1,024 / 2,048 / 4,096-token rows |
| Loss | per-slot NLL on the answer tokens, with CogCalib's gated confidence penalty (weight 0.1 on the slots in the lowest 30% of NLL) |
| EMA | an EMA of the adapters (decay 0.999) tracked and evaluated alongside the raw adapters; selection on the raw adapters |
| Selection | the held-out score (calibrated validation and held-out NLL) at every full evaluation (every 200 steps, every 50 in the cooldown) |
| Hardware | one TPU v6e chip per run (Colab), Pallas GDN megakernel, splash attention, merged LoRA at ≤ 4B |

## Training data

**mix-v7.5** (the code repository's `docs/data.md`): 672,075 unique training rows, 22.4% French
by sampled share, built from public decision, NLI, stance and QA datasets, a set of generated
contrastive / distractor / posterior tasks, and French machine translations of English
sources (Helsinki opus-mt, kept only above a CometKiwi quality threshold). The core sources
are seen about twice over a 5,000-step run (once in the cooldown-branch model).

Licence use of the training rows, by sampled share:

| Mix | commercial | non-commercial | unspecified |
|---|---|---|---|
| mix-v7.5 (these adapters) | 92.7% | 2.3% | 5.0% |
| mix-v7.5c (commercial-only variant) | 94.3% | — | 5.7% |

The non-commercial rows are the French stance and NLI sets (cxm_fr, x_stance_fr, xnli_fr and
its translation) and a part of tasksource. Licences were read from the dataset cards and must
be re-checked before any commercial use. Every mix is decontaminated against JevBench public
(exact and near-duplicate matching) before training.

## Intended use

Yes / no / unclear decisions on a question with its context, in French or English, with a
probability meant to be taken at face value: the adapters are selected and calibrated so that
a 0.8 answer is right about 80% of the time on the held-out sets. Serving goes through the
code repository's decision API, which applies the checkpoint's per-type temperatures and the
"noul-commit" policy (a 0.1 margin between the yes / no probabilities below which the model
says unclear).

Not intended for: open-ended generation, decisions outside the question-with-context format,
or any use where a wrong confident answer is costly without a human in the loop.

## Evaluation

JevBench v1.5 (public, 231 items; per-type temperature, noul-commit at margin 0.1, two
answer orders averaged), Intelligence / Calibration:

| Model | Zero-shot base | Final adapters (long run) | Final adapters (1-epoch branch) |
|---|---|---|---|
| 0.8B | 0.532 acc / ECE 0.100 | _pending_ | _pending_ |
| 2B | 0.584 acc / ECE 0.129 | _pending_ | _pending_ |
| 4B | 0.779 acc / ECE 0.062 | _pending_ | _pending_ |
| 9B | _pending_ | _pending_ | _pending_ |

The reference points from the development runs (rank 16, mix-v7.5, 4B): 75.3 / 85.8 for the
1-epoch cooldown branch and 70.2 / 84.6 for the 5,000-step run. JevBench public is small:
neighbouring checkpoints of one run swing by 64–72 Intelligence, so the held-out selection
score is the measure the training selects on, and JevBench the report.

## Limitations

- **Small public benchmark.** The Intelligence score moves by several points between
  neighbouring checkpoints; differences under about 5 points are not resolved.
- **Calibration is bound to a checkpoint and a domain.** The temperatures were fitted on the
  training mix's held-out sets; other question styles need their own calibration.
- **Language.** About a fifth of the training rows are French, part of them machine
  translated; French calibration differs from English and is reported separately.
- **Base model behaviour** (knowledge cutoff, biases) is Qwen3.5's; the adapters change the
  decision head, not the knowledge.
- **The second epoch over-fits slightly.** The 5,000-step model needs a higher calibration
  temperature (1.41 against 1.21) than the 1-epoch branch.

## Licence

The adapters in this repository are released under **CC BY-NC-SA 4.0**
(Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International, `LICENSE`):
their training data includes non-commercial sources. The base models are Qwen3.5 by Alibaba
Cloud, **Apache-2.0**; the adapters do not contain base weights. A variant trained on
mix-v7.5c, the commercial-only mix, could be released under a more permissive licence once
trained.

## Citation

```
tjev: calibrated decision adapters for Qwen3.5 (2026). LoRA adapters trained on TPU v6e,
with the mix-v7.5 data; see the code repository's docs/ for the recipe and measurements.
```
