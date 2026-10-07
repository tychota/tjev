# The Jev-class landscape (researched 2026-10-01)

Public benchmarks, the open Jev-alikes, what Jev itself probably is, and what tjev should
change. Gathered by a research pass over the sources linked below; numbers are as published
(author-reported unless measured by benchmarkheaven). Not independently re-run.

## Benchmarks

- **JevBench (benchmarkheaven).** v1.4.1 (2026-09-22) is stale; **v1.5.4 is live**
  ([board](https://benchmarkheaven.com/jev-models),
  [JSON](https://benchmarkheaven.com/api/jevbench/v1.5.4),
  [method](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.5.md)).
  - Our "JevBench public 231" is its public set (easy 48, standard 72, hard 111).
  - Axes: Intelligence (chance-corrected accuracy; v1.5 weighs the sealed set 50%), Calibration
    (typed: choice ECE + TVD, noul ECE, score RPS + ECE), Speed, Cost; composite = harmonic mean
    with a gate below 50.
  - **v1.5 scores a noul P(yes) in [0.2, 0.8] as an abstention (wrong)**; score questions on the
    MAE of the expected level.
- **jabr classifier-benchmark v2** ([repo](https://github.com/jabr/classifier-benchmark)) = the
  "49-task" benchmark: Jev 0.967, GLiNER2.5-Decide 0.739, Von 0.724, Laya 0.58–0.62.
- **typed-decisions** ([HF](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)): zero-shot
  Jev 0.727 vs fitted Laya 0.766; not comparable (teacher-label ceiling 0.735).
- **Decision Index 0.2.1** ([repo](https://github.com/sinanuozdemir/decision-index)): 38 public
  benchmarks, `http` engine for `/v1/systemone`; Jev 57.89.
- AIMLAPI ([blog](https://aimlapi.com/blog/what-is-jev)), LangWatch, DecisionBench (HF space).

**JevBench v1.5.4, top of the board**

| System | Score | Intelligence | Calibration | Sealed noul | Architecture |
|---|---|---|---|---|---|
| Cygnet | 73.7 | 71.1 | 87.0 | 56 | frozen Gemma-4-12B-it, letter logits, T 3.4 |
| Winnow-12B | 73.2 | 74.4 | 84.1 | 66 | Gemma-4-12B LoRA r32 |
| Jev 1.13.0 | 72.1 | 72.0 | 88.0 | 49 | closed |
| JevK5 v0.3 | 71.9 | 56.3 | 88.3 | 15 | Qwen3.5-4B LoRA, letter logits |
| Plumb-4B | 71.6 | 55.8 | 87.4 | 17 | JevK5 + LoRA r16 |
| decider-4b v2 | 71.3 | 55.8 | 85.6 | 19 | Qwen3.5-4B-Base |
| Imajev-4B | 70.4 | 53.5 | 88.1 | 8 | Qwen3.5-4B + 255-way head |
| Laya / Von (encoders) | 0.0 | ≈0 | 74–84 | — | ModernBERT |

4B decoder LoRAs plateau at Intelligence ≈ 55; the gap to Jev and the 12B models is mostly the
sealed noul items. A frozen, well-prompted Qwen3.5-4B scores 0.80–0.81 on the public 231
(Jobe, local-jev, SemIf); prompt wording alone moves it ±8 points (reflex).

## Open Jev-alikes

| Model | Base, adapter | Readout | Data | Loss | Calibration |
|---|---|---|---|---|---|
| [Imajev](https://github.com/mohit67890/imajev) | Qwen3.5 2–9B, LoRA r16 | 255-way linear head | 504k + 475k teacher + 23k hard | soft CE | one T |
| [Plumb](https://github.com/crh225/plumb) | JevK5 + LoRA r16 (attention) | letter logits | 27B teacher, both-agree filter; replay; hard-mining; long documents | letter CE | one T (2.07), score T 1.2 |
| [JevK5](https://github.com/allebee/jevk5) | Qwen3.5-4B LoRA r16 (attention + DeltaNet) | letters A–P | 17.4k teacher + 30k replay, 1 epoch | soft CE | one T (per-type rejected) |
| [decider](https://github.com/Mapika/decider) | Qwen3.5 2B/4B/35B-A3B Base | letters, multi-slot | ~95 public sets + teacher + generators | CE (+ RL on 2B, no JevBench gain) | per-type T, T(n) by option count |
| [Cygnet](https://github.com/blockbrain-ai/cygnet-recipe) | frozen Gemma-4-12B-it | letter logits | none | — | one T 3.4; noul false first |
| [metask-jev](https://github.com/metask-ai/metask-jev) | Qwen3.5-4B LoRA r16, lr 2e-5, 1 epoch | letters | 61k | CE, 3 orderings | per-type T |
| [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) | Qwen3.8-27B LoRA r8 + head | — | 149k | **CE + 0.1·Brier** (= tjev) | — |
| [Laya](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large 421M + [MASK] per option | per-option logits | — | "RLCD": REINFORCE with logit noise + soft CE | per-type T |
| [Von](https://github.com/wfzyx/von) | ModernBERT-large 395M | option markers | ~290k | listwise CE + Brier | T map |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | DeBERTa-v3-large 340M | label tokens | undisclosed | — | — |

## Is Jev "an encoder + scoring head"?

Best estimate: **a pretrained decoder LM (likely ≥ tens of billions of parameters, possibly
MoE) post-trained to read one distribution per question from a shared state prefix**
(~75%); a bespoke long-context bidirectional encoder ~20%; a Laya-class 400M encoder < 5%.

- Official ([launch](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
  [primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md),
  [models](https://docs.typesafe.ai/models.md)): "a new model architecture, parallel sampler …
  and training method we call RLCD"; RLCD presented as a post-training path of *pretrained
  language models*; 64k tokens per request (32k state + longest question); "ingests the state
  once and evaluates every question against it in parallel". No paper, weights or reward.
- For a decoder: 32k context per question (ModernBERT: 8k); shared-prefix semantics; MMLU-Pro
  84.6% ([probes](https://archerhume.com/posts/jevs-architecture-unmasked/)); a Qwen-like
  tokenizer; position effects consistent with causal attention; latency grows with input
  length only; inference job posts mention vLLM.
- The deck's "Gaussian logit noise + REINFORCE + group baseline" recipe is **Laya's**, not
  TypeSafe's.

## What tjev should change (ranked)

1. **Evaluate with the official harness** (`jevbench.cli run` against tjev's `/v1/systemone`)
   on the public 231, per tier, plus the Decision Index kit; benchmark a frozen 4B with a good
   prompt as the control.
2. **Data**: teacher-labelled hard items with a both-agree filter, hard-mining (train on what
   the model gets wrong or is unsure of), long documents (2–16k); replay ~1:1. Plumb: hard
   0.739 → 0.802 from 5k teacher items. (tjev's `plumb` source is Plumb's published data.)
3. **Context**: 8k and 16k buckets (re-check the GDN gate gradients at 8k first).
4. **Noul**: fixed option order or two-order averaging; a deliberate serving policy against
   v1.5's [0.2, 0.8] abstention band (keep the calibrated probability in the API).
5. **Order averaging at eval** (reflex: hard +2.7, ECE halved elsewhere).
6. **Calibration**: compare per-type and global T on a domain-shifted calibration set.
7. **Bigger backbone**: 9B, or a 12B-class model (4B LoRAs plateau at I ≈ 55).
8. **More than 26 options** (grouped / knockout reads or a 255-way head).
9. **Shared-state prefill at serving** (verify probability equivalence).
10. **Keep the direct proper-scoring objective; no RLCD/REINFORCE** (exact gradients beat a
    noisy estimate of the same reward; published RLCD attempts did not help).
11. **LoRA rank 16** (the leaders' choice; tjev's sweep found 16 = 32 = 64), or a KL penalty to
    the base on replay rows.
12. **No encoder baseline**, except as a latency tier.

Status since (October 2026): 1 is `tjev eval-official` and the frozen-prompt control
([official-eval.md](../official-eval.md)); 2 is in mix-v7 / mix-v7.5 (teacher-annotated and
targeted sources, long documents; [data.md](../data.md)); 3 is the long buckets
([training.md](../training.md)); 4 and 5 are `tjev post --orders 2` and the serving
abstention policies ([serve.md](../serve.md)); 6 is the calibration study
([calibration.md](../calibration.md)); 11 is the `r16` arms and `loss.kl_base_weight`. 7, 8, 9
and 12 are still open.

Could not verify: KDnuggets (403); the deck's "33 ms vs 236–276 ms" figures; Tev1's price; the
601 published v1.5 items; some HF pages (401 / unreadable).
