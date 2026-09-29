# Data

## The decision item

Every source, generator and evaluation set is JSONL in one JevBench-native schema
(`tjev.data.item`):

```json
{"id": "…", "state": "free text, or a JSON object / list",
 "question": {"type": "noul | choice | score", "instructions": "…",
              "criteria": {"label": "description", "…": "…"}},
 "labels": ["…"],
 "expected": "label",
 "source": "…", "family": "…", "lang": "en | fr"}
```

- **noul** (yes/no): labels are `no` and `yes`, with criteria `{"false": …, "true": …}`.
- **choice**: labels are *categories defined by criteria*, not answer strings to copy.
- **score**: labels are ordinal levels `0 … n`, in order. The criteria may be a list of
  level descriptors.
- **Targets**: `expected` gives a one-hot target. `target: {label: p}` gives a soft one:
  base rates, interpolated similarity, annotator distributions.
- **Structured states** (the JevBench hard tier) are rendered as indented JSON text
  (`item.rendered_state`). Dedup and the contamination checks see that same text.

Parsing is fail-closed. Unknown types, a label missing from the criteria, probabilities that
do not sum to 1, empty states and duplicate labels are all errors.

## Rendering

`tjev.data.render` turns an item into one prompt with one answer slot:

- **Options.** Options become letters A–Z, so an item has at most 26 options (the builder
  drops items with more). Each option shows `label: description`.
- **Order.** Choice and noul options are shuffled in training. At eval the order is a
  fixed hash of the item id. Score levels keep their order.
- **Distractor subsampling.** In training, with probability 0.3, some distractors are
  dropped, keeping at least 4 options. A label with target mass is never dropped.
- **Layouts.** In training, half the prompts use the plain layout (`State: … Question
  (type): … Options: A. …`) and half use a JSON layout. Eval uses the plain one.
- **Framing.** The chat framing is written out explicitly: the Qwen3.5 template with
  thinking disabled, checked against the HF template by `tests/data/test_real_template.py`.
  `TEMPLATE_VERSION` is part of every run and calibration identity: bump it whenever the
  rendered text changes.

## mix-v3

`tjev data build OUT` builds the mix into `OUT/`:

| File | Content |
|---|---|
| `train/<source>.jsonl` | Training rows, capped per source |
| `validation.jsonl` | Per-source held-out rows, plus held-out generator rows |
| `calibration.jsonl` | Disjoint held-out rows for temperature fitting |
| `test/<source>.jsonl` | Per-source test rows, never used for selection |
| `mix.yaml` | The `data` section of a run config: train files, validation set, mixture weights |
| `manifest.json` | Counts, licenses, languages, drops and hashes per source |

### Blocks

The mixture is set by block shares (`tjev.data.mix.BLOCKS`). Within a block, sources are
weighted ∝ n^0.5.

| Block | Share | Content |
|---|---|---|
| `jev` | 0.40 | Contenders' public JevBench-style training data |
| `generators` | 0.25 | Code-labelled EN/FR decision generators (a rule engine or arithmetic decides; only the surface text varies) |
| `replay` | 0.20 | Public datasets recast as rubric decisions |
| `analysis` | 0.10 | Text analysis: authorship (human / AI, model family), formality, sentiment, emotion, offensiveness |
| `writing` | 0.05 | Templated writing checks (spelling, register), kept small because low diversity overfits |

- **Priority.** The target is JevBench-style decisions, so text classification
  (`analysis` + `writing`) stays at 15% of the sampled mix. New text-analysis sources
  split that share rather than grow it.
- **French.** 20% of the sampled mix is French. The `jev` block is English only and the
  generators draw 30% French, so French-only sources carry fixed shares of their blocks
  (`FR_TARGET`: replay 0.38, analysis 0.35). The block shares never change, and
  `tjev data reweight OUT` recomputes the weights of a built mix.

As built with the default seed and scale:

- **Size.** 47 sources and 456,718 training rows: jev 101k, replay 218k, analysis 96k,
  generators 34k, writing 8k.
- **Languages.** 379k English and 77k French rows; 20.4% of the *sampled* mix is French.
- **Held out.** 11,234 validation and 11,188 calibration items.
- **Contamination.** 47 training rows were dropped for JevBench overlap.

### Sources

The registries live in `tjev.data.sources` (one module per block), and every entry records
its license. **Commercial** = cleared for commercial use by its card; `--commercial-only`
builds without the others. Licenses are copied from the dataset cards and must be re-checked
before any commercial use.

| Block | Source | Dataset | Lang | Family | License | Commercial | Cap |
|---|---|---|---|---|---|---|---|
| jev | certo | altslate/certo-decisions-v2 | en | certo | MIT | yes | 15000 |
| jev | decider_teacher | github.com/Mapika/decider teacher_data | en | decider | Apache-2.0 | yes | 30000 |
| jev | mghafiri_scenarios | mghafiri/decision-model-scenarios-v2 | en | mghafiri | MIT (Claude-written) | yes | 20000 |
| jev | openjev | ZefanCai/Open-Jev-v1.1 | en | openjev | CC0-1.0 (generated) + CC BY 4.0 (WANLI) | yes | 40000 |
| jev | plumb | crh225/plumb-decisions | en | plumb | Apache-2.0 | yes | 6000 |
| generators | gen_base_rate, gen_invoice_total, gen_long_refund_policy, gen_refund_policy, gen_return_window, gen_schedule_conflict, gen_table_argmax, gen_ticket_triage, gen_weekday | generated | en + fr | probability, math, long_policy, policy, temporal, tables, routing | Apache-2.0 | yes | 4000 each (long policy 2000) |
| replay | allocine_fr | tblard/allocine | fr | sentiment | MIT | yes | 8000 |
| replay | banking77 | legacy-datasets/banking77 | en | routing | CC BY 4.0 | yes | 20000 |
| replay | boolq | google/boolq | en | reading | CC BY-SA 3.0 | yes | 20000 |
| replay | clinc_oos | clinc/clinc_oos | en | routing | CC BY 3.0 | yes | 20000 |
| replay | dbpedia | fancyzhx/dbpedia_14 | en | classification | CC BY-SA 3.0 | yes | 8000 |
| replay | gsm8k_check | openai/gsm8k | en | math | MIT | yes | 20000 |
| replay | helpsteer2_correctness, helpsteer2_helpfulness | nvidia/HelpSteer2 | en | judge | CC BY 4.0 | yes | 8000 |
| replay | jailbreak | jackhhao/jailbreak-classification | en | safety | Apache-2.0 | yes | 20000 |
| replay | massive_en, massive_fr | mteb/amazon_massive_intent | en / fr | routing | CC BY 4.0 | yes | 20000 |
| replay | mnli | nyu-mll/glue | en | entailment | GLUE (see card) | yes | 20000 |
| replay | paws, pawsx_fr | google-research-datasets/paws(-x) | en / fr | paraphrase | free use (card) | yes | 10000 |
| replay | prompt_injection | deepset/prompt-injections | en | safety | Apache-2.0 | yes | 20000 |
| replay | snli | stanfordnlp/snli | en | entailment | CC BY-SA 4.0 | yes | 10000 |
| replay | squad2 | rajpurkar/squad_v2 | en | abstain | CC BY-SA 4.0 | yes | 20000 |
| replay | sst5 | SetFit/sst5 | en | sentiment | see card | yes | 20000 |
| replay | stsb | sentence-transformers/stsb | en | similarity | CC BY-SA (card) | yes | 20000 |
| replay | vitaminc | tals/vitaminc | en | fact_check | CC BY-SA 3.0 | yes | 20000 |
| replay | xnli_fr | facebook/xnli | fr | entailment | CC BY-NC 4.0 | **no** | 15000 |
| analysis | go_emotions | google-research-datasets/go_emotions | en | emotion | Apache-2.0 | yes | 20000 |
| analysis | hc3_fr | almanach/hc3_french_ood | fr | authorship | CC BY-SA 4.0 | yes | 10000 |
| analysis | mage | yaful/MAGE | en | authorship | Apache-2.0 | yes | 20000 |
| analysis | mlma_en, mlma_fr | nedjmaou/MLMA_hate_speech | en / fr | offensive | MIT (card) | yes | 20000 |
| analysis | pavlick_formality | osyvokon/pavlick-formality-scores | en | style | CC BY 3.0 | yes | 9300 |
| analysis | raid | liamdugan/raid | en | authorship | MIT | yes | 20000 |
| analysis | textdetox_fr | textdetox/multilingual_toxicity_dataset | fr | offensive | openrail++ (use restrictions) | **no** | 20000 |
| analysis | tweet_sentiment_en, tweet_sentiment_fr | cardiffnlp/tweet_sentiment_multilingual | en / fr | sentiment | CC BY 3.0 + Twitter ToS | yes | 20000 |
| writing | gen_register, gen_spelling_error | generated | en + fr | writing | Apache-2.0 | yes | 4000 |

Deliberately excluded:

- **`SargeDev/jev-distill-corpus-v3`.** Its labels are distilled from Jev, whose license
  forbids distillation.
- **Data needing approvals we do not have, or forbidding redistribution of derived data:**
  GYAFC / XFORMAL (Yahoo Webscope) and PAN.

How the adapters recast public data:

- **Intent datasets become variable menus.** The gold label plus 3–11 sampled distractors,
  sometimes an "other" option, and sometimes *only* "other" with the true intent left off
  the menu. The model thus learns to abstain.
- **Several annotators give a soft target.** Where several annotators rated a text, the
  target is their distribution, never its argmax.

### Hygiene

- **Held-out splits.** Held-out public rows come from each dataset's evaluation split, or
  from a hashed group split of train (rows of one group never straddle the two). They are
  hash-partitioned into validation / calibration / test and deduplicated across the three
  parts.
- **What the builder drops from training:** rows whose normalised (state, instructions)
  equal a held-out row; 5-gram near-duplicates of held-out states (small sources); rows
  whose (state, instructions, criteria) occur with different gold labels; items with more
  than 26 options.
- **Held-out generators.** Generators use other seeds for their held-out parts, and their
  own held-out surface text: framings, questions, template sentences, vocabularies, names,
  and some parameter ranges. Evaluation then measures transfer, not memorised templates.
  The tests check that no key or template line is shared.
- **Contamination.** Every training row of every block is checked against the 231 public
  JevBench items (`tjev.data.jevbench.JevBenchFilter`). A row is dropped when it has an
  equal NFKC-normalised state, an equal non-generic instruction, or more than two shared
  word 8-grams with a public state. An instruction is generic when it is under 60
  characters or shared by ≥ 20 training rows.

## Evaluation sets

- **JevBench public** (`tjev data jevbench REPO OUT`): 231 items (easy 48, original 72,
  hard 111), for final measurement only. They are never part of a mixture, a validation
  set or a calibration fit. Sealed items are never used, and no generator is written from
  the sealed family names.
- **Held-out generators** (`tjev data heldout`). Two sets with different seeds: the
  *selection* set (seed 777, `data.heldout`, part of the checkpoint selection score) and
  the *reporting* set (seed 4242, evaluated by `tjev post`).
- **The mix's validation and calibration splits.**

## Training stream

`tjev.data.pipeline` is the training stream as a Grain pipeline:

```
MapDataset.range(next_index, ∞).map(SegmentAt.pair)      random access, 1:1
  .to_iter_dataset() [.mp_prefetch(workers)]              order independent of the workers
→ BucketStepIterDataset                                   bucket routing, multi-bin packing,
                                                          one-bucket steps [A, R, T]
→ ThreadPrefetchIterDataset                               host work overlaps the device
```

- **Deterministic segments.** Segment i draws its source, item, option order,
  subsampling and layout from `default_rng([seed, i])`.
- **Portable state.** The state is our own JSON, `{next_index, open, ready}`, independent
  of the Grain version, the worker count and the prefetch depth.
- **Pinned output.** `tests/data/test_pipeline.py` pins the stream's output, checks it
  against the sequential reference stream (`tjev.data.mixture.TrainStream`), and checks
  exact resume across worker counts.
- **Workers are optional.** Host work is under 1% of a step at 2B. Worker processes keep
  JAX on the CPU (`tjev/__init__.py`), so they never take accelerator memory.

## Known issues

- **Score questions are the weakest type everywhere** (accuracy ~0.55–0.59, ECE ~0.12). The
  targets are ordinal one-hot, which is noisy for human ratings, and the levels are read
  through letters.
- **Long documents are rare.** The "long" policy documents are ~3.8k tokens at most, so the
  4096 bucket carries few rows.
- **Held-out ranges test extrapolation.** Some held-out generator splits test range
  extrapolation (e.g. weekday offsets beyond the training range), and small models score
  low there by design.
- **Planned: translated contender items.** About 15% of the `jev` block is to be translated
  to French, with automatic number / entity / option checks. Temperatures would then be
  calibrated per (type × language). Not implemented.
