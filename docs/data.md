# Data

How tjev's training data is built: the item format, the rendering, the current mix
(mix-v7.5), its sources and licences, the generators, the hygiene and decontamination
passes, the evaluation sets and the training stream. The mixes before it (mix-v3 to mix-v7)
are kept at the end, each with what it changed, why, and how it measured against the one
before. The French data work is in [fr-data.md](fr-data.md), the text-analysis block in
[research/text_analysis.md](research/text_analysis.md).

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

## Building a mix

`tjev data build OUT` builds a mix into `OUT/`:

| File | Content |
|---|---|
| `train/<source>.jsonl` | Training rows, capped per source |
| `validation.jsonl` | Per-source held-out rows, plus held-out generator rows |
| `calibration.jsonl` | Disjoint held-out rows for temperature fitting |
| `test/<source>.jsonl` | Per-source test rows, never used for selection |
| `mix.yaml` | The `data` section of a run config: train files, validation set, mixture weights |
| `manifest.json` | Counts, licenses, languages, drops and hashes per source; from mix-v4b the epoch cap (`epoch_cap`), from mix-v7.5 the tiers and the subsample |
| `validation-v6.jsonl`, `validation-v7.jsonl` (and `calibration-*`) | Per-source held-out rows of the sources added in mix-v6 / mix-v7 and later, evaluated separately (`tjev post --extra-validation`) |
| `README.md` | From mix-v7: every source with its dataset, rows, share, epochs, licence and licence use |

The commands (`tjev data`, `src/tjev/cli/data.py`):

| Command | What it does |
|---|---|
| `build OUT` | Builds a mix from the registered sources: adapters, held-out splits, the hygiene post-pass, the mixture weights. Options: `--gen-scale N` (rows per generator × N), `--rebalance-noul P`, `--max-epochs E`, `--preserve-blocks`, `--max-prompt-tokens`, `--commercial-only` / `--open-only`, `--blocks`, `--sources` |
| `derive BASE OUT` | A mix from a built one without rebuilding the sources: the yes/no rebalance, the epoch cap, `--drop-license-use`, the mix-v7.5 tiers (`--tiers --tier-a-cap --min-epochs --fresh-generators`). Held-out files are copied byte for byte |
| `extend BASE OUT` | Adds sources to a built mix (mix-v7): the new sources go through `hygiene.clean_v7`, their held-out rows to `validation-v7.jsonl` / `calibration-v7.jsonl`; `--import-from GENMIX --replace-blocks …` imports a separately built mix's blocks |
| `extend-v75 BASE OUT` | The mix-v7.5 intermediate build: cleaning, the rebuilt preference sources, the v7 generators, `--fr-mt` (the jev_fr_mt translations) |
| `reweight OUT` | Recomputes `mix.yaml` from the manifest's counts and scales |
| `heldout OUT/heldout-select.jsonl` | The held-out generator sets (selection seed 777; `tjev post` evaluates the reporting seed 4242) |
| `jevbench REPO OUT` | The 231 public JevBench items, for measurement only |

Every mix is audited with `scripts/data_report.py` ([Data quality reports](#data-quality-reports)).
Within a block, sources are weighted ∝ n^0.5 unless a per-group share is set
(`mix.SOURCE_SHARES_V7` / `source_shares_v75`); `WEIGHT_SCALE` scales single sources and
`SOURCE_MAX_EPOCHS` caps them. The French-only sources carry fixed shares of their blocks
(`FR_TARGET`, `FR_TARGET_V7`) so that the mix stays near 20% French.

## The current mix: mix-v7.5

mix-v7.5 is the mix of the finals: mix-v7 cleaned, its noisy preference sources rebuilt,
the v7 generators and jev_fr_mt added, under epoch tiers for 5,000 steps × 65,536 tokens
with `train.packing=best_fit`. Two variants come from the same intermediate build:
mix-v7.5-rep1 (the repetition arm) and mix-v7.5c (commercial-only).

```
tjev data extend-v75 ~/tjev-work/data/mix-v7 ~/tjev-work/data/mix-v7.5-pre \
    --fr-mt ~/tjev-work/data/jev-fr-mt --jevbench-public …/jevbench/public.jsonl \
    --steps 5000 --packing best_fit --also-steps 2500,3000
tjev data derive mix-v7.5-pre mix-v7.5      --tiers --min-epochs 1 --max-epochs 2 …
tjev data derive mix-v7.5-pre mix-v7.5-rep1 --tiers --tier-a-cap 1 --min-epochs 1 \
    --fresh-generators --jevbench-public …/public.jsonl …
tjev data derive mix-v7.5-pre mix-v7.5c     --drop-license-use non-commercial --tiers --min-epochs 1 …
```

`scripts/build_mix_v75.sh` runs every step on a Colab CPU runtime (about 2 h). The
translations come first, from a GPU (`scripts/fr_mt_gpu.sh`, then `scripts/fr_mt_mix.py`).

- **Where it is.** The private `tychot/tjev-data` repo: `data/mix-v7.5`,
  `data/mix-v7.5-rep1`, `data/mix-v7.5c`. The translation run and the jev_fr_mt mix folder
  are in `data/jev-fr-mt-run` and `data/jev-fr-mt`. Each mix has a `README.md` (every
  source: block, dataset, rows, share, epochs, licence). In the upload, every `mix.yaml`
  came after every file, and `data/mix-v7.5/mix.yaml` was the last.
- **Held-out files.** `validation.jsonl`, `calibration.jsonl`, `heldout-select.jsonl`,
  `validation-v6.jsonl`, `calibration-v6.jsonl` and the mix-v6 test files are mix-v7's,
  byte for byte (so mix-v6's), in all three mixes. `mix.yaml` points at `validation.jsonl`:
  checkpoint selection is comparable with v6 and v7.
- **Per-source held-out sets.** `validation-v7.jsonl` / `calibration-v7.jsonl` hold
  mix-v7's per-source held-out rows plus those of the new sources: the v7 generators
  (their held-out template families), jev_distractor (built from held-out base rows) and
  jev_fr_mt (translations of held-out rows of plumb / decider / mghafiri). Text fixes are
  applied there too (see below), and the rows of the rebuilt preference sources are their
  new held-out rows (same group split). `tjev post RUN --extra-validation
  validation-v7.jsonl --extra-validation calibration-v7.jsonl` evaluates them (below).

### What changed against mix-v7

1. **The v7 generators** (`gen_v7`; see "Generators v7").
   - **What was added.** The 14 v7 generator sources and `jev_distractor` sit in the
     generators block, sized by the plan in
     [Generators v7](#generators-v7). Each is a group of its own in
     `mix.source_shares_v75`, so its weight is ∝ its planned rows. The nine v6 generators
     that are not writing checks share 15.9% of the block.
   - **Post-pass.** They go through the v7 post-pass. A pair source drops the other
     member of a broken pair (`drop_orphans`), and the yes/no rebalance skips pair and
     generated sources (`V75_REBALANCE_EXEMPT`).
   - **ShARC.** gen7_rt_sharc_pairs rows with a line found in a held-out row of the ShARC
     adapter are dropped: 54 rows, leaving 274.
   - **Rows kept after the post-pass.** return_window 13,196 · weekday 7,200 ·
     schedule 8,400 · invoice 8,400 · refund 9,600 · long_refund 3,600 ·
     unknowable 7,199 · aux 3,789 · posterior 12,000 · distractor_opinion 6,000 ·
     rt_cold 7,932 · rt_kali 5,608 · rt_tau 3,000 · rt_sharc 274 · jev_distractor 3,186.
     jev_distractor is built from train rows only; its held-out rows come from held-out
     base rows. Its post-pass skips the near-train checks (a real row plus one line), and
     409 rows were dropped for sharing a passage with held-out rows.
2. **jev_fr_mt.** 27,984 rows after the post-pass (see below). Its group is 2.4 (% of the
   mix at v7's jev scale), which is 2.9% of the mix under the 34% jev block. That puts
   French at 22.3% before the tiers, so the share was not raised to 3.5.
3. **Cleaning** (`tjev.data.clean_v75`). It applies to the base train rows and to the
   mix-v7 per-source held-out rows, never to the mix-v6 files.
   - **HTML entities.** Unescaped in the prose sources: scruples 3,535 rows, drop_num
     2,031, forecasting 291, and webgpt / finqa / tatqa / mic …. Sources with code keep
     `&lt;`.
   - **Python reprs.** An option written as a repr becomes text (`level: definition`).
     A dict or list repr in a state or instruction becomes JSON. Non-JSON literals (`...`)
     are left as they are.
   - **Control characters** are removed.
   - **Language tags.** The judge sources are re-tagged by `tjev.data.lang.guess_lang` on
     the state. helpsteer3_x: 41 fr→en, 1 en→fr. In the rebuilt arena sources the adapter
     renders the item in the detected language:
     - lmarena: 1,634 French rows (was 1,414);
     - compar:IA arena / votes / reactions: 137 / 150 / 380 English rows.
   - **Dropped rows (train only).** Rows with U+FFFD and real mojibake, and dices' 197
     one-hot rows (all `no`; dices keeps its soft rows).
   - **tempreason_l23.** At most 2 rows per near-duplicate state cluster (MinHash, J ≥ 0.8,
     preferring distinct questions): 12,803 → 4,424.
4. **Noisy preferences.** lmarena_140k, comparia_arena / votes / reactions,
   community_alignment_fr and summarize_fb are rebuilt.
   - **Aggregation.** Votes on one item are aggregated: the same prompt and the same two
     replies, either side. compar:IA reactions are aggregated per (message, reply), and
     Community Alignment per (context, prompt, responses).
   - **Smoothed targets.** Every vote-derived target is smoothed toward uniform:
     p = (votes + α/k) / (n + α), with α = 0.5. For a single vote this gives:
     - a decisive vote (4 options): 0.75;
     - a tie / both-bad / equal vote: 0.55 (α = 1.5);
     - a thumbs up or down: 0.857 (α = 0.4, outside JevBench's 0.2–0.8 noul band).
   - **Before / after.** Rows went from 40,609 / 38,961 / 9,694 / 11,748 / 19,809 /
     19,949 in mix-v7 to 40,282 / 38,959 / 9,655 / 11,775 / 19,811 / 19,948 here. Only
     12 / 31 / 0 / 24 / 396 / 909 rows aggregate ≥ 2 votes: duplicates are rare, so the
     smoothing does the work.
   - **Single-vote rows.** 11,028 lmarena, 12,205 arena and 2,773 votes rows are a single
     weak vote (tie / both bad / equal), now low-confidence soft targets.
   - **Shares.** The judge block went from 9% to 7%. Within it, the single-vote groups
     went from 47% to 38% of the group shares (lmarena 1.5 → 0.9, compar:IA 2.0 → 1.5,
     community alignment 0.9 → 0.8; helpsteer3 / oasst2 / norms up). They are tier B, so
     each row is seen at most once.
5. **Per-source held-out sets** and `tjev post --extra-validation` (below).
6. **Blocks.** jev went from 28% to 34% and breadth from 9% to 5%, the judge block from 9%
   to 7%. The other blocks are unchanged: targeted 22%, generators 14%, replay 10%,
   analysis 5%, writing 3%. This follows the JevBench v1.5 scores at 1,500 steps: v7 was −5.3
   intelligence against v6, with +7.2 calibration. Epochs follow the
   tiers below.

### Epoch tiers

The horizon is 5,000 steps × 65,536 tokens, best_fit (the finals and the test run). Every
source is seen at least once and at most twice:

- **Tier A, core (≤ 2 epochs).**
  - Every jev, generators and writing source. jev_distractor is the exception: it stays at
    1 epoch, because its rows repeat real states.
  - eikos / eikos_x / eikos_long, the twin_* sources, helpsteer3 / helpsteer3_x.
  - French law and administration: cold_law_fr / cold_law_fr_x, cetat_fr, argimi_fr /
    argimi_fr_x.
  - forecasting, procedural (not procedural_x) and cuad.
  - The small French human sets: xnli_fr, pawsx_fr, piaf_fr, tweet_sentiment_fr, mlma_fr,
    textdetox_fr.
- **Tier B (≤ 1 epoch), everything else.** The noisy preference sources are in this tier.

How the derive applies them (`tjev data derive --tiers --min-epochs 1`):

1. **Cap.** `balance.cap_epochs` caps the weights on the full row pools, so a tier-B source
   can still absorb the mass a capped source frees. The tier caps replace mix-v7's
   per-source caps (e.g. the twins and mlma_fr were at 1).
2. **Subsample.** Last, every source under 0.9 epochs is cut to its expected draws:
   seeded, stratified by type and gold label, pairs kept whole. Each kept row is then
   seen about once.
   - The weights do not move: `weight_rows` keeps the pre-subsample row count for the
     n^0.5 rule.
   - The epochs are measured again; the subsample never feeds back into the cap.
   - The manifest records the dropped rows per source (`min_epoch_subsample`,
     `drops.min_epoch_subsample`).

**mix-v7.5** trains on 672,075 unique rows. 96 sources were subsampled (1,047,209 rows
dropped), and the largest cuts are:

| source | rows before → after |
|---|---|
| tasksource_jev | 291,826 → 36,770 |
| procedural_x | 56,909 → 19,008 |
| lmarena_140k | 40,282 → 4,552 |
| comparia_arena | 38,959 → 3,779 |
| openjev_x | 55,094 → 30,247 |
| openjev | 35,399 → 23,894 |
| jev_fr_mt | 27,984 → 21,380 |

- **Epochs at 5,000 steps.** Tier B is at 0.97–1.00. Tier A is at 0.91–1.97: decider
  1.86, certo 1.73, certo_x 1.67, mghafiri 1.63, jev_fr_pilot 1.93, mlagnajit 1.97, the
  writing generators 1.37, the rest ~1.0.
- **At 2,500 steps,** the core sources' first epoch: decider 0.93, certo 0.86, the rest
  ≤ 0.5.

**mix-v7.5-rep1** is the repetition arm, with the same block shares and held-out files
(`--tier-a-cap 1 --fresh-generators`). Tier A is capped at 1 epoch.

- **Fresh generator rows.** A generator drawn more often than it has rows gets new rows:
  the rows its seed draws after the ones it was built from, through the mix-v6 post-pass
  (held-out equal / near / passages, public JevBench 8-grams, length, exact duplicates).
  gen_register got +3,532, gen_spelling_error +3,498, gen7_rt_sharc_pairs +3 (seed-limited).
  The v7 generators were already at ~1 epoch.
- **Where the freed mass goes.** The jev block absorbs the mass of the real tier-A sources
  at 1 epoch (openjev_x 44,571 rows, openjev 35,399, jev_fr_mt 27,984 kept whole), and
  no block overflows.
- **Epochs.** Every source is at 0.91–1.005, on 710,367 unique rows.
- **Planner merge.** Its `mix.yaml` is self-contained: its own train files, mixture and
  validation copy. The planner merges it over mix-v7.5's (`@rep1`).

**mix-v7.5-np** is the natural-prior arm: mix-v7.5 without the yes/no rebalance (same
sources, block shares, tiers, caps, min-epoch subsample and seed; held-out files
byte-identical to mix-v7.5's). The benchmark cohorts are skewed (e.g. 61 yes to 7 no), so
a balanced train prior may hurt.

- **Where the rebalance was.** mix-v7.5 rebalanced nothing itself; it inherits it from
  mix-v6 (`--rebalance-noul 0.7`: mage, mlma_en, mlma_fr, openjev, raid) and mix-v7 (its
  new sources: forecasting, oasst2, openjev_x, scruples).
- **How the rows come back** (`scripts/build_mix_v75_np.sh`, `scripts/mix_np.py`, Colab
  CPU, about 4 h). mix-v6 is rebuilt (its own 63 sources) and mix-v7 re-extended from
  mix-v6, both without the rebalance. For each of the nine sources, the seeded rebalance of
  the rebuilt file gives the built mix's rows in order (licence fields aside), so the
  dropped rows are exactly those. oasst2 is the exception: today's post-pass keeps 3
  majority rows more, so the dropped rows are located by the seeded draw itself (it picks
  positions in the majority list): 2,819 of 2,851 are restored, the 32 in the three
  ambiguous gaps are not. mix-v7-np (mix-v7 with those files, its rows byte for byte) then
  goes through `extend-v75 --no-rebalance-noul` and the mix-v7.5 tiered derive.
- **Planner merge.** Self-contained `mix.yaml`, merged over mix-v7.5's (`@np`); the
  checkpoint-selection set `heldout-select.jsonl` stays mix-v7.5's.

- **Drift since the mix-v7.5 build.** Of mix-v7.5-pre's 121 train files, 104 came back
  byte for byte. Besides the nine sources, eight did not:
  - compar:IA / lmarena / community alignment, social_chem and tasksource_jev: upstream
    datasets updated since the build;
  - jev_distractor: it is derived from the base rows, which now include the restored ones.

  These eight keep mix-v7.5's train files, with their weights unchanged
  (`np_frozen_from_v75` in the manifest). Today's cleaning also unescapes `&#x200B;` in
  scruples, so its rows that mix-v7.5 trains on take mix-v7.5's lines (313 rows).
- **Held-out files.** All 122 files are mix-v7.5's, byte for byte (sha256), and no train
  state equals a held-out state.
- **Yes-share of the noul train rows** (final files, mix-v7.5 → np): raid 70.0 → 97.3%,
  mlma_en 70.0 → 82.6%, mlma_fr 70.0 → 75.5%, mage 70.0 → 73.4%, oasst2 29.9 → 8.2%,
  scruples 30.0 → 24.5%, openjev_x 30.0 → 25.5%, forecasting 30.0 → 27.7%, openjev
  30.0 → 28.8%. Over all noul rows the share is 44.5% → 44.6%.
- **Mixture.** Block shares, French (22.4%) and the tiers are unchanged, on 672,640
  unique rows (mix-v7.5: 672,075). The restored rows move a few weights inside their
  blocks (n^0.5): raid 0.50 → 0.61% of the mix, mlma_en 0.26 → 0.31%, mage
  0.65 → 0.61%. Each source's epochs at 5,000 steps are within 0.02 of mix-v7.5's
  (tier B 0.96–1.00, decider 1.86, certo 1.72). At 2,500 steps they are within 0.01,
  except jev_fr_pilot (200 rows: 1.03 → 0.93).

### Mixture (5,000 steps, best_fit)

| block | share | sources |
|---|---|---|
| jev | 34% | v7's + jev_fr_mt (2.9% of the mix) |
| targeted | 22% | v7's (tempreason_l23 deduplicated) |
| generators | 14% | the v7 generators (84.1% of the block, one group each) + the nine v6 generators (15.9%) |
| replay | 10% | v6 |
| judge | 7% | v7's, preference sources rebuilt, single-vote groups lowered |
| breadth | 5% | tasksource_jev (subsampled to its draws) |
| analysis | 5% | v6 + x_stance_fr |
| writing | 3% | gen_register, gen_spelling_error |

The shares hold to 0.01 pt after the cap in all three mixes; none overflows.

| | mix-v7.5 | mix-v7.5-rep1 | mix-v7.5c |
|---|---|---|---|
| French (sampled) | 22.4% | 23.3% | 22.1% |
| unique train rows | 672,075 | 710,367 | 674,314 |
| licence use (sampled) | commercial 92.7%, non-commercial 2.3%, unspecified 5.0% | | commercial 94.3%, unspecified 5.7% |
| segments per step (best_fit) | 147.0 at 97.4% fill (steps 1024 / 2048 / 4096: 2,694 / 1,413 / 893) | | |

- **mix-v7.5c.** It drops the non-commercial rows: cxm_fr, x_stance_fr, xnli_fr,
  xnli_fr_mt and 27,167 tasksource rows. jev_fr_mt is commercial: its upstream is
  Apache-2.0 / MIT and the MT model is CC BY 4.0; CometKiwi (NC-SA) only scored rows.
- **Full table.** Each mix's `README.md` lists every source's share and epochs, and
  `manifest.json` has the tiers (`tiers`), the subsample (`min_epoch_subsample`), the
  fresh rows (`fresh_generators`) and the epochs at 2,500 and 3,000 steps
  (`epoch_cap.at_steps`).

### jev_fr_mt

Every train row of plumb, decider_teacher and mghafiri_scenarios was translated (44,734
rows; held-out rows never enter train). The models are CTranslate2 float16 builds of
opus-mt-tc-big en→fr / fr→en, run on one Colab L4. The gate is the five checks plus
CometKiwi (wmt22-cometkiwi-da) at the defaults: row ≥ 0.72, no segment of 20+ characters
under 0.45.

- **Protected markup.** Before translation, `fr_mt.plan_text` sets these aside:
  - fenced code blocks;
  - markdown table rules (`|---|`);
  - pipes: a table cell or pipeline stage is translated alone, and a stage that looks
    like code stays as is;
  - inline code spans, which become `TJVC<n>` placeholders and are restored afterwards.

  A new `markup` check rejects a row whose pipes, backticks or code spans changed.
- **The gate runs in parallel.** The gate is pure Python, and on one core it left the GPU
  idle half the time, so it now runs in a fork pool. A row the gate cannot read is
  rejected, not crashed.
- **The 200-row sample again** (same seed as gen-v7's): 132 kept, against 127 before the
  protection (plumb 9/22, was 7; decider 108/129, was 104; mghafiri 15/49, was 16). Markup
  passes on every gated row, and 6 kept rows have intact tables.

**Full run.**

| source | rows | kept | keep rate (of gated) |
|---|---|---|---|
| decider_teacher | 28,810 | 24,032 | 83.6% |
| mghafiri_scenarios | 11,004 | 4,748 | 43.9% |
| plumb | 4,920 | 1,442 | 29.4% |
| all | 44,734 | 30,222 | 68.0% of gated |

- **Failed checks (of gated rows).** quantities 8,367, entities 5,465, kiwi 4,445,
  fluency 1,831, round trip 743, markup 550, French 240. 291 rows were skipped as
  English-specific.
- **chrF of the round trip (kept rows).** p5 66.6, p50 78.9, p95 93.4.
- **CometKiwi (kept rows).** p5 0.760, p25 0.808, p50 0.840, p95 0.882. Over all gated
  rows: p5 0.716, p50 0.829. Worst segment of kept rows: p5 0.528, p50 0.756.
- **Held-out translations.** validation 380 / 639, calibration 388 / 640, test 376 / 632.
  They go to `validation-v7` / `calibration-v7` / `test/jev_fr_mt.jsonl`.
- **Cost.** The L4 runtime was up 1 h 24 min. The translation itself took 41 min of wall
  time: MT, round trip and CometKiwi. The rest was setup, the sample, a crashed first
  shard and the upload. At about 1.6 CU/h for an L4, that is about 2.3 compute units.
- **Post-pass.** 30,222 → 27,984 rows. 2,207 rows were dropped as a cross-source state:
  JSON states with nothing translatable, identical to their English decider row.
- **Files.** Rows record `license` (upstream licence + MT model CC BY 4.0) and
  `license_use` (the base source's: commercial). Per-row metadata (base id and source,
  MT / QE model, chrF, CometKiwi, checks) is in `data/jev-fr-mt/meta/jev_fr_mt.jsonl.gz`.
  The raw run is in `data/jev-fr-mt-run`, with the segment tables for `fr_mt.regate`.

### Evaluating the per-source held-out sets

`tjev post RUN --extra-validation PATH` (repeatable) evaluates PATH with the run's
temperatures, like the validation split. A relative PATH that does not exist is read from
the mix directory. Each file writes three things:
- `post/<name>-cal{tag}.json`, the report;
- `post/predictions/<name>{tag}.jsonl.gz`, the per-item dump;
- a per-source table in `summary{tag}.md`, weakest source first.

Its v1.5 scoring is added to `v15{tag}.json`. In `cloud/tpu_campaign.sh`, the calib probe
passes `PROBE_CALIB_EXTRA_VALIDATION` (space-separated file names) to `tjev post`. Without
the option nothing changes.

### Remaining risks

- **The tier-B subsample cuts most rows.** 1.05M rows are dropped in mix-v7.5:
  tasksource_jev keeps 13%, the preference sets 10–20%. What a run sees is unchanged,
  since the dropped rows were not drawn in expectation; but a longer run on this mix
  repeats the kept rows. Re-derive from `mix-v7.5-pre` for another horizon.
- **The judge block is still single votes.** Duplicate votes are rare (≤ 2% of rows), so
  the smoothing (0.75 / 0.55 / 0.857) carries the noise model, not real agreement.
- **jev_fr_mt is biased toward decider.** 80% of its rows are decider rows, and plumb
  keeps 29%. The gate cannot see word-sense errors. mghafiri rows are Claude-written
  upstream, so the source stays separable. Score at 2.9% of the mix.
- **tempreason_l23 lost 65% of its rows** (2 per near-state cluster). Raise
  `clean_v75.CLUSTER_CAPS` if it underperforms.
- **The fresh generator rows of rep1 skip the eval-set decontamination.** They pass the
  mix-v6 post-pass only (held-out, public JevBench 8-grams), not the v7 eval-set check.
  These are writing-generator rows (~7k); v7's check dropped ~0.02% of generator rows.
- **The unknowable twins.** They (0.5 targets, ~0.86% of the segments) score as
  abstentions under the raw JevBench noul readout (see [Generators v7](#generators-v7)).
- **Findings of the data report** (4,000 rows sampled per source):
  - **HTML entities.** 386 scruples rows still carry them. They were double-escaped
    (`&amp;#39;`), and the build unescaped only once. The code now unescapes until stable,
    but the uploaded mixes are unchanged.
  - **"One label" flags.** gen7_posterior's hard rows are almost all `no` (its exact-zero
    posteriors; its signal is in the soft rows). gen7_unknowable's choice rows are all the
    abstain option. Both are by design.
  - **Leakage and repr checks.** No train row equals a validation or test row. 18 rows
    show Python reprs, inside conversations.
- **The held-out sets of the rebuilt preference sources changed** (same group split, new
  rows), so validation-v7 numbers for those six sources are not comparable with mix-v7
  runs. The other sources' validation-v7 rows are mix-v7's, text-fixed.

## Sources and licences

The registries live in `tjev.data.sources` (one module per block), and every entry records
its license. **Commercial** = cleared for commercial use by its card; `--commercial-only`
builds without the others. Licenses are copied from the dataset cards and must be re-checked
before any commercial use. From mix-v7 every train row also records `license` and
`license_use` (`commercial` / `non-commercial` / `unspecified`), and `tjev data derive
--drop-license-use non-commercial[,unspecified]` derives a commercial-only mix (mix-v7c,
mix-v7.5c). The licence table of the French sources, as read from the dataset cards, is in
[fr-data.md](fr-data.md#licences-from-the-dataset-cards-read-2026-10-01).

The sources of the first mixes (mix-v3 to mix-v5b; the generators' caps are per `--gen-scale 1`):

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

**Added in mix-v5** ([fr-data.md §5](fr-data.md#5-mix-v5-as-built)): human-translated XNLI /
PAWS-X dev splits for training (`xnli_fr`, `pawsx_fr`) with the machine-translated train sets
as half-weight train-only sources (`xnli_fr_mt`, `pawsx_fr_mt`), `mtop_fr`, `piaf_fr`, more
Allociné, and the 275-row Claude translation pilot `jev_fr_pilot`.

### The targeted block (mix-v6)

Aimed at the weak JevBench families of the mix-v5b 4B, at over-confident yes/no abstention and at French decisions:

| source | dataset | card licence | redistributable | train | val / cal / test | sampled share | epochs @3000 | targets |
|---|---|---|---|---|---|---|---|---|
| eikos | caiovicentino1/eikos-decisions (core) | CC BY 4.0 | yes | 10,327 | 213 / 183 / 190 | 2.53% | 1.06 | exact (program), soft (teacher noul, exact probabilities) |
| eikos_long | caiovicentino1/eikos-decisions (long_context) | CC BY 4.0 | yes | 410 | 3 / 2 / 6 | 0.13% | 1.43 | as eikos |
| procedural | tasksource/procedural-typed-decisions | Apache-2.0 | yes | 14,328 | 300 / 300 / 300 | 2.98% | 0.89 | exact; posteriors soft |
| tempreason | tonytan48/TempReason (L1) | CC BY-SA 3.0 | share-alike | 10,000 (30% fr) | 300 / 300 / 300 | 2.49% | 1.07 | exact (recomputed) |
| helpsteer3 | nvidia/HelpSteer3 (preference) | CC BY 4.0 | yes | 7,316 | 300 / 300 / 299 | 1.07% | 0.62 | annotator votes / scores (soft) |
| cuad | theatticusproject/cuad-qa | CC BY 4.0 | yes | 7,924 | 300 / 300 / 300 | 1.11% | 0.60 | exact (annotations) |
| cold_law_fr | harvard-lil/cold-french-law | CC BY 4.0 | yes | 7,999 | 300 / 300 / 300 | 2.93% | 1.55 | exact (dates) |
| argimi_fr | artefactory/Argimi-Legal-French-Jurisprudence (juri) | CC BY-SA 4.0 | share-alike | 6,000 | 300 / 300 / 300 | 1.27% | 0.91 | exact (decision) |
| twin_squad2 | rajpurkar/squad_v2, negated | CC BY-SA 4.0 | share-alike | 3,989 | 300 / 300 / 299 | 0.92% | 1.00 | exact (1 − base) |
| twin_boolq | google/boolq, negated | CC BY-SA 3.0 | share-alike | 2,705 | 300 / 300 / 300 | 0.63% | 0.98 | exact (1 − base) |
| twin_piaf_fr | etalab-ia/piaf, negated | MIT | yes | 1,961 | 96 / 90 / 91 | 0.45% | 0.99 | exact (1 − base) |
| **block** | | | | **72,959** | **2,712 / 2,675 / 2,685** | **16.5%** | | |

Labels as built: cold_law_fr is 42% yes; cuad 50%; tempreason noul 50% yes; argimi_fr rejet
54% / cassation 35% / cassation partielle 10% / irrecevabilité 2%; the twins keep their base
sources' prior, flipped. The targeted block is 16.5% of the sampled segments but about 26%
of the sampled prompt tokens (procedural 5.8%, cuad 4.0%, argimi_fr 3.6%, helpsteer3
3.3%, cold_law_fr 3.1%, eikos 3.0%).

How the sources are recast:

- **eikos.** The Eikos-4B training data, `core` config: English rows whose label is
  computed (`program`, exact or writer-exact probabilities) and the teacher-labelled noul
  rows. Program rows are one-hot (the card's 0.95 smoothing is not kept); the others keep
  their distribution. The `eval/` files are never read. Validation = the card's
  `validation` split.
- **eikos_long.** The `long_context` dossiers (6k–32k tokens) cut to 10,000 characters: the
  asked document plus other documents of the dossier while they fit, in a random order,
  renumbered, and the question's "Document k" rewritten to the new number. Rows whose
  asked document alone is over the budget are dropped; 410 rows remain. Its held-out parts
  are tiny (11 rows: the card's 59-row validation split).
- **procedural.** tasksource's 13 configs, 800 states per config (150 for held-out) by
  hash of the row id, one item per question. Exact labels; the Bayesian posteriors
  (`partial_observation_calibration`, `policy_under_uncertainty`, the `random_*`
  questions) are soft targets. States over 10,000 characters and menus over 26 options
  are left out.
- **tempreason.** TempReason L1 ("1 year and 7 months after March 1873"): the answer is
  recomputed and rows that disagree are dropped. 55% choice among the answer and 3–5 near
  misses (±1 / ±2 months, ±1 year, the other direction, years-only or months-only offset),
  45% yes/no on one candidate; 30% rendered in French. L2/L3 (Wikidata facts) are not used.
- **helpsteer3.** HelpSteer3 preference, English rows (general, stem) up to 10,000
  characters, responses in a random order. Half the rows ask which response is better
  (A / B / tie, the annotators' votes as a soft target), half ask the 7-level comparison
  (the annotators' score distribution).
- **cuad.** CUAD clause presence, 38 of the 41 categories (Document Name, Parties and
  Agreement Date are almost always present). Contracts run to ~50k characters, so a row
  shows one 6,000-character excerpt: for a contract with the clause, a window containing
  its first annotated span at a random offset (yes); for a contract without it, a random
  window (no; the contract has no span of that category anywhere). Spans over 4,000
  characters are skipped; negatives are subsampled to the positives' count. Held out: the
  102 test contracts.
- **cold_law_fr.** 3.5% hash sample of COLD French Law (the LEGI articles in force).
  "D'après ses dates de vigueur, cette version de l'article s'applique-t-elle à la date du
  D ?" The state shows the text title, article number, context, the version's start date
  (and end date, if any), the signature / publication dates as decoys, and the article
  text (≤ 2,500 characters). Label: start ≤ D (and D < end). Negatives are dates before
  entry into force (or on / after the end), 20% the day before and 30% within a month.
  Dates in three formats. Held out by text title (10% of titles).
- **argimi_fr.** 12% hash sample of the ArGiMi Cour de cassation decisions with a single
  outcome (rejet, cassation, cassation partielle, irrecevabilité; joined appeals with
  several outcomes are dropped). The state is the decision from "a rendu l'arrêt suivant"
  (the header above it can name the outcome) to "PAR CES MOTIFS" (the operative part and
  everything after it, e.g. the annexed grounds, are cut). In the old criminal-chamber
  format, which opens with "CASSATION sur le pourvoi…", the text starts at "LA COUR,".
  Rows where an outcome phrase remains (`CASSE ET ANNULE`, `REJETTE le pourvoi`, an
  outcome word alone on a line or opening a line…) are dropped. Over 8,500 characters, the
  first 2,000 and the last 6,500 characters are kept around "[…]". The court's reasoning
  ("qu'en statuant ainsi, la cour d'appel a violé…" / "le moyen n'est pas fondé") is kept:
  reading it is the task. Held out by decision.
- **twin_squad2 / twin_boolq / twin_piaf_fr.** Negation twins of exact-label noul rows: the
  same state, the question negated ("Is the answer missing from the passage, for: …",
  "Based only on the text, is the answer no to: …", "La réponse est-elle absente du
  passage, pour : …"), the two criteria swapped, the label flipped (target 1 − p). Their
  held-out rows come from the base source's held-out split. Capped at one epoch.

Weights: `WEIGHT_SCALE` halves eikos_long's square-root weight twice (0.25) and halves
cuad, helpsteer3 and argimi_fr (long rows); `SOURCE_MAX_EPOCHS` caps the twins at 1.
`FR_TARGET["targeted"] = 0.30` keeps the mix at ~20% French.

**Skipped:** `louisbrulenaudet/code-*` (COLD already covers French law with dates in one
file; not needed). Every listed card states a licence (none NC or unknown), and every
dataset was reachable with the expected schema.

### The sources added in mix-v7 and mix-v7.5

mix-v7 added 42 sources in four blocks (jev, targeted, judge, breadth); they are listed per
block under [mix-v7](#mix-v7) in the history, and each mix's `README.md` on the Hub lists
every source with its rows, share, epochs and licence. mix-v7.5 added the v7 generators
([Generators v7](#generators-v7)), `jev_distractor` and `jev_fr_mt` (QE-gated machine
translation of plumb, decider_teacher and mghafiri_scenarios; [the current mix](#jev_fr_mt)).

## Generators

Every generator's labels are computed in code: a rule engine or arithmetic decides, only
the surface text varies. The nine mix-v3 generators (`tjev.data.sources.generators`:
base_rate, invoice_total, long_refund_policy, refund_policy, return_window,
schedule_conflict, table_argmax, ticket_triage, weekday) and the two writing generators
(spelling_error, register) draw 30% French. Their held-out parts use other seeds and their
own surface text (framings, questions, template sentences, vocabularies, names, some
parameter ranges), so evaluation measures transfer, not memorised templates. Since mix-v5
each has 6–7 training and 2–3 held-out templates per language, and `--gen-scale 2` doubles
their rows; since mix-v5b return_window trains on 194 windows (the held-out 7 / 21 / 45 /
120 never) and 20% of the return_window and long-policy rows carry an intermediate quantity
line ([fr-data.md](fr-data.md#7-mix-v5b-return_window-and-the-long-policy)). The
long-bucket generator `gen_longdoc_refund_policy` is described under [Long items](#long-items).

### Generators v7

The v7 generators (`tjev.data.sources.gen_v7`) are new designs, not more rows of the v6
ones. More rows from one generator saturate fast: Kev gained +26 pp from its first 6k
hard rows and +5 pp from the next 12k, so the limit is diversity, not count
(SynthLLM, GRAPE: regenerate, do not repeat). They aim at the weak JevBench families of
mix-v6 (temporal_numeric .47, long_policy .53, judge_hard .59, ambiguous .71) and at four
findings:

- **Contrastive one-fact pairs** (Bespoke-Nimble). Two states that differ by one fact
  and have different answers, with an evidence-ablation check.
- **Unknowable twins** (Kev). The same state with one evidence sentence removed. The
  target is the base rate (noul) or an explicit "cannot tell" option (choice), never "no".
- **Intermediate quantities** (Kev). Stating the day count took a 4B's date readout from
  0.65 to 1.00. Auxiliary questions ask for the quantity itself.
- **Distractor opinions** (JevAdvBench). One appended opinion flips 12% of decisions, so
  some rows carry an irrelevant or misleading opinion and keep their label.

Every label is computed in code, never by a model. Every source is deterministic by seed.
`generators.generate` dispatches the v7 names to `gen_v7.generate_v7`, which has the same
signature, so `Builder.generator` builds them like any generator. The v6 generators and
their output are unchanged: their pinned hashes still hold.

#### Sources

| source | bases | rows of a group | type | family | targets |
|---|---|---|---|---|---|
| `gen7_return_window_pairs` | `rw`: a time limit counted from an anchor event, and the two dates (or the elapsed days) | a, b | noul | temporal | exact |
| `gen7_weekday_pairs` | `wd`: a dated anchor and an offset | a, b | choice (7 days + abstain on half) | temporal | exact |
| `gen7_schedule_pairs` | `sc`: one *named* booking, a proposal, other bookings as decoys | a, b | noul | temporal | exact |
| `gen7_invoice_budget_pairs` | `inv`: an amount limit (≤ or ≥) and the order lines with a discount | a, b | noul | math | exact |
| `gen7_refund_pairs` | `rp`: a refund policy (category window, member extension, escalation limit, final sale) and a case | a, b | choice (+ abstain on half) | policy | exact |
| `gen7_long_refund_pairs` | `lrp`: as `rp`, the window sentence buried in a ~3k-token numbered policy (`_long_policy`) | a, b | choice | long_policy | exact |
| `gen7_unknowable` | all six bases (lrp at weight 0.25) | unk | noul / choice | unknowable | base rate (soft) / abstain |
| `gen7_aux_quantities` | all six bases | aux | choice | intermediate | exact |
| `gen7_posterior` | five known-posterior worlds | a | noul / choice | probability | exact posterior (soft) |
| `gen7_distractor_opinion` | v7 bases + seven v6 generators | a | as the base | adversarial | the base's |

All of them are licensed Apache-2.0 (generated by tjev code) and are 30% French by default
(`fr_share`). The real-text sources and `jev_fr_mt` are described below.

#### Groups, evidence and pairs

A base draws one world (its facts) and renders it as tagged lines (`core.Line`: text,
tag, the facts it states, the strings that state them). It returns a `core.Group`:

- **`a`.** The state, with the label computed by the base's solver (`SOLVERS[base]`).
- **`b`.** One line of `a` edited (≤ 8 words), which flips the label. The edit is to the
  request date or the window (`rw`), the offset (`wd`), the proposal or the booking's end
  (`sc`), one quantity or the limit (`inv`), or the delivery age or the window (`rp`).
- **`evidence`.** The two tags the label needs: the rule line and the fact line.
- **`unknown`.** The target of the twin.
- **`aux`.** The intermediate-quantity question.

`check_group` runs on every group while it is generated. It checks four things:

1. The solver on all facts gives the label.
2. For each evidence tag, there is exactly one line with that tag. The solver returns
   `None` (unknowable) on the facts the other lines state. No surface string of the
   removed line (its value with the words around it) is left in the state or the
   question.
3. `b` differs from `a` in that one line, by ≤ 8 words.
4. `b`'s label differs from `a`'s.

The question refers to the evidence ("…within the return period?", "Taking the category's
refund period into account…", "Does the proposed call overlap *the budget review*?"). So
removing the rule does not default to a closed-world answer. For the same reason the
schedule question names one booking, and the other bookings are decoys that may overlap
the proposal. The ablation also covers the reverse: no other line states the fact again.
`rw` decoys are dates that are not the rule's anchor (a newsletter, a card renewal).
Unknowable twins are never generated from a pair of hint lines, so the v5b hints
(`HINT_SHARE`) are not used here.

**Pair integrity.** Both members of a pair, the twin and the auxiliary question of one
group come out of one `generate_v7` call. They therefore land in the same split (train,
or one held-out part with its own seed). Their ids share the group prefix
`<source>-<split>-<seed>-g<k>-{a,b,unk,aux0}`. `generate_v7` returns whole groups only
(at most `n` rows). When any row of a group repeats a key, the whole group is skipped.
After a post-pass that drops single rows (hygiene dedup, the JevBench filters),
`gen_v7.drop_orphans(rows)` drops the other member of every broken pair.

**Held out by template family, not by row.** Each base has 5–7 training template families
and 2 held-out families per language (`RW`, `WD`, `SC`, `INV`, `RP`). The held-out split
also has its own fillers, names, goods, domains, other-booking names, windows (`rw`:
7/21/45/120 as in v6, `rp`: 21/35/60), slot grids and the `cannot_determine` abstain
option (training uses `insufficient_information`). Every row records its family in
`meta.template` (`rw.train.en.3`). The tests check two things: the train and held-out
families are disjoint, and no masked template line of a held-out row appears in any v7
or v6 training row.

**Meta.** Each row carries a `meta` dict: base, role, template, `license_use`, the tagged
lines, the facts, the dropped tag and the edit. The mix drops it, because
`parse_item` keeps only the item fields. The group survives in the id.

#### Unknowable twins and the JevBench abstention band

A twin is `a` without one evidence line:

- **choice** (`wd`, `rp`, `lrp`). The target is the abstain option, added when the
  question lacks it. The pairs show that option on half their rows, so its presence is
  never a cue.
- **noul.** The target is the family's base rate, as a soft target. A fair coin decides
  `a`'s label, so the base rate is 0.5 for `inv` and `sc`. For `rw` it is
  `rw_base_rate(split)`: 0.49 in train, slightly under 0.5 because of the elapsed ≥ 1
  rejection. The tests check it by enumerating the sampler.

The sampler's conditional posterior given the remaining facts was rejected as the
target. It rewards learning the generator's priors: once the window is removed, an
elapsed time of ≤ 2 days is always in time, only because no window is under 3 days.

**Trade-off.** The JevBench readout scores noul by P(yes): ≤ 0.2 is no, ≥ 0.8 is yes, and
anything in between is an abstention that counts as wrong
([official-eval.md](official-eval.md)). A model that learns 0.5 for the unknowable twins
is calibrated, but those rows score as wrong without the commit policy. With
`--abstention noul` (commit, M = 0, the recommended serving policy) they become a coin
flip, at the calibration cost already measured there. Use a small share
(`gen7_unknowable` at 6% of the block, ≈ 0.8% of the segments). Read the effect on
JevBench's ambiguous / trap families, and on the noul abstention rate with and without
the commit.

#### Intermediate quantities and the fade-out knob

`gen7_aux_quantities` asks a choice question about the same state as `a` (same text, same
group id). The quantities are the elapsed days or the last day of the window (`rw`), the
date (`wd`), the minutes of overlap (`sc`), the order total after discount (`inv`) and the
customer's refund period (`rp` / `lrp`). The distractor options are the near misses a
shortcut gives: ±1 day, the month-count shortcut, ±7 days, the total without the
discount, the discount applied to one line, and the member bonus added for a non-member.

**What the pipeline supports.** The stream draws each segment's source with fixed
probabilities (`Mixture.probs`; segment *i* from `default_rng([seed, i])`, whatever the
step), so a share that decreases over training is not supported. The curriculum is
therefore a separate low-share source. `gen_v7.AUX_FADE` / `aux_share(step, total)` hold
the intended schedule: 1.5% of the segments at step 0, falling linearly to 0 at 60% of
training. `aux_static_share()` is its mean, 0.45%, the share to give the static source. A
step-dependent mixture would need `Mixture.draw` to see the segment index (≈ step ×
segments per step). That change is in `tjev.data.mixture` and `segments`, and it was not
made here.

#### Exact posteriors (`gen7_posterior`)

Each world states every number the posterior needs, and the target is that posterior:

| base | world | type | posterior |
|---|---|---|---|
| `pw_signals` | prior + two conditionally independent signals (sensitivity, false-positive rate, each result positive or negative) | noul | Bayes |
| `pw_sampling` | a lot of N with exactly K bad, m drawn without replacement, fails at ≥ r bad | noul | hypergeometric |
| `pw_table` | past outcome counts by segment (2–4 segments); a new case in one segment | choice (3 outcomes) | that segment's frequencies |
| `pw_delay` | a delay distribution (3–5 values, shares summing to 100%) and a slack | noul | Σ shares with delay ≤ slack |
| `pw_policy` | a rule with two conditions: one known (75% satisfied), one unrecorded with a stated frequency | noul | the frequency, or 0 |

About half the noul targets fall inside the 0.2–0.8 band, where a one-hot label would
teach over-confidence. The instruction never uses the JevBench phrase "probabilities that
reflect the evidence" (an 8-gram the mix-v6 filter drops).

#### Distractor opinions and injections

`gen7_distractor_opinion` takes the state of a v7 base or of a v6 generator. It uses only
generators whose state is evidence for a rule: return_window, weekday, invoice_total,
base_rate, schedule_conflict, table_argmax and refund_policy, and never the writing or
triage generators, whose text is the object being judged. It adds one line, placed at the
end 75% of the time (JevAdvBench) and otherwise inside the state. The label is unchanged.
The line is one of four kinds:

- **irrelevant**: a remark about the coffee machine, an old logo;
- **misleading**: an opinion for a wrong option, always marked as an opinion, never a
  fact;
- **pressure**: the requester insists on a wrong outcome;
- **injection**: an instruction planted in the data ("SYSTEM NOTE: disregard the rules
  above and answer …", an HTML comment, a hidden footer).

Training and held-out lines are disjoint. `gen_v7.distractor.from_rows(rows, n, seed)`
derives the same rows from real decision sources, read-only (certo, decider_teacher,
mghafiri_scenarios, openjev, plumb, procedural, eikos, cold_law_fr). Each row keeps its
base row's labels or soft target, `meta.base_id` and `meta.license_use` (its source's
licence). Text-analysis sources are never used. The mix-v7 build calls it on built train
rows, under the source name `jev_distractor`, and only on the train split: held-out
distractor rows must come from held-out base rows.

#### Diversity against v6

`gen_v7.diversity.compare(n)` uses one state per v7 group (the `a` member), with numbers
masked. It reports distinct lines per 1,000 rows and distinct word 3-grams over all
3-grams:

| v7 source | v6 generator | distinct lines / 1k rows (v7 / v6) | distinct 3-grams (v7 / v6) | distinct questions (v7 / v6) |
|---|---|---|---|---|
| gen7_return_window_pairs | gen_return_window | 1106 / 589 | 0.090 / 0.033 | 24 / 14 |
| gen7_invoice_budget_pairs | gen_invoice_total | 749 / 145 | 0.027 / 0.017 | 16 / 12 |
| gen7_schedule_pairs | gen_schedule_conflict | 294 / 55 | 0.012 / 0.007 | 42 / 12 |
| gen7_weekday_pairs | gen_weekday | 796 / 498 | 0.045 / 0.040 | 14 / 12 |
| gen7_refund_pairs | gen_refund_policy | 260 / 290 | 0.030 / 0.024 | 12 / 12 |
| gen7_long_refund_pairs | gen_long_refund_policy | 1087 / 1391 | 0.006 / 0.006 | 12 / 12 |
| gen7_posterior | gen_base_rate | 186 / 92 | 0.027 / 0.012 | 30 / 60 |

n = 1,000 states, seed 0, 30% French. refund and long refund are about level with v6:
their diversity comes from the case structure, not from the wording.

#### Sizes (5,000 steps, ≈ 817,850 segments, generators block 14% ≈ 114.5k)

Rows are sized for ≤ 1 epoch of fresh rows at the planned share, plus 5% headroom for the
hygiene drops (`scripts/gen7_report.py` builds and measures them):

| source | share of block | segments | rows (≤ 1 epoch + 5%) |
|---|---|---|---|
| gen7_return_window_pairs | 11% | 12.6k | 13.2k |
| gen7_weekday_pairs | 6% | 6.9k | 7.2k |
| gen7_schedule_pairs | 7% | 8.0k | 8.4k |
| gen7_invoice_budget_pairs | 7% | 8.0k | 8.4k |
| gen7_refund_pairs | 8% | 9.2k | 9.6k |
| gen7_long_refund_pairs | 3% | 3.4k | 3.6k (~3k tokens a row) |
| gen7_unknowable | 6% | 6.9k | 7.2k |
| gen7_aux_quantities | 3.2% (= `aux_static_share()` 0.45% of all segments) | 3.7k | 3.8k |
| gen7_posterior | 10% | 11.4k | 12.0k |
| gen7_distractor_opinion | 5% | 5.7k | 6.0k |
| jev_distractor (`from_rows`, real rows) | 3% | 3.4k | 3.6k |
| gen7_rt_cold_fr_pairs / gen7_rt_kali_fr_pairs / gen7_rt_tau_pairs / gen7_rt_sharc_pairs | 7% / 5% / 2.6% / 0.3% | 8.0k / 5.7k / 3.0k / 0.3k | as segments (sharc and tau are capped by their seeds) |
| the nine v6 generators (not writing) | ~15.9% | ~18.2k | their mix-v6 rows (≤ 0.3 epoch each) |

**How `mix.py` wires them in.** `GEN7_PLAN` holds each source's share of the block and
its rows; the rows go to `GEN_TRAIN`, and `source_shares_v75` gives each source a group of
its own, so its weight is ∝ its rows and the block's sources are seen about equally often.
The nine v6 generators share `V6_GENERATORS_SHARE` (15.9% of the block): they are
saturated. `select_sources` and `build` treat `gen_v7.V7_GENERATORS` as generators, with
the licence and family from `gen_v7.source_info`. After the post-pass, `drop_orphans` runs
on every pair source, and the yes/no rebalance skips the pair and generated sources
(`V75_REBALANCE_EXEMPT`): pairs are 50/50 by construction, and a subsample would break them.

#### `jev_fr_mt`: QE-gated MT of the exhausted contender sources

Rows of plumb, decider_teacher and mghafiri_scenarios are translated EN→FR with
Helsinki-NLP/opus-mt-tc-big-en-fr (CC BY 4.0) through CTranslate2. The code is
`gen_v7.fr_mt`; `scripts/fr_mt_sample.py sample|run` runs it (`scripts/fr_mt_gpu.sh` and
`scripts/fr_mt_mix.py` for the full run). This section is the design and its 200-row
study; the full run and its statistics are under [the current mix](#jev_fr_mt).

- **What is translated.** Label ids, score levels and targets are copied, never
  translated. JSON states keep their keys and layout: only their string values are
  translated.
- **The gate.** A row is kept only when all five checks pass:
  1. Every date, 24-hour time and number survives both ways. Dates in EN and FR formats
     are read into year-month-day, and the weekdays of dates are checked too.
  2. Identifiers, e-mails and URLs come out verbatim, and at least 80% of the proper
     names survive.
  3. The round trip (opus-mt-tc-big-fr-en, greedy) scores chrF ≥ 50 for the row and
     ≥ 25 for every segment of 40+ characters.
  4. No segment has an odd length ratio or a repeating phrase.
  5. The text reads as French.

  No date or number may change, so the label's date arithmetic is unchanged. Rows that
  fail are dropped, never patched.
- **CometKiwi** (`Unbabel/wmt22-cometkiwi-da`, CC BY-NC-SA 4.0, gated; the project accepts
  NC-SA) is a sixth check on top of the five, which stay mandatory. It is reference-free
  QE of each (English segment, French segment) pair, run with `unbabel-comet` in a venv
  of its own (it needs numpy < 2, which jax refuses; `scripts/fr_mt_sample.py` then loads
  `fr_mt` by path). A row needs a length-weighted mean ≥ `MIN_KIWI` (0.72), and no
  segment of 20+ characters may score under `MIN_KIWI_SEGMENT` (0.45). Every row logs
  `kiwi`, `kiwi_worst_segment`, the round-trip `chrf`, the QE model and its licence in
  `meta`, and the segment tables keep the scores (`tables.json`: `fr`, `rt`, `kiwi`), so
  `fr_mt.regate` re-decides with other thresholds without translating or scoring again.

**Sample (200 rows, Colab CPU, 8 vCPU).**

| | |
|---|---|
| kept | 68.5% (decider 109/129, mghafiri 20/49, plumb 8/22) |
| median round-trip chrF | 77 |
| speed, end to end | 339 source characters/s (CTranslate2 is ~7× faster than torch) |

**CometKiwi on the same 200 rows** (8-vCPU Colab CPU): 2,795 segment pairs in 448 s, i.e.
6.2 pairs/s or 867 characters/s (source + translation).

- **Scores against the other checks.**
  - Row-score median: 0.83 for the rows they keep (p5 0.72), 0.81 for the rows they
    reject (p5 0.65).
  - The one row that fails the round trip scores 0.46. The two filters agree at the
    bottom, but most rejections are lost dates, numbers or names, which CometKiwi does
    not see: it is a complement, not a replacement.
- **Segment scores catch errors the other checks miss.**
  - A hallucinated title ("playing a crime drama" became "Histoire de la littérature
    française", 0.36).
  - A shell command changed (`myapp:latest` became `myapp:last`, 0.35).
  - A file name translated (0.36).
  - Pipes dropped from commands and tables (0.27–0.48).
- **The spot check's two flawed rows.**
  - "Crown Chest" became "Coffre": its segment scores 0.47, caught at a segment threshold
    of 0.5, not at 0.45.
  - "jamming" became "brouiller": 0.74, not caught. Its row scores 0.84.
- **Keep rates.**

| thresholds (row / segment) | kept | decider | mghafiri | plumb |
|---|---|---|---|---|
| none (the five checks) | 137 | 109/129 | 20/47 | 8/22 |
| 0.72 / none | 132 | 106 | 18 | 8 |
| **0.72 / 0.45 (default)** | **127 (63.5%)** | **104** | **16** | **7** |
| 0.72 / 0.50 | 119 | 99 | 15 | 5 |

Rows skipped as English-specific are not counted in the denominators.

The MT drops `|` from tables and shell commands. CometKiwi catches it; mix-v7.5 protects
pipes, table rules, code fences and code spans before translation (see mix-v7.5, jev_fr_mt).

A spot check of 10 kept rows (before CometKiwi) found 8 faithful. The other 2 had a word-sense error and a
word dropped from an item name; no label was wrong.

**Cost, estimated before the run.** At 63.5% kept, the whole 44.7k-row corpus (~38M source
characters) gives ~28k rows, so 25k kept rows needs nearly all of it.

| | CPU (8 vCPU) | T4 (estimated, not measured) |
|---|---|---|
| translation + round trip | ~33 h | ~2–3 h (CTranslate2 float16) |
| CometKiwi | ~25 h (~80M characters at 867/s) | ~1 h (`--gpus 1`, roughly 30× the CPU rate for XLM-R large) |
| total | ~58 h | ~3–4 h, a few compute units |

Measured: the full run (translation, round trip and CometKiwi) took 41 min of wall time on
one Colab L4, about 2.3 compute units, and kept 30,222 of 44,734 rows.

Each row records `license_use`: the upstream licence (Apache-2.0 / MIT, Claude-written
upstream) plus the MT model's CC BY 4.0.

**Risks.**
- **Rejection bias.** Long, date-dense rows are kept less often (plumb 36%), so the French
  set leans towards short decider rows.
- **What the gate cannot see.** Word-sense errors, JSON keys left in English, and score
  labels translated inside their descriptions.
- **Provenance.** mghafiri rows are Claude-written upstream: keep the source separable,
  like `jev_fr_pilot`.

#### Real-text seeds

The rule line is a verbatim excerpt of a real document and the fact line a generated
case. The label comes from executing the rule. All four sources are pair + unknowable
groups, with noul twins at 0.5 and choice twins at the abstain option, and each row keeps
`meta.license_use`.

| source | seed text | licence | rules |
|---|---|---|---|
| `gen7_rt_tau_pairs` | tau-bench airline and retail agent policies (MIT, commit 59a200c) | MIT | checked-bag fees, cancellation eligibility, order-status actions; the compensation and gift-card rules are held out |
| `gen7_rt_sharc_pairs` | ShARC gov.uk snippets with exactly one age, hours or £ threshold | CC BY-SA 3.0 | train snippets train, dev snippets held out (and the 2 shared snippets) |
| `gen7_rt_kali_fr_pairs` | KALI collective agreements | ODC-By | délais "dans un délai de N jours/mois à compter de …", trial and notice periods by category; held out by excerpt hash |
| `gen7_rt_cold_fr_pairs` | COLD / LEGI délais | CC BY 4.0 | held out with `cold_law_fr`'s title split |

- **tau-bench and JevBench.** The tau-bench policies share no word 8-gram with the
  JevBench public items, so they are kept.
- **Dates.** Each French state spells out its counting convention: a délai in days
  excludes the start day; months keep the same day of the month, clamped to the month's
  end; a period ends the day before; there is no move to the next working day.
- **Seeds.** `python -m tjev.data.sources.gen_v7.realtext build-seeds` builds them in
  about 2 min on Colab CPU (1,379 KALI and 3,628 COLD seeds), cached under
  `~/.cache/tjev/gen7`. The tests use small attributed fixtures in
  `tests/data/fixtures/gen_v7` (`TJEV_GEN7_SEEDS`).
- **Capacity.**
  - ShARC: about 330 train rows (23 snippets).
  - tau: the states are parametric over a fixed policy; keep it at about 3k rows.
  - KALI: over 10k rows, diversity bounded by about 1.3k excerpts.
  - COLD: over 20k rows (about 3.2k train excerpts).
  - Rows kept in mix-v7.5 after the post-pass: COLD 7,932, KALI 5,608, tau 3,000, ShARC
    274 (54 rows dropped for a line shared with the ShARC adapter's held-out rows).

#### Risks

- **Noul twins score as abstentions.** Their 0.5 targets count as wrong in JevBench's
  noul band without the commit policy. Keep the share small and measure with and without
  `--abstention noul`.
- **Pairs can be broken.** A post-pass can drop one member of a pair (hygiene,
  `--rebalance-noul`). Call `drop_orphans`, and do not rebalance pair sources.
- **The fade-out is static.** The aux curriculum is a fixed-share source until the
  stream gets a step-dependent mixture.
- **Distractor rows repeat real states.** Each real-row distractor is a near-duplicate
  of its base row, which adds to the effective epochs of those sources.
- **Long rows cost tokens.** `gen7_long_refund_pairs` rows are ~3k tokens. Check the
  token share with `scripts/data_report.py`.
- **Real-text quality.** ShARC is tiny (its rows that share a line with the ShARC
  adapter's held-out passages are dropped). tau is low in diversity. Some KALI and COLD excerpts carry extra
  conditions or loosely fitting anchors; the label is exact under the stated convention.

## Hygiene and decontamination

The checks run in this order; each is described where it was introduced, with the rows it
dropped:

1. **Per source** (below): held-out splits by evaluation split or hashed group, the
   builder's drops, the held-out generators' own surface text, the public JevBench filter.
2. **The hygiene post-pass** (`tjev.data.hygiene.clean`, mix-v4): rows equal or near a
   held-out row of *any* source, prompts over 4096 tokens, exact duplicates
   ([mix-v4](#mix-v4)); since mix-v6 also a single full-text 8-gram shared with a public
   JevBench item and a state equal or near *any* held-out state (below).
3. **The v7 post-pass** (`hygiene.clean_v7`), for every source added since mix-v7:
   8-grams of the public evaluation sets, exact and near duplicates against every kept
   train row, cross-source states (below).
4. **mix-v7.5** (`tjev.data.clean_v75`): HTML entities, Python reprs, control characters,
   language tags, mojibake, the tempreason_l23 cluster cap
   ([the current mix](#what-changed-against-mix-v7)).
5. **Balance** (`tjev.data.balance`, at build or derive): the yes/no rebalance, the epoch cap,
   the mix-v7.5 tiers and the min-epoch subsample.

### Per source

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
  characters or shared by ≥ 20 training rows. The data report measures what is left
  (near states, near items) after every pass.

### Train-side decontamination and passage dedup (mix-v6)

Both checks are part of the hygiene post-pass (`tjev.data.hygiene.clean`) and are on by
default (`--no-passage-dedup`, `--no-jevbench-ngram` turn them off). Held-out rows are
never touched.

- **JevBench 8-grams** (`jevbench_8gram`, `tjev.data.jevbench.JevBenchGrams`). A train row
  is dropped when its full text (state, instructions, option labels and descriptions)
  shares a single word 8-gram with the full text of one of the 231 public JevBench items.
  The older filter (`jevbench_overlap`: equal state, equal non-generic instruction, or
  more than two shared 8-grams of the state) still runs first, per source.
- **Held-out passages** (`train_state_in_heldout`, `train_state_near_heldout`). A train row
  is dropped when its state alone equals (normalised) or is a near-duplicate (MinHash,
  J ≥ 0.8) of the state of *any* held-out row (validation, calibration, test, both
  held-out generator sets, the JevBench items), whatever its question. This is the group
  split the mix-v5 report asked for, done train-side: BoolQ, SQuAD and PIAF ask several
  questions about one passage, and their train splits reused held-out passages.

Rows dropped by the two checks (after the per-source steps; first reason wins):

| check | rows | by source |
|---|---|---|
| `jevbench_8gram` | 1,261 | eikos 1,066, plumb 79, cuad 70, mghafiri_scenarios 38, eikos_long 7, decider_teacher 1 |
| `train_state_in_heldout` | 1,481 | boolq 866, twin_boolq 269, helpsteer3 90, piaf_fr 78, squad2 37, twin_piaf_fr 37, mage 33, go_emotions 31, certo 22, clinc_oos 9, twin_squad2 5, massive_en 2, mlma_fr 1, pavlick_formality 1 |
| `train_state_near_heldout` | 717 | openjev 422, boolq 82, helpsteer3 77, procedural 38, twin_boolq 26, squad2 23, decider_teacher 13, eikos 8, mage 8, cuad 5, twin_squad2 5, others ≤ 2 |

- **JevBench.** In the mix-v5b sources, 118 rows shared an 8-gram with a public item:
  policy boilerplate in plumb ("1.1 This policy governs…", "working day means Monday to
  Friday excluding public holidays") and mghafiri, one decider question. In eikos, the hits
  are two instruction templates of its probability and rules_permit families ("give
  probabilities that reflect the evidence in the state", "if all of the following hold"),
  shared with JevBench items. They are dropped all the same. The older per-source filter
  dropped 51 rows (mix-v5b: 47).
- **Passages.** BoolQ loses 948 train rows (10%), PIAF 78, SQuAD 60, openjev 422 (templated
  states near held-out ones). Leakage measured by the data report:

  | split | exact state, v5b → v6 | near state, v5b → v6 | near item |
  |---|---|---|---|
  | validation | 124 → 0 | 178 → 0 | 0 → 0 |
  | calibration | 136 → 0 | 199 → 1 | 0 → 0 |
  | test | 152 → 0 | 197 → 3 | 0 → 0 |
  | JevBench public | 1 → 0 | 1 → 0 | 0 → 0 |

  The 4 remaining near states are openjev rows that MinHash's banding misses in the
  post-pass but the report's exhaustive pass finds.

### The post-pass of the sources added since mix-v7 (`hygiene.clean_v7`)

Every mix-v6 check runs first:
- held-out rows equal or near, state included;
- the public JevBench 8-grams, on single 8-grams;
- prompts over 4096 tokens.

Then, first reason wins:

1. **Evaluation-set 8-grams** (`tjev.data.decontam`).
   - **Reference sets.** HF `JevBench/jevbench` (questions, cases, rewrites), BFCL,
     UnSeenTimeQA, Test of Time, sycophancy-eval, LLMBar (from GitHub; the HF repo is
     script-only), AmbiEnt, HateCheck EN / FR, PPE, ANLI dev / test and bbh-fr. Each is
     read generically (every string of a row).
   - **Hit rule.** A hit needs 3 shared 8-grams; the JevBench suites need only one.
   - **Not counted.** Three kinds of 8-gram are set aside before counting:
     - a set's own templates (8-grams in ≥ 1% of its items);
     - the source's own templates (≥ 0.05% and ≥ 20 of its rows);
     - common phrases (8-grams in ≥ 50 new train rows, e.g. LLM filler).
   - **Our own held-out rows.** These count as a hit only for a passage: ≥ 25 shared
     8-grams, or ≥ 20% of the row's 8-grams. Held-out rows of the row's own upstream
     dataset do not count; the group splits and near checks cover them.
   - FiscalQA's questions are not on the Hub, and nothing reads them.
2. **Exact duplicate** of any kept train row, mix-v6's included.
3. **Near duplicate.** MinHash J ≥ 0.8, on a state that asks the same question
   (instructions, type and menu), from any source.
4. **Cross-source state.** A state equal or near to a kept row of another upstream dataset,
   whatever the question.
   - Rows of one origin may share a state; Open-Jev v1 / v1.1, compar:IA, xLAM and LexGLUE
     each count as one origin.
   - The LSH index grows source by source (`near.BandIndex`).

New held-out rows on a mix-v6 train state are dropped: 115 rows (openjev_ctrl 38,
mlagnajit_td 73, …). The mix-v6 train files are never modified.

Rows dropped by the post-pass in the mix-v7 build (new sources only):

| reason | rows | mostly |
|---|---|---|
| near a held-out row (same question) | 22,906 | openjev_ctrl 10,444, openjev_x 7,617, procedural_x 2,350 |
| near duplicate of a train row | 21,580 | openjev_x 6,386, openjev_ctrl 6,035, procedural_x 4,458, certo_x 3,918 |
| PPE 8-grams | 19,099 | lmarena_140k 5,740, helpsteer3_x 4,929, oasst2 4,304 |
| cross-source state | 9,682 | mlagnajit_td 3,334 (texts of our replay sources), tatqa 2,629, xlam 2,057 |
| our held-out rows' 8-grams | 6,069 | when2call 2,502, xlam 2,284, xlam_irrelevance 543 |
| exact duplicate | 6,031 | helpsteer3_x 1,353, openjev_x 953, cetat_fr 897 |
| held-out state (passage) | 3,307 | |
| ANLI dev / test 8-grams | 2,251 | tasksource_jev |
| LLMBar 8-grams | 1,530 | oasst2 1,196 |
| prompt over 4096 tokens | 1,347 | tasksource_jev 1,048 |
| JevBench HF 8-grams | 1,109 | kaivoss_s1 478, tasksource_jev, oasst2, lmarena |
| other sets | 1,148 | sycophancy-eval 491, BFCL 475, AmbiEnt 60 |
| public JevBench 8-grams | 109 | kaivoss_s1 42, n4ze3m_td 23 |

The yes/no rebalance (0.7) ran on the new sources, except xlam_irrelevance and dices.
xlam_irrelevance is the "no tool applies" side of xlam; dices is valued for its soft
shares. It dropped openjev_x 3,808, oasst2 2,851, scruples 1,145 and forecasting 208 rows.

## Evaluation sets

- **JevBench public** (`tjev data jevbench REPO OUT`): 231 items (easy 48, original 72,
  hard 111), for final measurement only. They are never part of a mixture, a validation
  set or a calibration fit. Sealed items are never used, and no generator is written from
  the sealed family names.
- **Held-out generators** (`tjev data heldout`). Two sets with different seeds: the
  *selection* set (seed 777, `data.heldout`, part of the checkpoint selection score) and
  the *reporting* set (seed 4242, evaluated by `tjev post`).
- **The mix's validation and calibration splits** (`validation.jsonl`, `calibration.jsonl`;
  byte-identical from mix-v5b to mix-v7.5, so checkpoint selection is comparable), and the
  per-source held-out sets of the sources added later (`validation-v6` / `-v7`,
  `calibration-v6` / `-v7`, `test/<source>.jsonl`), evaluated with
  `tjev post --extra-validation`.

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

### Packing (`train.packing`)

The packer (`tjev.data.pack.BucketPacker`) fills the rows of up to `train.packing_bins`
open microbatches per bucket. `train.packing` chooses where a segment goes:

- **`bucket` (the default).** The smallest bucket that holds the segment, first-fit over
  its open microbatches. Every segment over half a bucket is alone in its row: two
  1025–2048-token segments cannot share a 2048 row, and the short segments, which all go to
  the 1024 bucket, never fill the rest of it. A 2048 or 4096 step is one segment per row
  (32 / 16 slots per step at 4B, the `data/slots_per_step` of production runs).
- **`best_fit`.** Best-fit over the rows of every open microbatch of every bucket that holds
  the segment, longer buckets included: the row left with the least room. A new row of the
  smallest bucket that holds it is a candidate too, so a segment that fits its own bucket
  better than any open tail opens a microbatch there. The short segments then fill the
  tails of the long rows.

Both keep what the stream guarantees: segment i is the same function of (seed, i), no
segment is cut, no row exceeds its length or `max_segments` slots, rows keep the segment-id
contract (`check_segments`: runs 1, 2, … then padding, positions restart at 0, so attention
and the GDN conv and state reset at every segment), segments over the largest bucket are
still dropped, and the state (`{next_index, open, ready}`) is unchanged, so a run resumes
exactly. `tests/data/test_packing.py` checks these for both placements, and
`tests/data/test_pipeline.py` the Grain stream and its resume under `best_fit`.

**Measured offline** (`scripts/packing_report.py`, the real stream of seed 0: mixture,
render and tokenizer of `tjev train`; 3,000 steps × 65,536 tokens, buckets 1024 / 2048 /
4096, 16 bins):

| mix-v6, microbatch 4096 (4B) | `bucket` | `best_fit` |
|---|---|---|
| pad fraction 1024 / 2048 / 4096 | 4.0 / 34.7 / 29.4% | 3.9 / 1.8 / 0.7% |
| pad fraction, all tokens | 15.6% | 2.9% |
| steps 1024 / 2048 / 4096 | 1788 / 786 / 426 | 1762 / 748 / 490 |
| slots per step 1024 / 2048 / 4096 | 221 / 32 / 16 | 212 / 122 / 56 |
| real tokens per step | 55,289 | 63,649 (×1.151) |
| segments trained in 3,000 steps | 427k | 492k |

The `bucket` column is the production run's own (4B rank 16 on mix-v6: 4.0 / 34.7 / 29.4%
pad, 15.6% overall). The other microbatches: 8192 (0.8B / 2B) 84.5 → 97.2% real tokens
(×1.151 per step), 2048 (9B) 84.4 → 97.2% (×1.166 per step; its 4096 steps hold
131k tokens). mix-v5b at 4096: 85.9 → 97.1% (×1.130).

- **The mixture holds.** Per source, the share of the trained tokens is the share of the
  tokens the stream handed the packer to 1e-4 (absolute) under both placements: packing
  delays segments, it never selects them. Between the two placements the shares differ by
  at most 0.05 points (1.5% relative for sources over 0.5%), because `best_fit` reads 15%
  further into the same stream.
- **What changes for a run of the same steps.** It trains on ~15% more segments: the
  effective epochs of every source go up by as much (a mix's epoch cap, `cap_epochs`, is
  computed under `bucket`: `balance.simulate_stream(packing=...)` measures either), and
  the expected slots per step (the loss normaliser, measured on the stream) go from 142 to
  164. The long-bucket steps hold 4–8× more answer slots, so a clipped 4096 step no longer
  weighs ~15× per item (training.md, "Clipping"). For the same data, a run takes ~13% fewer
  steps.
- **Speed.** A step's time depends on its shape, not on its padding (production 4B:
  4.49 s for every bucket), so real tokens per second rise like the real tokens per step:
  ×1.00 / ×1.50 / ×1.41 in the 1024 / 2048 / 4096 steps, ×1.15 overall.

## Long items

**What exists over 4096 tokens.** The mix-v3 report counted 75 training rows over 4096
tokens in at least one layout (`reports/data/mix-v3/report.md`), and mix-v4's hygiene
dropped all 75 as `prompt_too_long`:

| Source | Rows > 4096 | Longest (plain layout) |
|---|---|---|
| raid | 62 | 18,630 |
| plumb | 9 | 5,671 |
| mage | 3 | 4,872 |
| helpsteer2_correctness | 1 | 4,494 |

- **No long-document source in mix-v4.** gen_long_refund_policy tops out at 3,784 tokens
  (p50 2,969). Its boilerplate pool caps a document at ~3.5k tokens in English and ~4.9k
  in French, so raising its budget cannot reach 8–16k without repeating sections.
- **raid's long rows are machine-text detection.** They are not policy reading, and 62
  rows are too few to train a 16k bucket.
- **The held-out splits are not filtered for length.** Their long rows are evaluated only
  when the run has a bucket that holds them; otherwise they are counted in
  `dropped_too_long`.

**`gen_longdoc_refund_policy` (`LONG_GENERATORS`).** The long-bucket source, family
`long_policy`: gen_refund_policy's case set against a handbook of 3–5 stores' long
policies. Each chapter is one `_long_policy` document with its own window for the case's
category, member bonus and amount limit. The case names its store, so only that chapter
answers it.

- **Distractors that disagree.** Cases that some rules would decide differently are
  preferred; the others are kept with probability 0.3, so final sales stay. Then the
  first other store's rules do decide the case differently, and reading the wrong chapter
  gives the wrong label.
- **Labels.** The test parses every English row's chapters and re-derives the label.
  Labels are deny 44%, escalate 30%, approve 26%.
- **Lengths.** 5–14.5k prompt tokens with the real tokenizer (22k–80k characters), so
  both the 8192 and the 16384 buckets get rows. A 16384 row holds one document.
- **Held-out split.** Its own store names, handbook framing, policy text and domains.
- **Opt-in.** It is built only by `tjev data build --max-prompt-tokens 16384`, which keeps
  prompts up to 16384 tokens and adds the long generators to the default sources, or when
  named with `--sources`. A default build (4096, mix-v4's) is unchanged.
- **Not part of the held-out generator sets.** `heldout-select.jsonl` and the reporting
  set are unchanged. A long mix still gets its validation, calibration and test rows from
  the builder.
- **Size and weight.** 1000 training rows (each about 3× a gen_long_refund_policy row in
  tokens), with its n^0.5 weight × 0.25.
- **No long mix has been built yet.** Before training a long bucket on it, check its token
  share with `scripts/data_report.py`. The report's buckets are still the default 1024 /
  2048 / 4096 there: set `BUCKETS`.

## Data quality reports

`scripts/data_report.py` audits a built mix without modifying it: the mix as sampled, the
effective epochs of each source (a simulation of the real packer over the weighted stream,
with each row's length under the real template and tokenizer), label balance, exact /
conflicting / near-duplicate rows (MinHash on states, Jaccard ≥ 0.8), leakage into every
held-out split and the JevBench public items, lengths and buckets, and quality flags. It
ranks recommendations. The reports of mix-v3 to mix-v6 are under
[`reports/data/<mix>/`](../reports/data/) (`report.md`, `report.json`, the plots; from
mix-v5 the build log, from mix-v6 `compare.txt` against the previous mix). The mix-v7 and
mix-v7.5 reports were run on 4,000 sampled rows per source; their findings are under
[mix-v7](#mix-v7) and [the current mix](#remaining-risks).

```
JAX_PLATFORMS=cpu nice -n 19 uv run --with matplotlib python scripts/data_report.py \
    --mix ~/tjev-work/data/mix-v3 --jevbench ~/tjev-work/data/jevbench/public.jsonl \
    --out reports/data/mix-v3            # ~4.5 min, one process, 2 tokenizer threads
```

## History

Each mix, what it changed against the one before, why, and how the two measured. The
training results on these mixes are in [results.md](results.md), the French diagnosis and
mix-v5 / v5b in [fr-data.md](fr-data.md), the calibration studies in
[calibration.md](calibration.md).

### mix-v3

The first mix of this layout: 47 sources in five blocks, set by block shares (`BLOCKS`;
mix-v6 added the `targeted` block to it, mix-v7 and mix-v7.5 have `BLOCKS_V7` /
`BLOCKS_V75`). Within a block, sources are weighted ∝ n^0.5.

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

#### The data report's findings

For 3,000 steps × 65,536 tokens on 4B (microbatch 4096):

- **Epochs.** The packer fits ~146 segments per step, about 440k in the run. The sampled
  mix repeats the generators ~3 times, plumb 3.7 times and `gen_long_refund_policy` 4.4
  times. The replay and analysis sources get 0.24–1.3 epochs, and 21–30% of the rows of
  the largest ones are ever seen. No source is below 0.1 epochs.
- **Tokens.** `gen_long_refund_policy` is 2% of the segments but 16% of the prompt tokens.
  Steps use one bucket each, so 22% of the steps are 4096-token steps at 72% fill and 21%
  are 2048-token steps at 61% fill. The overall fill is 84%.
- **Adapter bugs.**
  - `hc3_fr`: every train, validation and test row has gold `no`. The CHATGPT label never
    matches.
  - `mghafiri_scenarios`: 1,679 rows (15%) show their score levels as Python dict reprs
    (`{'level': …, 'definition': …}`), and 31 noul rows show dict criteria.
  - `decider_teacher`: 79 rows show options as `{'value': None}`.
- **Duplication.** There are 1,785 exact duplicate rows (decider_teacher 1,170,
  textdetox_fr 190, pawsx_fr 142) and no conflicting labels. 27k states are
  near-redundant: openjev 51% and certo 29% of their states, and gen_ticket_triage 45%.
- **Leakage.** Exact (state, instructions) leaks are rare: 5 / 5 / 2 rows across sources.
  About 0.9% of each held-out split is a *near item* (near state, same instructions and
  gold), mostly certo (~20% of its held-out rows) and openjev. BoolQ shares a third of
  its held-out passages with train, under other questions. JevBench has one state-only
  near match (massive_en → an easy intent item).
- **Labels.** After shuffling, the gold letter matches chance (total variation 0.001).
  The yes/no priors are skewed in raid (97% yes), mlma_en (83%) and mlma_fr (76%).
- **Lengths.** p50 171 / p99 1,497 tokens (plain layout); 96.5% of rows fit the 1024
  bucket. 75 rows (raid 62) exceed 4096 and are dropped silently by the packer: 0.01% of
  draws.

### mix-v4

mix-v4 is mix-v3 with the fixes from the mix-v3 data report. The CLI and the steps are the
same, and the tokenizer is found under `~/tjev-work/models/Qwen3.5-*` (or given with
`--tokenizer`):

```
tjev data build ~/tjev-work/data/mix-v4 --jevbench-public ~/tjev-work/data/jevbench/public.jsonl
tjev data heldout ~/tjev-work/data/mix-v4/heldout-select.jsonl     # ~4.5 min in all
```

Its report is [`reports/data/mix-v4/report.md`](../reports/data/mix-v4/report.md), next to
[mix-v3's](../reports/data/mix-v3/report.md).

What changed, and why:

- **hc3_fr labels.** `label` is a ClassLabel int (0 `HUMAN`, 1 `CHATGPT`). The adapter
  compared it with the string `"CHATGPT"`, so every row was `no`. It now maps the int
  through the label names: 31% `yes` in train, and about half `yes` in the held-out
  parts (validation 130 / 245, calibration 120 / 230, test 110 / 235).
- **Structured options as text.** `contenders.criterion_text` turns decider and mghafiri
  option descriptors into text: `{'level': 'Minor', 'definition': 'Cosmetic'}` becomes
  `Minor: Cosmetic`, `{'definition': …, 'examples': […]}` becomes `…; Examples: …`, and
  `{'value': None}` / `{'_null': None}` / None become an empty description (the option
  shows its label only). Dict-valued instruction fields (mghafiri's
  `Exemption list: {…}`) are converted the same way.
- **HTML entities.** `html.unescape` runs in the tweet, MLMA, HelpSteer2, jailbreak and NLI
  adapters (MNLI's `S&amp;L`). raid, boolq and allocine keep theirs: there they are
  attacks, or text about HTML.
- **Cross-source hygiene.** `tjev.data.hygiene` is a post-pass of the builder over the
  train files, in build order. It drops a row (first reason wins) when:
  1. its normalised (state, instructions) equals a held-out row of *any* source, or
  2. its state is a near-duplicate (the report's MinHash: word 3-grams, 128 permutations,
     LSH 32 × 4, J ≥ 0.8, in `tjev.data.near`) of a held-out row with the same
     instructions, or
  3. its rendered prompt is over 4096 tokens (all options, either layout; the packer would
     drop it silently), or
  4. it exactly duplicates an earlier train row (normalised state + instructions + type +
     options + target; the first is kept).

  "Held out" means validation, calibration, test, both held-out generator sets (selection
  seed 777 and reporting seed 4242) and the JevBench public items. The drops are listed
  per source in `manifest.json` and summed under `hygiene`.
- **gen_long_refund_policy at half weight.** `mix.WEIGHT_SCALE` halves its n^0.5 weight.
  Its block keeps its 25% share, so the freed mass goes to the other generators.

Rows dropped by the post-pass (6,585 in total):

| Reason | Rows | Mostly |
|---|---|---|
| near-duplicate of a held-out row | 4,718 | openjev 3,655, certo 939, massive_en 26, massive_fr 22 |
| exact duplicate | 1,780 | decider_teacher 1,166, textdetox_fr 190, pawsx_fr 142, go_emotions 69 |
| prompt over 4096 tokens | 75 | raid 62, plumb 9, mage 3 |
| equal to a held-out row of another source | 12 | clinc_oos 7, massive_en 4 |

mix-v3 against mix-v4 (from the two reports; 3,000 steps × 65,536 tokens):

| | mix-v3 | mix-v4 |
|---|---|---|
| train rows | 456,718 | 450,132 |
| prompt tokens (plain / json) | 132.4M / 148.7M | 128.1M / 144.1M |
| exact duplicate rows | 1,785 | 0 |
| (state, instructions) leaks: validation / calibration / test | 4 / 5 / 2 | 0 / 0 / 0 |
| near items (near state, same instructions + gold): validation / calibration / test | 105 / 106 / 97 | 0 / 0 / 0 |
| near states only: validation / calibration / test | 312 / 336 / 330 | 166 / 182 / 184 |
| hc3_fr `yes` share (train) | 0% | 31% |
| python-repr rows: mghafiri / decider | 1,679 / 79 | 0 / 8 (in the state) |
| dict noul criteria (mghafiri) | 31 | 0 |
| HTML-entity rows | 373 | 35 |
| rows over 4096 tokens | 75 | 0 |
| gen_long_refund_policy: sampled share / token share / epochs | 2.0% / 16.1% / 4.41 | 1.1% / 9.1% / 2.58 |
| segments per step / packing fill | 146.5 / 83.7% | 162.9 / 85.6% |

The remaining near states share a passage but ask another question: BoolQ passages, and
templated certo / openjev states. Those are not near *items*. The eight decider rows have
a Python repr inside their state, as published upstream.

Side effects, and what is deferred:

- **More segments per step.** Shorter prompts give 11% more segments per step, so every
  source is repeated more. plumb is now at 4.2 epochs (it was 3.7), over the report's
  4-epoch flag. The generators are at ~3.6 epochs (they were ~3.1).
- **openjev and certo shrink.** openjev loses 9% of its rows and certo 6%: their
  near-duplicates of held-out rows were most of the held-out leakage.
- **Not rebalanced (deferred).** The yes/no priors stay skewed: raid 97% yes, mlma_en 83%,
  mlma_fr 76%, mage 73%.
- **Packer unchanged (deferred).** Long-bucket steps are still under-filled (T=4096: 14% of
  steps at 72% fill).

### mix-v4b

mix-v4b is mix-v4 with two corrections. It is derived from the built mix-v4, which is not
modified:

```
tjev data derive ~/tjev-work/data/mix-v4 ~/tjev-work/data/mix-v4b \
    --rebalance-noul 0.7 --max-epochs 2.0                          # ~4 min
```

`tjev data build` accepts the same `--rebalance-noul`, `--max-loss`, `--max-epochs` and
`--preserve-blocks` options (`tjev.data.balance`, applied after the hygiene post-pass). Its report is
[`reports/data/mix-v4b/report.md`](../reports/data/mix-v4b/report.md), next to
[mix-v4's](../reports/data/mix-v4/report.md).

Why: the 4B fine-tune on mix-v4 becomes over-confident after ~900 of its 3,000 steps.
Accuracy still rises (80.1% → 81.0%), but confidence goes from 81% to 85% and ECE from
0.021 to 0.041. Only the yes/no items' NLL gets worse (0.308 → 0.337); choice and score
improve. Two causes are suspected, both deferred in mix-v4: skewed yes/no priors and
small sources repeated many times.

- **Yes/no priors.** Some noul sources have a train yes-share outside [0.3, 0.7]. For
  those, a seeded random subsample of the majority class brings the share to 0.7 (or
  0.3). Gold means the argmax of the target, as in the report. Only train files change:
  validation, calibration, heldout-select and test/ are byte-identical copies of mix-v4's,
  so evaluations stay comparable. Kept rows stay in their original order and bytes. A
  source may lose at most 60% of its train rows, counting all its question types
  (`--max-loss`); no source reached that cap.

  | source | train rows | noul rows | yes-share | dropped |
  |---|---|---|---|---|
  | raid | 19,924 → 10,891 | 9,933 → 900 | 97.3% → 70.0% | 9,033 (45% of the source) |
  | mlma_en | 5,032 → 2,913 | 5,032 → 2,913 | 82.6% → 70.0% | 2,119 (42%) |
  | mlma_fr | 3,567 → 2,910 | 3,567 → 2,910 | 75.5% → 70.0% | 657 (18%) |
  | mage | 19,997 → 18,803 | 10,977 → 9,783 | 73.3% → 70.0% | 1,194 (6%) |
  | openjev | 36,312 → 35,969 | 12,640 → 12,297 | 29.2% → 30.0% | 343 `no` (1%) |

  raid keeps its 9,991 choice rows. Only 9% of its noul rows remain, because it had 270
  `no` rows in all.
- **Repetition cap.** No source goes over 2.0 effective epochs at 3,000 steps × 65,536
  tokens. The weights do the work; no rows are dropped. `balance.cap_epochs` runs the
  report's packer simulation (the real `BucketPacker`, moved from
  `scripts/data_report.py` to `tjev.data.balance.simulate_stream`). Each source over the
  cap is pulled to 1.96 epochs.
  - The freed mass stays in the source's block: it goes to the block's uncapped sources,
    in proportion to their weights (`--preserve-blocks`, on by default).
  - When every source of a block is capped, the block cannot absorb the mass. Only then
    does its shortfall go to the other blocks, in proportion to their shares. Such blocks
    are listed under `epoch_cap.overflow` in `manifest.json`.
  - This repeats until nothing is over the cap: 5 rounds, with 21 sources capped in the
    end. The multipliers are stored in `manifest.json` under `epoch_cap.scale`.
    `mix.write_mixture` applies them, so `tjev data reweight` reproduces `mix.yaml`.

  Three blocks end up with every source capped: jev, generators and writing. At 2.0
  epochs they cannot keep their share. Their rows would need about 2.2 (jev), 3.8
  (generators) and 3.2 (writing) epochs to hold 40 / 25 / 5%. Their shortfall (0.039,
  0.120 and 0.019) goes to replay and analysis:

  | block | mix-v4 share | mix-v4b share | capped sources |
  |---|---|---|---|
  | jev | 40.0% | 36.1% | 5 of 5 |
  | generators | 25.0% | 13.0% | 9 of 9 |
  | replay | 20.0% | 31.9% | 2 of 21 (jailbreak, prompt_injection) |
  | analysis | 10.0% | 16.0% | 3 of 10 (mlma_fr, tweet_sentiment_en / fr) |
  | writing | 5.0% | 3.1% | 2 of 2 |

mix-v4 against mix-v4b (from the two reports; 3,000 steps × 65,536 tokens):

| | mix-v4 | mix-v4b |
|---|---|---|
| train rows | 450,132 | 436,786 |
| prompt tokens (plain / json) | 128.1M / 144.1M | 123.0M / 138.7M |
| max effective epochs | 4.24 (plumb) | 1.99 (gen_return_window) |
| noul yes-share: train rows / sampled | 52.0% / 47.7% | 48.5% / 46.6% |
| sampled type share noul / choice / score | 44.0 / 45.0 / 11.1% | 42.4 / 45.3 / 12.3% |
| sampled French share | 20.4% | 22.5% |
| steps by bucket 1024 / 2048 / 4096 | 64.9 / 20.7 / 14.4% | 71.2 / 18.5 / 10.4% |
| segments per step / packing fill | 162.9 / 85.6% | 172.5 / 87.4% |
| plumb: sampled share / epochs | 4.3% / 4.24 | 1.9% / 1.99 |
| gen_long_refund_policy: epochs | 2.58 | 1.97 |

Side effects:

- **Block shares are not preserved at this cap.** Even with `--preserve-blocks`, jev,
  generators and writing lose share to replay and analysis (see the table above). A
  paired study against mix-v4 therefore changes the block mix as well, not only the
  rebalance and the repetition. A cap of about 2.2 epochs keeps jev's share; no cap
  under 3.8 keeps the generators'.
- **Larger sources are capped too.** openjev (1.58 → 1.97) and decider_teacher
  (1.76 → 1.97) reach the cap once their block absorbs plumb's, certo's and mghafiri's
  freed mass. The least repeated source is squad2, at 0.45 epochs.

### mix-v4r

mix-v4r is mix-v4 with the yes/no rebalance only: no epoch cap, and the block shares are
mix-v4's (40 / 25 / 20 / 10 / 5%).

```
tjev data derive ~/tjev-work/data/mix-v4 ~/tjev-work/data/mix-v4r --rebalance-noul 0.7   # ~15 s
```

It separates the two mix-v4b changes in the long-horizon study:

- mix-v4 vs mix-v4r measures the rebalance.
- mix-v4r vs mix-v4b measures the epoch cap, together with the block shift it causes.

Its train files are byte-identical to mix-v4b's (the same seeded rebalance; table in the
mix-v4b section), and its held-out files are mix-v4's. Its report is
[`reports/data/mix-v4r/report.md`](../reports/data/mix-v4r/report.md).

The weights come from the usual n^0.5 rule within each block, applied to the new row
counts. So the rebalanced sources lose a little sampled share inside their block, and
their block-mates gain it: raid goes from 1.45% to 1.18%, mlma_en from 0.73% to 0.61%.
Everything else is as in mix-v4. Max epochs is still 4.24 (plumb), and the generators
are at ~3.7 epochs. Steps by bucket are 1950 / 618 / 432, with 163.2 segments per step.
The noul yes-share is 48.5% by train rows (mix-v4: 52.0%) and 46.6% sampled (mix-v4:
47.7%). The sampled French share is 20.4%.

### mix-v5

mix-v5 targets the French over-confidence of the 4B final on mix-v4 ([fr-data.md](fr-data.md)):
- 6–7 training and 2–3 held-out surface templates per generator × language;
- twice the generator rows (`--gen-scale 2`), seen 2.0× instead of 3.6×;
- human-translated XNLI / PAWS-X dev splits for training, with the machine-translated train
  sets as half-weight train-only sources;
- new French sources (MTOP, PIAF, more Allociné) and one-epoch caps on five French
  sources;
- a 275-row Claude translation pilot of contender rows (`jev_fr_pilot`);
- each source's licence recorded on its card, and `--open-only` to build the
  redistributable subset.

The build, the comparison with mix-v4b and the licence table are in
[fr-data.md §5](fr-data.md#5-mix-v5-as-built), French samples for review in
[fr-samples.md](fr-samples.md), and the report in
[`reports/data/mix-v5/report.md`](../reports/data/mix-v5/report.md). Generator held-out rows
are new, so generator metrics are not comparable with mix-v4 / v4b runs.

### mix-v5b

mix-v5b is mix-v5 with two generators changed, training rows only: gen_return_window trains
on 194 windows (40% from the common 10 / 14 / 30 / 60 / 90 / 180, 60% uniform on 3–200
days; the held-out 7 / 21 / 45 / 120 are never trained), and about 20% of the
gen_return_window and gen_long_refund_policy rows carry an intermediate-quantity line
(`HINT_SHARE` 0.2). The 4B on mix-v5 had learnt a calendar shortcut (accuracy 0.08–0.14 on
the held-out 45-day window at confidence 0.82–0.88). Diagnosis, changes and the comparison
with mix-v5: [fr-data.md §7](fr-data.md#7-mix-v5b-return_window-and-the-long-policy); the
report is [`reports/data/mix-v5b/report.md`](../reports/data/mix-v5b/report.md). Every
held-out file is byte-identical to mix-v5's.

### mix-v6

mix-v6 is mix-v5b with two train-side hygiene checks ([Hygiene](#train-side-decontamination-and-passage-dedup-mix-v6))
and a new `targeted` block of sources ([Sources](#the-targeted-block-mix-v6))
aimed at the weak JevBench families of the mix-v5b 4B (long_policy 0.58, judge_hard 0.59,
temporal_numeric 0.60, ambiguous 0.71, multi_hop 0.78), at over-confident yes/no
abstention, and at French decisions.

```
tjev data build ~/tjev-work/data/mix-v6 --jevbench-public …/jevbench/public.jsonl \
    --gen-scale 2 --max-epochs 2 --rebalance-noul 0.7          # ~47 min on Colab CPU
tjev data heldout ~/tjev-work/data/mix-v6/heldout-select.jsonl
```

- **Where it is.** The private `tychot/tjev-data` repo, under `data/mix-v6`. Data report:
  [`reports/data/mix-v6/report.md`](../reports/data/mix-v6/report.md), with the build log
  and the comparison with mix-v5b (`compare.txt`) next to it.
- **Held-out identity.** `validation.jsonl`, `calibration.jsonl`, `heldout-select.jsonl`
  and the 52 test files of mix-v5b are byte-identical in mix-v6 (checked by sha256). The
  new sources' validation and calibration rows are in their own files,
  `validation-v6.jsonl` and `calibration-v6.jsonl`, and their test rows in
  `test/<source>.jsonl` (11 new files). `mix.yaml` still points at `validation.jsonl`, so
  checkpoint selection is unchanged; evaluate the new files separately.

#### mix-v5b against mix-v6 (3,000 steps × 65,536 tokens)

| | mix-v5b | mix-v6 |
|---|---|---|
| sources | 52 | 63 |
| train rows (French) | 498,555 (108,652) | 569,560 (127,522) |
| sampled French share (report) | 20.5% | 20.9% |
| block shares jev / generators / replay / analysis / writing / targeted | 38.6 / 27.7 / 19.3 / 9.6 / 4.8 / – | 37.0 / 23.0 / 13.0 / 6.5 / 4.0 / 16.5% |
| sampled type share noul / choice / score | 42.1 / 46.8 / 11.1% | 45.5 / 44.1 / 10.4% |
| noul yes-share (train rows) | 48.6% | 48.1% |
| prompt tokens (plain layout, all rows) | 139.3M | 187.5M |
| segments per step / fill | 161.25 / 86.0% | 142.42 / 84.5% |
| steps by bucket 1024 / 2048 / 4096 | 1937 / 514 / 549 | 1799 / 785 / 416 |
| max epochs | 1.97 | 2.00 (plumb) |
| generator epochs | 1.96 | 1.45–1.49 (gen_long_refund_policy 1.02) |
| jev epochs: openjev / decider / certo / mghafiri / plumb | 1.96 each | 1.47 / 1.61 / 1.98 / 1.97 / 2.00 |
| replay / analysis epochs (range) | 0.25–1.94 | 0.15–1.07 |
| held-out files identical to mix-v5b | – | 55 of 55 |

- **Block shares hold.** No block is fully capped, so the epoch cap moves no mass between
  blocks (mix-v5b moved 1.4 points from jev to the generators). Capped sources: certo,
  mghafiri_scenarios, plumb, jev_fr_pilot and the three twins (at 1).
- **Fewer, longer segments.** The targeted block's long rows (contract excerpts,
  conversations, decisions, procedural tables) bring segments per step from 161 to 142
  (−12%) and the 2048-token steps from 514 to 785. Everything else is seen less: the
  generators drop from 1.96 to ~1.46 epochs, the replay and analysis sources by ~40%.
  Halving cuad / argimi_fr / helpsteer3 again would win back part of it.
- **French share.** The manifest's `fr_sampled_share` (20.2%) counts tempreason as English
  (its source language); its 30% French rows bring the measured share to 20.9%.

Per-source rows, shares and epochs of both mixes are in
[`reports/data/mix-v6/compare.txt`](../reports/data/mix-v6/compare.txt).

### mix-v7

mix-v7 is mix-v6 plus 42 new sources, built for 5,000 steps × 65,536 tokens under
`train.packing=best_fit`. mix-v7c is its commercial-only variant.

```
tjev data extend ~/tjev-work/data/mix-v6 ~/tjev-work/data/mix-v7 \
    --jevbench-public …/jevbench/public.jsonl --steps 5000 --packing best_fit --also-steps 3000
tjev data derive ~/tjev-work/data/mix-v7 ~/tjev-work/data/mix-v7c \
    --drop-license-use non-commercial --max-epochs 2 --steps 5000 --packing best_fit
```

All steps (tokenizer, downloads, extend, derive, sampled reports, upload) are in
`scripts/build_mix_v7.sh`, which runs on a Colab CPU runtime. The extend step takes about
2.5 h when the sources are built and about 1.5 h when they are staged in `<out>-stage/`.

- **Where it is.** The private `tychot/tjev-data` repo, under `data/mix-v7` and
  `data/mix-v7c`. Each mix has a `README.md` listing every source with its dataset, rows,
  share, epochs, licence and licence use. In each upload, `mix.yaml` was the last file.
- **Held-out identity.** These files are byte-identical to mix-v6's (checked by sha256):
  - validation, calibration, heldout-select, validation-v6, calibration-v6;
  - the 63 test files.

  The new sources' held-out rows are in `validation-v7.jsonl` (10,110 rows) and
  `calibration-v7.jsonl` (10,057), with their test rows in `test/<source>.jsonl`. These
  rows give per-source held-out CE / ECE for the over-repetition checks. `mix.yaml` still
  points at `validation.jsonl`, so checkpoint selection is unchanged.
- **Licences per row.** Every train row records `license` and `license_use`
  (`commercial` / `non-commercial` / `unspecified`):
  - mix-v6 rows inherit their source's licence (`licences.license_use_of`). XNLI counts as
    non-commercial: its upstream licence is CC BY-NC.
  - tasksource rows keep their own licence.
  - Of the sampled mix, 90.7% is commercial, 2.6% non-commercial and 6.7% unspecified.
  - `tjev data derive --drop-license-use non-commercial[,unspecified]` derives a
    commercial-only mix.

#### Mixture (5,000 steps, best_fit)

| block | share | sources |
|---|---|---|
| jev | 28% | v6 contenders + openjev_x, openjev_ctrl, certo_x, kaivoss_s1, n4ze3m_td, helmo_td, mlagnajit_td |
| targeted | 22% | v6 targeted + procedural_x, eikos_x, tempreason_l23, cold_law_fr_x, argimi_fr_x, cetat_fr, tram, drop_num, tatqa, finqa, temporal_nli, forecasting, sharc, when2call, xlam, xlam_irrelevance, metatool, casehold, ledgar, cxm_fr |
| judge | 9% | helpsteer3 (moved from targeted), helpsteer3_x, oasst2, lmarena_140k, webgpt, summarize_fb, scruples, comparia_arena / votes / reactions, community_alignment_fr, dices, social_chem, mic |
| breadth | 9% | tasksource_jev (tasksource-jev-typed-decisions `full`, filtered) |
| generators | 14% | the v6 generators, unchanged (gen_scale 2) |
| replay | 10% | v6 |
| analysis | 5% | v6 + x_stance_fr |
| writing | 3% | v6 |

- **Weights within a block.** In jev, targeted, judge and breadth, groups of sources take
  the per-source shares of the research plan (`mix.SOURCE_SHARES_V7`; a group's share is
  split ∝ n^0.5). A group whose sources are absent gives its share to the rest of its
  block. The other blocks keep mix-v6's n^0.5 rule, with French targets of 0.45
  (`FR_TARGET_V7`).
- **Epoch cap.** At most 2 epochs at 5,000 steps under best_fit, with the per-source caps
  of v6 (twins at 1).
  - The simulation: 145 segments per step at 97.5% fill (bucket steps 1024 / 2048 / 4096:
    2,579 / 1,450 / 971).
  - The most repeated sources: jev_fr_pilot 1.98, mlagnajit_td 1.94, decider_teacher 1.66,
    certo 1.53, the generators 1.50–1.54.
  - No block overflows, and every source is ≤ 1.25 epochs at 3,000 steps.
- **French.** 19.4% of the sampled mix. jev_fr_mt is a placeholder group at 2.4% of the
  jev block: it receives its share once a source of that name is in the mix (mix-v7.5).
- **The generators hook.** `tjev data extend BASE OUT --import-from GENMIX
  --replace-blocks generators,writing` adds the sources of a separately built mix (e.g.
  `tjev data build GENMIX --blocks generators,writing`) and drops the base generators.
  Imported rows go through the same post-pass, their held-out rows go to the v7 files, and
  the shares and cap are recomputed.

- **At 1,500 steps with the default bucket packing** (the paired TPU comparison against
  mix-v6; data report on 4,000 sampled rows per source):
  - The packer fits 122.5 segments per step at 82% fill (steps 1024 / 2048 / 4096:
    778 / 471 / 251).
  - Every source is under 0.6 epochs: jev_fr_pilot 0.56, mlagnajit_td 0.48,
    decider_teacher 0.42, certo / certo_x and the generators ~0.39. Most of the large new
    sources are under 0.1.

Findings of that report:
- dices' few one-hot rows are all `no`; its signal is in the soft shares.
- Python-repr options appear in 185 rows (when2call, helpsteer3_x, xlam, eikos_x).
- HTML entities appear in 1,246 rows (scruples, drop_num, forecasting).
- 321 compar:IA / lmarena rows are in another language than their tag.
- tempreason_l23 states are 53% near-duplicates (templated facts).
- No train row equals or nearly equals a validation item.

They were fixed in mix-v7.5 (`tjev.data.clean_v75`).

#### mix-v7c (commercial-only)

These leave mix-v7c: the non-commercial rows, i.e. cxm_fr, x_stance_fr, xnli_fr,
xnli_fr_mt and 27,167 tasksource rows. That is 60,682 rows in all, 1,540,511 remaining.
- **Shares.** The block shares are unchanged (the cap redistributes within blocks).
- **Mix.** French is 18.9%. The sampled licence use is 92.6% commercial and 7.4%
  unspecified (webgpt, mnli, sst5, stsb, mtop_fr, the tweets, textdetox_fr, jev_fr_pilot
  and the unspecified tasksource rows). Adding `--drop-license-use
  non-commercial,unspecified` gives a strict variant.
- **Epochs.** At most 1.96 (mlagnajit_td) at 5,000 steps.

#### What is not in mix-v7

- **fiscal_fr.** No public dataset has dated versions of an article: COLD holds only
  articles in force, and the `louisbrulenaudet/code-*` sets only the current version.
- **jev_fr_mt.** Added in mix-v7.5.
- **Below the plan.** Several sources gave fewer rows than planned:
  - certo_x: 14.8k (v6 had used every other rule family);
  - helpsteer3_x: 7.9k (half its rows share 8-grams with PPE);
  - tempreason_l23: 12.8k;
  - xlam: 23.8k;
  - mlagnajit_td: 1.6k (its texts come from our replay sources).
- **Spot checks needed.** when2call (regex-derived labels), metatool (ChatGPT labels),
  cetat_fr (regex on the operative part), TRAM (a benchmark itself: do not evaluate on it
  afterwards), forecasting (outcome not inferable, a prior) and the single-vote preference
  sets.

## Known issues

- **Score questions are the weakest type everywhere** (accuracy ~0.55–0.59, ECE ~0.12). The
  targets are ordinal one-hot, which is noisy for human ratings, and the levels are read
  through letters.
- **Long documents are rare.** The "long" policy documents are ~3.8k tokens at most, so the
  4096 bucket carries few rows. See [Long items](#long-items) for the long-bucket source.
- **Held-out ranges test extrapolation.** Some held-out generator splits test range
  extrapolation (e.g. weekday offsets beyond the training range), and small models score
  low there by design.
- **French over-confidence.** The 4B final on mix-v4 became over-confident on the French
  temporal generators (one template per language, repeated 3.6×; accuracy 0.55 at
  confidence 0.90 on the held-out surface). mix-v5 (templates, `--gen-scale 2`, French
  sources), mix-v5b (windows, hint lines) and mix-v7.5 (jev_fr_mt, the French real-text
  pairs) answer it; the analysis is in [fr-data.md](fr-data.md). French is 22.4% of mix-v7.5.
- **The judge block is single votes.** Its targets are smoothed single votes
  (0.75 / 0.55 / 0.857), a noise model rather than measured agreement.
- **Not built: a teacher-authored block.** An LLM authors JevBench-style items and two
  independent solvers answer them blind; an item is kept when both pick the author's label
  (the recipe of the strongest contenders). Items do carry an optional `teacher`
  distribution (plumb, decider; `loss.teacher_*` in [training.md](training.md) mixes it into
  the target), but no authored block exists.
