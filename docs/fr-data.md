# French data and the 4B over-confidence

Why the 4B final run on mix-v4 (`final-4b-h3000`, W&B `tychota/tjev-sweep/9025cfdc`,
Qwen3.5-4B LoRA r64, 3,000 steps, WSD with a 20% sqrt cooldown) became over-confident on
French (§1–2), the data changes this called for (§3–4), mix-v5 as built from them (§5–6)
and mix-v5b, which fixed the calendar shortcut the 4B learnt on mix-v5 (§7). The later
mixes are in [data.md](data.md): mix-v6 added French law and administration sources, and
mix-v7.5 the machine-translated contender rows (`jev_fr_mt`) and French real-text pairs.

Sources used:

- the run's per-step eval files: `eval/step-*.json` for validation (64 rows per source)
  and `eval/step-*-heldout.json` for heldout-select. They are under
  `~/tjev-work/colab-results/tjev-sweep1-s6/…/runs/final-4b-h{3000,1500}`.
- the run's W&B summary and history (read only).
- the mix-v4 files and `reports/data/mix-v4/report.json`, which gives epochs and shares.

`final-4b-h1500` branches from the h3000 run at step 1200 and cools down over steps
1200–1500. It is the same run with a shorter horizon.

**Method caveat.** The evals log metrics per source, per language and per family, but
never per (source × language). The generators mix English and French. So French generator
numbers are derived: the French total, minus the French-only sources, gives the French
generator part. The temporal split goes one step further and assumes that the
non-temporal families score the same in French and English. Validation and heldout-select
are independent sets, and both give the same picture. The sets are small (58 and 24
French temporal items), so read the numbers as large effects, not as exact values.

## 1. Findings

### Where the French NLL rise comes from

Validation, `final-4b-h3000` (n: 512 rows from French-only sources, 213 French generator rows, 491 English generator rows):

| step | fr sources NLL | fr sources conf−acc | fr generators NLL | fr gen acc | fr gen conf−acc | en generators NLL | en gen acc | en gen conf−acc |
|---|---|---|---|---|---|---|---|---|
| 300 | 0.360 | +0.003 | 0.469 | 0.771 | +0.018 | 0.406 | 0.799 | +0.030 |
| 1200 | 0.334 | +0.030 | 0.518 | 0.783 | +0.080 | 0.389 | 0.824 | +0.054 |
| 2400 | 0.347 | +0.027 | 0.556 | 0.798 | +0.087 | 0.369 | 0.849 | +0.052 |
| 2700 | 0.340 | +0.049 | 0.758 | 0.766 | +0.154 | 0.376 | 0.872 | +0.064 |
| 3000 | 0.331 | +0.048 | **0.833** | 0.789 | **+0.139** | 0.411 | 0.872 | +0.069 |
| h1500 @1500 | 0.323 | +0.023 | 0.533 | 0.762 | +0.100 | 0.384 | 0.838 | +0.038 |

- **The French-only sources do not cause the rise.** Their NLL stays at 0.33 for the
  whole run.
- **French generator items do.** French generator accuracy stays at 0.77–0.79 from step
  300 onwards. English generator accuracy rises from 0.80 to 0.87. Confidence rises in both
  languages.
- **The French temporal items carry it.** These are gen_return_window, gen_weekday and
  gen_schedule_conflict:

| step | val temporal NLL fr / en | heldout temporal NLL fr / en | heldout temporal acc fr / en | heldout temporal conf fr / en |
|---|---|---|---|---|
| 300 | 0.91 / 0.77 | 0.84 / 0.96 | 0.58 / 0.64 | 0.65 / 0.72 |
| 1200 | 1.07 / 0.67 | 0.91 / 0.72 | 0.69 / 0.74 | 0.69 / 0.76 |
| 1800 | 1.17 / 0.55 | 1.17 / 0.69 | 0.55 / 0.75 | 0.68 / 0.78 |
| 2400 | 1.22 / 0.60 | 1.70 / 0.59 | 0.43 / 0.83 | 0.81 / 0.91 |
| 3000 | **2.15** / 0.68 | **2.16** / 0.92 | **0.55** / 0.85 | **0.90** / 0.98 |
| h1500 @1500 | 1.18 / 0.68 | 1.03 / 0.83 | 0.54 / 0.69 | 0.70 / 0.75 |

From 2400 to 3000, validation French NLL gains +51 nats in all (0.070 × 725). The
French temporal rows alone gain about +54 nats (58 rows, 1.22 → 2.15). The French-only
sources lose about 8. So **the late French rise is almost entirely the French temporal
generators**, on their held-out surface: French accuracy stays near 0.55 while French
confidence goes from 0.65 to 0.90. English temporal accuracy improves from 0.64 to 0.85.
In the h1500 branch (same checkpoint at 1200, cooled down 1200 steps earlier) French
temporal NLL stays at about 1.1.

**Composition does not explain the held-out gap.** heldout-select is 49% French
(`fr_share=0.5`) and is not balanced per generator. French has 15 base_rate items against
5 English, and 24 temporal items against 36. Weight the per-family NLL of step 3000 by each
language's family counts: French would then score 0.54 and English 0.62. The observed
values are 0.71 and 0.47. French is therefore worse *within* families.

### French in mix-v4

French is 76,832 rows, 20.4% of the sampled mix. The jev block (40% of the sampled mix) is
English only. Epochs at 1,500 steps are half those at 3,000: the stream is the same prefix,
at 163 segments per step.

| source | kind | fr rows | share of mix | share of French | epochs @3000 | @1500 | seen @3000 |
|---|---|---|---|---|---|---|---|
| xnli_fr | MT of MNLI, Moses-tokenised (train) / human (test) | 15,000 | 2.23% | 10.9% | 0.73 | 0.36 | 51% |
| massive_fr | human-localised (MASSIVE) | 11,388 | 1.94% | 9.5% | 0.83 | 0.42 | 56% |
| pawsx_fr | MT (train) / human (test) | 9,858 | 1.81% | 8.9% | 0.91 | 0.45 | 60% |
| allocine_fr | native reviews | 7,995 | 1.63% | 8.0% | 0.99 | 0.50 | 63% |
| hc3_fr | MT of HC3 QA (both splits) | 9,956 | 1.31% | 6.4% | 0.63 | 0.32 | 47% |
| textdetox_fr | native comments | 4,260 | 0.85% | 4.2% | 0.98 | 0.49 | 63% |
| mlma_fr | native tweets, keyword-sampled | 3,567 | 0.78% | 3.8% | 1.11 | 0.55 | 66% |
| tweet_sentiment_fr | native tweets | 1,804 | 0.56% | 2.7% | 1.47 | 0.73 | 77% |
| 9 generators | templated French surface (30% of each) | 10,310 | 7.6% | 37.3% | 3.6–3.7 (long policy 2.6) | 1.8 | 97% |
| 2 writing generators | templated French surface | 2,694 | 1.7% | 8.2% | 3.0–3.1 | 1.5 | 95% |

French is 45% generator rows (13,004 rows, repeated 3.6×) and 55% public classification
data (repeated 0.6–1.5×). By family, French has no rows from the contenders' decision data
(the jev block). That block holds about 15k English rows with dates, deadlines or
day-counting in varied wording (decider 22%, mghafiri 33%, plumb 79% of their rows).

### Generator templates

The same counts hold in English. A generator draws its language per item and then uses
one fixed template per (split, language).

| generator (fr) | train rows | train questions | distinct state frames (numbers / dates masked) | validation frames seen in train | epochs |
|---|---|---|---|---|---|
| gen_return_window | 1,190 | 1 | 1 (2 lines) | 0% | 3.71 |
| gen_weekday | 1,210 | 1 | 1 | 0% | 3.65 |
| gen_schedule_conflict | 1,200 | 1 | 4 (3 lines) | 33% of lines | 3.63 |
| gen_base_rate | 1,131 | 5 | 5 (1 line) | 0% | 3.62 |
| gen_invoice_total | 1,233 | 1 | 10 lines | 0% | 3.66 |
| gen_table_argmax | 1,211 | 6 | 12 lines | 8% | 3.67 |
| gen_refund_policy | 1,190 | 1 | 19 lines | 5% | 3.65 |
| gen_ticket_triage / spelling / register | 1.1–1.4k | 1 | ~1,100–1,250 (sentence pools) | 0% | 3.0–3.6 |

French gen_return_window repeats one two-line frame about 4,400 times (1,190 rows ×
3.7). The held-out frame differs, and so do the parameters. return_window holds out its
windows (7 / 21 / 45 / 120 days), and weekday draws half its offsets beyond the training
range. Those are exactly the families that fail in French.

### Labels, types and lengths: French against English

| French source | yes-share train / val | English counterpart | yes-share train / val | notes |
|---|---|---|---|---|
| xnli_fr | 34.0% / 31.1% | mnli | 34.3% / 36.6% | the same 50/50 noul/choice split; choice 3 options, balanced |
| pawsx_fr | 43.3% / 42.7% | paws | 44.1% / 51.3% | |
| allocine_fr | 50.5% / 48.7% | — | | |
| textdetox_fr | 49.6% / 46.3% | — | | |
| mlma_fr | 75.5% / 77.5% (7% soft) | mlma_en | 82.6% / 79.1% | keyword-sampled tweets; e.g. "barbecue mongol" is labelled offensive |
| hc3_fr | **31.2% / 53.1%** | mage / raid | 73% / 97% | train / held-out prior shift; accuracy is still 0.98 |
| French generators | 43–54% | the same generators in English | 42–54% | French priors are within 3 points of English |

Choice option counts are the same as in English: massive 8, tweet 3, xnli 3, generators
4–7. French score items come only from gen_register (3 balanced levels). French prompts are
5–15% longer in tokens than their English counterparts, for example xnli 174 against
mnli 142 and massive 208 against 187. Every French row fits the 1024 bucket except 1.3%
of hc3_fr.

### Translation quality and artefacts

- **xnli_fr train is machine translation.** 97.8% of the rows have Moses-style
  tokenisation (`J' ai`, ` .`, ` ,`); 0% of the validation rows do and 0.7% of the test
  rows do. The validation and test rows come from the human-translated XNLI test.
  Samples show MT damage that can break the label, for example "l' humour du côlon"
  (colon humour) and "mouvements de viande séchée" (jerky movements). The model therefore
  trains on one register and is evaluated on another.
- **pawsx_fr train is machine translation, but detokenised.** Translation sometimes
  undoes the word-order swaps PAWS is built on, or makes up a difference. For example, a
  pair labelled `yes` reads "manoir que Walter Sullivan a cédé" against "pénétré par
  effraction". About 1–2 of the 10 sampled pairs were damaged. pawsx_fr's
  confidence − accuracy is +0.06, against −0.00 for paws, at equal confidence.
- **hc3_fr is machine-translated in both splits.** There is no shift between train and
  held-out. The source is easy (accuracy 0.98, confidence 0.995).
- **Near-duplicates** in the public French sources are low (report MinHash J ≥ 0.8):
  pawsx_fr 6.0%, tweet_sentiment_fr 1.5%, mlma_fr 1.1%, the others under 0.5%. MinHash
  misses templating, because numbers change every 3-gram. The frame counts above are the
  right measure for the generators.

### French-only sources at step 3000 (validation, n = 64 each)

| source | acc | conf | conf − acc | NLL 1200 → 3000 | English counterpart conf − acc |
|---|---|---|---|---|---|
| tweet_sentiment_fr | 0.61 | 0.80 | **+0.19** | 0.88 → **1.00** | tweet_sentiment_en +0.06 (NLL 0.66 → 0.59) |
| pawsx_fr | 0.86 | 0.92 | +0.06 | 0.41 → 0.35 | paws −0.00 |
| massive_fr | 0.91 | 0.96 | +0.05 | 0.22 → 0.22 | massive_en +0.03 |
| mlma_fr | 0.82 | 0.86 | +0.05 | 0.44 → 0.45 | mlma_en +0.06 |
| xnli_fr | 0.86 | 0.91 | +0.05 | 0.46 → 0.47 | mnli +0.04 |
| allocine / hc3 / textdetox | 0.97–0.98 | 0.95–0.99 | −0.02 … +0.01 | flat or down | — |

## 2. Hypotheses, ranked

1. **Strong: French temporal generators learn a template-bound rule that confidence
   outruns.** Each (generator, language) has one training frame, and the held-out frame
   and parameter ranges differ. On held-out data, French temporal accuracy stays at about
   0.55 while confidence rises to 0.90 with the run's overall sharpening. This explains
   about 100% of the late French validation NLL rise (+54 of +51 nats), and the held-out
   set shows the same.
2. **Strong, same mechanism: repetition.** Generators are at 3.6 epochs at step 3,000
   and 1.8 at step 1,500. The h1500 cooldown starts from the same step-1200 checkpoint, and
   its French temporal NLL stays at about 1.1, against 2.15 at h3000. French temporal NLL
   already climbs before the h3000 cooldown (1.07 → 1.22 on validation; 0.91 → 1.70 on
   held-out, 1200 → 2400), and the cooldown amplifies it. This is confounded with total
   training. The clean test is `--gen-scale 2` at 3,000 steps (§3).
3. **Moderate: no French decision data, so skills are learned in English.** English
   temporal items have about 15k varied English jev rows behind them. French has only the
   single generator frame. English temporal accuracy rises from 0.64 to 0.85 on held-out
   data; French stays flat. Confidence carries across languages and accuracy does not.
4. **Weak to moderate: MT training text with label noise and a style shift.** In xnli_fr,
   the train text is Moses-tokenised and the held-out text is human. pawsx_fr shows MT
   label damage. Together they explain the native-source gap of +0.05 (pawsx +0.06 against
   paws 0.00), but not the NLL rise, because the NLL of these sources is flat.
5. **Weak: a small, noisy native source repeated.** tweet_sentiment_fr has 1.8k rows
   seen 1.47×. Its NLL goes from 0.85 to 1.00 and its accuracy from 0.67 to 0.61, so it
   memorises. Its share is small (0.56% of the mix).
6. **Rejected: skewed French label priors.** French yes-shares and option counts match
   English. The exception, hc3_fr (31% → 53%), is at 0.98 accuracy.
7. **Rejected: French held-out items are harder by composition.** Weighting by family
   predicts *lower* French NLL than English (0.54 against 0.62 observed for English). The
   gap is within families. Note that heldout-select has only 5–15 items per (generator ×
   language).

## 3. mix-v5 proposal

Ordered by expected effect per cost. The 20% French share stays: JevBench public is 100%
English, so the share is a product choice. Also note that **heldout-select is 49% French
and half the checkpoint-selection score**, so French temporal failures pull the selection
towards early checkpoints. The best step was 1200.

| # | change | cost | expected effect |
|---|---|---|---|
| 1 | **Generator surface variety.** 6–8 training frames per (generator, language), covering state wording, question and criteria, drawn per item. Held-out frames stay disjoint; extend `test_heldout_generators_use_their_own_surface_text`. | ~150 template sentences (fr + en), no new rows; generator code only | Removes the one-frame shortcut. French held-out temporal accuracy should move towards English (0.55 → ≳ 0.75) and its NLL from 2.2 to below 1. Most of the late +0.07 French validation NLL should disappear. |
| 2 | **`--gen-scale 2` (implemented).** 8,000 rows per generator (long policy 4,000). | +42k generated rows (generators + writing 42k → 84k), free; token and block shares unchanged | Generators drop from 3.6 to 1.8 epochs at 3,000 steps, and each row is seen as often as in h1500. With `--max-epochs 2`, the generators keep their 25% share (in mix-v4b they fell to 13%). Alone it does not add surface variety, so pair it with #1. |
| 3 | **French decision rows.** Translate ~15% of the jev block (~15k rows) to French, first the date / deadline / policy rows of decider, mghafiri and plumb. Check numbers, dates and options automatically. This is planned in `docs/data.md`. | ~15k LLM translations (~5–6M tokens); licences MIT / Apache / CC0 allow it | Gives French the varied support that English temporal and policy items have (hypothesis 3). To hold 20% French, lower `FR_TARGET` (replay 0.38 → ~0.25). |
| 4 | **XNLI / PAWS-X: human text for training.** Train on the human-translated validation splits (xnli fr 2,490 rows, paws-x fr 2,000) and keep test held out. Detokenise the MT train (` ,` → `,`, `l' ` → `l'`) and halve its weight. | Adapter change, a few k rows | Removes the style shift and part of the label noise; native-source conf − acc +0.05 → ~+0.03. |
| 5 | **tweet_sentiment_fr ≤ 1 epoch** (and mlma_fr ≤ 1). | weight only (`--max-epochs`, or a per-source cap) | Stops the memorisation of noisy labels; small effect overall. |
| 6 | **hc3_fr prior.** Bring train to ~45% yes. | `--rebalance-noul 0.65` touches it only marginally; a per-source target is needed | Removes the 31% / 53% shift; negligible NLL effect. |
| 7 | **Evaluation, so this can be measured.** Log (source × language) groups in `EvalSet.run` (`meta["source_lang"]`). Raise heldout-select to 40 per generator (about 20 per language). | eval code; +220 eval items | Per-generator French numbers without inference; a less noisy French selection signal. |

What was done: #1, #2, #4 and #5 in mix-v5 (§5); #3 first as the 275-row `jev_fr_pilot`
(§5) and then at scale as `jev_fr_mt` in mix-v7.5 (28k machine-translated rows gated on
numbers, entities, a round trip and CometKiwi; [data.md](data.md#jev_fr_mt)). #6 was not
done as a per-source target (hc3_fr is at 0.98 accuracy); the evaluation of #7 is covered
by the per-source held-out sets of the later mixes (`tjev post --extra-validation`).

**Cost of the core of mix-v5 (#1 + #2 + #4):** no external data. Generator code (#1),
+42k generated rows, a few thousand public rows. Re-run the data report to check that
generators are under 2 epochs and that French stays at about 20%.

**Quick test without a full rebuild (#2 only):**

```
tjev data build /tmp/gen2 --blocks generators,writing --gen-scale 2
```

Then copy its `train/gen_*.jsonl` files into a copy of mix-v4 and run
`tjev data reweight` on that copy.

- **Held-out files do not change.** Generator held-out rows come from their own seeds,
  and the first 4,000 train rows are mix-v4's (prefix property, tested).
- **Weights do not change either.** Every generator doubles, so the n^0.5 weights within
  the block stay the same.

## 4. Data and calibration (per-language temperatures)

- **What calibration fixes.** It fixes a uniform scaling of the logits: the general
  cooldown sharpening seen in both languages (conf − acc +0.05 → +0.07 on English
  generators as well).
- **What a single `T_fr` cannot do.** French miscalibration is concentrated in about 8% of
  the French validation items, the temporal generators (conf 0.90 against acc 0.55). The
  rest of French is at +0.03…+0.05, like English. A `T_fr` fitted on the calibration split
  is dominated by those items, so it over-softens the healthy French rows and still
  under-corrects the temporal ones. Its value also depends on how many French temporal
  rows the calibration split happens to hold.
- **Calibration cannot restore accuracy.** French temporal accuracy (0.55 against 0.85)
  is a transfer failure; only data fixes it (§3, #1–#3).
- **A diagnostic.** After mix-v5, fit `T_en` and `T_fr` again. If the data works,
  `T_fr − T_en` shrinks towards 0 and the per-language temperatures become unnecessary. A
  `T_fr` that is clearly above `T_en` on a mix-v5 run would point to a remaining data
  problem.
- **JevBench.** The public items are English only (231 of 231). `T_fr` therefore does not
  change the JevBench score; it changes French product behaviour and the selection score
  (heldout-select is 49% French).

## 5. mix-v5 as built

```
tjev data build ~/tjev-work/data/mix-v5-build --jevbench-public …/jevbench/public.jsonl \
    --gen-scale 2 --max-epochs 2 --rebalance-noul 0.7          # ~40 min on Colab CPU
tjev data derive ~/tjev-work/data/mix-v5-build ~/tjev-work/data/mix-v5 --max-epochs 2
tjev data heldout ~/tjev-work/data/mix-v5/heldout-select.jsonl
```

- **The derive pass.** It re-runs only the epoch cap, with the per-source caps added after
  the build (§6). Its train files, held-out files and weights are what a single build with
  the final code gives.
- **The noul rebalance.** `--rebalance-noul 0.7` is mix-v4b's yes/no rebalance, kept so
  that mix-v5 is comparable with mix-v4b.
- **Where it is.** The mix is uploaded to the private `tychot/tjev-data` repo under
  `data/mix-v5` (109 files). Its data report is
  [`reports/data/mix-v5/report.md`](../reports/data/mix-v5/report.md), with the build and
  derive logs next to it.

### What changed

| change | where | detail |
|---|---|---|
| Generator templates | `sources/generators.py` | 6–7 training and 2–3 held-out surface variants per (generator, language), covering state wording, question, options and date / time format. A separate *surface* rng draws the template, so for a given seed every item keeps mix-v4's parameters and label. Held-out templates share no template line with training (existing test). |
| French conventions | generators | "1er", "lundi 3 mars 2025", 03/03/2025, 24h times "14h30" / "14 h 30", J+n, comma decimals, a space before ":" (mix-v4's invoice lines lacked it). |
| More generator rows | `--gen-scale 2` | 8,000 rows per generator (long policy 4,000). |
| Human-translated XNLI / PAWS-X | `xnli_fr`, `pawsx_fr` | They now train on the dev splits (2,490 and 1,965 rows). Their held-out rows (test split) are identical to mix-v4's. |
| Machine-translated train kept at half weight | `xnli_fr_mt` (detokenised), `pawsx_fr_mt` | Train-only sources (`Source.train_only`); `WEIGHT_SCALE` 0.5. |
| New native French data | `mtop_fr`, `piaf_fr`, `allocine_fr` | MTOP intents (110, with French menu text), PIAF reading (answerable, or paired with another paragraph of the same article = unanswerable; held out by article), Allociné cap 8k → 16k. |
| Per-source epoch caps | `mix.SOURCE_MAX_EPOCHS` | 1 epoch for tweet_sentiment_fr, mlma_fr, hc3_fr, textdetox_fr and allocine_fr (`balance.cap_epochs(caps=…)`). |
| Translation pilot | `jev_fr_pilot` | 275 contender rows translated to French (see below). |
| Licences | `tjev.data.licences`, manifest | Each source's card licence and redistributable status are stored in `manifest.json`. `--open-only` builds the redistributable subset. |

**Not added: FQuAD.** The official repository (`illuin/fquad`) is not reachable (404 with
the user's token). The copies on the Hub either have no licence on their card
(`kheopsai/f_quad`, `zanellac/fquad-partages`), contain only a 100-row test sample
(`manu/fquad2_test`, tagged Apache-2.0 although FQuAD's own terms differ), or are prompt
reformulations. PIAF covers the same task.

### mix-v4b against mix-v5 (3,000 steps × 65,536 tokens)

| | mix-v4b | mix-v5 |
|---|---|---|
| train rows (French) | 436,786 (76,175) | 498,555 (108,680) |
| sampled French share | 22.5% | 20.5% |
| block shares jev / generators / replay / analysis / writing | 36.1 / 13.0 / 31.9 / 16.0 / 3.1% | 38.6 / 27.7 / 19.3 / 9.6 / 4.8% |
| generator epochs | 1.97 | 1.97 (twice the rows, twice the share) |
| French share from generators | 22% | 49% |
| rows per French generator template, × epochs | ~1,200 × 2.0 (one template) | ~370 × 2.0 (6–7 templates) |
| max epochs | 1.99 | 1.97 |
| sampled type share noul / choice / score | 42.4 / 45.3 / 12.3% | 42.1 / 46.8 / 11.1% |
| segments per step / fill | 172.5 / 87.4% | 161.3 / 86.0% |
| steps by bucket 1024 / 2048 / 4096 | 2135 / 554 / 311 | 1937 / 514 / 549 |
| gen_long_refund_policy: segments / prompt tokens | 0.8% / 6.8% | 1.6% / 13.9% |

- **Block shares.** The jev and generator blocks are fully capped at 2 epochs. In mix-v4b
  the generators fell to 13%; with twice the rows they now keep (and slightly exceed) their
  25%, because they absorb part of jev's shortfall.
- **Side effect: long documents.** The long-policy generator doubles its share, so 18% of
  the steps are 4096-token steps (mix-v4b: 10%) and there are 6% fewer segments per step.
  Halving its `WEIGHT_SCALE` again (0.5 → 0.25) would undo this. It is not done here.
- **Held-out comparability.** The held-out rows of every non-generator source are
  identical to mix-v4's (validation and calibration checked). The generators' validation,
  calibration, test and heldout-select rows are new: they use the new held-out templates.
  Generator metrics are therefore not comparable with mix-v4 / v4b runs; per-source metrics
  of the other sources are.

### French in mix-v5

| source | fr rows | share of mix | share of French | epochs @3000 |
|---|---|---|---|---|
| allocine_fr | 15,985 | 1.58% | 7.7% | 0.48 |
| mtop_fr | 11,503 | 1.34% | 6.5% | 0.57 |
| massive_fr | 11,384 | 1.33% | 6.5% | 0.57 |
| hc3_fr | 9,956 | 1.33% | 6.5% | 0.64 |
| 9 generators | 20,661 | 8.41% | 41.1% | 1.96–1.97 |
| 2 writing generators | 5,079 | 1.53% | 7.5% | 1.45–1.47 |
| textdetox_fr | 4,260 | 0.88% | 4.3% | 0.98 |
| piaf_fr | 3,553 | 0.74% | 3.6% | 1.03 |
| xnli_fr (human) / xnli_fr_mt | 2,490 / 10,000 | 0.62 / 0.62% | 3.0 / 3.0% | 1.20 / 0.30 |
| pawsx_fr (human) / pawsx_fr_mt | 1,965 / 6,930 | 0.55 / 0.52% | 2.7 / 2.5% | 1.39 / 0.36 |
| mlma_fr | 2,910 | 0.58% | 2.8% | 0.99 |
| tweet_sentiment_fr | 1,804 | 0.36% | 1.8% | 0.98 |
| jev_fr_pilot | 200 | 0.08% | 0.4% | 1.91 |

### Translation pilot (`jev_fr_pilot`)

- **Selection.** `scripts/jev_fr_pilot.py select` picks rows with temporal or policy
  wording (≤ 160 words, no English-specific task such as spelling or word play) from
  certo, mghafiri, openjev and decider, round-robin over families.
  - Pilot train rows come from the English *train* files.
  - Pilot held-out rows come from the English *held-out* files, so no French held-out row
    translates a row the model trains on.
- **Translation.** Three Claude subagents translated the rows by hand. They followed one
  brief:
  - Write natural French; do not add or remove information.
  - Keep label ids, codes, JSON keys and identifiers unchanged.
  - Localise prose dates and numbers, 24h times, "399 $" (no currency conversion).
  - Drop a row when its label depends on English wording.
- **Checks (`assemble`).**
  - The type, label ids or levels, and target are the original's.
  - Every number of the English state is still present after localisation: amounts,
    dates as integer parts, and 12h times converted to 24h.
  - The text is French.
- **Result.** 276 selected → 275 kept (certo 75, mghafiri 80, openjev 60, decider 60).
  - 200 rows train and 75 are held out (validation 20, calibration 25, the rest test).
  - One row was dropped by review: its policy matches English gendered words literally.
  - Two rows were first rejected by the number check (12h times). The check now converts
    them to 24h.
- **Weight.** At 200 rows the pilot is capped at 2 epochs: 0.08% of the mix. Its effect
  is meant to be read on its own held-out rows (French contender decisions, which mix-v4
  lacked), not on the totals.
- **Scaling up.** Run `select --train N --heldout M` on more rows, translate in chunks with
  the same brief, then `assemble`.
- **Licence.** Claude-generated rows are kept separate (`redistributable:
  claude-generated`, `commercial_ok: false`), so they can be left out of an open release.

### Licences (from the dataset cards, read 2026-10-01)

`redistributable` is one of: yes, share-alike, NC, no, unknown, claude-generated.
`tjev data build --open-only` keeps only *yes* and *share-alike* sources. "unknown" means
that the card states no licence, says "other" / "unknown", or adds terms that restrict
redistribution.

| source | dataset | card licence | redistributable |
|---|---|---|---|
| allocine_fr | tblard/allocine | MIT | yes |
| banking77 | legacy-datasets/banking77 | CC BY 4.0 | yes |
| boolq | google/boolq | CC BY-SA 3.0 | share-alike |
| certo | altslate/certo-decisions-v2 | MIT | yes |
| clinc_oos | clinc/clinc_oos | CC BY 3.0 | yes |
| dbpedia | fancyzhx/dbpedia_14 | CC BY-SA 3.0 | share-alike |
| decider_teacher | github.com/Mapika/decider | Apache-2.0 (LICENSE) | yes |
| gen_* (11) | tjev's own generators | Apache-2.0 | yes |
| go_emotions | google-research-datasets/go_emotions | Apache-2.0 | yes |
| gsm8k_check | openai/gsm8k | MIT | yes |
| hc3_fr | almanach/hc3_french_ood | CC BY-SA 4.0 | share-alike |
| helpsteer2_* | nvidia/HelpSteer2 | CC BY 4.0 | yes |
| jailbreak | jackhhao/jailbreak-classification | Apache-2.0 | yes |
| jev_fr_pilot | Claude translations of certo / decider / mghafiri / openjev rows | — | **claude-generated** (keep separable) |
| mage | yaful/MAGE | Apache-2.0 | yes |
| massive_en / massive_fr | mteb/amazon_massive_intent | Apache-2.0 (original AmazonScience/massive: CC BY 4.0) | yes |
| mghafiri_scenarios | mghafiri/decision-model-scenarios-v2 | MIT (Claude-written upstream) | yes |
| mlma_en / mlma_fr | nedjmaou/MLMA_hate_speech | MIT (tweets) | yes |
| mnli | nyu-mll/glue | "other": refers to each task's licence | unknown |
| mtop_fr | mteb/mtop_intent | "unknown" | **unknown** |
| openjev | ZefanCai/Open-Jev-v1.1 | CC0 1.0 + CC BY 4.0 (WANLI) | yes |
| pavlick_formality | osyvokon/pavlick-formality-scores | CC BY 3.0 | yes |
| paws / pawsx_fr / pawsx_fr_mt | google-research-datasets/paws(-x) | "may be freely used for any purpose" | yes |
| piaf_fr | etalab-ia/piaf | MIT (tag; licensing section empty) | yes |
| plumb | crh225/plumb-decisions | Apache-2.0 | yes |
| prompt_injection | deepset/prompt-injections | Apache-2.0 | yes |
| raid | liamdugan/raid | MIT | yes |
| snli | stanfordnlp/snli | CC BY-SA 4.0 | share-alike |
| squad2 | rajpurkar/squad_v2 | CC BY-SA 4.0 | share-alike |
| sst5 | SetFit/sst5 | none on the card | unknown |
| stsb | sentence-transformers/stsb | none on the card | unknown |
| textdetox_fr | textdetox/multilingual_toxicity_dataset | openrail++ (use restrictions) | unknown |
| tweet_sentiment_en / _fr | cardiffnlp/tweet_sentiment_multilingual | CC BY 3.0 + Twitter ToS | unknown |
| vitaminc | tals/vitaminc | CC BY-SA 3.0 | share-alike |
| xnli_fr / xnli_fr_mt | facebook/xnli | none ("more information needed") | **unknown** |

**Licences I could not verify:**
- **XNLI.** The card states no licence.
- **MTOP.** The mteb card says "unknown".
- **sst5, stsb and MNLI.** Their cards state none.
- **textdetox.** Its card licence is openrail++, which has use restrictions.
- **The tweet datasets.** They also require the Twitter terms of service.
- **FQuAD.** Not used (see above).
- **The pilot.** These are Claude outputs. Check the terms of use for Claude outputs
  before any open release of models trained on them.

## 6. Inputs from the calibration study (docs/calibration.md)

- **(a) gen_return_window and gen_invoice_total are over-confident in and out of
  distribution.** I looked for label and template defects.
  - **Labels.** I recomputed every label independently from the rendered text, over all
    new templates and both languages: dates parsed back from the state, invoice lines,
    discount and stated total re-read and summed. 6,000 rows each, 0 mismatches.
  - **Templates.** The defect was the template surface: one frame per language, repeated
    ~4,400 times in mix-v4. The tasks themselves are exact date and cent arithmetic, which
    a 4B model without reasoning cannot do reliably. Training rows repeated 3.6× are
    memorised (confident), and the confidence carries over to items it cannot compute.
  - **mix-v5.** It has 6–7 frames per language, rows seen 2.0× (not 3.6×), and three
    date / number formats. A further option, not done here, is soft targets for the
    near-boundary cases (`|elapsed − window| ≤ 1` and ±1 $ invoice errors).
- **(b) Easy French noul sources are memorised; French choice is over-confident (mostly
  tweet_sentiment_fr).**
  - **Caps.** hc3_fr, textdetox_fr and allocine_fr are capped at 1 epoch (0.64 / 0.98 /
    0.48 in mix-v5, against 1.14 / 1.80 / 1.69 in mix-v4b). tweet_sentiment_fr and
    mlma_fr are also at ≤ 1.
  - **Harder French choice data added:** mtop_fr (110-intent menus), the French
    generators' choice items (weekday, table, triage, refund; now varied) and the pilot.
- **(c) Keep the French share constant through the cooldown.** It already is constant.
  - **Why.** The stream draws each segment's source from fixed mixture weights
    (`tjev.data.pipeline`, deterministic per segment index). WSD changes only the
    learning rate, so the expected French share is 20.5% at every step.
  - **What varies.** Only the per-step share fluctuates: steps use one bucket, and French
    long-policy rows fall in 4096-token steps.
  - **No sampler change needed.** A schedule that *changed* the share during the cooldown
    (e.g. less French) would need one; nothing here needs it.

## 7. mix-v5b: return_window and the long policy

The 4B on mix-v5 (sweep-reg-4b-centre-s0, 1,500 steps) gets more confident during the
cooldown on two generators. gen_return_window validation NLL goes 1.69 → 2.45 (step 1100 →
1500) at accuracy 0.52 and confidence 0.94; gen_long_refund_policy goes 0.09 → 0.30. On the
held-out generators, long_policy is +0.71 NLL and temporal +0.22. The French held-out NLL
(.537 → .755) comes from these rows. The labels are right (§6: 0 mismatches).

### Diagnosis

The step-N eval files only have aggregates. The per-item 4B predictions of the mix-v4 runs
(`final-4b-h1500@1450`, `final-4b-h3000@1200`, `@3000`, `post/predictions`) were joined
with the regenerated item parameters. The same seed gives the same parameters and labels,
which was checked on every row.

**return_window: a calendar shortcut, not the boundary.**

| held-out window | rows | accuracy | confidence |
|---|---|---|---|
| 7 | 47 | 0.89–0.91 | 0.93–0.97 |
| 21 | 37 | 0.49–0.54 (answers "no" almost always) | 0.77–0.90 |
| 45 | 37 | **0.08–0.14** | 0.82–0.88 |
| 120 | 29 | 0.79–0.83 | 0.85–0.90 |

(validation rows, h3000 @1200 and @3000)

- **Not a boundary effect.** Rows far inside the window (elapsed ≤ window − 6) have accuracy
  0.25–0.33. The |elapsed − window| ≤ 1 rows sit at about chance, like the rest.
- **The 45-day rows.** The model answers "late" when the day of the month has passed since
  the purchase, and "in time" when it has not. For example, elapsed 37, bought on the 20th,
  request on the 26th: P(yes) = 0.01.
- **Why.** Four of the six training windows (30/60/90/180) are whole months. On those,
  "compare the month count, then the day of the month" decides 90–97% of the training rows
  without counting days (80% of all training rows). The 10- and 14-day windows teach
  "the month changed → late". The held-out windows 45 and 21 are where both shortcuts fail.
  The "{e} days ago" template, the only one that states the count, was 3% of the training
  rows: the generator's key dedup collapses it.

**long_refund_policy: the member bonus.**

- Rows where the member bonus decides (window < days ≤ window + bonus) have about 20%
  error, against 2–3% for the other rows (mix-v4 4B, 150 validation rows). Rows within
  5 days of the allowed period: accuracy 0.76.
- The distractor windows of other categories do not explain the errors. 4 of the 7 errors
  had a distractor window that could flip the decision, against 56% of all rows.
- The 4 mix-v5 held-out errors (20 rows, confidence 0.98) cannot be attributed without
  predictions. That set has no bonus-decided rows; it has amounts at 1.01× the limit,
  near-period rows and broken-seal non-medical cases. These samples are small (20 / 64).

### What changed (training rows only)

| change | detail |
|---|---|
| Training windows | 40% from the common windows 10/14/30/60/90/180; 60% uniform on 3–200 days. The held-out windows 7/21/45/120 are never trained. The month shortcut now decides 72% of the training rows (was 80%), 50% on 40–50-day windows. 194 distinct windows. |
| Intermediate quantities (`HINT_SHARE` 0.2) | About 20% of the training rows of the two generators get one more line with an exact quantity. return_window: "Days elapsed: N" / "Jours écoulés : N", or "Last day of the window: <date>" / "Dernier jour du délai : <date>". Long policy: "Case note: the window for X is in Section N; this customer is a member, so the refund period is 14 + 30 = 44 days" (FR likewise). Rows are chosen by a hash of the state, so parameters, template and label are those of the row without the line. Held-out rows never get one. |
| Ambiguous wording | The EN training template "(the receipt date counts)" could read "the receipt day is day 1", which would flip the label at elapsed = window. It now says "(the date on the receipt is the purchase date)". |

**Not changed.**
- **Soft targets.** No other wording is ambiguous, and the errors are not at the boundary.
- **Weights and epoch caps.** At 1,500 steps both sources are seen about once (return_window
  3.2% of segments over 8,000 rows), and the confidence grows within that first epoch.
- **Contrastive twins, auxiliary bucket questions, negation twins.** Left for later; the
  negation twins came in mix-v6 (`twin_*`), the contrastive pairs and auxiliary questions
  with the v7 generators in mix-v7.5 ([data.md](data.md#generators-v7)).
- **Near-duplicate held-out states (boolq, squad2, piaf).** These are shared passages with
  different questions (`near_item` = 0), not item leakage. Removing them would change the
  held-out sets.

### Held-out comparability

- **Generators.** Every generator's validation, calibration, test, heldout-select and
  report rows are byte-identical to mix-v5's (`test_heldout_generator_rows_are_unchanged_since_mix_v5`).
- **Training hashes.** They change only for gen_return_window and gen_long_refund_policy.
- **Evals.** mix-v5b evals compare directly with mix-v5's.

### mix-v5b as built

```
tjev data build ~/tjev-work/data/mix-v5b --jevbench-public …/jevbench/public.jsonl \
    --gen-scale 2 --max-epochs 2 --rebalance-noul 0.7          # ~25 min on Colab CPU
tjev data heldout ~/tjev-work/data/mix-v5b/heldout-select.jsonl
```

- **Where it is.** Uploaded to the private `tychot/tjev-data` repo under `data/mix-v5b`
  (109 files, as mix-v5). Data report:
  [`reports/data/mix-v5b/report.md`](../reports/data/mix-v5b/report.md).
- **Against mix-v5.**
  - Held-out files: validation, calibration, heldout-select and the 52 test files are
    byte-identical.
  - Train files: only `gen_return_window` and `gen_long_refund_policy` differ.
  - Weights: they move by < 1e-4. The epoch cap re-measures token lengths, and the hint
    lines add a few tokens.

| | mix-v5 | mix-v5b |
|---|---|---|
| train rows (French) | 498,555 (108,680) | 498,555 (108,652) |
| sampled French share | 20.5% | 20.5% |
| block shares jev / generators / replay / analysis / writing | 38.6 / 27.7 / 19.3 / 9.6 / 4.8% | 38.6 / 27.7 / 19.3 / 9.6 / 4.8% |
| segments per step / steps by bucket 1024 / 2048 / 4096 | 161.27 / 1937 / 514 / 549 | 161.25 / 1937 / 514 / 549 |
| max epochs (3,000 steps) | 1.968 | 1.967 |
| gen_return_window: share / epochs / mean prompt tokens | 3.23% / 1.97 / 128.8 | 3.23% / 1.96 / 130.5 |
| gen_return_window: yes share / distinct windows / common-window share / hinted | 0.492 / 6 / 100% / 0% | 0.495 / 194 / 40% / 18.5% |
| gen_long_refund_policy: share / token share / epochs | 1.64% / 13.9% / 1.96 | 1.64% / 14.0% / 1.96 |
| gen_long_refund_policy: approve / deny / escalate, hinted | 877 / 2178 / 945, 0% | 877 / 2178 / 945, 18.5% |
