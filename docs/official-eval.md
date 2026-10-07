# The official evaluation

`tjev eval-official` measures a model with the **official JevBench harness**
([fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench), pinned at
`bb05a335`, the v1.5.4 board's commit). It does not use tjev's own eval loop. tjev serves the
model on a local port, and the harness's TypeSafe adapter queries it over HTTP as it would
query any entrant. The harness is never modified: it runs as a subprocess of its own CLI,
and the URL and model name are passed as CLI flags.

```bash
uv sync --extra serve
git clone https://github.com/fstandhartinger/jevbench ~/tjev-work/jevbench   # then: git checkout bb05a335

tjev eval-official runs/final-4B --out reports/official/final-4B            # a run: selected step + post/calibration-jax.json
tjev eval-official base:4B --out reports/official/base-4B \
  --calibration reports/zeroshot-4B.json                                     # the base model zero-shot (s0 temperatures)
tjev eval-official base:4B --prompt semif --out reports/official/base-4B-semif   # the field's frozen-model control
tjev eval-official runs/final-4B --orders 2 --out reports/official/final-4B-o2  # two-order averaging
tjev eval-official base:0.8B --per-tier 2 --out /tmp/smoke                   # a smoke run: 2 items per tier
```

- **TARGET.** A run directory, an HF snapshot, or `base:SIZE`, which resolves to
  `--models`/`Qwen3.5-SIZE` (default `$TJEV_ROOT/models`).
- **Harness.** `--harness` defaults to `$TJEV_ROOT/jevbench`. A checkout at another commit is
  refused unless `--allow-unpinned` is given.
- **Output.** The report lands in `--out`/`report.json`, next to the harness's own files:
  - `harness-results.jsonl` (per-item records),
  - `harness-manifest.json`,
  - `harness.log`,
  - `raw/` (request and response bodies).

On the TPU it is a probe mode that needs no training:
`PROBE_ONLY=official PROBE_OFFICIAL_TARGETS=base:4B,final-4B-… bash cloud/tpu_campaign.sh probe`.
- It writes `reports/probe/official/<name>.json`.
- It logs each report to W&B project `tjev-benchmarks` (job type `official-eval`).
- Base targets run with both prompts (`tjev`, `semif`), runs with tjev's; one option order
  (`tjev eval-official --orders` for more).
- `PROBE_OFFICIAL_DI_SUITE` also runs the Decision Index kit.
- The script checks out the pinned harness at `$TJEV_ROOT/jevbench-harness`.

On a v6e-1 the public 231 should take a few minutes (an estimate: about 0.2M prompt tokens). Most of that is the per-bucket compiles at startup.
The harness is serial, and the server is started with `max_wait_ms=0`, so a request never
waits for a micro-batch.

## What the report contains

The harness implements v1–v1.4 scoring but not v1.5. v1.5 is a frozen method document
(`docs/METHOD-v1.5.md` plus the equal-types headline addendum). For that reason:
- the **v1.3/v1.4 numbers** come from the harness's own code (`jevbench.summarize`,
  `composite_v13`, `composite_v14`), imported from the checkout;
- the **v1.5 numbers** are tjev's implementation of the method (`tjev.eval.official`),
  unit-tested against the method's definitions.

| Report key | What it is | Official? |
|---|---|---|
| `accuracy`, `tiers.*.accuracy`, `types`, `families` | argmax accuracy (failed or invalid answers count as wrong), per tier (easy 48 / standard 72 / hard 111), per type, per family | harness code |
| `harness_summary` | the harness's `summarize`: Brier, ECE (10 bins), ordinal MAE, latency, paraphrase agreement, per family and split | harness code |
| `v13.intelligence_public` | chance-corrected tier accuracy with the published per-tier chances, weights 14/28/(judge 28)/30. Judge is not public, so the weights are renormalised. `intelligence_public_item_chance` uses the chance of the public items themselves | harness code |
| `v13.calibration` | hard tier: `100·(1 − ECE/0.5)` averaged with `100·(1 − mean TVD)`. The TVD is taken to the gold distributions of the 10 hard items that have one | harness code, harness formula |
| `latency`, `latency.*.speed` | harness wall time per request, p50/p95 (all items, and the standard tier, which is the public part of v1.5's standard + judge speed run). Adjusted ×2 + 0.15 s for an own server (`--endpoint-kind gpu`/`cpu`); `api` leaves it unadjusted | harness code |
| `cost` | v1.5 M2 floor: measured input tokens × the base model's reference price. For Qwen3.5-4B that is $0.03/M input from the frozen DeepInfra snapshot; a logit readout has 0 output tokens. Other sizes have no frozen reference, so cost is `null` unless `--price-in` is given | harness formula |
| `v15.intelligence_open_public` | v1.5 §3.1 on the public items. Choice CC uses the per-item chance. **Noul: P(yes) ≤ 0.2 is No, ≥ 0.8 is Yes, anything between is an abstention and counts wrong** (`types.noul.abstentions`). Score CC is 1 − nMAE/nMAE_chance, with the expected level as the prediction. Tier weights 10/20/30/40 are renormalised over tiers with items. Types are equal (`intelligence_equal`, the headline addendum) or 50/25/25 (`intelligence_frozen`) | tjev impl. of the method |
| `v15.calibration_typed` | choice: top-label ECE (all choice items; the v1.3 hard-tier-only ECE is reported as `ece_hard`, not scored) + TVD; noul: ECE of P(yes), with Brier reported; score: normalised RPS + top-level ECE. The draft that maps each part to points is not published, so each part is scored as in v1.3 (`100·(1 − ECE/0.5)`, `100·(1 − x)`) and the parts of a type are averaged | tjev interpretation |
| `v14.composite_public_proxy`, `v15.composites_public_proxy.{A,B,C}` | weighted harmonic mean of I, C, S and $, with gates: I below 50 (60 for C), or S or $ below 50, multiplies the score by (x/50)². **Proxies:** they take I_sealed = I_open, so no gap penalty applies | proxy |
| `v14.official`, `v15.official` | `null`. Both need the sealed set, which only the evaluator holds | — |
| `tjev`, `tiers.*.tjev` | tjev's own metrics on the served distributions: accuracy, NLL, Brier, 15-bin ECE | comparable to `tjev eval` |

Not computable here:
- the sealed half of Intelligence, and therefore the gap penalty and the official composites;
- the judge tier, which is not public;
- the 601 v1.5 published items, which are not in the harness repo (the 231 are the v1.2
  public split);
- the bootstrap ties.

### How tjev's internal numbers map

`tjev eval RUN data/jevbench/public.jsonl` (and `post/jevbench-cal.json`) scores the **same
231 items**, with these differences:
- the items go through tjev's eval loop, not over HTTP;
- the ECE uses 15 bins, not 10;
- noul is read by argmax, with no abstention band;
- the option order differs: it is a hash of the item id, whereas the server hashes the
  question's content.

Internal source `jevbench_original` is the official **standard** tier. The internal accuracy
should match `report.tjev.accuracy` and the official `accuracy` to within an item or two of
order effects. The v1.5 Intelligence is lower than any accuracy-based number for two
reasons:
- confident-but-uncalibrated noul answers are fine, but a calibrated 0.7 is wrong;
- chance is subtracted per item.

## v1.5 scoring on per-item predictions

```bash
tjev eval-v15 RUN/post/predictions/{jevbench,generators,validation}-k2.jsonl.gz \
  [--calibration ART | --fit-on POOL_DUMPS...] [--harness ~/tjev-work/jevbench] \
  [--abstention noul --commit-margin M --noul-temperature T --score-temperature T] [--sweep] [--out F]
python scripts/v15_study.py h3000@1200=RUN/post ... --tag -k2 --out reports/v15-study
```

`tjev.eval.v15` is the v1.5 scorer, shared by `eval-official` (on the harness's records) and
`tjev eval-v15` (on `tjev post` prediction dumps: no model, no server). It reports, per
dump: Intelligence per type × tier (CC, not clipped; tier weights 10/20/30/40 renormalised
over the tiers present; items with no known tier form one tier `all`), I_open with equal
types (headline) and 50/25/25, the typed Calibration, the noul abstention rate, per-family
accuracy (argmax, and v1.5 for choice + noul), and with a serving policy
([serve.md](serve.md#serving-policies)) the same numbers as the harness would score them.
`--sweep` adds the one-knob sweeps (commit margin, noul T, score T). Gold: the argmax of the
item's target (a tied target is skipped); a soft target, or `--harness`'s published
`gold_probs`, is the choice TVD reference. JevBench items get their tier from their source
(`jevbench_easy` / `_original` → standard / `_hard`). `tjev post` writes `post/v15{tag}.json`
(every dumped split, with the sweep) and dumps `predictions/jevbench{tag}.jsonl.gz`.

**Validation of the scorer.** The harness has no v1.5 code. `tests/eval/test_v15.py`
checks hand-computed cases (the band edges, the argmax tie rule, the per-item chance, the
expected-level nMAE and its uniform chance, the tier/type renormalisation, each calibration
part), and, on the 231 public items with random distributions, that the per-item result
recomputed from the distribution equals the one built from the harness's own
`jevbench.scoring.score_task` record (`official.v15_item`), that our ECE equals
`jevbench.metrics.ece_top_label`, and that a choice calibration's points equal the harness's
`composite_v13.calibration(ECE, TVD)`. Interpretations: the draft that maps a typed part
to points is unpublished, so each part is v1.3-scored (ECE → 100 − 200·ECE; TVD, RPS →
100·(1 − x)) and averaged; choice ECE is on all choice items (the v1.3 hard-tier-only ECE is
reported as `ece_hard`).

### Measured (K = 2 dumps, in-distribution per-type T)

These dumps predate the JevBench dump (post then wrote calibration / generators /
validation only), so the numbers below are on the mix validation split (in distribution,
11.2k items) and the held-out generators (220 items, 11 code generators, held-out surface
text; one tier, so CC = the tier's). They are not JevBench numbers: JevBench noul and score
items are fewer and harder. A `tjev post` rerun adds the JevBench dump (one forward pass
over 231 × K, the reports are kept) and `v15-k2.json`; the JevBench reading of the policy
below is still to be measured.

| run | split | I | I choice | I noul | I score | C | C choice | C noul | C score | noul abstain |
|---|---|---|---|---|---|---|---|---|---|---|
| h3000@1200 | validation | 63.4 | 79.6 | 46.3 | 64.4 | 91.9 | 85.6 | 96.7 | 93.5 | 23.4 % |
| h3000@1200 | generators | 61.4 | 76.4 | 10.0 | 97.8 | 88.9 | 91.9 | 77.0 | 97.9 | 42.0 % |
| h3000@3000 | validation | 65.4 | 83.1 | 47.4 | 65.8 | 91.1 | 84.6 | 95.0 | 93.8 | 23.4 % |
| h3000@3000 | generators | 61.5 | 79.0 | 10.0 | 95.4 | 82.2 | 77.2 | 74.0 | 95.4 | 40.0 % |
| h1500@1450 | validation | 64.9 | 81.0 | 47.8 | 65.8 | 91.7 | 86.2 | 96.6 | 92.2 | 22.5 % |
| h1500@1450 | generators | 59.8 | 76.4 | 8.0 | 95.0 | 88.4 | 90.4 | 79.6 | 95.1 | 42.0 % |

The noul band is the weak point: 23 % of validation and 40–42 % of held-out generator noul
answers are abstentions (generator math and probability families: v1.5 accuracy 0–37 %
against 45–70 % argmax accuracy).

**Noul commit, by margin M** (validation / generators; M = 0.3 is no commit):

| run | M | I val | C val | abstain val | I gen | C gen | abstain gen |
|---|---|---|---|---|---|---|---|
| h3000@1200 | 0 | **73.3** | 90.1 | 0 % | **75.4** | 86.2 | 0 % |
| | 0.05 | 72.5 | 90.6 | 2.7 % | 74.1 | 87.6 | 8 % |
| | 0.1 | 71.5 | 91.0 | 5.5 % | 72.7 | 88.2 | 12 % |
| | 0.2 | 68.7 | 91.7 | 13.0 % | 69.4 | 88.4 | 23 % |
| | 0.3 (off) | 63.4 | 91.9 | 23.4 % | 61.4 | 88.9 | 42 % |
| h3000@3000 | 0 | 75.8 | 90.4 | 0 % | 73.5 | 78.1 | 0 % |
| | 0.3 (off) | 65.4 | 91.1 | 23.4 % | 61.5 | 82.2 | 40 % |
| h1500@1450 | 0 | 74.7 | 90.1 | 0 % | 74.5 | 85.3 | 0 % |
| | 0.1 | 72.2 | 91.1 | 6.8 % | 69.8 | 88.6 | 19 % |
| | 0.3 (off) | 64.9 | 91.7 | 22.5 % | 59.8 | 88.4 | 42 % |

The commit is worth +9.8 to +10.4 Intelligence points on validation and +12.0 to +14.7 on
the held-out generators (I noul 46 → 76 and 10 → 52), for −0.7 to −1.8 Calibration on
validation and −2.7 to −4.1 on the generators (the committed answers sit at 0.2 / 0.8,
which the noul ECE sees). Every margin is a straight trade along that line; M = 0 minimises
1/I + 1/C (what preset A's composite depends on with Speed and Cost fixed) on validation at
all three checkpoints.

**Noul T.** Alone, a lower noul T recovers part of what the commit does at a larger
calibration cost (h3000@1200 validation, T 0.5: I 69.8, C 88.6). With the commit on it buys
no Intelligence (the side of 0.5 does not move) and only costs calibration (T 0.5: C 87.7
vs 90.1): chosen T = 1 (0.9 at h1500@1450, +0.1 C).

**Score T.** The expected level barely moves: T 0.2 gives I score +1.4 to +1.9 on validation
(64.4 → 66.3) for C score −25 to −28 (RPS and top-level ECE: 93.5 → 65.8); on the generators
(20 items, mostly right) it is ±0.5. Chosen T = 1 (0.9 at h1500@1450: +0.4 I score, +0.4 C
score; noise).

**Recommended policy: `--abstention noul` (commit, M = 0, ε = 0), no serving temperatures.**
On a v1.5 preset-A composite with Speed and Cost unchanged, +10 to +15 Intelligence for
−1 to −4 Calibration is a clear gain, and it is disclosed (serve.md). Measure it on the
JevBench items before quoting a number: rerun `tjev post` (adds `predictions/jevbench-k2`)
then `tjev eval-v15 … --abstention noul`, or `tjev eval-official … --abstention noul`.

## What the server does for the harness

- `usage.input_tokens` (= `prompt_tokens`) and `usage.output_tokens` (0). These are the
  names the harness prices.
- The displayed option order is a hash of the question's content (an earlier hash of the
  constant id `request:<name>` gave every served question with the same option count the
  same permutation, so all of them shared one position bias).
- `JaxBackend(orders=K)`: order averaging. Choice and noul items are read under `min(K, n)`
  evenly spread rotations of their options (noul at 2 means both orders), and the calibrated
  probabilities are averaged. Score levels keep their order. Tokens and cost scale with K.
  The calibration must have been fitted with the same K: a run's default is the `tjev post
  --orders K` artifact (`post/calibration-jax-k2.json` for K = 2); a K = 1 artifact is
  refused ([serve.md](serve.md#readout-options)).
- `--noul-order fixed` (no first) and the serving policies (`--abstention field | noul`,
  `--commit-margin`, `--noul-temperature`, `--score-temperature`; [serve.md](serve.md#serving-policies))
  are passed to the served app; `report.json` records them under `served` (`served.policy`).
- `JaxBackend(prompt="semif")`: the control prompt below. It is uncalibrated, because
  temperatures bind tjev's template.
- An empty state (`""`, `{}`) is answered, not refused with a 400. The refusal messages
  carry the Decision Index kit's capacity markers ("options per choice", "too many tokens"),
  so the kit counts them as unsupported rather than as errors.

## The frozen good-prompt control

A frozen Qwen3.5-4B scores 0.805–0.810 on the public 231, and three entrants share its
prompt:

| Entrant | Repo, commit | Public 231 |
|---|---|---|
| **SemIf** (ex-OpenJev) | [TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) `23cf1f3`, `src/semif_phase1/{core,direct}.py`; prompt unchanged since `d85b4ab` | 0.810 under the official harness (easy 48/48, standard 71/72, hard 68/111) |
| **Jobe** | [MantisShrimpdev/jobe](https://github.com/MantisShrimpdev/jobe) `0382f20` | 0.805 under the official harness |
| **local-jev** | [amithgc/local-jev](https://github.com/amithgc/local-jev) `64a0b31` | 0.805, with the same layout and slightly different option texts |

[reflex](https://github.com/kshetrajna12/reflex) (`231f896`) uses a markdown prompt (`# Evidence` / `# Criterion` / `# Options`):
- with one order it scores 0.810, and with two orders 0.823, both on reflex's own runner;
- its author used the public items for development;
- its own-words JSON variant scores 0.740, which is where the "±8 points from wording" figure
  comes from.

tjev implements SemIf's prompt as `--prompt semif` (`tjev.data.render.semif_user`). It is the
one measured by the official harness and reproduced independently.

```text
<|im_start|>system
Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.<|im_end|>
<|im_start|>user
{"evidence": <state>, "criterion": <instructions>, "options": [{"letter": "A", "description": "<id>: <text>"}, ...]}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The readout is the next-token logits over the letters, softmax at T = 1.

How it differs from tjev's template (`TEMPLATE_VERSION` 1). The chat framing and the letter
readout are the same. The differences:

1. **System wording.** SemIf: "Apply the supplied criterion to the supplied evidence…".
   tjev: "You are a decision engine…".
2. **User layout.** SemIf sends one JSON object (`evidence`, `criterion`,
   `options[{letter, description}]`). tjev uses plain text: `State:` / `Question (type):` /
   `Options:` with `A. label: description`.
   - local-jev measured a plain layout close to tjev's at 0.732, against 0.805 for the JSON
     one.
   - reflex shows the wording matters more than the JSON itself.
3. **No type hint.** tjev writes "(yes/no | choose one | rate on the scale)"; SemIf does not.
4. **Noul options.**
   - SemIf: `true: <criterion>` then `false: <criterion>`, always in that order (A = yes),
     with fallback "The proposition is true/false.".
   - tjev: `no` / `yes` with the descriptions, hash-shuffled.
5. **Option order.** SemIf uses the declared order and never shuffles; tjev hash-shuffles
   choice and noul.
   - Jobe measured 0.805 for the declared order against 0.784 for the reversed one; averaging
     the two gave 0.801.
   - reflex measured +2.7 on the hard tier from two orders.
6. **Structured states.** SemIf embeds the JSON object itself. tjev renders it as indented
   JSON text.

The control is a zero-shot read of the untrained base model. Comparing `base:4B --prompt
semif` with `base:4B` (tjev's prompt) isolates the prompt. Comparing it with a trained run
shows what training adds over the best frozen read.

## Decision Index

[sinanuozdemir/decision-index](https://github.com/sinanuozdemir/decision-index) 0.2.1, pinned
at `87d4650`, MIT. Run it with `--suite decision-index --di-kit KIT --di-suite SUITE`. This
runs the kit's own `run --engine http --option base_url=… --compact` and then `score`. It has
not been run end to end here, because the suite is not built locally.

- **Size.**
  - 44 benchmarks (38 indexed): 150,317 requests, 322,459 questions, about 100M tokens.
  - Up to 64 questions per request. All are choice except RAGTruth and part of PhishNChips
    (noul). There are no score items.
  - Nothing is shipped. `suite rebuild` downloads about 7 GB from many hosts (HF-gated HLE,
    a 2.2 GB HoVer DB) into about 17 GB of work space.
  - Several sources are non-commercial or bar redistribution, so results must not be
    republished with item text.
- **Runtime** on a v6e-1 at about 14k tokens/s for 4B. The kit is strictly serial.
  - tjev's server scores every question as its own item, so a shared state is re-read per
    question (about 165–200M tokens): **about 3.3–4 h of prefill plus about 25–50 min of
    per-request overhead, so 4–4.5 h**.
  - With shared-state prefill it would be about 2.5 h.
- **Coverage limits.**
  - 14,500 requests have more than 26 options: CLINC150 151, POP909 129, ChessBench up to
    81, BANKING77 77, API-Bank 53. Three of these are gold benchmarks.
  - Roughly 500–1,900 requests exceed the 8,192-token bucket (ToolRet, BRIGHT, ACOS,
    ContractNLI).
  - tjev refuses both kinds as unsupported, and the kit scores them 0.
  - A request is refused as a whole when one of its questions does not fit.
- **Feasibility verdict.** It is feasible as an overnight TPU job once the suite is built
  (on the host, with an HF login for HLE). It is not a quick probe. Its score is capped by
  the 26-letter readout until tjev reads more options (research doc item 8).
