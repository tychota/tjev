# Text analysis in the mix: tasks, datasets, synthetic data (EN + FR)

Design of the "analysis" and "writing" blocks of mix-v3. The goal is a model that analyses a
text the way a writing assistant such as Antidote does, and goes beyond it: who wrote it
(human or AI, which model family), register and formality, sentiment and emotion,
offensiveness, readability and style issues. Every dimension is asked in the item formats
the model is trained on (`choice`, `noul`, `score`), with soft targets where annotators
disagree.

Code: `src/tjev/data/sources/text_analysis.py` (`TEXT_SOURCES`, the analysis block) and the
`gen_spelling_error` / `gen_register` generators in `src/tjev/data/sources/generators.py`
(the writing block). **[unverified]** marks claims not confirmed on an official page.

## 1. Share of the mix

`BLOCKS` in `src/tjev/data/mix.py`:

| block | share | content |
|---|---|---|
| jev | 0.40 | contender data, JevBench-style decisions (EN) |
| generators | 0.25 | templated decision generators (30% FR) |
| replay | 0.20 | public classification/NLI datasets |
| **analysis** | **0.10** | `TEXT_SOURCES`: authorship, formality, sentiment, emotion, offensiveness |
| **writing** | **0.05** | `gen_spelling_error`, `gen_register` |

- Analysis + writing stay at **≤ 15% of the sampled mix**: the target is JevBench-style
  decisions, and text classification is a supporting skill.
- Within a block, sources are weighted ∝ n^0.5. French-only sources carry 35% of the
  analysis block (`FR_TARGET`), part of the 20% French target of the whole mix.

## 2. What Antidote analyses, and where we go further

Antidote 12 (Druide, 2024) groups its checks as:
- **Style:** repetitions; tournures (passives, impersonnelles, négatives, averbales,
  lourdeurs); vocabulaire (pléonasmes, niveau de langue, offensant, verbes ternes);
  lisibilité (long sentences, cascades of complements, rare words; Flesch, Flesch-Kincaid,
  Gunning Fog, Coleman-Liau); inclusivity.
- **Révision:** pragmatics, lexical semantics (weak / strong / negative / positive words),
  logic (connectives, quotes, parentheses).
- **Statistiques:** sizes, reading time, error density, frequent words, lexical fields.
- **Anti-Oups!:** hurtful tone (exclamations, capitals, emojis).

Its semantic analysis is **word- and span-level** and lexicon/rule-based. It has **no AI-text
detection, no document-level tone or genre, and no emotion classification**; those are the
dimensions this block adds.

## 3. Taxonomy, by how well it calibrates

| tier | dimensions | target |
|---|---|---|
| Objective (label comes from how the data was made) | authorship human / AI / AI-polished / AI-humanized; generator family; error present; simplified vs original; rule-derived style counts | one-hot |
| Moderately objective | register (CORE: narrative, informational, opinion, discussion, how-to, persuasion, lyrical, spoken); formality; sentiment; politeness; toxicity; subjectivity; stance; hedging; readability | soft, from multi-annotator labels |
| Subjective | emotions (Ekman, Plutchik-8, GoEmotions-27); irony; persuasion techniques; tone (académique, journalistique, familier, administratif, marketing, juridique…) | distributions only, never argmax |

Antidote-style sentence issues are deterministic (long sentences, passives, negatives,
verbless sentences, repetitions, readability indices). They become code-labelled score items
("how many passive clauses: 0 / 1 / 2 / 3+") in the generator framework.

## 4. Datasets

### 4.1 In the current mix

| source | dataset | lang | task, format | target | license |
|---|---|---|---|---|---|
| `raid` | [RAID](https://huggingface.co/datasets/liamdugan/raid) | EN | AI? (noul) or which family (choice) | one-hot | MIT |
| `mage` | [MAGE](https://huggingface.co/datasets/yaful/MAGE) | EN | same | one-hot | Apache-2.0 |
| `hc3_fr` | [HC3 French](https://huggingface.co/datasets/almanach/hc3_french_ood) (QA) | FR | human vs ChatGPT answer (noul) | one-hot | CC BY-SA 4.0 (machine-translated) |
| `pavlick_formality` | [Pavlick formality](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores) | EN | formality, 5 levels (score) | soft (interpolated mean of 5 raters) | CC BY 3.0 |
| `tweet_sentiment_en/fr` | [tweet_sentiment_multilingual](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) | EN, FR | 3-class sentiment (choice) | one-hot | CC BY 3.0 + Twitter ToS |
| `go_emotions` | [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) raw | EN | Ekman group + neutral (choice) | soft (per-rater votes, ≥ 2 raters) | Apache-2.0 |
| `mlma_en/fr` | [MLMA](https://huggingface.co/datasets/nedjmaou/MLMA_hate_speech) | EN, FR | offensive? (noul) | soft (share of non-`normal` labels) | MIT (HF card) [original unverified] |
| `textdetox_fr` | [textdetox multilingual](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset) | FR | toxic? (noul) | one-hot | openrail++ (`commercial_ok=False`) |

Loading notes:
- RAID's train CSV is 11.8 GB. `raid_sample` streams it once into a seeded 60k reservoir
  (cached as parquet), keeping attacked rows (paraphrase, homoglyphs, zero-width
  characters…) with probability 1/33, so ~25% of the sample is attacked. Held-out rows are
  split by `source_id`, so one source document never straddles train and held-out.
- Family labels come from generator-name prefixes (`model_family`): OpenAI, Meta, Mistral,
  Cohere, EleutherAI, Google T5, BigScience, MPT, GLM. A family question shows 4–6 options.
- Script-only Hub repos load from `refs/convert/parquet` (`datasets` 4.x runs no scripts).
- Every source carries its license and a `commercial_ok` flag in the build manifest;
  `tjev data build --commercial-only` drops the others.

### 4.2 Candidates not yet in the mix

| dim | dataset | lang | size / labels | license |
|---|---|---|---|---|
| Human / AI / polished / humanized | [LLM-DetectAIve](https://github.com/mbzuai-nlp/LLM-DetectAIve) | EN | ~300k | [unverified], gated |
| Attribution | [M4GT-Bench](https://github.com/mbzuai-nlp/m4gt-bench) | multi, no FR | which model | [unverified] |
| Real LLM outputs | [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | 74 incl. FR | 838k conversations, `model` field | ODC-BY |
| Register | [FreCORE](https://github.com/TurkuNLP/Multilingual-register-corpora), [multilingual-CORE](https://github.com/TurkuNLP/multilingual-CORE) | FR; 16 incl. EN, FR | CORE labels | CC BY; [unverified] |
| Sentiment | [Allociné](https://huggingface.co/datasets/tblard/allocine) | FR | 200k binary | MIT tag, scraped: legal check |
| Emotion | [XED](https://github.com/Helsinki-NLP/XED) | EN; FR projected [unverified] | Plutchik-8 multilabel | CC BY 4.0 |
| Politeness | [Stanford Politeness](https://convokit.cornell.edu/documentation/wiki_politeness.html), [TyDiP](https://github.com/Genius1237/TyDiP) | EN; EN + 9 incl. FR | per-rater scores | CC BY 4.0 |
| Readability | [VikiWiki](https://github.com/ionmadrazo/VikiWiki) | 6 incl. FR | Vikidia vs Wikipedia | CC BY-SA 3.0 |
| Error presence | [WiCoPaCo](https://wicopaco.limsi.fr/) | FR | Wikipedia correction edits | GFDL |

Recent-model and French AI-text data (checked on HF cards, 2026-09-29; details unverified
unless stated):

| dataset | generators | French | license | use |
|---|---|---|---|---|
| [DetectRL-X](https://huggingface.co/datasets/WUJUNCHAO/DetectRL-X) (ACL 2026) | DeepSeek-V3, Gemini-2.5-Flash, GPT-4o, Qwen-Max | native, ~15.6k FR pairs | MIT | FR detection + attribution; LLM-refined human class; 11 attacks |
| [arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | 53 models of 2025 | ~4.3k FR responses | prompts CC BY 4.0; outputs under provider terms | recent-model attribution |
| [HelpSteer3 edit](https://huggingface.co/datasets/nvidia/HelpSteer3) | ~20 open models | 549 FR human-edited | CC BY 4.0 | the human-edited-AI class |
| [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) | gpt-4o, gpt-4.1-mini, o1 | yes | ODC-BY + OpenAI terms | OpenAI family over time |
| [DACTYL](https://huggingface.co/datasets/ShantanuT01/DACTYL) | GPT-4o, Claude 3.5, Gemini 1.5, Llama 3.x, Mistral, DeepSeek-V3 | no | MIT | EN recent attribution |
| [OpAI-Bench](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench), [APT-Eval](https://huggingface.co/datasets/smksaha/apt-eval), [Beemo](https://huggingface.co/datasets/toloka/beemo) | various | no | Apache-2.0 / MIT | graded AI polishing; expert-edited AI text |

Not used (non-commercial or research-only terms, gated access, or no redistribution):
XFORMAL / GYAFC (Yahoo Webscope), SemEval-2023 Task 3, UniversalCEFR / CEFR-SP, x-stance,
MULTITuDE / MultiSocial, RealDet (CC BY-NC), PAN'25/26 Voight-Kampff, flaird-raid-pan26
(no license). A non-commercial source could enter a build only with `commercial_ok=False`.

**Gaps.** No public native-French model-attribution data with open terms besides
DetectRL-X (COLING-25 GenAI, MultiSocial and MultiGhostBench have no French); French tone
and style labels are missing; French subjectivity and irony data is weak. Synthetic data
(§5) covers them.

## 5. Synthetic data

### 5.1 In the current mix: the writing block

- `gen_spelling_error` (noul): one or two templated sentences; with probability 0.5 one
  word is replaced by a known misspelling or agreement error. EN + FR.
- `gen_register` (score, 3 levels): an opening, body and closing drawn from informal,
  neutral or formal template sets. EN + FR.
- Both use separate templates, names and question wordings for the held-out split, so the
  held-out generators test transfer, not template recall.

### 5.2 Recipes for later blocks

**Authorship and attribution (FR + EN).**
- Human anchors: permissively licensed human text dated before 2022 (FR Wikipedia /
  Vikidia, FreCORE, Allociné, OASST human turns, public administration text).
- Generation from the anchor's title, first sentence and key points, with a panel of
  open-weight models (Llama, Mistral, Qwen, Gemma, Lucie, DeepSeek). API models only after
  a terms-of-service review; several forbid training competing models on their outputs.
- Classes: fully generated; human then LLM-polished (light, heavy); human then paraphrased;
  AI then "humanized". Store the edit ratio so "AI-edited" can be a 0–3 score.
- Vary temperature, top-p, system prompt and persona, so no generator has one default style.
- Hold out one model version per family for out-of-distribution calibration.

**Tone, register and formality.**
- Rewrite one human source into each target tone and level with ≥ 2 LLMs.
- Verify each rewrite: a judge from another model family gives a distribution (the soft
  label); an NLI or embedding check confirms the meaning; a small double-annotated human set
  (≥ 200 per language) is used only for calibration and ECE.
- Every rewrite is also AI text: tone items compare rewrites with rewrites, or with human
  text of known register, so authorship never leaks into tone.

**Shortcuts to block.**
- Paired designs: the same source or topic appears in every class.
- Match length buckets across classes.
- Normalise markdown, "Certainly!" openers, em-dashes, quote styles and whitespace; keep
  RAID-style attacks as augmentation.
- A probe that predicts the source dataset, or a bag-of-words or length-only baseline, must
  stay near chance.
- Split by topic, source and generator, never by random row; MinHash-dedup across splits.

**Calibration.**
- Raw multi-rater labels as soft targets (GoEmotions raw, MLMA, Pavlick means spread over
  score levels; later Stanford Politeness and TyDiP). Never binarise a Likert mean.
- Temperature per question type × language after training; ECE reported per language.

## 6. Future work: word-level semantic analysis

Target: highlight each word or expression with a class and intensity, as Antidote's
Révision › Sémantique does: polarity (positive / negative / neutral), force (faible / neutre
/ fort), optionally emotion, register and offensiveness. A contextual model resolves what a
lexicon cannot: negation ("pas mauvais"), irony, domain sense ("une batterie qui chauffe"
is negative; "un accueil chaleureux" is positive).

**Format: several answer slots per segment.** Today each segment has one slot, its last
token (`RowBuilder` in `src/tjev/data/pack.py`). The extension:
- Render the full text, then a numbered list of candidate spans, each followed by its own
  slot: `Texte: … / Mots: 1. «mauvais» → [slot] 2. «vraiment» → [slot] …`. Every slot
  follows the whole text, so each label sees full context despite the causal model.
- One prefill gives all labels, which fits the on-device latency budget.
- Needed: a per-segment slot count in the packer, a per-slot label set, per-slot targets in
  the loss (which already normalises over slots).

**Candidate spans:** content words from a POS tagger plus multi-word expressions; or the
inverse question per class ("which of these words are negative? A…/none").

**Data** (licenses to verify):
- Silver lexicons, at low weight or as the prior to override: FEEL, Polarimots (FR); VADER
  (MIT), SentiWordNet, MPQA (EN). NRC EmoLex is non-commercial without a licence.
- Aspect/phrase-level sentiment: SemEval-2016 Task 5 ABSA (includes French), Stanford
  Sentiment Treebank phrase means (EN, soft).
- Synthetic: an LLM annotates polarity and force per span, a second model family
  re-annotates, agreement is the soft target; contrastive pairs where only a negation or
  intensifier changes ("bon" / "pas bon" / "très bon" / "plutôt bon").
- Rule-derived lists (verbes ternes, pléonasmes, mots rares) stay code-labelled items (§3).

**Evaluation:** ~200 hand-annotated FR and EN sentences with span polarity and force; the
model must beat a lexicon baseline on negation and irony cases.
