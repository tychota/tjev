# Serving

```bash
uv sync --extra serve
tjev serve --run runs/q2b --calibration runs/q2b/post/calibration-jax.json      # JAX
tjev serve --backend mlx --model mlx/q2b-bf16 --calibration mlx/q2b-bf16-cal.json  # Apple silicon
```

`tjev serve` runs a FastAPI app under uvicorn with two routes: `GET /health` and
`POST /v1/systemone`, the wire format of the JevBench TypeSafe adapter.

```json
{"state": "Do not cancel my membership. Please return the duplicate charge.",
 "questions": {
   "intent": {"type": "choice", "instructions": "Select the requested action.",
              "criteria": {"refund": "Return money already charged", "cancel": "End a subscription"}},
   "urgent": {"type": "noul", "instructions": "Is this urgent?",
              "criteria": {"false": "Can wait", "true": "Needs action today"}}}}
```

```json
{"answers": {"intent": {"type": "choice", "probabilities": {"refund": 0.93, "cancel": 0.07}, "choice": "refund"},
             "urgent": {"type": "noul", "probabilities": {"no": 0.71, "yes": 0.29}, "noul": 0.29}},
 "usage": {"prompt_tokens": 412, "input_tokens": 412, "output_tokens": 0, "seconds": 0.031, "batch_items": 2},
 "model": "tjev:q2b@600"}
```

- **One rubric per question.** Each question becomes one item with tjev's prompt and letter
  readout. Its probabilities come back in the question's own label order and sum to 1.
  `noul` is P(yes); `choice` is the most probable label.
- **Option order.** The displayed order is a fixed hash of the question's content
  (state, type, instructions, labels), so the same question always gets the same prompt.
  `--orders K` averages K evenly spread rotations of the options (choice, noul; see
  [Readout options](#readout-options)). `JaxBackend(prompt="semif")` serves the field's frozen-model control prompt
  ([official-eval.md](official-eval.md)).
- **Usage.** `input_tokens` equals `prompt_tokens`, and `output_tokens` is always 0 (a logit
  readout generates nothing). These are the names the JevBench harness prices.
- **Calibration.** The artifact must have been fitted on this very checkpoint (JAX) or on
  this MLX model and quantization, and with the same readout (`--orders`, `--noul-order`).
  Otherwise the server refuses to start.
- **Micro-batching** (`tjev.serve.batcher`). Requests are async. A single worker collects
  the items of concurrent requests for up to `--max-wait-ms` (5 ms by default) or
  `--max-items` (64), then scores them in one backend call in a thread. The accelerator is
  used by one call at a time, and a packed call amortises its fixed cost over every request.
  `usage.batch_items` reports how many items were scored together.
- **Buckets** (JAX). The JAX backend packs items into the smallest bucket that fits
  (1024 … 32768 tokens, a single long question as Jev serves it; `--max-tokens` caps the
  list and the warm-up time) and compiles every bucket once at startup. The MLX backend
  scores one prompt at a time (its prefill is compute-bound anyway).
- **Errors.** An invalid rubric (for example fewer than two labels) is 400; a schema error
  is 422; a state longer than the largest bucket, or more than 26 options, is 413. An empty
  state is answered.
- **Official evaluation.** `tjev eval-official` serves a model in-process and runs the
  official JevBench harness against it ([official-eval.md](official-eval.md)).

## Readout options

```bash
tjev post RUN --mix MIX --jevbench PUBLIC --orders 2            # writes post/calibration-jax-k2.json
tjev serve --run RUN --calibration RUN/post/calibration-jax-k2.json --orders 2
tjev serve --run RUN --calibration CAL --noul-order fixed        # CAL fitted with --noul-order fixed
tjev serve --run RUN --calibration CAL --abstention field         # + committed / decision (noul)
```

All off by default; the defaults serve exactly what tjev served before.

- **Order averaging (`--orders K`).** Each choice or noul question is read under
  `min(K, n)` evenly spread rotations of its displayed options (rotation `r·n // K'`; the
  first is the usual order, so K = 1 is unchanged; noul at K = 2 is both orders; K ≥ n puts
  every option once at every position). Score levels keep their order and take one pass:
  training never shows them reversed or rotated. Each rotation's letter logits are
  softmaxed at the calibrated temperature, un-permuted to the label order, and the
  distributions are averaged. `tjev eval`, `tjev calibrate`, `tjev post`, `tjev eval-base`,
  `tjev eval-official` and `tjev mlx calibrate | eval` take the same `--orders` and use the
  same rotations (`tjev.data.render.eval_specs`), so eval numbers are the served ones.
- **Arithmetic, not log-space, mean.** The mean of the probabilities is a mixture: its NLL
  is at most the mean NLL of the orderings (Jensen), and where the orderings disagree it is
  less confident, which is what removing a position bias should do. The log-space mean
  (softmax of the mean logit) is a product of experts: it sharpens where the orderings
  agree, and under temperature scaling it is just one more logit vector.
- **Calibration follows K.** Averaging already softens the distribution, so a temperature
  fitted on single orderings is wrong for averaged ones. `tjev calibrate` / `tjev post`
  with `--orders K` fit the temperatures on the averaged distributions; the
  artifact records `orders` and `noul_order` and every consumer (`tjev eval`, `tjev serve`,
  `tjev eval-official`, MLX) refuses a mismatch. Artifacts from before read as K = 1,
  shuffled. `tjev post --orders 2` writes its own files (`*-k2.json`, `summary-k2.md`).
- **Per-language temperatures.** A `type_lang` artifact (`--grouping type_lang`,
  [calibration.md](calibration.md)) also holds per (type, language) temperatures. The wire format has no
  language field, so the server guesses each question's language from its text (state,
  instructions, option descriptions) with the data report's stop-word heuristic
  (`tjev.data.lang.guess_lang`); an unsure guess, or a language the fit never saw, gets the
  type's temperature. A per-type (older) artifact needs no guess.
- **Cost.** Each rotation is a full prompt: the state is re-encoded per rotation (the packed
  rows hold independent segments; there is no shared-state prefix reuse yet, landscape
  item 9). On the JevBench public 231 (139 choice, 74 noul, 18 score) K = 1 is 231 prompts,
  K = 2 is 444 (1.92× the forward passes and about as many prompt tokens), K = 26 (every
  rotation) is 794 (3.4×). `usage.prompt_tokens` counts every rotation. A request's
  rotations are packed into the same backend call, so latency grows less than K× while
  they fit one row, and K× once rows are full (a TPU measurement).
- **Noul order (`--noul-order`).** Training shuffles noul like choice (unchanged). At eval
  the order is a hash of the item id, at serving a hash of the question's content (before
  the content hash, every request's `decision` question got the same permutation, which for
  noul was no first). `--noul-order fixed` shows A = no, B = yes for every question
  (Cygnet's "false first"), at eval and serving alike; with `--orders 2` it makes no
  difference (both orders are read either way).
- **Serving policies** (`--abstention`, `--noul-temperature`, `--score-temperature`): see
  [Serving policies](#serving-policies).
- `GET /health` reports `orders`, `noul_order`, `abstention` and `serving_temperatures`.

## Serving policies

```bash
tjev serve --run RUN --calibration CAL --abstention noul                      # noul commit
tjev serve --run RUN --calibration CAL --abstention noul --commit-margin 0.1   # commit unless |P - 0.5| <= 0.1
tjev serve --run RUN --calibration CAL --score-temperature 0.5 --noul-temperature 0.8
tjev eval-official RUN --out OUT --abstention noul --score-temperature 0.5      # what the harness then scores
tjev eval-v15 RUN/post/predictions/jevbench-k2.jsonl.gz --abstention noul --sweep   # offline, no model
```

All off by default (then the response is byte for byte what it was). A policy is applied
*after* calibration (`tjev.serve.policy`, the same code in the server, in `tjev eval-v15`
and in `tjev post`), the same way on the JAX and the MLX backends (it acts on the answer,
not in the backend).

- **Noul commit** (`--abstention field | noul`, `--band LOW HIGH`, `--snap`,
  `--commit-margin M`, `--commit-eps E`). JevBench v1.5 reads a noul P(yes) ≤ 0.2 as No,
  ≥ 0.8 as Yes, and anything between as an abstention, counted wrong. The harness's
  TypeSafe adapter (`jevbench/adapters/typesafe.py`) takes a noul answer from
  `answers.<name>.noul` only (as `{"yes": p, "no": 1 − p}`, for both the accuracy and the
  noul ECE) and never reads a noul's `probabilities`. The policy moves an in-band P(yes) to
  the band edge on its side (`--snap nearest`: the nearer edge; `argmax`: the side of 0.5;
  the same for the symmetric default band), `E` beyond it (0.2 − E / 0.8 + E; default 0, the
  edges themselves count as committed). With `M` > 0 an in-band P(yes) with |P(yes) − 0.5| ≤
  M is kept: a near coin flip stays an abstention. A noul answer gains `committed` and
  `decision` (`yes` / `no` / `abstain`). `field` adds only these fields (the harness scores
  exactly what it did); `noul` also reports the committed value as `noul`, i.e. what the
  harness scores. This is Plumb v5.2's `--noul-commit` (M = 0).
- **Serving temperatures** (`--noul-temperature`, `--score-temperature`, default 1). The
  calibrated distribution of a noul or score question is raised to 1/T and renormalised
  (T < 1 sharpens; the argmax is unchanged). On one ordering it is the calibrated
  temperature times T; on an order-averaged distribution it sharpens the mixture. A sharper
  score distribution puts its expected level closer to its mode, which is what v1.5 scores
  (the MAE of the expected level), at a calibration cost (RPS, top-level ECE).
- **Honest API.** Whenever a policy changes a value the harness scores (a serving
  temperature on that type, or the commit reported as `noul`), the answer also carries
  `calibrated`: the calibrated distribution, unchanged. `probabilities` is the served one
  (sharpened if a temperature is set); `noul` is P(yes) of it, or the committed value.

### Disclosure

These policies change the reported numbers, not the model. They exist because JevBench
v1.5 scores a noul P(yes) in (0.2, 0.8) as wrong and a score item by the MAE of its
expected level: a calibrated 0.7 that is right 70 % of the time is counted wrong every
time, and a calibrated score distribution with a long tail pays MAE for being honest.
Committing or sharpening trades calibration for those Intelligence points. **Any result
obtained with a policy on must say so**: the policy and its parameters (`served.policy` in
an `eval-official` report, `policy` in an `eval-v15` report, `/health` on a server), and
the same numbers without it. The benchmark maintainer reviews the leaderboard before every
release and closes loopholes with general, documented rules ("Before every release we
review the leaderboard for anomalies and close loopholes with general, documented rules",
the v1.5 headline addendum); a policy that a later rule neutralises is not a model
improvement, and the parameters are chosen on our development data (the mix validation
split), never on the JevBench items. The calibrated distribution stays in every response,
so a consumer that wants calibrated probabilities ignores the policy.

What the policies buy on our per-item predictions, and the recommended setting, is in
[official-eval.md](official-eval.md#v15-scoring-on-per-item-predictions).
