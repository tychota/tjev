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
 "usage": {"prompt_tokens": 412, "seconds": 0.031, "batch_items": 2},
 "model": "tjev:q2b@600"}
```

- **One rubric per question.** Each question becomes one item with tjev's prompt and letter
  readout. Its probabilities come back in the question's own label order and sum to 1.
  `noul` is P(yes); `choice` is the most probable label.
- **Calibration.** The artifact must have been fitted on this very checkpoint (JAX) or on
  this MLX model and quantization. Otherwise the server refuses to start.
- **Micro-batching** (`tjev.serve.batcher`). Requests are async. A single worker collects
  the items of concurrent requests for up to `--max-wait-ms` (5 ms by default) or
  `--max-items` (64), then scores them in one backend call in a thread. The accelerator is
  used by one call at a time, and a packed call amortises its fixed cost over every request.
  `usage.batch_items` reports how many items were scored together.
- **Buckets** (JAX). The JAX backend packs items into the smallest bucket that fits
  (1024 … 8192 tokens) and compiles every bucket once at startup. The MLX backend scores
  one prompt at a time (its prefill is compute-bound anyway).
- **Errors.** An invalid rubric (for example fewer than two labels) is 400; a schema error
  is 422; a state longer than the largest bucket is 413.
