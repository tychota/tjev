# Architecture

tjev turns a pretrained Qwen3.5 text decoder into a calibrated decision model with LoRA.
Every question type becomes a lettered option list. The model's next token after the
assistant header is the letter, and the probabilities are a softmax over the letter
logits of the displayed options. There is no generation and no extra head: the readout
reads K rows of the (tied) output embedding at one position.

## The decoder

Qwen3.5 dense models are hybrids. Every period of four layers holds three **Gated DeltaNet**
(GDN) layers, linear attention with a delta-rule state, followed by one **gated full
attention** layer.

| Model | Hidden | Layers (GDN + attn) | FFN | GDN k / v heads | Attn q / kv heads, head dim | Head |
|---|---|---|---|---|---|---|
| 0.8B | 1024 | 24 (18 + 6) | 3584 | 16 / 16 | 8 / 2, 256 | tied |
| 2B | 2048 | 24 (18 + 6) | 6144 | 16 / 16 | 8 / 2, 256 | tied |
| 4B | 2560 | 32 (24 + 8) | 9216 | 16 / 32 | 16 / 4, 256 | tied |
| 9B | 4096 | 32 (24 + 8) | 12288 | 16 / 32 | 16 / 4, 256 | untied |

All sizes share a 248,320-token vocabulary and partial RoPE (64 of 256 dims, θ = 1e7).
Qwen3.5's interleaved MRoPE reduces to plain RoPE for text.

- **Parsing is fail-closed.** `tjev.model.config.ModelConfig.from_hf` refuses any
  config.json feature the port does not implement (MoE, a non-periodic layer pattern,
  attention bias, another activation or RoPE type).
- **The weight import is strict** (`tjev.model.weights`). The vision tower and the
  multi-token-prediction head are skipped by name. Any other unexpected tensor, missing
  tensor or shape mismatch is an error. Norms, `A_log` and `dt_bias` stay fp32; matrices
  go to the storage dtype.
- **One scan over super-blocks** (`tjev.model.qwen35`). The weights of layer 4s + j are
  stacked over s, so the forward pass is one `lax.scan` over super-blocks (MaxText's
  `Qwen3_5ScannableBlock`), and compile time does not grow with depth.
- **Per-layer remat.** Remat is applied per decoder layer, not per super-block, so the
  backward pass holds one layer's activations at a time.

### Gated DeltaNet layer (`tjev.model.mixers.GatedDeltaNet`)

```
qkv, z, b, a = in_proj_qkv(x), in_proj_z(x), in_proj_b(x), in_proj_a(x)
q, k, v      = split(silu(causal_conv1d(qkv)))             # per segment, 4 taps
q, k         = l2norm(q) · Dk^-½, l2norm(k)                 # eps inside the rsqrt
β            = sigmoid(b);   g = −exp(A_log) · softplus(a + dt_bias)
o            = gated_delta_rule(q, k, v, g, β)              # state restarts per segment
out          = out_proj(RMSNorm(o) · silu(z))
```

- **The core is one checkpointed unit.** The conv, the gates and the delta rule are
  rematerialised together, so their fp32 intermediates live only during their own
  backward pass.
- **The conv has a custom VJP.** Its backward keeps only x, the weight and the segment
  ids.
- **The delta-rule implementation is selectable.** It is chosen by
  `compute.gdn_impl`: the XLA chunked form, the recurrence, or the Pallas TPU kernels (see
  [kernels.md](kernels.md)).

### Gated attention layer (`tjev.model.mixers.GatedAttention`)

`q_proj` emits `[query | gate]` per head. q and k go through a zero-centred RMSNorm and
RoPE; attention is GQA within segments; the output is multiplied by `sigmoid(gate)` before
`o_proj`. There are two implementations (`compute.attention`):

- **xla**: query-blocked `jax.nn.dot_product_attention` with O(block × T) memory.
- **splash**: Pallas TPU splash attention.

## LoRA

`tjev.model.layers.Linear` holds the frozen weight as a non-`Param` variable and the
adapters as `nnx.LoRAParam`, so `nnx.split(model, nnx.LoRAParam, ...)` separates the
trainable state.

- **Targets.** Adapters sit on every projection: `q/k/v/o_proj`, the GDN
  `in_proj_qkv/z/a/b` and `out_proj`, and the MLP `gate/up/down_proj`.
- **Frozen.** The conv, `A_log`, `dt_bias`, norms and embeddings stay frozen.
- **Scaling.** rsLoRA: α/√r with α = 32, rank 32 (0.8B, 2B) or 64 (4B, 9B).
- **Precision and init.** Adapters are fp32, cast to the compute dtype for the matmul.
  B starts at zero, so a fresh adapter is the identity.
- **Grouped A matrices.** Projections that share an input (`in_proj_*`, `q/k/v`,
  `gate/up`) concatenate their A matrices. x is then read once by one wider low-rank
  matmul (`layers.grouped`).

## The decision readout

`Qwen35.label_logits(hidden, slots, label_ids)` gathers the hidden state at each answer
slot and dots it with K output-head rows in fp32. The letters A–Z are single tokens (ids
32–57). The prompt ends with the Qwen3.5 chat framing with thinking disabled:
`<|im_start|>assistant\n<think>\n\n</think>\n\n`. Option shuffling, layouts and
subsampling are described in [data.md](data.md).

## Packing

A training row of T tokens holds several segments. Each segment is one rendered item,
with its answer slot at its last token.

- **Segment ids.** Tokens carry `segment_ids` 1, 2, 3, … in contiguous runs, and padding
  (0) comes last. Both mixers isolate segments, and positions restart at 0 in every
  segment, so a packed row computes exactly what each segment would compute alone.
  `tests/model/test_parity.py` checks this against HF on separate sequences.
- **Fixed shapes.** Every step holds microbatches of a single sequence bucket (1024, 2048
  or 4096 tokens), so shapes stay static and each bucket compiles once.
- **Bins.** Packing is first-fit over 16 open bins per bucket, which cut padding from 10%
  to 6% compared with next-fit.
- **Loss normalisation.** A 4096-token step holds far fewer answer slots than a 1024-token
  one, so the gradient is divided by the *expected* slots per step (measured once on the
  packer). Every item then weighs the same whatever its bucket.

The training stream is a Grain pipeline. Segment i is a pure function of (seed, i), and the
packer state (`next_index` plus the segment indices in open rows) is a small JSON object.
A resumed run therefore sees exactly the batches an uninterrupted one would, whatever the
number of data workers.

## Code layout

| Package | Responsibility |
|---|---|
| `tjev.config` | Frozen dataclass schema with the recommended defaults; presets by name (`tpu-v6e`, `qwen35-2b`) with `extends`; `key=value` overrides; fail-closed coercion |
| `tjev.model` | Architecture config, HF import and stacking, layers, mixers, the scanned decoder |
| `tjev.kernels` | One package per op: an XLA reference and Pallas TPU kernels, plus the `shard_map` helper |
| `tjev.sharding` | The (data, fsdp) mesh, frozen-base placement, batch sharding |
| `tjev.data` | Items, rendering, tokenizer, packing, mixture, Grain pipeline, `sources/`, the mix builder, JevBench import and filter |
| `tjev.train` | Objective, optimizers and schedule, jitted step, Orbax checkpoints, metric sinks, the loop, the ahead-of-time memory check |
| `tjev.eval` | Metrics, calibration, eval sets, loading finished runs, post-training, reports |
| `tjev.export` | PEFT and merged export, reference logits, MLX |
| `tjev.campaign` | TPU planner and cost model, job queue, bench |
| `tjev.cli` | The `tjev` command, one Typer module per command group |

Design choices that follow the JAX AI stack:

- **Split once, jit a pure function.** The NNX model is split once into (graphdef, LoRA,
  frozen), and each step is a `jax.jit` of a pure function. The frozen base is an
  argument, never a closure constant; the LoRA and optimizer state are donated.
- **Checkpoints hold trainable state only.** They store adapters, optimizer state and the
  data-stream state (Orbax, async). The base is identified by the hash of its snapshot.
- **Run identities.** A run's identity is the fingerprint of the training config, base
  snapshot, data files and template version. It refuses resuming or branching into a
  different training.
