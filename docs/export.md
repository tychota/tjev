# Export and MLX

## Export a run

```bash
tjev export RUN OUT [--step N] [--no-merged] [--no-reference] [--jevbench public.jsonl]
```

| File | Content |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | PEFT LoRA tensors under HF names (`base_model.model.model.language_model.…`), rsLoRA config |
| `model*.safetensors`, `config.json`, tokenizer files | The base snapshot with W + s·AB merged in fp32 and stored in bf16 (what `mlx_lm` converts) |
| `tjev_decision.json` | Template version, checkpoint identity, step, base model, letter token ids |
| `reference.json` | fp32 JAX letter logits (XLA reference kernels) on fixed held-out generator items, plus short JevBench public items with `--jevbench` |

The reference logits are computed after the bf16 model is released. They still hold a full
fp32 copy of the base, so on a small accelerator run the export with `JAX_PLATFORMS=cpu`.

## MLX on Apple silicon

```bash
uv sync --extra mlx          # on the Mac
tjev mlx convert EXPORT mlx/NAME-bf16 --quant bf16            # bf16 | 8bit | 4bit | 4bit-emb
tjev mlx check mlx/NAME-bf16 EXPORT/reference.json            # parity with JAX
tjev mlx calibrate mlx/NAME-bf16 data/mix-v3/calibration.jsonl mlx/NAME-bf16-cal.json
tjev mlx eval mlx/NAME-bf16 data/jevbench/public.jsonl --calibration mlx/NAME-bf16-cal.json
```

- **The prompt and the readout are tjev's.** Only the backend differs. The scorer reads the
  final-norm hidden state at the last prompt token and the letter rows of the tied
  embedding (`lm_head` for the untied 9B). `tjev.export.mlx` imports no JAX model code.
- **`check`** reports max |Δlogit|, top-1 agreement and KL against the fp32 JAX reference.
- **Calibrate on the converted model.** Quantization shifts the logits, so the temperatures
  are refitted there, on up to 2,000 items of the mix calibration split, never on JevBench.
  The artifact is bound to the exported checkpoint *and* the quantization
  (`<checkpoint>/mlx-<quant>`), and `tjev mlx eval` refuses any other model.
- **`eval`** reports accuracy, NLL, ECE by type and language, and the p50 / p95 latency of
  one decision (tokenize → forward → softmax over letters).

## Which quantization

Measured on an M3 Pro (18 GB) with the base models:

| Model | Advice | Why |
|---|---|---|
| 2B | bf16 | As fast as 4-bit, because prefill is compute-bound; 4-bit cost 0.035 JevBench accuracy |
| 4B | 8bit | bf16 swaps on 18 GB; 4-bit costs ~0.07 accuracy |
| both | try `4bit-emb` | 4-bit weights with the tied embedding (the answer readout) kept in bf16: the 4-bit readout rows are the likely cause of the logit errors (up to 5) |

Latency, p50 seconds per decision by prompt length:

| Model | JevBench acc / ECE | 100 | 300 | 500 | 1000 | 2000 tokens |
|---|---|---|---|---|---|---|
| 2B bf16 | 0.592 / 0.141 | 0.11 | 0.24 | 0.32 | 0.60 | 1.26 |
| 4B 4-bit | 0.711 / 0.173 | 0.23 | 0.52 | 0.86 | 1.72 | 3.64 |

The 2B meets 800 ms up to about 1,200 prompt tokens; the 4B up to about 450. MLX bf16
matches JAX: on 2B, top-1 agreement is 0.96 and KL 0.003.
