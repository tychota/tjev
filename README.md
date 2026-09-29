# tjev

Calibrated typed decisions from **Qwen3.5** label logits: LoRA fine-tuning in **JAX / Flax
NNX / Optax / Orbax / Grain**, with Pallas TPU kernels, and export to **MLX** for Apple
silicon.

A request gives a *state* (free text or JSON) and a bounded rubric: a question whose labels
are defined by criteria. It is either yes/no (`noul`), a choice among options, or an
ordinal score. The model reads the state once and returns a probability distribution over
the labels from the letter logits at a single answer slot. This is the JevBench "Jev-class"
task (<https://benchmarkheaven.com/jev-models>). tjev is an independent implementation, not
affiliated with TypeSafe AI or the JevBench maintainers (see [NOTICE](NOTICE)).

## Highlights

- **Qwen3.5 in Flax NNX, from scratch.** A hybrid Gated DeltaNet + gated attention decoder,
  scanned over super-blocks, with segment-aware packing. It matches HF transformers with a
  max |Δlogit| of 2.6e-5 in fp32 on the real 0.8B checkpoint.
- **Training on TPU.** A fused Pallas GDN forward, a Pallas reverse-recurrence backward and
  splash attention, all under `shard_map`. Exact resume, WSD cooldown branches, and
  checkpoint selection by calibrated NLL.
- **Calibration built in.** Each question type gets its own temperature, and the artifact is
  bound to the checkpoint (and, on MLX, to the quantization) it was fitted on.
- **The mix-v3 data recipe.** Contenders' public JevBench-style data, code-labelled EN/FR
  generators, public datasets recast as rubric decisions, and a small text-analysis block.
  Every training row is checked against the public JevBench items for contamination.
- **A TPU campaign that fits the budget.** A planner, a queue and cost estimates, with
  runners for free Kaggle v5e-8 sessions, Colab and GCP (flex-start, spot).

## Status

| Area | State |
|---|---|
| Model | HF parity on real checkpoints: fp32 0.8B max \|Δlogit\| 2.6e-5, top-1 100% (also with the Pallas kernels in interpret mode: 2.8e-5); bf16 0.8B / 2B / 4B top-1 100%, KL ≤ 1.2e-3 |
| Kernels | Pallas TPU GDN and splash attention match the XLA reference in interpret mode and lower for TPU. **They have not been compiled on a real TPU yet**: campaign phase `s0` tests them first and falls back to the XLA paths if they fail |
| Training | Every path is tested end to end on CPU (4 virtual devices): resume, cooldown branches, selection, the TPU kernels under `shard_map` |
| Data | The mix-v3 builder, the JevBench contamination filter, held-out generator sets |
| Export | PEFT adapters, a merged HF snapshot, MLX conversion, calibration and parity (mlx-lm) |

Zero-shot controls on the 231 public JevBench items (temperatures fitted on the mix's
calibration split, never on JevBench):

| Qwen3.5 | Accuracy | ECE (calibrated) | Hard tier |
|---|---|---|---|
| 0.8B | 0.532 | 0.100 | 0.36 |
| 2B | 0.584 | 0.129 | 0.42 |
| 4B | 0.779 | 0.062 | 0.61 |

For measured training results so far, see [docs/results.md](docs/results.md).

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # dev environment (CPU JAX, tests, parity, data)
uv run pytest -n 4 --dist loadfile        # CPU suite; hardware- and weight-dependent tests are marked

# data: the base model, JevBench public (measurement only), the mix
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B
git clone --depth 1 https://github.com/fstandhartinger/jevbench jevbench
uv run tjev data jevbench jevbench data/jevbench
uv run tjev data build data/mix-v3 --jevbench-public data/jevbench/public.jsonl
uv run tjev data heldout data/mix-v3/heldout-select.jsonl

# train (on a TPU host: `uv sync --extra tpu`), then evaluate, calibrate, export
uv run tjev compile-check tpu-v6e qwen35-2b data/mix-v3/mix.yaml model.path=models/Qwen3.5-2B
uv run tjev train tpu-v6e qwen35-2b data/mix-v3/mix.yaml model.path=models/Qwen3.5-2B \
    data.heldout=data/mix-v3/heldout-select.jsonl name=q2b
uv run tjev post runs/q2b --mix data/mix-v3 --jevbench data/jevbench/public.jsonl
uv run tjev export runs/q2b runs/q2b/export

# on a Mac (uv sync --extra mlx)
uv run tjev mlx convert runs/q2b/export mlx/q2b-bf16 --quant bf16
uv run tjev mlx check mlx/q2b-bf16 runs/q2b/export/reference.json
uv run tjev mlx calibrate mlx/q2b-bf16 data/mix-v3/calibration.jsonl mlx/q2b-bf16-cal.json
```

To run the whole campaign (sweep, transfer, final runs, post-training) on Kaggle's free
v5e-8, use `bash cloud/kaggle.sh all s0 sweep transfer final`; see [docs/tpu.md](docs/tpu.md).

## Documentation

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | The decoder, the decision readout, packing, the code layout |
| [docs/data.md](docs/data.md) | Item format, rendering, mix-v3, sources and licenses, contamination checks |
| [docs/training.md](docs/training.md) | Configuration, the recipe, selection, calibration, evaluation, logging |
| [docs/tpu.md](docs/tpu.md) | The TPU campaign: phases, Kaggle / Colab / GCP, costs |
| [docs/kernels.md](docs/kernels.md) | The gated delta rule and the Pallas TPU kernels |
| [docs/export.md](docs/export.md) | PEFT, merged snapshots, MLX conversion and calibration |
| [docs/results.md](docs/results.md) | Everything measured so far |
| [docs/research/](docs/research/) | Muon on LoRA, LoRA optimizers, training notes, text analysis |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow |

## Layout

```
src/tjev/
  config/     typed run schema, presets (hardware/, models/), layered loading
  model/      Qwen3.5: HF import, LoRA, norms, RoPE, GDN, gated attention, MLP, scanned decoder
  kernels/    gated_delta_rule/ and attention/ (pure-JAX reference + Pallas TPU), conv1d, shard_map
  data/       items, rendering, packing, Grain pipeline, sources/, the mix builder, JevBench
  train/      objective, optimizers, jitted step, checkpoints, metrics, the loop
  eval/       metrics, calibration, eval sets, run loading, post-training, reports
  export/     PEFT and merged export, reference logits, MLX
  campaign/   TPU planner and cost model, job queue, bench
  cli/        the `tjev` command (Typer)
cloud/        GCP, Kaggle and Colab runners
scripts/      development tools (HF parity on real checkpoints)
tests/        mirrors src/; CPU with 4 virtual devices
```

## License

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Qwen3.5 weights are Apache-2.0.
Each mix's manifest records the license of every data source. Some sources are
non-commercial (CC BY-NC), and `tjev data build --commercial-only` excludes them.
