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
  scanned over super-blocks, with segment-aware packing. On a TPU v6e-1 in fp32 it matches
  the official model to a max |Δlogit| of 3.5e-5, with a worst gradient cosine of 0.9999998.
- **A GDN megakernel on TPU.** The Gated DeltaNet core, its conv, gates and norms run in
  one Pallas kernel with a fused backward (`mega2`), 3.2× the XLA chunked path at 0.8B and
  2.6× at 9B; with merged LoRA, 4× at 0.8B. Every iteration from XLA to the megakernel is
  measured on real training ([kernel-journey.md](docs/kernel-journey.md)). An XLA:TPU
  miscompile of the fused causal conv was found and worked around.
- **Training on TPU.** Exact resume, WSD cooldown branches, checkpoint selection by
  calibrated NLL, run-health kill rules, regularisers (label smoothing, KL to the base,
  CogCalib, teacher targets, EMA weights), LoRA inits (PiSSA, EVA, LoRA-GA), Muon.
- **Calibration built in.** Each question type gets its own temperature, bound to the
  checkpoint (and, on MLX, to the quantization) it was fitted on. A study of some twenty calibrators
  on the 4B finals keeps per-type temperature scaling ([calibration.md](docs/calibration.md)).
- **The mix-v7.5 data recipe.** Contenders' public JevBench-style data, code-labelled
  EN/FR decision generators, public datasets recast as rubric decisions, text analysis,
  French machine translation filtered by CometKiwi, and epoch tiers for 5,000 steps. Every
  training row is checked against the public JevBench items for contamination, and each
  mix ships a data-quality report ([data.md](docs/data.md)).
- **A TPU campaign that fits a budget.** A planner, a queue and a cost model, a headless
  Colab v6e-1 runner (chained time-limited sessions), and a staged 0.8B-proxy
  sweep whose results set the recipe ([sweep-plan.md](docs/sweep-plan.md),
  [results.md](docs/results.md)).
- **Evaluation two ways.** tjev's own eval loop (v1.3 and v1.5 scoring, two-order
  averaging, abstention policies) and the official JevBench harness querying tjev's
  server over HTTP ([official-eval.md](docs/official-eval.md)).

## Status

| Area | State |
|---|---|
| Model | TPU v6e-1 fp32 against the official Qwen3.5: every layer ≤ 1e-6, max \|Δlogit\| 3.5e-5, top-1 100%; the bf16 paths sit inside the official model's own bf16 spread (KL ≤ 2.9e-3, top-1 ≥ 99.4%) |
| Kernels | Pallas GDN megakernel (`mega2`, chunk 128, blocked inverse), splash attention, merged LoRA, all compiled and measured on v6e-1; TPU unit tests at fp32 level; interpret-mode tests on CPU |
| Training | The 0.8B sweep (4 stages, v6e-1) measured; 4B finals trained at horizons 1500 and 3000 (LoRA r64, WSD), their checkpoints calibrated and studied; every path tested end to end on CPU (4 virtual devices) |
| Data | mix-v7.5 (and its `-rep1` and commercial-only variants), built and reported; mix-v3 to mix-v7 kept as history with their reports under `reports/data/` |
| Eval, export and serving | v1.3 / v1.5 scoring, the official harness, calibration studies; PEFT adapters, merged HF snapshots, MLX conversion and parity; a FastAPI decision API with micro-batching and serving policies |

Zero-shot controls on the 231 public JevBench items (temperatures fitted on the mix's
calibration split, never on JevBench):

| Qwen3.5 | Accuracy | ECE (calibrated) | Hard tier |
|---|---|---|---|
| 0.8B | 0.532 | 0.100 | 0.36 |
| 2B | 0.584 | 0.129 | 0.42 |
| 4B | 0.779 | 0.062 | 0.61 |

Everything measured, with how, is in [docs/results.md](docs/results.md).

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # dev environment (CPU JAX, tests, parity, data)
uv run pytest -n 4 --dist loadfile        # CPU suite; hardware- and weight-dependent tests are marked

# data: the base model, JevBench public (measurement only), a mix built from scratch
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B
git clone --depth 1 https://github.com/fstandhartinger/jevbench jevbench
uv run tjev data jevbench jevbench data/jevbench
uv run tjev data build data/mix --jevbench-public data/jevbench/public.jsonl
uv run tjev data heldout data/mix/heldout-select.jsonl

# train (on a TPU host: `uv sync --extra tpu`), then evaluate, calibrate, export
uv run tjev compile-check tpu-v6e qwen35-2b data/mix/mix.yaml model.path=models/Qwen3.5-2B
uv run tjev train tpu-v6e qwen35-2b data/mix/mix.yaml model.path=models/Qwen3.5-2B \
    data.heldout=data/mix/heldout-select.jsonl name=q2b
uv run tjev post runs/q2b --mix data/mix --jevbench data/jevbench/public.jsonl
uv run tjev export runs/q2b runs/q2b/export

# on a Mac (uv sync --extra mlx)
uv run tjev mlx convert runs/q2b/export mlx/q2b-bf16 --quant bf16
uv run tjev mlx check mlx/q2b-bf16 runs/q2b/export/reference.json
uv run tjev mlx calibrate mlx/q2b-bf16 data/mix/calibration.jsonl mlx/q2b-bf16-cal.json

# serve the decisions (uv sync --extra serve), and score with the official harness
uv run tjev serve --run runs/q2b --calibration runs/q2b/post/calibration-jax.json
uv run tjev eval-official runs/q2b --out reports/official/q2b
```

`tjev data build` makes a mix-v3-style mix from the public sources. The finals' mix-v7.5 is
derived from mix-v7 by `scripts/build_mix_v75.sh` (`tjev data extend-v75`, then
`tjev data derive` for the tiers and variants); [docs/data.md](docs/data.md) has every step.

To run a campaign (sweep, transfer, final runs, post-training) on a TPU:
`TPU=v6e1 bash cloud/colab.sh all s0 sweep transfer final` (headless Colab sessions; see
[docs/tpu.md](docs/tpu.md)). `PROBE_ONLY=<mode> bash cloud/tpu_campaign.sh probe` runs one
probe (validate, bench, analyse, smoke, resume, journey, calib, official, finish) on any TPU
host; `PROBE_ONLY=analyse` profiles the final train step of every size
([docs/profiling.md](docs/profiling.md)).

## Documentation

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | The decoder, the decision readout, packing, the code layout |
| [docs/data.md](docs/data.md) | Item format, rendering, the mixes (mix-v3 to mix-v7.5), the v7 generators, sources and licenses, contamination checks |
| [docs/fr-data.md](docs/fr-data.md), [docs/fr-samples.md](docs/fr-samples.md) | French data and the 4B over-confidence; the French samples reviewed |
| [docs/training.md](docs/training.md) | Configuration, the recipe, long buckets, regularisers, selection, checkpoints and branches, logging |
| [docs/sweep-plan.md](docs/sweep-plan.md) | The staged sweep under the Colab v6e-1 budget, run-health rules, the W&B layout, the full-training plan |
| [docs/calibration.md](docs/calibration.md) | The calibration study on the 4B finals: per-type temperature against some twenty alternatives |
| [docs/tpu.md](docs/tpu.md) | The TPU campaign: phases, the Colab runner, the probes, costs |
| [docs/kernels.md](docs/kernels.md) | The gated delta rule, the Pallas TPU GDN kernels and megakernels, merged LoRA, splash, remat |
| [docs/kernel-journey.md](docs/kernel-journey.md) | Every GDN iteration measured on real training: speed, parity, profiles, compiler IR |
| [docs/profiling.md](docs/profiling.md) | The xprof capture protocol, the analyse probe, readable op names, how to read a summary |
| [docs/export.md](docs/export.md) | PEFT, merged snapshots, MLX conversion and calibration |
| [docs/serve.md](docs/serve.md) | The decision API: FastAPI, micro-batching, readout options, serving policies |
| [docs/official-eval.md](docs/official-eval.md) | The official JevBench harness against tjev's server; v1.5 scoring; the frozen-prompt control; Decision Index |
| [docs/model-card.md](docs/model-card.md) | The model card of the final adapters (HF `tychot/tjev-final`, private): recipe, data and licences, intended use, evaluation, limitations |
| [docs/results.md](docs/results.md) | Everything measured so far |
| [docs/research/](docs/research/) | The Jev landscape, LoRA and data notes, Muon on LoRA, LoRA optimizers, training notes, text analysis |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow |

## Layout

```
src/tjev/
  config/     typed run schema, presets (configs/hardware/, configs/models/), layered loading
  model/      Qwen3.5: HF import, LoRA and its inits, norms, RoPE, GDN, gated attention, MLP, scanned decoder
  kernels/    gated_delta_rule/ (XLA reference, Pallas v1 / v2 / megakernels), attention/ (splash), lora/, conv1d, shard_map
  data/       items, rendering, packing, Grain pipeline, sources/ (incl. gen_v7/), the mix builder, cleaning, JevBench
  train/      objective and regularisers, optimizers, jitted step, checkpoints, health rules, metrics, the loop
  eval/       metrics, calibration, eval sets, v1.5 scoring, the official harness, post-training, reports
  export/     PEFT and merged export, reference logits, MLX
  serve/      the /v1/systemone API (FastAPI, asyncio micro-batching), policies, JAX and MLX backends
  campaign/   TPU planner and cost model, job queue, bench, parity, kernel journey, profiling and IR analysis
  cli/        the `tjev` command (Typer)
cloud/        the Colab runner (colab.sh, session_run.py), tpu_campaign.sh (phases and probes)
scripts/      development tools: HF parity and ground truth, mix builds, data reports, studies
reports/      committed results: data-quality reports per mix, the calibration study
tests/        mirrors src/; CPU with 4 virtual devices
```

## License

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Qwen3.5 weights are Apache-2.0.
Each mix's manifest records the license of every data source. Some sources are
non-commercial (CC BY-NC), and `tjev data build --commercial-only` excludes them
(mix-v7.5c is the commercial-only derivation of the finals' mix).
