# Contributing

## Setup

```bash
uv sync                       # Python 3.12, CPU JAX, the dev group (pytest, ruff, ty, pre-commit)
uv run pre-commit install     # ruff check + format and ty on every commit
```

## Checks

The same gate runs in CI (`.github/workflows/ci.yml`):

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -n 4 --dist loadfile
```

## Tests

- **Four virtual CPU devices.** Tests run on CPU with 4 virtual devices
  (`tests/conftest.py`), so the sharded paths and the `shard_map` around the Pallas kernels
  run for real. The Pallas TPU kernels run in interpret mode and are also lowered for TPU
  with `jax.export`.
- **Tests never skip.** A skip is a failure. Tests that need hardware or real weights are
  *deselected* by marker instead:
  - `real`: real Qwen3.5 snapshots under `TJEV_MODELS` (default `./models`):
    `uv run pytest -m real`.
  - `tpu`: the compiled kernels on a TPU VM: `JAX_PLATFORMS=tpu pytest -m tpu`.
  - `slow`: trains the tiny fixture end to end. It runs by default; add `-m "not slow"`
    for a quick pass.
- **The fixture.** `tjev.testing.make_tiny_snapshot` writes a random Qwen3.5-format
  snapshot with a byte-level tokenizer; `tiny_items` makes learnable synthetic decisions.
- **Pinned outputs.** Some outputs are training data and are pinned by hash: the generator
  output (`tests/data/test_generators.py`) and the packed stream
  (`tests/data/test_pipeline.py`). Change a hash only together with a deliberate data change,
  and say so in the commit message.
- **Real checkpoints.** For changes to the model or the kernels, also run
  `uv run python scripts/check_parity.py models/Qwen3.5-0.8B --dtype float32`.

## Conventions

- **Imports.** Absolute everywhere (`from tjev.data.item import …`); `__init__.py` files
  re-export their package's modules relatively.
- **Types.** The source is type-checked strictly by ty. Annotate public functions, and use
  `jax.Array` and `jax.typing.ArrayLike`.
- **Configuration.** New run options go into `tjev.config.schema` with a comment saying what
  they change and why the default is what it is. Anything that changes the training (not
  just its logging or cadence) becomes part of the run identity automatically.
- **Rendered prompts.** If the rendered prompt changes, bump `TEMPLATE_VERSION` in
  `tjev.data.render`: calibration artifacts and run identities bind to it.
- **Docs.** Docs live in `docs/` and describe the code as it is; measured numbers go to
  `docs/results.md` with how they were measured.
- **Commits.** One logical change per commit, with a subject line naming the area and a body
  listing what changed and why.
