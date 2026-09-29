"""The ``tjev`` command line (Typer): one sub-app per command group.

Commands that take a run config accept presets (named by stem: ``tpu-v6e``, ``qwen35-2b``,
or given as paths) and ``key=value`` overrides, in any order: ``tjev config tpu-v6e
qwen35-2b optim.lr=3e-5``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from tjev.cli import data, evaluation, training
from tjev.cli.common import echo_json

app = typer.Typer(
    name="tjev",
    help="Calibrated typed decisions from Qwen3.5 label logits (JAX / Flax NNX on TPU).",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
app.add_typer(data.app, name="data")
training.register(app)
evaluation.register(app)

ConfigItems = Annotated[
    list[str] | None, typer.Argument(help="presets / YAML files and key=value overrides")
]


@app.callback()
def _environment() -> None:
    # Persistent compilation cache: a 2B train step compiles in ~100 s cold, ~5 s warm;
    # every resume, eval and restart benefits.
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR", str(Path("~/.cache/tjev/jax-compile").expanduser())
    )


@app.command()
def config(items: ConfigItems = None) -> None:
    """Print the resolved run config as YAML."""
    import yaml

    from tjev.config import load_config, split_args

    files, overrides = split_args(items or [])
    typer.echo(yaml.safe_dump(load_config(*files, overrides=overrides).to_dict(), sort_keys=False))


@app.command()
def doctor() -> None:
    """JAX version, backend and devices."""
    import jax

    devices = jax.devices()
    echo_json(
        {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in devices],
            "device_kind": devices[0].device_kind,
        }
    )


def main() -> None:
    app()


__all__ = ["app", "main"]
