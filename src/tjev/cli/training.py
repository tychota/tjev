"""``tjev train`` and ``tjev compile-check``."""

from __future__ import annotations

from typing import Annotated

import typer

from tjev.cli.common import echo_json

ConfigItems = Annotated[
    list[str] | None, typer.Argument(help="presets / YAML files (mix.yaml) and key=value overrides")
]


def train(items: ConfigItems = None) -> None:
    """Train (or resume) a run: `tjev train tpu-v6e qwen35-2b MIX/mix.yaml model.path=… name=…`."""
    from tjev.config import load_config, split_args
    from tjev.train.loop import train as run

    files, overrides = split_args(items or [])
    echo_json(run(load_config(*files, overrides=overrides)))


def compile_check(items: ConfigItems = None) -> None:
    """Compile the train step per bucket ahead of time and report its memory (no weights)."""
    from tjev.config import load_config, split_args
    from tjev.train.compile_check import compile_check as check

    files, overrides = split_args(items or [])
    echo_json(check(load_config(*files, overrides=overrides)))


def register(app: typer.Typer) -> None:
    app.command()(train)
    app.command("compile-check")(compile_check)
