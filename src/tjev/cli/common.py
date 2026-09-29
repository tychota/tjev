"""Helpers shared by the command modules."""

from __future__ import annotations

import json

import typer


def echo_json(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, default=str))


def comma_list(value: str) -> list[str]:
    return [v for v in value.split(",") if v]
