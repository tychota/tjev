"""Helpers shared by the command modules."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from tjev.data.render import NoulOrder
    from tjev.serve.policy import ServingPolicy


def echo_json(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, default=str))


def comma_list(value: str) -> list[str]:
    return [v for v in value.split(",") if v]


Orders = Annotated[
    int,
    typer.Option(
        min=1,
        help="order averaging: score K option orderings per question and average the "
        "probabilities (1: off; 2: + the reversed order / the noul swap; >2: + cyclic shifts)",
    ),
]
NoulOrderOpt = Annotated[
    str,
    typer.Option(help="noul option order: shuffled (by item / question hash) | fixed (no first)"),
]
Abstention = Annotated[
    str,
    typer.Option(
        help="noul abstention-band policy: off | field (add committed / decision) | noul "
        "(also report the committed value as `noul`, the field JevBench scores)"
    ),
]
Band = Annotated[tuple[float, float], typer.Option(help="the abstention band: LOW HIGH")]
SnapOpt = Annotated[
    str, typer.Option(help="an in-band P(yes) goes to the nearest edge | the argmax side")
]
CommitMargin = Annotated[
    float,
    typer.Option(help="commit an in-band P(yes) only when |P(yes) - 0.5| > MARGIN (0: always)"),
]
CommitEps = Annotated[
    float, typer.Option(help="commit to LOW - EPS / HIGH + EPS instead of the band edge")
]
NoulTemperature = Annotated[
    float, typer.Option(help="noul serving temperature, after calibration (<1 sharpens; 1: off)")
]
ScoreTemperature = Annotated[
    float,
    typer.Option(help="score serving temperature, after calibration (<1 sharpens; 1: off)"),
]


def noul_order_value(value: str) -> NoulOrder:
    from tjev.data.render import NOUL_ORDERS

    if value not in NOUL_ORDERS:
        raise typer.BadParameter(f"--noul-order must be one of {', '.join(NOUL_ORDERS)}")
    return value


def serving_policy(
    mode: str,
    band: tuple[float, float],
    snap: str,
    margin: float = 0.0,
    eps: float = 0.0,
    noul_temperature: float = 1.0,
    score_temperature: float = 1.0,
) -> ServingPolicy | None:
    """The serving policy of the CLI options (None: all off; docs/serve.md)."""
    from tjev.serve.policy import AbstentionPolicy, ServingPolicy, as_policy

    if mode not in ("off", "field", "noul"):
        raise typer.BadParameter("--abstention must be off, field or noul")
    if snap not in ("nearest", "argmax"):
        raise typer.BadParameter("--snap must be nearest or argmax")
    temps = {k: t for k, t in (("noul", noul_temperature), ("score", score_temperature))
             if t != 1.0}  # fmt: skip
    try:
        abstention = None
        if mode != "off":
            abstention = AbstentionPolicy(low=band[0], high=band[1], snap=snap,
                                          into_noul=mode == "noul", margin=margin, eps=eps)  # fmt: skip
        return as_policy(ServingPolicy(abstention=abstention, temperatures=temps))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
