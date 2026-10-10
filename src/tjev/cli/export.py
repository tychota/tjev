"""``tjev export``, ``tjev soup`` and ``tjev mlx convert | check | calibrate | eval``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from tjev.cli.common import NoulOrderOpt, Orders, echo_json, noul_order_value

mlx_app = typer.Typer(
    help="Apple silicon: convert, check, calibrate, evaluate.", no_args_is_help=True
)
Model = Annotated[Path, typer.Argument(help="an MLX model folder (tjev mlx convert)")]


def export(
    run: Path,
    out: Path,
    step: Annotated[int | None, typer.Option(help="default: the selected step")] = None,
    merged: Annotated[bool, typer.Option(help="also write the merged HF snapshot")] = True,
    reference: Annotated[bool, typer.Option(help="fp32 JAX reference logits")] = True,
    jevbench: Annotated[Path | None, typer.Option(help="add public items to the reference")] = None,
) -> None:
    """PEFT adapters, the merged HF snapshot and reference logits of a trained run."""
    from tjev.export.peft import export as run_export

    echo_json(
        run_export(run, out, step=step, merged=merged, reference=reference, jevbench=jevbench)
    )


def soup(
    members: Annotated[
        list[str], typer.Argument(help="RUN[@STEP][:ema] ... (ema: the EMA adapters)")
    ],
    out: Annotated[Path, typer.Option(help="the soup's run directory (must not exist)")],
    greedy: Annotated[
        bool, typer.Option(help="greedy soup: keep a member only if the held-out metric improves")
    ] = False,
    select_on: Annotated[
        list[Path] | None,
        typer.Option(help="greedy: held-out data (JSONL files / directories, never JevBench)"),
    ] = None,
    metric: Annotated[str, typer.Option(help="greedy: nll-cal | nll | accuracy")] = "nll-cal",
    orders: Orders = 1,
    noul_order: NoulOrderOpt = "shuffled",
) -> None:
    """Weight-space average of LoRA runs (the mean ΔW, exact at rank Σ r_i) as a run
    directory that `tjev post` / `tjev eval` / `tjev export` take."""
    from tjev.eval.soup import soup as make_soup

    if greedy and not select_on:
        raise typer.BadParameter("--greedy needs --select-on")
    on = (
        (select_on[0] if len(select_on) == 1 else list(select_on)) if greedy and select_on else None
    )
    echo_json(make_soup(members, out, greedy_on=on, metric=metric, orders=orders,
                        noul_order=noul_order_value(noul_order)))  # fmt: skip


@mlx_app.command()
def convert(
    export_dir: Annotated[Path, typer.Argument(help="a merged export (tjev export)")],
    out: Path,
    quant: Annotated[str, typer.Option(help="bf16 | 8bit | 4bit | 4bit-emb")] = "bf16",
) -> None:
    """Convert a merged export to MLX."""
    from tjev.export.mlx import convert as mlx_convert

    typer.echo(str(mlx_convert(export_dir, out, quant)))


@mlx_app.command()
def check(model: Model, reference: Annotated[Path, typer.Argument(help="reference.json")]) -> None:
    """Letter-logit parity of an MLX model against the fp32 JAX reference."""
    from tjev.export.mlx import Scorer, parity

    echo_json(parity(Scorer(model), json.loads(reference.read_text(encoding="utf-8"))))


@mlx_app.command()
def calibrate(
    model: Model,
    data: Annotated[Path, typer.Argument(help="the mix calibration split (JSONL)")],
    out: Path,
    limit: Annotated[int, typer.Option(help="items sampled for the fit")] = 2000,
    orders: Orders = 1,
    noul_order: NoulOrderOpt = "shuffled",
) -> None:
    """Per-type temperatures fitted on the MLX model (quantization shifts the logits)."""
    import numpy as np

    from tjev.data.item import read_jsonl
    from tjev.export.mlx import Scorer, fit_calibration

    items = list(read_jsonl(data))
    if len(items) > limit:
        keep = sorted(np.random.default_rng(0).choice(len(items), limit, replace=False))
        items = [items[i] for i in keep]
    art = fit_calibration(
        Scorer(model), items, out, orders=orders, noul_order=noul_order_value(noul_order)
    )
    echo_json(art["temperatures"])


@mlx_app.command("eval")
def evaluate(
    model: Model,
    data: Path,
    calibration: Annotated[Path | None, typer.Option(help="an MLX calibration artifact")] = None,
    out: Annotated[Path | None, typer.Option(help="write the full report here")] = None,
    orders: Orders = 1,
    noul_order: NoulOrderOpt = "shuffled",
) -> None:
    """Quality and per-decision latency of an MLX model on DATA."""
    from tjev.data.item import read_jsonl
    from tjev.data.render import TEMPLATE_VERSION
    from tjev.eval.calibrate import validate
    from tjev.export.mlx import Scorer, mlx_identity
    from tjev.export.mlx import evaluate as mlx_evaluate
    from tjev.utils import write_json

    temperatures = None
    if calibration:
        art = json.loads(calibration.read_text(encoding="utf-8"))
        temperatures = validate(art, checkpoint=mlx_identity(model), template=TEMPLATE_VERSION,
                                backend="mlx", orders=orders, noul_order=noul_order_value(noul_order))  # fmt: skip
    report = mlx_evaluate(Scorer(model), list(read_jsonl(data)), temperatures, orders=orders,
                          noul_order=noul_order_value(noul_order))  # fmt: skip
    if out:
        write_json(out, report)
    echo_json(report["all"])


def register(app: typer.Typer) -> None:
    app.command()(export)
    app.command()(soup)
    app.add_typer(mlx_app, name="mlx")
