"""``tjev eval``, ``tjev calibrate``, ``tjev eval-base`` and ``tjev post``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tjev.cli.common import echo_json
from tjev.utils import write_json

Step = Annotated[int | None, typer.Option(help="checkpoint step (default: the selected one)")]
Out = Annotated[Path | None, typer.Option(help="write the full report here")]


def evaluate(
    run: Path,
    data: Path,
    step: Step = None,
    calibration: Annotated[Path | None, typer.Option(help="a calibration artifact")] = None,
    out: Out = None,
) -> None:
    """Grouped report of a trained run on DATA (JSONL items); prints the overall metrics."""
    from tjev.eval.runs import evaluate as run_eval

    report = run_eval(run, data, step=step, calibration=calibration)
    if out:
        write_json(out, report)
    echo_json(report["all"])


def calibrate(run: Path, data: Path, step: Step = None, out: Out = None) -> None:
    """Fit per-type temperatures of a run on DATA (the mix calibration split)."""
    from tjev.eval.runs import calibrate as run_calibrate

    echo_json(run_calibrate(run, data, step=step, out=out))


def eval_base(
    model: Path,
    data: Path,
    items: Annotated[list[str] | None, typer.Argument(help="presets and key=value")] = None,
    calibrate_on: Annotated[
        Path | None, typer.Option(help="fit per-type temperatures here (never on DATA)")
    ] = None,
    per_source: Annotated[int, typer.Option(help="first N items per source (0: all)")] = 0,
    out: Out = None,
) -> None:
    """Zero-shot control: the untrained base MODEL with the same prompt and readout."""
    from tjev.config import load_config, split_args
    from tjev.eval.runs import evaluate_base

    files, overrides = split_args(items or [])
    report = evaluate_base(
        model,
        data,
        calibrate_on=calibrate_on,
        cfg=load_config(*files, overrides=overrides),
        per_source=per_source,
    )
    if out:
        write_json(out, report)
    echo_json({k: report[k]["all"] for k in ("raw", "calibrated") if k in report})


def post(
    run: Path,
    mix: Annotated[Path, typer.Option(help="the built mix directory")],
    jevbench: Annotated[Path, typer.Option(help="JevBench public.jsonl")],
    zeroshot: Annotated[Path | None, typer.Option(help="`tjev eval-base` report")] = None,
    quick: Annotated[bool, typer.Option(help="skip the full validation pass")] = False,
) -> None:
    """After training: calibrate, evaluate (JevBench, held-out generators, validation), summary."""
    from tjev.eval.post import post_train

    typer.echo(
        post_train(run, mix=mix, jevbench=jevbench, zeroshot=zeroshot, quick=quick).read_text()
    )


def register(app: typer.Typer) -> None:
    app.command("eval")(evaluate)
    app.command()(calibrate)
    app.command("eval-base")(eval_base)
    app.command()(post)
