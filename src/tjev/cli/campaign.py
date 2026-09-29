"""``tjev campaign plan | fit | select | cost | queue | bench``: the TPU campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from tjev.cli.common import comma_list, echo_json

app = typer.Typer(help="Plan, run and fit the TPU campaign.", no_args_is_help=True)

Runs = Annotated[Path, typer.Option(help="the runs directory")]
Mix = Annotated[str, typer.Option(help="the built mix directory")]
Models = Annotated[str, typer.Option(help="directory of Qwen3.5-<size> snapshots")]
Hardware = Annotated[str, typer.Option(help="v6e | v5e")]
Sizes = Annotated[str, typer.Option(help="final sizes, longest first")]
PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


def _campaign(runs: Path, mix: str, models: str, hardware: str, sizes: str, kernels: bool):
    from tjev.campaign.plan import HARDWARE, campaign_from_env

    if hardware not in HARDWARE:
        raise typer.BadParameter(f"hardware must be one of {sorted(HARDWARE)}")
    return campaign_from_env(runs, mix, models, hardware=hardware,
                             final_sizes=tuple(comma_list(sizes)), kernels=kernels)  # fmt: skip


@app.command()
def plan(
    phase: Annotated[str, typer.Argument(help="sweep | transfer | final")],
    runs: Runs,
    mix: Mix,
    models: Models,
    hardware: Hardware = "v6e",
    sizes: Sizes = "4B,2B,0.8B",
    kernels: Annotated[bool, typer.Option(help="Pallas TPU kernels (else the XLA paths)")] = True,
    out: Annotated[Path | None, typer.Option(help="write the queue here")] = None,
) -> None:
    """The queue of one phase (planned from the finished runs of the phases before)."""
    from tjev.campaign.plan import plan as make_plan

    text = (
        "\n".join(make_plan(phase, _campaign(runs, mix, models, hardware, sizes, kernels))) + "\n"
    )
    if out:
        out.write_text(text)
    else:
        typer.echo(text, nl=False)


@app.command()
def fit(runs: Runs, hardware: Hardware = "v6e", out: Path | None = None) -> None:
    """Decisions and per-size recipes from the finished runs."""
    from tjev.campaign.plan import fit as fit_runs

    report = fit_runs(_campaign(runs, "MIX", "MODELS", hardware, "4B,2B,0.8B", True))
    if out:
        out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    echo_json(report)


@app.command()
def select(runs: Runs) -> None:
    """Per size, the best final run (long run or cooldown branch)."""
    from tjev.campaign.plan import select as select_runs

    echo_json({size: r["name"] for size, r in select_runs(runs).items()})


@app.command()
def cost(
    runs: Runs = Path("runs"),
    hardware: Hardware = "v6e",
    sizes: Sizes = "4B,2B,0.8B",
    chips: Annotated[int, typer.Option(help="chips of the VM")] = 8,
    rate: Annotated[float | None, typer.Option(help="USD per chip-hour")] = None,
    bench: Annotated[Path | None, typer.Option(help="`tjev campaign bench` JSON")] = None,
) -> None:
    """Chip-hours, wall time on one VM and USD per phase (phases after the sweep are
    priced from the current fit, i.e. the priors before any run)."""
    from tjev.campaign.plan import SIZES, tokens_per_second
    from tjev.campaign.plan import cost as phase_cost

    c = _campaign(runs, "MIX", "MODELS", hardware, sizes, True)
    measured = json.loads(bench.read_text()) if bench else None
    table = phase_cost(c, bench=measured, vm_chips=chips, rate=rate)
    typer.echo(f"{'phase':10s} {'jobs':>5s} {'chip h':>8s} {'wall h':>7s} {'USD':>8s}   "
               f"({hardware}-{chips}, ${rate if rate is not None else c.hw['rate']}/chip-h)")  # fmt: skip
    for phase, row in table.items():
        typer.echo(f"{phase:10s} {row['jobs']:5d} {row['chip_hours']:8.1f} {row['wall_h']:7.2f} "
                   f"{row['usd']:8.2f}")  # fmt: skip
    for size in SIZES:
        typer.echo(f"  {size:5s} {tokens_per_second(c, size, measured):9.0f} tokens/s/chip")


@app.command(context_settings=PASSTHROUGH)
def queue(ctx: typer.Context) -> None:
    """Run a queue on this host (arguments: `tjev campaign queue --help`)."""
    from tjev.campaign.queue import main

    raise typer.Exit(main(ctx.args))


@app.command(context_settings=PASSTHROUGH)
def bench(ctx: typer.Context) -> None:
    """Kernel and train-step benchmarks on a TPU (arguments: `tjev campaign bench --help`)."""
    from tjev.campaign.bench import main

    raise typer.Exit(main(ctx.args))
