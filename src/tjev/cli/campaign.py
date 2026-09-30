"""``tjev campaign plan | fit | select | cost | queue | bench``: the TPU campaign.

The budget knobs (``--seeds``, ``--arm-seeds``, ``--sweep-steps``, ``--arms``, ``--muon``)
must be the same for ``plan``, ``fit`` and ``cost`` of one campaign: ``fit`` reads the runs
the plan named.
"""

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
Seeds = Annotated[int, typer.Option(help="seeds of the AdamW grid (the centre gets one more)")]
ArmSeeds = Annotated[int, typer.Option(help="seeds of every arm and of the Muon grid")]
SweepSteps = Annotated[int, typer.Option(help="steps of a sweep run")]
Arms = Annotated[str, typer.Option(help="comma list of sweep arms (default: all)")]
Muon = Annotated[bool, typer.Option(help="sweep a Muon (Polar Express) rate bracket")]
PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


def _campaign(runs: Path, mix: str = "MIX", models: str = "MODELS", *, hardware: str = "v6e",
              sizes: str = "4B,2B,0.8B", kernels: bool = True, seeds: int = 1,
              arm_seeds: int = 1, sweep_steps: int = 600, arms: str = "", muon: bool = True):  # fmt: skip
    from tjev.campaign.plan import ARMS, HARDWARE, campaign_from_env

    if hardware not in HARDWARE:
        raise typer.BadParameter(f"hardware must be one of {sorted(HARDWARE)}")
    chosen = tuple(comma_list(arms)) or tuple(ARMS)
    if unknown := set(chosen) - set(ARMS):
        raise typer.BadParameter(f"unknown arms {sorted(unknown)} (arms: {', '.join(ARMS)})")
    return campaign_from_env(runs, mix, models, hardware=hardware, kernels=kernels,
                             final_sizes=tuple(comma_list(sizes)), seeds=seeds,
                             arm_seeds=arm_seeds, sweep_steps=sweep_steps, arms=chosen,
                             muon=muon)  # fmt: skip


@app.command()
def plan(
    phase: Annotated[str, typer.Argument(help="sweep | transfer | final")],
    runs: Runs,
    mix: Mix,
    models: Models,
    hardware: Hardware = "v6e",
    sizes: Sizes = "4B,2B,0.8B",
    kernels: Annotated[bool, typer.Option(help="Pallas TPU kernels (else the XLA paths)")] = True,
    seeds: Seeds = 1,
    arm_seeds: ArmSeeds = 1,
    sweep_steps: SweepSteps = 600,
    arms: Arms = "",
    muon: Muon = True,
    out: Annotated[Path | None, typer.Option(help="write the queue here")] = None,
) -> None:
    """The queue of one phase (planned from the finished runs of the phases before)."""
    from tjev.campaign.plan import plan as make_plan

    c = _campaign(runs, mix, models, hardware=hardware, sizes=sizes, kernels=kernels,
                  seeds=seeds, arm_seeds=arm_seeds, sweep_steps=sweep_steps, arms=arms,
                  muon=muon)  # fmt: skip
    text = "\n".join(make_plan(phase, c)) + "\n"
    if out:
        out.write_text(text)
    else:
        typer.echo(text, nl=False)


@app.command()
def fit(
    runs: Runs,
    hardware: Hardware = "v6e",
    sizes: Sizes = "4B,2B,0.8B",
    seeds: Seeds = 1,
    arm_seeds: ArmSeeds = 1,
    sweep_steps: SweepSteps = 600,
    arms: Arms = "",
    muon: Muon = True,
    out: Path | None = None,
) -> None:
    """Decisions and per-size recipes from the finished runs."""
    from tjev.campaign.plan import fit as fit_runs

    c = _campaign(runs, hardware=hardware, sizes=sizes, seeds=seeds, arm_seeds=arm_seeds,
                  sweep_steps=sweep_steps, arms=arms, muon=muon)  # fmt: skip
    report = fit_runs(c)
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
    seeds: Seeds = 1,
    arm_seeds: ArmSeeds = 1,
    sweep_steps: SweepSteps = 600,
    arms: Arms = "",
    muon: Muon = True,
    chips: Annotated[int, typer.Option(help="chips of the VM")] = 8,
    rate: Annotated[float | None, typer.Option(help="USD per chip-hour")] = None,
    bench: Annotated[Path | None, typer.Option(help="`tjev campaign bench` JSON")] = None,
) -> None:
    """Chip-hours, wall time on one VM and USD per phase (phases after the sweep are
    priced from the current fit, i.e. the priors before any run)."""
    from tjev.campaign.plan import SIZES, tokens_per_second
    from tjev.campaign.plan import cost as phase_cost

    c = _campaign(runs, hardware=hardware, sizes=sizes, seeds=seeds, arm_seeds=arm_seeds,
                  sweep_steps=sweep_steps, arms=arms, muon=muon)  # fmt: skip
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
