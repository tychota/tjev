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
Proxy = Annotated[str, typer.Option(help="the sweep's proxy size (0.8B, 2B)")]
Centre = Annotated[float, typer.Option(help="the AdamW grid rate the arms pair with")]
Stage = Annotated[str, typer.Option(help="sweep stage: lr | arms | all | reg (4B regularisers)")]
Horizon = Annotated[int, typer.Option(help="final horizon in steps (0: the fitted one)")]
FinalRecipe = Annotated[
    str, typer.Option(help="final phase: decided (rank 16, CogCalib, EMA, h/2 branch) | generic")
]
FinalMix = Annotated[str, typer.Option(help="final runs on the sibling mix <mix>-NAME (e.g. rep1)")]
Lean = Annotated[
    bool, typer.Option(help="sweep economy: no quick eval, eval every 100, no post, health kill")
]
PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


def _campaign(runs: Path, mix: str = "MIX", models: str = "MODELS", *, hardware: str = "v6e",
              sizes: str = "4B,2B,0.8B", kernels: bool = True, seeds: int = 1,
              arm_seeds: int = 1, sweep_steps: int = 600, arms: str = "", muon: bool = True,
              proxy: str = "2B", centre: float | None = None, stage: str = "all",
              lean: bool = False, horizon: int = 0, max_bucket: int = 4096,
              final_recipe: str = "decided", final_mix: str = ""):  # fmt: skip
    from tjev.campaign.plan import ARMS, CENTRE, HARDWARE, REG_ARMS, SIZES, campaign_from_env

    if hardware not in HARDWARE:
        raise typer.BadParameter(f"hardware must be one of {sorted(HARDWARE)}")
    known = (*REG_ARMS, "nocentre") if stage == "reg" else ARMS  # nocentre: the arms only
    chosen = tuple(comma_list(arms)) or tuple(a for a in ARMS if stage != "reg")
    if unknown := set(chosen) - set(known):
        raise typer.BadParameter(f"unknown arms {sorted(unknown)} (arms: {', '.join(known)})")
    if proxy not in SIZES:
        raise typer.BadParameter(f"proxy must be one of {sorted(SIZES)}")
    try:
        return campaign_from_env(runs, mix, models, hardware=hardware, kernels=kernels,
                                 final_sizes=tuple(comma_list(sizes)), seeds=seeds,
                                 arm_seeds=arm_seeds, sweep_steps=sweep_steps, arms=chosen,
                                 muon=muon, proxy=proxy, centre=centre or CENTRE,
                                 stage=stage, lean=lean, horizon=horizon,
                                 max_bucket=max_bucket, final_recipe=final_recipe,
                                 final_mix=final_mix)  # fmt: skip
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


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
    proxy: Proxy = "2B",
    centre: Centre = 0.0,
    stage: Stage = "all",
    lean: Lean = False,
    horizon: Horizon = 0,
    max_bucket: Annotated[
        int, typer.Option(help="largest bucket: 8192 / 16384 add the long buckets (v6e1)")
    ] = 4096,
    final_recipe: FinalRecipe = "decided",
    final_mix: FinalMix = "",
    out: Annotated[Path | None, typer.Option(help="write the queue here")] = None,
) -> None:
    """The queue of one phase (planned from the finished runs of the phases before)."""
    from tjev.campaign.plan import plan as make_plan

    c = _campaign(runs, mix, models, hardware=hardware, sizes=sizes, kernels=kernels,
                  seeds=seeds, arm_seeds=arm_seeds, sweep_steps=sweep_steps, arms=arms,
                  muon=muon, proxy=proxy, centre=centre, stage=stage, lean=lean,
                  horizon=horizon, max_bucket=max_bucket, final_recipe=final_recipe,
                  final_mix=final_mix)  # fmt: skip
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
    proxy: Proxy = "2B",
    centre: Centre = 0.0,
    stage: Stage = "all",
    lean: Lean = False,
    horizon: Horizon = 0,
    final_recipe: FinalRecipe = "decided",
    final_mix: FinalMix = "",
    out: Path | None = None,
) -> None:
    """Decisions and per-size recipes from the finished runs (the planner's knobs as for
    ``plan``, so one BUDGET string drives plan, fit and cost)."""
    from tjev.campaign.plan import fit as fit_runs

    c = _campaign(runs, hardware=hardware, sizes=sizes, seeds=seeds, arm_seeds=arm_seeds,
                  sweep_steps=sweep_steps, arms=arms, muon=muon, proxy=proxy,
                  centre=centre, stage=stage, lean=lean, horizon=horizon,
                  final_recipe=final_recipe, final_mix=final_mix)  # fmt: skip
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
    proxy: Proxy = "2B",
    centre: Centre = 0.0,
    stage: Stage = "all",
    lean: Lean = False,
    horizon: Horizon = 0,
    final_recipe: FinalRecipe = "decided",
    final_mix: FinalMix = "",
    chips: Annotated[int, typer.Option(help="chips of the VM")] = 8,
    rate: Annotated[float | None, typer.Option(help="USD per chip-hour")] = None,
    bench: Annotated[Path | None, typer.Option(help="`tjev campaign bench` JSON")] = None,
) -> None:
    """Chip-hours, wall time on one VM and USD per phase (phases after the sweep are
    priced from the current fit, i.e. the priors before any run)."""
    from tjev.campaign.plan import SIZES, tokens_per_second
    from tjev.campaign.plan import cost as phase_cost

    c = _campaign(runs, hardware=hardware, sizes=sizes, seeds=seeds, arm_seeds=arm_seeds,
                  sweep_steps=sweep_steps, arms=arms, muon=muon, proxy=proxy, centre=centre,
                  stage=stage, lean=lean, horizon=horizon, final_recipe=final_recipe,
                  final_mix=final_mix)  # fmt: skip
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
def parity(ctx: typer.Context) -> None:
    """Kernels vs the XLA reference on real weights and data (`tjev campaign parity --help`)."""
    from tjev.campaign.parity import main

    raise typer.Exit(main(ctx.args))


@app.command(context_settings=PASSTHROUGH)
def bench(ctx: typer.Context) -> None:
    """Kernel and train-step benchmarks on a TPU (arguments: `tjev campaign bench --help`)."""
    from tjev.campaign.bench import main

    raise typer.Exit(main(ctx.args))
