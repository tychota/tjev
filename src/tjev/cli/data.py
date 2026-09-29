"""``tjev data``: build the decision mixture and import the evaluation sets."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tjev.cli.common import comma_list, echo_json
from tjev.data.heldout import REPORT_SEED, SELECT_SEED

app = typer.Typer(help="Build the mix and import datasets.", no_args_is_help=True)


@app.command()
def build(
    out: Path,
    sources: Annotated[str, typer.Option(help="comma list of sources (default: all)")] = "",
    blocks: Annotated[str, typer.Option(help="comma list of blocks (default: all)")] = "",
    commercial_only: bool = False,
    scale: Annotated[float, typer.Option(help="multiplies every source's cap")] = 1.0,
    seed: int = 0,
    jevbench_public: Annotated[
        Path | None,
        typer.Option(help="public.jsonl of `tjev data jevbench` (contamination filter)"),
    ] = None,
) -> None:
    """Build the decision mixture into OUT (train/, validation, calibration, test/, mix.yaml)."""
    from tjev.data.mix import build as build_mix

    manifest = build_mix(
        out,
        sources=comma_list(sources),
        blocks=comma_list(blocks),
        commercial_only=commercial_only,
        scale=scale,
        seed=seed,
        jevbench_public=jevbench_public,
    )
    echo_json({k: manifest[k] for k in ("train_rows_by_lang", "fr_sampled_share", "mixture")})


@app.command()
def reweight(out: Path) -> None:
    """Recompute mix.yaml (the source weights) of a built mix."""
    from tjev.data.mix import reweight as reweight_mix

    echo_json(reweight_mix(out)["mixture"])


@app.command()
def jevbench(
    repo: Annotated[Path, typer.Argument(help="a checkout of github.com/fstandhartinger/jevbench")],
    out: Path,
) -> None:
    """Import the JevBench public items (final measurement only, never training)."""
    from tjev.data.jevbench import prepare

    echo_json(prepare(repo, out))


@app.command()
def heldout(
    out: Path,
    per: Annotated[int, typer.Option(help="items per generator")] = 20,
    seed: int = SELECT_SEED,
) -> None:
    """Held-out generator items (data.heldout: part of the checkpoint selection score)."""
    from tjev.data.heldout import write_heldout

    if seed == REPORT_SEED:
        raise typer.BadParameter(f"seed {REPORT_SEED} is the post-training reporting set")
    echo_json({"items": write_heldout(out, per, seed), "out": str(out)})
