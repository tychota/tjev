"""``tjev serve``: the decision API (FastAPI + uvicorn, the ``serve`` extra)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def serve(
    run: Annotated[Path | None, typer.Option(help="a trained run directory (JAX)")] = None,
    model: Annotated[Path | None, typer.Option(help="a base model (JAX) or an MLX model")] = None,
    backend: Annotated[str, typer.Option(help="jax | mlx")] = "jax",
    calibration: Annotated[Path | None, typer.Option(help="a calibration artifact")] = None,
    step: Annotated[int | None, typer.Option(help="default: the selected step")] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_items: Annotated[int, typer.Option(help="items per micro-batch")] = 64,
    max_wait_ms: Annotated[float, typer.Option(help="how long a micro-batch collects")] = 5.0,
) -> None:
    """Serve POST /v1/systemone and GET /health."""
    import uvicorn

    from tjev.serve.app import create_app
    from tjev.serve.backends import JaxBackend, MlxBackend

    if (run is None) == (model is None):
        raise typer.BadParameter("give exactly one of --run and --model")
    if backend == "mlx":
        if model is None:
            raise typer.BadParameter("--backend mlx needs --model")
        scorer = MlxBackend(model_path=model, calibration=calibration)
    elif backend == "jax":
        scorer = JaxBackend(run_dir=run, model_path=model, calibration=calibration, step=step)
    else:
        raise typer.BadParameter("backend must be jax or mlx")
    app = create_app(scorer, max_items=max_items, max_wait_ms=max_wait_ms)
    uvicorn.run(app, host=host, port=port, log_level="info")


def register(app: typer.Typer) -> None:
    app.command()(serve)
