"""The HTTP service (FastAPI): ``POST /v1/systemone`` and ``GET /health``.

    tjev serve --run RUN [--calibration CAL.json]            # JAX, a trained run
    tjev serve --backend mlx --model MLX_DIR --calibration C # Apple silicon

Requests are async; their items are micro-batched (:mod:`tjev.serve.batcher`) into one
backend call at a time. Invalid rubrics are 400, schema errors 422, and a state too long
for the largest bucket is 413.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from tjev import __version__
from tjev.serve.backends import Backend
from tjev.serve.batcher import MicroBatcher
from tjev.serve.wire import DecisionRequest, DecisionResponse, Usage, answer, to_items


def create_app(backend: Backend, *, max_items: int = 64, max_wait_ms: float = 5.0) -> FastAPI:
    batcher = MicroBatcher(backend, max_items=max_items, max_wait_ms=max_wait_ms)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await batcher.start()
        yield
        await batcher.stop()

    app = FastAPI(title="tjev", version=__version__, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": backend.name}

    @app.post("/v1/systemone")
    async def systemone(request: DecisionRequest) -> DecisionResponse:
        try:
            named = to_items(request)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        try:
            scored = await batcher.score([item for _, item in named])
        except ValueError as e:  # e.g. a state longer than the largest bucket
            raise HTTPException(status_code=413, detail=str(e)) from e
        answers = {
            name: answer(item, p) for (name, item), p in zip(named, scored.probs, strict=True)
        }
        usage = Usage(prompt_tokens=scored.prompt_tokens, seconds=scored.seconds,
                      batch_items=scored.batch_items)  # fmt: skip
        return DecisionResponse(answers=answers, usage=usage, model=backend.name)

    return app
