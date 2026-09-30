"""Micro-batching: concurrent requests are scored together, one backend call at a time.

Each request puts its items on an asyncio queue with a future. One worker task takes the
first waiting request, then keeps collecting for up to ``max_wait_ms`` or until
``max_items`` items, scores the lot in a single backend call (in a thread, so the event
loop keeps accepting requests), and resolves every future with its own slice. The
accelerator is used by one call at a time, and a packed call amortises a step's fixed cost
over every request in it.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

import numpy as np

from tjev.data.item import Item
from tjev.serve.backends import Backend


@dataclass
class _Pending:
    items: list[Item]
    future: asyncio.Future = field(repr=False)


@dataclass(frozen=True)
class Scored:
    probs: list[np.ndarray]
    prompt_tokens: int
    seconds: float
    batch_items: int


class MicroBatcher:
    def __init__(self, backend: Backend, *, max_items: int = 64, max_wait_ms: float = 5.0):
        self.backend = backend
        self.max_items = max_items
        self.max_wait = max_wait_ms / 1000
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def score(self, items: list[Item]) -> Scored:
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(_Pending(items, future))
        return await future

    async def _collect(self) -> list[_Pending]:
        batch = [await self._queue.get()]
        deadline = time.monotonic() + self.max_wait
        while sum(len(p.items) for p in batch) < self.max_items:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout))
            except TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        while True:
            batch = await self._collect()
            items = [item for p in batch for item in p.items]
            try:
                probs, usage = await asyncio.to_thread(self.backend.score, items)
            except Exception as e:  # every waiting request gets the error, the worker survives
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(e)
                continue
            start = 0
            for p in batch:
                n = len(p.items)
                share = sum(len(i.state) for i in p.items) / max(
                    1, sum(len(i.state) for i in items)
                )
                if not p.future.done():
                    p.future.set_result(Scored(probs[start : start + n],
                                               round(usage["prompt_tokens"] * share),
                                               usage["seconds"], len(items)))  # fmt: skip
                start += n
