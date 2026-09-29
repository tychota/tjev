"""Weighted mixture sampling, and the sequential reference driver of the training stream.

Segment ``i`` is a pure function of (seed, i) (tjev.data.segments.SegmentAt); the state
is the next index plus the indices in partially filled rows (tjev.data.pack.BucketPacker),
so a resumed run sees exactly the batches an uninterrupted run would have seen. Training
runs the same logic as a Grain pipeline (tjev.data.pipeline).
"""

from __future__ import annotations

import numpy as np

from tjev.data.item import Item
from tjev.data.pack import Batch, BucketPacker
from tjev.data.segments import SegmentAt


class Mixture:
    """Source sampling probabilities ∝ weight (explicit) or ∝ n^alpha (default)."""

    def __init__(
        self,
        sources: dict[str, list[Item]],
        weights: dict[str, float] | None = None,
        alpha: float = 0.5,
    ):
        if not sources:
            raise ValueError("empty mixture")
        self.names = sorted(sources)
        self.sources = sources
        weights = weights or {}
        unknown = set(weights) - set(self.names)
        if unknown:
            raise KeyError(f"mixture weights for unknown sources: {sorted(unknown)}")
        raw = np.asarray([weights.get(n, len(sources[n]) ** alpha) for n in self.names], np.float64)
        if (raw < 0).any() or raw.sum() <= 0:
            raise ValueError("invalid mixture weights")
        self.probs = raw / raw.sum()

    def draw(self, rng: np.random.Generator) -> tuple[str, int]:
        name = self.names[int(rng.choice(len(self.names), p=self.probs))]
        return name, int(rng.integers(len(self.sources[name])))

    def describe(self) -> dict[str, dict]:
        return {
            n: {"items": len(self.sources[n]), "prob": round(float(p), 5)}
            for n, p in zip(self.names, self.probs, strict=True)
        }


class TrainStream:
    """The training stream driven sequentially in-process: the reference for the Grain
    pipeline (tjev.data.pipeline), which runs the same SegmentAt and BucketPacker."""

    def __init__(self, mixture: Mixture, tok, vocab, *, seed: int, spec=None, **packing):
        self.packer = BucketPacker(SegmentAt(mixture, tok, vocab, seed=seed, spec=spec), **packing)

    def next_step(self) -> Batch:
        """One optimizer step: ``accumulation`` microbatches of one bucket, [A,R,T]."""
        while (batch := self.packer.pop_step()) is None:
            self.packer.place(self.packer.segment_at(self.packer.next_index))
        return batch

    def state(self) -> dict:
        return self.packer.state()

    def restore(self, state: dict) -> None:
        self.packer.restore(state)
