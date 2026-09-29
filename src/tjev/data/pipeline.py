"""The training stream as a Grain pipeline (docs/data.md, "Training stream").

    MapDataset.range(next_index, ∞).map(SegmentAt.pair)      random access, 1:1
      .to_iter_dataset()  [.mp_prefetch(W workers)]           order independent of W
    → BucketStepIterDataset: bucket routing + RowBuilder rows + one-bucket steps [A,R,T]
    → ThreadPrefetchIterDataset                               host work overlaps the device

The stateful stage owns its parent pipeline and keeps our own state
``{next_index, open, ready}`` (the same JSON as before): a resumed run sees exactly the
batches of an uninterrupted run, whatever the number of workers or prefetch depth, and
checkpoints stay independent of the Grain version and of the host count.
"""

from __future__ import annotations

import grain

from tjev.data.pack import Batch, BucketPacker
from tjev.data.segments import SegmentAt

_UNBOUNDED = 2**62  # MapDataset.range stop: the stream never ends (steps decide)


class _BucketStepIterator(grain.DatasetIterator):
    def __init__(self, ds: BucketStepIterDataset):
        super().__init__()  # no Grain parents: this stage owns (and rebuilds) its pipeline
        self._ds = ds
        self.packer = BucketPacker(ds.segment_at, **ds.packing)
        self._it = None

    def _parent(self):
        if self._it is None:
            start = self.packer.next_index
            segments = grain.MapDataset.range(start, _UNBOUNDED).map(self._ds.segment_at.pair)
            pipeline = segments.to_iter_dataset(
                grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
            )
            if self._ds.workers:
                pipeline = pipeline.mp_prefetch(
                    grain.MultiprocessingOptions(
                        num_workers=self._ds.workers, per_worker_buffer_size=self._ds.worker_buffer
                    )
                )
            self._it = iter(pipeline)
        return self._it

    def __next__(self) -> Batch:
        while (batch := self.packer.pop_step()) is None:
            i, seg = next(self._parent())
            if i != self.packer.next_index:  # the pipeline before this stage must be 1:1
                raise RuntimeError(f"segment {i} arrived, expected {self.packer.next_index}")
            self.packer.place(seg)
        return batch

    def get_state(self) -> dict:
        return self.packer.state()

    def set_state(self, state: dict) -> None:
        self.close()  # rebuilt lazily at the restored next_index
        self.packer.restore(state)

    def close(self) -> None:
        if self._it is not None:
            close = getattr(self._it, "close", None)
            if close is not None:
                close()
            self._it = None


class BucketStepIterDataset(grain.IterDataset):
    def __init__(
        self, segment_at: SegmentAt, *, workers: int = 0, worker_buffer: int = 32, **packing
    ):
        super().__init__()
        self.segment_at, self.packing = segment_at, packing
        self.workers, self.worker_buffer = workers, worker_buffer

    def __iter__(self) -> _BucketStepIterator:
        return _BucketStepIterator(self)


def train_iterator(
    segment_at: SegmentAt,
    *,
    buckets: tuple[int, ...],
    microbatch_tokens: int,
    accumulation: int,
    slots: int,
    labels: int,
    workers: int = 0,
    worker_buffer: int = 32,
    prefetch: int = 2,
    bins: int = 1,
) -> grain.DatasetIterator:
    """Iterator of train steps. ``get_state()`` after ``next()`` is the state to checkpoint
    with that step (ThreadPrefetch hands out the state captured with each element)."""
    if workers:
        from absl import flags

        # Grain 0.2.18 mp_prefetch reads absl flags that a non-absl CLI never parses
        # (UnparsedFlagAccessError; fixed upstream after 0.2.18, google/grain#1355).
        if not flags.FLAGS.is_parsed():
            flags.FLAGS.mark_as_parsed()
    ds: grain.IterDataset = BucketStepIterDataset(
        segment_at,
        workers=workers,
        worker_buffer=worker_buffer,
        buckets=buckets,
        microbatch_tokens=microbatch_tokens,
        accumulation=accumulation,
        slots=slots,
        labels=labels,
        bins=bins,
    )
    if prefetch:
        ds = grain.experimental.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=prefetch)
    return iter(ds)
