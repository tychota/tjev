"""BucketPacker placements (train.packing): every segment whole in one row, no row over its
length or slots, deterministic, exact resume, and best_fit's tails filled by short
segments while "bucket" leaves them empty."""

import json
from typing import Any

import numpy as np
import pytest

from tjev.data.pack import PACKINGS, BucketPacker, Segment, check_segments

BUCKETS = (1024, 2048, 4096)
PACKING: dict[str, Any] = {"buckets": BUCKETS, "microbatch_tokens": 4096, "accumulation": 4,
                           "slots": 8, "labels": 26, "bins": 4}  # fmt: skip


def _lengths(count: int, seed: int = 0) -> np.ndarray:
    """A mix-like length profile: mostly short, a tail of 1-4k-token segments (and a few
    over the largest bucket, dropped)."""
    rng = np.random.default_rng(seed)
    short = rng.integers(60, 900, count)
    long = rng.integers(1025, 4300, count)
    return np.where(rng.random(count) < 0.25, long, short)


class Stream:
    """Segment i: its tokens are i's own id (to check segments are never cut or mixed)."""

    def __init__(self, lengths: np.ndarray, labels: int = 3):
        self.lengths, self.labels = lengths, labels

    def __call__(self, i: int) -> Segment:
        n = int(self.lengths[i])
        return Segment(tokens=np.full(n, i + 1, np.int32), label_ids=np.arange(self.labels,
                       dtype=np.int32), target=np.full(self.labels, 1 / self.labels, np.float32),
                       type_id=0, source_id=i % 5, lang_id=0, family_id=0, index=i)  # fmt: skip


def _steps(packer: BucketPacker, count: int) -> list:
    out = []
    while len(out) < count:
        step = packer.pop_step()
        if step is None:
            packer.place(packer.segment_at(packer.next_index))
        else:
            out.append(step)
    return out


def _packer(stream, packing, **kw) -> BucketPacker:
    return BucketPacker(stream, **(PACKING | kw), packing=packing)


def _pad(steps) -> float:
    return float(np.mean([1.0 - (s.segment_ids > 0).mean() for s in steps]))


@pytest.mark.parametrize("packing", PACKINGS)
def test_segments_are_whole_in_one_row_and_rows_never_overflow(packing):
    stream = Stream(_lengths(4000))
    packer = _packer(stream, packing)
    steps = _steps(packer, 40)
    seen: list[int] = []
    for step in steps:
        a, r, t = step.tokens.shape
        assert t in BUCKETS
        for m in range(a):
            check_segments(step.segment_ids[m])
            for row in range(r):
                ids = step.segment_ids[m, row]
                n_seg = int(ids.max())
                assert n_seg <= PACKING["slots"]
                assert (step.weight[m, row] > 0).sum() == n_seg
                for j in range(1, n_seg + 1):
                    where = np.flatnonzero(ids == j)
                    toks = step.tokens[m, row, where]
                    i = int(toks[0]) - 1  # the segment's index (its token value)
                    assert (toks == i + 1).all() and len(where) == stream.lengths[i]
                    assert (step.positions[m, row, where] == np.arange(len(where))).all()
                    assert step.slots[m, row, j - 1] == where[-1]  # the answer slot: its end
                    assert step.index[m, row, j - 1] == i
                    seen.append(i)
    # each segment trained at most once; with the in-flight ones, every index exactly once
    in_flight = [s.index for group in (*packer.open.values(), *packer.ready.values())
                 for b in group for s in b.segments()]  # fmt: skip
    dropped = packer.stats["dropped_too_long"]
    assert len(seen) == len(set(seen))
    assert sorted(seen + in_flight) == sorted(
        i for i in range(packer.next_index) if stream.lengths[i] <= max(BUCKETS)
    )
    assert dropped == sum(stream.lengths[: packer.next_index] > max(BUCKETS))


@pytest.mark.parametrize("packing", PACKINGS)
def test_same_stream_same_steps_and_exact_resume(packing):
    stream = Stream(_lengths(4000, seed=1))
    want = _steps(_packer(stream, packing), 30)
    again = _steps(_packer(stream, packing), 30)
    first = _packer(stream, packing)
    head = _steps(first, 11)
    state = json.loads(json.dumps(first.state()))  # what a checkpoint stores
    resumed = _packer(stream, packing)
    resumed.restore(state)
    got = head + _steps(resumed, 19)
    for w, a, g in zip(want, again, got, strict=True):
        for name in vars(w):
            x = getattr(w, name)
            if x is None:
                continue
            np.testing.assert_array_equal(x, getattr(a, name), err_msg=name)
            np.testing.assert_array_equal(x, getattr(g, name), err_msg=name)
    assert resumed.state() == _resumed_state(stream, packing, 30)


def _resumed_state(stream, packing, steps):
    packer = _packer(stream, packing)
    _steps(packer, steps)
    return packer.state()


def test_best_fit_fills_the_long_rows_with_short_segments():
    stream = Stream(_lengths(6000, seed=2))
    bucket = _steps(_packer(stream, "bucket"), 60)
    best = _steps(_packer(stream, "best_fit"), 60)

    def per_row(steps, length):
        rows = [s.segment_ids[m] for s in steps if s.tokens.shape[-1] == length
                for m in range(len(s.tokens))]  # fmt: skip
        return np.concatenate(rows).max(axis=-1)  # segments per row

    # "bucket": a segment over half a bucket is alone in its row
    assert (per_row(bucket, 2048) == 1).all() and (per_row(bucket, 4096) == 1).all()
    assert per_row(best, 2048).mean() > 1.5 and per_row(best, 4096).mean() > 1.5
    assert _pad(best) < 0.5 * _pad(bucket)


def test_best_fit_takes_the_tightest_row_and_opens_in_the_smallest_bucket():
    lengths = np.asarray([1500, 900, 300, 500, 100])
    packer = BucketPacker(Stream(lengths), buckets=(1024, 2048), microbatch_tokens=2048,
                          accumulation=8, slots=8, labels=26, bins=2, packing="best_fit")  # fmt: skip
    for i in range(5):
        packer.place(packer.segment_at(i))
        packer.next_index = i + 1
    (long,) = packer.open[2048]
    (short,) = packer.open[1024]
    # 1500 opens a 2048 row; 900 does not fit its 548-token tail: it opens a 1024
    # microbatch (two rows). 300: the 2048 tail (248 left) over the empty 1024 row (724
    # left). 500: only the empty 1024 row. 100: the tightest, the 900 row (24 left).
    assert [[s.index for s in row] for row in long.content] == [[0, 2]]
    assert [[s.index for s in row] for row in short.content] == [[1, 4], [3]]


def test_unknown_packing_is_refused():
    with pytest.raises(ValueError, match="packing"):
        BucketPacker(lambda i: None, **PACKING, packing="first_fit")
