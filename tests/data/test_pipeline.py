"""The Grain training stream (tjev.data.pipeline) against the reference TrainStream:
identical steps for any worker count, exact resume across worker counts, same state."""

import json
from typing import Any

import numpy as np
import pytest

from tjev.data.item import parse_item
from tjev.data.mixture import Mixture, TrainStream
from tjev.data.pack import Vocab
from tjev.data.pipeline import train_iterator
from tjev.data.segments import SegmentAt
from tjev.data.tokenize import PromptTokenizer
from tjev.testing import make_tiny_snapshot, tiny_items

# tiny segments are 276-468 tokens: these buckets split them ~50/25/25 (routing exercised)
PACKING: dict[str, Any] = {
    "buckets": (320, 400, 1024),
    "microbatch_tokens": 2048,
    "accumulation": 2,
    "slots": 8,
    "labels": 26,
}
SEED = 5


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("grain")
    make_tiny_snapshot(root / "model")
    tok = PromptTokenizer(root / "model")
    items = [parse_item(r) for r in tiny_items(300, seed=4)]
    sources: dict = {}
    for it in items:
        sources.setdefault(it.source, []).append(it)
    return Mixture(sources), tok, Vocab.from_items(items)


def _reference(setup, steps):
    mixture, tok, vocab = setup
    stream = TrainStream(mixture, tok, vocab, seed=SEED, **PACKING)
    return [stream.next_step() for _ in range(steps)], stream


def _grain(setup, workers=0, prefetch=2):
    mixture, tok, vocab = setup
    return train_iterator(
        SegmentAt(mixture, tok, vocab, seed=SEED), workers=workers, prefetch=prefetch, **PACKING
    )


def _assert_same(want, got):
    assert want.shape_key == got.shape_key
    for name in vars(want):
        a, b = getattr(want, name), getattr(got, name)
        assert a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=name)


@pytest.mark.parametrize("workers", [0, 2])
def test_grain_stream_equals_trainstream(setup, workers):
    want, _ = _reference(setup, 8)
    it = _grain(setup, workers=workers)
    try:
        got = [next(it) for _ in range(8)]
    finally:
        it.close()
    assert len({b.shape_key for b in want}) > 1  # several buckets were exercised
    for w, g in zip(want, got, strict=True):
        _assert_same(w, g)


def test_grain_resume_exact_across_workers(setup):
    want, _ = _reference(setup, 6)
    it = _grain(setup, workers=0)
    try:
        for _ in range(3):
            next(it)
        state = json.loads(json.dumps(it.get_state()))
    finally:
        it.close()
    ref = TrainStream(*setup, seed=SEED, **PACKING)
    for _ in range(3):
        ref.next_step()
    assert state == ref.state()  # the same portable state as the reference stream
    resumed = _grain(setup, workers=2)
    try:
        resumed.set_state(state)
        for w in want[3:]:
            _assert_same(w, next(resumed))
    finally:
        resumed.close()


def test_state_is_paired_with_the_prefetched_step(setup):
    plain, deep = _grain(setup, prefetch=0), _grain(setup, prefetch=3)
    try:
        for _ in range(4):
            _assert_same(next(plain), next(deep))
            assert plain.get_state() == deep.get_state()
    finally:
        plain.close()
        deep.close()


# Stream output is training data: this pins it across refactors (update only together with
# a deliberate data change). Equal to the jev 0.5 stream without its teacher fields.
STREAM_HASH = "39f9ed95062917c9"


def test_stream_output_is_pinned(setup):
    import hashlib

    it = _grain(setup)
    digest = hashlib.sha256()
    try:
        for _ in range(8):
            batch = next(it)
            for name in sorted(vars(batch)):
                digest.update(name.encode())
                digest.update(getattr(batch, name).tobytes())
    finally:
        it.close()
    assert digest.hexdigest()[:16] == STREAM_HASH


def _padding(batches):
    return float(np.mean([1.0 - (b.segment_ids > 0).mean() for b in batches]))


def test_multi_bin_packing_resumes_exactly_and_pads_less(setup):
    mixture, tok, vocab = setup
    packing: dict[str, Any] = {
        **PACKING,
        "buckets": (1024,),
    }  # one bucket: next-fit wastes row tails

    def iterator(bins):
        return train_iterator(SegmentAt(mixture, tok, vocab, seed=SEED), bins=bins, **packing)

    one, many = iterator(1), iterator(8)
    try:
        base = [next(one) for _ in range(6)]
        packed = [next(many) for _ in range(3)]
        state = json.loads(json.dumps(many.get_state()))
        packed += [next(many) for _ in range(3)]
    finally:
        one.close()
        many.close()
    assert _padding(packed) < _padding(base)
    resumed = iterator(8)
    try:
        resumed.set_state(state)
        for w in packed[3:]:
            _assert_same(w, next(resumed))
    finally:
        resumed.close()
