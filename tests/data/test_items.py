"""Items, rendering, segments, packing, the reference stream and the mixture."""

import json
from typing import Any

import numpy as np
import pytest

from tjev.data.item import parse_item
from tjev.data.mixture import Mixture, TrainStream
from tjev.data.pack import RowBuilder, Vocab, check_segments, make_segment, pack_eval
from tjev.data.render import RenderSpec, render
from tjev.data.tokenize import PromptTokenizer
from tjev.testing import make_tiny_snapshot, tiny_items


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tok")
    make_tiny_snapshot(folder)
    return PromptTokenizer(folder)


def test_parse_rejects_bad_items():
    good = tiny_items(3)[0]
    parse_item(good)
    for mutate in (
        lambda r: r.update(expected="purple"),
        lambda r: r["question"].update(type="rank"),
        lambda r: r.update(state=" "),
        lambda r: r.update(target={"red": 0.5}),
    ):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        with pytest.raises((ValueError, KeyError)):
            parse_item(bad)


def test_noul_and_score_labels():
    noul = parse_item(tiny_items(3)[1])
    assert noul.question.labels == ("no", "yes")
    score = parse_item(tiny_items(3)[2])
    assert score.question.labels == ("0", "1", "2", "3")


def test_render_train_shuffles_and_eval_is_stable():
    item = parse_item(tiny_items(1)[0])
    spec = RenderSpec(train=False)
    assert render(item, spec) == render(item, spec)
    orders = {tuple(render(item, RenderSpec(), np.random.default_rng(s))[1]) for s in range(30)}
    assert len(orders) > 1
    for s in range(30):  # sub-sampling never drops the gold option
        _, order = render(item, RenderSpec(subsample_prob=1.0), np.random.default_rng(s))
        assert item.gold in order


def test_score_keeps_level_order():
    item = parse_item(tiny_items(3)[2])
    for s in range(10):
        assert render(item, RenderSpec(), np.random.default_rng(s))[1] == [0, 1, 2, 3]


def test_segment_targets_follow_displayed_order(tok):
    item = parse_item(tiny_items(1)[0])
    rng = np.random.default_rng(3)
    seg = make_segment(item, tok, RenderSpec(), Vocab.from_items([item]), 0, rng)
    rng = np.random.default_rng(3)
    _, order = render(item, RenderSpec(), rng)
    assert seg.target.argmax() == order.index(item.gold)
    assert seg.tokens[-1] != tok.letter_ids[0]  # slot is the last prompt token, not a letter


def test_packing_isolates_segments(tok):
    items = [parse_item(r) for r in tiny_items(20)]
    vocab = Vocab.from_items(items)
    segs = [make_segment(it, tok, RenderSpec(train=False), vocab, i) for i, it in enumerate(items)]
    batches, dropped = pack_eval(segs, (1024, 2048), 4096, 8, 26)
    assert not dropped
    total = sum(int(b.weight.sum()) for b in batches)
    assert total == len(items)
    for b in batches:
        for r in range(b.tokens.shape[0]):
            for s in range(int((b.weight[r] > 0).sum())):
                slot = b.slots[r, s]
                assert b.segment_ids[r, slot] == s + 1
                assert slot + 1 == b.tokens.shape[1] or b.segment_ids[r, slot + 1] != s + 1
            assert (
                b.positions[r][b.segment_ids[r] == 1] == np.arange((b.segment_ids[r] == 1).sum())
            ).all()


def test_row_builder_rejects_oversize(tok):
    item = parse_item(tiny_items(1)[0])
    seg = make_segment(item, tok, RenderSpec(train=False), Vocab.from_items([item]), 0)
    with pytest.raises(ValueError):
        RowBuilder(1, 16, 4, 26).add(seg)


def test_stream_resume_is_exact(tok):
    items = [parse_item(r) for r in tiny_items(200)]
    sources = {}
    for it in items:
        sources.setdefault(it.source, []).append(it)
    vocab = Vocab.from_items(items)

    def stream():
        return TrainStream(
            Mixture(sources),
            tok,
            vocab,
            buckets=(1024,),
            microbatch_tokens=4096,
            accumulation=2,
            slots=8,
            labels=26,
            seed=5,
        )

    a = stream()
    first = [a.next_step() for _ in range(3)]
    state = a.state()
    rest = [a.next_step() for _ in range(3)]
    b = stream()
    b.restore(json.loads(json.dumps(state)))
    for want, got in zip(rest, [b.next_step() for _ in range(3)], strict=True):
        for name in vars(want):
            np.testing.assert_array_equal(getattr(want, name), getattr(got, name))
    del first


def test_mixture_weights():
    items: dict[str, list[Any]] = {"a": [None] * 100, "b": [None] * 400}
    m = Mixture(items)  # n^0.5: 10 vs 20
    np.testing.assert_allclose(m.probs, [1 / 3, 2 / 3])
    m = Mixture(items, {"a": 3.0, "b": 1.0})
    np.testing.assert_allclose(m.probs, [0.75, 0.25])
    with pytest.raises(KeyError):
        Mixture(items, {"c": 1.0})


def test_segment_contract():
    check_segments(np.array([[1, 1, 2, 2, 2, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]))
    for bad in ([1, 2, 1, 0], [0, 1, 1, 0], [1, 3, 3, 0], [2, 2, 0, 0], [1, 0, 1, 0]):
        with pytest.raises(ValueError, match="contiguous"):
            check_segments(np.array([bad]))


def _segment(n: int, index: int):
    from tjev.data.pack import Segment

    return Segment(tokens=np.ones(n, np.int32), label_ids=np.arange(2, dtype=np.int32),
                   target=np.asarray([1.0, 0.0], np.float32), type_id=0, source_id=0,
                   lang_id=0, family_id=0, index=index)  # fmt: skip


def test_buckets_longer_than_the_microbatch_take_one_row_per_device():
    from tjev.data.pack import BucketPacker, rows_for

    assert rows_for(1024, 4096) == 4 and rows_for(16384, 4096) == 1
    assert rows_for(1024, 4 * 4096, devices=4) == 16  # per-device budget × devices
    assert rows_for(16384, 4 * 4096, devices=4) == 4  # one long row per device
    segs = [_segment(n, i) for i, n in enumerate([9000, 700, 15000, 5000, 300])]
    packer = BucketPacker(lambda i: segs[i], buckets=(1024, 8192, 16384),
                          microbatch_tokens=4096, accumulation=1, slots=8, labels=26)  # fmt: skip
    for seg in segs:
        packer.place(seg)
    step = packer.pop_step()  # 9000 then 15000 tokens: one 16384 row each, never together
    assert step is not None and step.tokens.shape == (1, 1, 16384)
    assert [len(r) for r in packer.open[16384][0].content] == [1]
    assert packer.stats.get("dropped_too_long", 0) == 0
    batches, dropped = pack_eval(segs, (1024, 8192, 16384), 4096, 8, 26, devices=2)
    assert not dropped
    assert {b.tokens.shape for b in batches} == {(2, 16384), (2, 8192), (4, 1024)}


def test_train_spec_validates_buckets_and_microbatch():
    from tjev.config.schema import TrainSpec

    spec = TrainSpec(seq_buckets=(1024, 2048, 4096, 8192, 16384), microbatch_tokens=4096)
    assert spec.microbatch_tokens_at(16384) == 16384 and spec.microbatch_tokens_at(1024) == 4096
    for bad in ((2048, 1024), (1024, 1024), (1000,), ()):
        with pytest.raises(ValueError, match="seq_buckets"):
            TrainSpec(seq_buckets=bad)
    with pytest.raises(ValueError, match="multiple of the buckets"):
        TrainSpec(seq_buckets=(1024, 4096), microbatch_tokens=6144)


def test_the_teacher_travels_in_displayed_order_and_only_into_teacher_batches(tok):
    """Segment.teacher follows the option permutation (renormalised over the displayed
    letters); RowBuilder(teacher=True) packs it, the default leaves Batch.teacher None."""
    from tjev.data.pack import stack_batches

    raw = {"id": "t", "state": "s", "question": {"type": "choice", "instructions": "q",
           "criteria": {"a": "", "b": "", "c": ""}}, "expected": "a",
           "teacher": {"a": 0.2, "b": 0.7, "c": 0.1}}  # fmt: skip
    item = parse_item(raw)
    for seed in range(5):
        seg = make_segment(item, tok, RenderSpec(), Vocab(), 0, np.random.default_rng(seed))
        gold = int(np.argmax(seg.target))
        assert seg.teacher is not None and abs(seg.teacher.sum() - 1) < 1e-6
        assert seg.teacher[gold] == pytest.approx(0.2)
        assert sorted(seg.teacher.round(4)) == [0.1, 0.2, 0.7]
    plain = RowBuilder(1, 4096, 4, 26)
    with_teacher = RowBuilder(1, 4096, 4, 26, teacher=True)
    assert plain.add(seg) and with_teacher.add(seg)
    assert plain.build().teacher is None
    built = with_teacher.build()
    np.testing.assert_array_equal(built.teacher[0, 0, :3], seg.teacher)
    assert built.teacher[0, 1:].sum() == 0  # empty slots: no teacher
    assert stack_batches([plain.build(), plain.build()]).teacher is None
    assert stack_batches([built, built]).teacher.shape == (2, 1, 4, 26)
