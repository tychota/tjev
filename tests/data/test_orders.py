"""Eval orderings for order averaging, and the fixed noul order (eval / serving only)."""

import numpy as np
import pytest

from tjev.data.item import parse_item
from tjev.data.pack import Vocab, make_segment
from tjev.data.render import RenderSpec, choose_order, eval_orders, eval_specs, render
from tjev.data.tokenize import PromptTokenizer
from tjev.testing import make_tiny_snapshot, tiny_items

EVAL = RenderSpec(train=False)


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tok")
    make_tiny_snapshot(folder)
    return PromptTokenizer(folder)


def _choice(n: int, item_id: str = "c"):
    labels = [f"opt{i}" for i in range(n)]
    return parse_item({"id": item_id, "state": "s", "expected": labels[0], "question": {
        "type": "choice", "instructions": "?", "criteria": {lab: lab for lab in labels}}})  # fmt: skip


def _noul(item_id: str):
    return parse_item({"id": item_id, "state": "s", "expected": "yes", "question": {
        "type": "noul", "instructions": "?", "criteria": {"false": "f", "true": "t"}}})  # fmt: skip


def _rotations(order):
    return {tuple(order[s:] + order[:s]) for s in range(len(order))}


def test_first_ordering_is_the_plain_eval_order_then_spread_rotations():
    for item in (_choice(5), _noul("n"), _choice(4, "y")):
        base = choose_order(item, EVAL, None)[0]
        assert eval_orders(item, EVAL, 1) == [base]
        two = eval_orders(item, EVAL, 2)
        n = len(base)
        assert two == [base, base[n // 2 :] + base[: n // 2]]  # half a turn apart
    noul = _noul("n")
    assert eval_orders(noul, EVAL, 2)[1] == eval_orders(noul, EVAL, 2)[0][::-1]  # the swap


def test_orderings_are_distinct_and_capped():
    assert len(eval_orders(_noul("n"), EVAL, 5)) == 2  # the two noul orders, never one twice
    for n in (3, 4, 6):
        item = _choice(n)
        base = choose_order(item, EVAL, None)[0]
        for k in range(1, n + 3):
            orders = eval_orders(item, EVAL, k)
            assert len(orders) == min(k, n) == len({tuple(o) for o in orders})
            assert {tuple(o) for o in orders} <= _rotations(base)
        full = np.asarray(eval_orders(item, EVAL, n))  # every option once at each position
        assert all(sorted(full[:, j]) == list(range(n)) for j in range(n))
    score = parse_item(tiny_items(3)[2])
    assert eval_orders(score, EVAL, 4) == [[0, 1, 2, 3]]  # score levels keep their order
    specs = eval_specs(_choice(4), EVAL, 4)
    assert [s.rotation for s in specs] == [0, 1, 2, 3]


def test_bad_order_counts_and_train_specs_are_refused():
    with pytest.raises(ValueError, match=">= 1"):
        eval_orders(_choice(3), EVAL, 0)
    with pytest.raises(ValueError, match="train"):
        eval_orders(_choice(3), RenderSpec(), 2)


def test_fixed_noul_order_is_no_first_at_eval_only():
    fixed = RenderSpec(train=False, noul_order="fixed")
    shuffled = {tuple(render(_noul(f"id{i}"), EVAL)[1]) for i in range(20)}
    assert shuffled == {(0, 1), (1, 0)}  # the default: the item-id hash, like choice
    assert {tuple(render(_noul(f"id{i}"), fixed)[1]) for i in range(20)} == {(0, 1)}
    assert "A. no: f\nB. yes: t" in render(_noul("id"), fixed)[0]
    item = _choice(5, "x")  # choice is unaffected
    assert render(item, fixed) == render(item, EVAL)
    train = RenderSpec(noul_order="fixed")  # training always shuffles noul
    seen = {tuple(render(_noul("t"), train, np.random.default_rng(s))[1]) for s in range(20)}
    assert seen == {(0, 1), (1, 0)}


def test_rotated_fixed_noul_and_rotated_segments(tok):
    fixed = RenderSpec(train=False, noul_order="fixed")
    assert eval_orders(_noul("id"), fixed, 2) == [[0, 1], [1, 0]]
    item = _choice(4)
    for spec in eval_specs(item, EVAL, 4):
        seg = make_segment(item, tok, spec, Vocab.from_items([item]), 0)
        text, order = render(item, spec)
        assert list(seg.tokens) == tok.encode(text)
        assert seg.target.argmax() == order.index(item.gold)
