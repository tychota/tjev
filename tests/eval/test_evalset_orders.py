"""Order averaging in a packed eval set on the tiny random model, against prompts scored
one at a time: K = 1 is the plain eval, K = 2 the mean of an ordering and its reverse, and
the mean of half-a-turn rotations (noul: both orders), and averaging every rotation does
not depend on which one the eval shows first."""

import jax
import numpy as np
import pytest

from tjev.config import load_config
from tjev.data.item import parse_item
from tjev.data.pack import Vocab, make_segment, pack_eval
from tjev.data.render import RenderSpec, choose_order, eval_orders, eval_specs
from tjev.data.tokenize import PromptTokenizer
from tjev.eval.evalset import EvalSet, pad_rows
from tjev.eval.metrics import softmax, view_probs
from tjev.model import build_model
from tjev.sharding import batch_sharding, install_mesh, make_mesh, place_frozen
from tjev.testing import make_tiny_snapshot, tiny_items
from tjev.train.step import make_eval_step, split_model

TINY = [
    "model.dtype=float32", "train.seq_buckets=[1024,4096]", "train.microbatch_tokens=4096",
    "train.max_segments=4", "compute.gdn_chunk=16",
]  # fmt: skip


class Tiny:
    def __init__(self, folder):
        self.cfg = load_config(overrides=TINY)
        _, model = build_model(folder, self.cfg.compute, None, dtype=self.cfg.model.dtype)
        self.mesh = make_mesh(self.cfg.mesh)
        install_mesh(self.cfg.compute, self.mesh)
        graphdef, self.lora, frozen = split_model(model)
        self.frozen = place_frozen(frozen, self.mesh)
        self.eval_fn = jax.jit(make_eval_step(graphdef))
        self.tok = PromptTokenizer(folder)

    def evalset(self, items, **readout) -> EvalSet:
        return EvalSet(items, self.tok, Vocab.from_items(items), self.cfg, self.mesh, **readout)

    def probs(self, items, **readout) -> np.ndarray:
        es = self.evalset(items, **readout)
        a = es.arrays(self.eval_fn, self.lora, self.frozen, self.mesh, None)
        return view_probs(a["logits"], a["mask"])

    def single(self, item, spec) -> np.ndarray:
        """One prompt alone in its own batch: the distribution in label order."""
        order = choose_order(item, spec, None)[0]
        seg = make_segment(item, self.tok, spec, Vocab.from_items([item]), 0)
        (batch,), _ = pack_eval([seg], (1024, 4096), 4096, 4, self.cfg.train.max_labels)
        batch = pad_rows(batch, self.mesh.size)
        sharding = batch_sharding(self.mesh, stacked=False)
        out = np.asarray(self.eval_fn(self.lora, self.frozen, jax.device_put(batch, sharding)))
        shown = softmax(out[0, 0, : len(order)][None], np.ones((1, len(order)), bool))[0]
        p = np.zeros(len(item.question.labels))
        p[order] = shown
        return p


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    folder = tmp_path_factory.mktemp("model")
    make_tiny_snapshot(folder)
    return Tiny(folder)


ITEMS = [parse_item(r) for r in tiny_items(12, seed=4)]
EVAL = RenderSpec(train=False)


def _noul(item_id: str):
    return parse_item({"id": item_id, "state": "The door is open.", "expected": "yes",
                       "question": {"type": "noul", "instructions": "Is the door open?",
                                    "criteria": {"false": "closed", "true": "open"}}})  # fmt: skip


def test_one_ordering_is_the_plain_eval(tiny):
    got = tiny.probs(ITEMS)
    for item, p in zip(ITEMS, got, strict=True):
        n = len(item.question.labels)
        want = tiny.single(item, EVAL)
        np.testing.assert_allclose(p[:n], want, atol=2e-5)
        assert not p[n:].any()


def test_two_orderings_average_two_rotations(tiny):
    got = tiny.probs(ITEMS, orders=2)
    gaps = []
    for item, p in zip(ITEMS, got, strict=True):
        views = [tiny.single(item, spec) for spec in eval_specs(item, EVAL, 2)]
        assert len(views) == (1 if item.question.type == "score" else 2)
        gaps.append(np.abs(views[0] - views[-1]).max())
        np.testing.assert_allclose(p[: len(views[0])], np.mean(views, 0), atol=2e-5)
    assert max(gaps) > 1e-3  # the orderings really disagree (the test is not vacuous)
    es = tiny.evalset(ITEMS, orders=2)
    one = tiny.evalset(ITEMS)
    scores = sum(it.question.type == "score" for it in ITEMS)
    assert one.segments == len(ITEMS)
    assert es.segments == 2 * len(ITEMS) - scores  # the eval cost of K = 2: score is 1 pass
    report = es.run(tiny.eval_fn, tiny.lora, tiny.frozen, tiny.mesh, None)
    assert report["orders"] == 2 and report["all"]["n"] == len(ITEMS)
    assert report["cost"]["segments"] == es.segments


def test_noul_two_order_average_is_the_same_for_a_fixed_or_shuffled_first_order(tiny):
    items = [_noul(f"door-{i}") for i in range(8)]
    firsts = {tuple(eval_orders(it, EVAL, 1)[0]) for it in items}
    assert firsts == {(0, 1), (1, 0)}  # the hash shows some items yes-first
    shuffled = tiny.probs(items, orders=2)
    fixed = tiny.probs(items, orders=2, noul_order="fixed")
    np.testing.assert_allclose(shuffled, fixed, atol=2e-5)
    one_fixed = tiny.probs(items, noul_order="fixed")
    no_first = tiny.single(items[0], RenderSpec(train=False, noul_order="fixed"))
    np.testing.assert_allclose(one_fixed[:, :2], no_first[None].repeat(len(items), 0), atol=2e-5)


def test_every_rotation_averages_the_same_whichever_comes_first(tiny):
    raw = tiny_items(1)[0]  # a choice question, 3 options
    criteria = {c: f"the light is {c}" for c in ("red", "green", "blue")}

    def light(i):
        return parse_item({**raw, "question": {**raw["question"], "criteria": criteria},
                           "expected": "red", "id": f"light-{i}"})  # fmt: skip

    def cycle(order):  # the rotation class of an order
        return min(tuple(order[s:] + order[:s]) for s in range(len(order)))

    by_first = {}
    for i in range(40):  # items whose first orders differ but rotate into each other
        item = light(i)
        by_first.setdefault(tuple(eval_orders(item, EVAL, 1)[0]), item)
    target = cycle(next(iter(by_first)))
    items = [it for first, it in by_first.items() if cycle(first) == target]
    assert len(items) == 3  # the 3 rotations of one cyclic class, each shown first once
    got = tiny.probs(items, orders=3)
    np.testing.assert_allclose(got, got[:1].repeat(len(items), 0), atol=2e-5)
    single = tiny.probs(items)
    assert np.abs(single - single[:1]).max() > 1e-3  # unaveraged, the first order matters
