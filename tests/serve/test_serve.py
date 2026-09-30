"""The decision API: wire parsing, answers in label order, micro-batching of concurrent
requests, errors."""

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tjev.serve.app import create_app
from tjev.serve.backends import JaxBackend
from tjev.serve.batcher import MicroBatcher
from tjev.serve.wire import DecisionRequest, answer, to_items
from tjev.testing import make_tiny_snapshot

REQUEST = {
    "state": "Do not cancel my membership. Please return the duplicate charge.",
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "Select the primary requested action.",
            "criteria": {
                "refund": "Return money already charged",
                "cancel": "End a subscription",
                "status": "Learn delivery progress",
                "other": "None of these",
            },
        },
        "urgent": {
            "type": "noul",
            "instructions": "Is this urgent?",
            "criteria": {"false": "Can wait", "true": "Needs action today"},
        },
        "tone": {
            "type": "score",
            "instructions": "How upset is the customer?",
            "criteria": ["calm", "annoyed", "angry"],
        },
    },
}


def test_wire_parsing_and_answers():
    named = to_items(DecisionRequest.model_validate(REQUEST))
    assert [n for n, _ in named] == ["intent", "urgent", "tone"]
    out = answer(named[0][1], np.array([0.1, 0.2, 0.3, 0.4]))
    assert out.choice == "other"
    assert abs(sum(out.probabilities.values()) - 1) < 1e-9
    assert answer(named[1][1], np.array([0.25, 0.75])).noul == pytest.approx(0.75)
    assert named[2][1].question.labels == ("0", "1", "2")


class CountingBackend:
    """Uniform probabilities; records how many items each call scored."""

    name = "counting"

    def __init__(self):
        self.calls: list[int] = []

    def score(self, items):
        self.calls.append(len(items))
        probs = [np.full(len(i.question.labels), 1 / len(i.question.labels)) for i in items]
        return probs, {"prompt_tokens": 10 * len(items), "seconds": 0.0}


def test_concurrent_requests_are_micro_batched():
    backend = CountingBackend()
    named = to_items(DecisionRequest.model_validate(REQUEST))
    items = [item for _, item in named]

    async def run():
        batcher = MicroBatcher(backend, max_items=64, max_wait_ms=50)
        await batcher.start()
        try:
            return await asyncio.gather(*(batcher.score(items) for _ in range(5)))
        finally:
            await batcher.stop()

    results = asyncio.run(run())
    assert backend.calls == [15]  # 5 requests × 3 questions in one backend call
    assert all(len(r.probs) == 3 and r.batch_items == 15 for r in results)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    folder = tmp_path_factory.mktemp("serve-model")
    make_tiny_snapshot(folder)
    backend = JaxBackend(model_path=folder, buckets=(1024, 2048), warmup=False)
    with TestClient(create_app(backend)) as c:
        yield c, backend


def test_endpoint_returns_distributions_in_label_order(client):
    c, backend = client
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/v1/systemone", json=REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["answers"]) == {"intent", "urgent", "tone"}
    assert set(body["answers"]["tone"]["probabilities"]) == {"0", "1", "2"}
    assert 0 <= body["answers"]["urgent"]["noul"] <= 1
    assert body["usage"]["prompt_tokens"] > 0
    named = to_items(DecisionRequest.model_validate(REQUEST))
    direct, _ = backend.score([item for _, item in named])
    for (name, item), p in zip(named, direct, strict=True):
        got = [body["answers"][name]["probabilities"][label] for label in item.question.labels]
        np.testing.assert_allclose(got, p / p.sum(), atol=1e-6)


def test_bad_requests(client):
    c, _ = client
    assert c.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 422
    bad = {"state": "x", "questions": {"q": {"type": "choice", "criteria": {"only": "one"}}}}
    assert c.post("/v1/systemone", json=bad).status_code == 400  # fewer than 2 labels
    long = {**REQUEST, "state": "word " * 3000}  # beyond the 2048-token bucket
    assert c.post("/v1/systemone", json=long).status_code == 413
    assert c.post("/v1/other", json=REQUEST).status_code == 404
