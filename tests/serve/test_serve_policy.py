"""Serving policies (tjev.serve.policy): serving temperatures per type and the noul commit
with a margin, applied after calibration, the calibrated distribution kept in the answer."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tjev.cli import app as cli
from tjev.serve.app import create_app
from tjev.serve.policy import AbstentionPolicy, ServingPolicy, sharpen
from tjev.serve.wire import DecisionRequest, answer, to_items

STATE = "The parcel is three days late and the customer asks for a refund."
QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Is this urgent?",
               "criteria": {"false": "Can wait", "true": "Needs action today"}},
    "pick": {"type": "choice", "instructions": "Select the action.",
             "criteria": {"refund": "Refund", "wait": "Wait", "escalate": "Escalate"}},
    "tone": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed",
                                                                         "angry"]},
}  # fmt: skip
FIXED = {"noul": [0.4, 0.6], "choice": [0.2, 0.5, 0.3], "score": [0.1, 0.6, 0.3]}


def _items():
    named = to_items(DecisionRequest.model_validate({"state": STATE, "questions": QUESTIONS}))
    return dict(named)


class Fixed:
    """The same calibrated distribution for every question of a type."""

    name = "fixed"

    def score(self, items):
        return [np.asarray(FIXED[i.question.type]) for i in items], {"prompt_tokens": 1,
                                                                     "seconds": 0.0}  # fmt: skip


def test_serving_temperature_keeps_the_calibrated_distribution():
    items = _items()
    policy = ServingPolicy(temperatures={"score": 0.5})
    tone = answer(items["tone"], np.asarray(FIXED["score"]), policy)
    assert list(tone.probabilities.values()) == pytest.approx(sharpen(np.array(FIXED["score"]),
                                                                      0.5))  # fmt: skip
    assert list(tone.calibrated.values()) == pytest.approx(FIXED["score"])
    assert tone.choice == "1"  # the argmax is unchanged
    pick = answer(items["pick"], np.asarray(FIXED["choice"]), policy)
    assert "calibrated" not in pick.model_dump()  # no policy on choice: the wire as before
    assert pick.model_dump() == answer(items["pick"], np.asarray(FIXED["choice"])).model_dump()
    noul = answer(items["urgent"], np.asarray(FIXED["noul"]),
                  ServingPolicy(temperatures={"noul": 0.5}))  # fmt: skip
    assert noul.noul == pytest.approx(0.36 / 0.52) and noul.calibrated["yes"] == pytest.approx(0.6)


def test_commit_margin_answers():
    item = _items()["urgent"]
    keep = AbstentionPolicy(into_noul=True, margin=0.15)
    out = answer(item, np.array([0.4, 0.6]), keep)
    assert (out.noul, out.committed, out.decision) == (pytest.approx(0.6), pytest.approx(0.6),
                                                       "abstain")  # fmt: skip
    out = answer(item, np.array([0.3, 0.7]), keep)
    assert (out.noul, out.decision) == (0.8, "yes") and out.calibrated["yes"] == pytest.approx(0.7)
    eps = AbstentionPolicy(into_noul=True, eps=0.001)
    assert answer(item, np.array([0.7, 0.3]), eps).noul == pytest.approx(0.199)
    # noul T then the commit: 0.6 sharpened at T 0.25 is 0.835, out of the band
    both = ServingPolicy(abstention=keep, temperatures={"noul": 0.25})
    out = answer(item, np.array([0.4, 0.6]), both)
    assert out.decision == "yes" and out.noul == pytest.approx(0.6**4 / (0.6**4 + 0.4**4))


def test_endpoint_and_health_with_a_policy():
    request = {"state": STATE, "questions": QUESTIONS}
    policy = ServingPolicy(abstention=AbstentionPolicy(into_noul=True, margin=0.05),
                           temperatures={"score": 0.5})  # fmt: skip
    with TestClient(create_app(Fixed(), policy=policy)) as c:
        body = c.post("/v1/systemone", json=request).json()["answers"]
        assert body["urgent"]["noul"] == 0.8 and body["urgent"]["decision"] == "yes"
        assert body["urgent"]["calibrated"] == pytest.approx({"no": 0.4, "yes": 0.6})
        assert body["tone"]["calibrated"] == pytest.approx({"0": 0.1, "1": 0.6, "2": 0.3})
        assert "calibrated" not in body["pick"]
        health = c.get("/health").json()
        assert (
            health["serving_temperatures"] == "score=0.5" and "margin 0.05" in health["abstention"]
        )
    with TestClient(create_app(Fixed(), policy=ServingPolicy())) as c:  # all off: as before
        body = c.post("/v1/systemone", json=request).json()["answers"]
        assert set(body["tone"]) == {"type", "probabilities", "choice", "noul"}
        assert c.get("/health").json()["serving_temperatures"] == "off"


def test_cli_rejects_bad_policy_options(tmp_path):
    runner = CliRunner()
    for extra, message in ((["--abstention", "noul", "--commit-margin", "0.7"], "margin"),
                           (["--score-temperature", "0"], "temperature"),
                           (["--noul-temperature", "50"], "temperature")):  # fmt: skip
        result = runner.invoke(cli, ["serve", "--model", str(tmp_path), *extra])
        assert result.exit_code != 0
        assert message in result.output, result.output
