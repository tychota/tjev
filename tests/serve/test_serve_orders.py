"""Serving readout options: order averaging (K orderings, calibration bound to K), the
fixed noul order, and the optional abstention-band policy (off: the wire format is
unchanged)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tjev.cli import app as cli
from tjev.data.render import TEMPLATE_VERSION, RenderSpec, eval_orders
from tjev.eval import calibrate as cal
from tjev.eval.runs import base_identity
from tjev.serve.app import create_app
from tjev.serve.backends import JaxBackend
from tjev.serve.wire import AbstentionPolicy, DecisionRequest, answer, to_items
from tjev.testing import make_tiny_snapshot

STATE = "Do not cancel my membership. Please return the duplicate charge."
NOUL = {"type": "noul", "instructions": "Is this urgent?",
        "criteria": {"false": "Can wait", "true": "Needs action today"}}  # fmt: skip
CHOICE = {"type": "choice", "instructions": "Select the primary requested action.",
          "criteria": {"refund": "Return money already charged", "cancel": "End a subscription",
                       "status": "Learn delivery progress"}}  # fmt: skip


def _items(questions: dict):
    named = to_items(DecisionRequest.model_validate({"state": STATE, "questions": questions}))
    return [item for _, item in named]


def _yes_first_noul() -> dict:
    """A noul question whose (content-hashed) display order is yes first."""
    for i in range(50):
        question = {**NOUL, "instructions": f"Is this urgent? ({i})"}
        (item,) = _items({"u": question})
        if eval_orders(item, RenderSpec(train=False), 1)[0] == [1, 0]:
            return question
    raise AssertionError("no yes-first question in 50")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    folder = tmp_path_factory.mktemp("serve-model")
    make_tiny_snapshot(folder)
    return folder


# The served base model runs in bfloat16: one prompt's probabilities move by up to ~1e-3
# with its offset in a packed row (chunk alignment), so serving equivalences hold to BF16;
# the float32 ones (tests/eval/test_evalset_orders.py) hold to 2e-5.
BF16 = 3e-3


def _backend(model, **readout):
    return JaxBackend(model_path=model, buckets=(1024, 2048), warmup=False, **readout)


def test_two_orders_average_the_two_noul_orders(model):
    question = _yes_first_noul()
    (yes_first,), single = _backend(model).score(_items({"u": question}))
    (no_first,), _ = _backend(model, noul_order="fixed").score(_items({"u": question}))
    assert np.abs(yes_first - no_first).max() > 30 * BF16  # a real position effect
    for noul_order in ("shuffled", "fixed"):
        two = _backend(model, orders=2, noul_order=noul_order)
        (got,), usage = two.score(_items({"u": question}))
        np.testing.assert_allclose(got, (yes_first + no_first) / 2, atol=BF16)
        assert usage["prompt_tokens"] == 2 * single["prompt_tokens"]  # K× the prompt tokens
        assert two.name.endswith("+orders2" + ("+noul-fixed" if noul_order == "fixed" else ""))


def test_rotations_per_type(model):
    three = _backend(model, orders=3)
    tone = {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "angry"]}
    items = _items({"pick": CHOICE, "tone": tone})
    assert [len(three.specs(it)) for it in items] == [3, 1]  # score levels keep their order
    (pick, score), usage = three.score(items)
    _, one = _backend(model).score(items)
    np.testing.assert_allclose([pick.sum(), score.sum()], 1.0)
    assert usage["prompt_tokens"] > 1.9 * one["prompt_tokens"]  # 3 + 1 prompts vs 1 + 1


def test_calibration_must_match_the_served_orders(model, tmp_path):
    fit = {"temperatures": {"noul": 2.0, "choice": 1.5, "score": 1.0}}
    common = {"checkpoint": base_identity(model), "data": "d", "template": TEMPLATE_VERSION,
              "backend": "jax"}  # fmt: skip
    path = tmp_path / "cal-k1.json"
    path.write_text(__import__("json").dumps(cal.artifact(fit, **common)))
    assert _backend(model, calibration=path).temperatures == fit["temperatures"]
    with pytest.raises(ValueError, match="orders=1, not 2"):
        _backend(model, calibration=path, orders=2)
    path2 = tmp_path / "cal-k2.json"
    path2.write_text(__import__("json").dumps(cal.artifact(fit, orders=2, **common)))
    assert _backend(model, calibration=path2, orders=2).temperatures == fit["temperatures"]
    with pytest.raises(ValueError, match="orders=2, not 1"):
        _backend(model, calibration=path2)


def test_abstention_policy_commits_in_band_noul_only():
    (noul,) = _items({"u": NOUL})
    (choice,) = _items({"c": CHOICE})
    off = answer(noul, np.array([0.7, 0.3]))
    assert off.model_dump() == {"type": "noul", "probabilities": {"no": 0.7, "yes": 0.3},
                                "choice": None, "noul": 0.3}  # fmt: skip
    field = AbstentionPolicy()
    out = answer(noul, np.array([0.7, 0.3]), field)
    assert (out.noul, out.committed, out.decision) == (0.3, 0.2, "no")
    assert out.probabilities == off.probabilities  # the calibrated distribution is kept
    into = AbstentionPolicy(into_noul=True)
    out = answer(noul, np.array([0.45, 0.55]), into)
    assert (out.noul, out.committed, out.decision) == (0.8, 0.8, "yes")
    assert out.probabilities["yes"] == pytest.approx(0.55)
    for p_yes in (0.0, 0.1, 0.2, 0.8, 0.95):  # outside the open band (edges included): kept
        out = answer(noul, np.array([1 - p_yes, p_yes]), into)
        assert out.committed == pytest.approx(p_yes) and out.noul == pytest.approx(p_yes)
        assert out.decision == ("yes" if p_yes >= 0.8 else "no")
    assert field.commit(0.5) == 0.8  # the tie goes up
    skewed = (0.1, 0.6)  # nearest edge vs the argmax side differ on an asymmetric band
    assert AbstentionPolicy(*skewed).commit(0.4) == 0.6
    assert AbstentionPolicy(*skewed, snap="argmax").commit(0.4) == 0.1
    assert (
        answer(choice, np.array([0.2, 0.5, 0.3]), into).model_dump()
        == answer(choice, np.array([0.2, 0.5, 0.3])).model_dump()
    )
    for bad in ((0.8, 0.2), (-0.1, 0.5), (0.2, 1.5)):
        with pytest.raises(ValueError, match="band"):
            AbstentionPolicy(*bad)


class Uniform:
    name = "uniform"

    def score(self, items):
        probs = [np.full(len(i.question.labels), 1 / len(i.question.labels)) for i in items]
        return probs, {"prompt_tokens": 1, "seconds": 0.0}


def _typesafe_noul(answer: dict) -> float:
    """What the JevBench TypeSafe adapter reads from a noul answer."""
    p = answer["noul"]
    assert isinstance(p, float) and 0.0 <= p <= 1.0
    return p


def test_endpoint_fields_with_and_without_the_policy():
    request = {"state": STATE, "questions": {"decision": NOUL, "pick": CHOICE}}
    with TestClient(create_app(Uniform())) as c:
        body = c.post("/v1/systemone", json=request).json()
        assert set(body["answers"]["decision"]) == {"type", "probabilities", "choice", "noul"}
        assert set(body["answers"]["pick"]) == {"type", "probabilities", "choice", "noul"}
        assert _typesafe_noul(body["answers"]["decision"]) == 0.5
        assert c.get("/health").json()["abstention"] == "off"
    for policy, noul in ((AbstentionPolicy(), 0.5), (AbstentionPolicy(into_noul=True), 0.8)):
        with TestClient(create_app(Uniform(), policy=policy)) as c:
            body = c.post("/v1/systemone", json=request).json()
            ans = body["answers"]["decision"]
            assert (ans["committed"], ans["decision"]) == (0.8, "yes")
            assert _typesafe_noul(ans) == noul
            assert ans["probabilities"] == {"no": 0.5, "yes": 0.5}
            assert "committed" not in body["answers"]["pick"]


def test_serve_cli_rejects_bad_readout_options(model):
    runner = CliRunner()
    for extra, message in ((["--abstention", "maybe"], "abstention"),
                           (["--noul-order", "yes-first"], "noul-order"),
                           (["--abstention", "field", "--band", "0.9", "0.1"], "band"),
                           (["--orders", "0"], "orders")):  # fmt: skip
        result = runner.invoke(cli, ["serve", "--model", str(model), *extra])
        assert result.exit_code != 0
        assert message in result.output, result.output
