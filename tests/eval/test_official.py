"""The official evaluation: v1.5 scoring rules, the control prompt, order averaging, and a
CPU smoke run of the real JevBench harness against a tiny random model served by tjev."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tjev.data.item import parse_item
from tjev.data.render import RenderSpec, choose_order, render
from tjev.eval import official
from tjev.testing import make_tiny_snapshot


def task(kind, labels, expected, gold=None):
    return SimpleNamespace(question={"type": kind}, labels=labels, expected=expected,
                           provenance={"gold_probs": gold} if gold else {})  # fmt: skip


def test_v15_noul_band_is_an_abstention():
    t = task("noul", ["no", "yes"], "yes")
    hit = official.v15_item(t, {"valid": True, "probs": {"yes": 0.85, "no": 0.15}})
    unsure = official.v15_item(t, {"valid": True, "probs": {"yes": 0.79, "no": 0.21}})
    failed = official.v15_item(t, {"valid": False})
    assert hit["correct"] and not hit["abstained"]
    assert not unsure["correct"] and unsure["abstained"]  # argmax right, still wrong in v1.5
    assert not failed["correct"] and not failed["abstained"]
    assert official.v15_cc([hit, unsure]) == pytest.approx(0.0)  # 50 % = chance


def test_v15_score_and_choice_chance():
    t = task("score", ["0", "1", "2", "3"], 0)
    uniform = official.v15_item(t, {"valid": True, "ordinal_ev": 1.5, "probs": {}})
    # the uniform guess's nMAE at gold 0 of 4 levels: (0+1+2+3)/(4·3) = 0.5
    assert uniform["nmae_chance"] == pytest.approx(0.5)
    assert official.v15_cc([uniform]) == pytest.approx(0.0)
    worst = official.v15_item(t, {"valid": False})
    assert worst["nmae"] == 1.0
    c = task("choice", list("abcd"), "a")
    right = official.v15_item(c, {"valid": True, "correct": True, "probs": {}})
    wrong = official.v15_item(c, {"valid": True, "correct": False, "probs": {}})
    assert official.v15_cc([right]) == pytest.approx(100.0)
    assert official.v15_cc([right, wrong, wrong, wrong]) == pytest.approx(0.0)  # 25 % = chance


def test_v15_aggregation_renormalises_missing_tiers():
    items = [("easy", {"type": "choice", "correct": True, "chance": 0.5}),
             ("hard", {"type": "choice", "correct": False, "chance": 0.5})]  # fmt: skip
    out = official.v15_intelligence(items)
    # easy 100, hard -100; weights 0.1 / 0.4 renormalised (no standard, no judge)
    assert out["types"]["choice"]["cc"] == pytest.approx((0.1 * 100 - 0.4 * 100) / 0.5)
    assert out["intelligence_equal"] == out["types"]["choice"]["cc"]  # one supported type


def test_typed_calibration_parts():
    rows = [
        (task("noul", ["no", "yes"], "yes"), {"valid": True, "probs": {"yes": 0.9, "no": 0.1}}),
        (task("score", ["0", "1", "2"], 2), {"valid": True, "probs": {"0": 0, "1": 0, "2": 1.0}}),
        (task("choice", ["a", "b"], "a", gold={"a": 0.6, "b": 0.4}),
         {"valid": True, "probs": {"a": 0.6, "b": 0.4}}),
    ]  # fmt: skip
    cal = official.v15_calibration(rows)
    assert cal["types"]["score"]["rps"] == pytest.approx(0.0)
    assert cal["types"]["choice"]["tvd"] == pytest.approx(0.0)
    assert cal["types"]["noul"]["ece_p_yes"] == pytest.approx(0.1)


def test_harmonic_gates():
    axes = {"intelligence": 80.0, "calibration": 80.0, "speed": 80.0, "cost": 80.0}
    assert official.harmonic(axes, (0.25,) * 4, 50) == pytest.approx(80.0)
    low = axes | {"intelligence": 40.0}
    h = 4 / (1 / 40 + 3 / 80)
    assert official.harmonic(low, (0.25,) * 4, 50) == pytest.approx(h * (40 / 50) ** 2)
    mid = axes | {"intelligence": 55.0}
    floor_60 = official.harmonic(mid, (0.25,) * 4, 60)
    assert floor_60 is not None and floor_60 == pytest.approx(
        4 / (1 / 55 + 3 / 80) * (55 / 60) ** 2
    )
    root = harness_root()
    if root is not None:  # the same as the harness's v1.4 composite (equal weights, gate 50)
        v14 = official.harness_modules(root)["composite_v14"]
        for a in (axes, low, mid, axes | {"speed": 30.0}, axes | {"cost": 0.0}):
            assert official.harmonic(a, (0.25,) * 4, 50) == pytest.approx(v14.harmonic(a))


NOUL = {"id": "n1", "state": "The door is open.", "expected": "yes",
        "question": {"type": "noul", "instructions": "Is it open?",
                     "criteria": {"false": "closed", "true": "open"}}}  # fmt: skip
CHOICE = {"id": "c1", "state": {"order": 7, "status": "late"}, "expected": "late",
          "question": {"type": "choice", "instructions": "Status?",
                       "criteria": {"early": "", "on_time": "On time", "late": "Late"}}}  # fmt: skip


def test_semif_control_prompt():
    noul, choice = parse_item(NOUL), parse_item(CHOICE)
    text, order = render(noul, RenderSpec(train=False, prompt="semif"))
    assert order == [1, 0]  # true (yes) first, always
    assert text.startswith("<|im_start|>system\nApply the supplied criterion")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    user = text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
    assert json.loads(user) == {"evidence": "The door is open.", "criterion": "Is it open?",
                                "options": [{"letter": "A", "description": "true: open"},
                                            {"letter": "B", "description": "false: closed"}]}  # fmt: skip
    text, order = render(choice, RenderSpec(train=False, prompt="semif"))
    payload = json.loads(text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0])
    assert order == [0, 1, 2]  # the declared order
    assert payload["evidence"] == {"order": 7, "status": "late"}  # the structured state
    assert [o["description"] for o in payload["options"]] == ["early: early", "on_time: On time",
                                                              "late: Late"]  # fmt: skip
    with pytest.raises(ValueError):
        render(choice, RenderSpec(train=True, prompt="semif"), np.random.default_rng(0))


def test_rotations_put_each_option_elsewhere():
    choice = parse_item(CHOICE)
    base = choose_order(choice, RenderSpec(train=False), None)[0]
    turned = choose_order(choice, RenderSpec(train=False, rotation=1), None)[0]
    assert sorted(turned) == sorted(base) and all(a != b for a, b in zip(base, turned, strict=True))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    folder = tmp_path_factory.mktemp("official") / "Qwen3.5-tiny"
    make_tiny_snapshot(folder)
    return folder


def test_order_averaging_averages_the_rotations(tiny):
    from tjev.serve.backends import JaxBackend

    one = JaxBackend(model_path=tiny, buckets=(1024,), warmup=False)
    two = JaxBackend(model_path=tiny, buckets=(1024,), warmup=False, orders=2)
    item = parse_item(NOUL)
    (p1,), _ = one.score([item])
    (p2,), usage = two.score([item])
    _, single = one.score([item])
    assert two.name.endswith("+orders2") and len(two.specs(item)) == 2
    assert usage["prompt_tokens"] == 2 * single["prompt_tokens"]  # both orders were read
    assert np.isclose(p2.sum(), 1.0) and not np.allclose(p1, p2)


def harness_root() -> Path | None:
    for root in (os.environ.get("JEVBENCH_HARNESS"), Path("~/tjev-work/jevbench").expanduser()):
        if root and (Path(root) / "jevbench" / "cli.py").exists():
            return Path(root)
    return None


@pytest.mark.skipif(harness_root() is None, reason="no JevBench harness checkout")
def test_official_harness_smoke(tiny, tmp_path):
    """The real harness (its own CLI, a subprocess) against tjev served in-process: two
    items per tier, order averaging and the abstention policy (into ``noul``) on. A random
    model: only the plumbing is checked."""
    from tjev.serve.backends import JaxBackend
    from tjev.serve.wire import AbstentionPolicy

    root = harness_root()
    backend = JaxBackend(model_path=tiny, buckets=(1024, 2048, 4096), warmup=False, orders=2)
    out = tmp_path / "report"
    report = official.evaluate_official(
        "Qwen3.5-4B-tiny", harness=root, out=out, per_tier=2, allow_unpinned=True,
        endpoint_kind="cpu", backend=backend, orders=2,
        policy=AbstentionPolicy(into_noul=True),
    )  # fmt: skip
    assert report["served"]["policy"]["abstention"]["into_noul"] is True
    assert (out / "report.json").exists() and (out / "harness-results.jsonl").exists()
    assert report["harness"]["commit"] and report["harness"]["adapter"] == "typesafe"
    assert report["price_per_m"]["input"] == 0.03  # the Qwen3.5-4B M2 reference price
    assert set(report["tiers"]) == {"easy", "standard", "hard"}
    assert report["n"] == 6
    ok = list(map(json.loads, (out / "harness-results.jsonl").read_text().splitlines()))
    assert sum(r["ok"] for r in ok) >= 3  # easy + standard fit; long hard states may be 413
    for r in ok:
        if r["ok"]:
            assert r["valid"] and abs(sum(r["probs"].values()) - 1) < 1e-6
            assert r["usage"]["input_tokens"] > 0 and r["cost_usd"] > 0
            if set(r["probs"]) == {"yes", "no"}:  # noul: the harness scored the committed P(yes)
                assert not 0.2 < r["probs"]["yes"] < 0.8, r["probs"]
    v15 = report["v15"]
    assert v15["official"] is None and report["v14"]["official"] is None
    assert v15["intelligence_open_public"]["intelligence_equal"] is not None
    assert report["latency"]["all"]["p50_s"] > 0 and report["cost"]["usd_per_1000"] > 0
    assert report["v13"]["intelligence_public"] is not None
