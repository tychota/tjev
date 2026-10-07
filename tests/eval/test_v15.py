"""JevBench v1.5 scoring on per-item predictions (tjev.eval.v15): hand-computed cases, the
serving policies' effect, prediction dumps, the calibration pool, and a cross-check against
the harness's own per-item scoring and v1.3 calibration formula."""

import importlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from tjev.cli import app as cli
from tjev.eval import calibrate as cal
from tjev.eval import official, v15
from tjev.eval import predictions as dump
from tjev.eval.metrics import view_probs
from tjev.serve.policy import AbstentionPolicy, ServingPolicy, sharpen


def row(kind, labels, gold, probs, tier=v15.UNTIERED, gold_probs=None, family="f"):
    return v15.Row(id=f"{kind}-{gold}-{random.random()}", type=kind, labels=tuple(labels),
                   gold=gold, probs=None if probs is None else np.asarray(probs, float),
                   tier=tier, family=family,
                   gold_probs=None if gold_probs is None else np.asarray(gold_probs))  # fmt: skip


NOUL = ("no", "yes")


def test_noul_band_edges_and_abstention():
    def item(p, gold="yes"):
        return v15.score_item("noul", NOUL, gold, {"no": 1 - p, "yes": p})

    assert item(0.8)["correct"] and not item(0.8)["abstained"]  # the edge commits
    assert item(0.2, "no")["correct"] and not item(0.2, "no")["abstained"]
    assert not item(0.79)["correct"] and item(0.79)["abstained"]  # argmax right, still wrong
    assert not item(0.95, "no")["correct"] and not item(0.95, "no")["abstained"]
    failed = v15.score_item("noul", NOUL, "yes", None)
    assert not failed["correct"] and not failed["abstained"]
    # CC = 100 (acc - 0.5) / 0.5: 1 of 4 right is -50
    assert v15.cc([item(0.9), item(0.5), item(0.5), item(0.1)]) == pytest.approx(-50.0)


def test_choice_argmax_ties_and_per_item_chance():
    tie = v15.score_item("choice", ["b", "a"], "a", {"b": 0.5, "a": 0.5})
    assert tie["correct"]  # the harness's tie-break: the smallest label
    items = [v15.score_item("choice", list("ab"), "a", {"a": 0.7, "b": 0.3}),  # chance 1/2
             v15.score_item("choice", list("abcd"), "b", {"a": 0.4, "b": 0.3, "c": 0.2,
                                                         "d": 0.1})]  # fmt: skip
    # acc 0.5, mean chance (1/2 + 1/4) / 2 = 0.375: CC = 100 · 0.125 / 0.625 = 20
    assert v15.cc(items) == pytest.approx(20.0)


def test_score_expected_level_and_uniform_chance():
    labels = [str(i) for i in range(5)]
    split = v15.score_item("score", labels, "2", {"0": 0, "1": 0.5, "2": 0, "3": 0.5, "4": 0})
    assert split["nmae"] == pytest.approx(0.0)  # E[level] = 2, though no mass on 2
    assert split["nmae_chance"] == pytest.approx((2 + 1 + 0 + 1 + 2) / 20)
    uniform = v15.score_item("score", labels[:4], "0", dict.fromkeys(labels[:4], 0.25))
    assert uniform["nmae"] == pytest.approx(0.5) == uniform["nmae_chance"]
    assert v15.cc([uniform]) == pytest.approx(0.0)
    assert v15.score_item("score", labels, "1", None)["nmae"] == pytest.approx(3 / 4)  # worst
    # CC = 100 (1 - mean nMAE / mean chance): (0 + 0.5) / (0.3 + 0.5)
    assert v15.cc([split, uniform]) == pytest.approx(100 * (1 - 0.25 / 0.4))


def test_intelligence_tier_and_type_weights():
    right = {"type": "choice", "correct": True, "chance": 0.5}
    wrong = {"type": "choice", "correct": False, "chance": 0.5}
    yes = {"type": "noul", "correct": True, "abstained": False}
    unsure = {"type": "noul", "correct": False, "abstained": True}
    scored = [("easy", right), ("standard", right), ("standard", wrong), ("hard", wrong),
              ("hard", yes), ("hard", unsure)]  # fmt: skip
    out = v15.intelligence(scored)
    # choice: easy 100, standard 0, hard -100; weights 0.1 / 0.2 / 0.4 renormalised (no judge)
    choice = (0.1 * 100 + 0.2 * 0 + 0.4 * -100) / 0.7
    assert out["types"]["choice"]["cc"] == pytest.approx(choice)
    assert out["types"]["noul"]["cc"] == pytest.approx(0.0)
    assert out["types"]["noul"]["abstention_rate"] == pytest.approx(0.5)
    assert out["intelligence_equal"] == pytest.approx((choice + 0.0) / 2)  # score unsupported
    assert out["intelligence_frozen"] == pytest.approx((0.5 * choice) / 0.75)
    # an untiered set is one tier
    assert v15.intelligence([("all", right)])["types"]["choice"]["cc"] == pytest.approx(100.0)


def test_typed_calibration_hand_computed():
    rows = [
        row("choice", "ab", "a", [0.6, 0.4], tier="hard", gold_probs=[0.7, 0.3]),
        row("noul", NOUL, "yes", [0.1, 0.9]),
        row("noul", NOUL, "no", [0.1, 0.9]),
        row("score", "012", "0", [0.0, 1.0, 0.0]),
    ]
    out = v15.calibration(rows)
    choice, noul, score = (out["types"][k] for k in ("choice", "noul", "score"))
    # choice: ECE |1 - 0.6| = 0.4 → 20 points; TVD 0.1 → 90; mean 55
    assert choice["ece"] == pytest.approx(0.4) == choice["ece_hard"]
    assert choice["tvd"] == pytest.approx(0.1) and choice["points"] == pytest.approx(55.0)
    # noul: both in the 0.9 bin, yes rate 0.5: ECE 0.4 → 100 - 200 · 0.4 = 20
    assert noul["ece_p_yes"] == pytest.approx(0.4) and noul["points"] == pytest.approx(20.0)
    assert noul["brier"] == pytest.approx((2 * 0.1**2 + 2 * 0.9**2) / 2)
    # score: RPS = ((0 - 1)^2 + (1 - 1)^2) / 2 = 0.5 → 50; ECE 1.0 → 0; mean 25
    assert score["rps"] == pytest.approx(0.5) and score["points"] == pytest.approx(25.0)
    assert out["calibration_equal"] == pytest.approx((55 + 20 + 25) / 3)
    assert out["calibration_frozen"] == pytest.approx(0.5 * 55 + 0.25 * 20 + 0.25 * 25)


def test_commit_margin_and_eps():
    m = AbstentionPolicy(into_noul=True, margin=0.1)
    assert m.commit(0.55) == 0.55 and m.commit(0.45) == 0.45  # near a coin flip: kept
    assert m.commit(0.65) == 0.8 and m.commit(0.35) == 0.2
    assert m.commit(0.9) == 0.9 and m.commit(0.2) == 0.2  # outside the open band: kept
    e = AbstentionPolicy(eps=0.01)
    assert e.commit(0.65) == pytest.approx(0.81) and e.commit(0.3) == pytest.approx(0.19)
    for bad in ({"margin": 0.6}, {"margin": -0.1}, {"eps": 0.2}):
        with pytest.raises(ValueError, match=r"margin|eps"):
            AbstentionPolicy(**bad)


def test_sharpen_is_a_temperature_on_one_ordering():
    assert sharpen(np.array([0.5, 0.25, 0.25]), 0.5) == pytest.approx([2 / 3, 1 / 6, 1 / 6])
    assert sharpen(np.array([0.0, 0.4, 0.6]), 0.3)[0] == 0.0  # zeros stay zero
    z = np.array([[1.0, 2.5, -0.3]])
    once = view_probs(z, np.ones_like(z, bool), 1.3 * 0.6)[0]
    twice = sharpen(view_probs(z, np.ones_like(z, bool), 1.3)[0], 0.6)
    assert twice == pytest.approx(once)
    with pytest.raises(ValueError, match="unknown type"):
        ServingPolicy(temperatures={"nope": 0.5})
    with pytest.raises(ValueError, match="outside"):
        ServingPolicy(temperatures={"score": 0.0})


def test_policies_change_what_is_scored():
    rows = [row("noul", NOUL, "yes", [0.35, 0.65]), row("noul", NOUL, "no", [0.55, 0.45]),
            row("score", "012", "1", [0.1, 0.6, 0.3])]  # fmt: skip
    base = v15.report(rows)
    assert base["noul_abstention_rate"] == 1.0
    assert base["intelligence"]["types"]["noul"]["cc"] == pytest.approx(-100.0)
    commit = v15.report(rows, AbstentionPolicy(into_noul=True))
    assert commit["noul_abstention_rate"] == 0.0
    assert commit["intelligence"]["types"]["noul"]["cc"] == pytest.approx(100.0)
    field = v15.report(rows, AbstentionPolicy())  # field mode: the harness scores `noul`
    assert field["intelligence"] == base["intelligence"]
    margin = v15.report(rows, AbstentionPolicy(into_noul=True, margin=0.1))
    assert margin["noul_abstention_rate"] == 0.5  # 0.45 is kept, 0.65 committed
    sharp = v15.report(rows, ServingPolicy(temperatures={"score": 0.5}))
    nmae = sharp["intelligence"]["types"]["score"]["nmae"]
    assert nmae < base["intelligence"]["types"]["score"]["nmae"]
    p = sharpen(np.array([0.1, 0.6, 0.3]), 0.5)
    assert nmae == pytest.approx(abs(p @ np.arange(3) - 1) / 2)
    sweep = v15.sweep(rows, margins=(0.0, 0.3), noul_temperatures=(1.0,),
                      score_temperatures=(1.0,))  # fmt: skip
    assert [p["abstain"] for p in sweep["commit_margin"]] == [0.0, 1.0]  # 0.3: never commits


def records(rng, n=60):
    """A synthetic prediction dump: two views per choice / noul, one per score."""
    out = []
    for i in range(n):
        kind = ("choice", "noul", "score")[i % 3]
        labels = {"choice": ["a", "b", "c"], "noul": ["no", "yes"], "score": ["0", "1", "2"]}[kind]
        gold = int(rng.integers(len(labels)))
        views = 1 if kind == "score" else 2
        logits = rng.normal(size=(views, len(labels))) * 3
        logits[:, gold] += 4  # overconfident and mostly right
        target = np.eye(len(labels))[gold]
        out.append({"id": f"x{i}", "source": f"src{i % 4}", "lang": "en", "lang_guess": "en",
                    "family": f"fam{i % 2}", "type": kind, "labels": labels,
                    "target": target.tolist(), "logits": logits.round(4).tolist(),
                    "orders": [list(range(len(labels)))] * views,
                    "probs": view_probs(logits[None], np.ones_like(logits[None], bool))[0].tolist()})  # fmt: skip
    return out


def test_rows_from_predictions(tmp_path):
    rng = np.random.default_rng(0)
    recs = records(rng, 9)
    recs[0] |= {"source": "jevbench_hard"}
    recs[1] |= {"source": "jevbench_easy", "target": [0.0, 0.5, 0.5]}  # tied gold: skipped
    recs[2] |= {"target": [0.0, 0.2, 0.8]}  # soft: a gold distribution
    rows, skipped = v15.rows_from_predictions(recs, {"choice": 2.0, "noul": 1.0, "score": 0.5})
    assert skipped == {"tied_gold": 1} and len(rows) == 8
    assert rows[0].tier == "hard" and rows[1].tier == v15.UNTIERED
    assert rows[1].gold == "2" and rows[1].gold_probs == pytest.approx([0.0, 0.2, 0.8])
    z = np.asarray(recs[0]["logits"])
    assert rows[0].probs == pytest.approx(view_probs(z[None], np.ones_like(z[None], bool),
                                                     2.0)[0])  # fmt: skip
    as_dumped, _ = v15.rows_from_predictions(recs)  # the dump's own probabilities
    assert as_dumped[0].probs == pytest.approx(recs[0]["probs"])


def test_pool_fit_and_cli(tmp_path):
    rng = np.random.default_rng(1)
    recs = records(rng, 300)
    fit = cal.fit_from_predictions(recs)
    assert fit["pool"]["n"] == 300 and set(fit["pool"]["sources"]) == {f"src{i}" for i in range(4)}
    assert fit["temperatures"]["choice"] > 1.0  # the synthetic logits are overconfident
    path = dump.write(tmp_path / "gen.jsonl.gz", recs)
    pool = dump.write(tmp_path / "pool.jsonl.gz", records(rng, 300))
    out = tmp_path / "v15.json"
    result = CliRunner().invoke(cli, ["eval-v15", str(path), "--fit-on", str(pool),
                                      "--abstention", "noul", "--commit-margin", "0.05",
                                      "--score-temperature", "0.5", "--sweep",
                                      "--out", str(out)])  # fmt: skip
    assert result.exit_code == 0, result.output
    full = json.loads(out.read_text())
    rep = full["reports"]["gen"]
    assert full["pool_fit"]["pool"]["n"] == 300
    assert rep["policy"]["policy"]["abstention"]["margin"] == 0.05
    assert rep["policy"]["noul_abstention_rate"] <= rep["calibrated"]["noul_abstention_rate"]
    assert set(rep["sweep"]) == {"commit_margin", "noul_temperature", "score_temperature"}
    brief = json.loads(result.output)
    assert brief["gen"]["calibrated"]["I"] == pytest.approx(v15.headline(rep["calibrated"])["I"])
    bad = CliRunner().invoke(cli, ["eval-v15", str(path), "--calibration", str(out),
                                   "--fit-on", str(pool)])  # fmt: skip
    assert bad.exit_code != 0


def test_pool_files(tmp_path):
    from tjev.data.item import parse_item, write_jsonl
    from tjev.eval.runs import data_fingerprint, pool_files, read_pool

    def item(i):
        return parse_item({"id": f"i{i}", "state": "s", "question": {
            "type": "noul", "instructions": "?", "criteria": {}}, "expected": "yes"})  # fmt: skip

    (tmp_path / "d").mkdir()
    write_jsonl(tmp_path / "d" / "b.jsonl", [item(1), item(2)])
    write_jsonl(tmp_path / "d" / "a.jsonl", [item(0)])
    write_jsonl(tmp_path / "c.jsonl", [item(2), item(3)])
    assert [p.name for p in pool_files(tmp_path / "d")] == ["a.jsonl", "b.jsonl"]
    pool = read_pool([tmp_path / "d", tmp_path / "c.jsonl"])
    assert [i.id for i in pool] == ["i0", "i1", "i2", "i3"]  # i2 once
    one = data_fingerprint(tmp_path / "c.jsonl", 2)
    assert one != data_fingerprint([tmp_path / "c.jsonl"], 2)  # a pool records its files
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        pool_files(tmp_path / "empty")


def harness() -> Path | None:
    """The JevBench harness checkout (as tests/eval/test_official.py finds it)."""
    for root in (os.environ.get("JEVBENCH_HARNESS"), Path("~/tjev-work/jevbench").expanduser()):
        if root and (Path(root) / "jevbench" / "cli.py").exists():
            return Path(root)
    return None


@pytest.mark.skipif(harness() is None, reason="no JevBench harness checkout")
def test_matches_the_harness_per_item_and_v13_formula():
    """Random distributions on the public items: the harness's own ``score_task`` record,
    scored by :func:`official.v15_item` (from the harness's fields), equals
    :func:`v15.score_item` (recomputed from the distribution); the harness's ECE and its
    v1.3 calibration formula equal ours."""
    root = harness()
    mods = official.harness_modules(root)
    rng = np.random.default_rng(7)
    tasks = official.public_tasks(root, mods)
    scoring = importlib.import_module("jevbench.scoring")  # on sys.path: harness_modules
    conf, hit = [], []
    for task, _, _ in tasks:
        labels = [str(x) for x in task.labels]
        p = rng.dirichlet(np.full(len(labels), 0.4))
        if task.question["type"] == "noul" and rng.random() < 0.5:
            p = np.array([0.5, 0.5])  # in the band
        probs = dict(zip(labels, p.tolist(), strict=True))
        rec = scoring.score_task(probs, task)
        ours = v15.score_item(task.question["type"], labels, str(task.expected), rec["probs"])
        theirs = official.v15_item(task, rec)
        for key, value in theirs.items():
            assert ours[key] == pytest.approx(value), (task.id, key)
        conf.append(max(rec["probs"].values()))
        hit.append(bool(rec["correct"]))
    assert v15.ece(conf, hit) == pytest.approx(mods["metrics"].ece_top_label(
        list(zip(conf, hit, strict=True)))["ece"])  # fmt: skip
    rows = [
        row("choice", "ab", "a", [0.6, 0.4], gold_probs=[0.7, 0.3]),
        row("choice", "ab", "b", [0.8, 0.2], gold_probs=[0.5, 0.5]),
    ]
    out = v15.calibration(rows)["types"]["choice"]
    assert out["points"] == pytest.approx(mods["composite_v13"].calibration(out["ece"], out["tvd"]))
