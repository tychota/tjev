"""Statistical rigour for v1.5 (tjev.eval.v15_stats): the weighted headline reproduces the
scorer, bootstrap intervals, paired comparisons, and the calibration diagnostics on
hand-computed and synthetic cases."""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from tjev.cli import app as cli
from tjev.eval import predictions as dump
from tjev.eval import v15
from tjev.eval import v15_stats as st
from tjev.eval.metrics import view_probs
from tjev.serve.policy import AbstentionPolicy, ServingPolicy

POLICY = ServingPolicy(abstention=AbstentionPolicy(into_noul=True, margin=0.1))


def records(rng, n=150, shift=4.0, prefix="x"):
    """A synthetic dump: tiers, langs, soft choice golds, two views per choice / noul."""
    out = []
    for i in range(n):
        kind = ("choice", "noul", "score")[i % 3]
        labels = {"choice": ["a", "b", "c"], "noul": ["no", "yes"],
                  "score": ["0", "1", "2", "3"]}[kind]  # fmt: skip
        gold = int(rng.integers(len(labels)))
        views = 1 if kind == "score" else 2
        logits = rng.normal(size=(views, len(labels))) * 2
        logits[:, gold] += shift
        target = np.eye(len(labels))[gold]
        if kind == "choice" and i % 7 == 0:
            target = 0.7 * target + 0.1
        source = ("jevbench_easy", "jevbench_original", "jevbench_hard")[i % 5 % 3]
        out.append({"id": f"{prefix}{i}", "source": source, "lang": ("en", "fr")[i % 2],
                    "family": f"fam{i % 2}", "type": kind, "labels": labels,
                    "target": target.tolist(), "logits": logits.round(4).tolist(),
                    "orders": [list(range(len(labels)))] * views,
                    "probs": view_probs(logits[None], np.ones_like(logits[None], bool))[0].tolist()})  # fmt: skip
    return out


def rows_of(recs):
    rows, _ = v15.rows_from_predictions(recs)
    return rows


@pytest.mark.parametrize("policy", [None, POLICY])
def test_weighted_headline_reproduces_the_scorer(policy):
    rows = rows_of(records(np.random.default_rng(0)))
    rows[3] = v15.Row(**{**rows[3].__dict__, "probs": None})  # a failed answer
    want = v15.headline(v15.report(rows, policy, with_families=False))
    got = st.weighted_headline(st.items(rows, policy), np.ones((1, len(rows))))
    for k, v in want.items():
        assert got[k][0] == pytest.approx(v, abs=1e-9), k
    # integer multiplicities = the scorer on the duplicated rows
    w = np.random.default_rng(1).multinomial(len(rows), np.full(len(rows), 1 / len(rows)))
    dup = [r for r, k in zip(rows, w, strict=True) for _ in range(k)]
    want = v15.headline(v15.report(dup, policy, with_families=False))
    got = st.weighted_headline(st.items(rows, policy), w[None])
    for k, v in want.items():
        if v is None:
            assert np.isnan(got[k][0])
        else:
            assert got[k][0] == pytest.approx(v, abs=1e-9), k


def test_bootstrap_interval_contains_the_point():
    rows = rows_of(records(np.random.default_rng(2), 300))
    ci = st.bootstrap(st.items(rows), n=300, seed=0)
    assert set(ci) == set(st.METRICS)
    for k, v in ci.items():
        assert v["lo"] <= v["point"] <= v["hi"], k
        assert v["se"] > 0
    assert ci["I"]["point"] == pytest.approx(v15.headline(v15.report(rows))["I"])
    again = st.bootstrap(st.items(rows), n=300, seed=0)
    assert again == ci  # seeded


def test_paired_identical_inputs_is_zero():
    rows = rows_of(records(np.random.default_rng(3)))
    res = st.paired(st.items(rows), st.items(rows), n=200)
    assert res["n_common"] == len(rows) and res["only_a"] == res["only_b"] == 0
    for k, v in res["metrics"].items():
        assert v["diff"] == 0 and v["lo"] == 0 and v["hi"] == 0, k
        assert v["p_a_gt_b"] == 0.5


def test_paired_matches_by_id_and_drops_the_rest():
    rng = np.random.default_rng(4)
    a = rows_of(records(rng, 120, shift=6.0))
    b = rows_of(records(rng, 120, shift=0.0))  # same ids, much worse
    b = list(reversed(b[10:]))  # order does not matter; 10 items only in A
    b.append(v15.Row(**{**b[0].__dict__, "id": "only-b"}))
    res = st.paired(st.items(a), st.items(b), n=300)
    assert (res["n_common"], res["only_a"], res["only_b"]) == (110, 10, 1)
    i = res["metrics"]["I"]
    assert i["diff"] > 0 and i["lo"] > 0 and i["p_a_gt_b"] > 0.99
    assert i["lo"] <= i["diff"] <= i["hi"]


def test_smooth_ece_calibrated_vs_overconfident():
    rng = np.random.default_rng(5)
    f = rng.uniform(0.0, 1.0, 20000)
    y = (rng.uniform(size=f.size) < f).astype(float)
    assert st.smooth_ece(f, y) < 0.02
    over = 0.5 + 0.5 * (f - 0.5)  # the true rate is half as far from 0.5 as stated
    y_over = (rng.uniform(size=f.size) < over).astype(float)
    assert st.smooth_ece(f, y_over) > 0.08
    assert st.smooth_ece(np.full(100, 0.9), np.zeros(100)) == pytest.approx(0.9, abs=1e-6)
    assert st.smooth_ece([], []) is None


def test_adaptive_ece_and_brier_decomposition():
    conf = [0.9, 0.9, 0.6, 0.6]
    hit = [1, 0, 1, 1]
    # 2 equal-mass bins: (0.6, 0.6) hits 1.0 → 0.4; (0.9, 0.9) hits 0.5 → 0.4
    assert st.adaptive_ece(conf, hit, bins=2) == pytest.approx(0.4)
    d = st.brier_decomposition(conf, hit, bins=2)
    assert d["brier"] == pytest.approx((0.01 + 0.81 + 0.16 + 0.16) / 4)
    assert d["reliability"] == pytest.approx(0.16)
    assert d["resolution"] == pytest.approx(0.0625)  # base 0.75; bins at 1.0 and 0.5
    assert d["uncertainty"] == pytest.approx(0.1875)
    # distinct confidences per bin: Brier = REL - RES + UNC exactly
    assert d["brier"] == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"])


def test_aurc_hand_computed():
    # by confidence: 0.9 right, 0.8 wrong, 0.7 right: risks 0, 1/2, 1/3
    out = st.aurc([0.7, 0.9, 0.8], [1, 1, 0])
    assert out["aurc"] == pytest.approx((0 + 1 / 2 + 1 / 3) / 3)
    # oracle (the error last): 0, 0, 1/3
    assert out["eaurc"] == pytest.approx(out["aurc"] - (1 / 3) / 3)
    assert st.aurc([0.9, 0.1], [1, 0])["eaurc"] == pytest.approx(0.0)


def test_rps_and_diagnostics_per_type_and_lang():
    # gold 0 of 3 levels, p = (0.5, 0.5, 0): CDF (0.5, 1) vs (1, 1): 0.25 / 2
    assert v15.rps({"0": 0.5, "1": 0.5, "2": 0.0}, 0, 3) == pytest.approx(0.125)
    rows = rows_of(records(np.random.default_rng(6)))
    diag = st.calibration_diagnostics(rows)
    assert set(diag["types"]) == {"choice", "noul", "score"}
    assert set(diag["langs"]) == {"en", "fr"}
    assert "rps" in diag["types"]["score"] and "rps" not in diag["types"]["choice"]
    for kind, part in diag["types"].items():
        assert 0 <= part["smece"] <= 1 and part["aurc"] >= part["eaurc"] >= 0
        assert part["n"] == sum(diag["langs"][lang][kind]["n"] for lang in ("en", "fr"))
    assert sum(p["n"] for p in diag["types"].values()) == len(rows)


def test_cli_bootstrap_and_compare(tmp_path):
    rng = np.random.default_rng(7)
    a = dump.write(tmp_path / "a" / "jevbench-k2.jsonl.gz", records(rng, 90, shift=6.0))
    b = dump.write(tmp_path / "b" / "jevbench-k2.jsonl.gz", records(rng, 90, shift=1.0))
    out = tmp_path / "v15.json"
    res = CliRunner().invoke(cli, ["eval-v15", str(a), "--abstention", "noul",
                                   "--commit-margin", "0.1", "--bootstrap", "100",
                                   "--diagnostics", "--out", str(out)])  # fmt: skip
    assert res.exit_code == 0, res.output
    brief = json.loads(res.output)["jevbench-k2"]
    lo, hi = brief["ci95"]["policy"]["I"]
    assert lo <= brief["policy"]["I"] <= hi
    assert "smece" in brief["calibration_diagnostics"]["types"]["noul"]
    full = json.loads(out.read_text())["reports"]["jevbench-k2"]
    assert full["bootstrap"]["n"] == 100
    plain = CliRunner().invoke(cli, ["eval-v15", str(a)])  # default: unchanged brief
    assert set(json.loads(plain.output)["jevbench-k2"]) == {"calibrated"}

    res = CliRunner().invoke(cli, ["eval-compare", str(a), "--against", str(tmp_path / "b"),
                                   "--abstention", "noul", "--bootstrap", "100"])  # fmt: skip
    assert res.exit_code == 0, res.output
    cmp = json.loads(res.output)["jevbench-k2"]
    assert cmp["n_common"] == 90 and cmp["only_a"] == cmp["only_b"] == 0
    assert cmp["calibrated"]["I"]["diff"] > 0 and cmp["policy"]["I"]["p_a_gt_b"] > 0.9
    res = CliRunner().invoke(cli, ["eval-compare", str(a), "--against", str(b)])
    assert res.exit_code == 0, res.output
    missing = CliRunner().invoke(cli, ["eval-compare", str(a), "--against", str(tmp_path)])
    assert missing.exit_code != 0
