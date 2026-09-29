"""The mix builder: block routing, French reweighting, held-out hygiene, a generator-only
build end to end through the CLI (no dataset download)."""

import json

import yaml
from typer.testing import CliRunner

from tjev.cli import app
from tjev.config import load_config
from tjev.data.heldout import REPORT_SEED, SELECT_SEED, heldout_rows
from tjev.data.item import read_jsonl
from tjev.data.mix import (
    BLOCKS,
    FR_TARGET,
    block_of_source,
    conflicting_keys,
    dedup_parts,
    mixture_weights,
)


def test_blocks_and_their_shares():
    assert abs(sum(BLOCKS.values()) - 1.0) < 1e-9
    assert BLOCKS["analysis"] + BLOCKS["writing"] <= 0.20  # text classification stays small
    assert block_of_source("plumb") == "jev"
    assert block_of_source("raid") == "analysis"
    assert block_of_source("gen_register") == "writing"
    assert block_of_source("gen_weekday") == "generators"


def _entry(block, train, lang):
    return {"block": block, "train": train, "lang": lang}


def test_french_target_moves_weights_within_the_block_only():
    sources = {
        "en_a": _entry("replay", 900, "en"),
        "en_b": _entry("replay", 400, "en"),
        "fr_a": _entry("replay", 100, "fr"),
        "gen": _entry("generators", 400, {"en": 280, "fr": 120}),
    }
    w = mixture_weights(sources, {n: s["block"] for n, s in sources.items()})
    replay = w["en_a"] + w["en_b"] + w["fr_a"]
    total = BLOCKS["replay"] + BLOCKS["generators"]
    assert abs(replay - BLOCKS["replay"] / total) < 1e-5  # block share unchanged
    assert abs(w["fr_a"] / replay - FR_TARGET["replay"]) < 1e-5
    assert abs(w["gen"] - BLOCKS["generators"] / total) < 1e-5


def test_heldout_parts_are_disjoint_and_conflicts_are_found():
    row = {"state": "S", "question": {"instructions": "Q?", "criteria": {"a": ""}}}
    parts = {"validation": [row], "calibration": [dict(row)], "test": [dict(row)]}
    assert dedup_parts(parts) == 2
    assert parts["calibration"] == parts["test"] == []
    a = {**row, "expected": "a"}
    b = {**row, "expected": "b"}
    assert len(conflicting_keys([a, b])) == 1
    assert not conflicting_keys([a, dict(a)])


def test_selection_and_reporting_sets_differ():
    assert SELECT_SEED != REPORT_SEED
    select = {r["state"] for r in heldout_rows(5, SELECT_SEED)}
    report = {r["state"] for r in heldout_rows(5, REPORT_SEED)}
    assert not select & report


def test_generator_build_through_the_cli(tmp_path):
    out = tmp_path / "mix"
    args = ["data", "build", str(out), "--blocks", "generators,writing", "--scale", "0.01"]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["jevbench_filter"].startswith("disabled")
    assert set(manifest["mixture"]) == set(manifest["sources"])
    assert abs(sum(manifest["mixture"].values()) - 1.0) < 1e-5
    mix = yaml.safe_load((out / "mix.yaml").read_text())
    cfg = load_config(str(out / "mix.yaml"))
    assert cfg.data.train == tuple(mix["data"]["train"])
    validation = list(read_jsonl(out / "validation.jsonl"))
    train_states = {i.state for f in cfg.data.train for i in read_jsonl(f)}
    assert validation
    assert not {i.state for i in validation} & train_states
    before = dict(manifest["mixture"])
    assert runner.invoke(app, ["data", "reweight", str(out)]).exit_code == 0
    assert json.loads((out / "manifest.json").read_text())["mixture"] == before
