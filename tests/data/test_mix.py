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
    Builder,
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


def test_long_generators_are_opt_in_and_the_prompt_limit_is_a_parameter(tmp_path, monkeypatch):
    """The long-doc generator is built only into a long mix (--max-prompt-tokens > 4096) or
    when named; the hygiene drop of prompts over the limit follows the build's limit."""
    from tjev.data import mix as mix_mod

    assert "gen_longdoc_refund_policy" not in mix_mod.select_sources(set(), set())
    assert "gen_longdoc_refund_policy" in mix_mod.select_sources(set(), set(), long=True)
    named = mix_mod.select_sources({"gen_longdoc_refund_policy"}, set())
    assert named == ["gen_longdoc_refund_policy"]
    # characters as tokens: every handbook (> 20k characters) is over 4096, none over 100k
    monkeypatch.setattr(mix_mod, "token_counter", lambda _: lambda texts: [len(t) for t in texts])
    for limit, kept in ((4096, False), (100_000, True)):
        out = tmp_path / f"mix-{limit}"
        manifest = mix_mod.build(out, sources=["gen_longdoc_refund_policy"], scale=0.01,
                                 tokenizer="chars", max_prompt_tokens=limit)  # fmt: skip
        entry = manifest["sources"]["gen_longdoc_refund_policy"]
        assert (entry["train"] > 0) == kept, entry
        assert manifest["hygiene"]["max_prompt_tokens"] == limit


def test_gen_scale_adds_generator_rows_without_changing_the_first_ones():
    base, _, _ = Builder(scale=0.01).generator("gen_weekday")
    more, parts, _ = Builder(scale=0.01, gen_scale=2.0).generator("gen_weekday")
    assert len(base) == 40 and len(more) == 80
    assert more[: len(base)] == base  # a superset: same seed, same first rows
    assert parts == Builder(scale=0.01).generator("gen_weekday")[1]  # held-out unchanged


def test_gen_scale_through_the_cli(tmp_path):
    out = tmp_path / "mix"
    args = ["data", "build", str(out), "--sources", "gen_weekday", "--scale", "0.01",
            "--gen-scale", "2"]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["gen_scale"] == 2.0
    assert manifest["sources"]["gen_weekday"]["train"] == 80


def test_open_only_keeps_redistributable_sources_and_records_licences(tmp_path):
    out = tmp_path / "mix"
    args = ["data", "build", str(out), "--sources", "gen_weekday,gen_register", "--scale",
            "0.01", "--open-only"]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["open_only"] is True
    for entry in manifest["sources"].values():
        assert entry["redistributable"] == "yes" and entry["card_licence"].startswith("Apache")
