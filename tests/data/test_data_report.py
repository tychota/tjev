"""scripts/data_report.py on a tiny synthetic mix: hashing, near-duplicates, language and
artefact heuristics, label balance, the packer simulation, and the whole report."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from tjev.data.tokenize import PromptTokenizer
from tjev.testing import byte_tokenizer_json, tiny_items

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("data_report", ROOT / "scripts" / "data_report.py")
dr = importlib.util.module_from_spec(_spec)
sys.modules["data_report"] = dr  # dataclasses resolve their module
_spec.loader.exec_module(dr)


def _sig(text: str, hasher=None) -> np.ndarray:
    return (hasher or dr.MinHasher()).signature(dr.shingle_hashes(text))


def test_minhash_estimates_jaccard_and_clusters_near_duplicates():
    hasher = dr.MinHasher()
    base = " ".join(f"word{i}" for i in range(200))
    near = base.replace("word100", "changed")  # 3 of ~198 shingles differ
    other = " ".join(f"other{i}" for i in range(200))
    sigs = np.stack([_sig(t, hasher) for t in (base, near, other, other + " tail")])
    est = dr.estimated_jaccard(sigs, np.array([0, 0]), np.array([1, 2]))
    assert est[0] > 0.9
    assert est[1] < 0.1
    labels = dr.near_duplicate_clusters(sigs)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    match = dr.near_matches(sigs[[1, 2]], sigs[[0]])
    assert match.tolist() == [0, -1]


def test_group_counts_finds_conflicts():
    keys = np.array([1, 1, 2, 2, 3], np.uint64)
    values = np.array([7, 8, 9, 9, 9], np.uint64)
    size, distinct = dr.group_counts(keys, values)
    assert size.tolist() == [2, 2, 2, 2, 1]
    assert distinct.tolist() == [2, 2, 1, 1, 1]


def test_language_and_artefact_heuristics():
    assert dr.guess_lang("Le chat est sur la table et il ne veut pas descendre.") == "fr"
    assert dr.guess_lang("The cat is on the table and it does not want to come down.") == "en"
    assert dr.guess_lang("42 17 99") is None
    assert "html_entity" in dr.artefact_flags("Tom &amp; Jerry")
    assert "mojibake" in dr.artefact_flags("cafÃ© crÃ¨me")
    assert "placeholder" in dr.artefact_flags("Dear {customer_name}, …")
    assert dr.artefact_flags('{\n "a": 1\n}') == []


def test_raw_problems():
    assert dr.raw_problems(
        {
            "id": 1,
            "state": "s",
            "question": {},
            "source": "x",
            "lang": "en",
            "family": "f",
            "target": {"a": 0.5, "b": 0.2},
        }
    ) == ["target_sum"]
    assert "target_nan" in dr.raw_problems({"target": {"a": float("nan")}})


def test_simulation_counts_epochs_and_drops():
    lengths = {
        "short": (np.full(100, 100), np.full(100, 120), np.full(100, 3)),
        "long": (np.full(10, 5000), np.full(10, 5000), np.full(10, 2)),
    }
    sim = dr.simulate_stream(
        lengths, {"short": 0.9, "long": 0.1}, steps=20, tokens_per_step=4096, microbatch_tokens=1024
    )
    assert sim["steps"] == 20
    assert sim["by_source"]["long"]["trained"] == 0
    assert sim["dropped_too_long"] > 0
    assert sim["by_source"]["short"]["trained"] == sim["segments"]
    assert 0.8 < sim["fill"] <= 1.0


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_report_on_a_tiny_mix(tmp_path):
    mix = tmp_path / "mix"
    rows = tiny_items(60, seed=0)
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    # an exact duplicate, a conflicting label and a French row written in English
    doors = by_source["doors"]
    doors.append(dict(doors[0], id="dup"))
    flipped = "no" if doors[0]["expected"] == "yes" else "yes"
    doors.append(dict(doors[0], id="conflict", expected=flipped))
    doors.append(
        dict(
            doors[1],
            id="lang",
            lang="fr",
            state="The door of the house is open and the wind is in the hall.",
        )
    )
    for name, items in by_source.items():
        _write(mix / "train" / f"{name}.jsonl", items)
    held = tiny_items(9, seed=1)
    held[0] = dict(by_source["colors"][0], id="leak")  # an exact copy of a train row
    _write(mix / "validation.jsonl", held)
    _write(mix / "calibration.jsonl", tiny_items(6, seed=2))
    weights = {n: 1.0 / len(by_source) for n in by_source}
    (mix / "mix.yaml").write_text(yaml.safe_dump({"data": {"mixture": weights}}))
    (tmp_path / "tok").mkdir()
    (tmp_path / "tok" / "tokenizer.json").write_text(json.dumps(byte_tokenizer_json()))
    tok = PromptTokenizer(tmp_path / "tok")

    def encode(texts):
        return [len(tok.encode(t)) for t in texts]

    report = dr.analyse(
        mix,
        None,
        encode_batch=encode,
        steps=5,
        tokens_per_step=8192,
        microbatch_tokens=4096,
        log=lambda _: None,
    )
    assert report["overall"]["rows"] == 63
    assert report["duplication"]["exact_duplicate_rows"] >= 1
    assert report["duplication"]["conflicting_rows"] >= 2
    assert report["leakage"]["validation"]["exact_state_instructions"] >= 1
    assert report["flags"]["lang_mismatch:state"]["total"] == 1
    assert report["simulation"]["steps"] == 5
    for name in by_source:
        assert report["sources"][name]["epochs"] > 0
        assert report["sources"][name]["tokens_plain"]["p50"] > 0
    titles = [r["title"] for r in report["recommendations"]]
    assert "Conflicting labels" in titles
    assert any(t.startswith("Train overlaps validation") for t in titles)
    md = dr.markdown(report)
    assert "## Leakage" in md
    assert "doors" in md
