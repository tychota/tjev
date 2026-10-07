"""mix-v4b corrections on tiny synthetic data: the yes/no rebalance (target, cap,
determinism, train only) and the epoch cap through the weights (tjev.data.balance), and a
mix derived end to end through the CLI."""

import json
from pathlib import Path

import numpy as np
import yaml
from typer.testing import CliRunner

from tjev.cli import app
from tjev.data.balance import cap_epochs, rebalance_file, rebalance_noul, redistribute
from tjev.data.item import parse_item, read_jsonl
from tjev.data.mix import capped_weights, derive, reweight
from tjev.testing import byte_tokenizer_json


def _noul(i: int, yes: bool, source: str = "skewed") -> dict:
    return {
        "id": f"{source}-{i}",
        "state": f"Report {i}: the door is {'open' if yes else 'closed'}.",
        "question": {
            "type": "noul",
            "instructions": "Is the door open?",
            "criteria": {"false": "closed", "true": "open"},
        },
        "expected": "yes" if yes else "no",
        "source": source,
        "family": "extraction",
        "lang": "en",
    }


def _choice(i: int, source: str = "colors") -> dict:
    colors = ["red", "green", "blue"]
    return {
        "id": f"{source}-{i}",
        "state": f"Item {i}: the light is {colors[i % 3]}.",
        "question": {
            "type": "choice",
            "instructions": "Which colour is the light?",
            "criteria": {c: f"the light is {c}" for c in colors},
        },
        "expected": colors[i % 3],
        "source": source,
        "family": "extraction",
        "lang": "en",
    }


def _items(rows: list[dict]):
    return [parse_item(r) for r in rows]


def _share(items) -> float:
    noul = [it for it in items if it.question.type == "noul"]
    return sum(it.gold == 1 for it in noul) / len(noul)


def test_rebalance_brings_the_majority_down_to_the_target():
    items = _items([_noul(i, i % 10 != 0) for i in range(100)] + [_choice(i) for i in range(20)])
    kept, stats = rebalance_noul(items, 0.7, seed=3)
    out = [items[i] for i in kept]
    assert stats["yes_before"] == 90 and stats["yes_after"] == 23  # 23 / 33 ≈ 0.7
    assert abs(_share(out) - 0.7) < 0.01
    assert sum(it.question.type == "choice" for it in out) == 20  # other types untouched
    assert sum(it.gold == 0 for it in out if it.question.type == "noul") == 10  # minority kept
    assert kept == sorted(kept)  # file order kept
    assert rebalance_noul(items, 0.7, seed=3)[0] == kept  # seeded
    assert rebalance_noul(items, 0.7, seed=4)[0] != kept
    assert not stats["capped"]


def test_rebalance_low_side_in_range_and_loss_cap():
    low = _items([_noul(i, i % 5 == 0) for i in range(100)])  # 20% yes
    _, stats = rebalance_noul(low, 0.7, seed=0)
    assert stats["yes_after"] == 20 and stats["noul_after"] == 67  # 20 / 67 ≈ 0.3
    fair = _items([_noul(i, i % 2 == 0) for i in range(50)])
    assert rebalance_noul(fair, 0.7)[0] == list(range(50))
    raid = _items([_noul(i, i % 100 >= 3) for i in range(100)])  # 97% yes, nothing else
    _, stats = rebalance_noul(raid, 0.7, max_loss=0.6)
    assert stats["dropped"] == 60 and stats["capped"]
    assert stats["yes_share_after"] == round(37 / 40, 4)
    mixed = _items([_noul(i, i % 100 >= 3) for i in range(100)] + [_choice(i) for i in range(100)])
    _, stats = rebalance_noul(mixed, 0.7, max_loss=0.6)  # the cap counts all the source's rows
    assert not stats["capped"] and stats["yes_share_after"] == 0.7


def test_rebalance_file_keeps_lines_byte_identical(tmp_path):
    rows = [_noul(i, i % 10 != 0) for i in range(50)]
    rows[1]["state"] += " line separator"  # not a line break in JSONL
    src = tmp_path / "in.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    stats, kept = rebalance_file(src, tmp_path / "out.jsonl", 0.7, max_loss=0.6, seed=0)
    lines = (tmp_path / "out.jsonl").read_text().split("\n")[:-1]
    original = src.read_text().split("\n")
    assert len(lines) == len(kept) == stats["rows_after"]
    assert all(line in original for line in lines)
    assert [it.id for it in read_jsonl(tmp_path / "out.jsonl")] == [it.id for it in kept]


def _lengths(n: int, tokens: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.full(n, tokens), np.full(n, tokens + 20), np.full(n, 2)


def test_epoch_cap_moves_the_freed_mass_to_the_others():
    lengths = {"small": _lengths(50, 100), "mid": _lengths(400, 100), "big": _lengths(4000, 100)}
    weights = {"small": 0.4, "mid": 0.3, "big": 0.3}
    run = {"steps": 20, "tokens_per_step": 4096, "microbatch_tokens": 1024}
    cap = cap_epochs(lengths, weights, 2.0, log=lambda _: None, **run)
    assert cap["history"][0]["max_epochs"] > 2.0  # small was over the cap
    assert max(cap["epochs"].values()) <= 2.0
    assert cap["capped"][0] == "small" and cap["scale"]["small"] < 1.0
    assert cap["scale"]["big"] > 1.0  # the freed mass went to the uncapped sources
    if "mid" not in cap["capped"]:
        assert abs(cap["scale"]["big"] - cap["scale"]["mid"]) < 1e-6  # proportionally
    final = capped_weights({n: round(w, 6) for n, w in weights.items()}, cap)
    assert abs(sum(final.values()) - 1.0) < 1e-5
    assert all(abs(final[n] - cap["weights"][n]) < 2e-6 for n in final)


def test_redistribute_keeps_block_shares_and_overflows_full_blocks():
    w = {"a1": 0.2, "a2": 0.2, "b1": 0.3, "b2": 0.1, "c1": 0.2}
    block_of = {"a1": "a", "a2": "a", "b1": "b", "b2": "b", "c1": "c"}
    shares = {"a": 0.4, "b": 0.4, "c": 0.2}
    out, overflow = redistribute(w, {"a1": 0.1, "b1": 0.2}, block_of, shares)
    assert overflow == {}
    assert abs(out["a2"] - 0.3) < 1e-12  # a1's freed 0.1 stays in block a
    assert abs(out["b2"] - 0.2) < 1e-12 and abs(out["c1"] - 0.2) < 1e-12
    out, overflow = redistribute(w, {"c1": 0.1}, block_of, shares)  # c is all capped
    assert abs(overflow["c"] - 0.1) < 1e-12
    assert abs(out["a1"] + out["a2"] - 0.45) < 1e-12  # c's 0.1 shared ∝ a, b shares
    assert abs(out["b1"] + out["b2"] - 0.45) < 1e-12
    assert abs(out["a1"] / out["a2"] - 1.0) < 1e-12  # proportionally within a block
    assert abs(sum(out.values()) - 1.0) < 1e-12


def test_epoch_cap_preserves_block_shares():
    lengths = {
        "small": _lengths(50, 100),
        "big": _lengths(4000, 100),
        "other": _lengths(4000, 100),
        "gen": _lengths(30, 100),
    }
    weights = {"small": 0.3, "big": 0.2, "other": 0.3, "gen": 0.2}
    block_of = {"small": "jev", "big": "jev", "other": "replay", "gen": "generators"}
    run = {"steps": 20, "tokens_per_step": 4096, "microbatch_tokens": 1024}
    cap = cap_epochs(lengths, weights, 2.0, block_of=block_of, log=lambda _: None, **run)
    assert max(cap["epochs"].values()) <= 2.0
    assert cap["preserve_blocks"] and {"small", "gen"} <= set(cap["capped"])
    shares = cap["block_shares"]
    assert "generators" in cap["overflow"]  # its only source is capped: it overflows
    assert shares["generators"]["after"] < 0.2
    gen_short = 0.2 - shares["generators"]["after"]
    assert abs(shares["replay"]["after"] - 0.3 - gen_short * 0.3 / 0.8) < 1e-4
    assert abs(shares["jev"]["after"] - 0.5 - gen_short * 0.5 / 0.8) < 1e-4  # big took small's


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _tiny_mix(root: Path) -> Path:
    mix = root / "mix"
    train = {
        "skewed": [_noul(i, i % 5 != 0) for i in range(120)],
        "colors": [_choice(i) for i in range(150)],
        "gen_small": [_noul(i, i % 2 == 0, "gen_small") for i in range(12)],
    }
    blocks = {"skewed": "replay", "colors": "replay", "gen_small": "generators"}
    for name, rows in train.items():
        _write(mix / "train" / f"{name}.jsonl", rows)
    _write(mix / "validation.jsonl", [_noul(1000 + i, True) for i in range(5)])
    _write(mix / "calibration.jsonl", [_noul(2000 + i, True) for i in range(5)])
    _write(mix / "test" / "skewed.jsonl", [_noul(3000 + i, True) for i in range(5)])
    manifest = {
        "sources": {
            n: {"block": blocks[n], "train": len(r), "lang": "en", "drops": {}}
            for n, r in train.items()
        },
        "files": {},
    }
    (mix / "manifest.json").write_text(json.dumps(manifest))
    (root / "tok").mkdir()
    (root / "tok" / "tokenizer.json").write_text(json.dumps(byte_tokenizer_json()))
    return mix


def test_derive_rebalances_train_only_and_caps_epochs(tmp_path):
    mix = _tiny_mix(tmp_path)
    before = {p.name: p.read_bytes() for p in mix.rglob("*.jsonl")}
    run = {"steps": 10, "tokens_per_step": 8192, "microbatch_tokens": 4096}
    out = tmp_path / "mix-b"
    manifest = derive(
        mix, out, rebalance_noul=0.7, max_epochs=2.0, tokenizer=tmp_path / "tok", **run
    )
    assert {p.name: p.read_bytes() for p in mix.rglob("*.jsonl")} == before  # src untouched
    for name in ("validation.jsonl", "calibration.jsonl", "test/skewed.jsonl"):
        assert (out / name).read_bytes() == (mix / name).read_bytes()  # held-out as is
    table = manifest["rebalance_noul"]["sources"]
    assert set(table) == {"skewed"} and table["skewed"]["yes_share_after"] == 0.7
    assert manifest["sources"]["skewed"]["train"] == table["skewed"]["rows_after"]
    assert abs(_share(list(read_jsonl(out / "train" / "skewed.jsonl"))) - 0.7) < 0.01
    assert (out / "train" / "colors.jsonl").read_bytes() == before["colors.jsonl"]
    cap = manifest["epoch_cap"]
    assert "gen_small" in cap["capped"] and max(cap["epochs"].values()) <= 2.0
    assert cap["preserve_blocks"] and "generators" in cap["overflow"]
    mixture = yaml.safe_load((out / "mix.yaml").read_text())["data"]["mixture"]
    assert abs(sum(mixture.values()) - 1.0) < 1e-5
    assert all(
        str(out.resolve()) in f
        for f in yaml.safe_load((out / "mix.yaml").read_text())["data"]["train"]
    )
    assert reweight(out)["mixture"] == mixture  # the cap is part of the manifest


def test_derive_through_the_cli(tmp_path):
    mix = _tiny_mix(tmp_path)
    out = tmp_path / "mix-b"
    args = ["data", "derive", str(mix), str(out), "--rebalance-noul", "0.7"]
    args += ["--max-epochs", "2.0", "--tokenizer", str(tmp_path / "tok"), "--steps", "10"]
    args += ["--tokens-per-step", "8192", "--microbatch-tokens", "4096"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["derived_from"]["mix"] == str(mix)
    assert manifest["rebalance_noul"]["target"] == 0.7


def test_epoch_cap_honours_per_source_caps():
    lengths = {"noisy": _lengths(100, 100), "mid": _lengths(400, 100), "big": _lengths(4000, 100)}
    weights = {"noisy": 0.3, "mid": 0.3, "big": 0.4}
    run = {"steps": 20, "tokens_per_step": 4096, "microbatch_tokens": 1024}
    free = cap_epochs(lengths, weights, 3.0, log=lambda _: None, **run)
    assert free["epochs"]["noisy"] > 1.0
    cap = cap_epochs(lengths, weights, 3.0, caps={"noisy": 1.0}, log=lambda _: None, **run)
    assert cap["epochs"]["noisy"] <= 1.0
    assert cap["epochs"]["mid"] > free["epochs"]["mid"]  # the freed mass went to the others
    assert cap["caps"] == {"noisy": 1.0}
