"""mix-v7.5 machinery end to end on a tiny base (no download): base cleaned, a source
rebuilt, v7 generators with whole pairs, jev_distractor from train rows only, an imported
jev_fr_mt folder, v6 held-out files byte-identical and v7 held-out rows kept + appended."""

import json

import pytest
from tests.data.test_v7_mix import _fake_adapter, _fake_load, _row

from tjev.data import mix as mixmod
from tjev.data.item import parse_item, read_jsonl, write_jsonl
from tjev.data.sources import ALL_SOURCES
from tjev.data.sources.base import Source


def _base(tmp_path):
    base = tmp_path / "v7"
    boolq = [_row(f"Base train {i} about apples and pears.", id=f"b{i}", source="boolq")
             for i in range(20)]  # fmt: skip
    plumb = [_row(f"Plumb case {i}: the invoice of {i} euros is due on day {i}.",
                  id=f"plumb-train-{i}-0", source="plumb", license="Apache-2.0",
                  license_use="commercial") for i in range(30)]  # fmt: skip
    scruples = [_row(f"Tom &amp; Jerry fought {i} times.", id=f"s{i}", source="scruples",
                     license="x", license_use="commercial") for i in range(10)]  # fmt: skip
    old = [_row(f"Old fake row {i}.", id=f"f{i}", source="fake_v7") for i in range(5)]
    for name, rows in (("boolq", boolq), ("plumb", plumb), ("scruples", scruples),
                       ("fake_v7", old)):  # fmt: skip
        write_jsonl(base / "train" / f"{name}.jsonl", [parse_item(r) for r in rows])
    write_jsonl(base / "validation.jsonl",
                [parse_item(_row("Held apples.", id="v", source="boolq")),
                 parse_item(_row("Plumb held: the invoice of 7 euros.", id="pv",
                                 source="plumb"))])  # fmt: skip
    write_jsonl(
        base / "calibration.jsonl",
        [parse_item(_row("Plumb cal: the bill is 9 euros.", id="pc", source="plumb"))],
    )
    write_jsonl(base / "validation-v7.jsonl",
                [parse_item(_row("Held &amp; scruple.", id="sv", source="scruples")),
                 parse_item(_row("Old fake held.", id="fv", source="fake_v7"))])  # fmt: skip
    write_jsonl(
        base / "calibration-v7.jsonl",
        [parse_item(_row("Cal scruple.", id="sc", source="scruples"))],
    )
    write_jsonl(
        base / "test" / "boolq.jsonl", [parse_item(_row("Test pears.", id="t", source="boolq"))]
    )
    write_jsonl(
        base / "test" / "fake_v7.jsonl",
        [parse_item(_row("Old fake test.", id="ft", source="fake_v7"))],
    )
    entry = {"lang": "en", "drops": {}}
    (base / "manifest.json").write_text(json.dumps({"sources": {
        "boolq": {**entry, "block": "replay", "train": 20, "origin": "base"},
        "plumb": {**entry, "block": "jev", "train": 30, "origin": "base"},
        "scruples": {**entry, "block": "judge", "train": 10, "origin": "v7"},
        "fake_v7": {**entry, "block": "targeted", "train": 5, "origin": "v7"},
    }}))  # fmt: skip
    return base


def _frmt(tmp_path):
    d = tmp_path / "frmt"
    rows = [_row(f"Cas plumb {i} : la facture de {i} euros est due le jour {i}.",
                 id=f"jev_fr_mt-plumb-train-{i}-0", source="jev_fr_mt", lang="fr",
                 license="Apache-2.0 + CC BY 4.0", license_use="commercial") for i in range(8)]  # fmt: skip
    for r in rows:
        r["lang"] = "fr"
    write_jsonl(d / "train" / "jev_fr_mt.jsonl", [parse_item(r) for r in rows])
    write_jsonl(d / "validation.jsonl", [parse_item({**_row("Facture tenue : 7 euros.",
                id="jev_fr_mt-pv", source="jev_fr_mt"), "lang": "fr"})])  # fmt: skip
    write_jsonl(d / "calibration.jsonl", [])
    (d / "manifest.json").write_text(json.dumps({"sources": {"jev_fr_mt": {
        "block": "jev", "train": 8, "lang": {"fr": 8}, "license_use": "commercial",
        "drops": {}}}}))  # fmt: skip
    return d


def test_extend_v75_end_to_end(tmp_path, monkeypatch):
    base, out = _base(tmp_path), tmp_path / "v75"
    fake = Source("fake_v7", "fake/repo", None, "train", "validation", _fake_adapter, "fake",
                  "en", "MIT", True, cap=40, load=_fake_load, block="targeted",
                  license_use="commercial")  # fmt: skip
    monkeypatch.setitem(ALL_SOURCES, "fake_v7", fake)
    monkeypatch.setattr(mixmod, "heldout_rows", lambda seed: [])
    monkeypatch.setitem(mixmod.GEN_TRAIN, "gen7_return_window_pairs", 40)
    monkeypatch.setitem(mixmod.GEN_TRAIN, "gen7_posterior", 20)
    monkeypatch.setattr(mixmod, "_staged", lambda out, name, scale, seed, f: f())
    manifest = mixmod.extend_v75(
        base, out, fr_mt=_frmt(tmp_path), sources=["fake_v7"],
        generators=["gen7_return_window_pairs", "gen7_posterior"], distractor_rows=10,
        references=(), rebalance_noul=0.7, max_epochs=None,
    )  # fmt: skip
    # mix-v6 held-out files: byte-identical
    for name in ("validation.jsonl", "calibration.jsonl", "test/boolq.jsonl"):
        assert (out / name).read_bytes() == (base / name).read_bytes(), name
    # the base's v7 held-out rows kept (cleaned), the rebuilt source's replaced
    held = list(read_jsonl(out / "validation-v7.jsonl"))
    by = {it.source for it in held}
    assert "Held & scruple." in {it.state for it in held} and "fv" not in {it.id for it in held}
    assert {"fake_v7", "gen7_return_window_pairs", "jev_fr_mt"} <= by
    assert "Old fake test." not in (out / "test" / "fake_v7.jsonl").read_text()
    # base train rows cleaned
    assert all("&amp;" not in it.state for it in read_jsonl(out / "train" / "scruples.jsonl"))
    # pairs whole, ids grouped
    pairs = list(read_jsonl(out / "train" / "gen7_return_window_pairs.jsonl"))
    groups = {}
    for it in pairs:
        groups.setdefault(it.id.rsplit("-", 1)[0], set()).add(it.id.rsplit("-", 1)[1])
    assert pairs and all(v == {"a", "b"} for v in groups.values())
    assert {it.license_use for it in pairs} == {"commercial"}
    # distractor rows from plumb train rows only, with plumb's licence
    dis = list(read_jsonl(out / "train" / "jev_distractor.jsonl"))
    assert dis and {it.license for it in dis} == {"Apache-2.0"}
    dis_held = [it for it in held if it.source == "jev_distractor"]
    assert all("held" in it.state.lower() for it in dis_held)
    fr = list(read_jsonl(out / "train" / "jev_fr_mt.jsonl"))
    assert len(fr) == 8 and manifest["sources"]["jev_fr_mt"]["block"] == "jev"
    assert manifest["blocks"] == mixmod.BLOCKS_V75
    assert manifest["sources"]["fake_v7"]["rebuilt_from"]["train"] == 5
    assert "gen7_return_window_pairs" not in manifest["rebalance_noul"]["sources"]
    mix = (out / "mix.yaml").read_text()
    assert "validation.jsonl" in mix and "jev_fr_mt.jsonl" in mix
    assert manifest["mixture"]["jev_fr_mt"] > 0 and manifest["mixture"]["gen7_posterior"] > 0


def test_v75_shares():
    shares = mixmod.source_shares_v75(3.5)
    assert shares["jev_fr_mt"][1] == 3.5
    gen = sum(v[1] for v in shares.values() if v[0] == "generators")
    assert gen == pytest.approx(100 * mixmod.BLOCKS_V75["generators"])
    assert sum(mixmod.BLOCKS_V75.values()) == pytest.approx(1.0)
    assert mixmod.BLOCKS_V75["jev"] >= 0.33 and mixmod.BLOCKS_V75["generators"] >= 0.12


def _tiered_mix(tmp_path):
    from tests.data.test_balance import _choice, _noul, _write

    from tjev.data.sources.generators import generate
    from tjev.testing import byte_tokenizer_json

    mix = tmp_path / "mix"
    train = {
        "colors": [_choice(i) for i in range(3000)],  # tier B, drawn well under once
        "skewed": [_noul(i, i % 3 != 0) for i in range(1000)],
        "gen_weekday": generate("gen_weekday", 4, 0),  # tier A, drawn more than its rows
    }
    blocks = {"colors": "replay", "skewed": "replay", "gen_weekday": "generators"}
    for name, rows in train.items():
        _write(mix / "train" / f"{name}.jsonl", rows)
    _write(mix / "validation.jsonl", [_noul(1000 + i, True) for i in range(5)])
    _write(mix / "calibration.jsonl", [_noul(2000 + i, True) for i in range(5)])
    manifest = {"scale": 0.001, "gen_scale": 1.0, "files": {},
                "sources": {n: {"block": blocks[n], "train": len(r), "lang": "en", "drops": {}}
                            for n, r in train.items()}}  # fmt: skip
    (mix / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "tok").mkdir()
    (tmp_path / "tok" / "tokenizer.json").write_text(json.dumps(byte_tokenizer_json()))
    return mix


def test_tiers_min_epoch_subsample_and_fresh_generators(tmp_path, monkeypatch):
    monkeypatch.setattr(mixmod, "heldout_rows", lambda seed: [])
    mix = _tiered_mix(tmp_path)
    run = {"steps": 12, "tokens_per_step": 8192, "microbatch_tokens": 4096}
    out = tmp_path / "v75"
    m = mixmod.derive(mix, out, max_epochs=2.0, tiers=True, min_epochs=1.0,
                      tokenizer=tmp_path / "tok", **run)  # fmt: skip
    assert m["tiers"] == {"colors": "B", "gen_weekday": "A", "skewed": "B"}
    sub = m["min_epoch_subsample"]
    assert "colors" in sub and sub["colors"]["rows_after"] < 3000
    assert m["sources"]["colors"]["weight_rows"] == 3000  # the weight does not move
    ep = m["epoch_cap"]["epochs"]
    assert 0.6 <= ep["colors"] <= 1.0 and ep["gen_weekday"] <= 2.0
    assert (out / "validation.jsonl").read_bytes() == (mix / "validation.jsonl").read_bytes()
    rep1 = tmp_path / "rep1"
    r = mixmod.derive(mix, rep1, max_epochs=2.0, tiers=True, tier_a_cap=1.0, min_epochs=1.0,
                      fresh_generators=True, tokenizer=tmp_path / "tok", **run)  # fmt: skip
    fresh = r["fresh_generators"]["gen_weekday"]
    assert fresh["fresh_rows"] > 0 and r["sources"]["gen_weekday"]["train"] > 4
    rows = list(read_jsonl(rep1 / "train" / "gen_weekday.jsonl"))
    assert len({it.id for it in rows}) == len(rows) and rows[-1].license_use == "commercial"
    assert r["epoch_cap"]["epochs"]["gen_weekday"] <= 1.0
    assert (mix / "train" / "gen_weekday.jsonl").read_text().count("\n") == 4  # src untouched
