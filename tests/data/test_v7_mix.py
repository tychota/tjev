"""mix-v7 machinery: per-row licences, the best_fit epoch simulation and cap, the
decontamination of a planted overlap, the cross-source dedup, the per-source shares, and
an end-to-end ``extend`` of a tiny base mix with a fake source (no download)."""

import json

import numpy as np
import pytest

from tjev.data import mix as mixmod
from tjev.data.balance import cap_epochs, make_packer, simulate_stream
from tjev.data.decontam import RefGrams, row_text
from tjev.data.hygiene import HeldIndex, TrainIndex, clean_v7
from tjev.data.item import item_to_json, parse_item, read_jsonl, write_jsonl
from tjev.data.licences import license_use_of
from tjev.data.mix import (
    BLOCKS_V7,
    SOURCE_SHARES_V7,
    filter_license_use,
    mixture_weights_v7,
    origin_of,
)
from tjev.data.pack import BucketPacker
from tjev.data.sources import ALL_SOURCES, V7_SOURCES
from tjev.data.sources.base import Source


def _row(state, instructions="Is it allowed?", expected="yes", **kw):
    return {
        "id": kw.pop("id", state[:20]),
        "state": state,
        "question": {
            "type": "noul",
            "instructions": instructions,
            "criteria": {"false": "no", "true": "yes"},
        },
        "expected": expected,
        "source": kw.pop("source", "s"),
        "family": "f",
        "lang": "en",
        **kw,
    }


# --- licences ------------------------------------------------------------------------------


def test_rows_carry_their_licence_and_use():
    item = parse_item(_row("A state.", license="CC BY-NC 4.0", license_use="non-commercial"))
    assert (item.license, item.license_use) == ("CC BY-NC 4.0", "non-commercial")
    out = item_to_json(item)
    assert out["license_use"] == "non-commercial" and parse_item(out) == item
    assert "license" not in item_to_json(parse_item(_row("No licence.")))  # v6 bytes unchanged
    with pytest.raises(ValueError):
        parse_item(_row("x y z", license_use="research-only"))


def test_licence_use_of_the_v6_sources():
    assert license_use_of("boolq") == "commercial"  # CC BY-SA: share-alike is commercial
    assert license_use_of("xnli_fr") == "non-commercial"
    assert license_use_of("mnli") == "unspecified"
    assert license_use_of("gen_weekday") == "commercial"


def test_v7_sources_name_their_block_and_licence_use():
    for name, source in V7_SOURCES.items():
        assert source.block in BLOCKS_V7, name
        assert source.license_use in ("commercial", "non-commercial", "unspecified"), name
        assert source.commercial_ok == (source.license_use == "commercial"), name
        assert name in ALL_SOURCES and mixmod.block_of_source(name) == source.block


def test_filter_license_use_drops_rows_not_sources(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    rows = [
        _row(
            f"state {i}",
            id=f"r{i}",
            license="x",
            license_use="non-commercial" if i % 2 else "commercial",
        )
        for i in range(6)
    ]
    write_jsonl(src / "train" / "a.jsonl", [parse_item(r) for r in rows])
    write_jsonl(src / "train" / "b.jsonl", [parse_item(r) for r in rows if "1" in r["id"]])
    manifest = {"sources": {"a": {"train": 6, "lang": "en"}, "b": {"train": 1, "lang": "en"}}}
    present = {"a": "jev", "b": "judge"}
    filter_license_use(src, out, manifest, present, ("non-commercial",))
    kept = list(read_jsonl(out / "train" / "a.jsonl"))
    assert len(kept) == 3 and {it.license_use for it in kept} == {"commercial"}
    assert present == {"a": "jev"} and manifest["sources"]["b"]["train"] == 0


# --- best_fit simulation and the cap ----------------------------------------------------


def _lengths(rng, n, lo, hi, opts=4):
    return (rng.integers(lo, hi, n), rng.integers(lo, hi + 100, n), np.full(n, opts))


def test_best_fit_packs_tighter_and_keeps_the_mixture():
    rng = np.random.default_rng(0)
    lengths = {"short": _lengths(rng, 4000, 60, 500), "long": _lengths(rng, 1500, 900, 3000)}
    probs = {"short": 0.8, "long": 0.2}
    run = {"steps": 150, "tokens_per_step": 65536, "microbatch_tokens": 4096}
    bucket = simulate_stream(lengths, probs, **run, packing="bucket")
    best = simulate_stream(lengths, probs, **run, packing="best_fit")
    assert best["fill"] > 0.95 > bucket["fill"]
    assert best["segments_per_step"] > bucket["segments_per_step"]
    share = best["by_source"]["short"]["trained"] / best["segments"]
    assert abs(share - 0.8) < 0.03  # packing delays segments, it never selects them
    with pytest.raises(ValueError):
        make_packer(
            "next_fit",
            segment_at=None,
            buckets=(1024,),
            microbatch_tokens=4096,
            accumulation=1,
            slots=8,
            labels=26,
        )
    assert isinstance(
        make_packer(
            "bucket",
            segment_at=None,
            buckets=(1024,),
            microbatch_tokens=4096,
            accumulation=1,
            slots=8,
            labels=26,
        ),
        BucketPacker,
    )


def test_cap_epochs_under_best_fit_holds_every_source_and_reports_other_lengths():
    rng = np.random.default_rng(1)
    lengths = {
        "tiny": _lengths(rng, 300, 80, 400),
        "mid": _lengths(rng, 3000, 80, 400),
        "big": _lengths(rng, 20000, 80, 600),
        "other": _lengths(rng, 20000, 80, 600),
    }
    weights = {"tiny": 0.3, "mid": 0.2, "big": 0.3, "other": 0.2}
    block_of = {"tiny": "a", "mid": "a", "big": "a", "other": "b"}
    cap = cap_epochs(
        lengths,
        weights,
        2.0,
        block_of=block_of,
        steps=120,
        packing="best_fit",
        also_steps=(60,),
        log=lambda m: None,
    )
    assert max(cap["epochs"].values()) <= 2.0
    assert "tiny" in cap["capped"] and cap["packing"] == "best_fit"
    assert cap["block_shares"]["b"]["after"] == pytest.approx(0.2, abs=1e-4)  # block kept
    assert cap["at_steps"]["60"]["epochs"]["tiny"] < cap["epochs"]["tiny"]


# --- decontamination and dedup ----------------------------------------------------------


PLANTED = "the warehouse closes its loading dock at seven every friday evening"


def test_planted_overlap_is_dropped_and_templates_are_not():
    refs = RefGrams()
    refs.add(
        "evalset",
        [
            f"Q{i}: boilerplate header line repeated in every single item here, case {i}"
            for i in range(40)
        ]
        + [f"Note: {PLANTED}."],
    )
    assert refs.stats["evalset"]["template_grams"] > 0
    held = HeldIndex.of([parse_item(_row("held out passage about refunds", id="h"))])
    items = [
        parse_item(_row(f"Shipping memo. {PLANTED.upper()}! Plan ahead.", id="a")),
        parse_item(_row("boilerplate header line repeated in every single item here, ok", id="b")),
        parse_item(_row("An unrelated state about invoices and totals.", id="c")),
    ]
    kept, drops = clean_v7(items, held, set(), TrainIndex.empty(), "repo/x", refs=refs)
    assert [it.id for it in kept] == ["b", "c"]
    assert drops["decontam_8gram:evalset"] == 1


def test_row_text_reads_every_string_and_json_strings():
    text = row_text({"q": [{"content": "hello"}], "fn": json.dumps({"doc": "deep text"}), "n": 3})
    assert "hello" in text and "deep text" in text


def test_cross_source_dedup_respects_origins():
    passage = "The museum opens at nine and closes at five, except on public holidays " * 3
    held = HeldIndex.of([])
    train, seen = TrainIndex.empty(), set()
    base = [parse_item(_row(passage, "Is it open at ten?", id="base", source="v6src"))]
    clean_v7(base, held, seen, train, "google/boolq", index_only=True)
    same_origin = [parse_item(_row(passage, "Is it open on holidays?", "no", id="o"))]
    kept, drops = clean_v7(same_origin, held, seen, train, "google/boolq")
    assert len(kept) == 1  # one passage, another question, same dataset: kept
    other = [
        parse_item(_row(passage, "Is it open on holidays?", "no", id="t1")),  # same q + state
        parse_item(_row(passage + " Tickets cost ten euros.", "Do tickets cost money?", id="t2")),
        parse_item(_row("A fresh passage about trains and their timetables.", id="t3")),
    ]
    kept, drops = clean_v7(other, held, seen, train, "tasksource/x")
    assert [it.id for it in kept] == ["t3"]
    assert drops["train_exact_duplicate"] + drops["train_near_duplicate"] == 1
    assert drops["train_cross_source_state"] == 1


def test_origins_group_related_repos():
    assert origin_of("gen_weekday").startswith("generated")
    assert origin_of("boolq") == origin_of("twin_boolq") == "google/boolq"


# --- shares ------------------------------------------------------------------------------


def _src(block, train, lang="en"):
    return {"block": block, "train": train, "lang": lang}


def test_v7_shares_blocks_groups_and_absent_groups():
    assert abs(sum(BLOCKS_V7.values()) - 1) < 1e-9
    sources = {
        "openjev": _src("jev", 35000),
        "openjev_x": _src("jev", 75000),
        "decider_teacher": _src("jev", 28000),
        "plumb": _src("jev", 5000),
        "gen_a": _src("generators", 8000),
        "gen_b": _src("generators", 2000),
        "tasksource_jev": _src("breadth", 300000),
        "cold_law_fr": _src("targeted", 8000, "fr"),
        "sharc": _src("targeted", 19000),
        "mystery": _src("targeted", 1000),
    }
    block_of = {n: s["block"] for n, s in sources.items()}
    manifest = {"sources": sources, "source_shares": SOURCE_SHARES_V7, "blocks": BLOCKS_V7}
    w = mixture_weights_v7(manifest, block_of)
    present = {"jev", "generators", "breadth", "targeted"}
    total = sum(BLOCKS_V7[b] for b in present)
    for block in present:
        mass = sum(v for n, v in w.items() if block_of[n] == block)
        assert mass == pytest.approx(BLOCKS_V7[block] / total, abs=1e-5)
    jev = BLOCKS_V7["jev"] / total
    # only openjev / decider / plumb groups are present in jev: 8.0 : 6.0 : 0.6
    assert w["openjev"] + w["openjev_x"] == pytest.approx(jev * 8.0 / 14.6, abs=1e-5)
    assert w["openjev_x"] / w["openjev"] == pytest.approx((75000 / 35000) ** 0.5, rel=1e-3)
    assert w["gen_a"] / w["gen_b"] == pytest.approx(2.0, rel=1e-3)  # n^0.5 outside groups
    assert w["mystery"] > 0  # an unlisted source of a listed block gets a group of its own


# --- extend end to end ---------------------------------------------------------------------


def _fake_load(split):
    from datasets import Dataset

    n = 60 if split == "train" else 30
    return Dataset.from_list(
        [
            {
                "text": f"Fake {split} case {i}: the parcel weighs {i} kilograms and ships "
                f"on day {i % 7}.",
                "y": i % 2,
            }
            for i in range(n)
        ]
    )


def _fake_adapter(row, rng, meta):
    return [
        {
            "state": row["text"],
            "question": {
                "type": "noul",
                "instructions": "Is the parcel heavy?",
                "criteria": {"false": "light", "true": "heavy"},
            },
            "expected": "yes" if row["y"] else "no",
        }
    ]


def test_extend_keeps_the_base_and_adds_a_source(tmp_path, monkeypatch):
    base, out = tmp_path / "v6", tmp_path / "v7"
    rows = [
        _row(f"Base train {i} about apples and pears.", id=f"b{i}", source="boolq")
        for i in range(20)
    ]
    write_jsonl(base / "train" / "boolq.jsonl", [parse_item(r) for r in rows])
    write_jsonl(
        base / "validation.jsonl", [parse_item(_row("Held apples.", id="v", source="boolq"))]
    )
    write_jsonl(
        base / "test" / "boolq.jsonl", [parse_item(_row("Test pears.", id="t", source="boolq"))]
    )
    (base / "manifest.json").write_text(
        json.dumps(
            {"sources": {"boolq": {"block": "replay", "train": 20, "lang": "en", "drops": {}}}}
        )
    )
    fake = Source(
        "fake_v7",
        "fake/repo",
        None,
        "train",
        "validation",
        _fake_adapter,
        "fake",
        "en",
        "CC BY-NC 4.0",
        False,
        cap=40,
        load=_fake_load,
        block="targeted",
        license_use="non-commercial",
    )
    monkeypatch.setitem(ALL_SOURCES, "fake_v7", fake)
    monkeypatch.setattr(mixmod, "heldout_rows", lambda seed: [])
    manifest = mixmod.extend(
        base, out, sources=["fake_v7"], references=(), rebalance_noul=None, max_epochs=None
    )
    assert (out / "validation.jsonl").read_bytes() == (base / "validation.jsonl").read_bytes()
    assert (out / "test" / "boolq.jsonl").read_bytes() == (
        base / "test" / "boolq.jsonl"
    ).read_bytes()
    base_rows = list(read_jsonl(out / "train" / "boolq.jsonl"))
    assert len(base_rows) == 20 and {it.license_use for it in base_rows} == {"commercial"}
    new = list(read_jsonl(out / "train" / "fake_v7.jsonl"))
    assert 0 < len(new) <= 40 and {it.license_use for it in new} == {"non-commercial"}
    held = list(read_jsonl(out / "validation-v7.jsonl"))
    assert held and {it.source for it in held} == {"fake_v7"}
    assert manifest["sources"]["fake_v7"]["block"] == "targeted"
    assert manifest["block_sampled_share"]["targeted"] == pytest.approx(
        BLOCKS_V7["targeted"] / (BLOCKS_V7["targeted"] + BLOCKS_V7["replay"]), abs=1e-4
    )
    assert 0 < manifest["license_use_sampled_share"]["non-commercial"] < 1
    yaml_text = (out / "mix.yaml").read_text()
    assert "validation.jsonl" in yaml_text and "fake_v7.jsonl" in yaml_text


def test_a_source_template_shared_with_held_out_rows_is_not_an_overlap():
    frame = "please decide according to the operating policy of the regional depot below"
    refs = RefGrams()
    refs.add_groups(
        "heldout",
        {
            "src": [f"{frame}: held case {i}" for i in range(3)]
            + [f"Unique held passage: {PLANTED}."]
        },
    )
    items = [
        parse_item(_row(f"{frame}: train case number {i} with {i} boxes", id=f"r{i}"))
        for i in range(30)
    ]
    items.append(parse_item(_row(f"Copied: {PLANTED}", id="leak")))
    kept, drops = clean_v7(items, HeldIndex.of([]), set(), TrainIndex.empty(), "repo/y", refs=refs)
    assert len(kept) == 30 and drops["decontam_8gram:heldout"] == 1


def test_band_index_agrees_with_near_hits():
    from tjev.data.near import BandIndex, MinHasher, near_hits, shingle_hashes

    rng = np.random.default_rng(3)
    words = [f"w{i}" for i in range(400)]
    base = [" ".join(rng.choice(words, 60)) for _ in range(200)]
    query = [t if i % 2 else " ".join(rng.choice(words, 60)) for i, t in enumerate(base[:80])]
    query = [q.replace("w1 ", "w2 ", 1) for q in query]  # near, not equal
    h = MinHasher()
    sig = lambda ts: np.stack([h.signature(shingle_hashes(t)) for t in ts])  # noqa: E731
    index = BandIndex()
    index.add(sig(base[:100]), tag=0)
    index.add(sig(base[100:]), tag=1)
    got = index.query(sig(query))
    assert (got == near_hits(sig(query), sig(base))).all() and got.sum() == 40
    assert not index.query(sig(query), exclude_tag=0).any()  # all near rows are tag 0
    keyed = BandIndex(keyed=True)
    keyed.add(sig(base), np.zeros(len(base), np.uint64))
    assert keyed.query(sig(query), np.ones(len(query), np.uint64)).sum() == 0
    assert keyed.query(sig(query), np.zeros(len(query), np.uint64)).sum() == 40


def test_held_out_rows_of_the_same_origin_do_not_count_as_overlap():
    passage = " ".join(f"visa{i} article{i}" for i in range(40))  # long shared boilerplate
    refs = RefGrams()
    refs.add_groups(
        "heldout",
        {"repo/a": [passage], "repo/b": ["other text " * 5]},
        min_shared=25,
        min_share=0.2,
    )
    item = [parse_item(_row(f"{passage} Decision text.", id="x"))]
    kept, _ = clean_v7(item, HeldIndex.of([]), set(), TrainIndex.empty(), "repo/a", refs=refs)
    assert len(kept) == 1
    kept, drops = clean_v7(item, HeldIndex.of([]), set(), TrainIndex.empty(), "repo/c", refs=refs)
    assert not kept and drops["decontam_8gram:heldout"] == 1
