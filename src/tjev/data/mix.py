"""Build the decision mixture (mix-v4) as JevBench-native JSONL (``tjev data build``), or
derive a corrected mix from a built one (mix-v4b, ``tjev data derive``).

Layout of a built mix::

    OUT/train/<source>.jsonl   training rows (capped per source)
    OUT/validation.jsonl       per-source held-out rows + held-out generator seeds/surface text
    OUT/calibration.jsonl      disjoint held-out rows for temperature fitting
    OUT/test/<source>.jsonl    per-source test rows (never used for selection)
    OUT/mix.yaml               the data section of a run config: train files, validation, mixture
    OUT/manifest.json          counts, licenses, languages, drops, hashes

Held-out public rows come from each dataset's evaluation split (or, without one, from a
hashed group split of train), hash-partitioned into validation / calibration / test.
Training rows whose normalised (state, instructions) match a held-out row are removed, and
so are rows whose (state, instructions, criteria) occur with different gold labels.
Generators use other seeds and their own held-out surface text for the evaluation parts.
Every training row of every source is checked against the public JevBench items
(:class:`tjev.data.jevbench.JevBenchFilter`); drops are counted as ``jevbench_overlap``.
Once every source is built, a cross-source post-pass (:mod:`tjev.data.hygiene`) drops train
rows equal or near-duplicate to any held-out row, prompts over the largest bucket, and exact
duplicates. Two optional corrections follow (:func:`correct`, :mod:`tjev.data.balance`): a
yes/no prior rebalance of the train files and a cap on each source's effective epochs
through the mixture weights (mix-v4b).
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tjev.data.heldout import REPORT_SEED, SELECT_SEED, heldout_rows
from tjev.data.hygiene import MAX_PROMPT_TOKENS, HeldIndex, clean, token_counter
from tjev.data.item import parse_item, read_jsonl, write_jsonl
from tjev.data.jevbench import JevBenchFilter, JevBenchGrams
from tjev.data.licences import GENERATED, is_open, licence_of, license_use_of
from tjev.data.near import normalized
from tjev.data.render import RenderSpec
from tjev.data.sources import (
    ALL_SOURCES,
    GENERATORS,
    JEV_SOURCES,
    LONG_GENERATORS,
    TARGETED_SOURCES,
    TEXT_SOURCES,
    Source,
    generate,
)
from tjev.utils import file_hash, write_json

# Block shares of the training mixture (docs/data.md). "jev" is the contender data, the
# largest block: the target is JevBench-style decisions. "analysis" + "writing" (text
# classification and templated writing checks) stay at 15% together. Within a block,
# sources are weighted ∝ n^0.5. Blocks absent from a build are redistributed.
# mix-v6: the "targeted" block (sources aimed at the weak JevBench families, abstention
# and French decisions; docs/data.md, mix-v6) takes its share mostly from replay / analysis.
BLOCKS = {"jev": 0.37, "generators": 0.23, "replay": 0.13, "analysis": 0.065, "writing": 0.04,
          "targeted": 0.165}  # fmt: skip
# French share of the sampled mix (target 20%): the jev block is English only and the
# generators draw 30% French, so the French-only sources carry these shares of their block:
# 0.4·0 + 0.25·0.30 + 0.2·0.38 + 0.1·0.35 + 0.05·0.34 ≈ 0.20. Only the weights move.
# mix-v6: the targeted block carries its French sources (COLD, ArGiMi, the PIAF twins, the
# French TempReason rows) at "targeted" of the block, to keep the mix at ~20% French.
FR_TARGET = {"replay": 0.38, "analysis": 0.35, "targeted": 0.30}
WRITING = frozenset({"gen_spelling_error", "gen_register"})
# rows per generator (default GEN_DEFAULT); a long-doc row is ~10k tokens, ~3× a
# gen_long_refund_policy row
GEN_TRAIN = {"gen_long_refund_policy": 2000, "gen_longdoc_refund_policy": 1000}
GEN_DEFAULT = 4000
# Multiplies a source's n^0.5 weight before its block is renormalised (the block share
# never changes). gen_long_refund_policy is 2% of the segments but 16% of the prompt
# tokens and 4.4 epochs at 3,000 steps in mix-v3 (reports/data/mix-v3): halved in mix-v4.
# mix-v5: the machine-translated XNLI / PAWS-X train sets are second, train-only sources
# next to the human-translated dev splits; they keep half their weight.
WEIGHT_SCALE = {
    "gen_long_refund_policy": 0.5,
    "gen_longdoc_refund_policy": 0.25,
    "xnli_fr_mt": 0.5,
    "pawsx_fr_mt": 0.5,
    # mix-v6: long rows (a dossier is ~3k tokens, an excerpt or a conversation ~1.5k)
    "eikos_long": 0.25,
    "cuad": 0.5,
    "helpsteer3": 0.5,
    "argimi_fr": 0.5,
}
# mix-v5: per-source epoch caps under --max-epochs (docs/fr-data.md): small French sources
# with noisy labels (tweet_sentiment_fr, mlma_fr) and easy French yes/no sources the model
# memorises (hc3_fr, textdetox_fr, allocine_fr; docs/calibration.md) get one epoch
SOURCE_MAX_EPOCHS = {
    "tweet_sentiment_fr": 1.0,
    "mlma_fr": 1.0,
    "hc3_fr": 1.0,
    "textdetox_fr": 1.0,
    "allocine_fr": 1.0,
    # mix-v6: negation twins repeat passages of their base sources: one epoch
    "twin_squad2": 1.0,
    "twin_boolq": 1.0,
    "twin_piaf_fr": 1.0,
}
# mix-v4b: a source loses at most this share of its train rows to the yes/no rebalance.
REBALANCE_MAX_LOSS = 0.6
HELD_OUT_PER_PART = 300
HELD_OUT_GROUP_SHARE = 0.1  # sources without an eval split: share of groups held out
MAX_OPTIONS = RenderSpec().max_options  # answer letters A..Z
PARTS = ("validation", "calibration", "test")
# mix-v6: the targeted sources' validation / calibration rows go to their own files, so the
# mix-v5 held-out files stay byte-identical and the new sources are evaluated apart
SEPARATE_HELD = {"targeted": "-v6"}

Row = dict[str, Any]
Parts = dict[str, list[Row]]


def block_of_source(name: str) -> str:
    source = ALL_SOURCES.get(name)
    if source is not None and source.block is not None:  # mix-v7 sources name their block
        return source.block
    if name in TARGETED_SOURCES:
        return "targeted"
    if name in JEV_SOURCES:
        return "jev"
    if name in TEXT_SOURCES:
        return "analysis"
    if name in ALL_SOURCES:
        return "replay"
    return "writing" if name in WRITING else "generators"


def key_of(row: Row) -> str:
    return normalized(row["state"]) + "\x1f" + normalized(row["question"]["instructions"])


def menu_key_of(row: Row) -> str:
    """(state, instructions, criteria): gold labels are comparable only on one menu (the
    same question over two different option sets is not a conflict)."""
    criteria = row["question"].get("criteria")
    return key_of(row) + "\x1f" + json.dumps(criteria, sort_keys=True, ensure_ascii=False)


def part_of(identifier: str) -> str:
    return PARTS[int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) % 3]


def _seed_of(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def label_names(ds: Any) -> list[str] | None:
    for column in ("label", "intent"):
        feature = ds.features.get(column) if ds.features else None
        if feature is not None and hasattr(feature, "names"):
            return list(feature.names)
    if "label" in ds.column_names and isinstance(ds[0]["label"], str):
        return sorted(set(ds["label"]))
    return None


def adapt(
    source: Source, ds: Any, indices: np.ndarray, seed: int, meta: dict, split: str
) -> tuple[list[Row], Counter]:
    """Dataset rows → valid item rows (ids, source, family, language, licence set), and drop
    counts. The licence and its use come from the source unless the adapter set them per
    row (mix-v7: every row records them)."""
    card, use = licence_of(source.name)[0], license_use_of(source.name)
    rows, drops = [], Counter()
    for i in indices:
        rng = np.random.default_rng([seed, int(i)])
        try:
            produced = source.adapter(ds[int(i)], rng, meta)
        except (KeyError, ValueError, TypeError) as e:
            drops[f"adapter:{type(e).__name__}"] += 1
            continue
        for j, row in enumerate(produced):
            if isinstance(row["state"], str) and len(re.sub(r"\W+", "", row["state"])) < 3:
                drops["empty_state"] += 1  # e.g. a template around empty fields
                continue
            row.update(  # adapters may set a per-row family / lang (contenders)
                id=f"{source.name}-{split}-{i}-{j}",
                source=source.name,
                family=row.get("family") or source.family,
                lang=row.get("lang") or source.lang,
                license=row.get("license") or card,
                license_use=row.get("license_use") or use,
            )
            try:
                item = parse_item(row)
            except (KeyError, ValueError) as e:
                drops[f"invalid:{str(e)[:40]}"] += 1
                continue
            if len(item.question.labels) > MAX_OPTIONS:
                drops["too_many_options"] += 1
                continue
            rows.append(row)
    return rows, drops


def load(source: Source, split: str) -> Any:
    from datasets import load_dataset

    if source.load is not None:
        return source.load(split)
    return load_dataset(source.repo, source.config, split=split, **source.load_kwargs)


def group_split(source: Source, ds: Any) -> tuple[np.ndarray, np.ndarray]:
    """(train, held-out) indices of a source without an eval split, by hashed group."""
    assert source.group is not None
    held = np.asarray(
        [
            int(hashlib.sha256(source.group(row).encode()).hexdigest()[:8], 16) % 1000
            < HELD_OUT_GROUP_SHARE * 1000
            for row in ds
        ]
    )
    return np.flatnonzero(~held), np.flatnonzero(held)


def shingles(text: str, n: int = 5) -> set[str]:
    words = normalized(text).split()
    return {" ".join(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


def near_duplicates(
    train_rows: list[Row], held_rows: list[Row], threshold: float = 0.8
) -> set[int]:
    """Indices of train rows whose state has 5-gram Jaccard ≥ threshold with a held-out
    state (exhaustive: meant for small sources such as jailbreak)."""
    held = [shingles(r["state"]) for r in held_rows]
    out = set()
    for i, row in enumerate(train_rows):
        s = shingles(row["state"])
        if any(len(s & h) >= threshold * len(s | h) for h in held):
            out.add(i)
    return out


def conflicting_keys(rows: list[Row]) -> set[str]:
    """(state, instructions, criteria) keys that occur with different gold labels."""
    labels: dict[str, set] = {}
    for r in rows:
        gold = r.get("expected") or json.dumps(r.get("target"), sort_keys=True)
        labels.setdefault(menu_key_of(r), set()).add(str(gold))
    return {k for k, v in labels.items() if len(v) > 1}


def dedup_parts(parts: Parts) -> int:
    """Drop held-out rows whose key already occurs in an earlier part (validation, then
    calibration, then test) or earlier in the same part; returns the number dropped."""
    seen, dropped = set(), 0
    for name in PARTS:
        kept = []
        for row in parts[name]:
            key = key_of(row)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            kept.append(row)
        parts[name] = kept
    return dropped


@dataclass(frozen=True)
class Builder:
    """How to build one source: size scale, seed, and the contamination filter (or None).
    ``gen_scale`` multiplies the generators' row counts on top of ``scale``: generated
    rows are free, and a generator's first n rows do not depend on how many are drawn, so
    a larger count only adds rows (fewer repeats at the same block share)."""

    scale: float = 1.0
    seed: int = 0
    jevbench: JevBenchFilter | None = None
    gen_scale: float = 1.0

    def filtered(self, rows: list[Row]) -> tuple[list[Row], Counter]:
        if self.jevbench is None:
            return rows, Counter()
        return self.jevbench.filter(rows)

    def public(self, name: str) -> tuple[list[Row], Parts, Counter]:
        source = ALL_SOURCES[name]
        train = load(source, source.train_split)
        meta = {"label_names": label_names(train)}
        rng = np.random.default_rng([self.seed, _seed_of(name)])
        cap = int(source.cap * self.scale)
        if source.train_only:
            ev, pool, ev_idx = train, np.arange(len(train)), np.arange(0)
        elif source.eval_split:
            ev, pool = load(source, source.eval_split), np.arange(len(train))
            ev_idx = rng.permutation(len(ev))[: 3 * HELD_OUT_PER_PART * 2]
        elif source.group is not None:
            ev = train
            pool, ev_idx = group_split(source, train)
            ev_idx = rng.permutation(ev_idx)[: 3 * HELD_OUT_PER_PART * 2]
        else:
            raise ValueError(f"{name}: no eval split and no group key for a held-out split")
        held, drops = adapt(source, ev, ev_idx, self.seed + 1, meta, "eval")
        train_idx = rng.permutation(pool)[: int(cap * 1.2)]
        train_rows, train_drops = adapt(source, train, train_idx, self.seed, meta, "train")
        drops.update(train_drops)
        parts: Parts = {p: [] for p in PARTS}
        for row in held:
            part = parts[part_of(row["id"])]
            if len(part) < HELD_OUT_PER_PART:
                part.append(row)
        drops["heldout_duplicate"] += dedup_parts(parts)
        held_rows = [r for p in parts.values() for r in p]
        held_keys = {key_of(r) for r in held_rows}
        kept = [r for r in train_rows if key_of(r) not in held_keys]
        drops["train_overlaps_heldout"] += len(train_rows) - len(kept)
        kept, overlap = self.filtered(kept)
        drops.update(overlap)
        if source.near_dedup:
            near = near_duplicates(kept, held_rows)
            kept = [r for i, r in enumerate(kept) if i not in near]
            drops["train_near_duplicate_of_heldout"] += len(near)
        if conflicts := conflicting_keys(kept):
            n = len(kept)
            kept = [r for r in kept if menu_key_of(r) not in conflicts]
            drops["train_label_conflict"] += n - len(kept)
        return kept[:cap], parts, drops

    def generator(self, name: str) -> tuple[list[Row], Parts, Counter]:
        n = int(GEN_TRAIN.get(name, GEN_DEFAULT) * self.scale * self.gen_scale)
        train = generate(name, n, self.seed)
        parts: Parts = {
            part: generate(name, HELD_OUT_PER_PART // 2, self.seed + offset, split="heldout")
            for part, offset in zip(PARTS, (101, 202, 303), strict=True)
        }
        drops = Counter({"heldout_duplicate": dedup_parts(parts)})
        held_keys = {key_of(r) for p in parts.values() for r in p}
        kept = [r for r in train if key_of(r) not in held_keys]
        drops["train_overlaps_heldout"] = len(train) - len(kept)
        kept, overlap = self.filtered(kept)
        drops.update(overlap)
        return kept, parts, drops


def select_sources(sources: set[str], blocks: set[str], long: bool = False) -> list[str]:
    """Source and generator names to build (all by default), in build order. The long
    generators (prompts over 4096 tokens) are in the default set only when ``long``."""
    names = [*ALL_SOURCES, *GENERATORS, *LONG_GENERATORS]
    unknown = (sources - set(names)) | (blocks - set(BLOCKS))
    if unknown:
        raise KeyError(f"unknown sources/blocks: {sorted(unknown)}")
    return [
        n
        for n in names
        if (n in sources if sources else long or n not in LONG_GENERATORS)
        and (not blocks or block_of_source(n) in blocks)
    ]


def build(
    out: str | Path,
    *,
    sources: Iterable[str] = (),
    blocks: Iterable[str] = (),
    commercial_only: bool = False,
    scale: float = 1.0,
    gen_scale: float = 1.0,
    seed: int = 0,
    jevbench_public: str | Path | None = None,
    tokenizer: str | Path | None = None,
    rebalance_noul: float | None = None,
    max_loss: float = REBALANCE_MAX_LOSS,
    max_epochs: float | None = None,
    preserve_blocks: bool = True,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    open_only: bool = False,
    passage_dedup: bool = True,
    jevbench_ngram: bool = True,
) -> dict:
    """Build the mix into ``out``; returns the manifest. ``sources`` / ``blocks`` build a
    subset (the block shares are then renormalised over the blocks present). ``tokenizer``
    (a folder with tokenizer.json) enables the drop of prompts over ``max_prompt_tokens``
    (the run's largest bucket: 4096 for mix-v4; 16384 for the long buckets, which also adds
    the long generators to the default sources).
    ``gen_scale`` multiplies the generators' rows (:class:`Builder`). ``open_only`` keeps
    only sources whose card licence allows redistribution (:mod:`tjev.data.licences`:
    permissive or share-alike; no unknown, NC or Claude-generated source).
    ``rebalance_noul`` / ``max_epochs``: the mix-v4b corrections (:func:`correct`).
    ``passage_dedup`` / ``jevbench_ngram``: the mix-v6 train checks of the post-pass
    (:mod:`tjev.data.hygiene`: no train state equal or near to a held-out state; no train
    row sharing a word 8-gram with a public JevBench item). The blocks of
    ``SEPARATE_HELD`` write their validation / calibration rows to their own files."""
    out = Path(out)
    manifest: dict[str, Any] = {"sources": {}, "blocks": BLOCKS, "seed": seed, "scale": scale,
                                "max_prompt_tokens": max_prompt_tokens}  # fmt: skip
    if gen_scale != 1.0:
        manifest["gen_scale"] = gen_scale
    if open_only:
        manifest["open_only"] = True
    jevbench, grams = None, None
    if jevbench_public is not None and Path(jevbench_public).exists():
        jevbench = JevBenchFilter.from_jsonl(jevbench_public)
        grams = JevBenchGrams.from_jsonl(jevbench_public) if jevbench_ngram else None
        manifest["jevbench_filter"] = {
            "public": str(jevbench_public),
            "sha256": file_hash(jevbench_public),
        }
        print(f"[jevbench] filtering against {len(jevbench.states)} public states", flush=True)
    else:
        manifest["jevbench_filter"] = "disabled: public items not found (tjev data jevbench)"
        print("[jevbench] WARNING: no public items, contamination filter off", flush=True)
    manifest["hygiene_checks"] = {"passage_dedup": passage_dedup,
                                  "jevbench_8gram": grams is not None}  # fmt: skip
    builder = Builder(scale=scale, seed=seed, jevbench=jevbench, gen_scale=gen_scale)
    held: dict[str, Any] = {"validation": [], "calibration": [], "test": {}}
    for suffix in SEPARATE_HELD.values():
        held[f"validation{suffix}"], held[f"calibration{suffix}"] = [], []
    block_of: dict[str, str] = {}
    long = max_prompt_tokens > MAX_PROMPT_TOKENS
    for name in select_sources(set(sources), set(blocks), long):
        if open_only and not is_open(name):
            continue
        if name in GENERATORS or name in LONG_GENERATORS:
            train, parts, drops = builder.generator(name)
            langs: Any = dict(Counter(r["lang"] for r in train))
            info = {"license": "generated (Apache-2.0)", "commercial_ok": True,
                    "family": {**GENERATORS, **LONG_GENERATORS}[name][1]}  # fmt: skip
        else:
            source = ALL_SOURCES[name]
            if commercial_only and not source.commercial_ok:
                continue
            print(f"[public] {name} …", flush=True)
            try:
                train, parts, drops = builder.public(name)
            except Exception as e:  # a dead mirror must not kill the whole build
                print(f"  FAILED {name}: {type(e).__name__}: {e}", flush=True)
                manifest["sources"][name] = {"error": f"{type(e).__name__}: {e}"}
                continue
            langs = source.lang
            info = {"license": source.license, "commercial_ok": source.commercial_ok,
                    "family": source.family}  # fmt: skip
        write_jsonl(out / "train" / f"{name}.jsonl", [parse_item(r) for r in train])
        block_of[name] = block_of_source(name)
        suffix = SEPARATE_HELD.get(block_of[name], "")
        held[f"validation{suffix}"] += parts["validation"]
        held[f"calibration{suffix}"] += parts["calibration"]
        held["test"][name] = parts["test"]
        card, redistributable, note = licence_of(name)
        manifest["sources"][name] = {
            "block": block_of[name],
            "train": len(train),
            **{k: len(v) for k, v in parts.items()},
            "lang": langs,
            **info,
            "card_licence": card,
            "redistributable": redistributable,
            "licence_note": note,
            "drops": dict(drops),
        }
        print(f"  {name}: {len(train)} train, drops {dict(drops)}", flush=True)
    parts_out = [p for p in held if p != "test" and (held[p] or "-" not in p)]
    for part in parts_out:
        write_jsonl(out / f"{part}.jsonl", [parse_item(r) for r in held[part]])
    for name, rows in held["test"].items():
        write_jsonl(out / "test" / f"{name}.jsonl", [parse_item(r) for r in rows])
    held_rows = [*chain(*(held[p] for p in parts_out)), *chain(*held["test"].values())]
    held_rows += [*heldout_rows(seed=SELECT_SEED), *heldout_rows(seed=REPORT_SEED)]
    held_items = [parse_item(r) for r in held_rows]
    if jevbench_public is not None and Path(jevbench_public).exists():
        held_items += list(read_jsonl(jevbench_public))
    manifest["hygiene"] = hygiene(out, manifest, list(block_of), held_items, tokenizer,
                                  max_prompt_tokens, passages=passage_dedup,
                                  grams=grams)  # fmt: skip
    block_of = {n: b for n, b in block_of.items() if manifest["sources"][n]["train"] > 0}
    return correct(
        out, out, manifest, block_of, rebalance_noul=rebalance_noul, max_loss=max_loss,
        max_epochs=max_epochs, preserve_blocks=preserve_blocks, tokenizer=tokenizer, seed=seed,
    )  # fmt: skip


def correct(
    src: Path,
    out: Path,
    manifest: dict,
    block_of: dict[str, str],
    *,
    rebalance_noul: float | None = None,
    max_loss: float = REBALANCE_MAX_LOSS,
    max_epochs: float | None = None,
    preserve_blocks: bool = True,
    tokenizer: str | Path | None = None,
    seed: int = 0,
    tiers: bool = False,
    tier_a_cap: float | None = None,
    min_epochs: float | None = None,
    fresh_generators: bool = False,
    held_dir: Path | None = None,
    jevbench_public: str | Path | None = None,
    **run: Any,
) -> dict:
    """The mix-v4b corrections (:mod:`tjev.data.balance`), then ``mix.yaml``: train files
    of ``src`` are rebalanced into ``out`` (yes-share into [1 - r, r]), then the weights
    are capped at ``max_epochs`` effective epochs, the freed mass staying in its block
    when ``preserve_blocks`` (``run``: steps, tokens_per_step,
    microbatch_tokens of the simulated run). Held-out parts are never touched."""
    from tjev.data import balance

    items: dict[str, list] = {}
    if rebalance_noul is not None:
        table = {}
        for name in block_of:
            path, dest = src / "train" / f"{name}.jsonl", out / "train" / f"{name}.jsonl"
            stats, kept = balance.rebalance_file(
                path, dest, rebalance_noul, max_loss=max_loss, seed=_seed_of(name) + seed
            )
            items[name] = kept
            if stats["dropped"] or stats.get("capped"):
                table[name] = stats
                entry = manifest["sources"][name]
                entry["train"] = stats["rows_after"]
                entry["drops"] = {**entry["drops"], "noul_rebalance": stats["dropped"]}
                if isinstance(entry["lang"], dict):
                    entry["lang"] = dict(Counter(it.lang for it in kept))
                print(
                    f"  [rebalance] {name}: yes {stats['yes_share_before']} → "
                    f"{stats['yes_share_after']}, {stats['rows_before']} → "
                    f"{stats['rows_after']} rows",
                    flush=True,
                )
        manifest["rebalance_noul"] = {
            "target": rebalance_noul, "max_loss": max_loss, "seed": seed, "sources": table,
        }  # fmt: skip
    elif src != out:
        for name in block_of:
            dest = out / "train" / f"{name}.jsonl"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / "train" / f"{name}.jsonl", dest)
    manifest.pop("epoch_cap", None)
    if max_epochs is not None:
        if tokenizer is None:
            raise ValueError("the epoch cap measures prompt lengths: it needs a tokenizer")
        count = token_counter(tokenizer)
        lengths, rows = {}, {}
        for name in block_of:
            its = items.get(name) or list(read_jsonl(out / "train" / f"{name}.jsonl"))
            lengths[name] = balance.prompt_lengths(its, count)
            rows[name] = len(its)
            print(f"  [epoch-cap] {name}: {rows[name]} rows measured", flush=True)
        base = weights_of(manifest, block_of)
        caps = {
            n: c
            for n, c in {**SOURCE_MAX_EPOCHS, **V7_MAX_EPOCHS, **V75_MAX_EPOCHS}.items()
            if n in block_of
        }  # fmt: skip
        if tiers:  # mix-v7.5: tier A (core) ≤ tier_a_cap, tier B (the rest) ≤ 1 epoch
            caps = tier_caps(block_of, tier_a_cap or max_epochs)
            manifest["tiers"] = {n: tier_of(n, b) for n, b in sorted(block_of.items())}
        if fresh_generators:  # mix-v7.5-rep1: generators get new rows instead of repeats
            manifest["fresh_generators"] = fresh_generator_rows(
                out, manifest, block_of, base, lengths, rows, count, held_dir or out,
                jevbench_public, seed=seed, **run,
            )  # fmt: skip

        def run_cap() -> dict:
            return balance.cap_epochs(
                lengths, {n: base[n] for n in block_of}, max_epochs,
                block_of=block_of if preserve_blocks else None, rows=rows, seed=seed,
                caps=caps, **run,
            )  # fmt: skip

        cap = run_cap()  # the cap runs on the full pools (tier B can absorb)
        if min_epochs:  # then, last, every source ≥ min_epochs: rows resized to their draws
            manifest["min_epoch_subsample"] = min_epoch_subsample(
                out, manifest, cap["epochs"], rows, lengths, items, min_epochs, seed
            )
            if manifest["min_epoch_subsample"]:  # measured again, never re-capped
                cap = balance.remeasure(cap, lengths, rows, seed=seed, **run)
        cap.pop("weights")
        manifest["epoch_cap"] = cap
    return write_mixture(out, manifest, block_of)


HELD_FILES = ("validation.jsonl", "calibration.jsonl", "heldout-select.jsonl",
              "validation-v6.jsonl", "calibration-v6.jsonl",
              "validation-v7.jsonl", "calibration-v7.jsonl")  # fmt: skip


def derive(
    src: str | Path,
    out: str | Path,
    *,
    rebalance_noul: float | None = None,
    max_loss: float = REBALANCE_MAX_LOSS,
    max_epochs: float | None = None,
    preserve_blocks: bool = True,
    tokenizer: str | Path | None = None,
    seed: int = 0,
    drop_license_use: Iterable[str] = (),
    refresh_shares: bool = False,
    v75_shares: float | None = None,
    tiers: bool = False,
    tier_a_cap: float | None = None,
    min_epochs: float | None = None,
    fresh_generators: bool = False,
    jevbench_public: str | Path | None = None,
    **run: Any,
) -> dict:
    """A new mix from a built one (mix-v4 → mix-v4b): the held-out files (validation,
    calibration, heldout-select, test/) are copied byte for byte, the train files go
    through :func:`correct`. ``src`` is never modified. ``drop_license_use`` (mix-v7c:
    ``non-commercial``) first removes the train rows of those licence uses (per row; a
    source left empty leaves the mix)."""
    src, out = Path(src).expanduser(), Path(out).expanduser()
    if src.resolve() == out.resolve():
        raise ValueError("derive writes a new mix: OUT must differ from SRC")
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    for name in HELD_FILES:
        if (src / name).exists():
            shutil.copyfile(src / name, out / name)
    (out / "test").mkdir(exist_ok=True)
    for path in sorted((src / "test").glob("*.jsonl")):
        shutil.copyfile(path, out / "test" / path.name)
    manifest["derived_from"] = {
        "mix": str(src),
        "files": dict(manifest.get("files", {})),
    }
    if refresh_shares:  # mix-v7: the current BLOCKS_V7 / SOURCE_SHARES_V7 / FR_TARGET_V7
        manifest.update(blocks=BLOCKS_V7, source_shares=SOURCE_SHARES_V7, fr_target=FR_TARGET_V7)
    if v75_shares is not None:  # mix-v7.5: the v7.5 shares, jev_fr_mt's group at that share
        manifest.update(blocks=BLOCKS_V75, source_shares=source_shares_v75(v75_shares),
                        fr_target=FR_TARGET_V7)  # fmt: skip
    present = {n: s["block"] for n, s in manifest["sources"].items() if s.get("train", 0) > 0}
    if drop_license_use := tuple(drop_license_use):
        src = filter_license_use(src, out, manifest, present, drop_license_use)
    if any((tiers, tier_a_cap, min_epochs, fresh_generators)):
        manifest["epoch_policy"] = {"tiers": tiers, "tier_a_cap": tier_a_cap,
                                    "min_epochs": min_epochs,
                                    "fresh_generators": fresh_generators}  # fmt: skip
    if fresh_generators and src != out:  # new rows are appended to out's train files
        for name in present:
            dest = out / "train" / f"{name}.jsonl"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src / "train" / f"{name}.jsonl", dest)
        src = out
    return correct(
        src, out, manifest, present, rebalance_noul=rebalance_noul, max_loss=max_loss,
        max_epochs=max_epochs, preserve_blocks=preserve_blocks, tokenizer=tokenizer,
        seed=seed, tiers=tiers, tier_a_cap=tier_a_cap, min_epochs=min_epochs,
        fresh_generators=fresh_generators, held_dir=out, jevbench_public=jevbench_public,
        **run,
    )  # fmt: skip


def hygiene(
    out: Path,
    manifest: dict,
    names: list[str],
    held: list,
    tokenizer: str | Path | None,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    *,
    passages: bool = False,
    grams: JevBenchGrams | None = None,
) -> dict:
    """The cross-source post-pass (:mod:`tjev.data.hygiene`) over the train files, in build
    order: rewrites them, updates the manifest's counts and drops."""
    index = HeldIndex.of(held)
    count = token_counter(tokenizer) if tokenizer is not None else None
    if count is None:
        print("[hygiene] WARNING: no tokenizer, prompts over the bucket are kept", flush=True)
    seen: set[int] = set()
    total: Counter = Counter()
    for name in names:
        path = out / "train" / f"{name}.jsonl"
        kept, drops = clean(list(read_jsonl(path)), index, seen, count_tokens=count,
                            limit=max_prompt_tokens, passages=passages,
                            grams=grams)  # fmt: skip
        write_jsonl(path, kept)
        entry = manifest["sources"][name]
        entry["train"] = len(kept)
        entry["drops"] = {**entry["drops"], **drops}
        if isinstance(entry["lang"], dict):
            entry["lang"] = dict(Counter(it.lang for it in kept))
        total.update(drops)
        if drops:
            print(f"  [hygiene] {name}: {len(kept)} train, drops {dict(drops)}", flush=True)
    return {
        "held_rows": len(held),
        "tokenizer": str(tokenizer) if tokenizer is not None else "disabled",
        "max_prompt_tokens": max_prompt_tokens,
        "passage_dedup": passages,
        "jevbench_8gram": grams is not None,
        "drops": dict(total),
    }


def reweight(out: str | Path) -> dict:
    """Recompute ``mix.yaml`` (and the manifest's weights) of a built mix."""
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    present = {n: s for n, s in manifest["sources"].items() if s.get("train", 0) > 0}
    return write_mixture(out, manifest, {n: s["block"] for n, s in present.items()})


def _weight_rows(entry: dict) -> float:
    """The row count a source's n^0.5 weight is computed on: its train rows, or (mix-v7.5)
    ``weight_rows``, the count before a min-epoch subsample / fresh generator rows, so
    that resizing a source to its draws does not move the mixture."""
    return entry.get("weight_rows") or entry["train"]


def fr_fraction(entry: dict) -> float:
    lang = entry["lang"]
    return lang.get("fr", 0) / sum(lang.values()) if isinstance(lang, dict) else float(lang == "fr")


def mixture_weights(
    sources: dict, block_of: dict[str, str], fr_target: dict[str, float] | None = None
) -> dict[str, float]:
    """Block share (absent blocks redistributed) × n^0.5 within the block; then, in the
    blocks of FR_TARGET, French-only sources are rescaled (and the rest shrunk) so that the
    block is that share French. Block shares never change."""
    shares = {b: s for b, s in BLOCKS.items() if b in set(block_of.values())}
    total = sum(shares.values())
    weights = {}
    for block, share in shares.items():
        members = [n for n, b in block_of.items() if b == block]
        sizes = {n: _weight_rows(sources[n]) ** 0.5 * WEIGHT_SCALE.get(n, 1.0) for n in members}
        norm = sum(sizes.values())
        w = {n: share / total * sizes[n] / norm for n in members}
        targets = FR_TARGET if fr_target is None else fr_target
        if block in targets:
            french = [n for n in members if fr_fraction(sources[n]) == 1.0]
            fixed = sum(w[n] * fr_fraction(sources[n]) for n in members if n not in french)
            mass = sum(w.values())
            fr_mass = sum(w[n] for n in french)
            want = max(targets[block] * mass - fixed, 0.0)  # French mass of FR-only sources
            if french and 0 < want < mass:
                for n in members:
                    w[n] *= want / fr_mass if n in french else (mass - want) / (mass - fr_mass)
        weights.update({n: round(v, 6) for n, v in w.items()})
    return weights


def capped_weights(weights: dict[str, float], cap: dict | None) -> dict[str, float]:
    """The weights times the repetition cap's multipliers (manifest ``epoch_cap.scale``,
    :func:`tjev.data.balance.cap_epochs`), renormalised; unchanged without a cap."""
    if not cap:
        return weights
    scaled = {n: w * cap["scale"].get(n, 1.0) for n, w in weights.items()}
    total = sum(scaled.values())
    return {n: round(w / total, 6) for n, w in scaled.items()}


def weights_of(manifest: dict, block_of: dict[str, str]) -> dict[str, float]:
    """The base weights of a built mix: mix-v7's per-source shares when the manifest has
    them (``source_shares``), else the block shares × n^0.5 rule."""
    if manifest.get("source_shares"):
        return mixture_weights_v7(manifest, block_of)
    return mixture_weights(manifest["sources"], block_of)


def write_mixture(out: Path, manifest: dict, block_of: dict[str, str]) -> dict:
    weights = capped_weights(weights_of(manifest, block_of), manifest.get("epoch_cap"))
    train_files = sorted(str((out / "train" / f"{n}.jsonl").resolve()) for n in block_of)
    mix = {
        "data": {
            "train": train_files,
            "validation": str((out / "validation.jsonl").resolve()),
            "mixture": weights,
        }
    }
    (out / "mix.yaml").write_text(yaml.safe_dump(mix, sort_keys=True, allow_unicode=True))
    manifest["mixture"] = weights
    manifest["files"] = {p: file_hash(p) for p in train_files}
    langs: Counter = Counter()
    for n in block_of:
        lang = manifest["sources"][n]["lang"]
        if isinstance(lang, dict):
            langs.update(lang)
        else:
            langs[lang] += manifest["sources"][n]["train"]
    manifest["train_rows_by_lang"] = dict(langs)
    manifest["fr_sampled_share"] = round(
        sum(w * fr_fraction(manifest["sources"][n]) for n, w in weights.items()), 4
    )
    manifest["jevbench_overlap"] = sum(
        s.get("drops", {}).get("jevbench_overlap", 0) for s in manifest["sources"].values()
    )
    if any("license_use_rows" in manifest["sources"][n] for n in block_of):
        uses: Counter = Counter()
        for n, w in weights.items():
            rows = manifest["sources"][n].get("license_use_rows") or {"unspecified": 1}
            total = sum(rows.values())
            for use, k in rows.items():
                uses[use] += w * k / total
        manifest["license_use_sampled_share"] = {u: round(v, 4) for u, v in sorted(uses.items())}
        blocks: Counter = Counter()
        for n, w in weights.items():
            blocks[block_of[n]] += w
        manifest["block_sampled_share"] = {b: round(v, 4) for b, v in sorted(blocks.items())}
    write_json(out / "manifest.json", manifest)
    if manifest.get("source_shares"):
        write_readme(out, manifest, weights)
    return manifest


def write_readme(out: Path, manifest: dict, weights: dict[str, float]) -> None:
    """``README.md`` of a mix-v7 build: per source its block, upstream dataset, rows,
    sampled share, epochs, licence and licence use (the data card of the private repo)."""
    epochs = (manifest.get("epoch_cap") or {}).get("epochs", {})
    lines = [
        f"# {out.name}",
        "",
        (
            "tjev decision mix (docs/data.md). Train rows record `license` and `license_use` "
            "(commercial / non-commercial / unspecified); held-out files of mix-v6 are "
            "byte-identical, the new sources' held-out rows are in validation-v7 / "
            "calibration-v7 / test/."
        ),
        "",
        f"- French sampled share: {manifest.get('fr_sampled_share')}",
        f"- Block shares: {manifest.get('block_sampled_share')}",
        f"- Licence use (sampled): {manifest.get('license_use_sampled_share')}",
        "",
        "| source | block | dataset | train rows | share % | epochs | licence | use |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(weights, key=lambda n: (manifest["sources"][n]["block"], n)):
        e = manifest["sources"][name]
        repo = e.get("repo") or (ALL_SOURCES[name].repo if name in ALL_SOURCES else "generated")
        uses = e.get("license_use_rows") or {}
        use = "/".join(f"{u} {k}" for u, k in sorted(uses.items())) if len(uses) > 1 else (
            next(iter(uses), e.get("license_use", "")))  # fmt: skip
        lines.append(
            f"| {name} | {e['block']} | {repo} | {e.get('train', 0)} | "
            f"{100 * weights[name]:.2f} | {epochs.get(name, '')} | "
            f"{e.get('card_licence') or e.get('license', '')} | {use} |"
        )
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ======================================================================================
# mix-v7: mix-v6 extended with new sources (``tjev data extend``; docs/data.md, mix-v7)
#
# The mix-v6 files are kept: its held-out files byte for byte (validation, calibration,
# heldout-select, validation-v6, calibration-v6, test/), its train files row for row (each
# row now records its licence and licence use). The new sources are built, their held-out
# rows go to validation-v7 / calibration-v7 / test/<source>, and their train rows go through
# the v7 post-pass (tjev.data.hygiene.clean_v7: every mix-v6 check, the evaluation-set
# 8-grams, cross-source exact / near dedup). Then the yes/no rebalance of the new sources,
# the per-source shares below, and the epoch cap for 5,000 steps under best_fit packing.

# Block shares (sum 1). New blocks: "judge" (preference / rating data, helpsteer3 moves
# there) and "breadth" (the tasksource typed-decision recasts).
BLOCKS_V7 = {"jev": 0.28, "targeted": 0.22, "judge": 0.09, "breadth": 0.09,
             "generators": 0.14, "replay": 0.10, "analysis": 0.05, "writing": 0.03}  # fmt: skip
BLOCK_OVERRIDES_V7 = {"helpsteer3": "judge"}
# French share of the blocks under the n^0.5 rule (mix-v6: replay 0.38, analysis 0.35):
# raised so that the mix stays ≥ 18% French without a translated jev block (jev_fr_mt)
FR_TARGET_V7 = {"replay": 0.45, "analysis": 0.45}
# Per-source shares (% of the sampled mix) for groups of sources, from the mix-v7 plan
# (notes/2026-10-05/research/mix-v7/plan.py). A group's share is split over its present
# members ∝ n^0.5 (× WEIGHT_SCALE); a block's listed groups are renormalised to the block
# share (absent groups give their share to the block's other groups); a block without
# listed groups (generators, writing, replay, analysis) keeps mix-v6's n^0.5 rule and
# FR_TARGET. ``jev_fr_mt`` (QE-gated translations, the gen-v7 work) is a placeholder: it
# gets 2.4% of the jev block once a source of that name is in the mix.
SOURCE_SHARES_V7: dict[str, tuple[str, float, tuple[str, ...]]] = {
    # group: (block, share %, members)
    "openjev": ("jev", 8.0, ("openjev", "openjev_x", "openjev_ctrl")),
    "decider": ("jev", 6.0, ("decider_teacher",)),
    "certo": ("jev", 5.5, ("certo", "certo_x")),
    "mghafiri": ("jev", 2.0, ("mghafiri_scenarios",)),
    "plumb": ("jev", 0.6, ("plumb",)),
    "kaivoss": ("jev", 1.5, ("kaivoss_s1",)),
    "n4ze3m": ("jev", 1.0, ("n4ze3m_td",)),
    "helmo": ("jev", 0.6, ("helmo_td",)),
    "mlagnajit": ("jev", 0.4, ("mlagnajit_td",)),
    "jev_fr_pilot": ("jev", 0.1, ("jev_fr_pilot",)),
    "jev_fr_mt": ("jev", 2.4, ("jev_fr_mt",)),
    "procedural": ("targeted", 4.0, ("procedural", "procedural_x")),
    "eikos": ("targeted", 2.2, ("eikos", "eikos_x", "eikos_long")),
    "cuad": ("targeted", 1.0, ("cuad",)),
    "tempreason": ("targeted", 2.0, ("tempreason", "tempreason_l23")),
    "tram": ("targeted", 1.2, ("tram",)),
    "temporal_nli": ("targeted", 0.6, ("temporal_nli",)),
    "drop": ("targeted", 1.0, ("drop_num",)),
    "fin_tables": ("targeted", 0.9, ("tatqa", "finqa")),
    "sharc": ("targeted", 1.2, ("sharc",)),
    "when2call": ("targeted", 0.7, ("when2call",)),
    "xlam": ("targeted", 1.0, ("xlam", "xlam_irrelevance")),
    "metatool": ("targeted", 0.5, ("metatool",)),
    "cold": ("targeted", 2.3, ("cold_law_fr", "cold_law_fr_x")),
    "argimi": ("targeted", 1.8, ("argimi_fr", "argimi_fr_x", "cetat_fr")),
    "twins": ("targeted", 1.4, ("twin_squad2", "twin_boolq", "twin_piaf_fr")),
    "fiscal": ("targeted", 0.5, ("fiscal_fr",)),
    "lexglue": ("targeted", 0.5, ("casehold", "ledgar")),
    "forecasting": ("targeted", 0.3, ("forecasting",)),
    "cxm": ("targeted", 0.4, ("cxm_fr",)),
    "helpsteer3": ("judge", 2.0, ("helpsteer3", "helpsteer3_x")),
    "oasst2": ("judge", 1.5, ("oasst2",)),
    "lmarena": ("judge", 1.5, ("lmarena_140k",)),
    "graded": ("judge", 1.0, ("webgpt", "summarize_fb")),
    "scruples": ("judge", 0.7, ("scruples",)),
    "comparia": ("judge", 2.0, ("comparia_votes", "comparia_reactions", "comparia_arena")),
    "community_alignment": ("judge", 0.9, ("community_alignment_fr",)),
    "norms": ("judge", 0.8, ("dices", "social_chem", "mic")),
    "tasksource": ("breadth", 9.0, ("tasksource_jev",)),
}
# mix-v7 per-source epoch caps (under --max-epochs): repeated passages / small noisy sets
V7_MAX_EPOCHS: dict[str, float] = {}
# the decontamination reference sets of the v7 post-pass (tjev.data.decontam)
V7_REFERENCES = ("jevbench_hf", "bfcl", "unseentimeqa", "test_of_time", "sycophancy_eval",
                 "llmbar", "ambient", "hatecheck", "hatecheck_fr", "ppe", "anli_eval",
                 "bbh_fr")  # fmt: skip
V7_HELD_SUFFIX = "-v7"
# a train row overlaps our held-out rows when it shares ≥ 25 of their 8-grams (a ~32-word
# passage) or ≥ 20% of its own 8-grams with them
HELDOUT_MIN_SHARED, HELDOUT_MIN_SHARE = 25, 0.2
# an evaluation-only set is hit by ≥ 3 shared 8-grams (~10 consecutive words) once the
# common phrases are set aside; the JevBench suites by a single one (mix-v6's rule)
EVAL_MIN_SHARED = 3
STRICT_REFERENCES = frozenset({"jevbench_hf"})
# an 8-gram in ≥ COMMON_MIN new train rows (all sources together) is a common phrase
COMMON_MIN = 50
# sources the yes/no rebalance leaves alone: their prior is the point (xlam_irrelevance is
# the "no tool applies" side of xlam, which supplies the yes side) and dices (its value
# is in the raters' soft shares; 94% argmax "no")
V7_REBALANCE_EXEMPT = frozenset({"xlam_irrelevance", "dices"})
# sources whose upstream datasets are one origin for the cross-source state dedup
_ORIGIN_PREFIXES = {"ministere-culture/comparia": "comparia", "Salesforce/xlam": "xlam",
                    "ZefanCai/Open-Jev": "openjev",
                    "MadeAgents/xlam": "xlam", "coastalcph/lex_glue": "lex_glue"}  # fmt: skip


# ======================================================================================
# mix-v7.5: mix-v7 cleaned (tjev.data.clean_v75), the noisy preference sources rebuilt with
# aggregated / smoothed votes, the v7 generators (gen_v7) and jev_fr_mt (QE-gated EN→FR MT
# of plumb, decider_teacher, mghafiri_scenarios) added (docs/data.md, mix-v7.5).
#
# Block shares: v7's, with jev raised back toward mix-v6 (v7's broader mix diluted the
# JevBench-like block: 1,500-step JevBench v1.5, v7 −5.3 intelligence vs v6), taken from
# breadth and judge; generators stay at 14%.
BLOCKS_V75 = {"jev": 0.34, "targeted": 0.22, "judge": 0.07, "breadth": 0.05,
              "generators": 0.14, "replay": 0.10, "analysis": 0.05, "writing": 0.03}  # fmt: skip
# the nine mix-v6 generators that are not writing checks (saturated: kept at a small share)
V6_GENERATORS = ("gen_base_rate", "gen_invoice_total", "gen_long_refund_policy",
                 "gen_refund_policy", "gen_return_window", "gen_schedule_conflict",
                 "gen_table_argmax", "gen_ticket_triage", "gen_weekday")  # fmt: skip
# v7 generators: (share of the generators block %, train rows). Rows are ≤ 1 epoch at the
# share over 5,000 steps plus 5% headroom (docs/data.md, "Generators v7"); weights ∝ rows
# (a group per source), so the block's sources are seen about equally often.
GEN7_PLAN: dict[str, tuple[float, int]] = {
    "gen7_return_window_pairs": (11.0, 13200),
    "gen7_weekday_pairs": (6.0, 7200),
    "gen7_schedule_pairs": (7.0, 8400),
    "gen7_invoice_budget_pairs": (7.0, 8400),
    "gen7_refund_pairs": (8.0, 9600),
    "gen7_long_refund_pairs": (3.0, 3600),
    "gen7_unknowable": (6.0, 7200),
    "gen7_aux_quantities": (3.2, 3800),
    "gen7_posterior": (10.0, 12000),
    "gen7_distractor_opinion": (5.0, 6000),
    "jev_distractor": (3.0, 3600),
    "gen7_rt_cold_fr_pairs": (7.0, 8000),
    "gen7_rt_kali_fr_pairs": (5.0, 5700),
    "gen7_rt_tau_pairs": (2.6, 3000),
    "gen7_rt_sharc_pairs": (0.3, 330),
}
V6_GENERATORS_SHARE = 15.9  # % of the generators block, split ∝ n^0.5 over the nine
GEN_TRAIN.update({n: rows for n, (_, rows) in GEN7_PLAN.items() if n != "jev_distractor"})
GEN7_NAMES = tuple(n for n in GEN7_PLAN if n != "jev_distractor")
# the preference sources rebuilt with aggregated, smoothed votes (judge.vote_target)
PREF_REBUILD = ("lmarena_140k", "comparia_arena", "comparia_votes", "comparia_reactions",
                "community_alignment_fr", "summarize_fb")  # fmt: skip
JEV_FR_MT_SHARE = 2.4  # % of the mix at mix-v7's jev scale (renormalised to the jev block)


def source_shares_v75(jev_fr_mt: float = JEV_FR_MT_SHARE) -> dict:
    """mix-v7.5 group shares: mix-v7's, the judge block re-weighted toward multi-rater
    sources (single-vote sources capped), the generators block as groups (GEN7_PLAN and
    the v6 generators), jev_fr_mt at ``jev_fr_mt``."""
    shares = {g: v for g, v in SOURCE_SHARES_V7.items() if v[0] != "judge"}
    shares["jev_fr_mt"] = ("jev", jev_fr_mt, ("jev_fr_mt",))
    shares |= {  # judge: the single-vote sources (lmarena, compar:IA, community alignment,
        # summarize) from 47% to ~38% of the block; multi-rater sources up
        "helpsteer3": ("judge", 2.3, ("helpsteer3", "helpsteer3_x")),
        "oasst2": ("judge", 1.7, ("oasst2",)),
        "lmarena": ("judge", 0.9, ("lmarena_140k",)),
        "graded": ("judge", 0.9, ("webgpt", "summarize_fb")),
        "scruples": ("judge", 0.8, ("scruples",)),
        "comparia": ("judge", 1.5, ("comparia_votes", "comparia_reactions", "comparia_arena")),
        "community_alignment": ("judge", 0.8, ("community_alignment_fr",)),
        "norms": ("judge", 0.9, ("dices", "social_chem", "mic")),
    }  # fmt: skip
    block = BLOCKS_V75["generators"] * 100 / 100  # group shares as % of the mix
    for name, (share, _) in GEN7_PLAN.items():
        shares[name] = ("generators", round(share * block, 4), (name,))
    shares["gen_v6"] = ("generators", round(V6_GENERATORS_SHARE * block, 4), V6_GENERATORS)
    return shares


SOURCE_SHARES_V75 = source_shares_v75()
# rows a post-pass must not rebalance: pairs (a subsample would break them), soft votes,
# generated / translated rows (balanced by construction or by their base source)
V75_REBALANCE_EXEMPT = V7_REBALANCE_EXEMPT | set(GEN7_PLAN) | {"jev_fr_mt"} | set(PREF_REBUILD)
# per-source epoch caps: distractor rows repeat real states (near-duplicates of their base)
V75_MAX_EPOCHS = {"jev_distractor": 1.0}


def origin_of(name: str) -> str:
    """The upstream dataset of a source (rows of one origin may share a state)."""
    if name.startswith(("gen_", "gen7_")):
        return f"generated:{name}"
    source = ALL_SOURCES.get(name)
    repo = source.repo if source is not None else f"imported:{name}"
    for prefix, origin in _ORIGIN_PREFIXES.items():
        if repo.startswith(prefix):
            return origin
    return repo


def mixture_weights_v7(manifest: dict, block_of: dict[str, str]) -> dict[str, float]:
    """mix-v7 weights: ``manifest['blocks']`` shares; in a block with listed groups
    (``manifest['source_shares']``), each present group gets its share (renormalised to
    the block), split over its members ∝ n^0.5 × WEIGHT_SCALE; other blocks follow
    :func:`mixture_weights` (n^0.5, FR_TARGET). Unlisted sources of a listed block get a
    group of their own at the block's mean group share."""
    sources, groups = manifest["sources"], manifest["source_shares"]
    blocks = {b: s for b, s in manifest["blocks"].items() if b in set(block_of.values())}
    total = sum(blocks.values())
    listed_blocks = {g[0] for g in groups.values()}
    weights: dict[str, float] = {}
    for block in sorted(set(block_of.values()) - listed_blocks):  # n^0.5 and FR_TARGET
        sub = mixture_weights(sources, {n: b for n, b in block_of.items() if b == block},
                              manifest.get("fr_target"))  # fmt: skip
        weights.update({n: w * blocks[block] / total for n, w in sub.items()})
    for block in sorted(set(block_of.values()) & listed_blocks):
        members = {n for n, b in block_of.items() if b == block}
        present: dict[str, tuple[float, list[str]]] = {}
        for g, (gb, share, names) in groups.items():
            here = [n for n in names if n in members]
            if gb == block and here:
                present[g] = (share, here)
        grouped = {n for _, ns in present.values() for n in ns}
        mean = np.mean([s for s, _ in present.values()]) if present else 1.0
        for n in sorted(members - grouped):
            present[f"_{n}"] = (float(mean), [n])
        norm = sum(s for s, _ in present.values())
        for share, names in present.values():
            sizes = {n: _weight_rows(sources[n]) ** 0.5 * WEIGHT_SCALE.get(n, 1.0) for n in names}
            z = sum(sizes.values())
            for n in names:
                weights[n] = blocks[block] / total * share / norm * sizes[n] / z
    return {n: round(v, 6) for n, v in weights.items()}


def annotate_licence(items: list, name: str) -> list:
    """Rows without a recorded licence get their source's (mix-v6 train rows)."""
    import dataclasses

    card, use = licence_of(name)[0], license_use_of(name)
    return [
        it
        if it.license_use
        else dataclasses.replace(it, license=it.license or card, license_use=use)
        for it in items
    ]


def license_rows(items) -> dict[str, int]:
    return dict(Counter(it.license_use or "unspecified" for it in items))


def filter_license_use(
    src: Path, out: Path, manifest: dict, present: dict[str, str], drop: tuple[str, ...]
) -> Path:
    """Write ``src``'s train files without the rows of the ``drop`` licence uses into
    ``out``/train; sources left empty leave ``present``. Returns ``out`` (the new train
    source for :func:`correct`)."""
    table = {}
    for name in list(present):
        items = list(read_jsonl(src / "train" / f"{name}.jsonl"))
        kept = [it for it in items if (it.license_use or license_use_of(name)) not in drop]
        entry = manifest["sources"][name]
        if len(kept) < len(items):
            table[name] = {"rows_before": len(items), "rows_after": len(kept)}
            if entry.get("weight_rows"):  # mix-v7.5: the weight basis shrinks with the rows
                entry["weight_rows"] = round(entry["weight_rows"] * len(kept) / len(items))
        entry["train"] = len(kept)
        entry["license_use_rows"] = license_rows(kept) if kept else {}
        if isinstance(entry.get("lang"), dict):
            entry["lang"] = dict(Counter(it.lang for it in kept))
        if kept:
            write_jsonl(out / "train" / f"{name}.jsonl", kept)
        else:
            del present[name]
    manifest["license_filter"] = {"dropped_uses": list(drop), "sources": table}
    return out


def _staged(out: Path, name: str, scale: float, seed: int, build_one):
    """``build_one()`` (a source's train rows, held-out parts and drops), cached next to
    ``out`` (``<out>-stage/``) under a key of the source's module code, scale and seed, so
    a re-run of the post-pass does not rebuild every source."""
    import gzip
    import inspect
    import pickle

    module = inspect.getmodule(ALL_SOURCES[name].adapter)
    code = Path(module.__file__).read_bytes() if module and module.__file__ else b""
    key = hashlib.sha256(code + f"{name}:{scale}:{seed}".encode()).hexdigest()[:16]
    path = out.with_name(out.name + "-stage") / f"{name}-{key}.pkl.gz"
    if path.exists():
        with gzip.open(path, "rb") as f:
            print(f"  (staged: {path.name})", flush=True)
            return pickle.load(f)
    result = build_one()
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path.with_suffix(".tmp"), "wb") as f:
        pickle.dump(result, f)
    path.with_suffix(".tmp").replace(path)
    return result


def _held_file_rows(path: Path) -> list:
    return list(read_jsonl(path)) if path.exists() else []


def _with_licence(items: list, licence: str, use: str) -> list:
    import dataclasses

    return [dataclasses.replace(it, license=it.license or licence,
                                license_use=it.license_use or use) for it in items]  # fmt: skip


def _drop_orphans(items: list) -> tuple[list, int]:
    """gen_v7.drop_orphans on items: a pair member whose other member was dropped goes."""
    from tjev.data.sources.gen_v7 import drop_orphans

    keep = {r["id"] for r in drop_orphans([{"id": it.id} for it in items])}
    kept = [it for it in items if it.id in keep]
    return kept, len(items) - len(kept)


def _sharc_overlap(rows: list[dict], held: list) -> set[int]:
    """Indices of gen7_rt_sharc_pairs rows with a line (the rule snippet: 40+ characters) in
    a held-out row of the ShARC adapter: the v7 sharc source evaluates on those passages."""
    states = [normalized(it.state) for it in held if it.source == "sharc"]
    out = set()
    for i, r in enumerate(rows):
        lines = [normalized(x) for x in str(r["state"]).split("\n")]
        if any(len(x) >= 40 and any(x in s for s in states) for x in lines):
            out.add(i)
    return out


def _distractor(base_rows: dict[str, list], held_parts: dict[str, list], n: int, seed: int):
    """jev_distractor: ``gen_v7.distractor.from_rows`` on the base train rows of its real
    sources (train only), and its held-out rows from those sources' held-out rows (each
    part from the same part). Rows keep their base row's licence."""
    from tjev.data.item import item_to_json
    from tjev.data.sources.gen_v7.distractor import REAL_SOURCES, from_rows

    def derive(items: list, k: int, split: str) -> list[dict]:
        by_id = {it.id: it for it in items}
        rows = from_rows([item_to_json(it) for it in items], k, seed, split=split)
        for r in rows:
            base = by_id[r["meta"]["base_id"]]
            r["license"], r["license_use"] = base.license, base.license_use
        return rows

    pool = [it for name in REAL_SOURCES for it in base_rows.get(name, [])]
    train = derive(pool, n, "train")
    parts = {
        part: derive([it for it in rows if it.source in REAL_SOURCES], HELD_OUT_PER_PART,
                     f"heldout-{part}")
        for part, rows in held_parts.items()
    }  # fmt: skip
    return train, parts


def extend(
    base: str | Path,
    out: str | Path,
    *,
    sources: Iterable[str] | None = None,
    jevbench_public: str | Path | None = None,
    tokenizer: str | Path | None = None,
    seed: int = 0,
    scale: float = 1.0,
    rebalance_noul: float | None = 0.7,
    max_loss: float = REBALANCE_MAX_LOSS,
    max_epochs: float | None = 2.0,
    preserve_blocks: bool = True,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    references: Iterable[str] = V7_REFERENCES,
    import_from: Iterable[str | Path] = (),
    replace_blocks: Iterable[str] = (),
    generators: Iterable[str] = (),
    distractor_rows: int = 0,
    replace_sources: Iterable[str] = (),
    clean_base: bool = False,
    blocks: dict | None = None,
    source_shares: dict | None = None,
    fr_target: dict | None = None,
    rebalance_exempt: Iterable[str] = V7_REBALANCE_EXEMPT,
    **run: Any,
) -> dict:
    """mix-v7: the built mix ``base`` (mix-v6) plus the new sources (default: every
    ``V7_SOURCES`` entry), into ``out``. ``import_from``: built mixes (e.g. the gen-v7
    generators, ``tjev data build --blocks generators,writing``; jev_fr_mt) whose train
    files are added as sources, their validation / calibration rows going to the v7
    held-out files and their test files copied; ``replace_blocks`` drops the base sources
    of those blocks first (e.g. ``generators,writing``). ``run``: steps, tokens_per_step,
    microbatch_tokens, packing, also_steps of the epoch-cap simulation.

    mix-v7.5 (``base`` = mix-v7, :func:`extend_v75`): ``generators`` (v7 generator names)
    are built like any generator and go through the post-pass, their pairs kept whole
    (``drop_orphans``); ``distractor_rows`` rows of ``jev_distractor`` are derived from the
    base train rows (held-out ones from held-out rows); ``replace_sources`` are base sources
    rebuilt here (their base train and v7 held-out rows are dropped); ``clean_base`` applies
    :mod:`tjev.data.clean_v75` to the base train rows and to the base's v7 held-out rows
    (the mix-v6 held-out files stay byte-identical). The base's ``validation-v7`` /
    ``calibration-v7`` rows are kept and the new held-out rows appended to them whenever
    the base has them."""
    from tjev.data import balance
    from tjev.data.decontam import RefGrams, load_references
    from tjev.data.hygiene import TrainIndex, clean_v7, signatures
    from tjev.data.item import item_to_json
    from tjev.data.near import MinHasher
    from tjev.data.sources import V7_SOURCES

    base, out = Path(base).expanduser(), Path(out).expanduser()
    if base.resolve() == out.resolve():
        raise ValueError("extend writes a new mix: OUT must differ from BASE")
    out.mkdir(parents=True, exist_ok=True)
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    replace = set(replace_blocks)
    rebuilt = set(replace_sources)
    names = list(V7_SOURCES) if sources is None else list(sources)
    unknown = [n for n in names if n not in ALL_SOURCES]
    if unknown:
        raise KeyError(f"unknown sources: {unknown}")
    generators = list(generators)
    from tjev.data.sources.gen_v7 import V7_GENERATORS, source_info

    unknown = [n for n in generators if n not in V7_GENERATORS and n not in GENERATORS]
    if unknown:
        raise KeyError(f"unknown generators: {unknown}")
    manifest: dict[str, Any] = {
        "extends": {
            "mix": str(base),
            "files": dict(base_manifest.get("files", {})),
            "held_files": {},
        },
        "blocks": blocks or BLOCKS_V7,
        "source_shares": source_shares or SOURCE_SHARES_V7,
        "fr_target": fr_target or FR_TARGET_V7,
        "seed": seed,
        "scale": scale,
        "max_prompt_tokens": max_prompt_tokens,
        "gen_scale": base_manifest.get("gen_scale", 1.0),
        "sources": {},
    }
    if rebuilt or clean_base:
        manifest["extends"]["replaced_sources"] = sorted(rebuilt)
        manifest["extends"]["clean_base"] = clean_base
    from tjev.data.clean_v75 import clean_rows

    def is_v7(name: str) -> bool:  # a source the base itself added (its held-out rows: v7)
        return base_manifest["sources"].get(name, {}).get("origin") not in (None, "base")

    # 1. the base held-out files, byte for byte (the base's v7 files: cleaned, the rebuilt
    # sources' rows dropped)
    v7_held = {p: [] for p in ("validation", "calibration")}
    clean_stats: dict[str, Counter] = {}
    for name in HELD_FILES:
        if not (base / name).exists():
            continue
        part = name.removesuffix(f"{V7_HELD_SUFFIX}.jsonl")
        if part in v7_held:  # kept, the new held-out rows appended (step 4)
            rows = [it for it in read_jsonl(base / name) if it.source not in rebuilt]
            if clean_base:
                by_source: dict[str, list] = {}
                for it in rows:
                    by_source.setdefault(it.source, []).append(it)
                rows = []
                for src, its in by_source.items():
                    block = base_manifest["sources"].get(src, {}).get("block", "")
                    fixed, stats = clean_rows(its, src, block, train=False)
                    clean_stats.setdefault(f"held:{src}", Counter()).update(stats)
                    rows += fixed
            v7_held[part] = rows
            continue
        shutil.copyfile(base / name, out / name)
        manifest["extends"]["held_files"][name] = file_hash(out / name)
    (out / "test").mkdir(exist_ok=True)
    for path in sorted((base / "test").glob("*.jsonl")):
        if path.stem in rebuilt:
            continue
        if clean_base and is_v7(path.stem):
            block = base_manifest["sources"][path.stem].get("block", "")
            fixed, stats = clean_rows(list(read_jsonl(path)), path.stem, block, train=False)
            clean_stats.setdefault(f"held:{path.stem}", Counter()).update(stats)
            write_jsonl(out / "test" / path.name, fixed)
        else:
            shutil.copyfile(path, out / "test" / path.name)
            manifest["extends"]["held_files"][f"test/{path.name}"] = file_hash(
                out / "test" / path.name
            )
    # 2. the base train files, row for row (+ licence fields; cleaned with clean_base),
    # indexed for the dedup
    hasher = MinHasher()
    train_index, seen = TrainIndex.empty(), set()
    block_of: dict[str, str] = {}
    base_rows: dict[str, list] = {}
    for name, entry in base_manifest["sources"].items():
        if entry.get("train", 0) <= 0 or name in rebuilt:
            continue
        block = BLOCK_OVERRIDES_V7.get(name, entry["block"])
        if block in replace:
            continue
        items = annotate_licence(list(read_jsonl(base / "train" / f"{name}.jsonl")), name)
        entry = dict(entry)
        if clean_base:
            before = len(items)
            items, stats = clean_rows(items, name, block, train=True)
            if stats:
                clean_stats[name] = stats
                entry["clean_v75"] = {"rows_before": before, **dict(stats)}
            entry["train"] = len(items)
            if isinstance(entry.get("lang"), dict) or any(k.startswith("fix:lang") for k in stats):
                entry["lang"] = dict(Counter(it.lang for it in items))
        write_jsonl(out / "train" / f"{name}.jsonl", items)
        clean_v7(items, None, seen, train_index, origin_of(name), hasher=hasher,
                 index_only=True)  # fmt: skip
        base_rows[name] = items
        manifest["sources"][name] = {**entry, "block": block, "origin": "base",
                                     "license_use": license_use_of(name),
                                     "license_use_rows": license_rows(items)}  # fmt: skip
        block_of[name] = block
        print(f"[base] {name}: {len(items)} rows indexed", flush=True)
    if clean_stats:
        manifest["clean_v75"] = {k: dict(v) for k, v in clean_stats.items()}
    # 3. the new sources
    jevbench = grams = None
    if jevbench_public is not None and Path(jevbench_public).exists():
        jevbench = JevBenchFilter.from_jsonl(jevbench_public)
        grams = JevBenchGrams.from_jsonl(jevbench_public)
        manifest["jevbench_filter"] = {"public": str(jevbench_public),
                                       "sha256": file_hash(jevbench_public)}  # fmt: skip
    else:
        manifest["jevbench_filter"] = "disabled: public items not found"
        print("[jevbench] WARNING: no public items, contamination filter off", flush=True)
    builder = Builder(scale=scale, seed=seed, jevbench=jevbench)
    new_train: dict[str, list] = {}
    new_held: dict[str, Parts] = {}
    for name in names:
        source = ALL_SOURCES[name]
        print(f"[v7] {name} …", flush=True)
        try:
            train, parts, drops = _staged(
                out, name, scale, seed, functools.partial(builder.public, name)
            )
        except Exception as e:  # recorded; the mix is built without it
            print(f"  FAILED {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            manifest["sources"][name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            continue
        new_train[name] = [parse_item(r) for r in train]
        new_held[name] = parts
        card, redistributable, note = licence_of(name)
        manifest["sources"][name] = {
            "block": block_of_source(name), "origin": "v7" if not rebuilt else "v7.5",
            "repo": source.repo,
            "train_built": len(train), "lang": source.lang, "license": source.license,
            "commercial_ok": source.commercial_ok, "family": source.family,
            "card_licence": card, "redistributable": redistributable, "licence_note": note,
            "license_use": license_use_of(name), "drops": dict(drops),
        }  # fmt: skip
        if name in rebuilt:
            old = base_manifest["sources"].get(name, {})
            manifest["sources"][name]["rebuilt_from"] = {"train": old.get("train"),
                                                         "drops": old.get("drops")}  # fmt: skip
        print(f"  {name}: {len(train)} train, held {[len(v) for v in parts.values()]}, "
              f"drops {dict(drops)}", flush=True)  # fmt: skip
    base_held = [*v7_held["validation"], *v7_held["calibration"]]
    for path in sorted((out / "test").glob("*.jsonl")):
        if path.stem == "sharc":
            base_held += list(read_jsonl(path))
    for name in generators:
        print(f"[gen] {name} …", flush=True)
        train, parts, drops = builder.generator(name)
        if name == "gen7_rt_sharc_pairs":
            overlap = _sharc_overlap(train, base_held)
            train = [r for i, r in enumerate(train) if i not in overlap]
            drops["sharc_heldout_snippet"] += len(overlap)
        info = source_info(name) if name in V7_GENERATORS else {
            "license": GENERATED[0], "commercial_ok": True, "family": GENERATORS[name][1],
            "redistributable": "yes", "note": GENERATED[2]}  # fmt: skip
        use = "commercial" if info["commercial_ok"] else "unspecified"
        new_train[name] = _with_licence([parse_item(r) for r in train], info["license"], use)
        new_held[name] = parts
        manifest["sources"][name] = {
            "block": block_of_source(name), "origin": "v7.5", "repo": "generated",
            "train_built": len(train), "lang": dict(Counter(r["lang"] for r in train)),
            "license": info["license"], "commercial_ok": info["commercial_ok"],
            "family": info["family"], "card_licence": info["license"],
            "redistributable": info["redistributable"], "licence_note": info["note"],
            "license_use": use, "drops": dict(drops),
        }  # fmt: skip
        print(f"  {name}: {len(train)} train, held {[len(v) for v in parts.values()]}, "
              f"drops {dict(drops)}", flush=True)  # fmt: skip
    if distractor_rows:
        held_parts = {"validation": [], "calibration": [], "test": []}
        for part in ("validation", "calibration"):
            held_parts[part] = _held_file_rows(out / f"{part}.jsonl") + v7_held[part]
        for path in sorted((out / "test").glob("*.jsonl")):
            held_parts["test"] += list(read_jsonl(path))
        train, parts = _distractor(base_rows, held_parts, distractor_rows, seed)
        new_train["jev_distractor"] = [parse_item(r) for r in train]
        new_held["jev_distractor"] = parts
        uses = license_rows(new_train["jev_distractor"])
        manifest["sources"]["jev_distractor"] = {
            "block": "generators", "origin": "v7.5", "repo": "derived (gen_v7.distractor)",
            "train_built": len(train), "lang": dict(Counter(r["lang"] for r in train)),
            "license": "its base rows' licences", "family": "adversarial",
            "card_licence": "base rows' licences", "redistributable": "as the base rows",
            "licence_note": "real rows + one generated opinion line (Apache-2.0 code)",
            "license_use": max(uses, key=uses.get) if uses else "unspecified", "drops": {},
        }  # fmt: skip
        print(f"[distractor] {len(train)} train, held {[len(v) for v in parts.values()]}",
              flush=True)  # fmt: skip
    for path in import_from:
        path = Path(path).expanduser()
        imported = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        held_rows = {p: _held_file_rows(path / f"{p}.jsonl") for p in ("validation", "calibration")}
        for name, entry in imported["sources"].items():
            if entry.get("train", 0) <= 0:
                continue
            new_train[name] = annotate_licence(list(read_jsonl(path / "train" / f"{name}.jsonl")),
                                               name)  # fmt: skip
            new_held[name] = {
                p: [item_to_json(it) for it in held_rows[p] if it.source == name]
                for p in ("validation", "calibration")
            } | {
                "test": [
                    item_to_json(it) for it in _held_file_rows(path / "test" / f"{name}.jsonl")
                ]
            }
            manifest["sources"][name] = {**entry, "origin": f"imported:{path}",
                                         "license_use": entry.get("license_use")
                                         or license_use_of(name)}  # fmt: skip
            print(f"[import] {name}: {len(new_train[name])} rows from {path}", flush=True)
    # 4. new held-out rows: none on a base train passage (the base train files stay as
    # they are, so the held-out row goes)
    base_states = set().union(*train_index.states.values()) if train_index.states else set()
    held_drops: Counter = Counter()
    for name, parts in new_held.items():
        for part, rows in parts.items():
            if not rows:
                continue
            its = [parse_item(r) for r in rows]
            near = train_index.by_state.query(signatures(its, hasher))
            keep = [r for r, it, nr in zip(rows, its, near, strict=True)
                    if not nr and normalized(it.state) not in base_states]  # fmt: skip
            held_drops[name] += len(rows) - len(keep)
            parts[part] = keep
        if name in manifest["sources"]:
            manifest["sources"][name].update({p: len(v) for p, v in parts.items()})
            manifest["sources"][name]["heldout_on_base_train"] = held_drops[name]
    manifest["heldout_on_base_train"] = dict(held_drops)
    v_rows = [r for p in new_held.values() for r in p.get("validation", [])]
    c_rows = [r for p in new_held.values() for r in p.get("calibration", [])]
    write_jsonl(out / f"validation{V7_HELD_SUFFIX}.jsonl",
                [*v7_held["validation"], *(parse_item(r) for r in v_rows)])  # fmt: skip
    write_jsonl(out / f"calibration{V7_HELD_SUFFIX}.jsonl",
                [*v7_held["calibration"], *(parse_item(r) for r in c_rows)])  # fmt: skip
    for name, parts in new_held.items():
        if parts.get("test"):
            write_jsonl(out / "test" / f"{name}.jsonl", [parse_item(r) for r in parts["test"]])
    # 5. the v7 post-pass over the new train rows
    held_items = [*v_rows, *c_rows, *(r for p in new_held.values() for r in p.get("test", []))]
    held_all = [parse_item(r) for r in held_items] + v7_held["validation"] + v7_held["calibration"]
    for name in ("validation.jsonl", "calibration.jsonl", "validation-v6.jsonl",
                 "calibration-v6.jsonl", "heldout-select.jsonl"):  # fmt: skip
        held_all += _held_file_rows(out / name)
    for path in sorted((out / "test").glob("*.jsonl")):
        if path.stem not in new_held:
            held_all += list(read_jsonl(path))
    held_all += [parse_item(r) for r in heldout_rows(seed=REPORT_SEED)]
    if jevbench_public is not None and Path(jevbench_public).exists():
        held_all += list(read_jsonl(jevbench_public))
    from tjev.data.hygiene import HeldIndex, token_counter
    from tjev.data.jevbench import item_text

    index = HeldIndex.of(held_all, hasher)
    refs = RefGrams()
    by_source: dict[str, list[str]] = {}
    for it in held_all:
        by_source.setdefault(origin_of(it.source), []).append(item_text(it))  # by origin
    # our held-out rows: a passage overlap, not one shared phrase (synthetic sources share
    # incidental 8-grams with their own held-out rows)
    refs.add_groups(
        "heldout", by_source, min_shared=HELDOUT_MIN_SHARED, min_share=HELDOUT_MIN_SHARE
    )
    failed = load_references(refs, references)
    for name in refs.grams:
        if name != "heldout" and name not in STRICT_REFERENCES:
            refs.loose[name] = (EVAL_MIN_SHARED, 0.0)
    from tjev.data.decontam import gram_hashes

    per_item = [gram_hashes(item_text(it)) for its in new_train.values() for it in its]
    allg, counts = np.unique(
        np.concatenate([g for g in per_item if len(g)] or [np.zeros(0, np.int64)]),
        return_counts=True,
    )
    common = allg[counts >= COMMON_MIN]
    del per_item, allg, counts
    print(f"[decontam] {len(common)} common 8-grams (≥ {COMMON_MIN} new train rows)", flush=True)
    manifest["decontam"] = {"references": refs.stats, "failed": failed,
                            "common_grams": len(common), "eval_min_shared": EVAL_MIN_SHARED,
                            "heldout_min": [HELDOUT_MIN_SHARED, HELDOUT_MIN_SHARE],
                            "held_rows": len(held_all)}  # fmt: skip
    count = token_counter(tokenizer) if tokenizer is not None else None
    total: Counter = Counter()
    for name, items in new_train.items():
        # distractor rows are near-duplicates of their base rows by design (one line added)
        kept, drops = clean_v7(items, index, seen, train_index, origin_of(name),
                               count_tokens=count, hasher=hasher, limit=max_prompt_tokens,
                               grams=grams, refs=refs, common=common,
                               near_train=name != "jev_distractor")  # fmt: skip
        if name.endswith("_pairs"):
            kept, orphans = _drop_orphans(kept)
            if orphans:
                drops["pair_orphan"] = orphans
        write_jsonl(out / "train" / f"{name}.jsonl", kept)
        entry = manifest["sources"][name]
        entry["train"] = len(kept)
        entry["drops"] = {**entry.get("drops", {}), **drops}
        entry["lang"] = dict(Counter(it.lang for it in kept)) or entry.get("lang")
        entry["license_use_rows"] = license_rows(kept)
        total.update(drops)
        print(f"  [hygiene-v7] {name}: {len(items)} → {len(kept)}, drops {dict(drops)}", flush=True)
        if kept:
            block_of[name] = entry.get("block") or block_of_source(name)
    manifest["hygiene"] = {"drops": dict(total), "tokenizer": str(tokenizer)}
    # 6. yes/no rebalance of the new sources, then the weights and the epoch cap
    exempt = set(rebalance_exempt)
    if rebalance_noul is not None:
        table = {}
        for name in new_train:
            if name not in block_of or name in exempt:
                continue
            path = out / "train" / f"{name}.jsonl"
            stats, kept = balance.rebalance_file(path, path, rebalance_noul, max_loss=max_loss,
                                                 seed=_seed_of(name) + seed)  # fmt: skip
            if stats["dropped"]:
                table[name] = stats
                entry = manifest["sources"][name]
                entry["train"] = stats["rows_after"]
                entry["drops"] = {**entry["drops"], "noul_rebalance": stats["dropped"]}
                entry["lang"] = dict(Counter(it.lang for it in kept))
                entry["license_use_rows"] = license_rows(kept)
        manifest["rebalance_noul"] = {"target": rebalance_noul, "max_loss": max_loss,
                                      "sources": table, "scope": "new sources",
                                      "exempt": sorted(exempt)}  # fmt: skip
    block_of = {n: b for n, b in block_of.items() if manifest["sources"][n].get("train", 0) > 0}
    manifest["held_files"] = {
        str(p.relative_to(out)): file_hash(p)
        for p in sorted([*out.glob("*.jsonl"), *(out / "test").glob("*.jsonl")])
    }
    return correct(
        out,
        out,
        manifest,
        block_of,
        max_epochs=max_epochs,
        preserve_blocks=preserve_blocks,
        tokenizer=tokenizer,
        seed=seed,
        **run,
    )


def extend_v75(
    base: str | Path,
    out: str | Path,
    *,
    fr_mt: str | Path | None = None,
    jev_fr_mt_share: float = JEV_FR_MT_SHARE,
    distractor_rows: int = GEN7_PLAN["jev_distractor"][1],
    sources: Iterable[str] = PREF_REBUILD,
    generators: Iterable[str] = GEN7_NAMES,
    **kw: Any,
) -> dict:
    """mix-v7.5 from mix-v7 (docs/data.md, mix-v7.5): the base cleaned (``clean_v75``), the
    noisy preference sources rebuilt (aggregated, smoothed votes), the v7 generators and
    jev_distractor added, jev_fr_mt imported from ``fr_mt`` (a mix folder written by
    ``scripts/fr_mt_mix.py``), the v7.5 block / group shares."""
    sources = list(sources)
    return extend(
        base, out, sources=sources, generators=generators, distractor_rows=distractor_rows,
        replace_sources=sources, clean_base=True, import_from=[fr_mt] if fr_mt else [],
        blocks=BLOCKS_V75, source_shares=source_shares_v75(jev_fr_mt_share),
        rebalance_exempt=V75_REBALANCE_EXEMPT, **kw,
    )  # fmt: skip


# mix-v7.5 epoch tiers (5,000 steps × 65,536 tokens, best_fit): tier A, the core sources,
# may repeat (≤ 2 epochs); tier B, everything else, is seen about once (≤ 1 epoch, and
# resized to its expected draws when under: every kept row seen ~once).
TIER_A_BLOCKS = frozenset({"jev", "generators", "writing"})
TIER_A_SOURCES = frozenset({
    "eikos", "eikos_x", "eikos_long", "twin_squad2", "twin_boolq", "twin_piaf_fr",
    "helpsteer3", "helpsteer3_x", "cold_law_fr", "cold_law_fr_x", "cetat_fr", "argimi_fr",
    "argimi_fr_x", "forecasting", "procedural", "cuad", "xnli_fr", "pawsx_fr", "piaf_fr",
    "tweet_sentiment_fr", "mlma_fr", "textdetox_fr",
})  # fmt: skip
# below their tier: derived rows that repeat real states (near-duplicates of their base rows)
TIER_OVERRIDES = {"jev_distractor": 1.0}
SUBSAMPLE_BELOW = 0.9  # a source under this share of min_epochs is resized to its draws


def tier_of(name: str, block: str) -> str:
    return "A" if block in TIER_A_BLOCKS or name in TIER_A_SOURCES else "B"


def tier_caps(block_of: dict[str, str], tier_a_cap: float) -> dict[str, float]:
    caps = {n: tier_a_cap if tier_of(n, b) == "A" else 1.0 for n, b in block_of.items()}
    return {n: min(c, TIER_OVERRIDES.get(n, c)) for n, c in caps.items()}


def _strata(items: list, pairs: bool) -> dict:
    """Sampling units by stratum: (type, gold label) of a row, or of a pair group's first
    row (pairs are sampled whole)."""
    from tjev.data.sources.gen_v7 import group_of

    units: dict = {}
    for i, it in enumerate(items):
        key = group_of(it.id) if pairs else i
        units.setdefault(key, []).append(i)
    strata: dict = {}
    for unit in units.values():
        it = items[unit[0]]
        strata.setdefault((it.question.type, it.question.labels[it.gold]), []).append(unit)
    return strata


def subsample(items: list, k: int, seed: int, pairs: bool = False) -> list[int]:
    """Indices (file order) of about ``k`` rows, stratified by type and gold label
    (largest remainder), seeded; pair groups kept whole."""
    strata = _strata(items, pairs)
    n_units = sum(len(v) for v in strata.values())
    per_unit = len(items) / max(n_units, 1)
    want = k / per_unit
    quota = {s: want * len(v) / n_units for s, v in strata.items()}
    take = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda s: quota[s] - take[s], reverse=True)[
        : round(want) - sum(take.values())
    ]:
        take[s] += 1
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for s in sorted(strata, key=str):
        units = strata[s]
        for j in sorted(rng.permutation(len(units))[: min(take[s], len(units))].tolist()):
            keep += units[j]
    return sorted(keep)


def min_epoch_subsample(out: Path, manifest: dict, epochs: dict[str, float], rows: dict,
                        lengths: dict, items: dict, min_epochs: float, seed: int) -> dict:  # fmt: skip
    """Sources under ``SUBSAMPLE_BELOW · min_epochs`` resized to their expected draws (rows
    × epochs / min_epochs): train file rewritten, ``lengths`` / ``rows`` updated in place,
    ``weight_rows`` keeps the mixture weight. Returns the table of resized sources."""
    from tjev.data import balance  # noqa: F401  (lengths layout)

    table = {}
    for name, e in sorted(epochs.items()):
        if e >= SUBSAMPLE_BELOW * min_epochs or rows[name] <= 1:
            continue
        path = out / "train" / f"{name}.jsonl"
        its = items.get(name) or list(read_jsonl(path))
        k = max(1, round(rows[name] * e / min_epochs))
        idx = subsample(its, k, _seed_of(name) + seed, pairs=name.endswith("_pairs"))
        kept = [its[i] for i in idx]
        write_jsonl(path, kept)
        lengths[name] = tuple(np.asarray(a)[idx] for a in lengths[name])
        entry = manifest["sources"][name]
        entry.setdefault("weight_rows", entry["train"])
        table[name] = {"rows_before": rows[name], "rows_after": len(kept),
                       "epochs_before": round(e, 3)}  # fmt: skip
        entry["train"] = rows[name] = len(kept)
        entry["drops"] = {
            **entry.get("drops", {}),
            "min_epoch_subsample": table[name]["rows_before"] - len(kept),
        }
        if isinstance(entry.get("lang"), dict):
            entry["lang"] = dict(Counter(it.lang for it in kept))
        if "license_use_rows" in entry:
            entry["license_use_rows"] = license_rows(kept)
        print(f"  [min-epoch] {name}: {e:.2f} epochs, {table[name]['rows_before']} → "
              f"{len(kept)} rows", flush=True)  # fmt: skip
    return table


def _held_items(held_dir: Path, jevbench_public) -> list:
    held = []
    for path in sorted([*held_dir.glob("*.jsonl"), *(held_dir / "test").glob("*.jsonl")]):
        held += list(read_jsonl(path))
    held += [parse_item(r) for r in heldout_rows(seed=REPORT_SEED)]
    held += [parse_item(r) for r in heldout_rows(seed=SELECT_SEED)]
    if jevbench_public is not None and Path(jevbench_public).exists():
        held += list(read_jsonl(jevbench_public))
    return held


def fresh_generator_rows(out: Path, manifest: dict, block_of: dict[str, str], base: dict,
                         lengths: dict, rows: dict, count, held_dir: Path, jevbench_public,
                         *, seed: int = 0, steps: int = 5000, tokens_per_step: int = 65536,
                         microbatch_tokens: int = 4096, packing: str = "bucket",
                         also_steps: tuple = ()) -> dict:  # fmt: skip
    """mix-v7.5-rep1: each generator expected to be drawn more often than it has rows (at
    the uncapped weights) gets new rows, the ones its seed draws after the rows it was
    built from (generators are prefix-stable), up to 1.05 × its draws. New rows go
    through the mix-v6 post-pass (held-out rows equal / near, passages, public JevBench
    8-grams, prompt length, exact duplicates); pairs stay whole. Seed-limited sources
    (tau, ShARC) return what they can and are then capped."""
    from tjev.data import balance
    from tjev.data.hygiene import HeldIndex, exact_key
    from tjev.data.sources.gen_v7 import V7_GENERATORS, source_info

    sim = balance.simulate_stream(
        lengths, balance.rounded({n: base[n] for n in block_of}), steps=steps,
        tokens_per_step=tokens_per_step, microbatch_tokens=microbatch_tokens, seed=seed,
        packing=packing,
    )  # fmt: skip
    gens = [n for n in block_of if n in GENERATORS or n in V7_GENERATORS]
    want = {n: int(np.ceil(1.05 * sim["by_source"][n]["trained"])) for n in gens}
    want = {n: w for n, w in want.items() if w > rows[n]}
    if not want:
        return {}
    index = HeldIndex.of(_held_items(held_dir, jevbench_public))
    grams = None
    if jevbench_public is not None and Path(jevbench_public).exists():
        grams = JevBenchGrams.from_jsonl(jevbench_public)
    table = {}
    for name, target in sorted(want.items()):
        entry = manifest["sources"][name]
        v7 = name in V7_GENERATORS
        scale = manifest.get("scale", 1.0) * (1.0 if v7 else manifest.get("gen_scale", 1.0))
        built = int(GEN_TRAIN.get(name, GEN_DEFAULT) * scale)
        path = out / "train" / f"{name}.jsonl"
        have = list(read_jsonl(path))
        extra = generate(name, built + 2 * (target - rows[name]), seed)[built:]
        if v7:
            info = source_info(name)
            lic, use = info["license"], "commercial" if info["commercial_ok"] else "unspecified"
        else:
            lic, use = GENERATED[0], "commercial"
        new = [parse_item({**r, "license": lic, "license_use": use}) for r in extra]
        seen = {exact_key(it) for it in have}
        kept, drops = clean(new, index, seen, count_tokens=count, limit=MAX_PROMPT_TOKENS,
                            passages=True, grams=grams)  # fmt: skip
        if name.endswith("_pairs"):
            from tjev.data.sources.gen_v7 import drop_orphans

            ok = {r["id"] for r in drop_orphans([{"id": it.id} for it in kept])}
            drops["pair_orphan"] = len(kept) - len(ok)
            kept = [it for it in kept if it.id in ok]
        kept = kept[: max(0, target - rows[name])]
        if name.endswith("_pairs"):  # the cut may split the last pair
            from tjev.data.sources.gen_v7 import drop_orphans

            ok = {r["id"] for r in drop_orphans([{"id": it.id} for it in kept])}
            kept = [it for it in kept if it.id in ok]
        if not kept:
            continue
        write_jsonl(path, [*have, *kept])
        add = balance.prompt_lengths(kept, count)
        lengths[name] = tuple(np.concatenate([np.asarray(a), np.asarray(b)])
                              for a, b in zip(lengths[name], add, strict=True))  # fmt: skip
        entry.setdefault("weight_rows", entry["train"])
        table[name] = {"rows_before": rows[name], "fresh_rows": len(kept), "draws": target,
                       "drops": dict(drops)}  # fmt: skip
        entry["train"] = rows[name] = rows[name] + len(kept)
        entry["fresh_rows"] = len(kept)
        if isinstance(entry.get("lang"), dict):
            entry["lang"] = dict(Counter(it.lang for it in [*have, *kept]))
        entry["license_use_rows"] = license_rows([*have, *kept])
        print(f"  [fresh] {name}: +{len(kept)} rows (draws {target})", flush=True)
    return table
