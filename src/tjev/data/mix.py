"""Build the decision mixture (mix-v3) as JevBench-native JSONL (``tjev data build``).

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
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tjev.data.item import parse_item, write_jsonl
from tjev.data.jevbench import JevBenchFilter
from tjev.data.render import RenderSpec
from tjev.data.sources import ALL_SOURCES, GENERATORS, JEV_SOURCES, TEXT_SOURCES, Source, generate
from tjev.utils import file_hash, write_json

# Block shares of the training mixture (docs/data.md). "jev" is the contender data, the
# largest block: the target is JevBench-style decisions. "analysis" + "writing" (text
# classification and templated writing checks) stay at 15% together. Within a block,
# sources are weighted ∝ n^0.5. Blocks absent from a build are redistributed.
BLOCKS = {"jev": 0.40, "generators": 0.25, "replay": 0.20, "analysis": 0.10, "writing": 0.05}
# French share of the sampled mix (target 20%): the jev block is English only and the
# generators draw 30% French, so the French-only sources carry these shares of their block:
# 0.4·0 + 0.25·0.30 + 0.2·0.38 + 0.1·0.35 + 0.05·0.34 ≈ 0.20. Only the weights move.
FR_TARGET = {"replay": 0.38, "analysis": 0.35}
WRITING = frozenset({"gen_spelling_error", "gen_register"})
GEN_TRAIN = {"gen_long_refund_policy": 2000}  # rows per generator (default GEN_DEFAULT)
GEN_DEFAULT = 4000
HELD_OUT_PER_PART = 300
HELD_OUT_GROUP_SHARE = 0.1  # sources without an eval split: share of groups held out
MAX_OPTIONS = RenderSpec().max_options  # answer letters A..Z
PARTS = ("validation", "calibration", "test")

Row = dict[str, Any]
Parts = dict[str, list[Row]]


def block_of_source(name: str) -> str:
    if name in JEV_SOURCES:
        return "jev"
    if name in TEXT_SOURCES:
        return "analysis"
    if name in ALL_SOURCES:
        return "replay"
    return "writing" if name in WRITING else "generators"


def normalized(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


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
    """Dataset rows → valid item rows (ids, source, family, language set), and drop counts."""
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
    """How to build one source: size scale, seed, and the contamination filter (or None)."""

    scale: float = 1.0
    seed: int = 0
    jevbench: JevBenchFilter | None = None

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
        if source.eval_split:
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
        n = int(GEN_TRAIN.get(name, GEN_DEFAULT) * self.scale)
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


def select_sources(sources: set[str], blocks: set[str]) -> list[str]:
    """Source and generator names to build (all by default), in build order."""
    names = [*ALL_SOURCES, *GENERATORS]
    unknown = (sources - set(names)) | (blocks - set(BLOCKS))
    if unknown:
        raise KeyError(f"unknown sources/blocks: {sorted(unknown)}")
    return [
        n
        for n in names
        if (not sources or n in sources) and (not blocks or block_of_source(n) in blocks)
    ]


def build(
    out: str | Path,
    *,
    sources: Iterable[str] = (),
    blocks: Iterable[str] = (),
    commercial_only: bool = False,
    scale: float = 1.0,
    seed: int = 0,
    jevbench_public: str | Path | None = None,
) -> dict:
    """Build the mix into ``out``; returns the manifest. ``sources`` / ``blocks`` build a
    subset (the block shares are then renormalised over the blocks present)."""
    out = Path(out)
    manifest: dict[str, Any] = {"sources": {}, "blocks": BLOCKS, "seed": seed, "scale": scale}
    jevbench = None
    if jevbench_public is not None and Path(jevbench_public).exists():
        jevbench = JevBenchFilter.from_jsonl(jevbench_public)
        manifest["jevbench_filter"] = {
            "public": str(jevbench_public),
            "sha256": file_hash(jevbench_public),
        }
        print(f"[jevbench] filtering against {len(jevbench.states)} public states", flush=True)
    else:
        manifest["jevbench_filter"] = "disabled: public items not found (tjev data jevbench)"
        print("[jevbench] WARNING: no public items, contamination filter off", flush=True)
    builder = Builder(scale=scale, seed=seed, jevbench=jevbench)
    held: dict[str, Any] = {"validation": [], "calibration": [], "test": {}}
    block_of: dict[str, str] = {}
    for name in select_sources(set(sources), set(blocks)):
        if name in GENERATORS:
            train, parts, drops = builder.generator(name)
            langs: Any = dict(Counter(r["lang"] for r in train))
            info = {"license": "generated (Apache-2.0)", "commercial_ok": True,
                    "family": GENERATORS[name][1]}  # fmt: skip
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
        held["validation"] += parts["validation"]
        held["calibration"] += parts["calibration"]
        held["test"][name] = parts["test"]
        block_of[name] = block_of_source(name)
        manifest["sources"][name] = {
            "block": block_of[name],
            "train": len(train),
            **{k: len(v) for k, v in parts.items()},
            "lang": langs,
            **info,
            "drops": dict(drops),
        }
        print(f"  {name}: {len(train)} train, drops {dict(drops)}", flush=True)
    for part in ("validation", "calibration"):
        write_jsonl(out / f"{part}.jsonl", [parse_item(r) for r in held[part]])
    for name, rows in held["test"].items():
        write_jsonl(out / "test" / f"{name}.jsonl", [parse_item(r) for r in rows])
    block_of = {n: b for n, b in block_of.items() if manifest["sources"][n]["train"] > 0}
    return write_mixture(out, manifest, block_of)


def reweight(out: str | Path) -> dict:
    """Recompute ``mix.yaml`` (and the manifest's weights) of a built mix."""
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    present = {n: s for n, s in manifest["sources"].items() if s.get("train", 0) > 0}
    return write_mixture(out, manifest, {n: s["block"] for n, s in present.items()})


def fr_fraction(entry: dict) -> float:
    lang = entry["lang"]
    return lang.get("fr", 0) / sum(lang.values()) if isinstance(lang, dict) else float(lang == "fr")


def mixture_weights(sources: dict, block_of: dict[str, str]) -> dict[str, float]:
    """Block share (absent blocks redistributed) × n^0.5 within the block; then, in the
    blocks of FR_TARGET, French-only sources are rescaled (and the rest shrunk) so that the
    block is that share French. Block shares never change."""
    shares = {b: s for b, s in BLOCKS.items() if b in set(block_of.values())}
    total = sum(shares.values())
    weights = {}
    for block, share in shares.items():
        members = [n for n, b in block_of.items() if b == block]
        sizes = {n: sources[n]["train"] ** 0.5 for n in members}
        norm = sum(sizes.values())
        w = {n: share / total * sizes[n] / norm for n in members}
        if block in FR_TARGET:
            french = [n for n in members if fr_fraction(sources[n]) == 1.0]
            fixed = sum(w[n] * fr_fraction(sources[n]) for n in members if n not in french)
            mass = sum(w.values())
            fr_mass = sum(w[n] for n in french)
            want = max(FR_TARGET[block] * mass - fixed, 0.0)  # French mass of FR-only sources
            if french and 0 < want < mass:
                for n in members:
                    w[n] *= want / fr_mass if n in french else (mass - want) / (mass - fr_mass)
        weights.update({n: round(v, 6) for n, v in w.items()})
    return weights


def write_mixture(out: Path, manifest: dict, block_of: dict[str, str]) -> dict:
    weights = mixture_weights(manifest["sources"], block_of)
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
    write_json(out / "manifest.json", manifest)
    return manifest
