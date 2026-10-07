"""Data-quality report of a built mix (mix-v3): mix, duplication, leakage, lengths, flags.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 nice -n 19 python scripts/data_report.py \\
        --mix ~/tjev-work/data/mix-v3 --jevbench ~/tjev-work/data/jevbench/public.jsonl \\
        --out reports/data/mix-v3 [--sample N]

Writes ``OUT/report.json`` (every number), ``OUT/report.md`` (tables and ranked
recommendations) and a few PNG plots (``--no-plots`` skips them; they need matplotlib, and
``--from-json`` re-derives the recommendations, report.md and plots from an existing
report.json without re-reading the mix). Nothing in the mix is modified.

What it measures, per source and overall:

1. **Mix.** Raw rows, rendered prompt tokens (the real chat template and tokenizer, both
   layouts, every option shown), the sampled share (``mix.yaml`` weights) against the raw
   share, and the *effective epochs* of each source in a run of ``--steps`` steps of
   ``--tokens-per-step`` tokens. Those come from a simulation of the real packer
   (``tjev.data.pack.BucketPacker``, multi-bin first-fit) over a stream drawn like the
   training stream (source by weight, item uniform, layout 50/50), using each row's
   measured length; distractor sub-sampling is ignored (prompts are slightly shorter in
   training, so the epochs are a small underestimate). Language and type splits, option
   counts, and label balance: the gold letter under the training render (one draw of the
   real ``choose_order``) against chance, the stored gold position, noul yes-share, score
   levels, soft targets and their normalised entropy.
2. **Duplication.** Exact duplicates in train (normalised state + instructions + options +
   target), equal inputs with different targets (conflicting labels; the builder drops
   these within a source only), equal (state, instructions) over different menus, and
   near-duplicate states (MinHash, 128 permutations over word 3-gram shingles, LSH 32 bands
   × 4 rows, pairs kept at an estimated Jaccard ≥ 0.8). Leakage of train into validation,
   calibration, heldout-select, test and the JevBench public items: exact (normalised
   state + instructions, and state alone) and near (same MinHash test).
3. **Length.** Token-length histograms and p50 / p90 / p99 / max, bucket assignment, rows
   over the largest bucket (the packer drops them: ``BucketPacker.place`` counts
   ``dropped_too_long``), packing fill per bucket from the simulation, very short states.
4. **Quality flags.** Malformed rows (``tjev.data.item.parse_item`` is the reference, plus
   raw checks: missing fields, NaN, targets not summing to 1, duplicate ids), language
   mismatch (a stop-word heuristic on the state and on the instructions), template
   artefacts (unrendered placeholders, HTML tags / entities, mojibake, U+FFFD, control
   characters), extreme length outliers. At most ``EXAMPLES`` examples per flag.
5. **Recommendations**, ranked, each with its expected effect and what it touches (the
   mixture weights are part of the run config identity; the train files' hashes are part
   of the data identity; rendering changes bump ``TEMPLATE_VERSION``).

Light on the host: one process, streaming JSONL, the tokenizer on ``RAYON_NUM_THREADS``
(2 by default), hashes for every exact test. ``--sample N`` reads the first N rows of each
train file (the builder writes them in a random order) for a quick look.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # tjev.data.pack imports jax
os.environ.setdefault("RAYON_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tjev.data.balance import simulate_stream  # the real BucketPacker on lengths
from tjev.data.item import TYPES, Item, parse_item
from tjev.data.lang import guess_lang
from tjev.data.near import (
    BANDS,
    NEAR_THRESHOLD,
    NUM_PERM,
    SHINGLE,
    MinHasher,
    _band_pairs,
    band_keys,
    estimated_jaccard,
    near_matches,
    normalized,
    shingle_hashes,
    stable_hash,
)
from tjev.data.pack import bucket_for
from tjev.data.render import LETTERS, RenderSpec, choose_order, render

STEPS = 3000
TOKENS_PER_STEP = 65536
MICROBATCH_TOKENS = 4096  # 4B on one v6e chip (tjev.campaign.plan); accumulation 16
BUCKETS = (1024, 2048, 4096)
MAX_SEGMENTS = 32
PACKING_BINS = 16
MAX_LABELS = 26
EXAMPLES = 20
SHORT_STATE_WORDS = 3
HIGH_EPOCHS, LOW_EPOCHS = 4.0, 0.1
HELD_SPLITS = ("validation", "calibration", "heldout-select", "test", "jevbench")
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


# --------------------------------------------------------------------------------------
# Hashing and near-duplicates


def near_duplicate_clusters(sigs: np.ndarray, threshold: float = NEAR_THRESHOLD) -> np.ndarray:
    """Cluster label per signature: connected components of the LSH pairs whose estimated
    Jaccard is ≥ threshold (each candidate is checked against its group's first member)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(sigs)
    if n == 0:
        return np.zeros(0, np.int64)
    src, dst = [], []
    for first, member in _band_pairs(band_keys(sigs)):
        ok = estimated_jaccard(sigs, first, member) >= threshold
        src.append(first[ok])
        dst.append(member[ok])
    i = np.concatenate(src) if src else np.zeros(0, np.int64)
    j = np.concatenate(dst) if dst else np.zeros(0, np.int64)
    graph = coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def group_counts(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: how many rows share its key, and how many distinct values its key has."""
    if len(keys) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    _, inv, size = np.unique(keys, return_inverse=True, return_counts=True)
    pairs = np.unique(np.stack([inv, np.searchsorted(np.unique(values), values)], 1), axis=0)
    distinct = np.bincount(pairs[:, 0], minlength=len(size))
    return size[inv], distinct[inv]


# --------------------------------------------------------------------------------------
# Text heuristics

ARTEFACTS = {
    "placeholder": re.compile(
        r"\{\{[^}]*\}\}|\$\{[^}]*\}|\{[A-Za-z_][A-Za-z0-9_]{0,30}\}|<(?:PLACEHOLDER|MASK|unk)>"
    ),
    "html_tag": re.compile(
        r"</?(?:br|p|div|span|a|b|i|u|li|ul|ol|table|tr|td|img|strong|em|h[1-6])\b[^>]{0,80}>",
        re.IGNORECASE,
    ),
    "html_entity": re.compile(
        r"&(?:amp|lt|gt|quot|apos|nbsp|#\d{2,5}|#x[0-9a-f]{2,4});", re.IGNORECASE
    ),
    "mojibake": re.compile(r"Ã[\u0080-¿©¨ª«®¯§¢]|â€[™œ\u009d˜”“¦]|Â[ -¿]"),
    "replacement_char": re.compile("�"),
    "control_char": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
    "python_repr": re.compile(r"\{'[A-Za-z_]+': (?:None|'|\[|\{|\d)"),
    "none_literal": re.compile(r"^(?:None|nan|NaN|null)$|: (?:None|nan)\b"),
}


# (severity, advice) of each artefact flag in the recommendations
ARTEFACT_ADVICE = {
    "python_repr": (
        6.0,
        (
            "Options (criteria) rendered as Python dict reprs such as "
            "`{'level': …, 'definition': …}` or `{'value': None}`: the adapter passes "
            "structured criteria through and parse_item str()s them. Map them to text "
            "(e.g. `level: definition`) in the adapter."
        ),
    ),
    "nonstring_criterion": (
        5.0,
        (
            "Noul criteria that are dicts: parse_item keeps them unconverted "
            "and the plain layout prints their repr. Same fix as python_repr."
        ),
    ),
    "html_entity": (3.0, "Unescape (`html.unescape`) in the adapters."),
    "replacement_char": (2.5, "Encoding damage in the source data (U+FFFD); drop or leave."),
    "mojibake": (2.0, "Double-encoded UTF-8; `ftfy`-style repair or drop."),
    "html_tag": (
        1.5,
        (
            "Mostly legitimate (questions about HTML, code); a few scraped tables "
            "(dbpedia). Review before stripping."
        ),
    ),
    "placeholder": (
        1.5,
        "Mostly legitimate shell / template syntax (`{{.ID}}`, `${VAR}`); review, no blanket fix.",
    ),
    "none_literal": (
        2.0,
        (
            "Literal None / nan in text: part are python reprs (see python_repr), "
            "part legitimate (`Witnesses: None`)."
        ),
    ),
}


def artefact_flags(text: str) -> list[str]:
    return [name for name, pattern in ARTEFACTS.items() if pattern.search(text)]


def raw_problems(raw: dict) -> list[str]:
    """Raw-row checks beyond parse_item (which is fail-closed and normalises targets)."""
    out = [
        f"missing:{k}"
        for k in ("id", "state", "question", "source", "lang", "family")
        if k not in raw
    ]
    target = raw.get("target") or raw.get("gold_probs")
    if target is not None:
        values = [float(v) for v in target.values()]
        if any(not math.isfinite(v) for v in values):
            out.append("target_nan")
        elif abs(sum(values) - 1.0) > 1e-3:
            out.append("target_sum")
    elif "expected" not in raw:
        out.append("missing:target")
    return out


# --------------------------------------------------------------------------------------
# Rows → columns


@dataclass
class Columns:
    """Per-row features of one split (parallel lists; ``arrays()`` freezes them)."""

    split: str
    names: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    sigs: list[np.ndarray] = field(default_factory=list)

    def add(self, **values: Any) -> None:
        for k, v in values.items():
            self.names[k].append(v)

    def __len__(self) -> int:
        return len(self.names["source"])

    def arrays(self) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v) for k, v in self.names.items()}
        for k in ("h_full", "h_input", "h_si", "h_state", "h_target", "h_instr", "h_gold"):
            if k in out:
                out[k] = out[k].astype(np.uint64)
        out["sig"] = np.stack(self.sigs) if self.sigs else np.zeros((0, NUM_PERM), np.uint32)
        return out


def item_keys(item: Item) -> dict[str, int]:
    q = item.question
    state, instr = normalized(item.state), normalized(q.instructions)
    menu = json.dumps([q.labels, q.descriptions], ensure_ascii=False)
    target = json.dumps([round(p, 4) for p in item.target])
    return {
        "h_state": stable_hash(state),
        "h_si": stable_hash(state, instr),
        "h_input": stable_hash(state, instr, q.type, menu),
        "h_target": stable_hash(target),
        "h_full": stable_hash(state, instr, q.type, menu, target),
        "h_instr": stable_hash(instr),
        "h_gold": stable_hash(q.type, q.labels[item.gold]),
    }


def normalised_entropy(target: tuple[float, ...]) -> float:
    h = -sum(p * math.log(p) for p in target if p > 0)
    return h / math.log(len(target))


class Flags:
    """Counts per (flag, source) and ≤ EXAMPLES examples per flag."""

    def __init__(self) -> None:
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.examples: dict[str, list[dict]] = defaultdict(list)

    def add(self, flag: str, source: str, row_id: str, text: str = "", **extra: Any) -> None:
        self.counts[flag][source] += 1
        if len(self.examples[flag]) < EXAMPLES:
            self.examples[flag].append(
                {"source": source, "id": row_id, "text": text[:240], **extra}
            )

    def to_json(self) -> dict:
        return {
            flag: {
                "total": sum(c.values()),
                "by_source": dict(c.most_common()),
                "examples": self.examples[flag],
            }
            for flag, c in sorted(self.counts.items())
        }


def iter_jsonl(path: Path, limit: int | None = None) -> Iterable[tuple[int, dict | None, str]]:
    with path.open(encoding="utf-8") as f:
        n = 0
        for line in f:
            if not line.strip():
                continue
            if limit is not None and n >= limit:
                return
            n += 1
            try:
                yield n, json.loads(line), line
            except json.JSONDecodeError:
                yield n, None, line


class Scanner:
    """Streams JSONL files into :class:`Columns`, tokenizing train rows in batches."""

    def __init__(self, encode_batch: Callable[[list[str]], list[int]] | None, seed: int = 0):
        self.encode_batch = encode_batch
        self.minhash = MinHasher()
        self.flags = Flags()
        self.seed = seed
        self.eval_plain = RenderSpec(train=False, layout="plain")
        self.eval_json = RenderSpec(train=False, layout="json")
        self.train_spec = RenderSpec()
        self.ids: Counter = Counter()

    def scan(self, cols: Columns, path: Path, limit: int | None = None) -> None:
        train = cols.split == "train"
        pending: list[tuple[int, str, str]] = []
        for n, raw, line in iter_jsonl(path, limit):
            where = f"{path.name}:{n}"
            if raw is None:
                self.flags.add("malformed:json", path.stem, where, line)
                continue
            source = str(raw.get("source", path.stem))
            for problem in raw_problems(raw):
                self.flags.add(f"malformed:{problem}", source, str(raw.get("id", where)))
            try:
                item = parse_item(raw, where)
            except (KeyError, ValueError, TypeError) as e:
                self.flags.add("malformed:parse", source, where, f"{type(e).__name__}: {e}")
                continue
            if train:
                self.ids[item.id] += 1
                if self.ids[item.id] == 2:
                    self.flags.add("malformed:duplicate_id", source, item.id)
            row = len(cols)
            self.features(cols, item, train)
            if train and self.encode_batch is not None:
                pending.append(
                    (row, render(item, self.eval_plain)[0], render(item, self.eval_json)[0])
                )
                if len(pending) >= 512:
                    self.flush(cols, pending)
        if pending:
            self.flush(cols, pending)

    def flush(self, cols: Columns, pending: list[tuple[int, str, str]]) -> None:
        texts = [t for _, plain, js in pending for t in (plain, js)]
        lengths = self.encode_batch(texts)
        for k, (row, _, _) in enumerate(pending):
            cols.names["tokens_plain"][row] = lengths[2 * k]
            cols.names["tokens_json"][row] = lengths[2 * k + 1]
        pending.clear()

    def features(self, cols: Columns, item: Item, train: bool) -> None:
        q = item.question
        soft = max(item.target) < 1.0
        state_words = len(normalized(item.state).split())
        values: dict[str, Any] = {
            "source": item.source,
            "lang": item.lang,
            "type": q.type,
            "n_options": len(q.labels),
            "gold_pos": item.gold,
            "gold_label": f"{q.type}:{q.labels[item.gold]}",
            "soft": soft,
            "entropy": normalised_entropy(item.target) if soft else 0.0,
            "state_words": state_words,
            "state_chars": len(item.state),
            "eval_letter": choose_order(item, self.eval_plain, None)[0].index(item.gold),
            **item_keys(item),
        }
        if train:
            rng = np.random.default_rng([self.seed, len(cols)])
            order, _ = choose_order(item, self.train_spec, rng)
            values |= {
                "train_letter": order.index(item.gold),
                "train_shown": len(order),
                "tokens_plain": -1,
                "tokens_json": -1,
                "yes": q.type == "noul" and item.gold == 1,
            }
            self.quality(item, state_words)
        cols.add(**values)
        cols.sigs.append(self.minhash.signature(shingle_hashes(item.state)))

    def quality(self, item: Item, state_words: int) -> None:
        q, src = item.question, item.source
        if state_words < SHORT_STATE_WORDS:
            self.flags.add("short_state", src, item.id, item.state)
        for part, text in (("state", item.state), ("instructions", q.instructions)):
            guess = guess_lang(text)
            if guess is not None and guess != item.lang:
                self.flags.add(f"lang_mismatch:{part}", src, item.id, text, lang=item.lang)
        bad = [d for d in q.descriptions if not isinstance(d, str)]
        if bad:  # noul criteria pass through parse_item unconverted: rendered as a repr
            self.flags.add("artefact:nonstring_criterion", src, item.id, repr(bad[0]))
        text = "\n".join([item.state, q.instructions, *map(str, q.descriptions)])
        for flag in artefact_flags(text):
            self.flags.add(f"artefact:{flag}", src, item.id, _around(text, ARTEFACTS[flag]))


def _around(text: str, pattern: re.Pattern) -> str:
    m = pattern.search(text)
    start = max(0, m.start() - 80) if m else 0
    return text[start : start + 240]


# --------------------------------------------------------------------------------------
# Analyses


def percentiles(x: np.ndarray) -> dict[str, float]:
    if len(x) == 0:
        return {}
    p = np.percentile(x, [50, 90, 99])
    return {
        "mean": round(float(x.mean()), 1),
        "p50": float(p[0]),
        "p90": float(p[1]),
        "p99": float(p[2]),
        "max": int(x.max()),
    }


LENGTH_EDGES = np.unique(np.round(np.geomspace(16, 16384, 49)).astype(int))


def bucket_shares(lengths: np.ndarray, buckets: tuple[int, ...]) -> dict[str, float]:
    if len(lengths) == 0:
        return {}
    out = Counter(str(bucket_for(int(n), buckets) or f">{max(buckets)}") for n in lengths)
    return {k: round(v / len(lengths), 5) for k, v in sorted(out.items())}


def label_balance(t: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    """Gold letter under the training render vs chance (E[1/shown]); stored gold position
    vs chance (E[1/n]); noul yes-share; score levels; soft targets."""
    out: dict[str, Any] = {}
    cs = mask & (t["type"] != "score")
    if cs.any():
        shown = t["train_shown"][cs]
        observed = np.bincount(t["train_letter"][cs], minlength=26)[:26] / cs.sum()
        expected = (
            np.array([(shown > k).astype(float) @ (1.0 / shown) for k in range(26)]) / cs.sum()
        )
        stored = np.bincount(t["gold_pos"][cs], minlength=26)[:26] / cs.sum()
        n = t["n_options"][cs]
        exp_stored = np.array([(n > k).astype(float) @ (1.0 / n) for k in range(26)]) / cs.sum()
        last = int(max(shown.max(), n.max()))
        out["choice_noul"] = {
            "rows": int(cs.sum()),
            "train_letter": [round(float(v), 4) for v in observed[:last]],
            "train_letter_chance": [round(float(v), 4) for v in expected[:last]],
            "train_letter_tvd": round(float(0.5 * np.abs(observed - expected).sum()), 4),
            "train_letter_A_ratio": round(float(observed[0] / max(expected[0], 1e-9)), 3),
            "stored_position": [round(float(v), 4) for v in stored[:last]],
            "stored_position_chance": [round(float(v), 4) for v in exp_stored[:last]],
            "stored_A_ratio": round(float(stored[0] / max(exp_stored[0], 1e-9)), 3),
            "stored_tvd": round(float(0.5 * np.abs(stored - exp_stored).sum()), 4),
        }
    noul = mask & (t["type"] == "noul")
    if noul.any():
        out["noul_yes_share"] = round(float(t["yes"][noul].mean()), 4)
        out["noul_rows"] = int(noul.sum())
    score = mask & (t["type"] == "score")
    if score.any():
        levels = np.bincount(t["gold_pos"][score])
        out["score_levels"] = [round(float(v), 4) for v in levels / levels.sum()]
        out["score_rows"] = int(score.sum())
    hard = mask & ~t["soft"]
    if hard.any():
        label, count = Counter(t["gold_label"][hard].tolist()).most_common(1)[0]
        out["majority_label"] = label
        out["majority_share"] = round(count / int(hard.sum()), 4)
        per_type = {}
        for kind in TYPES:
            m = hard & (t["type"] == kind)
            if m.any():
                lab, c = Counter(t["gold_label"][m].tolist()).most_common(1)[0]
                per_type[kind] = {
                    "label": lab,
                    "share": round(c / int(m.sum()), 4),
                    "rows": int(m.sum()),
                }
        out["majority_by_type"] = per_type
    soft = mask & t["soft"]
    out["soft_rows"] = int(soft.sum())
    out["soft_share"] = round(float(soft.sum() / max(mask.sum(), 1)), 4)
    if soft.any():
        out["soft_entropy_mean"] = round(float(t["entropy"][soft].mean()), 4)
    return out


def duplication(train: dict[str, np.ndarray]) -> dict:
    """Exact, conflicting and near duplicates in train, overall and per source."""
    src = train["source"]
    out: dict[str, Any] = {}
    size_full, _ = group_counts(train["h_full"], train["h_full"])
    _, unique_idx = np.unique(train["h_full"], return_index=True)
    extra = np.ones(len(src), bool)
    extra[unique_idx] = False  # rows beyond the first of each exact-duplicate group
    _, cross_full = group_counts(train["h_full"], src)
    out["exact_duplicate_rows"] = int(extra.sum())
    out["exact_duplicate_groups"] = int(np.unique(train["h_full"][size_full > 1]).size)
    out["exact_duplicate_cross_source_rows"] = int((extra & (cross_full > 1)).sum())
    _, distinct_t = group_counts(train["h_input"], train["h_target"])
    conflict = distinct_t > 1
    _, cross_in = group_counts(train["h_input"], src)
    out["conflicting_rows"] = int(conflict.sum())
    out["conflicting_groups"] = int(np.unique(train["h_input"][conflict]).size)
    out["conflicting_cross_source_rows"] = int((conflict & (cross_in > 1)).sum())
    _, distinct_menu = group_counts(train["h_si"], train["h_input"])
    out["same_question_other_menu_rows"] = int((distinct_menu > 1).sum())
    size_state, _ = group_counts(train["h_state"], train["h_state"])
    _, cross_state = group_counts(train["h_state"], src)
    out["shared_state_rows"] = int((size_state > 1).sum())
    out["shared_state_cross_source_rows"] = int((cross_state > 1).sum())
    pairs = Counter()
    if (cross_state > 1).any():
        key_sources: dict[int, set] = defaultdict(set)
        for k, s in zip(
            train["h_state"][cross_state > 1].tolist(), src[cross_state > 1].tolist(), strict=True
        ):
            key_sources[k].add(s)
        for ss in key_sources.values():
            pairs[" + ".join(sorted(ss))] += 1
    out["shared_state_cross_source_pairs"] = dict(pairs.most_common(10))
    # near-duplicates on distinct normalised states
    _, first = np.unique(train["h_state"], return_index=True)
    first = np.sort(first)
    labels = near_duplicate_clusters(train["sig"][first])
    cluster_size = np.bincount(labels)[labels]
    _, lead = np.unique(labels, return_index=True)
    redundant = np.ones(len(first), bool)
    redundant[lead] = False
    _, cross_near = group_counts(labels, src[first])
    out["distinct_states"] = len(first)
    out["near_duplicate_clusters"] = int((np.bincount(labels) > 1).sum())
    out["near_duplicate_states"] = int((cluster_size > 1).sum())
    out["near_redundant_states"] = int(redundant.sum())
    out["near_cross_source_states"] = int((cross_near > 1).sum())
    per: dict[str, dict] = {}
    for name in sorted(set(src.tolist())):
        m = src == name
        mf = src[first] == name
        n_first = int(mf.sum())
        # within-source near redundancy: states beyond the first per (cluster, source)
        sub = labels[mf]
        per[name] = {
            "rows": int(m.sum()),
            "exact_duplicate_rows": int((extra & m).sum()),
            "conflicting_rows": int((conflict & m).sum()),
            "shared_state_rows": int((size_state[m] > 1).sum()),
            "distinct_states": n_first,
            "near_redundant_states": int(n_first - np.unique(sub).size),
            "near_redundant_share": round(
                float((n_first - np.unique(sub).size) / max(n_first, 1)), 4
            ),
        }
    out["by_source"] = per
    return out


def leakage(train: dict[str, np.ndarray], held: dict[str, dict[str, np.ndarray]]) -> dict:
    """Exact (state + instructions; state alone) and near (MinHash) overlap of train with
    every held-out split, counted on both sides."""
    out: dict[str, Any] = {}
    _, first = np.unique(train["h_state"], return_index=True)
    index_sigs = train["sig"][first]
    index_src = train["source"][first]
    train_pairs = set(zip(train["h_si"].tolist(), train["source"].tolist(), strict=True))
    for split, h in held.items():
        if len(h.get("source", ())) == 0:
            continue
        si = np.isin(h["h_si"], train["h_si"])
        st = np.isin(h["h_state"], train["h_state"])
        near = near_matches(h["sig"], index_sigs)
        nm = near >= 0
        lead = first[np.maximum(near, 0)]
        near_item = (
            nm & (train["h_instr"][lead] == h["h_instr"]) & (train["h_gold"][lead] == h["h_gold"])
        )
        train_si = np.isin(train["h_si"], h["h_si"])
        train_state = np.isin(train["h_state"], h["h_state"])
        cross = Counter()
        for i in np.flatnonzero(nm):
            cross[f"{index_src[near[i]]} → {h['source'][i]}"] += 1
        same_src_exact = sum(
            (k, s) in train_pairs
            for k, s in zip(h["h_si"][si].tolist(), h["source"][si].tolist(), strict=True)
        )
        out[split] = {
            "rows": len(h["source"]),
            "exact_state_instructions": int(si.sum()),
            "exact_state_instructions_same_source": same_src_exact,
            "exact_state": int(st.sum()),
            "near_state": int(nm.sum()),
            "near_state_share": round(float(nm.mean()), 4),
            "near_item": int(near_item.sum()),
            "near_item_share": round(float(near_item.mean()), 4),
            "train_rows_exact_state_instructions": int(train_si.sum()),
            "train_rows_exact_state": int(train_state.sum()),
            "near_by_pair": dict(cross.most_common(15)),
            "by_heldout_source": {
                s: {
                    "rows": int((h["source"] == s).sum()),
                    "exact_si": int((si & (h["source"] == s)).sum()),
                    "exact_state": int((st & (h["source"] == s)).sum()),
                    "near": int((nm & (h["source"] == s)).sum()),
                    "near_item": int((near_item & (h["source"] == s)).sum()),
                }
                for s in sorted(set(h["source"].tolist()))
                if (si | st | nm)[h["source"] == s].any()
            },
        }
    return out


# --------------------------------------------------------------------------------------
# The report


def read_mixture(mix: Path) -> tuple[dict[str, float], dict[str, str]]:
    weights = yaml.safe_load((mix / "mix.yaml").read_text())["data"]["mixture"]
    blocks = {}
    manifest = mix / "manifest.json"
    if manifest.exists():
        for name, entry in json.loads(manifest.read_text())["sources"].items():
            blocks[name] = entry.get("block", "?")
    return weights, blocks


def count_rows(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


def held_files(mix: Path, jevbench: Path | None) -> dict[str, list[Path]]:
    files = {
        # mix-v6: the targeted block's held-out rows are in their own files
        # mix-v7: the new sources' held-out rows are in validation-v7 / calibration-v7
        "validation": [
            mix / "validation.jsonl",
            mix / "validation-v6.jsonl",
            mix / "validation-v7.jsonl",
        ],
        "calibration": [
            mix / "calibration.jsonl",
            mix / "calibration-v6.jsonl",
            mix / "calibration-v7.jsonl",
        ],
        "heldout-select": [mix / "heldout-select.jsonl"],
        "test": sorted((mix / "test").glob("*.jsonl")),
    }
    if jevbench is not None:
        files["jevbench"] = [jevbench]
    return {k: [p for p in v if p.exists()] for k, v in files.items()}


def make_encoder(tokenizer: Path) -> Callable[[list[str]], list[int]]:
    from tjev.data.tokenize import PromptTokenizer

    tok = PromptTokenizer(tokenizer)  # checks the letters and the answer slot

    def encode(texts: list[str]) -> list[int]:
        return [len(e.ids) for e in tok.tokenizer.encode_batch(texts, add_special_tokens=False)]

    return encode


def find_tokenizer() -> Path | None:
    models = Path("~/tjev-work/models").expanduser()
    preferred = models / "Qwen3.5-0.8B"
    if (preferred / "tokenizer.json").exists():
        return preferred
    found = sorted(models.glob("Qwen3.5*/tokenizer.json"))
    return found[0].parent if found else None


def analyse(
    mix: Path,
    jevbench: Path | None,
    *,
    encode_batch: Callable[[list[str]], list[int]],
    sample: int | None = None,
    steps: int = STEPS,
    tokens_per_step: int = TOKENS_PER_STEP,
    microbatch_tokens: int = MICROBATCH_TOKENS,
    seed: int = 0,
    log: Callable[[str], None] = print,
    packing: str = "bucket",
) -> dict:
    t0 = time.time()
    weights, blocks = read_mixture(mix)
    scanner = Scanner(encode_batch, seed=seed)
    train_cols = Columns("train")
    train_files = sorted((mix / "train").glob("*.jsonl"))
    full_rows = {}
    for path in train_files:
        full_rows[path.stem] = count_rows(path)
        scanner.scan(train_cols, path, sample)
        log(f"[scan] {path.stem}: {len(train_cols)} rows so far ({time.time() - t0:.0f}s)")
    train = train_cols.arrays()
    held = {}
    for split, paths in held_files(mix, jevbench).items():
        cols = Columns(split)
        for p in paths:
            scanner.scan(cols, p)
        held[split] = cols.arrays()
    log(f"[scan] held-out splits done ({time.time() - t0:.0f}s)")

    sources = sorted(set(train["source"].tolist()) | set(weights))
    total_w = sum(weights.values())
    total_rows = sum(full_rows.values())
    tok_mean = 0.5 * (train["tokens_plain"] + train["tokens_json"])
    lengths = {
        s: (
            train["tokens_plain"][train["source"] == s],
            train["tokens_json"][train["source"] == s],
            train["n_options"][train["source"] == s],
        )
        for s in sources
    }
    probs = {s: weights.get(s, 0.0) / total_w for s in sources if weights.get(s, 0.0) > 0}
    sim = simulate_stream(
        {s: lengths[s] for s in probs},
        probs,
        steps=steps,
        tokens_per_step=tokens_per_step,
        microbatch_tokens=microbatch_tokens,
        seed=seed,
        packing=packing,
    )
    log(
        f"[simulate] {sim['steps']} steps, {sim['segments_per_step']} segments/step "
        f"({time.time() - t0:.0f}s)"
    )
    token_mass = {
        s: probs.get(s, 0.0) * float(tok_mean[train["source"] == s].mean()) for s in probs
    }
    token_total = sum(token_mass.values())

    per_source: dict[str, dict] = {}
    for s in sources:
        m = train["source"] == s
        rows = full_rows.get(s, 0)
        simulated = sim["by_source"].get(s, {"trained": 0, "seen_share_of_sample": 0.0})
        both = np.concatenate([train["tokens_plain"][m], train["tokens_json"][m]])
        per_source[s] = {
            "block": blocks.get(s, "?"),
            "rows": rows,
            "rows_analysed": int(m.sum()),
            "raw_share": round(rows / total_rows, 5),
            "sampled_share": round(probs.get(s, 0.0), 5),
            "sampled_token_share": round(token_mass.get(s, 0.0) / token_total, 5)
            if token_total
            else 0.0,
            "sampled_over_raw": round(probs.get(s, 0.0) / max(rows / total_rows, 1e-12), 3),
            "epochs": round(simulated["trained"] / max(rows, 1), 3),
            "seen_share": simulated["seen_share_of_sample"],
            "lang": dict(Counter(train["lang"][m].tolist())),
            "type": dict(Counter(train["type"][m].tolist())),
            "options": percentiles(train["n_options"][m]),
            "tokens_plain": percentiles(train["tokens_plain"][m]),
            "tokens_json": percentiles(train["tokens_json"][m]),
            "buckets": bucket_shares(both, BUCKETS),
            "over_max_bucket_rows": int(
                (
                    (train["tokens_plain"][m] > max(BUCKETS))
                    | (train["tokens_json"][m] > max(BUCKETS))
                ).sum()
            ),
            "state_words": percentiles(train["state_words"][m]),
            "length_hist": np.histogram(both, LENGTH_EDGES)[0].tolist(),
            "labels": label_balance(train, m),
        }
    everything = np.ones(len(train["source"]), bool)
    both_all = np.concatenate([train["tokens_plain"], train["tokens_json"]])
    lang_sampled = Counter()
    type_sampled = Counter()
    for s, p in probs.items():
        m = train["source"] == s
        for k, v in Counter(train["lang"][m].tolist()).items():
            lang_sampled[k] += p * v / m.sum()
        for k, v in Counter(train["type"][m].tolist()).items():
            type_sampled[k] += p * v / m.sum()
    overall = {
        "sources": len(sources),
        "rows": total_rows,
        "rows_analysed": len(train["source"]),
        "sample_per_source": sample,
        "prompt_tokens_plain": int(train["tokens_plain"].sum()),
        "prompt_tokens_json": int(train["tokens_json"].sum()),
        "prompt_tokens_mean_layout": int(tok_mean.sum()),
        "lang_rows": dict(Counter(train["lang"].tolist())),
        "lang_sampled": {k: round(v, 4) for k, v in lang_sampled.items()},
        "type_rows": dict(Counter(train["type"].tolist())),
        "type_sampled": {k: round(v, 4) for k, v in type_sampled.items()},
        "options": percentiles(train["n_options"]),
        "options_hist": dict(sorted(Counter(train["n_options"].tolist()).items())),
        "tokens_plain": percentiles(train["tokens_plain"]),
        "tokens_json": percentiles(train["tokens_json"]),
        "buckets": bucket_shares(both_all, BUCKETS),
        "over_max_bucket_rows": int(sum(v["over_max_bucket_rows"] for v in per_source.values())),
        "length_edges": LENGTH_EDGES.tolist(),
        "length_hist": np.histogram(both_all, LENGTH_EDGES)[0].tolist(),
        "labels": label_balance(train, everything),
        "eval_letter_validation": _eval_letters(held.get("validation")),
    }
    log(f"[dedup] … ({time.time() - t0:.0f}s)")
    dup = duplication(train)
    log(f"[leakage] … ({time.time() - t0:.0f}s)")
    leak = leakage(train, held)
    outliers = _outliers(train, scanner.flags)
    report = {
        "mix": str(mix),
        "jevbench": str(jevbench) if jevbench else None,
        "settings": {
            "steps": steps,
            "tokens_per_step": tokens_per_step,
            "microbatch_tokens": microbatch_tokens,
            "buckets": list(BUCKETS),
            "max_segments": MAX_SEGMENTS,
            "packing_bins": PACKING_BINS,
            "near_threshold": NEAR_THRESHOLD,
            "num_perm": NUM_PERM,
            "bands": BANDS,
            "shingle_words": SHINGLE,
            "seed": seed,
        },
        "overall": overall,
        "simulation": {k: v for k, v in sim.items() if k != "by_source"},
        "sources": per_source,
        "duplication": dup,
        "leakage": leak,
        "flags": scanner.flags.to_json(),
        "outliers": outliers,
    }
    report["recommendations"] = recommendations(report)
    report["runtime_s"] = round(time.time() - t0, 1)
    return report


def _eval_letters(h: dict[str, np.ndarray] | None) -> dict:
    if h is None or len(h.get("source", ())) == 0:
        return {}
    m = h["type"] != "score"
    counts = np.bincount(h["eval_letter"][m], minlength=26)[:26] / max(m.sum(), 1)
    n = h["n_options"][m]
    chance = np.array([(n > k).astype(float) @ (1.0 / n) for k in range(26)]) / max(m.sum(), 1)
    last = int(n.max()) if len(n) else 0
    return {
        "observed": [round(float(v), 4) for v in counts[:last]],
        "chance": [round(float(v), 4) for v in chance[:last]],
    }


def _outliers(train: dict[str, np.ndarray], flags: Flags) -> dict:
    """Rows above 4× their source's p99 token length, and above the largest bucket."""
    out = {}
    longest = np.maximum(train["tokens_plain"], train["tokens_json"])
    over = longest > max(BUCKETS)
    out["over_max_bucket"] = int(over.sum())
    extreme = 0
    for s in sorted(set(train["source"].tolist())):
        m = train["source"] == s
        p99 = np.percentile(longest[m], 99)
        extreme += int((longest[m] > 4 * p99).sum())
    out["above_4x_source_p99"] = extreme
    for i in np.flatnonzero(over)[:EXAMPLES]:
        flags.add(
            "length:over_max_bucket", str(train["source"][i]), f"row {i}", tokens=int(longest[i])
        )
    return out


# --------------------------------------------------------------------------------------
# Recommendations


def recommendations(report: dict) -> list[dict]:
    """Ranked actions from the numbers; severity orders them (higher first)."""
    recs: list[dict] = []
    src = report["sources"]
    dup, leak, flags = report["duplication"], report["leakage"], report["flags"]

    def add(severity: float, title: str, action: str, effect: str, touches: str) -> None:
        recs.append(
            {
                "severity": round(severity, 2),
                "title": title,
                "action": action,
                "effect": effect,
                "touches": touches,
            }
        )

    for split in ("jevbench", "test", "validation", "calibration", "heldout-select"):
        info = leak.get(split)
        if not info:
            continue
        exact = info["train_rows_exact_state_instructions"]
        near = info["near_state"]
        if exact or near:
            strong = exact or info["near_item"]
            worst = sorted(
                info["by_heldout_source"].items(),
                key=lambda x: -(x[1]["near_item"] * 10 + x[1]["exact_state"] + x[1]["near"]),
            )[:4]
            where = "; ".join(
                f"{s}: {v['near_item']} near items, {v['exact_state']} equal states, "
                f"{v['near']} near states (of {v['rows']})"
                for s, v in worst
            )
            add(
                (10 if split in ("jevbench", "test") else 8) if strong else 3,
                f"Train overlaps {split}",
                f"{exact} train rows equal a {split} row on (state, instructions); "
                f"{info['train_rows_exact_state']} on the state alone; {near} {split} rows "
                f"have a near-duplicate (J≥0.8) train state, {info['near_item']} of them with the "
                f"same instructions and gold label. Mostly {where}. Drop them in the builder "
                f"(cross-source key check, near check for every source).",
                f"Removes optimistic bias of {split} metrics "
                f"({info['near_item_share']:.1%} of {split} rows are near-duplicate items).",
                "data rebuild (train file hashes → data identity)",
            )
    if dup["conflicting_rows"]:
        add(
            7,
            "Conflicting labels",
            f"{dup['conflicting_rows']} rows ({dup['conflicting_groups']} inputs) share the "
            f"same state, instructions and menu with different targets "
            f"({dup['conflicting_cross_source_rows']} across sources). Drop or merge them into "
            f"a soft target.",
            "Removes contradictory gradients; small direct effect.",
            "data rebuild",
        )
    if dup["exact_duplicate_rows"]:
        add(
            5 + min(dup["exact_duplicate_rows"] / 5000, 2),
            "Exact duplicates",
            f"{dup['exact_duplicate_rows']} rows are exact duplicates of another train row "
            f"({dup['exact_duplicate_cross_source_rows']} across sources). Dedup.",
            "Their sources' effective epochs on the duplicated items fall; no new data.",
            "data rebuild",
        )
    high = {s: v["epochs"] for s, v in src.items() if v["epochs"] > HIGH_EPOCHS}
    if high:
        listing = ", ".join(f"{s} {e:.1f}" for s, e in sorted(high.items(), key=lambda x: -x[1]))
        add(
            6,
            f"Sources repeated > {HIGH_EPOCHS:g} epochs",
            f"At {report['settings']['steps']} steps: {listing}. Lower their weights or raise "
            f"their caps / generate more rows (generators are cheap to regenerate).",
            "Less memorisation of small sources; the freed share goes to sources seen < 1 epoch.",
            "weights: run config identity only (mix.yaml); caps / generators: data rebuild",
        )
    low = {s: v["epochs"] for s, v in src.items() if v["epochs"] < LOW_EPOCHS}
    if low:
        listing = ", ".join(f"{s} {e:.2f}" for s, e in sorted(low.items(), key=lambda x: x[1]))
        add(
            3,
            f"Sources barely seen (< {LOW_EPOCHS:g} epoch)",
            f"{listing}. Most of their rows are never trained on: lower their caps (smaller, "
            f"faster build) or raise their weights if they matter.",
            "No quality change at equal weights; a smaller build.",
            "caps: data rebuild; weights: run config identity",
        )
    over = report["outliers"]["over_max_bucket"]
    if over:
        frac = report["simulation"]["dropped_too_long"] / max(report["simulation"]["segments"], 1)
        add(
            4,
            "Rows longer than the largest bucket",
            f"{over} rows exceed {max(BUCKETS)} tokens in at least one layout; the packer drops "
            f"them silently ({report['simulation']['dropped_too_long']} draws in the simulation, "
            f"{frac:.2%}). Cap them in the builder or add a larger bucket.",
            "No silent drop; the sampled mix matches mix.yaml.",
            "data rebuild (or config identity for a new bucket)",
        )
    for part in ("state", "instructions"):
        info = flags.get(f"lang_mismatch:{part}")
        if info and info["total"] > 100:
            top = ", ".join(f"{s} {n}" for s, n in list(info["by_source"].items())[:5])
            add(
                3 + min(info["total"] / 10000, 2),
                f"Language mismatch ({part})",
                f"{info['total']} rows whose {part} looks like another language than their "
                f"`lang` ({top}). Check the examples; fix the lang tag or the adapter.",
                "Correct per-language metrics and the French share.",
                "data rebuild",
            )
    artefacts = {k: v for k, v in flags.items() if k.startswith("artefact:")}
    for k, v in artefacts.items():
        kind = k.split(":", 1)[1]
        severity, advice = ARTEFACT_ADVICE.get(kind, (2.0, "Review the examples."))
        if v["total"] > 50 or (kind in ("python_repr", "nonstring_criterion") and v["total"]):
            top = ", ".join(f"{s} {n}" for s, n in list(v["by_source"].items())[:5])
            add(
                severity + min(v["total"] / 20000, 1),
                f"Template artefacts ({kind})",
                f"{v['total']} rows ({top}). {advice}",
                "Cleaner inputs; the model stops reading artefacts as features.",
                "data rebuild",
            )
    near = [
        (s, v["near_redundant_share"])
        for s, v in dup["by_source"].items()
        if v["near_redundant_share"] > 0.3
    ]
    if near:
        listing = ", ".join(f"{s} {r:.0%}" for s, r in sorted(near, key=lambda x: -x[1]))
        add(
            2.5,
            "Sources with many near-duplicate states",
            f"Share of states that are near-duplicates (J≥0.8) of another state of the source: "
            f"{listing}. Expected for templated generators; for public data, near-dedup.",
            "Effective diversity is lower than the row count suggests.",
            "data rebuild",
        )
    letters = report["overall"]["labels"].get("choice_noul", {})
    if letters and (
        letters["train_letter_tvd"] > 0.02 or abs(letters["train_letter_A_ratio"] - 1) > 0.05
    ):
        add(
            4,
            "Answer-letter imbalance after shuffling",
            f"Gold letters under the training render are {letters['train_letter_tvd']:.3f} "
            f"(total variation) from chance, A at {letters['train_letter_A_ratio']:.2f}× chance. "
            f"Check choose_order.",
            "Removes a positional prior.",
            "TEMPLATE_VERSION bump",
        )
    sim = report["simulation"]
    steps = max(sim["steps"], 1)
    under = {
        b: f
        for b, f in sim["fill_by_bucket"].items()
        if f < 0.85 and sim["steps_by_bucket"].get(b, 0) / steps > 0.05
    }
    if under:
        listing = ", ".join(
            f"T={b}: {sim['steps_by_bucket'][b] / steps:.0%} of steps at {f:.0%} fill"
            for b, f in under.items()
        )
        add(
            5,
            "Long-bucket steps are under-filled",
            f"{listing} (overall fill {sim['fill']:.0%}). A step is one bucket, and a long "
            f"segment leaves a tail that only same-bucket segments may fill. Let short segments "
            f"fill the tails of open 2048 / 4096 rows (first-fit across buckets ≥ their own), "
            f"or shrink the long sources' share.",
            f"Up to {1 - sim['fill']:.0%} more segments per step at the same cost.",
            "packer change: stream identity (pinned stream test) and slots_per_step",
        )
    hogs = [
        (s, v)
        for s, v in src.items()
        if v["sampled_token_share"] > 0.05 and v["sampled_token_share"] > 3 * v["sampled_share"]
    ]
    for s, v in hogs:
        add(
            4.5,
            f"{s} dominates the token budget",
            f"{s} is {v['sampled_share']:.1%} of the sampled segments but "
            f"{v['sampled_token_share']:.1%} of the prompt tokens (p50 "
            f"{v['tokens_plain'].get('p50', 0):.0f} tokens), at {v['epochs']:.1f} epochs.",
            "Halving its weight frees ~half its token share for other sources; regenerating "
            "more rows lowers its epochs without changing its share.",
            "weight: run config identity; more rows: data rebuild",
        )
    constant = [
        (s, v["labels"])
        for s, v in src.items()
        if v["labels"].get("majority_share", 0) >= 0.97 and v["rows"] >= 200
    ]
    for s, lab in constant:
        add(
            9,
            f"{s}: one label for (almost) every row",
            f"{lab['majority_share']:.1%} of {s}'s hard-target rows have gold "
            f"`{lab['majority_label']}`. The model learns a constant, and the source's held-out "
            f"metrics are meaningless. Likely an adapter bug (label mapping): fix the adapter, "
            f"or drop the source until it is fixed (its weight goes to its block).",
            "Removes a degenerate prior from its block (and from the French share).",
            "data rebuild (or weight 0: run config identity)",
        )
    skewed = [
        (s, v["labels"]["noul_yes_share"])
        for s, v in src.items()
        if "noul_yes_share" in v["labels"]
        and v["labels"].get("noul_rows", 0) >= 200
        and not 0.25 <= v["labels"]["noul_yes_share"] <= 0.75
    ]
    if skewed:
        listing = ", ".join(f"{s} {y:.0%}" for s, y in sorted(skewed, key=lambda x: x[1]))
        add(
            2,
            "Skewed yes/no base rates",
            f"Yes-share outside 25–75%: {listing}. Fine if it mirrors the task; otherwise "
            f"rebalance in the adapter (shuffling letters cannot fix a label prior).",
            "Better-calibrated yes/no on balanced evaluation sets.",
            "data rebuild",
        )
    return sorted(recs, key=lambda r: -r["severity"])


# --------------------------------------------------------------------------------------
# Markdown and plots


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def markdown(report: dict) -> str:
    o, sim, dup, s = (
        report["overall"],
        report["simulation"],
        report["duplication"],
        report["settings"],
    )
    out = [
        f"# Data quality: {Path(report['mix']).name}",
        "",
        f"Generated by `scripts/data_report.py` in {report['runtime_s']:.0f} s. "
        f"Rows analysed: {o['rows_analysed']:,} of {o['rows']:,}"
        + (f" (first {o['sample_per_source']} per source)." if o["sample_per_source"] else ".")
        + f" Run assumed: {s['steps']} steps × {s['tokens_per_step']:,} tokens, microbatch "
        f"{s['microbatch_tokens']} tokens (accumulation {sim['accumulation']}), buckets "
        f"{s['buckets']}, {s['packing_bins']} packing bins.",
        "",
    ]
    out += ["## Recommendations (ranked)", ""]
    for k, r in enumerate(report["recommendations"], 1):
        out.append(
            f"{k}. **{r['title']}.** {r['action']} *Effect:* {r['effect']} "
            f"*Touches:* {r['touches']}."
        )
    out += [
        "",
        "## Overview",
        "",
        _table(
            ["", "value"],
            [
                ["sources", o["sources"]],
                ["train rows", f"{o['rows']:,}"],
                [
                    "prompt tokens (plain / json layout)",
                    f"{o['prompt_tokens_plain']:,} / {o['prompt_tokens_json']:,}",
                ],
                ["rows by language", ", ".join(f"{k} {v:,}" for k, v in o["lang_rows"].items())],
                [
                    "sampled language share",
                    ", ".join(f"{k} {_pct(v)}" for k, v in o["lang_sampled"].items()),
                ],
                ["rows by type", ", ".join(f"{k} {v:,}" for k, v in o["type_rows"].items())],
                [
                    "sampled type share",
                    ", ".join(f"{k} {_pct(v)}" for k, v in o["type_sampled"].items()),
                ],
                [
                    "options p50 / p90 / max",
                    f"{o['options']['p50']:.0f} / {o['options']['p90']:.0f} / {o['options']['max']}",
                ],
                [
                    "soft targets",
                    f"{o['labels']['soft_rows']:,} ({_pct(o['labels']['soft_share'])}), mean normalised entropy {o['labels'].get('soft_entropy_mean', 0):.2f}",
                ],
                [
                    "tokens plain p50 / p90 / p99 / max",
                    "{p50:.0f} / {p90:.0f} / {p99:.0f} / {max}".format(**o["tokens_plain"]),
                ],
                [
                    "tokens json p50 / p90 / p99 / max",
                    "{p50:.0f} / {p90:.0f} / {p99:.0f} / {max}".format(**o["tokens_json"]),
                ],
                [
                    "bucket shares (both layouts)",
                    ", ".join(f"{k}: {_pct(v)}" for k, v in o["buckets"].items()),
                ],
                ["rows over the largest bucket", o["over_max_bucket_rows"]],
                ["segments per step (simulated)", sim["segments_per_step"]],
                [
                    "packing fill (by bucket)",
                    f"{_pct(sim['fill'])} ({', '.join(f'{k}: {_pct(v)}' for k, v in sim['fill_by_bucket'].items())})",
                ],
                [
                    "steps by bucket",
                    ", ".join(f"{k}: {v}" for k, v in sim["steps_by_bucket"].items()),
                ],
                ["draws dropped as too long", sim["dropped_too_long"]],
            ],
        ),
    ]
    out += [
        "",
        "## Mix per source",
        "",
        (
            "Sampled share = mix.yaml weight; token share = sampled share × mean prompt "
            "length; epochs = segments of the source trained on in the simulated run / rows; "
            "seen = rows trained on at least once."
        ),
        "",
    ]
    rows = []
    for name, v in sorted(report["sources"].items(), key=lambda x: (x[1]["block"], x[0])):
        flag = " ⚠" if v["epochs"] > HIGH_EPOCHS or v["epochs"] < LOW_EPOCHS else ""
        rows.append(
            [
                name,
                v["block"],
                f"{v['rows']:,}",
                "/".join(f"{k}" for k in v["lang"]),
                ", ".join(f"{k} {n}" for k, n in v["type"].items()),
                f"{v['options'].get('p50', 0):.0f}",
                _pct(v["raw_share"]),
                _pct(v["sampled_share"]),
                _pct(v["sampled_token_share"]),
                f"{v['epochs']:.2f}{flag}",
                _pct(v["seen_share"]),
            ]
        )
    out.append(
        _table(
            [
                "source",
                "block",
                "rows",
                "lang",
                "types",
                "opts p50",
                "raw",
                "sampled",
                "token share",
                "epochs",
                "seen",
            ],
            rows,
        )
    )
    cn0 = o["labels"].get("choice_noul", {})
    out += [
        "",
        "## Labels",
        "",
        (
            "Gold letter under the training render (choice + noul), "
            "observed vs chance (E[1/shown]: with fewer options, A and B are more often "
            f"right by construction): total variation {cn0.get('train_letter_tvd', 0)}, A at "
            f"{cn0.get('train_letter_A_ratio', 0)}× chance. As stored (before shuffling): total "
            f"variation {cn0.get('stored_tvd', 0)}, first label at {cn0.get('stored_A_ratio', 0)}× "
            "chance; this order is what a model would see with shuffling off. Score levels keep "
            "their order, so their letter is the level."
        ),
        "",
    ]
    cn = o["labels"].get("choice_noul", {})
    if cn:
        out.append(
            _table(
                ["letter", *LETTERS[: len(cn["train_letter"])]][:11],
                [
                    ["observed", *[_pct(x) for x in cn["train_letter"]]][:11],
                    ["chance", *[_pct(x) for x in cn["train_letter_chance"]]][:11],
                    ["stored position", *[_pct(x) for x in cn["stored_position"]]][:11],
                    ["stored chance", *[_pct(x) for x in cn["stored_position_chance"]]][:11],
                ],
            )
        )
    rows = []
    for name, v in sorted(report["sources"].items()):
        lab = v["labels"]
        cnl = lab.get("choice_noul", {})
        maj = (
            f"{lab.get('majority_label', '')} {_pct(lab['majority_share'])}"
            if "majority_share" in lab
            else ""
        )
        rows.append(
            [
                name,
                f"{cnl.get('train_letter_tvd', '')}",
                f"{cnl.get('stored_A_ratio', '')}",
                maj,
                _pct(lab["noul_yes_share"]) if "noul_yes_share" in lab else "",
                " ".join(f"{x:.2f}" for x in lab.get("score_levels", [])),
                _pct(lab["soft_share"]),
                f"{lab.get('soft_entropy_mean', '')}",
            ]
        )
    out += [
        "",
        _table(
            [
                "source",
                "letter TV",
                "stored A/chance",
                "majority gold",
                "noul yes",
                "score levels",
                "soft",
                "soft entropy",
            ],
            rows,
        ),
    ]
    out += [
        "",
        "## Duplication (train)",
        "",
        _table(
            ["", "rows"],
            [
                [
                    "exact duplicates (beyond the first)",
                    f"{dup['exact_duplicate_rows']:,} ({dup['exact_duplicate_cross_source_rows']:,} cross-source)",
                ],
                [
                    "conflicting targets (same input + menu)",
                    f"{dup['conflicting_rows']:,} in {dup['conflicting_groups']:,} groups ({dup['conflicting_cross_source_rows']:,} cross-source)",
                ],
                [
                    "same state + instructions, other menu",
                    f"{dup['same_question_other_menu_rows']:,}",
                ],
                [
                    "rows whose state occurs in another row",
                    f"{dup['shared_state_rows']:,} ({dup['shared_state_cross_source_rows']:,} cross-source)",
                ],
                ["distinct states", f"{dup['distinct_states']:,}"],
                [
                    "near-duplicate clusters / states in them",
                    f"{dup['near_duplicate_clusters']:,} / {dup['near_duplicate_states']:,}",
                ],
                [
                    "near-redundant states (beyond one per cluster)",
                    f"{dup['near_redundant_states']:,} ({dup['near_cross_source_states']:,} in cross-source clusters)",
                ],
            ],
        ),
    ]
    if dup["shared_state_cross_source_pairs"]:
        out += [
            "",
            "Cross-source shared states (distinct states): "
            + "; ".join(f"{k}: {v}" for k, v in dup["shared_state_cross_source_pairs"].items()),
        ]
    rows = [
        [
            n,
            f"{v['rows']:,}",
            v["exact_duplicate_rows"],
            v["conflicting_rows"],
            v["shared_state_rows"],
            f"{v['distinct_states']:,}",
            _pct(v["near_redundant_share"]),
        ]
        for n, v in sorted(dup["by_source"].items())
    ]
    out += [
        "",
        _table(
            [
                "source",
                "rows",
                "exact dup",
                "conflicts",
                "shared state",
                "distinct states",
                "near-redundant",
            ],
            rows,
        ),
    ]
    out += [
        "",
        "## Leakage (train vs held-out)",
        "",
        (
            "Held-out rows whose normalised (state, instructions) / state occurs in train, or "
            "whose state has a near-duplicate train state (est. Jaccard ≥ 0.8; *near item*: "
            "also the same instructions and gold label); and the train rows that would be "
            "dropped for the exact matches."
        ),
        "",
    ]
    rows = []
    for split, v in report["leakage"].items():
        rows.append(
            [
                split,
                f"{v['rows']:,}",
                v["exact_state_instructions"],
                v["exact_state_instructions_same_source"],
                v["exact_state"],
                f"{v['near_state']} ({_pct(v['near_state_share'])})",
                f"{v['near_item']} ({_pct(v['near_item_share'])})",
                v["train_rows_exact_state_instructions"],
                v["train_rows_exact_state"],
            ]
        )
    out.append(
        _table(
            [
                "split",
                "rows",
                "exact s+i",
                "…same source",
                "exact state",
                "near state",
                "near item",
                "train rows s+i",
                "train rows state",
            ],
            rows,
        )
    )
    for split, v in report["leakage"].items():
        if v["near_by_pair"]:
            out += [
                "",
                f"Near pairs into {split} (train source → held-out source): "
                + "; ".join(f"{k}: {n}" for k, n in v["near_by_pair"].items()),
            ]
    out += [
        "",
        "## Length (prompt tokens, real template and tokenizer)",
        "",
        "![lengths](lengths.png)",
        "",
    ]
    rows = []
    for name, v in sorted(report["sources"].items()):
        tp, tj = v["tokens_plain"], v["tokens_json"]
        rows.append(
            [
                name,
                f"{tp['p50']:.0f}",
                f"{tp['p90']:.0f}",
                f"{tp['p99']:.0f}",
                tp["max"],
                f"{tj['p50']:.0f}",
                tj["max"],
                ", ".join(f"{k}: {_pct(x)}" for k, x in v["buckets"].items()),
                v["over_max_bucket_rows"],
                f"{v['state_words'].get('p50', 0):.0f}",
            ]
        )
    out.append(
        _table(
            [
                "source",
                "plain p50",
                "p90",
                "p99",
                "max",
                "json p50",
                "json max",
                "buckets",
                f"> {max(BUCKETS)}",
                "state words p50",
            ],
            rows,
        )
    )
    out += [
        "",
        "## Quality flags",
        "",
        _table(
            ["flag", "rows", "top sources"],
            [
                [k, v["total"], ", ".join(f"{s} {n}" for s, n in list(v["by_source"].items())[:5])]
                for k, v in report["flags"].items()
            ],
        ),
        "",
        (
            f"Extreme outliers: {report['outliers']['above_4x_source_p99']} rows above 4× "
            f"their source's p99 length. Examples (≤ {EXAMPLES} per flag) are in report.json."
        ),
        "",
        "## Plots",
        "",
        "![shares](shares.png)",
        "",
        "![labels](labels.png)",
        "",
    ]
    return "\n".join(out)


BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"


def plots(report: dict, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.edgecolor": GREY,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 110,
        }
    )
    written = []
    o, src = report["overall"], report["sources"]
    edges = np.asarray(o["length_edges"])
    names = sorted(src, key=lambda n: src[n]["tokens_plain"].get("p50", 0))

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 6.5), gridspec_kw={"width_ratios": [1, 1.3]})
    hist = np.asarray(o["length_hist"])
    a.bar(
        edges[:-1],
        hist,
        width=np.diff(edges),
        align="edge",
        color=BLUE,
        edgecolor="white",
        linewidth=0.5,
    )
    a.set_xscale("log")
    for bkt in BUCKETS:
        a.axvline(bkt, color=GREY, linestyle="--", linewidth=1)
        a.text(bkt, a.get_ylim()[1] * 0.95, f" {bkt}", color=MUTED, fontsize=7)
    a.set_title("Prompt tokens, all train rows (both layouts)", loc="left")
    a.set_xlabel("tokens")
    a.set_ylabel("rows × layouts")
    y = np.arange(len(names))
    for key, color, label in (("p50", BLUE, "p50"), ("p99", ORANGE, "p99")):
        b.scatter(
            [src[n]["tokens_plain"].get(key, 0) for n in names],
            y,
            s=14,
            color=color,
            label=label,
            zorder=3,
        )
    b.scatter(
        [src[n]["tokens_plain"].get("max", 0) for n in names],
        y,
        s=10,
        marker="|",
        color=INK,
        label="max",
        zorder=3,
    )
    b.set_yticks(y, names, fontsize=6.5)
    b.set_xscale("log")
    for bkt in BUCKETS:
        b.axvline(bkt, color=GREY, linestyle="--", linewidth=1)
    b.set_title("Plain-layout tokens per source", loc="left")
    b.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    written.append(out / "lengths.png")
    fig.savefig(written[-1])
    plt.close(fig)

    order = sorted(src, key=lambda n: src[n]["sampled_share"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 7.5), sharey=True)
    y = np.arange(len(order))
    a.barh(y - 0.2, [src[n]["raw_share"] * 100 for n in order], 0.4, color=GREY, label="raw rows")
    a.barh(
        y + 0.2,
        [src[n]["sampled_share"] * 100 for n in order],
        0.4,
        color=BLUE,
        label="sampled (mix.yaml)",
    )
    a.set_yticks(y, order, fontsize=6.5)
    a.set_xlabel("share of the mix (%)")
    a.set_title("Raw vs sampled share", loc="left")
    a.legend(frameon=False, loc="lower right")
    ep = [max(src[n]["epochs"], 1e-3) for n in order]
    b.scatter(
        ep,
        y,
        s=14,
        color=[ORANGE if e > HIGH_EPOCHS or e < LOW_EPOCHS else BLUE for e in ep],
        zorder=3,
    )
    b.set_xscale("log")
    for v in (LOW_EPOCHS, 1, HIGH_EPOCHS):
        b.axvline(v, color=GREY, linestyle="--", linewidth=1)
    b.set_xlabel(f"effective epochs at {report['settings']['steps']} steps")
    b.set_title("Effective epochs (orange: > 4 or < 0.1)", loc="left")
    fig.tight_layout()
    written.append(out / "shares.png")
    fig.savefig(written[-1])
    plt.close(fig)

    cn = o["labels"].get("choice_noul")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    if cn:
        k = min(len(cn["train_letter"]), 12)
        x = np.arange(k)
        a.bar(
            x - 0.2,
            np.asarray(cn["train_letter"][:k]) * 100,
            0.4,
            color=BLUE,
            label="observed (training render)",
        )
        a.bar(
            x + 0.2,
            np.asarray(cn["train_letter_chance"][:k]) * 100,
            0.4,
            color=GREY,
            label="chance E[1/options]",
        )
        a.set_xticks(x, LETTERS[:k])
        a.set_ylabel("% of choice + noul rows")
        a.set_title("Gold letter after shuffling", loc="left")
        a.legend(frameon=False)
        b.bar(
            x - 0.2,
            np.asarray(cn["stored_position"][:k]) * 100,
            0.4,
            color=ORANGE,
            label="stored gold position",
        )
        b.bar(
            x + 0.2,
            np.asarray(cn["stored_position_chance"][:k]) * 100,
            0.4,
            color=GREY,
            label="chance",
        )
        b.set_xticks(x, [str(i + 1) for i in range(k)])
        b.set_title("Gold position as stored (before shuffling)", loc="left")
        b.legend(frameon=False)
    fig.tight_layout()
    written.append(out / "labels.png")
    fig.savefig(written[-1])
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mix", type=Path, required=True, help="a built mix directory")
    ap.add_argument("--jevbench", type=Path, default=None, help="JevBench public.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="report directory")
    ap.add_argument("--sample", type=int, default=None, help="first N rows of each train file")
    ap.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="folder with tokenizer.json (default: a Qwen3.5 under ~/tjev-work/models)",
    )
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--tokens-per-step", type=int, default=TOKENS_PER_STEP)
    ap.add_argument("--microbatch-tokens", type=int, default=MICROBATCH_TOKENS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--packing", default="bucket", help="bucket | best_fit (train.packing)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument(
        "--from-json",
        action="store_true",
        help="re-derive recommendations, report.md and plots from OUT/report.json",
    )
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.from_json:
        report = json.loads((args.out / "report.json").read_text())
        report["recommendations"] = recommendations(report)
        (args.out / "report.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False) + "\n"
        )
        (args.out / "report.md").write_text(markdown(report))
    else:
        tokenizer = args.tokenizer or find_tokenizer()
        if tokenizer is None:
            raise SystemExit("no Qwen3.5 tokenizer found: pass --tokenizer")
        report = analyse(
            args.mix.expanduser(),
            args.jevbench.expanduser() if args.jevbench else None,
            encode_batch=make_encoder(tokenizer),
            sample=args.sample,
            steps=args.steps,
            tokens_per_step=args.tokens_per_step,
            microbatch_tokens=args.microbatch_tokens,
            seed=args.seed,
            log=lambda m: print(m, flush=True),
            packing=args.packing,
        )
        report["tokenizer"] = str(tokenizer)
        (args.out / "report.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False) + "\n"
        )
        (args.out / "report.md").write_text(markdown(report))
    if not args.no_plots:
        try:
            for p in plots(report, args.out):
                print(f"[plot] {p}")
        except ImportError:
            print("[plot] matplotlib missing: plots skipped (uv run --with matplotlib …)")
    print(f"[done] {args.out} in {report['runtime_s']:.0f} s of analysis")


if __name__ == "__main__":
    main()
