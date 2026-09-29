"""Segments → packed fixed-shape rows.

A segment is one rendered (state, question) prompt; its answer slot is its last token.
Rows of ``T`` tokens hold several segments (``segment_ids`` 1..n, 0 = padding) and at
most ``S`` answer slots; each batch uses a single bucket ``T`` so shapes stay static.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, fields

import jax
import numpy as np

from tjev.data.item import TYPES, Item
from tjev.data.render import RenderSpec, render
from tjev.data.tokenize import PromptTokenizer


@dataclass
class Segment:
    tokens: np.ndarray  # int32 [n]
    label_ids: np.ndarray  # int32 [k]
    target: np.ndarray  # float32 [k] (displayed order)
    type_id: int
    source_id: int
    lang_id: int
    family_id: int
    index: int  # item index in its dataset (eval bookkeeping), or stream position


@jax.tree_util.register_dataclass
@dataclass
class Batch:
    tokens: np.ndarray  # [R,T] int32
    # [R,T] int32. Contract (check_segments): per row, ids 1, 2, 3, … each form one
    # contiguous run, padding (0) only at the end. The Mosaic GDN and attention kernels skip
    # blocks and carry state on that basis; the XLA paths would silently disagree otherwise.
    segment_ids: np.ndarray
    positions: np.ndarray  # [R,T] int32
    slots: np.ndarray  # [R,S] int32
    label_ids: np.ndarray  # [R,S,K] int32
    label_mask: np.ndarray  # [R,S,K] bool
    target: np.ndarray  # [R,S,K] float32
    weight: np.ndarray  # [R,S] float32 (0 = empty slot)
    type_id: np.ndarray  # [R,S] int32
    source_id: np.ndarray  # [R,S] int32
    lang_id: np.ndarray  # [R,S] int32
    family_id: np.ndarray  # [R,S] int32
    index: np.ndarray  # [R,S] int32 (-1 = empty)

    @property
    def shape_key(self) -> tuple[int, ...]:
        return self.tokens.shape


def check_segments(segment_ids: np.ndarray) -> None:
    """Raise unless every row is 1, 2, … in contiguous runs followed by padding 0s."""
    ids = np.asarray(segment_ids)
    real = ids > 0
    step = np.diff(ids, axis=-1)
    prefix = (real[..., :-1] >= real[..., 1:]).all()  # real tokens first, padding after
    starts = (np.where(real[..., 0], ids[..., 0] == 1, True)).all()
    runs = np.where(real[..., 1:], (step == 0) | (step == 1), True).all()
    if not (prefix and starts and runs):
        raise ValueError("segment ids must be contiguous runs 1, 2, … with padding (0) last")


def stack_batches(batches: list[Batch]) -> Batch:
    return Batch(**{f.name: np.stack([getattr(b, f.name) for b in batches]) for f in fields(Batch)})


class Vocab:
    """Stable string → id maps for metadata (persisted with a run)."""

    def __init__(self, names: dict[str, list[str]] | None = None):
        self.names = {
            k: list(v) for k, v in (names or {"source": [], "lang": [], "family": []}).items()
        }
        self.frozen = False  # training: every name is known up front (from_items)

    @classmethod
    def from_items(cls, *collections) -> Vocab:
        """Deterministic ids (sorted names), independent of sampling order."""
        seen: dict[str, set[str]] = {"source": set(), "lang": set(), "family": set()}
        for items in collections:
            for item in items:
                seen["source"].add(item.source)
                seen["lang"].add(item.lang)
                seen["family"].add(item.family)
        return cls({k: sorted(v) for k, v in seen.items()})

    def id(self, kind: str, name: str) -> int:
        table = self.names[kind]
        if name not in table:
            if self.frozen:  # a new id minted in a data worker would be lost silently
                raise KeyError(f"unknown {kind} {name!r} (vocab is frozen)")
            table.append(name)
        return table.index(name)


def make_segment(
    item: Item,
    tok: PromptTokenizer,
    spec: RenderSpec,
    vocab: Vocab,
    index: int,
    rng: np.random.Generator | None = None,
) -> Segment:
    text, order = render(item, spec, rng)
    tokens = np.asarray(tok.encode(text), np.int32)
    target = np.asarray([item.target[i] for i in order], np.float32)
    target = target / target.sum()  # sub-sampling never drops positives; renormalise anyway
    return Segment(
        tokens=tokens,
        label_ids=np.asarray(tok.letter_ids[: len(order)], np.int32),
        target=target,
        type_id=TYPES.index(item.question.type),
        source_id=vocab.id("source", item.source),
        lang_id=vocab.id("lang", item.lang),
        family_id=vocab.id("family", item.family),
        index=index,
    )


class RowBuilder:
    """First-fit packing of segments into ``rows`` rows of ``length`` tokens."""

    def __init__(self, rows: int, length: int, slots: int, labels: int):
        self.rows, self.length, self.slots, self.labels = rows, length, slots, labels
        self.content: list[list[Segment]] = [[] for _ in range(rows)]
        self.used = [0] * rows

    def add(self, seg: Segment) -> bool:
        n = len(seg.tokens)
        if n > self.length or len(seg.label_ids) > self.labels:
            raise ValueError(f"segment of {n} tokens / {len(seg.label_ids)} labels does not fit")
        for r in range(self.rows):
            if self.used[r] + n <= self.length and len(self.content[r]) < self.slots:
                self.content[r].append(seg)
                self.used[r] += n
                return True
        return False

    def empty(self) -> bool:
        return not any(self.content)

    def fill(self) -> float:
        return sum(self.used) / (self.rows * self.length)

    def segments(self) -> list[Segment]:
        return [s for row in self.content for s in row]

    def build(self) -> Batch:
        R, T, S, K = self.rows, self.length, self.slots, self.labels
        b = Batch(
            tokens=np.zeros((R, T), np.int32),
            segment_ids=np.zeros((R, T), np.int32),
            positions=np.zeros((R, T), np.int32),
            slots=np.zeros((R, S), np.int32),
            label_ids=np.zeros((R, S, K), np.int32),
            label_mask=np.zeros((R, S, K), bool),
            target=np.zeros((R, S, K), np.float32),
            weight=np.zeros((R, S), np.float32),
            type_id=np.zeros((R, S), np.int32),
            source_id=np.zeros((R, S), np.int32),
            lang_id=np.zeros((R, S), np.int32),
            family_id=np.zeros((R, S), np.int32),
            index=np.full((R, S), -1, np.int32),
        )
        for r, row in enumerate(self.content):
            start = 0
            for s, seg in enumerate(row):
                n, k = len(seg.tokens), len(seg.label_ids)
                b.tokens[r, start : start + n] = seg.tokens
                b.segment_ids[r, start : start + n] = s + 1
                b.positions[r, start : start + n] = np.arange(n)
                b.slots[r, s] = start + n - 1
                b.label_ids[r, s, :k] = seg.label_ids
                b.label_mask[r, s, :k] = True
                b.target[r, s, :k] = seg.target
                b.weight[r, s] = 1.0
                b.type_id[r, s] = seg.type_id
                b.source_id[r, s] = seg.source_id
                b.lang_id[r, s] = seg.lang_id
                b.family_id[r, s] = seg.family_id
                b.index[r, s] = seg.index
                start += n
        check_segments(b.segment_ids)
        return b


def bucket_for(n: int, buckets: tuple[int, ...]) -> int | None:
    for length in sorted(buckets):
        if n <= length:
            return length
    return None


def rows_for(length: int, microbatch_tokens: int) -> int:
    return max(1, microbatch_tokens // length)


def pack_eval(
    segments: list[Segment],
    buckets: tuple[int, ...],
    microbatch_tokens: int,
    slots: int,
    labels: int,
) -> tuple[list[Batch], list[int]]:
    """Pack a finite set (longest first). Returns batches and indices of dropped segments."""
    builders: dict[int, RowBuilder] = {}
    batches, dropped = [], []
    for seg in sorted(segments, key=lambda s: -len(s.tokens)):
        length = bucket_for(len(seg.tokens), buckets)
        if length is None:
            dropped.append(seg.index)
            continue
        builder = builders.setdefault(
            length, RowBuilder(rows_for(length, microbatch_tokens), length, slots, labels)
        )
        if not builder.add(seg):
            batches.append(builder.build())
            builder = builders[length] = RowBuilder(builder.rows, length, slots, labels)
            builder.add(seg)
    batches.extend(b.build() for b in builders.values() if not b.empty())
    return batches, dropped


class BucketPacker:
    """Routes segments to the smallest bucket that fits and packs them into microbatches,
    emitting a step once ``accumulation`` microbatches of one bucket are complete.

    ``bins`` open microbatches per bucket (Grain's multi-bin first-fit): a segment goes to
    the first open one with room; when none has room and all bins are open, the fullest is
    closed. Nearly full ones (>= ``FULL``) close early. ``bins=1`` is next-fit per bucket:
    the original packing.
    """

    FULL = 0.99

    def __init__(
        self,
        segment_at: Callable[[int], Segment],
        *,
        buckets: tuple[int, ...],
        microbatch_tokens: int,
        accumulation: int,
        slots: int,
        labels: int,
        bins: int = 1,
    ):
        self.segment_at = segment_at
        self.buckets = tuple(sorted(buckets))
        self.microbatch_tokens, self.accumulation = microbatch_tokens, accumulation
        self.slots, self.labels, self.bins = slots, labels, bins
        self.next_index = 0
        self.open: dict[int, list[RowBuilder]] = {}
        self.ready: dict[int, list[RowBuilder]] = {b: [] for b in self.buckets}
        self.stats: Counter = Counter()

    def _builder(self, length: int) -> RowBuilder:
        return RowBuilder(rows_for(length, self.microbatch_tokens), length, self.slots, self.labels)

    def place(self, seg: Segment) -> None:
        self.next_index += 1
        length = bucket_for(len(seg.tokens), self.buckets)
        if length is None or len(seg.label_ids) > self.labels:
            reason = "too_long" if length is None else "too_many_labels"
            self.stats[f"dropped_{reason}"] += 1
            return
        builders = self.open.setdefault(length, [])
        target = next((b for b in builders if b.add(seg)), None)
        if target is None:
            if len(builders) >= self.bins:
                fullest = max(range(len(builders)), key=lambda i: builders[i].fill())
                self.ready[length].append(builders.pop(fullest))
            target = self._builder(length)
            target.add(seg)
            builders.append(target)
        if self.bins > 1 and target.fill() >= self.FULL:
            builders.remove(target)
            self.ready[length].append(target)

    def pop_step(self) -> Batch | None:
        for length in self.buckets:
            if len(self.ready[length]) >= self.accumulation:
                group = self.ready[length][: self.accumulation]
                self.ready[length] = self.ready[length][self.accumulation :]
                self.stats[f"steps_T{length}"] += 1
                return stack_batches([g.build() for g in group])
        return None

    def state(self) -> dict:
        def rows(builder: RowBuilder):
            return [[s.index for s in row] for row in builder.content]

        return {
            "next_index": self.next_index,
            "open": {str(k): [rows(b) for b in v] for k, v in self.open.items() if v},
            "ready": {str(k): [rows(b) for b in v] for k, v in self.ready.items() if v},
        }

    def restore(self, state: dict) -> None:
        def rebuild(length: int, rows: list[list[int]]) -> RowBuilder:
            builder = self._builder(length)
            for r, indices in enumerate(rows):
                for i in indices:
                    seg = self.segment_at(i)
                    builder.content[r].append(seg)
                    builder.used[r] += len(seg.tokens)
            return builder

        self.next_index = int(state["next_index"])
        self.open = {int(k): [rebuild(int(k), b) for b in v] for k, v in state["open"].items()}
        self.ready = {b: [] for b in self.buckets}
        for k, groups in state.get("ready", {}).items():
            self.ready[int(k)] = [rebuild(int(k), g) for g in groups]
