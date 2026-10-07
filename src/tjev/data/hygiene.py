"""Cross-source train hygiene of a built mix (since mix-v4): a post-pass over train files.

``tjev.data.mix.build`` builds every source on its own (per-source held-out keys, label
conflicts, the JevBench filter), then re-reads each train file in build order and drops,
first reason wins:

``train_equals_heldout``
    normalised (state, instructions) equal to a held-out row of *any* source
    (validation, calibration, test, the held-out generator sets, the JevBench public items);
``train_near_heldout``
    a state near-duplicate (MinHash, :mod:`tjev.data.near`: word 3-grams, 128
    permutations, LSH 32 × 4, estimated Jaccard ≥ 0.8) of a held-out row with the same
    normalised instructions;
``train_state_in_heldout`` / ``train_state_near_heldout`` (``passages``, since mix-v6)
    the state alone equals (normalised) or is a near-duplicate of a held-out state, whatever
    the question: a passage that a held-out row asks about (BoolQ, SQuAD, PIAF reuse one
    passage for several questions) is not trained on, so held-out sets stay unchanged and
    a model is never evaluated on a passage it read in training;
``jevbench_8gram`` (``grams``, since mix-v6)
    the row's full text (state, instructions, options) shares a word 8-gram with a public
    JevBench item (:class:`tjev.data.jevbench.JevBenchGrams`);
``prompt_too_long``
    the rendered prompt (every option shown, plain or JSON layout) is longer than the
    largest bucket (``limit``: ``MAX_PROMPT_TOKENS`` by default, mix-v4's; 16384 for a mix
    trained with the long buckets): the packer would drop it silently;
``train_exact_duplicate``
    the same normalised state + instructions + type + options + target as an earlier
    train row of any source (the first is kept).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tjev.data.item import Item
from tjev.data.jevbench import JevBenchGrams
from tjev.data.near import BandIndex, MinHasher, near_hits, normalized, shingle_hashes, stable_hash
from tjev.data.render import RenderSpec, render

MAX_PROMPT_TOKENS = 4096  # mix-v4's limit: the default largest bucket (config.seq_buckets)
EVAL_LAYOUTS = (RenderSpec(train=False, layout="plain"), RenderSpec(train=False, layout="json"))

TokenCounter = Callable[[list[str]], list[int]]


def si_key(item: Item) -> str:
    """Normalised (state, instructions): ``tjev.data.mix.key_of`` on a parsed item."""
    return normalized(item.state) + "\x1f" + normalized(item.question.instructions)


def exact_key(item: Item) -> int:
    """Normalised state + instructions, the type, the menu and the target (the exact
    duplicate key of ``scripts/data_report.py``)."""
    q = item.question
    menu = json.dumps([q.labels, q.descriptions], ensure_ascii=False)
    target = json.dumps([round(p, 4) for p in item.target])
    return stable_hash(normalized(item.state), normalized(q.instructions), q.type, menu, target)


def instructions_hash(item: Item) -> int:
    return stable_hash(normalized(item.question.instructions))


@dataclass(frozen=True)
class HeldIndex:
    """Every held-out row: exact (state, instructions) keys, normalised states, state
    signatures and instruction hashes."""

    keys: frozenset[str]
    sigs: np.ndarray
    instructions: np.ndarray
    states: frozenset[str] = frozenset()

    @classmethod
    def of(cls, items: Iterable[Item], hasher: MinHasher | None = None) -> HeldIndex:
        hasher = hasher or MinHasher()
        items = list(items)
        return cls(
            keys=frozenset(si_key(it) for it in items),
            sigs=signatures(items, hasher),
            instructions=np.asarray([instructions_hash(it) for it in items], np.uint64),
            states=frozenset(normalized(it.state) for it in items),
        )


def signatures(items: list[Item], hasher: MinHasher) -> np.ndarray:
    if not items:
        return np.zeros((0, hasher.num_perm), np.uint32)
    return np.stack([hasher.signature(shingle_hashes(it.state)) for it in items])


def too_long(
    items: list[Item], count_tokens: TokenCounter | None, limit: int = MAX_PROMPT_TOKENS
) -> np.ndarray:
    """Rows whose eval render (all options) exceeds ``limit`` tokens in either layout. A
    byte-level BPE token is at least one byte, so prompts of ≤ ``limit`` UTF-8 bytes are
    never tokenized."""
    out = np.zeros(len(items), bool)
    if count_tokens is None:
        return out
    for i, item in enumerate(items):
        texts = [render(item, spec)[0] for spec in EVAL_LAYOUTS]
        long = [t for t in texts if len(t.encode()) > limit]
        if long and max(count_tokens(long)) > limit:
            out[i] = True
    return out


def reasons(
    items: list[Item],
    held: HeldIndex,
    *,
    sigs: np.ndarray,
    count_tokens: TokenCounter | None = None,
    limit: int = MAX_PROMPT_TOKENS,
    passages: bool = False,
    grams: JevBenchGrams | None = None,
) -> list[str | None]:
    """Each row's first drop reason among the held-out and length checks (``None``: kept
    so far; the exact-duplicate test is the caller's)."""
    equal = np.asarray([si_key(it) in held.keys for it in items], bool)
    near = near_hits(
        sigs,
        held.sigs,
        query_key=np.asarray([instructions_hash(it) for it in items], np.uint64),
        index_key=held.instructions,
    )
    off = np.zeros(len(items), bool)
    state_equal = (
        np.asarray([normalized(it.state) in held.states for it in items], bool) if passages else off
    )
    state_near = near_hits(sigs, held.sigs) if passages else off
    overlap = np.asarray([grams.overlaps(it) for it in items], bool) if grams else off
    long = too_long(items, count_tokens, limit)
    out: list[str | None] = []
    for eq, nr, se, sn, ov, lg in zip(equal, near, state_equal, state_near, overlap, long,
                                      strict=True):  # fmt: skip
        if eq:
            out.append("train_equals_heldout")
        elif nr:
            out.append("train_near_heldout")
        elif se:
            out.append("train_state_in_heldout")
        elif sn:
            out.append("train_state_near_heldout")
        elif ov:
            out.append("jevbench_8gram")
        elif lg:
            out.append("prompt_too_long")
        else:
            out.append(None)
    return out


def clean(
    items: list[Item],
    held: HeldIndex,
    seen: set[int],
    *,
    count_tokens: TokenCounter | None = None,
    hasher: MinHasher | None = None,
    limit: int = MAX_PROMPT_TOKENS,
    passages: bool = False,
    grams: JevBenchGrams | None = None,
) -> tuple[list[Item], Counter]:
    """One source's train rows → the kept rows and drop counts. ``seen`` holds the exact
    keys of the train rows kept so far (earlier sources), and is updated. ``passages``
    and ``grams``: the mix-v6 checks (held-out passages, JevBench 8-grams)."""
    hasher = hasher or MinHasher()
    drops: Counter = Counter()
    why = reasons(items, held, sigs=signatures(items, hasher), count_tokens=count_tokens,
                  limit=limit, passages=passages, grams=grams)  # fmt: skip
    kept = []
    for item, reason in zip(items, why, strict=True):
        if reason is not None:
            drops[reason] += 1
            continue
        key = exact_key(item)
        if key in seen:
            drops["train_exact_duplicate"] += 1
            continue
        seen.add(key)
        kept.append(item)
    return kept, drops


def find_tokenizer(root: str | Path | None = None) -> Path | None:
    """The Qwen3.5 tokenizer folder under ``root``/models (default ~/tjev-work): every
    Qwen3.5 size shares it."""
    models = Path(root or "~/tjev-work").expanduser() / "models"
    preferred = models / "Qwen3.5-0.8B"
    if (preferred / "tokenizer.json").exists():
        return preferred
    found = sorted(models.glob("Qwen3.5*/tokenizer.json"))
    return found[0].parent if found else None


def token_counter(folder: str | Path) -> TokenCounter:
    from tjev.data.tokenize import PromptTokenizer

    tok = PromptTokenizer(folder)

    def count(texts: list[str]) -> list[int]:
        return [len(e.ids) for e in tok.tokenizer.encode_batch(texts, add_special_tokens=False)]

    return count


# --------------------------------------------------------------------------------------
# mix-v7: the post-pass of the new sources (tjev.data.mix.extend)


def question_hash(item: Item) -> int:
    """Normalised instructions + type + menu: two rows that ask the same question."""
    q = item.question
    menu = json.dumps([q.labels, q.descriptions], ensure_ascii=False)
    return stable_hash(normalized(q.instructions), q.type, normalized(menu))


class TrainIndex:
    """The kept train rows of every source so far, grown source by source: two LSH
    indexes of their state signatures (:class:`tjev.data.near.BandIndex`), one tagged
    with each row's origin (upstream dataset), one keyed by its question hash, plus the
    normalised states of each origin."""

    def __init__(self, num_perm: int = 128):
        self.by_state = BandIndex(num_perm)
        self.by_question = BandIndex(num_perm, keyed=True)
        self.states: dict[int, set[str]] = {}
        self.origin_ids: dict[str, int] = {}

    @classmethod
    def empty(cls) -> TrainIndex:
        return cls()

    def origin_id(self, origin: str) -> int:
        return self.origin_ids.setdefault(origin, len(self.origin_ids))

    def add(self, items: list[Item], sigs: np.ndarray, origin: str) -> None:
        if not items:
            return
        oid = self.origin_id(origin)
        self.by_state.add(sigs, tag=oid)
        self.by_question.add(sigs, np.asarray([question_hash(it) for it in items], np.uint64))
        self.states.setdefault(oid, set()).update(normalized(it.state) for it in items)

    def state_elsewhere(self, state: str, oid: int) -> bool:
        return any(state in s for o, s in self.states.items() if o != oid)

    def __len__(self) -> int:
        return len(self.by_state)


def clean_v7(
    items: list[Item],
    held: HeldIndex,
    seen: set[int],
    train: TrainIndex,
    origin: str,
    *,
    count_tokens: TokenCounter | None = None,
    hasher: MinHasher | None = None,
    limit: int = MAX_PROMPT_TOKENS,
    grams: JevBenchGrams | None = None,
    refs=None,
    common: np.ndarray | None = None,
    index_only: bool = False,
    near_train: bool = True,
) -> tuple[list[Item], Counter]:
    """The mix-v7 post-pass of one new source: every mix-v6 check (:func:`reasons`, with
    the held-out passages and the public JevBench 8-grams), then, first reason wins:

    ``decontam_8gram:<set>``
        a word 8-gram shared with an evaluation-only set or with our held-out rows
        (``refs``: :class:`tjev.data.decontam.RefGrams`; against our held-out rows,
        a passage: ``mix.HELDOUT_MIN_SHARED`` grams or ``HELDOUT_MIN_SHARE`` of the
        row's; the held-out rows of the row's own origin do not count). The source's own
        train templates (8-grams in ≥ 0.05% and ≥ 20 of its
        rows) never count: generator boilerplate is not contamination;
    ``train_exact_duplicate``
        as in :func:`clean`, against every kept train row (mix-v6's included);
    ``train_near_duplicate``
        a state near-duplicate (MinHash, J ≥ 0.8) of a kept train row of any source that
        asks the same question (instructions, type and menu);
    ``train_cross_source_state``
        a state equal or near to a kept train row of *another* upstream dataset
        (``origin``), whatever the question: one passage in two sources (e.g. a tasksource
        recast of a dataset the mix already has). Rows of one origin may share a state
        (one passage, several questions).

    Kept rows are added to ``train`` and their exact keys to ``seen``. ``index_only``: the
    rows are only indexed (the mix-v6 train files, kept as they are). ``near_train`` False
    skips the two near checks against train rows (mix-v7.5 jev_distractor: a real row plus
    one opinion line, near its base row by design); held-out and exact checks stay."""
    hasher = hasher or MinHasher()
    sigs = signatures(items, hasher)
    if index_only:
        seen.update(exact_key(it) for it in items)
        train.add(items, sigs, origin)
        return items, Counter()
    why = reasons(items, held, sigs=sigs, count_tokens=count_tokens, limit=limit,
                  passages=True, grams=grams)  # fmt: skip
    drops: Counter = Counter(r for r in why if r is not None)
    idx = [i for i, r in enumerate(why) if r is None]
    items, sigs = [items[i] for i in idx], sigs[idx] if idx else sigs[:0]
    oid = train.origin_id(origin)
    if items and near_train:
        questions = np.asarray([question_hash(it) for it in items], np.uint64)
        same_q = train.by_question.query(sigs, questions)
        cross = train.by_state.query(sigs, exclude_tag=oid, per_group=32)
    else:
        same_q = cross = np.zeros(0, bool)
    grams_of, ignore = [], None
    if refs is not None:
        from tjev.data.decontam import gram_hashes, train_templates
        from tjev.data.jevbench import item_text

        grams_of = [gram_hashes(item_text(it)) for it in items]
        templates = train_templates(grams_of)  # the source's own frames
        if common is not None:  # phrases common to the new train rows of every source
            templates = np.union1d(templates, common)
        ignore = dict.fromkeys(refs.grams, templates)
        # our held-out rows of the same upstream dataset: passage reuse within a dataset is
        # the group splits' and the near-duplicate checks' job; shared boilerplate (legal
        # visas, tool descriptions) is not a leak
        for name, groups in refs.groups.items():
            if origin in groups:
                ignore[name] = np.union1d(templates, groups[origin])
    kept, kept_rows = [], []
    for k, item in enumerate(items):
        ref = refs.hit("", grams=grams_of[k], ignore=ignore) if refs is not None else None
        key = exact_key(item)
        if ref is not None:
            drops[f"decontam_8gram:{ref}"] += 1
        elif key in seen:
            drops["train_exact_duplicate"] += 1
        elif near_train and same_q[k]:
            drops["train_near_duplicate"] += 1
        elif near_train and (cross[k] or train.state_elsewhere(normalized(item.state), oid)):
            drops["train_cross_source_state"] += 1
        else:
            seen.add(key)
            kept.append(item)
            kept_rows.append(k)
    train.add(kept, sigs[kept_rows] if kept_rows else sigs[:0], origin)
    return kept, drops
