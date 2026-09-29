"""JevBench public items: import for final measurement, and the contamination filter.

The public split (231 items: easy 48, original 72, hard 111) is for *final measurement
only*: never part of a training mixture, a validation set or a calibration fit (JevBench
README; sealed items are never used at all). :class:`JevBenchFilter` drops training rows
that overlap a public item: an equal normalised state, an equal non-generic instruction,
or more than two shared word 8-grams with a public state.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from tjev.data.item import Item, parse_item, read_jsonl, rendered_state, write_jsonl

TIERS = ("easy", "original", "hard")


def load_public(root: str | Path) -> list[Item]:
    root = Path(root)
    items = []
    for tier in TIERS:
        path = root / "datasets" / "public" / f"{tier}.jsonl"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            row = {
                "id": raw["id"],
                "state": raw["state"],
                "question": raw["question"],
                "labels": raw.get("labels"),
                "expected": raw["expected"],
                "source": f"jevbench_{tier}",
                "family": raw.get("family", "unknown"),
                "lang": "en",
            }
            if raw["question"]["type"] == "noul":
                row.pop("labels")  # JevBench noul labels are no/yes; ours are fixed
            items.append(parse_item(row, f"{path}:{n}"))
    return items


def prepare(repo: str | Path, out: str | Path) -> dict[str, int]:
    """A JevBench checkout → ``out/public.jsonl`` plus one file per tier; counts per tier."""
    items = load_public(repo)
    out = Path(out)
    write_jsonl(out / "public.jsonl", items)
    counts = {}
    for tier in TIERS:
        counts[tier] = write_jsonl(
            out / f"{tier}.jsonl", [i for i in items if i.source == f"jevbench_{tier}"]
        )
    return counts


GENERIC_INSTRUCTION_CHARS = 60
GENERIC_INSTRUCTION_ROWS = 20
MAX_SHARED_8GRAMS = 2


def norm_text(text: str) -> str:
    """NFKC, lower case, collapsed whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def word_ngrams(text: str, n: int = 8) -> set[str]:
    words = re.findall(r"\w+", norm_text(text))
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


class JevBenchFilter:
    """Overlap with the public JevBench items (``public`` = (state, instructions) pairs)."""

    def __init__(self, public: list[tuple[str, str]]):
        self.states = {norm_text(s) for s, _ in public}
        self.instructions = {norm_text(i) for _, i in public if i.strip()}
        self.grams: dict[str, set[int]] = defaultdict(set)
        for k, (state, _) in enumerate(public):
            for g in word_ngrams(state):
                self.grams[g].add(k)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> JevBenchFilter:
        return cls([(i.state, i.question.instructions) for i in read_jsonl(path)])

    def reason(self, state: str, instructions: str, generic: bool = False) -> str | None:
        """Why a row overlaps (``state``, ``instructions``, ``8gram``) or None."""
        if norm_text(state) in self.states:
            return "state"
        if not generic and norm_text(instructions) in self.instructions:
            return "instructions"
        shared = Counter()
        for g in word_ngrams(state):
            for k in self.grams.get(g, ()):
                shared[k] += 1
        if shared and max(shared.values()) > MAX_SHARED_8GRAMS:
            return "8gram"
        return None

    def filter(self, rows: list[dict]) -> tuple[list[dict], Counter]:
        """Rows without overlap, and drop counts by reason. An instruction is generic when
        shorter than 60 characters or shared by ≥20 of ``rows`` (a source's training rows)."""
        counts = Counter(norm_text(r["question"]["instructions"]) for r in rows)
        kept, drops = [], Counter()
        for r in rows:
            instructions = r["question"]["instructions"]
            key = norm_text(instructions)
            generic = (
                len(key) < GENERIC_INSTRUCTION_CHARS or counts[key] >= GENERIC_INSTRUCTION_ROWS
            )
            why = self.reason(str(rendered_state(r["state"])), instructions, generic)
            if why is None:
                kept.append(r)
            else:
                drops["jevbench_overlap"] += 1
                drops[f"jevbench_overlap:{why}"] += 1
        return kept, drops
