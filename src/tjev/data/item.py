"""The decision item: JevBench-native, fail-closed.

JSONL row::

    {"id": "...", "state": "...",
     "question": {"type": "noul|choice|score", "instructions": "...",
                  "criteria": {"label": "description", ...}},   # noul: {"false": .., "true": ..}
     "labels": ["..."],                  # optional for choice/score if criteria given
     "expected": "label",                # or "target": {"label": prob} (soft), or "gold_probs"
     "source": "...", "family": "...", "lang": "en|fr|..."}

Noul questions use labels ``["no", "yes"]``; score questions ``["0".."n"]`` in order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

QuestionType = Literal["noul", "choice", "score"]
TYPES: tuple[QuestionType, ...] = ("noul", "choice", "score")
NOUL_LABELS = ("no", "yes")


@dataclass(frozen=True)
class Question:
    type: QuestionType
    instructions: str
    labels: tuple[str, ...]
    descriptions: tuple[str, ...]  # aligned with labels ("" when none)


@dataclass(frozen=True)
class Item:
    id: str
    state: str
    question: Question
    target: tuple[float, ...]  # aligned with question.labels, sums to 1
    source: str = "unknown"
    family: str = "unknown"
    lang: str = "en"
    positives: tuple[int, ...] = field(default=())  # labels with target mass (never dropped)

    @property
    def gold(self) -> int:
        return max(range(len(self.target)), key=self.target.__getitem__)


def _distribution(raw: dict, labels: tuple[str, ...], where: str) -> tuple[float, ...]:
    unknown = set(raw) - set(labels)
    if unknown:
        raise ValueError(f"{where}: probabilities for unknown labels {sorted(unknown)}")
    values = [float(raw.get(label, 0.0)) for label in labels]
    if any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError(f"{where}: invalid probability")
    total = sum(values)
    if not 0.98 <= total <= 1.02:
        raise ValueError(f"{where}: probabilities sum to {total}")
    return tuple(v / total for v in values)


def parse_item(raw: dict, where: str = "item") -> Item:
    """One JSONL row → :class:`Item`; malformed rows raise ``ValueError`` / ``KeyError``."""
    q = raw["question"]
    kind = q["type"]
    if kind not in TYPES:
        raise ValueError(f"{where}: unknown question type {kind!r}")
    criteria = q.get("criteria") or {}
    if kind == "noul":
        labels = NOUL_LABELS
        if isinstance(criteria, dict) and set(criteria) <= {"false", "true", "no", "yes"}:
            descriptions = (
                criteria.get("false", criteria.get("no", "")),
                criteria.get("true", criteria.get("yes", "")),
            )
        else:
            raise ValueError(f"{where}: noul criteria must be {{false,true}}")
        if raw.get("labels") and tuple(raw["labels"]) not in (NOUL_LABELS, NOUL_LABELS[::-1]):
            raise ValueError(f"{where}: noul labels must be no/yes")
    else:
        if isinstance(criteria, list):  # score levels as a list of descriptors
            labels = tuple(raw.get("labels") or [str(i) for i in range(len(criteria))])
            descriptions = tuple(str(c) for c in criteria)
        else:
            labels = tuple(raw.get("labels") or list(criteria))
            descriptions = tuple(str(criteria.get(label, "")) for label in labels)
        if len(labels) != len(descriptions):
            raise ValueError(f"{where}: labels and criteria differ in length")
        if kind == "score" and labels != tuple(str(i) for i in range(len(labels))):
            raise ValueError(f"{where}: score labels must be '0'..'n' in order")
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError(f"{where}: need ≥2 distinct labels")
    if "target" in raw or "gold_probs" in raw:
        target = _distribution(raw.get("target") or raw["gold_probs"], labels, where)
    else:
        expected = raw["expected"]
        if isinstance(expected, bool):
            expected = "yes" if expected else "no"
        expected = str(expected)
        if expected not in labels:
            raise ValueError(f"{where}: expected {expected!r} not in labels")
        target = tuple(1.0 if label == expected else 0.0 for label in labels)
    state = raw["state"]
    if isinstance(state, (dict, list)) and state:
        # Structured states (JevBench hard tier) are rendered as indented JSON text.
        state = json.dumps(state, ensure_ascii=False, indent=1)
    if not isinstance(state, str) or not state.strip():
        raise ValueError(f"{where}: empty state")
    return Item(
        id=str(raw.get("id", where)),
        state=state,
        question=Question(kind, str(q.get("instructions", "")), labels, descriptions),
        target=target,
        source=str(raw.get("source", "unknown")),
        family=str(raw.get("family", "unknown")),
        lang=str(raw.get("lang", "en")),
        positives=tuple(i for i, p in enumerate(target) if p > 0),
    )


def item_to_json(item: Item) -> dict:
    q = item.question
    if q.type == "noul":
        criteria: dict | list = {"false": q.descriptions[0], "true": q.descriptions[1]}
    elif q.type == "score":
        criteria = list(q.descriptions)
    else:
        criteria = dict(zip(q.labels, q.descriptions, strict=True))
    out = {
        "id": item.id,
        "state": item.state,
        "question": {"type": q.type, "instructions": q.instructions, "criteria": criteria},
        "labels": list(q.labels),
        "source": item.source,
        "family": item.family,
        "lang": item.lang,
    }
    if max(item.target) == 1.0:
        out["expected"] = q.labels[item.gold]
    else:
        out["target"] = dict(zip(q.labels, item.target, strict=True))
    return out


def read_jsonl(path: str | Path) -> Iterator[Item]:
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                yield parse_item(json.loads(line), f"{path}:{n}")


def write_jsonl(path: str | Path, items: Iterable[Item]) -> int:
    """Atomic write (temporary file, then rename); returns the number of rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item_to_json(item), ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count
