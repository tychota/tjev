"""Attach the teacher distributions to a built mix's plumb / decider_teacher rows.

Mixes built before ``Item.teacher`` (mix-v6 and older) dropped plumb's ``teacher_probs`` and
decider's ``teacher_p``. This recomputes them from the raw sources with the current adapters
(rows are matched by id, ``<source>-<split>-<row>-<question>``, and checked against the
mix's state and question), counts the rows that carry a teacher and those that pass the
gate of ``loss.teacher_gate`` (the teacher puts ≥ gate on a non-gold option), and with
``--out`` writes annotated copies of the two train files (every other field unchanged).

    python scripts/teacher_annotate.py MIX/train [--out OUT/train] [--gate 0.6]

It also counts the decider questions the adapter drops for teacher disagreement
(``teacher_ok`` false, or a custom question with ``teacher_p`` < 0.5), and how many of
them would pass the gate: the rows the gated-teacher target is about.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tjev.data.item import parse_item
from tjev.data.sources.contenders import (
    _labels,
    decider,
    decider_load,
    decider_teacher_dist,
    plumb,
    plumb_load,
)


def passes(item, gate: float) -> bool:
    if item.teacher is None:
        return False
    return any(p >= gate for i, p in enumerate(item.teacher) if i != item.gold)


def annotate(path: Path, adapter, load, gate: float, out: Path | None) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    raw = None
    counts: Counter = Counter(rows=len(rows))
    annotated = []
    for row in rows:
        _, _, i, j = row["id"].rsplit("-", 3)
        if raw is None:  # train rows (plumb: its train split; decider: a group split of it)
            raw = load("train")
        produced = adapter(raw[int(i)], None, {})
        fresh = produced[int(j)] if int(j) < len(produced) else None
        same = (
            fresh is not None
            and parse_item({**fresh, "id": row["id"]}).state == parse_item(row).state
            and fresh["question"]["instructions"] == row["question"]["instructions"]
        )
        if not same:
            counts["unmatched"] += 1
        elif "teacher" in fresh:
            row = {**row, "teacher": fresh["teacher"]}
        item = parse_item(row)
        if item.teacher is not None:
            counts["teacher"] += 1
            counts[f"teacher/{item.question.type}"] += 1
            if passes(item, gate):
                counts["gate"] += 1
                counts[f"gate/{item.question.type}"] += 1
        annotated.append(row)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        with (out / path.name).open("w", encoding="utf-8") as f:
            for row in annotated:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dict(counts)


def dropped_decider(gate: float) -> dict:
    """The decider questions the adapter drops for teacher disagreement, and those whose
    teacher distribution (as the adapter would build it) passes the gate."""
    counts: Counter = Counter()
    for row in decider_load("train"):
        for q in json.loads(row["questions"]):
            p = q.get("teacher_p")
            dropped = q.get("teacher_ok") is False or (
                q.get("teacher_ok") is None and p is not None and p < 0.5
            )
            if not dropped:
                continue
            counts["dropped"] += 1
            kind, answer = q["type"], q["answer"]
            if kind == "noul":
                gold = "yes" if answer is True or str(answer).lower() == "true" else "no"
                criteria = None
            else:
                gold = str(int(answer)) if kind == "score" else str(answer)
                criteria = q.get("criteria")
            dist = decider_teacher_dist(
                kind, _labels(kind, criteria), gold, p, q.get("teacher_pred")
            )
            if dist and any(v >= gate for k, v in dist.items() if k != gold):
                counts["dropped_gate"] += 1
                counts[f"dropped_gate/{kind}"] += 1
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("train", type=Path, help="the mix's train/ directory")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gate", type=float, default=0.6)
    args = ap.parse_args()
    report = {
        "plumb": annotate(args.train / "plumb.jsonl", plumb, plumb_load, args.gate, args.out),
        "decider_teacher": annotate(
            args.train / "decider_teacher.jsonl", decider, decider_load, args.gate, args.out
        ),
        "decider_dropped": dropped_decider(args.gate),
        "gate": args.gate,
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
