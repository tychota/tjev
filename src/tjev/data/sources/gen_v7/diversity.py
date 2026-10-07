"""A small surface-diversity metric for generator output (mix-v7 vs the v6 generators).

On ``n`` rows of a source, numbers masked (``#``) so that only the wording counts:

- ``lines``: distinct masked lines (state lines and the question) per 1,000 rows, the
  template count as the reader sees it;
- ``distinct3``: distinct word 3-grams / all word 3-grams (distinct-n), on states and
  questions;
- ``questions``: distinct masked questions.

``compare`` pairs each v7 source with its v6 counterpart (``V6_OF``), on ``n`` states each
(the first member of each v7 pair: its twin repeats it but for one fact).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

WORDS = re.compile(r"[^\W\d_]+|#")

V6_OF = {
    "gen7_return_window_pairs": "gen_return_window",
    "gen7_invoice_budget_pairs": "gen_invoice_total",
    "gen7_schedule_pairs": "gen_schedule_conflict",
    "gen7_weekday_pairs": "gen_weekday",
    "gen7_refund_pairs": "gen_refund_policy",
    "gen7_long_refund_pairs": "gen_long_refund_policy",
    "gen7_posterior": "gen_base_rate",
}


def _mask(text: str) -> str:
    return re.sub(r"\d+([.,]\d+)?", "#", text.lower())


def metrics(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    lines, questions, grams, total = set(), set(), set(), 0
    for r in rows:
        state = r["state"] if isinstance(r["state"], str) else str(r["state"])
        q = _mask(r["question"]["instructions"])
        questions.add(q)
        for line in re.split(r"\n|(?<=[.!?])\s+", state):
            if line.strip():
                lines.add(_mask(line.strip()))
        lines.add(q)
        words = WORDS.findall(_mask(state) + " " + q)
        tri = list(zip(words, words[1:], words[2:], strict=False))
        grams.update(tri)
        total += len(tri)
    n = max(1, len(rows))
    return {
        "rows": len(rows),
        "lines_per_1k": round(1000 * len(lines) / n, 1),
        "distinct3": round(len(grams) / max(1, total), 4),
        "questions": len(questions),
    }


def compare(n: int = 1000, seed: int = 0, fr_share: float = 0.3) -> list[dict]:
    from tjev.data.sources.generators import generate

    out = []
    for v7, v6 in V6_OF.items():
        # one state per group: a pair's second member repeats its first one but one fact
        rows7 = [r for r in generate(v7, 2 * n, seed, fr_share=fr_share)
                 if r["meta"]["role"] == "a"][:n]  # fmt: skip
        m7 = metrics(rows7)
        m6 = metrics(generate(v6, n, seed, fr_share=fr_share))
        out.append({"v7": v7, "v6": v6, **{f"v7_{k}": v for k, v in m7.items()},
                    **{f"v6_{k}": v for k, v in m6.items()}})  # fmt: skip
    return out
