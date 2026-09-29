"""Held-out generator items: one set selects checkpoints, another one reports on them.

Every code-labelled generator with ``split="heldout"`` (held-out surface text and parameter
ranges), half French. The selection set (``data.heldout``, part of the selection score) and
the reporting set (post-training evaluation) use different seeds, so the items used to
*choose* a checkpoint are never the ones it is *reported* on.
"""

from __future__ import annotations

from pathlib import Path

from tjev.data.item import parse_item, write_jsonl
from tjev.data.sources import GENERATORS, generate

SELECT_SEED = 777
REPORT_SEED = 4242


def heldout_rows(per: int = 20, seed: int = SELECT_SEED, fr_share: float = 0.5) -> list[dict]:
    return [
        row
        for name in GENERATORS
        for row in generate(name, per, seed=seed, split="heldout", fr_share=fr_share)
    ]


def write_heldout(out: str | Path, per: int = 20, seed: int = SELECT_SEED) -> int:
    return write_jsonl(out, [parse_item(r) for r in heldout_rows(per, seed)])
