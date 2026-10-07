"""Build the mix-v7 generator sources at their planned sizes and report on them.

    JAX_PLATFORMS=cpu uv run python scripts/gen7_report.py [--scale 1.0] [--out DIR]

For each v7 source: the planned share of the generators block, the rows that give at most
one epoch at that share (5,000 steps, ≈ 817,850 segments, block 14%), the rows generated
(train and the builder's three held-out parts), the time it took, the label balance, the
French share, the length, and the train / held-out key overlap. Then the diversity
comparison with the v6 generators (``gen_v7.diversity``). ``--out`` writes the rows
(train/<source>.jsonl, heldout/<source>.jsonl) and report.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from tjev.data.item import parse_item
from tjev.data.sources.gen_v7 import V7_GENERATORS, generate_v7, key_of, source_info
from tjev.data.sources.gen_v7.diversity import compare

SEGMENTS = 817_850  # 5,000 steps × 65,536 tokens at the v6 length mix (mix-v7 plan)
BLOCK = 0.14
# share of the generators block (the rest: real-text seeds, the real-row distractors and
# the v6 generators; see docs/data.md, "Generators v7")
PLAN = {
    "gen7_return_window_pairs": 0.11,
    "gen7_weekday_pairs": 0.06,
    "gen7_schedule_pairs": 0.07,
    "gen7_invoice_budget_pairs": 0.07,
    "gen7_refund_pairs": 0.08,
    "gen7_long_refund_pairs": 0.03,
    "gen7_unknowable": 0.06,
    "gen7_aux_quantities": 0.032,
    "gen7_posterior": 0.10,
    "gen7_distractor_opinion": 0.05,
}
REALTEXT_SHARE = 0.15  # split evenly over the real-text sources that are registered
HEADROOM = 1.05  # the hygiene post-pass drops a few rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    realtext = [n for n in V7_GENERATORS if n not in PLAN]
    plan = dict(PLAN)
    for n in realtext:
        plan[n] = REALTEXT_SHARE / max(1, len(realtext))
    report = {"segments": SEGMENTS, "block": BLOCK, "sources": {}}
    for name, share in plan.items():
        segs = SEGMENTS * BLOCK * share
        want = int(segs * HEADROOM * args.scale)
        t = time.time()
        try:
            train = generate_v7(name, want, args.seed)
        except Exception as e:  # a real-text source without its seed cache
            report["sources"][name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{name}: FAILED {e}", flush=True)
            continue
        took = time.time() - t
        held = [r for k in (101, 202, 303) for r in generate_v7(name, 150, args.seed + k,
                                                                 split="heldout")]  # fmt: skip
        for r in train[:200] + held[:50]:
            parse_item(r)
        gold = Counter(r.get("expected") or max(r["target"], key=r["target"].get) for r in train)
        soft = sum("target" in r for r in train)
        chars = np.array([len(r["state"]) for r in train])
        overlap = len({key_of(r) for r in train} & {key_of(r) for r in held})
        info = {
            "share_of_block": round(share, 4),
            "segments": round(segs),
            "rows_wanted": want,
            "rows": len(train),
            "epochs": round(segs / max(1, len(train)), 3),
            "heldout_rows": len(held),
            "seconds": round(took, 1),
            "rows_per_s": round(len(train) / max(took, 1e-9)),
            "fr_share": round(sum(r["lang"] == "fr" for r in train) / max(1, len(train)), 3),
            "gold": dict(gold.most_common(8)),
            "soft_share": round(soft / max(1, len(train)), 3),
            "chars_p50_p95": [int(np.median(chars)), int(np.percentile(chars, 95))],
            "train_heldout_key_overlap": overlap,
            **source_info(name),
        }
        report["sources"][name] = info
        print(name, json.dumps(info, ensure_ascii=False), flush=True)
        if args.out:
            for part, rows in (("train", train), ("heldout", held)):
                path = args.out / part / f"{name}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["diversity"] = compare(1000)
    for row in report["diversity"]:
        print("diversity", json.dumps(row), flush=True)
    if args.out:
        (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
