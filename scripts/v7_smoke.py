"""Smoke-test mix-v7 source adapters on the real data (run on Colab CPU: it downloads).

    python scripts/v7_smoke.py --sources sharc,tram [--limit 2000] [--show 2]

For each source: loads the train split (and the eval split, or the group split), adapts
the first ``--limit`` rows of a seeded permutation, and prints the row counts, the drop
reasons, the question types, the gold-label balance, the soft-target share, the licence
uses and ``--show`` rendered sample items (truncated). Exits non-zero when a source
fails to load or yields no valid item.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

import numpy as np


def main() -> int:
    from tjev.data.item import parse_item
    from tjev.data.mix import adapt, block_of_source, label_names, load
    from tjev.data.render import RenderSpec, render
    from tjev.data.sources import ALL_SOURCES

    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--chars", type=int, default=1200)
    args = ap.parse_args()
    failed = []
    for name in [s for s in args.sources.split(",") if s]:
        source = ALL_SOURCES[name]
        t0 = time.time()
        print(f"\n===== {name} ({source.repo}, block {block_of_source(name)}) =====", flush=True)
        try:
            splits = [source.train_split] + ([source.eval_split] if source.eval_split else [])
            for split in splits:
                ds = load(source, split)
                meta = {"label_names": label_names(ds)}
                idx = np.random.default_rng(0).permutation(len(ds))[: args.limit]
                rows, drops = adapt(source, ds, idx, 0, meta, split)
                items = [parse_item(r) for r in rows]
                kinds = Counter(it.question.type for it in items)
                gold = Counter(
                    (it.question.type, it.question.labels[it.gold]) for it in items
                ).most_common(12)
                soft = sum(max(it.target) < 1.0 for it in items)
                uses = Counter(it.license_use for it in items)
                langs = Counter(it.lang for it in items)
                fams = Counter(it.family for it in items).most_common(8)
                nopt = Counter(len(it.question.labels) for it in items).most_common(6)
                chars = sorted(len(it.state) for it in items) or [0]
                print(
                    f"[{split}] dataset rows {len(ds)}, adapted {len(idx)} → items {len(items)}"
                    f" ({time.time() - t0:.0f}s)\n  drops {dict(drops)}\n  types {dict(kinds)}"
                    f" soft {soft}\n  gold {gold}\n  options {nopt}\n  langs {dict(langs)}"
                    f" uses {dict(uses)}\n  families {fams}\n  state chars p50 "
                    f"{chars[len(chars) // 2]} p99 {chars[int(len(chars) * 0.99)]}",
                    flush=True,
                )
                if not items:
                    failed.append(f"{name}:{split}: no item")
                for it in items[: args.show]:
                    text = render(it, RenderSpec(train=False, layout="plain"))[0]
                    tgt = {
                        lab: round(p, 3)
                        for lab, p in zip(it.question.labels, it.target, strict=True)
                        if p
                    }
                    print(f"  --- sample {it.id} target {tgt}\n{text[-args.chars :]}", flush=True)
        except Exception as e:
            print(f"  FAILED {name}: {type(e).__name__}: {str(e)[:500]}", flush=True)
            failed.append(name)
    print(f"\n[smoke] failed: {failed or 'none'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
