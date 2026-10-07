"""Offline packing report: the real training stream of a mix under each ``train.packing``.

    python scripts/packing_report.py MIX_DIR --tokenizer models/Qwen3.5-0.8B \
        --steps 1500 --microbatch 4096 --out packing.json

Segment ``i`` is the run's own (tjev.data.segments.SegmentAt: the mixture, render and
tokenizer of ``tjev train`` with ``train.seed``), computed once (in ``--workers``
processes) and reduced to what the packer reads: tokens, labels and source. Every packing
then runs the real :class:`tjev.data.pack.BucketPacker` on that same sequence for
``--steps`` steps per microbatch budget (one device: the v6e1 plan's microbatch per size,
4096 for 4B), and reports per bucket the pad fraction, steps and slots, the per-source
token share of what was trained, and the real tokens per step. It also checks, on the
real lengths, that a packing is deterministic (two runs, the same steps) and resumes
exactly (a JSON round trip of the state mid-run, then the same steps).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from tjev.data.item import read_jsonl
from tjev.data.mixture import Mixture
from tjev.data.pack import PACKINGS, BucketPacker, Segment, Vocab
from tjev.data.segments import SegmentAt
from tjev.data.tokenize import PromptTokenizer

_AT: SegmentAt | None = None


def _init(at: SegmentAt) -> None:
    global _AT
    _AT = at


def _lengths(span: tuple[int, int]) -> np.ndarray:
    assert _AT is not None
    out = np.zeros((span[1] - span[0], 3), np.int32)
    for k, i in enumerate(range(*span)):
        seg = _AT(i)
        out[k] = (len(seg.tokens), len(seg.label_ids), seg.source_id)
    return out


def load_mix(mix: Path) -> tuple[dict, dict, float]:
    raw = yaml.safe_load((mix / "mix.yaml").read_text(encoding="utf-8"))["data"]
    files = [mix / "train" / Path(p).name for p in raw["train"]]  # mix.yaml paths are the host's
    sources: dict[str, list] = defaultdict(list)
    for path in files:  # tjev.train.loop.load_sources: file order, then row order
        for item in read_jsonl(path):
            sources[item.source].append(item)
    return dict(sources), raw.get("mixture") or {}, float(raw.get("mixture_alpha", 0.5))


def segment_table(at: SegmentAt, count: int, workers: int, chunk: int = 2000) -> np.ndarray:
    spans = [(a, min(a + chunk, count)) for a in range(0, count, chunk)]
    if workers <= 1:
        _init(at)
        return np.concatenate([_lengths(s) for s in spans])
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(at,)) as pool:
        return np.concatenate(pool.map(_lengths, spans))


class LengthStream:
    """Length-only segments in stream order (the packer reads tokens, labels, index)."""

    def __init__(self, table: np.ndarray):
        self.table = table

    def __call__(self, i: int) -> Segment:
        if i >= len(self.table):
            raise IndexError(f"segment {i}: the table has {len(self.table)}; raise --segments")
        n, k, s = (int(x) for x in self.table[i])
        return Segment(tokens=np.empty(n, np.int8), label_ids=np.empty(k, np.int8), target=None,
                       type_id=0, source_id=s, lang_id=0, family_id=0, index=i)  # fmt: skip


def run(stream: LengthStream, steps: int, *, packing: str, buckets, microbatch: int,
        tokens_per_step: int, bins: int, slots: int, labels: int,
        resume_at: int | None = None) -> dict:  # fmt: skip
    """``steps`` steps of the packer; the digest of their segment layout; with
    ``resume_at``, the packer is rebuilt from its JSON state after that many steps."""
    accumulation = max(1, tokens_per_step // microbatch)
    kw = {"buckets": buckets, "microbatch_tokens": microbatch, "accumulation": accumulation,
              "slots": slots, "labels": labels, "bins": bins, "packing": packing}  # fmt: skip
    packer = BucketPacker(stream, **kw)
    digest = hashlib.sha256()
    per_bucket: dict[int, dict] = defaultdict(lambda: defaultdict(float))
    sources = int(stream.table[:, 2].max()) + 1
    src_tokens = np.zeros(sources, np.int64)
    src_segments = np.zeros(sources, np.int64)
    for step in range(steps):
        if resume_at is not None and step == resume_at:
            state = json.loads(json.dumps(packer.state()))
            packer = BucketPacker(stream, **kw)
            packer.restore(state)
        while True:
            group = None
            for length in packer.buckets:
                if len(packer.ready[length]) >= accumulation:
                    group = packer.ready[length][:accumulation]
                    packer.ready[length] = packer.ready[length][accumulation:]
                    break
            if group is not None:
                break
            packer.place(packer.segment_at(packer.next_index))
        b = per_bucket[length]
        b["steps"] += 1
        b["capacity"] += sum(g.rows * g.length for g in group)
        for g in group:
            for r, row in enumerate(g.content):
                digest.update(np.asarray([length, r, *[s.index for s in row], -1], np.int64))
                for seg in row:
                    n = len(seg.tokens)
                    b["real"] += n
                    b["slots"] += 1
                    src_tokens[seg.source_id] += n
                    src_segments[seg.source_id] += 1
    drawn = stream.table[: packer.next_index]
    stream_tokens = np.bincount(drawn[:, 2], weights=drawn[:, 0], minlength=sources)
    total_real = sum(b["real"] for b in per_bucket.values())
    total_cap = sum(b["capacity"] for b in per_bucket.values())
    return {
        "packing": packing,
        "steps": steps,
        "segments_consumed": int(packer.next_index),
        "segments_trained": int(src_segments.sum()),
        "real_fraction": total_real / total_cap,
        "real_tokens_per_step": total_real / steps,
        "by_bucket": {
            str(k): {
                "steps": int(b["steps"]),
                "step_share": b["steps"] / steps,
                "pad_fraction": 1 - b["real"] / b["capacity"],
                "slots_per_step": b["slots"] / b["steps"],
                "real_tokens_per_step": b["real"] / b["steps"],
            }
            for k, b in sorted(per_bucket.items())
        },
        "source_token_share": (src_tokens / max(src_tokens.sum(), 1)).tolist(),
        "source_segment_share": (src_segments / max(src_segments.sum(), 1)).tolist(),
        # the same share over every segment the stream handed the packer (in flight included)
        "stream_token_share": (stream_tokens / stream_tokens.sum()).tolist(),
        "digest": digest.hexdigest()[:16],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mix", type=Path)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--microbatch", type=int, nargs="+", default=[4096])
    ap.add_argument("--tokens-per-step", type=int, default=65536)
    ap.add_argument("--buckets", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--labels", type=int, default=26)
    ap.add_argument("--segments", type=int, default=0, help="0: steps × 200")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--table", type=Path, help="cache of the segment table (.npy)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    t0 = time.time()
    if args.table and args.table.exists():
        table = np.load(args.table)
        names = json.loads(args.table.with_suffix(".json").read_text())
    else:
        sources, weights, alpha = load_mix(args.mix)
        print(f"[load] {sum(map(len, sources.values()))} items, {len(sources)} sources "
              f"({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
        mixture = Mixture(sources, weights or None, alpha)
        vocab = Vocab.from_items(*sources.values())
        vocab.frozen = True
        at = SegmentAt(mixture, PromptTokenizer(args.tokenizer), vocab, seed=args.seed)
        table = segment_table(at, args.segments or args.steps * 200, args.workers)
        names = vocab.names["source"]
        if args.table:
            np.save(args.table, table)
            args.table.with_suffix(".json").write_text(json.dumps(names))
    print(f"[segments] {len(table)}, mean {table[:, 0].mean():.0f} tokens "
          f"({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
    stream = LengthStream(table)
    buckets = tuple(args.buckets)
    report: dict = {"mix": str(args.mix), "seed": args.seed, "steps": args.steps,
                    "segments_table": len(table), "sources": names, "runs": {}}  # fmt: skip
    for micro in args.microbatch:
        kw = {"buckets": buckets, "microbatch": micro, "tokens_per_step": args.tokens_per_step,
                  "bins": args.bins, "slots": args.slots, "labels": args.labels}  # fmt: skip
        runs = {}
        for packing in PACKINGS:
            r = run(stream, args.steps, packing=packing, **kw)
            again = run(stream, args.steps, packing=packing, **kw)
            resumed = run(stream, args.steps, packing=packing, resume_at=args.steps // 3, **kw)
            r["deterministic"] = again["digest"] == r["digest"]
            r["resume_exact"] = resumed["digest"] == r["digest"]
            runs[packing] = r
            print(f"[micro {micro}] {packing}: real {r['real_fraction']:.4f}, "
                  + ", ".join(f"T{k} pad {v['pad_fraction']:.3f} ({v['steps']} steps, "
                              f"{v['slots_per_step']:.0f} slots)" for k, v in r["by_bucket"].items())
                  + f"; deterministic {r['deterministic']}, resume {r['resume_exact']} "
                  f"({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
        old, new = runs["bucket"], runs["best_fit"]
        a, b = np.asarray(old["source_token_share"]), np.asarray(new["source_token_share"])
        runs["source_share_diff"] = {
            "max_abs": float(np.abs(a - b).max()),
            "max_rel_of_shares_over_0.5pct": float((np.abs(a - b)[a > 0.005] / a[a > 0.005]).max()),
            "total_variation": float(0.5 * np.abs(a - b).sum()),
        }
        for name, r in runs.items():
            if name in PACKINGS:
                t, d = np.asarray(r["source_token_share"]), np.asarray(r["stream_token_share"])
                r["trained_vs_stream_max_abs"] = float(np.abs(t - d).max())
        runs["real_tokens_per_step_gain"] = (
            new["real_tokens_per_step"] / old["real_tokens_per_step"]
        )
        report["runs"][str(micro)] = runs
        print(f"[micro {micro}] source token share: {runs['source_share_diff']}; real tokens "
              f"per step × {runs['real_tokens_per_step_gain']:.4f}", flush=True)  # fmt: skip
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
