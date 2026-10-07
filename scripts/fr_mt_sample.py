"""jev_fr_mt on a sample: draw rows, translate EN→FR, QE-gate, write the outputs and stats.

    # 1. a stratified sample of the contender train files (seconds, local)
    python scripts/fr_mt_sample.py sample --in ~/tjev-work/data/mix-v6/train/{plumb,decider_teacher,mghafiri_scenarios}.jsonl \
        --n 200 --seed 0 --out experiments/fr_mt_sample/in.jsonl
    # 2. translate and gate (CPU heavy: on Colab, never on the Mac)
    TJEV_CPU_SESSION=tjev-cpu-gen-mt bash cloud/colab_cpu.sh --keep \
        "uv pip install -q --python /content/cpu/venv/bin/python sentencepiece sacremoses ctranslate2 && \
         python scripts/fr_mt_sample.py run --in experiments/fr_mt_sample/in.jsonl --out /content/fr_mt"

``run`` writes OUT/translated.jsonl (accepted items, with ``license_use`` and ``meta``),
OUT/rejected.jsonl (ids, reasons, measurements) and OUT/stats.json (throughput, model load
time, accept rate per check and per base source).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from tjev.data.sources.gen_v7 import fr_mt
except ImportError:  # a venv without tjev's dependencies (unbabel-comet needs numpy < 2,
    # which jax refuses): fr_mt itself only needs the standard library
    import importlib.util

    _path = Path(__file__).resolve().parents[1] / "src/tjev/data/sources/gen_v7/fr_mt.py"
    _spec = importlib.util.spec_from_file_location("fr_mt", _path)
    fr_mt = importlib.util.module_from_spec(_spec)
    sys.modules["fr_mt"] = fr_mt  # dataclasses look their module up
    _spec.loader.exec_module(fr_mt)


def read(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with Path(p).expanduser().open(encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def sample(paths: list[str], n: int, seed: int) -> list[dict]:
    """``n`` rows, proportional to each file's size (at least one per file), seeded."""
    files = [read([p]) for p in paths]
    total = sum(map(len, files))
    rng = np.random.default_rng(seed)
    out = []
    for rows in files:
        k = max(1, round(n * len(rows) / total))
        out += [
            rows[i] for i in sorted(rng.choice(len(rows), size=min(k, len(rows)), replace=False))
        ]
    return out[:n] if len(out) > n else out


def cpu_info() -> dict:
    info = {"nproc": os.cpu_count(), "machine": platform.machine()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        info["cpu"] = platform.processor()
    return info


def run(args) -> dict:
    rows = read(args.inputs)
    t0 = time.perf_counter()
    if args.backend == "tables":  # re-gate saved translations offline (no model)
        mt, rt = fr_mt.table_translators(json.loads(Path(args.tables).read_text()))
    elif args.backend == "ct2":
        ct2 = {"threads": args.threads, "device": args.device, "compute_type": args.compute_type}
        mt = fr_mt.ct2_translator(args.mt_model, beams=args.beams, batch_size=args.batch, **ct2)
        rt = fr_mt.ct2_translator(args.rt_model, beams=1, batch_size=args.batch, **ct2)
    else:
        import torch

        torch.set_num_threads(args.threads or os.cpu_count())
        mt = fr_mt.hf_translator(args.mt_model, beams=args.beams, batch_size=args.batch)
        rt = fr_mt.hf_translator(args.rt_model, beams=1, batch_size=args.batch)
    qe = None
    if args.kiwi != "none":
        saved = json.loads(Path(args.tables).read_text()).get("kiwi") if args.tables else None
        if args.backend == "tables" and saved and not args.rescore:
            qe = fr_mt.table_scorer(saved)
        else:
            qe = fr_mt.kiwi_scorer(args.kiwi, batch_size=args.kiwi_batch, gpus=args.gpus)
    load = time.perf_counter() - t0
    t1 = time.perf_counter()
    accepted, rejected, timing = fr_mt.translate_rows(
        rows,
        mt,
        rt,
        mt_model=args.mt_model,
        qe=qe,
        min_kiwi=args.min_kiwi,
        min_kiwi_segment=args.min_kiwi_segment,
        qe_model=args.kiwi,
    )
    wall = time.perf_counter() - t1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.json").write_text(json.dumps(timing.pop("tables"), ensure_ascii=False))
    for name, data in (("translated", accepted), ("rejected", rejected)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
    gated = [r["meta"] for r in accepted] + [r for r in rejected if "checks" in r and "chrf" in r]
    per_check = Counter()
    for g in gated:
        for k, v in g["checks"].items():
            per_check[k] += bool(v)
    by_source: dict[str, Counter] = {}
    for r in accepted:
        by_source.setdefault(r["meta"]["base_source"], Counter())["accepted"] += 1
    for r in rejected:
        by_source.setdefault(r["base_source"], Counter())["rejected"] += 1
    src_chars = sum(len(fr_mt._flat(r)) for r in rows)
    chrfs = sorted(g["chrf"] for g in gated)
    stats = {
        "rows": len(rows),
        "accepted": len(accepted),
        "accept_rate": round(len(accepted) / max(1, len(rows)), 3),
        "skipped_english_specific": sum(1 for r in rejected if "chrf" not in r),
        "pass_rate_per_check (of gated rows)": {
            k: round(v / max(1, len(gated)), 3) for k, v in per_check.items()
        },
        "by_base_source": {k: dict(v) for k, v in by_source.items()},
        "chrf_quantiles": {q: chrfs[int(q / 100 * (len(chrfs) - 1))] for q in (5, 25, 50, 75, 95)}
        if chrfs else {},
        "model_load_seconds": round(load, 1),
        "wall_seconds": round(wall, 1),
        "rows_per_second": round(len(rows) / wall, 3),
        "source_chars": src_chars,
        "source_chars_per_second (end to end)": round(src_chars / wall, 1),
        "mt": {"seconds": round(timing["mt_seconds"], 1), "segments": timing["segments"],
               "chars": timing["segment_chars"],
               "chars_per_second": round(timing["segment_chars"] / max(timing["mt_seconds"], 1e-9), 1),
               "beams": args.beams, "model": args.mt_model},
        "round_trip": {"seconds": round(timing["rt_seconds"], 1),
                       "segments": timing["rt_segments"], "chars": timing["rt_chars"],
                       "chars_per_second": round(timing["rt_chars"] / max(timing["rt_seconds"], 1e-9), 1),
                       "beams": 1, "model": args.rt_model},
        "backend": args.backend,
        "device": getattr(args, "device", "cpu"),
        "compute_type": getattr(args, "compute_type", "int8"),
        "threads": args.threads or os.cpu_count(),
        "batch": args.batch,
        **cpu_info(),
    }  # fmt: skip
    if qe is not None:
        stats["kiwi"] = kiwi_report(accepted, rejected, timing)
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    return stats


def _q(xs: list[float]) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}
    return {q: round(xs[int(q / 100 * (len(xs) - 1))], 3) for q in (5, 10, 25, 50, 75, 90, 95)}


def kiwi_report(accepted: list[dict], rejected: list[dict], timing: dict) -> dict:
    """CometKiwi against the existing gate: score quantiles of the rows the gate keeps and
    rejects, agreement with the round trip, and the keep rate per source when a threshold
    is added (regate on the logged scores)."""
    gated = [r["meta"] for r in accepted] + [r for r in rejected if "chrf" in r]
    base = {id(g): fr_mt.regate(g) for g in gated}
    kept = [g["kiwi"] for g in gated if base[id(g)]]
    dropped = [g["kiwi"] for g in gated if not base[id(g)]]
    trip = [g for g in gated if g["checks"].get("round_trip") is not None]
    sweep = {}
    sources = sorted({g["base_source"] for g in gated})
    totals = Counter(g["base_source"] for g in gated)
    for t in (0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84):
        keep = Counter(g["base_source"] for g in gated if fr_mt.regate(g, min_kiwi=t))
        sweep[t] = {"kept": sum(keep.values()),
                    **{s: f"{keep[s]}/{totals[s]}" for s in sources}}  # fmt: skip
    return {
        "model": fr_mt.KIWI_MODEL,
        "licence": fr_mt.KIWI_LICENCE,
        "seconds": round(timing["qe_seconds"], 1),
        "segments": timing["qe_segments"],
        "segments_per_second": round(timing["qe_segments"] / max(timing["qe_seconds"], 1e-9), 2),
        "chars_per_second (src+mt)": round(timing["qe_chars"] / max(timing["qe_seconds"], 1e-9), 1),
        "kept_by_gate": _q(kept),
        "rejected_by_gate": _q(dropped),
        "round_trip_pass": _q([g["kiwi"] for g in trip if g["checks"]["round_trip"]]),
        "round_trip_fail": _q([g["kiwi"] for g in trip if not g["checks"]["round_trip"]]),
        "worst_segment_kept": _q([g["kiwi_worst_segment"] for g in gated if base[id(g)]]),
        "keep_by_threshold": sweep,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--in", dest="inputs", nargs="+", required=True)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--in", dest="inputs", nargs="+", required=True)
    r.add_argument("--n", type=int, default=0, help="first n rows only (0: all)")
    r.add_argument("--out", required=True)
    r.add_argument("--mt-model", default=fr_mt.MT_MODEL)
    r.add_argument("--rt-model", default=fr_mt.RT_MODEL)
    r.add_argument("--beams", type=int, default=4)
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--threads", type=int, default=0)
    r.add_argument("--backend", choices=("hf", "ct2", "tables"), default="ct2")
    r.add_argument("--device", default="cpu", help="ct2: cpu or cuda")
    r.add_argument("--compute-type", default="int8", help="ct2: int8 (CPU), float16 (GPU)")
    r.add_argument("--tables", help="--backend tables: a run's OUT/tables.json")
    r.add_argument("--kiwi", default="none", help=f"a CometKiwi model ({fr_mt.KIWI_MODEL}) or none")
    r.add_argument("--min-kiwi", type=float, default=fr_mt.MIN_KIWI)
    r.add_argument("--min-kiwi-segment", type=float, default=fr_mt.MIN_KIWI_SEGMENT)
    r.add_argument("--kiwi-batch", type=int, default=16)
    r.add_argument("--gpus", type=int, default=0)
    r.add_argument("--rescore", action="store_true", help="score again even if tables have kiwi")
    args = ap.parse_args()
    if args.cmd == "sample":
        rows = sample(args.inputs, args.n, args.seed)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out).open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        print(len(rows), dict(Counter(fr_mt.base_source(r) for r in rows)))
    else:
        if args.n:
            rows = read(args.inputs)[: args.n]
            tmp = Path(args.out) / "_in.jsonl"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            args.inputs = [str(tmp)]
        run(args)


if __name__ == "__main__":
    main()
