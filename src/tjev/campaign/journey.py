"""The kernel journey: every iteration of the GDN core, measured the same way on real training.

    python -m tjev.campaign.journey final-line --plan PLAN --size 4B --horizon 3000 > LINE
    python -m tjev.campaign.journey queue --final-line LINE --size 0.8B --steps 200 >> Q
    python -m tjev.campaign.journey collect --runs RUNS --out DIR [--parity P] [--ir IR]
    python -m tjev.campaign.journey ir --sizes 0.8B --variants xla,fused-bwd,… --out DIR/ir
    python -m tjev.campaign.journey estimate [--steps 0.8B:20,4B:20] [--variants-4b …]

Driven by ``PROBE_ONLY=journey`` (cloud/tpu_campaign.sh, docs/kernel-journey.md):

1. ``queue``: per (size, variant) one real ``tjev train`` job, the planner's final-run
   configuration of the size (the ``final-<size>-h<horizon>`` line of ``tjev campaign plan
   final``, picked by its run name: ``final-line``) with the GDN /
   attention / LoRA / remat overrides replaced by the variant's (``bench.VARIANTS``; its
   environment, ``bench.VARIANT_ENV``, as ``env=`` queue tokens), on the real mix, for a
   fixed number of steps, without checkpoints or evals (``train.checkpoint_every=0``,
   ``train.quick_eval_every=0``), cold compiles (no persistent cache) all done before step 1
   (``train.precompile_buckets``), and an xprof capture
   of a few steps near the end (detail mode: ``TRACE_COMPUTE_AND_SYNC``). W&B: project
   tjev-benchmarks, group ``journey``, tags size and variant.
2. ``collect``: from each run's ``metrics.jsonl`` and ``stdout.log``: the step time
   distribution (p10 / p50 / p90) of the steady steps only (without the compile rows, the
   first and the last row, and the profiled window: ``steady_steps``), per
   sequence bucket too; tokens/s (real, non-padding, and padded), MFU, peak HBM, the compile
   time per bucket shape (the dispatch time of its step-0 precompile, or of the step that
   compiled it in a run without ``train.precompile_buckets``), the xprof summary
   (time by scope, by category, top ops, HBM bytes per op when the trace has them); the
   correctness column from ``tjev campaign parity`` (the same fixed batch for every variant,
   against the XLA reference with the conv barriers) and the IR statistics of ``ir``.
   Writes ``<size>-<variant>.json`` per run and ``journey.json``; logs numbers, tables and
   charts to W&B (never the raw traces).
3. ``ir``: one compile of the GDN core (fwd + bwd, the model's own code under the variant's
   ComputeSpec) per variant in its own process, with the HLO, Mosaic and LLO dumps on (the
   libtpu flags it accepts), parsed by :mod:`tjev.campaign.ir_stats`.
4. ``estimate``: the session's time and Colab compute units at the measured speeds.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# The story, in order (bench.VARIANTS names). Pallas v1 = the preset's fused forward kernel
# ("kernels") and its split fallback; v2 = fused forward + backward; mega1 / mega2: the
# megakernels; then chunk 128 with the blocked inverse, then merged LoRA (the sweep's path).
JOURNEY = (
    "xla-nobarrier",
    "xla",
    "kernels",
    "split",
    "fused-bwd",
    "fused-bwd+states",
    "mega1",
    "mega2",
    "mega2-c128-blocked",
    "mega2-c128-merged",
)
LABELS = {
    "xla-nobarrier": "XLA chunked, no conv barrier (miscompiled)",
    "xla": "XLA chunked",
    "kernels": "Pallas v1 (fused forward)",
    "split": "Pallas v1 (split)",
    "fused-bwd": "Pallas v2 (fused fwd + bwd)",
    "fused-bwd+states": "v2 + core_states remat",
    "mega1": "mega1",
    "mega2": "mega2 (conv in the kernel)",
    "mega2-c128-blocked": "mega2, chunk 128, blocked inverse",
    "mega2-c128-merged": "+ merged LoRA",
}
IR_VARIANTS = ("xla", "fused-bwd", "mega1", "mega2", "mega2-c128-blocked")
# the key iterations: what a size runs when the whole journey would cost too much (4B)
KEY_VARIANTS = ("xla", "fused-bwd+states", "mega2", "mega2-c128-blocked", "mega2-c128-merged")
# every bucket precompiled (train.precompile_buckets): the steps are all steady, and 20 give
# the median step time to ~±0.3% (docs/kernel-journey.md, "How many steps")
DEFAULT_STEPS = "0.8B:20,4B:20"
PROFILE_STEPS = 3
WANDB_PROJECT = "tjev-benchmarks"
WANDB_GROUP = "journey"
# replaced on the final line: the variant's compute, the benchmark's run length and logging
STRIPPED = (
    "compute.gdn_",
    "compute.lora_impl=",
    "compute.remat=",
    "compute.attention",
    "train.steps=",
    "train.checkpoint_every=",
    "train.checkpoint_secs=",
    "train.keep_checkpoints=",
    "train.quick_eval_every=",
    "train.profile_",
    "train.health_kill=",
    "train.branch_from=",
    "log.wandb_project=",
    "log.wandb_group=",
    "log.wandb_tags=",
    "chips=",
    "post=",
    "after=",
    "env=",
    "name=",
    "output=",
)


def parse_steps(spec: str) -> dict[str, int]:
    """``"0.8B:200,4B:100"`` → {"0.8B": 200, "4B": 100}."""
    out = {}
    for item in spec.split(","):
        if item.strip():
            size, _, steps = item.partition(":")
            out[size.strip()] = int(steps)
    return out


def variants_for(
    size: str, variants: list[str], size_variants: dict[str, list[str]] | None = None
) -> list[str]:
    """The variants a size runs: its own list in ``size_variants``, else ``variants``."""
    return list((size_variants or {}).get(size) or variants)


def run_name(size: str, variant: str) -> str:
    return f"journey-{size.lower()}-{variant}"


def final_run_name(size: str, horizon: int) -> str:
    """The planner's long final run of a size (plan.horizon_jobs: ``final-<size>-h<H>``)."""
    return f"final-{size.lower()}-h{horizon}"


def final_line_for(lines: list[str], size: str, horizon: int | None = None) -> str:
    """The size's long final run among the planner's lines, picked by its run name (never by
    position: the planner orders sizes as its ``--sizes`` says, and a caller may pass more
    sizes than asked): ``final-<size>-h<horizon>``, or with no horizon the size's only
    root line (no ``after=``: not a branch)."""
    named = [line for line in lines if line.split()]
    if horizon:
        want = final_run_name(size, horizon)
        found = [line for line in named if line.split()[0] == want]
    else:
        prefix = f"final-{size.lower()}-h"
        found = [line for line in named if line.split()[0].startswith(prefix)
                 and not any(t.startswith("after=") for t in line.split())]  # fmt: skip
    if len(found) != 1:
        names = [line.split()[0] for line in named]
        what = final_run_name(size, horizon) if horizon else f"one final-{size.lower()}-h* root"
        raise ValueError(f"{len(found)} lines for {what} in the plan ({', '.join(names)})")
    return found[0]


def check_final_line(final_line: str, size: str) -> None:
    """The final line is the size's own run: its name, and its model path when it has one
    (a 0.8B line under a 4B name trained 0.8B: tjev-journey2)."""
    tokens = final_line.split()
    if not tokens or not tokens[0].startswith(f"final-{size.lower()}-"):
        raise ValueError(f"the final line {tokens[:1]} is not a final-{size.lower()} run")
    paths = [t.partition("=")[2] for t in tokens if t.startswith("model.path=")]
    if paths and not paths[-1].rstrip("/").endswith(f"Qwen3.5-{size}"):
        raise ValueError(f"the final line of {size} trains {paths[-1]}")


def profile_window(steps: int, profile_steps: int = PROFILE_STEPS) -> tuple[int, int]:
    """(train.profile_start, train.profile_steps): a few steps near the end, after every
    bucket has compiled; (0, 0) when the run is too short to leave a warm-up."""
    if profile_steps <= 0 or steps < profile_steps + 8:
        return 0, 0
    return steps - profile_steps - 5, profile_steps


def journey_line(
    final_line: str,
    size: str,
    variant: str,
    steps: int,
    *,
    profile_steps: int = PROFILE_STEPS,
    detail: bool = True,
    cold: bool = True,
) -> str:
    """The queue line of one (size, variant) run from the planner's final-run line."""
    from tjev.campaign.bench import VARIANT_ENV, VARIANTS

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (bench.VARIANTS)")
    tokens = final_line.split()
    if not tokens:
        raise ValueError("empty final line")
    check_final_line(final_line, size)
    kept = [t for t in tokens[1:] if not t.startswith(STRIPPED)]
    start, count = profile_window(steps, profile_steps)
    env = dict(VARIANT_ENV.get(variant, {}))
    if start and detail:
        env["TJEV_PROFILE_DETAIL"] = "1"
    if cold:  # compile times as a fresh process sees them, not the host cache's
        env["JAX_ENABLE_COMPILATION_CACHE"] = "false"
        # why a step after the precompile still compiles (tjev-journey2: the first real step
        # of every splash / Pallas variant recompiled the 1024 bucket, ~24 s)
        env["JAX_EXPLAIN_CACHE_MISSES"] = "true"
    args = [
        *kept,
        *VARIANTS[variant],
        f"train.steps={steps}",
        "train.checkpoint_every=0",
        "train.checkpoint_secs=0",
        "train.quick_eval_every=0",
        "train.health_kill=false",
        "train.log_every=1",
        "train.precompile_buckets=true",  # every bucket compiled before step 1: all steps steady
        f"train.profile_start={start}",
        f"train.profile_steps={count}",
    ]
    if "log.wandb=true" in kept:
        args += [
            f"log.wandb_project={WANDB_PROJECT}",
            f"log.wandb_group={WANDB_GROUP}",
            f"log.wandb_tags=[journey,{size},variant:{variant}]",
        ]
    head = [
        run_name(size, variant),
        "chips=1",
        "post=none",
        *(f"env={k}={v}" for k, v in env.items()),
    ]
    return " ".join(head + args)


# ------------------------------------------------------------------------------------------
# Collect


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def events(stdout: str, kind: str) -> list[dict[str, Any]]:
    out = []
    for line in stdout.splitlines():
        if line.startswith('{"event"') and f'"{kind}"' in line:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == kind:
                out.append(record)
    return out


def percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    a = np.asarray(values, float)
    return {
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "mean": float(a.mean()),
        "n": len(values),
    }


def failure(stdout: str) -> str:
    """A short reason from a failed run's log: OOM, a compile error, or the last error line."""
    if "RESOURCE_EXHAUSTED" in stdout or "Ran out of memory" in stdout:
        return "oom"
    lines = [x for x in stdout.splitlines() if re.search(r"Error|error:|Traceback", x)]
    return lines[-1][:300] if lines else "failed (see stdout.log)"


def steady_steps(perf: list[dict[str, Any]], excluded: set[int]) -> list[dict[str, Any]]:
    """The rows that time one steady training step: not a compile row, not ``excluded``
    (the profiled window), and neither the first nor the last row. The loop times a row as
    the wall interval between two metric fetches, each made after the next step's dispatch
    (train.loop.MetricWindow, metrics one step late), so the first row holds the dispatch of
    steps 1 and 2: a compile there lands in it whatever its ``perf/includes_compile`` says
    (tjev-journey2: every splash / Pallas run recompiled the 1024 bucket at step 1 despite
    the precompile, a 25 s first row). The last row waits for its own step to finish (the
    final full fetch: no next step to overlap)."""
    if len(perf) <= 2:
        return []
    first, last = perf[0]["step"], perf[-1]["step"]
    return [r for r in perf if not r.get("perf/includes_compile") and r["step"] not in excluded
            and r["step"] not in (first, last)]  # fmt: skip


def step_stats(rows: list[dict[str, Any]], excluded: set[int]) -> dict[str, Any]:
    """Timing from the per-step metric rows (``perf/*``, ``data/seq_len``) of the steady
    steps (:func:`steady_steps`); tokens/s and MFU as totals over them. ``warmup_s``: the
    first row's seconds (with any compile the precompile missed) and its excess over the
    median step (``warmup_excess_s``)."""
    perf = [r for r in rows if "perf/step_seconds" in r]
    timed = steady_steps(perf, excluded)
    out: dict[str, Any] = {"steps_logged": len(perf), "steps_timed": len(timed)}
    if not timed:
        return out
    seconds = np.asarray([r["perf/step_seconds"] for r in timed], float)
    total = float(seconds.sum())
    real = sum(
        r.get("perf/tokens_per_second_per_device", 0.0) * r["perf/step_seconds"] for r in timed
    )
    padded = sum(
        r.get("perf/padded_tokens_per_second_per_device", 0.0) * r["perf/step_seconds"]
        for r in timed
    )
    out["step_s"] = percentiles(seconds.tolist())
    out["warmup_s"] = float(perf[0]["perf/step_seconds"])
    out["warmup_excess_s"] = out["warmup_s"] - out["step_s"]["p50"]
    out["tokens_per_second"] = real / total
    if padded:
        out["padded_tokens_per_second"] = padded / total
        out["pad_fraction"] = 1.0 - real / padded
    mfu = [(r["perf/mfu"], r["perf/step_seconds"]) for r in timed if r.get("perf/mfu") is not None]
    if mfu:
        out["mfu"] = sum(m * s for m, s in mfu) / sum(s for _, s in mfu)
    tflops = [
        r["perf/tflops_per_device"] * r["perf/step_seconds"]
        for r in timed
        if r.get("perf/tflops_per_device") is not None
    ]
    if tflops:
        out["tflops"] = sum(tflops) / total
    hbm = [r["perf/hbm_peak_gb"] for r in perf if r.get("perf/hbm_peak_gb") is not None]
    if hbm:
        out["hbm_peak_gb"] = max(hbm)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in timed:
        if "data/seq_len" in r:
            buckets.setdefault(str(int(r["data/seq_len"])), []).append(r)
    out["by_bucket"] = {}
    for length, group in sorted(buckets.items(), key=lambda kv: int(kv[0])):
        secs = [r["perf/step_seconds"] for r in group]
        tok = sum(
            r.get("perf/tokens_per_second_per_device", 0.0) * r["perf/step_seconds"] for r in group
        )
        out["by_bucket"][length] = {
            "step_s": percentiles(secs),
            "tokens_per_second": tok / sum(secs),
            "share_of_steps": len(group) / len(timed),
        }
    losses = [r["learning/loss"] for r in perf if "learning/loss" in r]
    if losses:
        k = max(1, min(10, len(losses) // 4))
        out["loss_first"] = float(np.mean(losses[:k]))
        out["loss_last"] = float(np.mean(losses[-k:]))
    return out


def run_report(run_dir: Path, size: str, variant: str) -> dict[str, Any]:
    """Everything ``collect`` reads about one run (missing parts are left out)."""
    from tjev.campaign.xprof_tools import summarize_any

    report: dict[str, Any] = {
        "size": size,
        "variant": variant,
        "label": LABELS.get(variant, variant),
        "run": run_dir.name,
    }
    if not run_dir.is_dir():
        return report | {"status": "missing"}
    stdout = (
        (run_dir / "stdout.log").read_text(encoding="utf-8", errors="replace")
        if (run_dir / "stdout.log").exists()
        else ""
    )
    report["status"] = "ok" if (run_dir / "queue.done").exists() else "failed"
    if report["status"] == "failed":
        report["error"] = failure(stdout)
    config = {}
    if (run_dir / "config.json").exists():
        config = json.loads((run_dir / "config.json").read_text())["config"]
    train = config.get("train", {})
    start, count = train.get("profile_start", 0), train.get("profile_steps", 0)
    # the profiled steps and their neighbours (start / stop of the trace land on them)
    excluded = set(range(start - 1, start + count + 2)) if start else set()
    report |= step_stats(_rows(run_dir / "metrics.jsonl"), excluded)
    # one compile per shape: the step-0 precompile (train.precompile_buckets) when there is
    # one, else the first step of that shape (its dispatch time, the step itself excluded)
    first: dict[tuple[int, ...], dict[str, Any]] = {}
    for c in events(stdout, "compiled"):
        key = tuple(c["shape"])
        if key not in first or (c.get("precompiled") and not first[key].get("precompiled")):
            first[key] = c
    report["compiles"] = [
        {
            "seq_len": c["shape"][-1],
            "shape": c["shape"],
            "step": c["step"],
            "seconds": c["seconds"],
            "precompiled": bool(c.get("precompiled")),
        }
        for c in sorted(first.values(), key=lambda c: (c["step"], c["shape"][-1]))
    ]
    compiles = report["compiles"]
    report["compile_s_total"] = round(sum(c["seconds"] for c in compiles), 2)
    for c in report["compiles"]:
        bucket = report.get("by_bucket", {}).setdefault(str(c["seq_len"]), {})
        bucket["compile_s"] = c["seconds"]
    if config:
        report["compute"] = config.get("compute", {})
        report["tokens_per_step"] = train.get("tokens_per_step")
        report["rank"] = config.get("lora", {}).get("rank")
    if (run_dir / "profile").is_dir():
        try:
            # the window traces profile_steps + 1 steps (start_trace before step `start`,
            # stop_trace after step start + count): the fallback count when the trace
            # carries no module rows / step markers (the summary's step_source says which)
            report["profile"] = summarize_any(run_dir / "profile", steps=count + 1 if count else 3)
        except Exception as e:  # the summary is a convenience: the trace is the record
            report["profile"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return report


def correctness(parity: dict[str, Any] | None, size: str, variant: str) -> dict[str, Any] | None:
    """The parity check of one variant (fixed batch, against the XLA reference)."""
    if not parity:
        return None
    result = parity.get("sizes", {}).get(size, {}).get("variants", {}).get(variant)
    if result is None:
        return None
    if "error" in result:
        return {"pass": False, "error": result["error"][:200]}
    keys = ("pass", "loss_rel", "logit_max_abs", "grad_rel_l2", "grad_cosine", "grad_bound")
    return {k: result[k] for k in keys if k in result}


def collect(
    runs: Path,
    sizes: list[str],
    variants: list[str],
    parity: dict | None = None,
    ir: dict | None = None,
    size_variants: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """``size_variants``: a size's own variant list (the others run ``variants``)."""
    every = list(dict.fromkeys(v for s in sizes for v in variants_for(s, variants, size_variants)))
    out: dict[str, Any] = {"sizes": sizes, "variants": every, "labels": LABELS, "runs": []}
    for size in sizes:
        for variant in variants_for(size, variants, size_variants):
            report = run_report(runs / run_name(size, variant), size, variant)
            if (check := correctness(parity, size, variant)) is not None:
                report["correctness"] = check
            out["runs"].append(report)
    for size in sizes:  # the speed-up over the XLA path (with the barriers) of the same size
        base = next(
            (
                r.get("tokens_per_second")
                for r in out["runs"]
                if r["size"] == size and r["variant"] == "xla"
            ),
            None,
        )
        for r in out["runs"]:
            if r["size"] == size and base and r.get("tokens_per_second"):
                r["speedup_vs_xla"] = r["tokens_per_second"] / base
    if ir:
        out["ir"] = ir
    return out


# ------------------------------------------------------------------------------------------
# IR dumps: one compile of the GDN core per variant, in its own process (libtpu reads its
# flags once), with the dumps the installed libtpu accepts


def ir_flag_candidates(dumps: Path) -> list[str]:
    """libtpu flags for the Mosaic and LLO dumps (patricktoulme.substack.com, "When XLA
    isn't enough"); each is probed before use: an unknown flag aborts libtpu."""
    return [
        f"--xla_mosaic_dump_to={dumps / 'mosaic'}",
        "--xla_mosaic_enable_dump_debug_info=true",
        "--xla_mosaic_enable_llo_source_annotations=true",
        f"--xla_jf_dump_to={dumps / 'llo'}",
        "--xla_jf_dump_llo_text=true",
        "--xla_jf_dump_hlo_text=true",
        "--xla_jf_dump_llo_html=false",
        "--xla_jf_dump_llo_static_gaps=true",
        "--xla_jf_emit_annotations=true",
        "--xla_jf_debug_level=2",
    ]


PROBE = "import jax, jax.numpy as jnp; print(float(jax.jit(lambda x: (x @ x).sum())(jnp.ones((256, 256)))))"


def probe_flags(
    candidates: list[str], base: str, timeout: int = 300
) -> tuple[list[str], list[str]]:
    """The candidates libtpu accepts, one at a time on top of ``base`` and those before."""
    ok: list[str] = []
    rejected: list[str] = []
    for flag in candidates:
        env = os.environ | {"LIBTPU_INIT_ARGS": " ".join([base, *ok, flag]).strip()}
        try:
            done = subprocess.run(
                [sys.executable, "-c", PROBE],
                env=env,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            (ok if done.returncode == 0 else rejected).append(flag)
        except subprocess.TimeoutExpired:
            rejected.append(flag)
    return ok, rejected


def _with_dir(flag: str, old: Path, new: Path) -> str:
    return flag.replace(str(old), str(new))


def core_function(
    variant: str, size: str, *, rows: int | None = None, length: int = 4096, model=None
):
    """(fwd + bwd of the model's GDN core under the variant's ComputeSpec, its inputs)."""
    import functools
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp

    from tjev.campaign.bench import VARIANTS
    from tjev.campaign.plan import HARDWARE, model_config
    from tjev.config import load_config
    from tjev.model.gdn import GatedDeltaNet

    hw = HARDWARE["v6e"]
    cfg = load_config(hw["preset"], overrides=list(VARIANTS[variant]))
    c = model or model_config(size)
    rows = rows or max(1, hw["microbatch"].get(size, 8192) // length)
    width = c.conv_dim
    heads = c.linear_num_value_heads
    keys = jax.random.split(jax.random.key(0), 6)
    module = SimpleNamespace(
        config=c,
        compute=cfg.compute,
        conv=(jax.random.normal(keys[0], (c.conv_kernel, width)) * 0.3).astype(jnp.bfloat16),
        A_log=jax.random.normal(keys[1], (heads,)) * 0.5,
        dt_bias=jax.random.normal(keys[2], (heads,)) * 0.5,
    )
    module.core_mega = functools.partial(GatedDeltaNet.core_mega, module)
    qkv = jax.random.normal(keys[3], (rows, length, width), jnp.bfloat16)
    b = jax.random.normal(keys[4], (rows, length, heads), jnp.bfloat16)
    a = jax.random.normal(keys[5], (rows, length, heads), jnp.bfloat16)
    pieces = 16 if length >= 1024 else 2  # packed segments, as a training row
    seg = jnp.repeat(jnp.arange(1, pieces + 1), length // pieces)
    seg = jnp.pad(seg, (0, length - seg.shape[0]))[None].repeat(rows, 0).astype(jnp.int32)

    def step(qkv, b, a):
        loss = lambda *x: jnp.sum(GatedDeltaNet.core(module, *x, seg).astype(jnp.float32))  # noqa: E731
        return jax.grad(loss, argnums=(0, 1, 2))(qkv, b, a)

    return step, (qkv, b, a), cfg.compute


def ir_compile(
    variant: str, size: str, out: Path, *, rows: int | None = None, length: int = 4096, model=None
) -> dict[str, Any]:
    """Lower and compile (never run) the variant's GDN core; its HLO dump in ``out/hlo``
    (the Mosaic / LLO dumps go where the process's libtpu flags say)."""
    import jax

    from tjev.campaign.bench import variant_env
    from tjev.config import MeshSpec
    from tjev.sharding import configure_runtime, install_mesh, make_mesh

    out.mkdir(parents=True, exist_ok=True)
    with variant_env(variant):
        step, inputs, compute = core_function(variant, size, rows=rows, length=length, model=model)
        configure_runtime(compute)
        install_mesh(compute, make_mesh(MeshSpec(data=1), jax.devices()[:1]))
        t0 = time.perf_counter()
        lowered = jax.jit(step).lower(*inputs)
        compiled = lowered.compile({"xla_dump_to": str(out / "hlo"), "xla_dump_hlo_as_text": True})
        seconds = time.perf_counter() - t0
    report: dict[str, Any] = {
        "variant": variant,
        "size": size,
        "compile_s": round(seconds, 2),
        "shape": [int(x) for x in inputs[0].shape],
        "device": jax.devices()[0].device_kind,
    }
    try:
        cost = compiled.cost_analysis()
        cost = cost[0] if isinstance(cost, list) else cost
        report["xla_cost"] = {
            k: float(v)
            for k, v in (cost or {}).items()
            if k in ("flops", "bytes accessed", "transcendentals")
        }
    except Exception as e:
        report["xla_cost"] = {"error": str(e)[:200]}
    mem = compiled.memory_analysis()
    if mem is not None:
        report["temp_gb"] = getattr(mem, "temp_size_in_bytes", 0) / 1e9
    (out / "compile.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def ir_all(
    sizes: list[str], variants: list[str], out: Path, *, probe: bool = True
) -> dict[str, Any]:
    """Every (size, variant) compile in its own process with its own dump directories."""
    from tjev.campaign.ir_stats import summarize_dumps

    out.mkdir(parents=True, exist_ok=True)
    base = os.environ.get("LIBTPU_INIT_ARGS", "")
    template = out / "_probe"
    flags, rejected = (
        probe_flags(ir_flag_candidates(template), base)
        if probe
        else (ir_flag_candidates(template), [])
    )
    report: dict[str, Any] = {
        "flags": [_with_dir(f, template, Path("<dumps>")) for f in flags],
        "rejected": rejected,
        "runs": {},
    }
    for size in sizes:
        for variant in variants:
            key = f"{size}-{variant}"
            dumps = out / key
            env = os.environ | {
                "LIBTPU_INIT_ARGS": " ".join(
                    [base, *(_with_dir(f, template, dumps) for f in flags)]
                ).strip()
            }
            cmd = [
                sys.executable,
                "-m",
                "tjev.campaign.journey",
                "ir-compile",
                "--variant",
                variant,
                "--size",
                size,
                "--out",
                str(dumps),
            ]
            dumps.mkdir(parents=True, exist_ok=True)
            with (dumps / "compile.log").open("w") as log:
                code = subprocess.run(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=3600
                ).returncode
            entry: dict[str, Any] = {"exit": code}
            if (dumps / "compile.json").exists():
                entry |= json.loads((dumps / "compile.json").read_text())
            try:
                entry["dumps"] = summarize_dumps(dumps)
            except Exception as e:
                entry["dumps"] = {"error": f"{type(e).__name__}: {e}"[:300]}
            report["runs"][key] = entry
            print(json.dumps({"ir": key, "exit": code, **brief_ir(entry)}), flush=True)
    (out / "ir.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def brief_ir(entry: dict[str, Any]) -> dict[str, Any]:
    """The article's numbers of one IR entry: bundles per kernel, MXU ops, VMEM, passes."""
    dumps = entry.get("dumps", {})
    llo = dumps.get("llo", {}).get("totals", {})
    mosaic = dumps.get("mosaic", {})
    hlo = dumps.get("hlo", {})
    fusions = max((v["fusions"] for v in hlo.values()), default=None) if hlo else None
    return {
        "compile_s": entry.get("compile_s"),
        "kernel_bundles": llo.get("kernel_bundles", {}),
        "fwd_bundles": llo.get("fwd_bundles"),
        "bwd_bundles": llo.get("bwd_bundles"),
        "xla_bundles": llo.get("xla_bundles"),
        "mxu_ops": llo.get("mxu_ops"),
        "kernel_vmem_bytes": llo.get("kernel_vmem_bytes")
        or mosaic.get("totals", {}).get("kernel_vmem_bytes"),
        "llo_passes": llo.get("llo_passes"),
        "mosaic_passes": mosaic.get("totals", {}).get("llo_passes"),
        "hlo_fusions": fusions,
        "xla_bytes_accessed": entry.get("xla_cost", {}).get("bytes accessed"),
    }


# ------------------------------------------------------------------------------------------
# Cost estimate (docs/kernel-journey.md): the measured speeds, 65,536 tokens per step

# tokens/s per v6e chip. 0.8B: docs/results.md (bench3 / bench4); Pallas v1: s0 bench (14.6k
# 0.8B, 5.6k 4B). 4B: the sweep path measured (14.1k merged, bench4), the others scaled by
# their 9B ratio to mega2-c128 (docs/results.md); "fused-bwd" without core_states assumed
# 0.9× v2 + states; no barrier: as fast as the barrier (the barriers are ~free).
SPEEDS: dict[str, dict[str, float]] = {
    "0.8B": {
        "xla-nobarrier": 13000,
        "xla": 13000,
        "kernels": 14600,
        "split": 14000,
        "fused-bwd": 25300,
        "fused-bwd+states": 28100,
        "mega1": 34200,
        "mega2": 40100,
        "mega2-c128-blocked": 41800,
        "mega2-c128-merged": 52700,
    },
    "4B": {
        "xla-nobarrier": 5000,
        "xla": 5000,
        "kernels": 5600,
        "split": 5400,
        "fused-bwd": 9800,
        "fused-bwd+states": 10900,
        "mega1": 11800,
        "mega2": 12600,
        "mega2-c128-blocked": 12900,
        "mega2-c128-merged": 14100,
    },
}
CU_PER_HOUR = 4.08  # Colab v6e-1
TOKENS_PER_STEP = 65536
OVERHEAD_S = {"0.8B": 240.0, "4B": 360.0}  # start, model load, cold compiles, trace, summary
BUCKETS = 3  # the precompile runs each bucket's step once (an all-padding batch, discarded)
PARITY_S = {"0.8B": 60.0, "4B": 150.0}  # one fixed batch per variant (one compile)
IR_S = 120.0  # one GDN-core compile with the dumps, per (size, variant)
FLAG_PROBE_S = 300.0


def estimate(
    steps: dict[str, int],
    variants: list[str],
    *,
    parity_sizes: list[str] | None = None,
    ir_sizes: list[str] | None = None,
    ir_variants: list[str] | None = None,
    tokens_per_step: int = TOKENS_PER_STEP,
    size_variants: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Seconds per (size, variant) training run at SPEEDS (the steps plus one precompile
    step per bucket), plus parity and IR, and the CU; ``size_variants`` as in collect."""
    rows = []
    for size, n in steps.items():
        for variant in variants_for(size, variants, size_variants):
            speed = SPEEDS.get(size, {}).get(variant)
            if speed is None:
                continue
            train_s = (n + BUCKETS) * tokens_per_step / speed
            rows.append(
                {
                    "size": size,
                    "variant": variant,
                    "steps": n,
                    "tokens_per_second": speed,
                    "train_s": train_s,
                    "total_s": train_s + OVERHEAD_S.get(size, 300.0),
                }
            )
    parity_s = sum(
        PARITY_S.get(s, 120.0) * (len(variants_for(s, variants, size_variants)) + 1)
        for s in parity_sizes or []
    )
    ir_s = FLAG_PROBE_S + IR_S * len(ir_sizes or []) * len(ir_variants or []) if ir_sizes else 0.0
    total = sum(r["total_s"] for r in rows) + parity_s + ir_s
    return {
        "runs": rows,
        "parity_s": parity_s,
        "ir_s": ir_s,
        "total_h": total / 3600,
        "cu": total / 3600 * CU_PER_HOUR,
    }


def estimate_markdown(est: dict[str, Any]) -> str:
    sizes = dict.fromkeys(r["size"] for r in est["runs"])
    per_size = ", ".join(
        f"{s} {sum(r['total_s'] for r in est['runs'] if r['size'] == s) / 3600:.2f} h"
        for s in sizes
    )
    lines = [
        "| Size | Variant | Steps | tok/s | Train | With overhead |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['size']} | {r['variant']} | {r['steps']} | {r['tokens_per_second']:,.0f} | "
        f"{r['train_s'] / 60:.1f} min | {r['total_s'] / 60:.1f} min |"
        for r in est["runs"]
    ]
    lines.append(
        f"\nTraining runs: {per_size}. "
        f"Parity {est['parity_s'] / 60:.0f} min, IR dumps {est['ir_s'] / 60:.0f} min; "
        f"total **{est['total_h']:.2f} h = {est['cu']:.1f} CU** at {CU_PER_HOUR} CU/h."
    )
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------
# W&B: numbers, tables and charts (never the raw traces: the free plan has 5 GB)


def flat_row(r: dict[str, Any]) -> dict[str, Any]:
    step = r.get("step_s") or {}
    check = r.get("correctness") or {}
    return {
        "size": r["size"],
        "variant": r["variant"],
        "status": r.get("status"),
        "tokens_per_second": r.get("tokens_per_second"),
        "padded_tokens_per_second": r.get("padded_tokens_per_second"),
        "speedup_vs_xla": r.get("speedup_vs_xla"),
        "mfu": r.get("mfu"),
        "step_p10_s": step.get("p10"),
        "step_p50_s": step.get("p50"),
        "step_p90_s": step.get("p90"),
        "hbm_peak_gb": r.get("hbm_peak_gb"),
        "compile_s_total": r.get("compile_s_total"),
        "loss_last": r.get("loss_last"),
        "correct": check.get("pass"),
        "loss_rel": check.get("loss_rel"),
        "logit_max_abs": check.get("logit_max_abs"),
        "grad_cosine": check.get("grad_cosine"),
        "error": r.get("error"),
    }


def log_wandb(journey: dict[str, Any], charts: list[Path]) -> None:
    from tjev.campaign.experiments import Experiment
    from tjev.campaign.xprof_tools import log_profile

    exp = Experiment(
        "journey",
        "journey",
        project=os.environ.get("TJEV_WANDB_BENCHMARKS_PROJECT", WANDB_PROJECT),
        config={
            "sizes": journey["sizes"],
            "variants": journey["variants"],
            "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        },
    )
    exp.table("runs", [flat_row(r) for r in journey["runs"]])
    exp.table(
        "buckets",
        [
            {
                "size": r["size"],
                "variant": r["variant"],
                "seq_len": int(k),
                "p50_s": (v.get("step_s") or {}).get("p50"),
                "tokens_per_second": v.get("tokens_per_second"),
                "compile_s": v.get("compile_s"),
            }
            for r in journey["runs"]
            for k, v in r.get("by_bucket", {}).items()
        ],
    )
    exp.table(
        "scopes_ms",
        [
            {"size": r["size"], "variant": r["variant"], **r["profile"]["ms_by_scope"]}
            for r in journey["runs"]
            if isinstance(r.get("profile"), dict) and "ms_by_scope" in r["profile"]
        ],
    )
    for r in journey["runs"]:  # scope × phase, top ops' roofline, peak buffers
        log_profile(exp, f"{r['size']}/{r['variant']}/profile/", r.get("profile"))
    for size in journey["sizes"]:
        runs = [r for r in journey["runs"] if r["size"] == size]
        for metric in ("tokens_per_second", "mfu", "hbm_peak_gb"):
            values = {r["variant"]: r[metric] for r in runs if r.get(metric) is not None}
            if values:
                exp.bar(f"{size}/{metric}", values, metric)
    exp.summary(
        {
            f"{r['size']}/{r['variant']}/{k}": v
            for r in journey["runs"]
            for k, v in flat_row(r).items()
            if isinstance(v, int | float | bool)
        }
    )
    if journey.get("ir"):
        exp.table(
            "ir",
            [
                {
                    "run": k,
                    **{
                        n: (json.dumps(v) if isinstance(v, dict) else v)
                        for n, v in brief_ir(e).items()
                    },
                }
                for k, e in journey["ir"].get("runs", {}).items()
            ],
        )
    for chart in charts:  # small PNGs (scripts/journey_report.py)
        exp.image(chart.stem, chart)
    exp.finish()


# ------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tjev.campaign.journey")
    sub = parser.add_subparsers(dest="cmd", required=True)
    fl = sub.add_parser("final-line", help="the size's final run from the planner's lines")
    fl.add_argument("--plan", type=Path, required=True, help="tjev campaign plan final output")
    fl.add_argument("--size", required=True)
    fl.add_argument("--horizon", type=int, default=0, help="0: the size's only root line")
    q = sub.add_parser("queue", help="queue lines of one size from the planner's final line")
    q.add_argument("--final-line", type=Path, required=True, help="a file: the final run's line")
    q.add_argument("--size", required=True)
    q.add_argument("--steps", type=int, required=True)
    q.add_argument("--variants", default=",".join(JOURNEY))
    q.add_argument("--profile-steps", type=int, default=PROFILE_STEPS)
    q.add_argument("--no-detail", action="store_true", help="profile in the timing mode")
    q.add_argument("--warm", action="store_true", help="keep the persistent compile cache")
    c = sub.add_parser("collect", help="per-run reports, journey.json, W&B")
    c.add_argument("--runs", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--sizes", default="0.8B,4B")
    c.add_argument("--variants", default=",".join(JOURNEY))
    c.add_argument("--variants-4b", default=",".join(KEY_VARIANTS), help="4B's own list")
    c.add_argument("--parity", type=Path, help="tjev campaign parity JSON (correctness)")
    c.add_argument("--ir", type=Path, help="the ir.json of the ir command")
    c.add_argument("--charts", type=Path, help="PNG charts to log (scripts/journey_report.py)")
    c.add_argument("--no-wandb", action="store_true")
    lg = sub.add_parser("log", help="log journey.json (and its PNG charts) to W&B")
    lg.add_argument("--journey", type=Path, required=True)
    lg.add_argument("--charts", type=Path, help="a directory of PNG charts")
    i = sub.add_parser("ir", help="GDN-core compiles with the HLO / Mosaic / LLO dumps")
    i.add_argument("--sizes", default="0.8B")
    i.add_argument("--variants", default=",".join(IR_VARIANTS))
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--no-probe", action="store_true", help="use the flags without probing")
    ic = sub.add_parser("ir-compile", help="(internal) one compile in this process")
    ic.add_argument("--variant", required=True)
    ic.add_argument("--size", required=True)
    ic.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("estimate", help="session time and CU at the measured speeds")
    e.add_argument("--steps", default=DEFAULT_STEPS)
    e.add_argument("--variants", default=",".join(JOURNEY))
    e.add_argument("--variants-4b", default=",".join(KEY_VARIANTS), help="4B's own list")
    e.add_argument("--parity-sizes", default="0.8B")
    e.add_argument("--ir-sizes", default="0.8B")
    e.add_argument("--ir-variants", default=",".join(IR_VARIANTS))
    args = parser.parse_args(argv)
    split = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    if args.cmd == "final-line":
        lines = args.plan.read_text().splitlines()
        try:
            print(final_line_for(lines, args.size, args.horizon or None))
        except ValueError as e:
            print(f"journey final-line: {e}", file=sys.stderr)
            return 1
        return 0
    if args.cmd == "queue":
        line = args.final_line.read_text().strip().splitlines()[0]
        for variant in split(args.variants):
            print(
                journey_line(
                    line,
                    args.size,
                    variant,
                    args.steps,
                    profile_steps=args.profile_steps,
                    detail=not args.no_detail,
                    cold=not args.warm,
                )
            )
        return 0
    if args.cmd == "collect":
        parity = (
            json.loads(args.parity.read_text()) if args.parity and args.parity.exists() else None
        )
        ir = json.loads(args.ir.read_text()) if args.ir and args.ir.exists() else None
        journey = collect(args.runs, split(args.sizes), split(args.variants), parity, ir,
                          size_variants={"4B": split(args.variants_4b)})  # fmt: skip
        args.out.mkdir(parents=True, exist_ok=True)
        for r in journey["runs"]:
            (args.out / f"{r['size']}-{r['variant']}.json").write_text(
                json.dumps(r, indent=2) + "\n"
            )
        (args.out / "journey.json").write_text(json.dumps(journey, indent=2) + "\n")
        for r in journey["runs"]:
            row = flat_row(r)
            print(
                json.dumps(
                    {
                        k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in row.items()
                        if v is not None
                    }
                ),
                flush=True,
            )
        if not args.no_wandb:
            charts = (
                sorted(args.charts.glob("*.png")) if args.charts and args.charts.is_dir() else []
            )
            log_wandb(journey, charts)
        return 0
    if args.cmd == "log":
        charts = sorted(args.charts.glob("*.png")) if args.charts and args.charts.is_dir() else []
        log_wandb(json.loads(args.journey.read_text()), charts)
        return 0
    if args.cmd == "ir":
        report = ir_all(split(args.sizes), split(args.variants), args.out, probe=not args.no_probe)
        return 0 if all(r["exit"] == 0 for r in report["runs"].values()) else 1
    if args.cmd == "ir-compile":
        print(json.dumps(ir_compile(args.variant, args.size, args.out)), flush=True)
        return 0
    est = estimate(
        parse_steps(args.steps),
        split(args.variants),
        parity_sizes=split(args.parity_sizes),
        ir_sizes=split(args.ir_sizes),
        ir_variants=split(args.ir_variants),
        size_variants={"4B": split(args.variants_4b)},
    )
    print(estimate_markdown(est))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
