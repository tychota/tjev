"""The profile-analysis probe (``PROBE_ONLY=analyse``, docs/profiling.md): per model size, the
planner's run on one v6e chip, captured three ways.

* ``aggregate``: a short timed run (``steps``, every bucket precompiled), ``traced`` steps
  profiled after the warm-up in the timing trace mode: the step tables (xprof_tools).
* ``detail``: one step in ``TRACE_COMPUTE_AND_SYNC`` (sync-flag and DMA waits; the shell adds
  the LLO debug info to ``LIBTPU_INIT_ARGS``): for the timeline only, never for timing.
* ``hlo``: ``tjev compile-check`` of the same items with an XLA dump (the optimized HLO, the
  buffer assignment, the memory-usage report): the static side of the same step.

    python -m tjev.campaign.analyse line --plan FINAL.plan --size 9B --mode aggregate
    python -m tjev.campaign.analyse items --line-file analyse-9b-aggregate.line
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tjev.campaign.journey import check_final_line, final_line_for

# replaced on the final line: the run length, its saves and evals, logging, the queue head
STRIPPED = (
    "train.steps=",
    "train.checkpoint_every=",
    "train.checkpoint_secs=",
    "train.checkpoint_tail_every=",
    "train.keep_checkpoints=",
    "train.keep_steps=",
    "train.quick_eval_every=",
    "train.full_eval_every=",
    "train.profile_",
    "train.health_kill=",
    "train.branch_from=",
    "train.precompile_buckets=",
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
MODES = ("aggregate", "detail")
WANDB_PROJECT = "tjev-benchmarks"
WANDB_GROUP = "analyse"
WARMUP = 3  # steps after the precompile before a trace starts (allocator, first dispatches)


def run_name(size: str, mode: str) -> str:
    return f"analyse-{size.lower()}-{mode}"


def window(mode: str, steps: int, traced: int) -> tuple[int, int, int]:
    """(train.steps, train.profile_start, train.profile_steps) of a capture. The loop traces
    ``profile_steps + 1`` steps (start_trace before step ``profile_start``, stop_trace after
    step ``profile_start + profile_steps``): ``traced`` steps are ``profile_steps =
    traced - 1``. aggregate: the window ends 2 steps before the run (its last steps carry
    the end-of-run work); detail: one step after the warm-up, then the run ends."""
    if mode == "detail":
        return WARMUP + 2, WARMUP + 1, 0
    if traced < 1 or steps < WARMUP + traced + 2:
        raise ValueError(f"{steps} steps leave no room for a {traced}-step trace")
    start = steps - traced - 1
    return steps, start, traced - 1


def analyse_line(
    final_line: str,
    size: str,
    mode: str,
    *,
    steps: int = 25,
    traced: int = 3,
    rank: int | None = 16,
    extra: list[str] | None = None,
    unroll: bool = False,
) -> str:
    """The queue line of one capture from the planner's final line of ``size`` (its compute,
    microbatch and LoRA targets kept; ``rank`` replaces the planner's; ``extra`` last)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: one of {MODES}")
    tokens = final_line.split()
    if not tokens:
        raise ValueError("empty final line")
    check_final_line(final_line, size)
    kept = [t for t in tokens[1:] if not t.startswith(STRIPPED)]
    if rank:
        kept = [t for t in kept if not t.startswith("lora.rank=")] + [f"lora.rank={rank}"]
    total, start, count = window(mode, steps, traced)
    env = {}
    if mode == "detail":  # sync-flag / semaphore waits: the timeline, not the timing
        env["TJEV_PROFILE_DETAIL"] = "1"
        # the GDN megakernel's named regions (prologue, chunk, state update, …: trace marks
        # inside the kernel, so only in this untimed capture)
        env["TJEV_KERNEL_REGIONS"] = "1"
        if unroll:  # the layers traced one by one: layer_00 … layer_NN in every op name
            env["TJEV_UNROLL_LAYERS"] = "1"
    args = [
        *kept,
        f"train.steps={total}",
        "train.checkpoint_every=0",  # a benchmark: no checkpoints, no full evals
        "train.checkpoint_secs=0",
        "train.quick_eval_every=0",  # no eval program inside the window
        "train.health_kill=false",
        "train.log_every=1",
        "train.precompile_buckets=true",  # every bucket compiled before step 1
        f"train.profile_start={start}",
        f"train.profile_steps={count}",
    ]
    if "log.wandb=true" in kept:
        args += [
            f"log.wandb_project={WANDB_PROJECT}",
            f"log.wandb_group={WANDB_GROUP}",
            f"log.wandb_tags=[analyse,{size},{mode}]",
        ]
    args += list(extra or [])
    head = [run_name(size, mode), "chips=1", "post=none", *(f"env={k}={v}" for k, v in env.items())]
    return " ".join(head + args)


def compile_items(line: str) -> list[str]:
    """``tjev compile-check`` items of a queue line: the presets, the mix and the overrides
    (the queue head, ``name=chips=post=env=after=``, and the run's own settings dropped)."""
    tokens = line.split()[1:]
    head = ("chips=", "post=", "env=", "after=", "name=", "output=", "log.")
    return [t for t in tokens if not t.startswith(head)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tjev.campaign.analyse")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("line", help="a capture's queue line from the planner's final plan")
    p.add_argument("--plan", type=Path, required=True, help="tjev campaign plan final output")
    p.add_argument("--size", required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--horizon", type=int, default=3000)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--traced", type=int, default=3, help="aggregate: steps in the trace")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank (0: the planner's)")
    p.add_argument("--extra", default="", help="more overrides, space-separated")
    p.add_argument("--unroll", action="store_true", help="detail: TJEV_UNROLL_LAYERS=1")
    i = sub.add_parser("items", help="compile-check items of a queue line")
    i.add_argument("--line-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "line":
        lines = args.plan.read_text(encoding="utf-8").splitlines()
        final = final_line_for(lines, args.size, args.horizon)
        print(
            analyse_line(final, args.size, args.mode, steps=args.steps, traced=args.traced,
                         rank=args.rank or None, extra=args.extra.split(),
                         unroll=args.unroll and args.mode == "detail")
        )  # fmt: skip
    else:
        print(" ".join(compile_items(args.line_file.read_text(encoding="utf-8").strip())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
