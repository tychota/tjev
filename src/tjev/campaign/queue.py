"""Run a job queue on one TPU VM: several jobs at once, each on its own chip group.

    tjev campaign queue QUEUE --runs RUNS --mix MIX --jevbench PUBLIC.jsonl [--chips 8]

QUEUE lines (written by ``tjev campaign plan``)::

    <name> chips=N [post=quick|full|none] [after=RUN | after=RUN@STEP] <tjev train args ...>

* A job gets N chips (1, 2, 4 or 8, aligned groups) through libtpu's per-process chip
  variables (TPU_VISIBLE_CHIPS, TPU_CHIPS_PER_PROCESS_BOUNDS, ...). A preflight checks
  that a process really sees N devices for each group size; if not, every job runs on all
  chips, one at a time (throughput per chip is the same within ~2%).
* ``after=RUN@STEP`` waits for that run's checkpoint STEP (a WSD cooldown branch starts
  as soon as its branch point exists); ``after=RUN`` waits for RUN to finish.
* A failed or stalled run (metrics.jsonl unchanged for --stall minutes) restarts from its
  last checkpoint (exact resume) up to --retries times.
* A finished run leaves RUNS/<name>/queue.done (then ``post``: ``tjev post``, quick or
  full, on the same chips) and is skipped when the queue is run again.
* ``$RUNS`` in the arguments is replaced by the runs directory.
* SIGTERM (VM shutdown) is forwarded to the runs, which checkpoint and exit 143; a run that
  exits 143 is interrupted, not failed. ``--bucket gs://…`` mirrors RUNS there every 10
  minutes and at the end (spot VMs: restore it on a new VM before running again).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

BOUNDS = {  # candidate chip bounds per group size on a 2x4 v6e-8 host (preflight picks one)
    1: ["1,1,1"],
    2: ["2,1,1", "1,2,1"],
    4: ["2,2,1", "1,4,1"],
}


@dataclass
class Job:
    name: str
    chips: int
    args: list[str]
    post: str = "quick"
    after: str | None = None
    tries: int = 0
    proc: subprocess.Popen | None = None
    group: list[int] = field(default_factory=list)
    phase: str = "train"  # train -> post -> done | failed
    last_size: int = -1
    last_change: float = 0.0


def parse_lines(lines: list[str], runs: Path) -> list[Job]:
    jobs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, *rest = line.split()
        opts, args = {}, []
        for token in rest:
            key = token.split("=", 1)[0]
            if key in ("chips", "post", "after") and "=" in token:
                opts[key] = token.split("=", 1)[1]
            else:
                args.append(token.replace("$RUNS", str(runs)))
        jobs.append(Job(name, int(opts.get("chips", 1)), args, opts.get("post", "quick"),
                        opts.get("after")))  # fmt: skip
    return jobs


def parse(path: Path, runs: Path) -> list[Job]:
    return parse_lines(path.read_text().splitlines(), runs)


def sync(runs: Path, bucket: str) -> None:
    subprocess.run(["gcloud", "storage", "rsync", "-r", "-q", str(runs), bucket], check=False)


def chip_env(group: list[int], total: int, bounds: dict[int, str]) -> dict[str, str]:
    if len(group) == total:
        return {}
    port = 8476 + group[0]
    return {
        "TPU_VISIBLE_CHIPS": ",".join(map(str, group)),
        "TPU_CHIPS_PER_PROCESS_BOUNDS": bounds[len(group)],
        "TPU_PROCESS_BOUNDS": "1,1,1",
        "TPU_PROCESS_PORT": str(port),
        "TPU_PROCESS_ADDRESSES": f"localhost:{port}",
        "ALLOW_MULTIPLE_LIBTPU_LOAD": "1",
    }


def preflight(sizes: set[int], total: int) -> dict[int, str] | None:
    """Chip bounds that give a process exactly N devices, per group size; None if any
    size fails (then jobs run on all chips, one at a time)."""
    chosen = {}
    probe = "import jax; print(jax.device_count())"
    for size in sorted(s for s in sizes if s < total):
        for candidate in BOUNDS.get(size, []):
            env = os.environ | chip_env(list(range(size)), total, {size: candidate})
            try:
                out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True,
                                     text=True, timeout=300)  # fmt: skip
                if out.returncode == 0 and out.stdout.strip().splitlines()[-1] == str(size):
                    chosen[size] = candidate
                    break
            except (subprocess.TimeoutExpired, IndexError):
                pass
        if size not in chosen:
            print(f"[queue] preflight: no chip bounds give {size} devices; serial mode")
            return None
    print(f"[queue] preflight ok: {chosen or 'all jobs on all chips'}")
    return chosen


def ready(job: Job, runs: Path, jobs: dict[str, Job]) -> bool | None:
    """True/False, or None when the dependency failed (the job can never start)."""
    if not job.after:
        return True
    run, _, step = job.after.partition("@")
    parent = jobs.get(run)
    if parent is not None and parent.phase == "failed" and not (runs / run / "queue.done").exists():
        return None
    if step:
        return (runs / run / "checkpoints" / step).is_dir()
    return (runs / run / "queue.done").exists()


@dataclass(frozen=True)
class PostInputs:
    """What ``tjev post`` needs after each run: the mix, JevBench public, the controls."""

    mix: str
    jevbench: str
    zeroshot: str = ""  # a directory of eval-base reports zeroshot-<size>.json (optional)


def _size(args: list[str]) -> str:
    return next((a for a in args if a.startswith("qwen35-")), "qwen35-?")[len("qwen35-") :].upper()


def start(job: Job, runs: Path, post: PostInputs, total: int, bounds) -> None:
    run_dir = runs / job.name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ | chip_env(job.group, total, bounds or {})
    log = (run_dir / ("stdout.log" if job.phase == "train" else "post.log")).open("a")
    if job.phase == "train":
        cmd = [sys.executable, "-m", "tjev", "train", *job.args,
               f"name={job.name}", f"output={runs}"]  # fmt: skip
    else:
        cmd = [sys.executable, "-m", "tjev", "post", str(run_dir), "--mix", post.mix,
               "--jevbench", post.jevbench]  # fmt: skip
        control = (
            Path(post.zeroshot) / f"zeroshot-{_size(job.args)}.json" if post.zeroshot else None
        )
        if control is not None and control.exists():
            cmd += ["--zeroshot", str(control)]
        if job.post == "quick":
            cmd.append("--quick")
    print(f"[queue] {time.strftime('%H:%M:%S')} start {job.phase} {job.name} chips={job.group}")
    job.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()  # the child keeps its own descriptor
    job.last_size, job.last_change = -1, time.time()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tjev campaign queue")
    parser.add_argument("queue", type=Path)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--mix", required=True, help="the built mix directory")
    parser.add_argument("--jevbench", default="", help="JevBench public.jsonl (post-training)")
    parser.add_argument("--zeroshot", default="", help="directory of zeroshot-<size>.json")
    parser.add_argument("--chips", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--stall", type=float, default=30.0, help="minutes without metrics")
    parser.add_argument("--bucket", help="gs://… mirror of the runs directory")
    parser.add_argument(
        "--deadline-hours", type=float, default=0.0,
        help="session limit (Kaggle 9 h, Colab): at it, every run checkpoints and the queue "
             "exits 3 (paused); re-running the queue in the next session resumes them",
    )  # fmt: skip
    args = parser.parse_args(argv)
    runs, total = args.runs, args.chips
    post = PostInputs(args.mix, args.jevbench, args.zeroshot)
    runs.mkdir(parents=True, exist_ok=True)
    jobs = {j.name: j for j in parse(args.queue, runs)}
    for job in jobs.values():
        if (runs / job.name / "queue.done").exists():
            job.phase = "done"
    for job in jobs.values():  # a smaller VM than planned: the job takes every chip
        job.chips = min(job.chips, total)
    bounds = preflight({j.chips for j in jobs.values() if j.phase != "done"}, total)
    if bounds is None:
        for job in jobs.values():
            job.chips = total
    free = set(range(total))
    stopping = threading.Event()

    def on_term(*_):
        stopping.set()
        for job in jobs.values():
            if job.proc is not None and job.proc.poll() is None:
                job.proc.send_signal(signal.SIGTERM)

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, on_term)
    last_sync = time.time()
    deadline = time.time() + args.deadline_hours * 3600 if args.deadline_hours else None
    paused = False

    def release(job: Job) -> None:
        free.update(job.group)
        job.group, job.proc = [], None

    while any(j.phase in ("train", "post") for j in jobs.values()):
        if args.bucket and time.time() - last_sync > 600:
            threading.Thread(target=sync, args=(runs, args.bucket), daemon=True).start()
            last_sync = time.time()
        if deadline and not stopping.is_set() and time.time() > deadline:
            print("[queue] session deadline: checkpointing every run and pausing")
            paused = True
            on_term()
        if stopping.is_set() and all(
            j.proc is None or j.proc.poll() is not None for j in jobs.values()
        ):
            print("[queue] stopped; runs resume from their checkpoints")
            break
        for job in jobs.values():  # finished processes
            if job.proc is None or job.proc.poll() is None:
                continue
            code = job.proc.returncode
            if job.phase == "post":
                if code:
                    print(f"[queue] post {job.name} failed ({code}); training result kept")
                job.phase = "done"
                release(job)
            elif code == 0:
                (runs / job.name / "queue.done").touch()
                print(f"[queue] {time.strftime('%H:%M:%S')} done {job.name}")
                job.phase = "post" if job.post != "none" and post.jevbench else "done"
                if job.phase == "post":
                    start(job, runs, post, total, bounds)  # keeps its chips
                else:
                    release(job)
            elif code == 143 or stopping.is_set():  # interrupted: checkpointed, not failed
                release(job)  # restarted by the scheduling pass below unless stopping
            else:
                job.tries += 1
                print(f"[queue] {job.name} exited {code} (try {job.tries}/{args.retries + 1})")
                if job.tries > args.retries:
                    job.phase = "failed"
                    release(job)
                else:
                    start(job, runs, post, total, bounds)
        for job in jobs.values():  # stall watchdog (training only)
            if job.proc is None or job.phase != "train":
                continue
            metrics = runs / job.name / "metrics.jsonl"
            size = metrics.stat().st_size if metrics.exists() else 0
            if size != job.last_size:
                job.last_size, job.last_change = size, time.time()
            elif time.time() - job.last_change > args.stall * 60:
                print(f"[queue] {job.name} stalled {args.stall:.0f} min: killing")
                job.proc.kill()
        for job in jobs.values():  # start what fits, in queue order
            if stopping.is_set() or job.phase != "train" or job.proc is not None:
                continue
            state = ready(job, runs, jobs)
            if state is None:
                print(f"[queue] {job.name}: dependency {job.after} failed; skipped")
                job.phase = "failed"
                continue
            if not state:
                continue
            groups = [list(range(i, i + job.chips)) for i in range(0, total, job.chips)]
            group = next((g for g in groups if set(g) <= free), None)
            if group is None:
                continue
            free.difference_update(group)
            job.group = group
            start(job, runs, post, total, bounds)
        time.sleep(10)
    if args.bucket:
        sync(runs, args.bucket)
    failed = [j.name for j in jobs.values() if j.phase == "failed"]
    print(f"[queue] {'paused' if paused else 'finished'}; failed: {failed or 'none'}")
    if paused:
        return 3
    return 1 if failed else 0
