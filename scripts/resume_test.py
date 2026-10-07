"""Resume on the real hardware: one run straight through, its twin stopped with SIGTERM near
step ``--stop-at`` and resumed, both from the planner's final line of a size (shortened to
``--steps``). Same seed, same batches: every step after the resume must match the straight run
(data fields exactly, losses and norms to bf16 noise) and so must the final adapters.

    python scripts/resume_test.py --line FINAL_LINE_FILE --runs DIR [--steps 120 --stop-at 60]

Prints one JSON verdict line ({"resume": "sound" | "mismatch", ...}); exit 0 on sound.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DATA = ("data/seq_len", "data/segments_consumed", "data/slots_per_step", "data/pad_fraction")
# the line's own horizon, cadence and branch settings: replaced by the test's
DROP = ("train.steps", "train.checkpoint_every", "train.keep_steps", "train.checkpoint_tail_every",
        "train.full_eval_tail_every", "train.quick_eval_every", "train.branch_from")  # fmt: skip
CLOSE = ("learning/loss", "learning/nll", "learning/grad_norm", "health/param_norm_a",
         "health/param_norm_b")  # fmt: skip


def job_args(line: str, steps: int, every: int) -> tuple[list[str], dict[str, str]]:
    """The train arguments of a queue line, shortened and checkpointed every ``every``."""
    _, *rest = line.split()
    args, env = [], {}
    for token in rest:
        key = token.split("=", 1)[0]
        if key == "env" and token.count("=") >= 2:
            var, value = token.split("=", 2)[1:]
            env[var] = value
        elif key in ("chips", "post", "after") or key.startswith(DROP):
            continue
        else:
            args.append(token)
    args += [f"train.steps={steps}", f"train.checkpoint_every={every}", "train.quick_eval_every=0",
             "train.keep_checkpoints=20"]  # fmt: skip
    return args, env


def train(args: list[str], env: dict[str, str], name: str, runs: Path, stop_at: int | None = None):
    """``tjev train``; with ``stop_at``, SIGTERM once metrics.jsonl reaches that step."""
    run = runs / name
    run.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "tjev", "train", *args, f"name={name}", f"output={runs}"]
    with (run / "stdout.log").open("a") as log:
        proc = subprocess.Popen(cmd, env=os.environ | env, stdout=log, stderr=subprocess.STDOUT)
        sent = None
        while proc.poll() is None:
            time.sleep(2)
            if stop_at is not None and sent is None and (step := last_step(run)) >= stop_at:
                proc.send_signal(signal.SIGTERM)
                sent = step
        return proc.returncode, sent


def last_step(run: Path) -> int:
    path = run / "metrics.jsonl"
    if not path.exists():
        return 0
    steps = [json.loads(x).get("step", 0) for x in path.read_text().splitlines() if x.strip()]
    return max(steps, default=0)


def rows(run: Path) -> dict[int, dict]:
    out = {}
    for line in (run / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "learning/loss" in r:
            out[r["step"]] = r  # a resumed step logged twice keeps its last record
    return out


def final_adapters(run: Path, step: int) -> dict:
    import numpy as np
    import orbax.checkpoint as ocp

    path = run / "checkpoints" / str(step) / "lora"
    tree = ocp.PyTreeCheckpointer().restore(str(path))
    return {k: np.asarray(v, dtype=np.float64) for k, v in tree.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--line", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--stop-at", type=int, default=60)
    p.add_argument("--every", type=int, default=30)
    a = p.parse_args()
    runs = Path(a.runs)
    args, env = job_args(Path(a.line).read_text().strip(), a.steps, a.every)
    rc_a, _ = train(args, env, "resume-straight", runs)
    rc_b1, sent = train(args, env, "resume-stopped", runs, stop_at=a.stop_at)
    rc_b2, _ = train(args, env, "resume-stopped", runs)
    straight, resumed = rows(runs / "resume-straight"), rows(runs / "resume-stopped")
    steps = sorted(set(straight) & set(resumed))
    data = [(s, k) for s in steps for k in DATA if straight[s].get(k) != resumed[s].get(k)]
    rel = {}
    for k in CLOSE:
        d = [abs(resumed[s][k] - straight[s][k]) / (abs(straight[s][k]) + 1e-12)
             for s in steps if k in straight[s] and k in resumed[s]]  # fmt: skip
        rel[k] = max(d, default=None)
    adapters = None
    try:
        x, y = (
            final_adapters(runs / "resume-straight", a.steps),
            final_adapters(runs / "resume-stopped", a.steps),
        )
        adapters = max(float(abs(x[k] - y[k]).max() / (abs(x[k]).max() + 1e-12)) for k in x)
    except Exception as e:  # reported, not fatal
        adapters = f"unreadable: {type(e).__name__}: {e}"
    sound = (rc_a == 0 and rc_b1 == 143 and rc_b2 == 0 and not data and len(steps) == a.steps
             and all(v is not None and v < 1e-2 for v in rel.values())
             and isinstance(adapters, float) and adapters < 1e-2)  # fmt: skip
    print(json.dumps({"resume": "sound" if sound else "mismatch", "sigterm_at": sent,
                      "exit": [rc_a, rc_b1, rc_b2], "common_steps": len(steps),
                      "data_mismatches": data[:5], "max_rel": rel, "adapters_max_rel": adapters}))  # fmt: skip
    return 0 if sound else 1


if __name__ == "__main__":
    raise SystemExit(main())
