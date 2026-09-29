"""One time-limited TPU session (Kaggle v5e-8, Colab v6e-1/v5e-1): run the campaign up to a
deadline, save what the next session needs to resume.

Run on the session host by cloud/kaggle.sh (as the kernel script, with JOB filled in) or by
cloud/colab.sh (``python session_run.py JOB.json``). Steps:

 1. fail fast unless JAX sees a TPU (Kaggle can silently give a CPU image: kaggle-cli #1197);
 2. unpack the code and data tarballs into $HOME/tjev-work (HOME = a scratch dir on the host;
    the models, 15 GB, stay there and are not saved);
 3. cloud/tpu_bootstrap.sh (uv Python 3.12 venv, jax[tpu], the HF snapshots);
 4. restore the previous session's outputs (runs with their checkpoints, reports, queues,
    logs) from ``restore`` dirs;
 5. cloud/tpu_campaign.sh PHASES with DEADLINE_HOURS = the session limit minus setup
    and a save margin: at the deadline every run checkpoints and the campaign pauses;
 6. copy the outputs to ``save`` (Kaggle: /kaggle/working, kept as the kernel output),
    keeping only the checkpoints a resume or a pending cooldown branch can still use.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

JOB: dict = {}  # cloud/kaggle.sh writes the job here (Kaggle kernels take no arguments)
START = time.time()
KEEP = ("runs", "reports", "queues", "logs")


def log(msg: str) -> None:
    print(f"[session] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def check_tpu() -> None:
    out = subprocess.run(
        [sys.executable, "-c", "import jax; print(jax.devices()[0].platform, jax.device_count())"],
        capture_output=True, text=True,
    )  # fmt: skip
    log(f"jax devices: {out.stdout.strip() or out.stderr.strip()[-300:]}")
    # the preinstalled JAX may be missing or too old (the venv installs ours): only a host
    # without TPU chips is fatal
    no_chips = not Path("/dev/accel0").exists() and not Path("/dev/vfio").exists()
    if not out.stdout.startswith("tpu") and no_chips:
        raise SystemExit("no TPU on this host (check the accelerator setting)")


def unpack(source: Path, into: Path, marker: str) -> None:
    """A tarball, a directory holding one, or an already extracted tree (Kaggle may unpack
    uploaded archives): ``marker`` is a relative path the extracted tree contains."""
    into.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        found = next((p.parent for p in source.rglob(marker) if p.exists()), None)
        if found is not None:  # extracted: the tree root is where the marker sits
            root = found
            for _ in Path(marker).parent.parts:
                root = root.parent
            shutil.copytree(root, into, dirs_exist_ok=True)
            return
        source = next(p for p in source.rglob("*") if p.name.endswith((".tgz", ".tar.gz", ".tar")))
    with tarfile.open(source) as tar:
        tar.extractall(into, filter="data")


def restore(sources: list[Path], root: Path) -> None:
    """Copy each previous session's saved outputs (the newest last) into the root."""
    for src in sources:
        base = src / "tjev-work"
        if not base.is_dir():
            continue
        for name in KEEP:
            if (base / name).is_dir():
                shutil.copytree(base / name, root / name, dirs_exist_ok=True)
        log(f"restored {base}")


def needed_checkpoints(run: Path, runs: Path) -> set[str]:
    """Checkpoint steps worth saving: the latest (resume), the selected one, and any a
    pending cooldown branch will start from."""
    ckpt = run / "checkpoints"
    steps = sorted(int(p.name) for p in ckpt.iterdir() if p.name.isdigit()) if ckpt.is_dir() else []
    keep = set()
    if steps and not (run / "queue.done").exists():
        keep.add(str(steps[-1]))
    sel = run / "selection.json"
    if sel.exists():
        keep.add(str(json.loads(sel.read_text())["step"]))
    for other in runs.iterdir():  # branches not yet finished that start from this run
        cfg = other / "config.json"
        if other == run or (other / "queue.done").exists() or not cfg.exists():
            continue
        branch = json.loads(cfg.read_text())["config"]["train"].get("branch_from", "")
        if branch and Path(branch.rpartition("@")[0]).name == run.name:
            keep.add(branch.rpartition("@")[2])
    return keep


def pending_branch_points(runs: Path, queues: Path) -> dict[str, set[str]]:
    """Branch points of queued jobs that have not started (no run dir yet)."""
    out: dict[str, set[str]] = {}
    for q in queues.glob("*.queue"):
        for line in q.read_text().splitlines():
            fields = dict(t.split("=", 1) for t in line.split() if "=" in t)
            name = line.split()[0] if line.strip() else ""
            src = fields.get("train.branch_from", "")
            if src and not (runs / name / "queue.done").exists():
                run, _, step = src.rpartition("@")
                out.setdefault(Path(run).name, set()).add(step)
    return out


def save(root: Path, dest: Path) -> None:
    target = dest / "tjev-work"
    if target.exists():
        shutil.rmtree(target)
    runs = root / "runs"
    pending = pending_branch_points(runs, root / "queues") if runs.is_dir() else {}
    for name in KEEP:
        if not (root / name).is_dir():
            continue
        if name != "runs":
            shutil.copytree(root / name, target / name)
            continue
        for run in runs.iterdir():
            keep = needed_checkpoints(run, runs) | pending.get(run.name, set())
            shutil.copytree(
                run, target / "runs" / run.name,
                ignore=lambda d, files, run=run, keep=keep: (
                    [f for f in files if f not in keep] if Path(d) == run / "checkpoints" else []
                ),
            )  # fmt: skip
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e9
    log(f"saved {target} ({size:.1f} GB)")


def main() -> int:
    job = dict(JOB)
    if len(sys.argv) > 1:
        job |= json.loads(Path(sys.argv[1]).read_text())
    if job["home"] == "auto":  # a large scratch disk that is not saved as output
        job["home"] = "/kaggle/temp/home" if Path("/kaggle/temp").is_dir() else "/tmp/tjev-home"
    home = Path(job["home"])
    root = home / "tjev-work"
    env = os.environ | {
        "HOME": str(home),
        "TJEV_ROOT": str(root),
        "MIX": job.get("mix", "mix-v3"),
        "MODELS": job.get("models", "0.8B 2B 4B"),
        "SIZES": job.get("sizes", "4B,2B,0.8B"),
        "HARDWARE": job.get("hardware", "v5e"),
        "CHIPS": str(job.get("chips", 8)),
        "TJEV_WANDB_CAMPAIGN": job.get("campaign", "tpu"),
        "PATH": f"{home}/.local/bin:{os.environ.get('PATH', '')}",
    }
    if job.get("wandb_key"):
        env["WANDB_API_KEY"] = job["wandb_key"]
    check_tpu()
    unpack(Path(job["code"]), root / "src", "pyproject.toml")
    unpack(Path(job["data"]), root, f"data/{env['MIX']}/mix.yaml")
    (root / "src" / "CODE_VERSION").write_text(job.get("code_version", "unknown"))
    mix_yaml = root / "data" / env["MIX"] / "mix.yaml"
    mix_yaml.write_text(mix_yaml.read_text().replace(job["data_root"], str(root)))
    if env.get("WANDB_API_KEY"):
        (root / ".wandb_key").write_text(env["WANDB_API_KEY"])
    restore([Path(p) for p in job.get("restore", [])], root)
    subprocess.run(["bash", str(root / "src/cloud/tpu_bootstrap.sh")], env=env, check=True)
    # the session limit, minus what setup used and a margin to save the outputs
    left = job["session_hours"] - (time.time() - START) / 3600 - job.get("save_margin_hours", 0.4)
    env["DEADLINE_HOURS"] = f"{max(left, 0.1):.2f}"
    log(f"campaign {job['phases']} with a {env['DEADLINE_HOURS']} h deadline")
    code = subprocess.run(
        ["bash", str(root / "src/cloud/tpu_campaign.sh"), *job["phases"].split()], env=env
    ).returncode
    log(f"campaign exited {code} (0 done, 3 paused: run the next session to resume)")
    save(root, Path(job["save"]))
    return 0  # the kernel succeeds so its output is kept; status is in logs/campaign.*


if __name__ == "__main__":
    sys.exit(main())
