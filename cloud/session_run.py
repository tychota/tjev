"""One time-limited TPU session (Colab v6e-1 / v5e-1): run the campaign up to a deadline,
save what the next session needs to resume.

Run on the session host by cloud/colab.sh (``python session_run.py JOB.json``). Steps:

 1. fail fast unless JAX sees a TPU;
 2. unpack the code and data tarballs into $HOME/tjev-work (HOME = the job's ``home``, a
    scratch dir on the host; the models, 15 GB, stay there and are not saved); ``data`` may
    instead be ``hf:USER/REPO``, a private HF dataset repo (read with ``hf_token``);
 3. cloud/tpu_bootstrap.sh (uv Python 3.12 venv, jax[tpu], the HF snapshots);
 4. restore the previous session's outputs (runs with their checkpoints, reports, queues,
    logs) from ``restore`` dirs; PROBE_FETCH_CKPTS ("RUN@STEP …", RUN the backup path
    <campaign>/<run> or <run>) fetches branch points from the private HF backup (PROBE_HF_REPO);
 5. cloud/tpu_campaign.sh PHASES with DEADLINE_HOURS = the session limit minus setup
    and a save margin: at the deadline every run checkpoints and the campaign pauses;
 6. with PROBE_SWEEP_BACKUP=1 (or set to the repo's name), the finished sweep and final
    runs' checkpoints to the HF backup (PROBE_HF_REPO), under <campaign>/<run>;
 7. copy the outputs to ``save``, keeping only the checkpoints a resume or a pending
    cooldown branch can still use.
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

START = time.time()
KEEP = ("runs", "reports", "queues", "logs")


def relocate_mix(text: str, mix: str, here: Path) -> str:
    """mix.yaml's absolute paths moved under ``here``: whatever directory the mix was built in
    (a laptop's TJEV_ROOT, or /content/out on a Colab CPU build), the part up to ``/<mix>/``."""
    import re

    return re.sub(rf"(?m)(?<=[\s'\"])/[^\s'\"]*?/{re.escape(mix)}/", f"{here}/", text)


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
    """A tarball, a directory holding one, or an already extracted tree: ``marker`` is a
    relative path the extracted tree contains."""
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


def fetch_hf(repo: str, into: Path, token: str | None) -> Path:
    """A private HF dataset repo holding ``data/<mix>/`` and ``data/jevbench/`` (the layout
    of $TJEV_ROOT on the launching machine): download it once, return the snapshot dir."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True
        )
        from huggingface_hub import snapshot_download
    path = snapshot_download(repo, repo_type="dataset", local_dir=into, token=token or None)
    log(f"data from hf:{repo}")
    return Path(path)


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
    # the selected checkpoint feeds post-training and export; a finished sweep run has
    # neither (its metrics, evals and selection.json are what the fit reads), and its
    # ~230 MB would make the next session's restore too large to upload (Colab: HTTP 500)
    if sel.exists() and not (run.name.startswith("sweep-") and (run / "queue.done").exists()):
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


BACKED_UP = ("sweep-", "final-")  # the run kinds PROBE_SWEEP_BACKUP uploads


def backup_path(run: str, campaign: str = "") -> str:
    """A run's path on the HF backup: ``<campaign>/<run>`` (runs of the same name from two
    campaigns, e.g. the reg arms rerun on another mix, overwrote each other's checkpoints
    under the bare ``<run>``), or ``<run>`` without a campaign."""
    return f"{campaign.strip('/')}/{run}" if campaign.strip("/") else run


def backup_sweep_runs(runs: Path, repo: str, token: str | None, campaign: str = "") -> None:
    """Finished sweep and final runs to a private HF model repo before save() drops their
    checkpoints (too large for the next session's restore): every checkpoint left after
    retention (the selected, the latest, train.keep_steps / keep_every), the optimizer state
    of the last one and of the selected one (the EMA adapters, optim.ema_decay, live in it:
    probe_calib's RUN:STEP:ema and a selection on the EMA need it), with the run's json /
    jsonl / txt and eval/, under ``<campaign>/<run>`` (backup_path): the layout tpu_campaign.sh's
    probe_calib reads (PROBE_CALIB_RUNS=<campaign>/<run>:<step>[:ema] runs JevBench on them).
    Never fatal."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token or None)
        assert api.model_info(repo).private, f"{repo} must be private"
        for run in sorted(runs.iterdir()):
            ckpt = run / "checkpoints"
            if not (
                run.name.startswith(BACKED_UP) and (run / "queue.done").exists() and ckpt.is_dir()
            ):
                continue
            steps = sorted(int(p.name) for p in ckpt.iterdir() if p.name.isdigit())
            if not steps:
                continue
            keep = [str(s) for s in steps]
            sel = run / "selection.json"
            selected = str(json.loads(sel.read_text()).get("step", "")) if sel.exists() else ""
            with_opt = list(dict.fromkeys([keep[-1], *([selected] if selected in keep else [])]))
            path = backup_path(run.name, campaign)
            api.upload_folder(
                repo_id=repo, folder_path=str(run), path_in_repo=path,
                ignore_patterns=["tb/**", "wandb/**", "profile/**",
                                 *(f"checkpoints/{s}/opt/**" for s in keep if s not in with_opt)],
                commit_message=f"{path}: checkpoints {', '.join(keep)}",
            )  # fmt: skip
            log(f"backed up {run.name} checkpoints {keep} (optimizer: {', '.join(with_opt)}) "
                f"to hf:{repo}/{path}")  # fmt: skip
    except Exception as e:  # the session's save matters more than the backup
        log(f"sweep backup to hf:{repo} FAILED: {type(e).__name__}: {e}")


def fetch_checkpoints(items: list[str], repo: str, runs: Path, token: str | None) -> None:
    """RUN@STEP checkpoints (with their optimizer state: a cooldown branch's source) from the
    private HF backup into RUNS/<run>/checkpoints/STEP, with the run's json files. RUN is the
    run's backup path: ``<campaign>/<run>`` (backup_path) or a bare ``<run>``."""
    from huggingface_hub import snapshot_download

    for item in items:
        path, _, step = item.partition("@")
        name = path.rsplit("/", 1)[-1]
        stage = runs.parent / "hf-fetch"
        snapshot_download(repo, local_dir=stage, token=token or None,
                          allow_patterns=[f"{path}/*.json", f"{path}/*.jsonl", f"{path}/*.txt",
                                          f"{path}/checkpoints/{step}/**"])  # fmt: skip
        shutil.copytree(stage / path, runs / name, dirs_exist_ok=True)
        shutil.rmtree(stage, ignore_errors=True)
        ok = (runs / name / "checkpoints" / step / "opt").is_dir()
        log(f"fetched {path} checkpoint {step} from hf:{repo} (optimizer state: {ok})")


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
    job = json.loads(Path(sys.argv[1]).read_text())
    home = Path(job["home"])  # a large scratch disk that is not saved as output
    root = home / "tjev-work"
    env = os.environ | {
        "HOME": str(home),
        "TJEV_ROOT": str(root),
        "MIX": job.get("mix", "mix-v7.5"),
        "MODELS": job.get("models", "0.8B 2B 4B"),
        "SIZES": job.get("sizes", "4B,2B,0.8B"),
        "BUDGET": job.get("budget", ""),
        "HARDWARE": job.get("hardware", "v6e1"),
        "CHIPS": str(job.get("chips", 1)),
        "TJEV_WANDB_CAMPAIGN": job.get("campaign", "tpu"),
        "TJEV_WANDB_ENTITY": job.get("wandb_entity", ""),  # "": the key's default entity
        "PATH": f"{home}/.local/bin:{os.environ.get('PATH', '')}",
    }
    # the launcher's PROBE_* settings (PROBE_ONLY, PROBE_SIZES, PROBE_VARIANTS, …)
    env |= {k: str(v) for k, v in job.get("env", {}).items() if k.startswith("PROBE_")}
    if job.get("wandb_key"):
        env["WANDB_API_KEY"] = job["wandb_key"]
    if job.get("hf_token"):
        env["HF_TOKEN"] = job["hf_token"]
    check_tpu()
    unpack(Path(job["code"]), root / "src", "pyproject.toml")
    data = job["data"]
    if data.startswith("hf:"):  # a private HF dataset repo instead of an uploaded tarball
        data = fetch_hf(data.removeprefix("hf:"), home / "hf-data", job.get("hf_token"))
    unpack(Path(data), root, f"data/{env['MIX']}/mix.yaml")
    (root / "src" / "CODE_VERSION").write_text(job.get("code_version", "unknown"))
    # every mix of the data (the campaign's, and siblings an arm reads, e.g. <mix>-rep1)
    for mix_yaml in sorted((root / "data").glob("*/mix.yaml")):
        mix = mix_yaml.parent.name
        mix_yaml.write_text(relocate_mix(mix_yaml.read_text(), mix, mix_yaml.parent))
    if env.get("WANDB_API_KEY"):
        (root / ".wandb_key").write_text(env["WANDB_API_KEY"])
    restore([Path(p) for p in job.get("restore", [])], root)
    if fetch := env.get("PROBE_FETCH_CKPTS"):  # "RUN@STEP ...": branch points from the HF backup
        fetch_checkpoints(fetch.split(), env.get("PROBE_HF_REPO", "tychot/tjev-4b"), root / "runs",
                          job.get("hf_token"))  # fmt: skip
    subprocess.run(["bash", str(root / "src/cloud/tpu_bootstrap.sh")], env=env, check=True)
    # the session limit, minus what setup used and a margin to save the outputs
    left = job["session_hours"] - (time.time() - START) / 3600 - job.get("save_margin_hours", 0.4)
    env["DEADLINE_HOURS"] = f"{max(left, 0.1):.2f}"
    log(f"campaign {job['phases']} with a {env['DEADLINE_HOURS']} h deadline")
    code = subprocess.run(
        ["bash", str(root / "src/cloud/tpu_campaign.sh"), *job["phases"].split()], env=env
    ).returncode
    log(f"campaign exited {code} (0 done, 3 paused: run the next session to resume)")
    # PROBE_SWEEP_BACKUP=1 (or, the older form, the repo's own name) turns the backup on
    backup = env.get("PROBE_SWEEP_BACKUP", "")
    if backup not in ("", "0") and (root / "runs").is_dir():
        repo = backup if "/" in backup else env.get("PROBE_HF_REPO", "tychot/tjev-4b")
        backup_sweep_runs(root / "runs", repo, job.get("hf_token"), env["TJEV_WANDB_CAMPAIGN"])
    save(root, Path(job["save"]))
    return 0  # the session's output is kept whatever happened; status is in logs/campaign.*


if __name__ == "__main__":
    sys.exit(main())
