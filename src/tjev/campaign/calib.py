"""A backed-up checkpoint made a run directory ``tjev post`` scores (cloud/tpu_campaign.sh,
``PROBE_ONLY=calib``, docs/calibration.md).

    python -m tjev.campaign.calib fetch REPO ENTRY RUNS   # prints: LOCAL_DIR HF_UPLOAD_PATH

ENTRY (one of PROBE_CALIB_RUNS): ``RUN:STEP`` scores the adapters of that checkpoint,
``RUN:STEP:ema`` their EMA (``optim.ema_decay``: it lives in the optimizer state, so the
backup must hold that step's ``opt/``, as cloud/session_run.py keeps it for the selected and
the last step). RUN is the run's path on the HF backup: ``<campaign>/<run>`` (the layout of
cloud/session_run.py's backup) or a bare ``<run>`` (the backups before the campaign prefix).
The checkpoint becomes the run's selection (``selection.json``: its step, and ``"weights":
"ema"`` for the EMA; tjev.eval.runs loads the weights the selection names); the original
selection stays in ``selection.orig.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

WEIGHTS = ("raw", "ema")


@dataclass(frozen=True)
class Entry:
    hf_run: str  # the run's path on the HF backup (``<campaign>/<run>`` or ``<run>``)
    step: int
    weights: str = "raw"

    @property
    def run(self) -> str:
        """The run's own name (its directory under RUNS)."""
        return self.hf_run.rsplit("/", 1)[-1]

    @property
    def tag(self) -> str:
        """``<step>`` or ``<step>-ema``: the local directory's and the upload's suffix."""
        return f"{self.step}-ema" if self.weights == "ema" else str(self.step)

    @property
    def local(self) -> str:
        return f"{self.run}@{self.tag}"

    @property
    def upload(self) -> str:
        """Where the post/ outputs go back on the HF backup, next to the run."""
        return f"{self.hf_run}/post-calib-{self.tag}"


def parse_entry(entry: str) -> Entry:
    """``RUN:STEP`` or ``RUN:STEP:ema`` (``raw`` accepted too)."""
    parts = entry.split(":")
    if len(parts) not in (2, 3) or not parts[0] or not parts[1].isdigit():
        raise ValueError(f"calib entry {entry!r}: RUN:STEP or RUN:STEP:ema")
    weights = parts[2] if len(parts) == 3 else "raw"
    if weights not in WEIGHTS:
        raise ValueError(f"calib entry {entry!r}: weights {weights!r} not in {WEIGHTS}")
    return Entry(parts[0].strip("/"), int(parts[1]), weights)


def prepare(src: Path, dest: Path, entry: Entry) -> Path:
    """``src`` (the downloaded run) copied to ``dest`` with the entry's checkpoint as its
    selection; refuses an EMA entry whose checkpoint has no optimizer state."""
    ckpt = src / "checkpoints" / str(entry.step)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"no checkpoint {entry.step} of {entry.hf_run} on the backup")
    if entry.weights == "ema" and not (ckpt / "opt").is_dir():
        raise FileNotFoundError(
            f"{entry.hf_run}@{entry.step}: no optimizer state backed up (the EMA lives there)"
        )
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(src, dest)
    sel_path = dest / "selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    (dest / "selection.orig.json").write_text(json.dumps(sel))
    sel["step"] = entry.step  # post reads its checkpoint (and weights) from the selection
    if entry.weights == "ema":
        sel["weights"] = "ema"
    else:
        sel.pop("weights", None)  # the adapters, whatever the run selected on
    sel_path.write_text(json.dumps(sel))
    (dest / "queue.done").touch()
    return dest


def fetch(repo: str, entry: Entry, runs: Path) -> Path:
    """The entry's checkpoint (with its optimizer state when backed up), the run's json /
    jsonl / txt files and eval/ from the private HF backup, prepared under RUNS."""
    from huggingface_hub import HfApi, snapshot_download

    assert HfApi().model_info(repo).private, "the backup repo must be private"
    r = entry.hf_run
    tmp = Path(snapshot_download(repo, allow_patterns=[
        f"{r}/*.json", f"{r}/*.jsonl", f"{r}/*.txt", f"{r}/eval/*",
        f"{r}/checkpoints/{entry.step}/**"], local_dir=str(runs / "hf-calib")))  # fmt: skip
    return prepare(tmp / r, runs / entry.local, entry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tjev.campaign.calib")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="a backed-up checkpoint as a scorable run dir")
    f.add_argument("repo")
    f.add_argument("entry", help="RUN:STEP or RUN:STEP:ema")
    f.add_argument("runs", type=Path)
    args = parser.parse_args(argv)
    entry = parse_entry(args.entry)
    dest = fetch(args.repo, entry, args.runs)
    print(f"{dest} {entry.upload}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
