"""Time-limited sessions (Kaggle, Colab): what cloud/session_run.py saves and unpacks."""

import importlib.util
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("session_run", ROOT / "cloud" / "session_run.py")
assert _spec is not None and _spec.loader is not None
session_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(session_run)


def _run(runs: Path, name: str, steps: list[int], *, done: bool, selected=None, branch=""):
    run = runs / name
    run.mkdir(parents=True)
    for s in steps:
        (run / "checkpoints" / str(s)).mkdir(parents=True)
        (run / "checkpoints" / str(s) / "x").write_text("x")
    cfg = {"config": {"train": {"branch_from": branch}}}
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "metrics.jsonl").write_text("{}\n")
    if selected is not None:
        (run / "selection.json").write_text(json.dumps({"step": selected}))
    if done:
        (run / "queue.done").touch()


def test_save_keeps_only_resume_selection_and_branch_checkpoints(tmp_path):
    root, runs = tmp_path / "tjev-work", tmp_path / "tjev-work" / "runs"
    _run(runs, "done", [10, 20, 30], done=True, selected=20)
    _run(runs, "running", [5, 10], done=False, selected=5)
    _run(runs, "long", [48, 96, 240], done=True, selected=240)
    _run(runs, "started-branch", [], done=False, branch=f"{runs}/long@48")
    (root / "queues").mkdir()
    (root / "queues" / "transfer.queue").write_text(
        "not-started chips=1 train.steps=120 train.branch_from=$RUNS/long@96\n"
    )  # fmt: skip
    (root / "reports").mkdir()
    (root / "reports" / "r.json").write_text("{}")
    session_run.save(root, tmp_path / "out")
    saved = tmp_path / "out" / "tjev-work" / "runs"

    def kept(name):
        return sorted(int(p.name) for p in (saved / name / "checkpoints").iterdir())

    assert kept("done") == [20]  # the selected checkpoint only
    assert kept("running") == [5, 10]  # latest (resume) + selected
    assert kept("long") == [48, 96, 240]  # both pending branch points + selected
    assert (saved / "done" / "metrics.jsonl").exists()
    assert (tmp_path / "out" / "tjev-work" / "reports" / "r.json").exists()


def test_unpack_accepts_a_tarball_or_an_extracted_tree(tmp_path):
    tree = tmp_path / "tree"
    (tree / "data" / "mix-v3").mkdir(parents=True)
    (tree / "data" / "mix-v3" / "mix.yaml").write_text("x")
    with tarfile.open(tmp_path / "data.tgz", "w:gz") as tar:
        tar.add(tree / "data", arcname="data")
    (tmp_path / "ds").mkdir()
    (tmp_path / "data.tgz").rename(tmp_path / "ds" / "data.tgz")
    session_run.unpack(tmp_path / "ds", tmp_path / "a", "data/mix-v3/mix.yaml")
    assert (tmp_path / "a" / "data" / "mix-v3" / "mix.yaml").exists()
    extracted = tmp_path / "kaggle-input" / "some-dir"
    (extracted / "data" / "mix-v3").mkdir(parents=True)
    (extracted / "data" / "mix-v3" / "mix.yaml").write_text("y")
    session_run.unpack(tmp_path / "kaggle-input", tmp_path / "b", "data/mix-v3/mix.yaml")
    assert (tmp_path / "b" / "data" / "mix-v3" / "mix.yaml").read_text() == "y"
