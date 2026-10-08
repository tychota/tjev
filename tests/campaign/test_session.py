"""Time-limited Colab sessions: what cloud/session_run.py saves and unpacks."""

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
    extracted = tmp_path / "extracted-input" / "some-dir"
    (extracted / "data" / "mix-v3").mkdir(parents=True)
    (extracted / "data" / "mix-v3" / "mix.yaml").write_text("y")
    session_run.unpack(tmp_path / "extracted-input", tmp_path / "b", "data/mix-v3/mix.yaml")
    assert (tmp_path / "b" / "data" / "mix-v3" / "mix.yaml").read_text() == "y"


def test_hf_data_unpacks_from_the_snapshot_dir(tmp_path, monkeypatch):
    import huggingface_hub

    def fake_snapshot(repo, repo_type, local_dir, token):  # the HF repo holds data/ at its root
        assert (repo, repo_type, token) == ("me/tjev-data", "dataset", "t0k")
        (Path(local_dir) / "data" / "mix-v3").mkdir(parents=True)
        (Path(local_dir) / "data" / "mix-v3" / "mix.yaml").write_text("x")
        (Path(local_dir) / ".cache" / "huggingface").mkdir(parents=True)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    snap = session_run.fetch_hf("me/tjev-data", tmp_path / "hf-data", "t0k")
    session_run.unpack(snap, tmp_path / "root", "data/mix-v3/mix.yaml")
    assert (tmp_path / "root" / "data" / "mix-v3" / "mix.yaml").read_text() == "x"


def test_sweep_backup_uploads_every_kept_checkpoint_of_finished_sweep_runs(tmp_path, monkeypatch):
    import huggingface_hub

    runs = tmp_path / "runs"
    _run(runs, "sweep-reg-4b-rank16-s0", [1300, 1450, 1500], done=True, selected=1300)
    _run(runs, "sweep-reg-4b-rank16-lr0.7-s0", [600], done=False, selected=600)  # paused
    _run(runs, "final-4b-h5000", [2500, 2750, 5000], done=True, selected=2750)
    _run(runs, "transfer-2b-hi-h2400", [2400], done=True, selected=2400)  # not backed up
    calls = []

    class Api:
        def __init__(self, token=None):
            pass

        def model_info(self, repo):
            return type("Info", (), {"private": True})()

        def upload_folder(self, **kw):
            calls.append(kw)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    session_run.backup_sweep_runs(runs, "me/backup", None, "tjev-v75b")
    # the final runs too, each under its campaign (same-named runs of two campaigns clashed)
    assert [c["path_in_repo"] for c in calls] == ["tjev-v75b/final-4b-h5000",
                                                  "tjev-v75b/sweep-reg-4b-rank16-s0"]  # fmt: skip
    final, sweep = calls
    # every checkpoint retention left; the optimizer state of the last and the selected one
    # (the EMA lives in it): never ignored
    assert sweep["ignore_patterns"][3:] == ["checkpoints/1450/opt/**"]
    assert final["ignore_patterns"][3:] == ["checkpoints/2500/opt/**"]
    assert "1300, 1450, 1500" in sweep["commit_message"]
    # no campaign: the bare run name (the older layout)
    calls.clear()
    session_run.backup_sweep_runs(runs, "me/backup", None)
    assert [c["path_in_repo"] for c in calls] == ["final-4b-h5000", "sweep-reg-4b-rank16-s0"]
    # per kind (the final models and the sweep go to different repos); a missing dir is fine
    calls.clear()
    session_run.backup_sweep_runs(runs, "me/final", None, "c1", kinds=("final-",))
    assert [c["path_in_repo"] for c in calls] == ["c1/final-4b-h5000"]
    smoke = tmp_path / "reports/probe/smoke"
    _run(smoke, "smoke-4b-h5000", [30], done=True, selected=30)
    calls.clear()
    session_run.backup_sweep_runs(smoke, "me/final", None, "c1", kinds=("smoke-",))
    session_run.backup_sweep_runs(tmp_path / "nowhere", "me/final", None, "c1")
    assert [c["path_in_repo"] for c in calls] == ["c1/smoke-4b-h5000"]


def test_backup_repo_follows_the_run_kind():
    """final-* and smoke-* runs live in the final models' repo, the sweep's in the sweep's;
    PROBE_HF_FINAL_REPO / PROBE_HF_REPO override the defaults; a backup path works too."""
    assert session_run.backup_repo("sweep-reg-4b-centre-s0", {}) == "tychot/tjev-4b"
    assert session_run.backup_repo("tjev-v75b/sweep-reg-4b-centre-s0", {}) == "tychot/tjev-4b"
    assert session_run.backup_repo("final-4b-h5000", {}) == "tychot/tjev-final"
    assert session_run.backup_repo("c1/final-4b-h5000-b2500", {}) == "tychot/tjev-final"
    assert session_run.backup_repo("smoke-4b-h5000", {"PROBE_HF_FINAL_REPO": "me/f"}) == "me/f"
    both = {"PROBE_HF_REPO": "me/s", "PROBE_HF_FINAL_REPO": "me/f"}
    assert session_run.backup_repo("sweep-x", both) == "me/s"


def test_fetch_checkpoints_reads_a_campaign_prefixed_backup(tmp_path, monkeypatch):
    import huggingface_hub

    def fake_snapshot(repo, local_dir, token, allow_patterns):
        assert allow_patterns[-1] == "c1/final-4b-h5000/checkpoints/2000/**"
        ckpt = Path(local_dir) / "c1" / "final-4b-h5000" / "checkpoints" / "2000" / "opt"
        ckpt.mkdir(parents=True)
        (Path(local_dir) / "c1" / "final-4b-h5000" / "config.json").write_text("{}")
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    runs = tmp_path / "tjev-work" / "runs"
    runs.mkdir(parents=True)
    session_run.fetch_checkpoints(["c1/final-4b-h5000@2000"], "me/backup", runs, None)
    assert (runs / "final-4b-h5000" / "checkpoints" / "2000" / "opt").is_dir()
    assert (runs / "final-4b-h5000" / "config.json").exists()
    assert not (tmp_path / "tjev-work" / "hf-fetch").exists()


def test_sweep_backup_refuses_a_public_repo_and_never_raises(tmp_path, monkeypatch):
    import huggingface_hub

    runs = tmp_path / "runs"
    _run(runs, "sweep-reg-4b-rank16-s0", [1500], done=True, selected=1500)
    uploads = []

    class Api:
        def __init__(self, token=None):
            pass

        def model_info(self, repo):
            return type("Info", (), {"private": False})()

        def upload_folder(self, **kw):
            uploads.append(kw)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    session_run.backup_sweep_runs(runs, "me/public", None)
    assert uploads == []
