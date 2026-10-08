"""probe_calib's fetch (tjev.campaign.calib): RUN:STEP[:ema] entries, campaign-prefixed backup
paths, and the selection that makes ``tjev post`` score the adapters or their EMA."""

import json

import pytest

from tjev.campaign import calib


def test_entries_name_the_step_the_weights_and_the_backup_path():
    raw = calib.parse_entry("final-4b-h3000:1200")
    assert (raw.hf_run, raw.run, raw.step, raw.weights) == (
        "final-4b-h3000", "final-4b-h3000", 1200, "raw")  # fmt: skip
    assert raw.local == "final-4b-h3000@1200" and raw.upload == "final-4b-h3000/post-calib-1200"
    ema = calib.parse_entry("tjev-v75b/final-4b-h5000:2750:ema")
    assert (ema.run, ema.step, ema.weights) == ("final-4b-h5000", 2750, "ema")
    assert ema.local == "final-4b-h5000@2750-ema"
    assert ema.upload == "tjev-v75b/final-4b-h5000/post-calib-2750-ema"
    for bad in ("final-4b", "run:x", "run:10:swa", ":10"):
        with pytest.raises(ValueError):
            calib.parse_entry(bad)


def _backup(root, step=2750, opt=True, selection=None):
    run = root / "c1" / "final-4b-h5000"
    ckpt = run / "checkpoints" / str(step)
    (ckpt / "lora").mkdir(parents=True)
    if opt:
        (ckpt / "opt").mkdir()
    (run / "config.json").write_text("{}")
    (run / "selection.json").write_text(json.dumps(selection or {"step": 5000, "score": 0.4}))
    return run


def test_an_ema_entry_selects_the_ema_weights_of_its_step(tmp_path):
    src = _backup(tmp_path / "hf")
    entry = calib.parse_entry("c1/final-4b-h5000:2750:ema")
    dest = calib.prepare(src, tmp_path / "runs" / entry.local, entry)
    sel = json.loads((dest / "selection.json").read_text())
    assert sel == {"step": 2750, "score": 0.4, "weights": "ema"}
    assert json.loads((dest / "selection.orig.json").read_text())["step"] == 5000
    assert (dest / "queue.done").exists()


def test_a_raw_entry_scores_the_adapters_even_of_an_ema_selected_run(tmp_path):
    src = _backup(tmp_path / "hf", selection={"step": 2750, "score": 0.4, "weights": "ema"})
    entry = calib.parse_entry("c1/final-4b-h5000:2750")
    sel = json.loads((calib.prepare(src, tmp_path / "d", entry) / "selection.json").read_text())
    assert sel == {"step": 2750, "score": 0.4}


def test_an_ema_entry_needs_the_backed_up_optimizer_state(tmp_path):
    src = _backup(tmp_path / "hf", opt=False)
    with pytest.raises(FileNotFoundError, match="optimizer state"):
        calib.prepare(src, tmp_path / "d", calib.parse_entry("c1/final-4b-h5000:2750:ema"))
    with pytest.raises(FileNotFoundError, match="no checkpoint 100"):
        calib.prepare(src, tmp_path / "d", calib.parse_entry("c1/final-4b-h5000:100"))


def test_fetch_downloads_the_backup_path_and_prepares_the_run(tmp_path, monkeypatch):
    import huggingface_hub

    class Api:
        def model_info(self, repo):
            return type("Info", (), {"private": True})()

    def fake_snapshot(repo, allow_patterns, local_dir):
        assert "c1/final-4b-h5000/checkpoints/2750/**" in allow_patterns
        _backup(tmp_path / "runs" / "hf-calib")
        return local_dir

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    entry = calib.parse_entry("c1/final-4b-h5000:2750:ema")
    dest = calib.fetch("me/backup", entry, tmp_path / "runs")
    assert dest == tmp_path / "runs" / "final-4b-h5000@2750-ema"
    assert json.loads((dest / "selection.json").read_text())["weights"] == "ema"
