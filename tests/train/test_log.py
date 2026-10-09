"""MetricLogger's W&B: a failed online init (bad key or entity, W&B down) never kills a run."""

import sys
import types

import pytest

from tjev.train.log import MetricLogger


def _fake_wandb(monkeypatch, fail_online: bool):
    calls: list[dict] = []

    def init(**kwargs):
        calls.append(kwargs)
        if fail_online and kwargs["mode"] == "online":
            raise RuntimeError("entity nope not found during upsertBucket")
        return types.SimpleNamespace(mode=kwargs["mode"])

    module = types.SimpleNamespace(init=init, teardown=lambda: calls.append({"teardown": True}))
    monkeypatch.setitem(sys.modules, "wandb", module)
    return calls


def _logger(tmp_path, mode="online", resume_step=0, parent=None):
    return MetricLogger(
        tmp_path, tensorboard=False, wandb=True, project="p", config={"a": 1}, name="sweep-x",
        resume_step=resume_step, entity="nope", group="", tags=(), mode=mode, parent=parent,
    )  # fmt: skip


def test_a_resume_reopens_the_run_without_rewind(tmp_path, monkeypatch):
    """W&B's rewind (resume_from) is a private preview that answered HTTP 400 and left runs
    offline: a resume is a plain resume="allow" of the same id."""
    calls = _fake_wandb(monkeypatch, fail_online=False)
    (tmp_path / "wandb_id.txt").write_text("abcd1234")
    _logger(tmp_path, resume_step=300)
    assert len(calls) == 1 and calls[0]["id"] == "abcd1234" and calls[0]["resume"] == "allow"
    assert "resume_from" not in calls[0] and "parent" not in calls[0]["config"]


def test_a_cooldown_branch_is_a_new_run_naming_its_parent(tmp_path, monkeypatch):
    calls = _fake_wandb(monkeypatch, fail_online=False)
    parent = {"run": "final-4b-h5000", "step": 2000, "wandb_id": "p4r3nt00"}
    _logger(tmp_path, parent=parent)
    assert len(calls) == 1 and calls[0]["id"] != "p4r3nt00" and "resume_from" not in calls[0]
    assert calls[0]["config"] == {"a": 1, "parent": parent}
    assert "final-4b-h5000@2000" in calls[0]["notes"] and "p4r3nt00" in calls[0]["notes"]
    assert (tmp_path / "wandb_id.txt").read_text() == calls[0]["id"]


def test_failed_online_init_falls_back_to_offline(tmp_path, monkeypatch, capsys):
    calls = _fake_wandb(monkeypatch, fail_online=True)
    logger = _logger(tmp_path)
    assert logger.wandb.mode == "offline"
    assert [c.get("mode", "teardown") for c in calls] == ["online", "teardown", "offline"]
    assert calls[0]["id"] == calls[2]["id"]  # the same run: `wandb sync` uploads it later
    assert "wandb_offline_fallback" in capsys.readouterr().out


def test_offline_init_failure_is_not_hidden(tmp_path, monkeypatch):
    def init(**kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init))
    with pytest.raises(RuntimeError, match="disk full"):
        _logger(tmp_path, mode="offline")
