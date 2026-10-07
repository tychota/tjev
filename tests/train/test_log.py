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


def _logger(tmp_path, mode="online"):
    return MetricLogger(
        tmp_path, tensorboard=False, wandb=True, project="p", config={}, name="sweep-x",
        resume_step=0, entity="nope", group="", tags=(), mode=mode,
    )  # fmt: skip


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
