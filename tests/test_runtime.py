"""libtpu flags from the TPU presets: applied before JAX starts, the environment wins, and
they are not part of a run's identity."""

import dataclasses
import os

from tjev.config import load_config
from tjev.sharding import configure_runtime
from tjev.train.loop import run_identity


def test_libtpu_flags_come_from_the_preset_unless_the_environment_sets_them(monkeypatch):
    cfg = load_config("tpu-v5e")  # through extends: the v6e set
    assert "--xla_tpu_enable_latency_hiding_scheduler=true" in cfg.compute.libtpu_flags
    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    configure_runtime(cfg.compute)
    assert os.environ["LIBTPU_INIT_ARGS"] == " ".join(cfg.compute.libtpu_flags)
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--probed")
    configure_runtime(cfg.compute)
    assert os.environ["LIBTPU_INIT_ARGS"] == "--probed"


def test_libtpu_flags_do_not_change_the_identity():
    cfg = load_config("tpu-v6e")
    bare = dataclasses.replace(cfg, compute=dataclasses.replace(cfg.compute, libtpu_flags=()))
    assert run_identity(cfg, "m") == run_identity(bare, "m")
