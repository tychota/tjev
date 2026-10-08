"""libtpu flags from the TPU presets: applied before JAX starts, the environment wins, and
they are not part of a run's identity."""

import dataclasses
import os

from tjev.config import load_config
from tjev.sharding import configure_runtime, libtpu_settings
from tjev.train.loop import run_identity


def test_libtpu_flags_come_from_the_preset_unless_the_environment_sets_them(monkeypatch):
    cfg = load_config("tpu-v5e")  # through extends: the v6e set
    assert "--xla_tpu_enable_latency_hiding_scheduler=true" in cfg.compute.libtpu_flags
    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    assert configure_runtime(cfg.compute) == "preset"
    assert os.environ["LIBTPU_INIT_ARGS"] == " ".join(cfg.compute.libtpu_flags)
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--probed")
    assert configure_runtime(cfg.compute) == "env"
    assert os.environ["LIBTPU_INIT_ARGS"] == "--probed"
    monkeypatch.delenv("LIBTPU_INIT_ARGS")
    assert configure_runtime(load_config().compute) == "none"
    assert "LIBTPU_INIT_ARGS" not in os.environ


def test_libtpu_settings_report_the_scoped_vmem_limit(monkeypatch):
    """What a run logs at its start: the flags and the scoped VMEM limit they imply (XLA's,
    and the GDN kernels' 90% of it), None for libtpu's default."""
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--a=1 --xla_tpu_scoped_vmem_limit_kib=98304")
    got = libtpu_settings("env")
    assert got == {"libtpu_init_args": "--a=1 --xla_tpu_scoped_vmem_limit_kib=98304",
                   "libtpu_source": "env", "scoped_vmem_limit_kib": 98304,
                   "kernel_vmem_limit_bytes": int(0.9 * 98304 * 1024)}  # fmt: skip
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--a=1")
    assert libtpu_settings()["scoped_vmem_limit_kib"] is None
    assert libtpu_settings()["kernel_vmem_limit_bytes"] is None


def test_libtpu_flags_do_not_change_the_identity():
    cfg = load_config("tpu-v6e")
    bare = dataclasses.replace(cfg, compute=dataclasses.replace(cfg.compute, libtpu_flags=()))
    assert run_identity(cfg, "m") == run_identity(bare, "m")


def test_new_compute_fields_at_their_default_keep_the_identity(monkeypatch):
    """Fields added after runs existed (IDENTITY_NEW_DEFAULTS): absent from the identity at
    their default, so older runs resume; any other value is part of it."""
    from tjev.train import loop

    cfg = load_config("tpu-v6e")
    seen = []
    monkeypatch.setattr(loop, "fingerprint", lambda d: seen.append(d) or "x")
    run_identity(cfg, "m")
    assert not set(loop.IDENTITY_NEW_DEFAULTS["compute"]) & set(seen[0]["compute"])
    monkeypatch.undo()
    for key, default in loop.IDENTITY_NEW_DEFAULTS["compute"].items():
        assert loop._same(getattr(cfg.compute, key), default), key  # a tuple field: a list
    changed = dataclasses.replace(
        cfg, compute=dataclasses.replace(cfg.compute, gdn_inverse="solve")
    )
    assert run_identity(changed, "m") != run_identity(cfg, "m")
