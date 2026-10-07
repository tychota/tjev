"""tjev campaign bench: the probe's debug dumps (compile log, XLA HLO, Mosaic flag)."""

import os

import jax
import jax.numpy as jnp

from tjev.campaign import bench


def test_debug_compiles_logs_and_dumps_one_compile(tmp_path):
    with bench.debug_compiles(tmp_path, "0.8B-split") as options:
        jax.jit(lambda x: jnp.sin(x) @ x).lower(jnp.ones((4, 4))).compile(options)
    dumped = list((tmp_path / "xla" / "0.8B-split").glob("*after_optimizations*.txt"))
    assert dumped
    assert "Compiling" in (tmp_path / "compiles-0.8B-split.log").read_text()
    assert not jax.config.jax_log_compiles  # restored after the block


def test_debug_compiles_is_off_without_a_dir():
    with bench.debug_compiles(None, "x") as options:
        assert options == {}


def test_mosaic_dump_keeps_the_environment_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--xla_tpu_scoped_vmem_limit_kib=98304")
    bench.enable_mosaic_dump(tmp_path)
    bench.enable_mosaic_dump(tmp_path)  # idempotent
    flags = os.environ["LIBTPU_INIT_ARGS"].split()
    assert flags == [
        "--xla_tpu_scoped_vmem_limit_kib=98304",
        f"--xla_mosaic_dump_to={tmp_path}/mosaic",
    ]


def test_variants_are_known_everywhere():
    from tjev.campaign import parity

    assert set(parity.VARIANTS) >= set(bench.VARIANTS)


def test_every_variant_builds_a_valid_config():
    """Each variant's overrides parse against the schema (a typo or bad value fails here,
    not in a paid TPU session); the cycle-3 options reach the ComputeSpec."""
    from tjev.campaign import parity
    from tjev.config import RunConfig
    from tjev.config.loader import apply_override, from_dict

    for overrides in parity.VARIANTS.values():
        raw = RunConfig().to_dict()
        for item in overrides:
            if item != parity.HIGHEST:
                apply_override(raw, item)
        from_dict(raw)  # raises on unknown keys and wrong types
    raw = RunConfig().to_dict()
    for item in bench.VARIANTS["mega2-hb16"] + [bench.MERGED, bench.BLOCKED]:
        apply_override(raw, item)
    compute = from_dict(raw).compute
    assert (compute.gdn_impl, compute.gdn_heads_per_step, compute.gdn_vmem_fraction) == (
        "pallas_tpu_mega2", 16, 0.85,
    )  # fmt: skip
    assert (compute.lora_impl, compute.gdn_inverse, compute.remat) == (
        "merged", "blocked", "core_states",
    )  # fmt: skip
