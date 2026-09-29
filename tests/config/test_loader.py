"""Layered config: presets by name, overrides, fail-closed coercion."""

import pytest

from tjev.config import ComputeSpec, LoRASpec, RunConfig, load_config, preset_names
from tjev.config.loader import build, from_dict


def test_defaults_are_the_recommended_recipe():
    cfg = load_config()
    assert cfg == RunConfig()
    assert cfg.optim.decay_shape == "sqrt"
    assert cfg.train.packing_bins == 16


def test_presets_layer_and_overrides_win():
    cfg = load_config(
        "tpu-v5e", "qwen35-4b", overrides=["optim.lr=1e-5", "train.seq_buckets=[1024]"]
    )
    assert cfg.compute.attention == "splash"  # from tpu-v6e, through extends
    assert cfg.train.microbatch_tokens == 4096  # tpu-v5e overrides its parent
    assert cfg.lora.rank == 64
    assert cfg.optim.lr == 1e-5
    assert cfg.train.seq_buckets == (1024,)


def test_every_preset_loads():
    names = preset_names()
    assert {"tpu-v6e", "tpu-v5e", "qwen35-2b"} <= set(names)
    for name in names:
        load_config(name)


def test_saved_configs_round_trip():
    cfg = load_config("tpu-v6e", "qwen35-2b", overrides=["data.mixture={a: 0.5, b: 0.5}"])
    assert from_dict(cfg.to_dict()) == cfg
    assert from_dict(cfg.to_dict()).fingerprint() == cfg.fingerprint()


def test_mixture_is_replaced_not_merged(tmp_path):
    mix = tmp_path / "mix.yaml"
    mix.write_text("data:\n  mixture: {a: 1.0}\n")
    cfg = load_config(str(mix), overrides=["data.mixture={b: 1.0}"])
    assert cfg.data.mixture == {"b": 1.0}


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (["compute.gdn_typo=1"], KeyError),
        (["compute.attention=mosaic"], ValueError),
        (["train.steps=1.5"], ValueError),
        (["log.wandb=maybe"], ValueError),
        (["lora=3"], TypeError),
        (["noequals"], ValueError),
    ],
)
def test_invalid_configs_fail(overrides, error):
    with pytest.raises(error):
        load_config(overrides=overrides)


def test_spec_invariants():
    with pytest.raises(ValueError, match="power-of-two"):
        build(ComputeSpec, {"gdn_impl": "pallas_tpu", "gdn_chunk": 48})
    assert LoRASpec(rank=64, alpha=32).scale == 4.0
    assert LoRASpec(rank=64, alpha=32, rslora=False).scale == 0.5
