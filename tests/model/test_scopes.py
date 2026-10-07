"""Named scopes (xprof op_profile / framework op stats group by them): every part of the
model names its ops, and the names survive remat (the recompute) and autodiff (the
backward), whose ops are most of a training step."""

import re

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from tjev.config import ComputeSpec, LoRASpec
from tjev.model import build_model
from tjev.testing import make_tiny_snapshot

SCOPES = {
    "embed", "layer0_gdn", "layer2_gdn", "layer3_attn", "norm", "gdn_proj", "gdn_conv", "gdn_gates",
    "gdn_qk_norm", "gdn_rule", "gdn_core", "gdn_out", "attn_proj", "attention_core",
    "attn_out", "mlp", "base", "lora", "final_norm", "readout",
}  # fmt: skip


@pytest.fixture(scope="module")
def op_names(tmp_path_factory):
    root = tmp_path_factory.mktemp("scopes")
    make_tiny_snapshot(root / "model")
    compute = ComputeSpec(remat="full", gdn_chunk=16)
    _, model = build_model(root / "model", compute, LoRASpec(rank=4), dtype="float32")
    graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)
    tokens = jnp.zeros((1, 32), jnp.int32)
    seg, pos = jnp.ones_like(tokens), jnp.arange(32)[None]
    slots, labels = jnp.array([[31]]), jnp.array([[[5, 6]]])

    def loss(lora):
        m = nnx.merge(graphdef, lora, rest)
        return jnp.sum(m.label_logits(m.hidden(tokens, seg, pos), slots, labels))

    lowered = jax.jit(jax.grad(loss)).lower(lora)
    # name stacks: scopes, possibly wrapped by autodiff ("transpose(jvp(final_norm))"), as
    # traced (locations) and as compiled (op_name metadata: what xprof reads)
    return {
        "traced": re.findall(r'loc\("([^"]+)"', lowered.as_text(debug_info=True)),
        "compiled": re.findall(r'op_name="([^"]+)"', str(lowered.compile().as_text())),
    }


def _scopes(name: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", name))


def test_every_part_of_the_model_is_named(op_names):
    seen = set().union(*map(_scopes, op_names["traced"]))
    assert seen >= SCOPES, SCOPES - seen
    # after XLA's fusions too, except the tiny readout (fused into its neighbours)
    seen = set().union(*map(_scopes, op_names["compiled"]))
    assert seen >= SCOPES - {"readout"}, SCOPES - seen


@pytest.mark.parametrize("scope", ["gdn_core", "gdn_rule", "base", "lora", "mlp", "norm"])
def test_names_survive_remat_and_the_backward(op_names, scope):
    named = [n for n in op_names["compiled"] if scope in _scopes(n)]
    assert [n for n in named if "rematted_computation" in n]  # the remat recompute
    assert [n for n in named if "transpose(" in n]  # the backward


def test_projections_are_named_down_to_the_adapter(op_names):
    """Each projection names its ops (q_proj, in_proj_qkv, up_proj, …): base, the adapter's
    A (shared by a group: gate_proj+up_proj/lora/A) and B, forward and backward."""
    names = op_names["traced"]
    for path in ("mlp/up_proj/base", "mlp/down_proj/base", "mlp/gate_proj+up_proj/lora/A",
                 "mlp/up_proj/lora/B", "attn_proj/q_proj/base", "attn_out/o_proj/lora/B",
                 "gdn_proj/in_proj_qkv/base", "gdn_out/out_proj/lora/A",
                 "layer1_gdn/residual"):  # fmt: skip
        assert any(path in n for n in names), path
    assert any("transpose(" in n and "up_proj/lora/B" in n for n in op_names["compiled"])


def test_merged_lora_is_named_per_projection(tmp_path):
    make_tiny_snapshot(tmp_path / "model")
    compute = ComputeSpec(remat="full", gdn_chunk=16, lora_impl="merged")
    _, model = build_model(tmp_path / "model", compute, LoRASpec(rank=4), dtype="float32")
    graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)
    tokens = jnp.zeros((1, 32), jnp.int32)
    seg, pos = jnp.ones_like(tokens), jnp.arange(32)[None]

    def loss(lora):
        m = nnx.merge(graphdef, lora, rest)
        return jnp.sum(m.label_logits(m.hidden(tokens, seg, pos), jnp.array([[31]]),
                                      jnp.array([[[5, 6]]])))  # fmt: skip

    text = jax.jit(jax.grad(loss)).lower(lora).as_text(debug_info=True)
    names = re.findall(r'loc\("([^"]+)"', text)
    for path in ("mlp/up_proj/base", "mlp/up_proj/lora/merge", "mlp/gate_proj+up_proj/lora"):
        assert any(path in n for n in names), path


def test_unrolled_layers_are_numbered_and_equal(tmp_path, monkeypatch):
    """TJEV_UNROLL_LAYERS=1 (a profiling capture): one name per layer (L00 …), the same hidden
    states as the scan to float32 rounding."""
    make_tiny_snapshot(tmp_path / "model")
    compute = ComputeSpec(remat="full", gdn_chunk=16)
    config, model = build_model(tmp_path / "model", compute, LoRASpec(rank=4), dtype="float32")
    tokens = jnp.arange(32, dtype=jnp.int32)[None] % 50
    seg, pos = jnp.ones_like(tokens), jnp.arange(32)[None]
    scanned = model.hidden(tokens, seg, pos)
    monkeypatch.setenv("TJEV_UNROLL_LAYERS", "1")
    unrolled = model.hidden(tokens, seg, pos)
    assert jnp.allclose(scanned, unrolled, atol=1e-5)
    graphdef, state = nnx.split(model)
    text = (
        jax.jit(lambda st: nnx.merge(graphdef, st).hidden(tokens, seg, pos))
        .lower(state)
        .as_text(debug_info=True)
    )
    last = config.num_super_blocks * config.full_attention_interval - 1
    assert "L00/layer0_gdn" in text and f"L{last:02d}/layer3_attn" in text
