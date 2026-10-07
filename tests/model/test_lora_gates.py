"""The GDN gate projections (in_proj_a / in_proj_b): their own adapter rank
(``lora.gate_rank``), direct training of their kernels (``lora.gate_full``), or frozen
(left out of ``lora.targets``)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from tjev.config import ComputeSpec, LoRASpec
from tjev.config.schema import GATES, LORA_TARGETS
from tjev.model import Qwen35
from tjev.model.weights import load_hf, stack_layers
from tjev.testing import make_tiny_snapshot
from tjev.train.step import split_model


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tiny-gates") / "model"
    make_tiny_snapshot(folder)
    return load_hf(folder, "float32")


def _model(snapshot, **lora):
    config, tensors = snapshot
    spec = LoRASpec(rank=8, alpha=8.0, **lora)
    model = Qwen35(stack_layers(dict(tensors), config), config, ComputeSpec(gdn_chunk=16), spec,
                   dtype=jnp.float32, rngs=nnx.Rngs(0))  # fmt: skip
    return model, spec


def _gdn(model):
    return model.blocks.gdn[0]


def _hidden(model):
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.integers(0, 512, (2, 32)), jnp.int32)
    seg = jnp.ones((2, 32), jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(32), (2, 32))
    return model.hidden(tokens, seg, pos)


def test_gate_rank_sets_the_gate_adapters_rank_and_scale(snapshot):
    model, spec = _model(snapshot, gate_rank=2)
    gdn = _gdn(model)
    for name in GATES:
        layer = getattr(gdn, name)
        assert layer.rank == 2 and layer.lora_a[...].shape[-1] == 2
        assert layer.scale == pytest.approx(8.0 / 2**0.5)  # rsLoRA at the gate rank
    assert gdn.in_proj_qkv.rank == 8 and gdn.in_proj_qkv.scale == pytest.approx(8.0 / 8**0.5)
    assert spec.rank_of("in_proj_a") == 2 and spec.rank_of("q_proj") == 8


def test_gates_left_out_of_targets_are_frozen(snapshot):
    targets = tuple(t for t in LORA_TARGETS if t not in GATES)
    model, _ = _model(snapshot, targets=targets)
    gdn = _gdn(model)
    for name in GATES:
        layer = getattr(gdn, name)
        assert layer.rank == 0 and not layer.full and not hasattr(layer, "lora_a")
    leaves = jax.tree_util.tree_leaves_with_path(split_model(model).lora)
    assert not any("in_proj_a" in jax.tree_util.keystr(p) for p, _ in leaves)


def test_gate_full_trains_the_gate_kernels_and_starts_at_the_base_model(snapshot):
    base, _ = _model(snapshot)
    model, _ = _model(snapshot, gate_full=True)
    gdn = _gdn(model)
    for name in GATES:
        layer = getattr(gdn, name)
        assert layer.full and layer.rank == 0 and not hasattr(layer, "lora_a")
        assert isinstance(layer.kernel, nnx.LoRAParam) and layer.kernel[...].dtype == jnp.float32
    # B = 0 elsewhere and the gate kernels are the base weights: the same output
    np.testing.assert_allclose(_hidden(model), _hidden(base), rtol=1e-6, atol=1e-6)
    # the gate kernels are trainable parameters, and get a gradient
    split = split_model(model)
    paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(split.lora)]
    assert any("in_proj_a" in p and "kernel" in p for p in paths)

    def loss(lora):
        m = nnx.merge(split.graphdef, lora, split.frozen)
        return jnp.sum(_hidden(m) ** 2)

    grads = jax.grad(loss)(split.lora)
    gate = [g for p, g in jax.tree_util.tree_leaves_with_path(grads)
            if "in_proj_a" in jax.tree_util.keystr(p) and "kernel" in jax.tree_util.keystr(p)]  # fmt: skip
    assert gate and float(jnp.abs(gate[0]).max()) > 0
