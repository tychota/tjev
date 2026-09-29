"""Selective remat policies (qwen35.REMAT_POLICIES): same gradients as full remat, and each
policy keeps exactly its named intermediates."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax._src.ad_checkpoint import saved_residuals

from tjev.config import ComputeSpec, LoRASpec
from tjev.model import build_model
from tjev.testing import make_tiny_snapshot

NAMES = {"mlp_gate", "mlp_up", "mixer_out", "lora_xa", "gdn_qkv", "gdn_z", "gdn_ba", "attn_qkv",
         "gdn_core_out"}  # fmt: skip


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("remat")
    make_tiny_snapshot(root / "model")
    return root / "model"


def _model(snapshot, remat):
    compute = ComputeSpec(remat=remat, gdn_chunk=16)
    _, model = build_model(snapshot, compute, LoRASpec(rank=4), dtype="float32", seed=0)
    for path, var in nnx.iter_graph(model):  # non-zero B: every adapter gets a gradient
        if isinstance(var, nnx.LoRAParam) and path[-1] == "lora_b":
            var[...] = 0.01 * jnp.ones_like(var[...])
    return model


TOKENS = jax.random.randint(jax.random.key(0), (1, 48), 3, 200)
SEG, POS = jnp.ones_like(TOKENS), jnp.arange(48)[None]


@pytest.mark.parametrize("policy", ["core", "minimal"])
def test_policy_gradients_equal_full_remat(snapshot, policy):
    def grads(remat):
        graphdef, lora, rest = nnx.split(_model(snapshot, remat), nnx.LoRAParam, ...)

        def loss(lora):
            h = nnx.merge(graphdef, lora, rest).hidden(TOKENS, SEG, POS)
            return jnp.sum(h.astype(jnp.float32) ** 2)

        return jax.grad(loss)(lora)

    want, got = grads("full"), grads(policy)

    def close(a, b):  # float32 reassociation only: ~1e-5 of the largest entry
        assert float(jnp.max(jnp.abs(a - b))) <= 1e-3 * float(jnp.max(jnp.abs(b))) + 1e-8

    jax.tree.map(close, got, want)


def _saved_names(snapshot, remat):
    """Named residuals of one super-block (outside the layer scan, where names show)."""
    model = _model(snapshot, remat)
    graphdef, state = nnx.split(model.blocks)
    block = nnx.merge(graphdef, jax.tree.map(lambda x: x[0], state))
    x = model.embed[...][TOKENS]

    def loss(x):
        return jnp.sum(block(x, SEG, POS) ** 2)

    return {n for _, where in saved_residuals(loss, x) for n in NAMES if f"'{n}'" in where}


def test_policies_save_exactly_their_names(snapshot):
    assert _saved_names(snapshot, "full") == set()
    assert _saved_names(snapshot, "core") == {"gdn_core_out"}  # attn_context: splash only
    # minimal: the GDN qkv/b/a outputs are kept too, as the inputs of the GDN core's own
    # checkpoint (listed as "output of remat2", not by name); with the projections saved,
    # the small x·A products are recomputed.
    assert _saved_names(snapshot, "minimal") == {
        "gdn_core_out", "mlp_gate", "mlp_up", "mixer_out", "gdn_z", "attn_qkv",
    }  # fmt: skip
