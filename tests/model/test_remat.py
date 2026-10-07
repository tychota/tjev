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
         "gdn_core_out", "gdn_states", "gdn_vnew", "gdn_tinv"}  # fmt: skip


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("remat")
    make_tiny_snapshot(root / "model")
    return root / "model"


def _model(snapshot, remat, gdn_impl="chunked", by_bucket=None):
    compute = ComputeSpec(remat=remat, gdn_chunk=16, gdn_impl=gdn_impl,
                          remat_by_bucket=by_bucket or {})  # fmt: skip
    _, model = build_model(snapshot, compute, LoRASpec(rank=4), dtype="float32", seed=0)
    for path, var in nnx.iter_graph(model):  # non-zero B: every adapter gets a gradient
        if isinstance(var, nnx.LoRAParam) and path[-1] == "lora_b":
            var[...] = 0.01 * jnp.ones_like(var[...])
    return model


TOKENS = jax.random.randint(jax.random.key(0), (1, 48), 3, 200)
SEG, POS = jnp.ones_like(TOKENS), jnp.arange(48)[None]


@pytest.mark.parametrize(
    ("policy", "gdn_impl"),
    [("core", "chunked"), ("minimal", "chunked"), ("core_states", "chunked"),
     ("core_states", "pallas_tpu_split"), ("core_states", "pallas_tpu"),
     ("core_states", "pallas_tpu_v2"), ("full", "pallas_tpu_v2"),
     ("core_states", "pallas_tpu_mega2"), ("full", "pallas_tpu_mega2"),
     ("core_states", "pallas_tpu_mega1"), ("minimal_states", "chunked"),
     ("minimal_states", "pallas_tpu_v2"), ("minimal_states", "pallas_tpu_mega2")],
)  # fmt: skip
def test_policy_gradients_equal_full_remat(snapshot, policy, gdn_impl):
    """(full, pallas_tpu_v2): against remat=none, the v2 kernels under the full policy."""

    def grads(remat):
        model = _model(snapshot, remat, gdn_impl)
        graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)

        def loss(lora):
            h = nnx.merge(graphdef, lora, rest).hidden(TOKENS, SEG, POS)
            return jnp.sum(h.astype(jnp.float32) ** 2)

        return jax.grad(loss)(lora)

    want, got = grads("none" if policy == "full" else "full"), grads(policy)

    def close(a, b):  # float32 reassociation only: ~1e-5 of the largest entry
        assert float(jnp.max(jnp.abs(a - b))) <= 1e-3 * float(jnp.max(jnp.abs(b))) + 1e-8

    jax.tree.map(close, got, want)


def _saved_names(snapshot, remat, gdn_impl="chunked", by_bucket=None, tokens=TOKENS):
    """Named residuals of one super-block (outside the layer scan, where names show)."""
    model = _model(snapshot, remat, gdn_impl, by_bucket)
    graphdef, state = nnx.split(model.blocks)
    block = nnx.merge(graphdef, jax.tree.map(lambda x: x[0], state))
    x = model.embed[...][tokens]
    seg, pos = jnp.ones_like(tokens), jnp.arange(tokens.shape[1])[None]

    def loss(x):
        return jnp.sum(block(x, seg, pos) ** 2)

    return {n for _, where in saved_residuals(loss, x) for n in NAMES if f"'{n}'" in where}


MINIMAL_STATES = {"gdn_core_out", "mlp_gate", "mlp_up", "mixer_out", "gdn_z", "attn_qkv",
                  "gdn_qkv", "gdn_ba"}  # fmt: skip


def test_minimal_states_saves_minimal_and_the_kernel_residuals(snapshot):
    # the GDN core is no checkpoint of its own: its qkv / b / a inputs are saved by name
    assert _saved_names(snapshot, "minimal_states") == MINIMAL_STATES  # XLA core: no kernel
    for impl in ("pallas_tpu_v2", "pallas_tpu_mega2"):
        assert _saved_names(snapshot, "minimal_states", impl) == {
            *MINIMAL_STATES, "gdn_states", "gdn_tinv",
        }, impl  # fmt: skip


def test_remat_by_bucket_picks_the_policy_of_the_row_length(snapshot):
    by_bucket = {"48": "minimal_states"}
    short = TOKENS[:, :32]
    assert _saved_names(snapshot, "core_states", "pallas_tpu_v2", by_bucket) == {
        *MINIMAL_STATES, "gdn_states", "gdn_tinv",
    }  # fmt: skip
    assert _saved_names(snapshot, "core_states", "pallas_tpu_v2", by_bucket, short) == {
        "gdn_core_out", "gdn_states", "gdn_tinv",
    }  # fmt: skip
    # EVA's probe passes set blocks.remat = "none": it wins over the bucket's policy
    blocks = _model(snapshot, "core_states", by_bucket=by_bucket).blocks
    assert blocks.remat_at(48) == "minimal_states" and blocks.remat_at(32) == "core_states"
    blocks.remat = "none"
    assert blocks.remat_at(48) == "none"


def test_remat_by_bucket_gradients_equal_its_policy(snapshot):
    """A per-bucket policy is the same function: gradients equal core_states' at T=48."""

    def grads(remat, by_bucket=None):
        model = _model(snapshot, remat, "pallas_tpu_v2", by_bucket)
        graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)

        def loss(lora):
            h = nnx.merge(graphdef, lora, rest).hidden(TOKENS, SEG, POS)
            return jnp.sum(h.astype(jnp.float32) ** 2)

        return jax.grad(loss)(lora)

    want = grads("core_states")
    got = grads("core_states", {"48": "minimal_states"})

    def close(a, b):
        assert float(jnp.max(jnp.abs(a - b))) <= 1e-3 * float(jnp.max(jnp.abs(b))) + 1e-8

    jax.tree.map(close, got, want)


def test_policies_save_exactly_their_names(snapshot):
    assert _saved_names(snapshot, "full") == set()
    assert _saved_names(snapshot, "core") == {"gdn_core_out"}  # attn_context: splash only
    # minimal: the GDN qkv/b/a outputs are kept too, as the inputs of the GDN core's own
    # checkpoint (listed as "output of remat2", not by name); with the projections saved,
    # the small x·A products are recomputed.
    assert _saved_names(snapshot, "minimal") == {
        "gdn_core_out", "mlp_gate", "mlp_up", "mixer_out", "gdn_z", "attn_qkv",
    }  # fmt: skip


def test_core_states_saves_the_gdn_kernel_residuals(snapshot):
    assert _saved_names(snapshot, "core_states") == {"gdn_core_out"}  # XLA core: no kernel
    for impl in ("pallas_tpu", "pallas_tpu_split"):
        assert _saved_names(snapshot, "core_states", impl) == {
            "gdn_core_out", "gdn_states", "gdn_vnew",
        }, impl  # fmt: skip
    for impl in ("pallas_tpu_v2", "pallas_tpu_mega1", "pallas_tpu_mega2"):
        assert _saved_names(snapshot, "core_states", impl) == {
            "gdn_core_out", "gdn_states", "gdn_tinv",
        }, impl  # fmt: skip


@pytest.mark.parametrize(
    ("remat", "forward_kernels", "impl"),
    [("full", 9, "pallas_tpu"), ("core", 6, "pallas_tpu"), ("core_states", 3, "pallas_tpu"),
     ("full", 9, "pallas_tpu_v2"), ("core_states", 3, "pallas_tpu_v2"),
     ("full", 9, "pallas_tpu_mega2"), ("core_states", 3, "pallas_tpu_mega2"),
     ("core_states", 3, "pallas_tpu_mega1"), ("minimal_states", 3, "pallas_tpu_mega2"),
     ("minimal_states", 3, "pallas_tpu_v2")],
)  # fmt: skip
def test_gdn_forward_kernel_runs_once_under_core_states(tmp_path, monkeypatch, remat,
                                                         forward_kernels, impl):  # fmt: skip
    """Pallas GDN custom calls in the TPU-lowered gradient of one super-block (3 GDN layers):
    3 backward kernels, and the forward kernel 3× per layer under full remat (primal, layer
    recompute, the core's own recompute), 2× under core, once under core_states."""
    from jax import export

    from tjev.kernels.gated_delta_rule import pallas_tpu

    monkeypatch.setattr(pallas_tpu, "_interpret", lambda: False)  # the real Mosaic path
    make_tiny_snapshot(tmp_path / "m", hidden_size=128, intermediate_size=256, num_layers=4,
                       linear_key_head_dim=128, linear_value_head_dim=128)  # fmt: skip
    compute = ComputeSpec(remat=remat, gdn_impl=impl)
    _, model = build_model(tmp_path / "m", compute, LoRASpec(rank=4), dtype="bfloat16")
    graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)
    tokens = jnp.zeros((1, 128), jnp.int32)
    seg, pos = jnp.ones_like(tokens), jnp.arange(128)[None]

    def loss(lora, rest):
        h = nnx.merge(graphdef, lora, rest).hidden(tokens, seg, pos)
        return jnp.sum(h.astype(jnp.float32) ** 2)

    mlir = export.export(jax.jit(jax.grad(loss)), platforms=["tpu"])(lora, rest).mlir_module()
    assert mlir.count("tpu_custom_call") == forward_kernels + 3
