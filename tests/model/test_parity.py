"""Numerical parity with the Hugging Face Qwen3.5 reference (tiny random model, fp32)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tjev.config import ComputeSpec, LoRASpec
from tjev.model import ModelConfig, Qwen35, load_hf, stack_layers


@pytest.fixture(scope="module")
def tiny_hf(tmp_path_factory):
    config = ModelConfig.tiny()
    raw = config.to_hf()
    raw.pop("architectures")
    raw.pop("model_type")
    hf_config = Qwen3_5TextConfig(**raw)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(hf_config).eval()
    with torch.no_grad():  # non-trivial norms, decays and conv (HF init zeros some)
        for name, p in model.named_parameters():
            if "norm" in name or "A_log" in name or "dt_bias" in name:
                p.copy_(torch.randn_like(p) * 0.3 + (1.0 if "linear_attn.norm" in name else 0.0))
    folder = tmp_path_factory.mktemp("tiny")
    model.save_pretrained(folder, safe_serialization=True)
    return folder, model


def _ours(folder, **compute):
    config, tensors = load_hf(folder, "float32")
    return Qwen35(stack_layers(tensors, config), config, ComputeSpec(**compute), dtype=jnp.float32)


def _ids(batch=2, length=70, vocab=512, seed=0):
    return np.random.default_rng(seed).integers(0, vocab, (batch, length))


@pytest.mark.parametrize("compute", [{}, {"gdn_impl": "recurrent"}, {"remat": "none"}])
def test_logits_match_hf(tiny_hf, compute):
    folder, hf = tiny_hf
    ids = _ids()
    with torch.no_grad():
        want = hf(torch.tensor(ids)).logits.numpy()
    model = _ours(folder, gdn_chunk=16, **compute)
    seg = jnp.ones(ids.shape, jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape)
    got = model.logits(model.hidden(jnp.asarray(ids), seg, pos))
    np.testing.assert_allclose(got, want, atol=2e-4, rtol=2e-4)


def test_packed_rows_equal_separate_sequences(tiny_hf):
    folder, hf = tiny_hf
    a, b = _ids(1, 23, seed=1)[0], _ids(1, 41, seed=2)[0]
    with torch.no_grad():
        want_a = hf(torch.tensor(a[None])).logits.numpy()[0]
        want_b = hf(torch.tensor(b[None])).logits.numpy()[0]
    model = _ours(folder, gdn_chunk=16)
    pad = 6
    tokens = np.concatenate([a, b, np.zeros(pad, int)])[None]
    seg = np.concatenate([np.full(23, 1), np.full(41, 2), np.zeros(pad, int)])[None]
    pos = np.concatenate([np.arange(23), np.arange(41), np.zeros(pad, int)])[None]
    got = np.asarray(
        model.logits(model.hidden(jnp.asarray(tokens), jnp.asarray(seg), jnp.asarray(pos)))
    )[0]
    np.testing.assert_allclose(got[:23], want_a, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(got[23:64], want_b, atol=2e-4, rtol=2e-4)
    assert np.all(np.isfinite(got))


def test_label_readout_equals_full_logit_gather(tiny_hf):
    folder, _ = tiny_hf
    model = _ours(folder)
    ids = jnp.asarray(_ids())
    seg = jnp.ones(ids.shape, jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape)
    hidden = model.hidden(ids, seg, pos)
    slots = jnp.asarray([[5, 69], [0, 33]])
    labels = jnp.asarray([[[1, 2, 3], [7, 8, 9]], [[4, 5, 6], [10, 11, 12]]])
    got = model.label_logits(hidden, slots, labels)
    full = model.logits(hidden)
    want = jnp.take_along_axis(jnp.take_along_axis(full, slots[..., None], axis=1), labels, axis=-1)
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_fresh_lora_is_identity(tiny_hf):
    folder, _ = tiny_hf
    config, tensors = load_hf(folder, "float32")
    base = _ours(folder)
    adapted = Qwen35(
        stack_layers(tensors, config),
        config,
        ComputeSpec(),
        LoRASpec(rank=4),
        dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    lora = nnx.state(adapted, nnx.LoRAParam)
    assert len(jnp.concatenate([x.ravel() for x in jax.tree.leaves(lora)])) > 0
    ids = jnp.asarray(_ids())
    seg = jnp.ones(ids.shape, jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape)
    np.testing.assert_allclose(adapted.hidden(ids, seg, pos), base.hidden(ids, seg, pos), atol=1e-5)


@pytest.mark.parametrize(
    "compute", [{"attention_block": 16}, {"attention_block": 16, "remat": "none"}]
)
def test_blocked_attention_matches_hf(tiny_hf, compute):
    folder, hf = tiny_hf
    ids = _ids(length=53)
    with torch.no_grad():
        want = hf(torch.tensor(ids)).logits.numpy()
    model = _ours(folder, gdn_chunk=16, **compute)
    seg = jnp.ones(ids.shape, jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape)
    got = model.logits(model.hidden(jnp.asarray(ids), seg, pos))
    np.testing.assert_allclose(got, want, atol=2e-4, rtol=2e-4)


def test_lora_gradients_independent_of_remat_and_blocking(tiny_hf):
    folder, _ = tiny_hf
    config, tensors = load_hf(folder, "float32")
    ids = jnp.asarray(_ids(length=45))
    seg = jnp.asarray(np.concatenate([np.full((2, 20), 1), np.full((2, 25), 2)], axis=1))
    pos = jnp.asarray(
        np.concatenate([np.tile(np.arange(20), (2, 1)), np.tile(np.arange(25), (2, 1))], axis=1)
    )
    grads = []
    for compute in (
        ComputeSpec(remat="none", attention_block=0),
        ComputeSpec(remat="full", attention_block=16),
        ComputeSpec(remat="minimal", attention_block=8, gdn_chunk=16),
    ):
        model = Qwen35(
            stack_layers(dict(tensors), config),
            config,
            compute,
            LoRASpec(rank=4),
            dtype=jnp.float32,
            rngs=nnx.Rngs(0),
        )
        g, lora, frozen = nnx.split(model, nnx.LoRAParam, ...)
        # non-zero B so gradients reach A too
        lora = jax.tree.map(lambda x: x + 0.01, lora)

        def loss(lo, g=g, frozen=frozen):
            m = nnx.merge(g, lo, frozen)
            return jnp.sum(m.hidden(ids, seg, pos) ** 2)

        grads.append(jax.grad(loss)(lora))
    # Same chunking: remat/blocking must not change gradients beyond fp32 noise. A different
    # chunk size reorders fp32 sums (loss is a large sum of squares), hence the looser bound.
    # Error normalised by each leaf's max magnitude (element-wise rtol amplifies noise in
    # near-cancelling entries). This randomly-normed HF fixture amplifies fp32 reordering
    # noise to ~4e-4; a structural bug gives O(1). (On the tjev.testing fixture: ~2e-5.)
    for other, tol in ((grads[1], 1e-3), (grads[2], 5e-3)):
        for a, b in zip(jax.tree.leaves(grads[0]), jax.tree.leaves(other), strict=True):
            err = np.max(np.abs(np.asarray(a) - np.asarray(b))) / (np.max(np.abs(a)) + 1e-12)
            assert err < tol, err


def test_untied_head_matches_hf(tmp_path):
    """9B layout: a separate lm_head (tie_word_embeddings false) is loaded and read out."""
    config = ModelConfig.tiny(tie_word_embeddings=False)
    raw = config.to_hf()
    raw.pop("architectures")
    raw.pop("model_type")
    torch.manual_seed(1)
    hf = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**raw)).eval()
    assert hf.lm_head.weight.data_ptr() != hf.model.embed_tokens.weight.data_ptr()
    hf.save_pretrained(tmp_path, safe_serialization=True)
    model = _ours(tmp_path, gdn_chunk=16)
    assert model.head is not None
    ids = _ids(seed=3)
    with torch.no_grad():
        want = hf(torch.tensor(ids)).logits.numpy()
    seg = jnp.ones(ids.shape, jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape)
    hidden = model.hidden(jnp.asarray(ids), seg, pos)
    np.testing.assert_allclose(model.logits(hidden), want, atol=2e-4, rtol=2e-4)
    slots = jnp.asarray([[5, 69], [0, 33]])
    labels = jnp.asarray([[[1, 2, 3], [7, 8, 9]], [[4, 5, 6], [10, 11, 12]]])
    got = model.label_logits(hidden, slots, labels)
    gathered = np.take_along_axis(np.take_along_axis(want, np.asarray(slots)[..., None], 1),
                                  np.asarray(labels), -1)  # fmt: skip
    np.testing.assert_allclose(got, gathered, atol=2e-4, rtol=2e-4)
