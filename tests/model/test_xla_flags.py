"""The XLA-side compute flags (ComputeSpec: gdn_norm, gdn_input_mask, flat_rows,
kernel_out_in, head_rows) compute what the defaults compute: the official model's
logits, and the default's loss and LoRA gradients in fp32 and bf16, on packed rows with
padding (junk tokens in it)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tjev.config import ComputeSpec, LoRASpec
from tjev.model import ModelConfig, Qwen35, load_hf, stack_layers
from tjev.model.norms import GatedRMSNorm

FLAGS = {
    "gdn_norm=flat": {"gdn_norm": "flat"},
    "gdn_input_mask=false": {"gdn_input_mask": False},
    "flat_rows": {"flat_rows": True},
    "kernel_out_in": {"kernel_out_in": ("in_proj_z", "out_proj", "o_proj", "k_proj")},
    "merged+kernel_out_in": {"lora_impl": "merged", "kernel_out_in": ("in_proj_z", "out_proj")},
    "flat+nomask+rows+out_in": {"gdn_norm": "flat", "gdn_input_mask": False, "flat_rows": True,
                                "kernel_out_in": ("in_proj_z", "o_proj")},
}  # fmt: skip
LETTER_IDS = (40, 41, 42, 43, 77)  # stand-ins for the answer letters' token ids


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A tiny random Qwen3.5 with an untied head (the 9B layout), saved as a HF snapshot."""
    config = ModelConfig.tiny(tie_word_embeddings=False)
    raw = config.to_hf()
    raw.pop("architectures")
    raw.pop("model_type")
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**raw)).eval()
    with torch.no_grad():  # non-trivial norms, decays and conv (HF init zeros some)
        for name, p in model.named_parameters():
            if "norm" in name or "A_log" in name or "dt_bias" in name:
                p.copy_(torch.randn_like(p) * 0.3 + (1.0 if "linear_attn.norm" in name else 0.0))
    folder = tmp_path_factory.mktemp("tiny-untied")
    model.save_pretrained(folder, safe_serialization=True)
    return folder, model


def _model(folder, dtype="float32", lora=False, **compute):
    config, tensors = load_hf(folder, dtype)
    return Qwen35(
        stack_layers(tensors, config), config, ComputeSpec(gdn_chunk=16, **compute),
        LoRASpec(rank=4) if lora else None, dtype=getattr(jnp, dtype), rngs=nnx.Rngs(0),
        head_rows=LETTER_IDS,
    )  # fmt: skip


def _packed(seed=0):
    """Two rows: segments of 23 + 30 tokens then 11 padding tokens (random ids, not 0), and
    one 64-token segment."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(1, 512, (2, 64))
    seg = np.zeros((2, 64), np.int32)
    seg[0, :23], seg[0, 23:53], seg[1] = 1, 2, 1
    pos = np.zeros((2, 64), np.int32)
    pos[0, :23], pos[0, 23:53], pos[1] = np.arange(23), np.arange(30), np.arange(64)
    return jnp.asarray(tokens), jnp.asarray(seg), jnp.asarray(pos)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_flat_gated_norm_equals_per_head(dtype):
    heads, dim = 4, 16
    kx, kg, kw, kc = jax.random.split(jax.random.key(0), 4)
    x = (3 * jax.random.normal(kx, (2, 7, heads * dim))).astype(dtype)
    gate = jax.random.normal(kg, (2, 7, heads * dim)).astype(dtype)
    norm = GatedRMSNorm(1 + 0.3 * jax.random.normal(kw, (dim,)), 1e-6)
    cot = jax.random.normal(kc, x.shape)

    def per_head(x, g):
        y = norm(x.reshape(2, 7, heads, dim), g.reshape(2, 7, heads, dim))
        return y.reshape(2, 7, -1)

    def flat(x, g):
        return norm.flat(x, g, heads)

    for fn in (per_head, flat):
        assert fn(x, gate).dtype == x.dtype
    want, got = per_head(x, gate), flat(x, gate)
    loss = lambda fn: lambda x, g: jnp.sum(fn(x, g).astype(jnp.float32) * cot)  # noqa: E731
    dwant = jax.grad(loss(per_head), argnums=(0, 1))(x, gate)
    dgot = jax.grad(loss(flat), argnums=(0, 1))(x, gate)
    if dtype == "float32":
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
        for a, b in zip(dgot, dwant, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
    else:  # the fp32 sums reordered: a bf16 rounding may flip (≤ 1 ulp, rarely)
        w, g = np.asarray(want, np.float32), np.asarray(got, np.float32)
        assert np.mean(w != g) < 0.02
        np.testing.assert_allclose(g, w, rtol=2**-7, atol=1e-6)
        for a, b in zip(dgot, dwant, strict=True):
            a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
            assert np.max(np.abs(a - b)) <= 2**-6 * np.max(np.abs(b))


@pytest.mark.parametrize(
    "compute",
    [*FLAGS.values(), {"head_rows": "letters"},
     {"gdn_norm": "flat", "gdn_impl": "pallas_tpu_mega2"},
     {"gdn_norm": "flat", "gdn_input_mask": False, "gdn_impl": "pallas_tpu_mega2"}],
    ids=[*FLAGS, "head_rows=letters", "flat+mega2", "flat+nomask+mega2"],
)  # fmt: skip
def test_flags_match_hf(tiny, compute):
    """Logits at every valid position of packed rows (HF runs each segment alone) and the
    letters' readout, fp32."""
    folder, hf = tiny
    tokens, seg, pos = _packed()
    model = _model(folder, **compute)
    hidden = model.hidden(tokens, seg, pos)
    t = np.asarray(tokens)
    with torch.no_grad():
        segments = [(0, 0, 23), (0, 23, 53), (1, 0, 64)]
        want = [hf(torch.tensor(t[r : r + 1, a:b])).logits.numpy()[0] for r, a, b in segments]
    slots = jnp.asarray([[5, 40], [0, 63]])
    labels = jnp.broadcast_to(jnp.asarray(LETTER_IDS), (2, 2, len(LETTER_IDS)))
    got = np.asarray(model.label_logits(hidden, slots, labels))
    for r, s in ((0, 0), (0, 1), (1, 0), (1, 1)):
        p = int(slots[r, s])
        idx = next(i for i, (q, a, b) in enumerate(segments) if q == r and a <= p < b)
        a = segments[idx][1]
        np.testing.assert_allclose(got[r, s], want[idx][p - a, list(LETTER_IDS)], atol=2e-4)
    if compute.get("head_rows") != "letters":
        full = np.asarray(model.logits(hidden))
        for (r, a, b), w in zip(segments, want, strict=True):
            np.testing.assert_allclose(full[r, a:b], w, atol=2e-4, rtol=2e-4)


def test_letters_head_keeps_only_letter_rows(tiny):
    folder, _ = tiny
    model = _model(folder, head_rows="letters")
    assert model.head[...].shape[0] == len(LETTER_IDS) + 1  # + the padding id 0
    with pytest.raises(ValueError, match="letters"):
        model.output_head()
    tokens, seg, pos = _packed()
    hidden = model.hidden(tokens, seg, pos)
    got = model.label_logits(
        hidden, jnp.asarray([[3], [4]]), jnp.asarray([[[40, 0, 41]], [[77, 5, 0]]])
    )
    assert np.isfinite(got[0, 0]).all() and np.isfinite(got[1, 0, [0, 2]]).all()
    assert np.isnan(got[1, 0, 1])  # not a letter: NaN (loud), never another row


def _loss_and_grads(folder, dtype, **compute):
    model = _model(folder, dtype, lora=True, **compute)
    g, lora, frozen = nnx.split(model, nnx.LoRAParam, ...)
    leaves, treedef = jax.tree.flatten(lora)
    offsets = [0.01 * jax.random.normal(jax.random.fold_in(jax.random.key(7), i), x.shape)
               for i, x in enumerate(leaves)]  # fmt: skip
    lora = jax.tree.unflatten(treedef, [x + o for x, o in zip(leaves, offsets, strict=True)])
    tokens, seg, pos = _packed()
    valid = (seg > 0)[..., None]
    slots = jnp.asarray([[5, 40], [0, 63]])
    labels = jnp.broadcast_to(jnp.asarray(LETTER_IDS), (2, 2, len(LETTER_IDS)))

    def loss(lo):
        m = nnx.merge(g, lo, frozen)
        h = m.hidden(tokens, seg, pos)
        letters = jax.nn.log_softmax(m.label_logits(h, slots, labels), axis=-1)
        return -jnp.sum(letters[..., 0]) + 1e-3 * jnp.sum(
            jnp.where(valid, h, 0).astype(jnp.float32) ** 2
        )

    value, grads = jax.jit(jax.value_and_grad(loss))(lora)
    return float(value), jax.tree.leaves(grads)


def _errors(got, want) -> tuple[float, float]:
    """(worst leaf's max |Δ| over its max |want|, relative L2 over all leaves)."""
    got = [np.asarray(a, np.float64) for a in got]
    want = [np.asarray(b, np.float64) for b in want]
    worst = max(
        np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30) for a, b in zip(got, want, strict=True)
    )
    flat_g, flat_w = (
        np.concatenate([a.ravel() for a in got]),
        np.concatenate([b.ravel() for b in want]),
    )
    return float(worst), float(np.linalg.norm(flat_g - flat_w) / np.linalg.norm(flat_w))


@pytest.fixture(scope="module")
def truth(tiny):
    """The fp32 defaults' loss and LoRA gradients: what fp32 and bf16 variants are held to."""
    return _loss_and_grads(tiny[0], "float32")


@pytest.fixture(scope="module")
def bf16_default(tiny):
    return _loss_and_grads(tiny[0], "bfloat16")


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    "compute", [*FLAGS.values(), {"head_rows": "letters"}], ids=[*FLAGS, "head_rows=letters"]
)
def test_flags_keep_loss_and_lora_gradients(tiny, truth, bf16_default, dtype, compute):
    """Loss and LoRA gradients (LoRA on every projection, packed rows with padding) against
    the fp32 defaults. fp32: equal up to reordered sums. bf16: no further from fp32 than the
    bf16 defaults are (the bf16 model's own rounding noise: on CPU any change of fusion
    moves bf16 roundings, so bf16 variants are not compared with each other)."""
    folder, _ = tiny
    got_loss, got = _loss_and_grads(folder, dtype, **compute)
    worst, l2 = _errors(got, truth[1])
    rel = abs(got_loss - truth[0]) / abs(truth[0])
    if dtype == "float32":
        print(f"fp32 {compute}: loss {rel:.1e}, grads worst leaf {worst:.1e}, L2 {l2:.1e}")
        assert rel < 1e-6 and worst < 1e-4
        return
    base_worst, base_l2 = _errors(bf16_default[1], truth[1])
    base_rel = abs(bf16_default[0] - truth[0]) / abs(truth[0])
    print(f"bf16 {compute}: loss {rel:.1e} (defaults {base_rel:.1e}), grads worst leaf "
          f"{worst:.1e} ({base_worst:.1e}), L2 {l2:.1e} ({base_l2:.1e})")  # fmt: skip
    # the tiny random model's bf16 noise is large and chaotic: over 4 seeds the flat norm's
    # gradient L2 against fp32 was 0.74-1.52x the defaults' (flat_rows 0.98-1.16x)
    assert rel < 2 * base_rel + 1e-4
    assert l2 < 2 * base_l2 + 1e-3
    assert worst < 2 * base_worst + 1e-3
