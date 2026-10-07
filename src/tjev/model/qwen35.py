"""Qwen3.5 dense text decoder in Flax NNX, scanned over super-blocks.

A super-block is one period of the layer pattern: three Gated DeltaNet layers followed by
one gated full-attention layer (MaxText ``Qwen3_5ScannableBlock``). Weights of all
super-blocks are stacked on a leading axis and the forward pass is one ``lax.scan``, so
compile time does not grow with depth.

Packed rows: ``segment_ids`` [B,T] (0 = padding) isolate segments in both mixers, and
``positions`` restart at 0 in every segment, so a packed row computes exactly what each
segment would compute alone. The decision readout (:meth:`Qwen35.label_logits`) reads only
the output-head rows of the answer letters at the answer slots: there is never a [T, V]
full-vocabulary projection in training.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tjev.config import ComputeSpec, LoRASpec
from tjev.model.attention import GatedAttention
from tjev.model.config import ModelConfig
from tjev.model.gdn import GatedDeltaNet
from tjev.model.lora import Linear, LinearFactory, dropout_scope
from tjev.model.mlp import MLP
from tjev.model.norms import RMSNorm
from tjev.model.params import Frozen, Probe, Weights, sub_weights
from tjev.model.weights import load_hf, stack_layers

_SAVE = jax.checkpoint_policies.save_only_these_names
# Per-decoder-layer remat policies (ComputeSpec.remat; "none" disables remat). With a
# frozen base the backward of a base GEMM is dX only, so full remat spends about a third of
# the GEMM time recomputing; the policies trade that for memory.
# The GDN core output and splash's attention context: the layer recompute skips the
# sequence-mixing cores (their own backward still recomputes their internals).
_CORE = ("gdn_core_out", "attn_context")
# + MLP gate/up, the mixer output (needed by the MLP's backward), LoRA x·A and the GDN /
# attention in-projections.
_MINIMAL = (*_CORE, "mlp_gate", "mlp_up", "mixer_out", "lora_xa", "gdn_qkv", "gdn_z", "gdn_ba")
# + the Pallas GDN kernel's residuals (the state before each chunk, v_new): with them saved
# and the core not in a checkpoint of its own, the forward kernel runs once per microbatch
# (under full / core it runs 3 / 2 times: the primal, the layer recompute, the core's own
# recompute for its residuals). Costs ~1.5 KB per token and value head per GDN layer (fp32).
_KERNEL = ("gdn_states", "gdn_vnew", "gdn_tinv")  # gdn_tinv: pallas_tpu_v2's (I+L)⁻¹
_STATES = (*_CORE, *_KERNEL)
REMAT_POLICIES = {
    "full": None,
    "core": _SAVE(*_CORE),
    "core_states": _SAVE(*_STATES),
    "minimal": _SAVE(*_MINIMAL, "attn_qkv"),
    # minimal + core_states: the layer recompute is down to the norms, the gates, the conv
    # and the small elementwise ops, and the GDN forward kernel runs once per microbatch.
    # Its saved activations grow with T: per bucket (ComputeSpec.remat_by_bucket), where it
    # fits.
    "minimal_states": _SAVE(*_MINIMAL, "attn_qkv", *_KERNEL),
}


_BLOCK: list[int] = []  # the super-block being traced, under TJEV_UNROLL_LAYERS


def unrolled_layers() -> bool:
    """TJEV_UNROLL_LAYERS=1: trace the super-blocks one by one instead of a ``lax.scan``
    (a profiling capture with per-layer names; same math, but not the training program)."""
    return os.environ.get("TJEV_UNROLL_LAYERS") == "1"


@contextlib.contextmanager
def _block_index(s: int):
    _BLOCK.append(s)
    try:
        yield
    finally:
        _BLOCK.pop()


class SuperBlock(nnx.Module):
    """3 × (norm, GDN, norm, MLP) + 1 × (norm, gated attention, norm, MLP)."""

    def __init__(
        self,
        layers: list[Weights],
        config: ModelConfig,
        compute: ComputeSpec,
        linear: LinearFactory,
    ):
        interval = config.full_attention_interval
        eps = config.rms_norm_eps
        self.input_norms = nnx.List([RMSNorm(w["input_layernorm.weight"], eps) for w in layers])
        self.post_norms = nnx.List(
            [RMSNorm(w["post_attention_layernorm.weight"], eps) for w in layers]
        )
        self.mlps = nnx.List([MLP(sub_weights(w, "mlp."), linear) for w in layers])
        self.gdn = nnx.List(
            [
                GatedDeltaNet(sub_weights(w, "linear_attn."), config, compute, linear)
                for w in layers[: interval - 1]
            ]
        )
        self.attention = GatedAttention(
            sub_weights(layers[-1], "self_attn."), config, compute, linear
        )
        self.remat = compute.remat
        # hashable (a static attribute of the graph): ((length, policy), ...)
        self.remat_by_bucket = tuple(sorted((int(k), v) for k, v in
                                            compute.remat_by_bucket.items()))  # fmt: skip

    def remat_at(self, length: int) -> str:
        """The policy at row length ``length``: ``remat`` when it is "none" (set so by EVA's
        probe passes), else the bucket's own policy if it has one."""
        if self.remat == "none":
            return "none"
        return dict(self.remat_by_bucket).get(length, self.remat)

    def layer(
        self, i: int, x: jax.Array, segment_ids: jax.Array, positions: jax.Array,
        remat: str | None = None,
    ) -> jax.Array:  # fmt: skip
        """``remat``: this layer's policy (None: ``compute.remat``, the GDN / attention own)."""
        # layer0_gdn … layer2_gdn, layer3_attn: the position in the super-block and the mixer
        # (xprof's framework op stats group every op, primal, remat recompute and backward,
        # by it, then by the inner scopes); the super-block itself is the scan's iteration
        kind = f"layer{i}_gdn" if i < len(self.gdn) else f"layer{i}_attn"
        block = _BLOCK[-1] if _BLOCK else None  # unrolled: the layer's own number, L00 … L31
        number = (
            jax.named_scope(f"L{block * len(self.mlps) + i:02d}")
            if block is not None
            else contextlib.nullcontext()
        )
        with number, jax.named_scope(kind):
            with jax.named_scope("norm"):
                h = self.input_norms[i](x)
            if i < len(self.gdn):
                h = self.gdn[i](h, segment_ids, remat=remat)
            else:
                h = self.attention(h, segment_ids, positions, remat=remat)
            with jax.named_scope("residual"):  # the stream the next layer reads (and saves)
                x = x + h
            with jax.named_scope("norm"):
                h = self.post_norms[i](x)
            h = self.mlps[i](h)
            with jax.named_scope("residual"):
                return x + h

    def __call__(self, x: jax.Array, segment_ids: jax.Array, positions: jax.Array,
                 dropout: tuple[jax.Array, float] | None = None) -> jax.Array:  # fmt: skip
        """``dropout``: (this block's key, rate) for the adapter-input dropout (training)."""
        # Remat per decoder layer (not per super-block): the backward pass then holds one
        # layer's activations at a time, ~4x less peak memory than block-level remat. The
        # policy is the bucket's: the row length is static, one compiled step per bucket.
        remat = self.remat_at(x.shape[1])
        for i in range(len(self.mlps)):

            def layer(h: jax.Array, i: int = i) -> jax.Array:
                if dropout is None:
                    return self.layer(i, h, segment_ids, positions, remat)
                with dropout_scope(jax.random.fold_in(dropout[0], i), dropout[1]):
                    return self.layer(i, h, segment_ids, positions, remat)

            fn: Callable[[jax.Array], jax.Array] = layer
            if remat != "none":
                fn = jax.checkpoint(layer, policy=REMAT_POLICIES[remat], prevent_cse=False)
            x = fn(x)
        return x


class Qwen35(nnx.Module):
    probing = False  # True while an EVA probe pass runs (tjev.model.lora_init)

    def __init__(
        self,
        weights: dict,
        config: ModelConfig,
        compute: ComputeSpec | None = None,
        lora: LoRASpec | None = None,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        head_rows: Sequence[int] | None = None,
    ):
        """``weights``: HF text-model names without the ``model.language_model.`` prefix,
        each ``layers.{i}.*`` tensor already stacked by :func:`stack_layers`.
        ``head_rows``: the token ids the readout may ask for (``compute.head_rows=letters``:
        the answer letters and the padding id 0); an untied output head then keeps only
        these rows."""
        self.config = config
        self.compute = compute = compute or ComputeSpec()
        self.dtype = dtype
        # lora.dropout drops the adapter input alone, which merged LoRA (x·(W + s·AB))
        # cannot: such a model builds the (equal) unmerged layers instead
        merged = compute.lora_impl == "merged" and not (lora is not None and lora.dropout)
        if compute.kernel_out_in and lora is not None and lora.init != "default":
            raise ValueError("compute.kernel_out_in: lora.init reads the kernels as [in, out]")
        linear = LinearFactory(lora, dtype, rngs or nnx.Rngs(0), merged=merged,
                               flat_rows=compute.flat_rows,
                               out_in=compute.kernel_out_in)  # fmt: skip
        self.embed = Frozen(jnp.asarray(weights["embed_tokens.weight"], dtype))
        # Output head rows: the embedding itself when tied (0.8B-4B), else lm_head (9B)
        head = head_ids = None
        if not config.tie_word_embeddings:
            head = weights["lm_head.weight"]
            # compute.head_rows=letters: only the rows the readout reads (the 9B lm_head is
            # 1.9 GiB of bf16, the letters 27 rows), sliced before it reaches the device;
            # head_ids: their token ids, sorted (the readout finds a row by binary search)
            if compute.head_rows == "letters":
                if head_rows is None:
                    raise ValueError("compute.head_rows=letters: pass head_rows (build_model does)")
                ids = np.unique(np.asarray([0, *head_rows], np.int32))
                head = head[ids]
                head_ids = Frozen(jnp.asarray(ids))
            head = Frozen(jnp.asarray(head, dtype))
        self.head, self.head_ids = head, head_ids
        self.blocks = SuperBlock(weights["blocks"], config, compute, linear)
        self.norm = RMSNorm(weights["norm.weight"], config.rms_norm_eps)

    def hidden(self, tokens: jax.Array, segment_ids: jax.Array, positions: jax.Array,
               dropout: tuple[jax.Array, float] | None = None) -> jax.Array:  # fmt: skip
        """Final-norm hidden states [B,T,D] in the compute dtype. ``dropout``: (key, rate)
        of the adapter-input dropout (training steps only; tjev.model.lora)."""
        with jax.named_scope("embed"):
            x = self.embed[...][tokens]
        graphdef, state = nnx.split(self.blocks)
        probing = self.probing
        rate = dropout[1] if dropout is not None else 0.0
        count = self.config.num_super_blocks
        keys = jax.random.split(dropout[0], count) if rate else None

        def body(h: jax.Array, xs) -> tuple[jax.Array, nnx.State | None]:
            block_state, key = xs
            block = nnx.merge(graphdef, block_state)
            h = block(h, segment_ids, positions, None if key is None else (key, rate))
            # EVA probes: the statistics each block added come out stacked like the weights
            return h, (nnx.state(block, Probe) if probing else None)

        if unrolled_layers() and not probing:
            # profiling only (TJEV_UNROLL_LAYERS=1): a Python loop over the super-blocks, so
            # every layer has its own ops, buffers and name (L00 … L31); the
            # compile grows with depth and XLA may fuse differently than under the scan
            n = self.config.num_super_blocks
            for s in range(n):
                xs = jax.tree.map(lambda t, s=s: t[s], (state, keys))
                with _block_index(s):
                    x, _ = body(x, xs)
            with jax.named_scope("final_norm"):
                return self.norm(x)
        x, probes = jax.lax.scan(body, x, (state, keys))
        if probing:
            nnx.update(self.blocks, probes)
        with jax.named_scope("final_norm"):
            return self.norm(x)

    def output_head(self) -> jax.Array:
        if self.head_ids is not None:
            raise ValueError("compute.head_rows=letters keeps only the letter rows of the head")
        return (self.embed if self.head is None else self.head)[...]

    def head_rows_of(self, label_ids: jax.Array) -> jax.Array:
        """Output-head rows [..., D] of ``label_ids`` (fp32). With a letters-only head an id
        outside it reads NaN: loud (the step rejects a non-finite update), never a wrong
        row."""
        if self.head_ids is None:
            return self.output_head()[label_ids].astype(jnp.float32)
        ids = self.head_ids[...]
        at = jnp.clip(jnp.searchsorted(ids, label_ids), 0, ids.shape[0] - 1)
        rows = self.head[...][at].astype(jnp.float32)
        return jnp.where((ids[at] == label_ids)[..., None], rows, jnp.nan)

    def label_logits(self, hidden: jax.Array, slots: jax.Array, label_ids: jax.Array) -> jax.Array:
        """FP32 logits of the label tokens at answer slots.

        hidden [B,T,D]; slots [B,S] (token index of each answer slot); label_ids [B,S,K].
        Only K output-head rows are read: no [T,V] full-vocabulary projection.
        """
        with jax.named_scope("readout"):
            h = jnp.take_along_axis(hidden, slots[..., None], axis=1).astype(jnp.float32)
            rows = self.head_rows_of(label_ids)  # [B,S,K,D]
            return jnp.einsum("bsd,bskd->bsk", h, rows, precision=jax.lax.Precision.HIGHEST)

    def logits(self, hidden: jax.Array) -> jax.Array:
        """Full-vocabulary logits (tests and parity only); fp32 accumulation, no fp32
        copy of the [V,D] embedding."""
        embed = self.output_head()
        return jnp.einsum(
            "btd,vd->btv", hidden.astype(embed.dtype), embed, preferred_element_type=jnp.float32
        )

    def adapted_linears(self) -> dict[str, tuple[Linear, int]]:
        """HF text-model weight name → (stacked Linear, super-block index), adapted only."""
        return {k: v for k, v in self._all_linears().items() if v[0].rank}

    def _all_linears(self) -> dict[str, tuple[Linear, int]]:
        out = {}
        interval = self.config.full_attention_interval
        blocks = self.blocks
        for s in range(self.config.num_super_blocks):
            for j in range(interval):
                base = f"layers.{s * interval + j}."
                mlp = blocks.mlps[j]
                for name in ("gate_proj", "up_proj", "down_proj"):
                    out[base + f"mlp.{name}.weight"] = (getattr(mlp, name), s)
                if j < interval - 1:
                    gdn = blocks.gdn[j]
                    for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                        out[base + f"linear_attn.{name}.weight"] = (getattr(gdn, name), s)
                else:
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                        out[base + f"self_attn.{name}.weight"] = (
                            getattr(blocks.attention, name),
                            s,
                        )
        return out

    def full_linears(self) -> dict[str, tuple[Linear, int]]:
        """HF text-model weight name → (stacked Linear, super-block index) for the
        projections whose kernel trains directly (``lora.gate_full``)."""
        return {k: v for k, v in self._all_linears().items() if v[0].full}


def build_model(
    folder: str | Path,
    compute: ComputeSpec | None = None,
    lora: LoRASpec | None = None,
    *,
    dtype: str = "bfloat16",
    seed: int = 0,
) -> tuple[ModelConfig, Qwen35]:
    """Load a HF Qwen3.5 snapshot into a (LoRA-adapted) scanned NNX model."""
    config, tensors = load_hf(folder, dtype)
    compute = compute or ComputeSpec()
    head_rows = None
    if compute.head_rows == "letters":
        from tjev.data.tokenize import PromptTokenizer

        head_rows = PromptTokenizer(folder).letter_ids
    model = Qwen35(
        stack_layers(tensors, config),
        config,
        compute,
        lora,
        dtype=getattr(jnp, dtype),
        rngs=nnx.Rngs(seed),
        head_rows=head_rows,
    )
    return config, model
