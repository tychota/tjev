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

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from tjev.config import ComputeSpec, LoRASpec
from tjev.model.config import ModelConfig
from tjev.model.layers import Frozen, Linear, RMSNorm, grouped
from tjev.model.mixers import GatedAttention, GatedDeltaNet, LinearFactory, Weights
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
REMAT_POLICIES = {
    "full": None,
    "core": _SAVE(*_CORE),
    "minimal": _SAVE(*_MINIMAL, "attn_qkv"),
}


class MLP(nnx.Module):
    def __init__(self, w: Weights, linear: LinearFactory):
        self.gate_proj = linear("gate_proj", w["gate_proj.weight"])
        self.up_proj = linear("up_proj", w["up_proj.weight"])
        self.down_proj = linear("down_proj", w["down_proj.weight"])

    def __call__(self, x: jax.Array) -> jax.Array:
        with jax.named_scope("mlp"):
            gate, up = grouped(x, self.gate_proj, self.up_proj, names=("mlp_gate", "mlp_up"))
            return self.down_proj(jax.nn.silu(gate) * up)


def _sub(w: Weights, prefix: str) -> Weights:
    return {k[len(prefix) :]: v for k, v in w.items() if k.startswith(prefix)}


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
        self.mlps = nnx.List([MLP(_sub(w, "mlp."), linear) for w in layers])
        self.gdn = nnx.List(
            [
                GatedDeltaNet(_sub(w, "linear_attn."), config, compute, linear)
                for w in layers[: interval - 1]
            ]
        )
        self.attention = GatedAttention(_sub(layers[-1], "self_attn."), config, compute, linear)
        self.remat = compute.remat

    def layer(
        self, i: int, x: jax.Array, segment_ids: jax.Array, positions: jax.Array
    ) -> jax.Array:
        with jax.named_scope("norm"):
            h = self.input_norms[i](x)
        if i < len(self.gdn):
            h = self.gdn[i](h, segment_ids)
        else:
            h = self.attention(h, segment_ids, positions)
        x = x + h
        with jax.named_scope("norm"):
            h = self.post_norms[i](x)
        return x + self.mlps[i](h)

    def __call__(self, x: jax.Array, segment_ids: jax.Array, positions: jax.Array) -> jax.Array:
        # Remat per decoder layer (not per super-block): the backward pass then holds one
        # layer's activations at a time, ~4x less peak memory than block-level remat.
        for i in range(len(self.mlps)):

            def layer(h: jax.Array, i: int = i) -> jax.Array:
                return self.layer(i, h, segment_ids, positions)

            fn: Callable[[jax.Array], jax.Array] = layer
            if self.remat != "none":
                fn = jax.checkpoint(layer, policy=REMAT_POLICIES[self.remat], prevent_cse=False)
            x = fn(x)
        return x


class Qwen35(nnx.Module):
    def __init__(
        self,
        weights: dict,
        config: ModelConfig,
        compute: ComputeSpec | None = None,
        lora: LoRASpec | None = None,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        """``weights``: HF text-model names without the ``model.language_model.`` prefix,
        each ``layers.{i}.*`` tensor already stacked by :func:`stack_layers`."""
        self.config = config
        self.compute = compute = compute or ComputeSpec()
        self.dtype = dtype
        linear = LinearFactory(lora, dtype, rngs or nnx.Rngs(0))
        self.embed = Frozen(jnp.asarray(weights["embed_tokens.weight"], dtype))
        # Output head rows: the embedding itself when tied (0.8B-4B), else lm_head (9B)
        self.head = (
            None
            if config.tie_word_embeddings
            else Frozen(jnp.asarray(weights["lm_head.weight"], dtype))
        )
        self.blocks = SuperBlock(weights["blocks"], config, compute, linear)
        self.norm = RMSNorm(weights["norm.weight"], config.rms_norm_eps)

    def hidden(self, tokens: jax.Array, segment_ids: jax.Array, positions: jax.Array) -> jax.Array:
        """Final-norm hidden states [B,T,D] in the compute dtype."""
        x = self.embed[...][tokens]
        graphdef, state = nnx.split(self.blocks)

        def body(h: jax.Array, block_state: nnx.State) -> tuple[jax.Array, None]:
            block = nnx.merge(graphdef, block_state)
            return block(h, segment_ids, positions), None

        x, _ = jax.lax.scan(body, x, state)
        return self.norm(x)

    def output_head(self) -> jax.Array:
        return (self.embed if self.head is None else self.head)[...]

    def label_logits(self, hidden: jax.Array, slots: jax.Array, label_ids: jax.Array) -> jax.Array:
        """FP32 logits of the label tokens at answer slots.

        hidden [B,T,D]; slots [B,S] (token index of each answer slot); label_ids [B,S,K].
        Only K output-head rows are read: no [T,V] full-vocabulary projection.
        """
        h = jnp.take_along_axis(hidden, slots[..., None], axis=1).astype(jnp.float32)
        rows = self.output_head()[label_ids].astype(jnp.float32)  # [B,S,K,D]
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
        return {k: v for k, v in out.items() if v[0].rank}


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
    model = Qwen35(
        stack_layers(tensors, config),
        config,
        compute or ComputeSpec(),
        lora,
        dtype=getattr(jnp, dtype),
        rngs=nnx.Rngs(seed),
    )
    return config, model
