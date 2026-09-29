"""Gated full-attention layer: [query | gate] projection, q/k norms, RoPE, GQA attention
within segments, output gate."""

from __future__ import annotations

import jax
from flax import nnx

from tjev.config import ComputeSpec
from tjev.kernels import attention
from tjev.model.config import ModelConfig
from tjev.model.lora import LinearFactory, grouped
from tjev.model.norms import RMSNorm
from tjev.model.params import Weights
from tjev.model.rope import rotary


class GatedAttention(nnx.Module):
    """q_proj emits [query | gate] per head; q/k RMSNorm, partial RoPE, GQA attention within
    segments, then the output gate σ(gate) before o_proj."""

    def __init__(
        self, w: Weights, config: ModelConfig, compute: ComputeSpec, linear: LinearFactory
    ):
        self.config = config
        self.compute = compute
        self.q_proj = linear("q_proj", w["q_proj.weight"])
        self.k_proj = linear("k_proj", w["k_proj.weight"])
        self.v_proj = linear("v_proj", w["v_proj.weight"])
        self.o_proj = linear("o_proj", w["o_proj.weight"])
        self.q_norm = RMSNorm(w["q_norm.weight"], config.rms_norm_eps)
        self.k_norm = RMSNorm(w["k_norm.weight"], config.rms_norm_eps)

    def __call__(self, x: jax.Array, segment_ids: jax.Array, positions: jax.Array) -> jax.Array:
        c = self.config
        batch, length, _ = x.shape
        with jax.named_scope("attn_proj"):
            qg, k, v = grouped(x, self.q_proj, self.k_proj, self.v_proj, names=("attn_qkv",) * 3)
            qg = qg.reshape(batch, length, c.num_heads, 2 * c.head_dim)
            q, gate = qg[..., : c.head_dim], qg[..., c.head_dim :]
            gate = gate.reshape(batch, length, -1)
            k = k.reshape(batch, length, c.num_kv_heads, c.head_dim)
            v = v.reshape(batch, length, c.num_kv_heads, c.head_dim)
            q = rotary(self.q_norm(q), positions, c.rotary_dim, c.rope_theta)
            k = rotary(self.k_norm(k), positions, c.rotary_dim, c.rope_theta)
        with jax.named_scope("attention_core"):
            out = attention(
                q,
                k,
                v,
                segment_ids,
                impl=self.compute.attention,
                block=self.compute.attention_block,
                remat=self.compute.remat != "none",
            )
        with jax.named_scope("attn_out"):
            out = out.reshape(batch, length, -1) * jax.nn.sigmoid(gate)
            return self.o_proj(out, name="mixer_out")
