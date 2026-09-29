"""Gated DeltaNet layer: projections, segmented causal conv, gates, the delta rule.

Packed rows: ``segment_ids`` [B,T] (0 = padding) isolate segments, so a packed row computes
exactly what each segment would compute alone. The delta rule and the conv are ops from
:mod:`tjev.kernels`, chosen by ``ComputeSpec``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name

from tjev.config import ComputeSpec
from tjev.kernels import gated_delta_rule, l2norm, segmented_causal_conv1d
from tjev.model.config import ModelConfig
from tjev.model.lora import LinearFactory, grouped
from tjev.model.norms import GatedRMSNorm
from tjev.model.params import Frozen, Weights


class GatedDeltaNet(nnx.Module):
    """in_proj (qkv, z, b, a) → causal conv + silu → l2norm(q, k), β = σ(b),
    g = −exp(A_log)·softplus(a + dt_bias) → delta rule → gated RMSNorm(·, z) → out_proj."""

    def __init__(
        self, w: Weights, config: ModelConfig, compute: ComputeSpec, linear: LinearFactory
    ):
        self.config = config
        self.compute = compute
        self.in_proj_qkv = linear("in_proj_qkv", w["in_proj_qkv.weight"])
        self.in_proj_z = linear("in_proj_z", w["in_proj_z.weight"])
        self.in_proj_b = linear("in_proj_b", w["in_proj_b.weight"])
        self.in_proj_a = linear("in_proj_a", w["in_proj_a.weight"])
        self.out_proj = linear("out_proj", w["out_proj.weight"])
        # HF conv1d.weight [C,1,K] -> [K,C]
        self.conv = Frozen(jnp.swapaxes(jnp.asarray(w["conv1d.weight"])[..., 0, :], -1, -2))
        self.A_log = Frozen(jnp.asarray(w["A_log"], jnp.float32))
        self.dt_bias = Frozen(jnp.asarray(w["dt_bias"], jnp.float32))
        self.norm = GatedRMSNorm(w["norm.weight"], config.rms_norm_eps)

    def core(self, qkv: jax.Array, b: jax.Array, a: jax.Array, segment_ids: jax.Array) -> jax.Array:
        """conv → split → gates → l2norm → delta rule. Rematerialised as one unit, so its
        fp32 intermediates live only during its own backward pass."""
        c = self.config
        batch, length, _ = qkv.shape
        valid = (segment_ids > 0)[..., None]
        with jax.named_scope("gdn_conv"):
            qkv = jax.nn.silu(segmented_causal_conv1d(qkv, self.conv[...], segment_ids))
            qkv = jnp.where(valid, qkv, 0).astype(qkv.dtype)
        q, k, v = jnp.split(qkv, [c.linear_key_dim, 2 * c.linear_key_dim], axis=-1)
        q = q.reshape(batch, length, c.linear_num_key_heads, c.linear_key_head_dim)
        k = k.reshape(batch, length, c.linear_num_key_heads, c.linear_key_head_dim)
        v = v.reshape(batch, length, c.linear_num_value_heads, c.linear_value_head_dim)
        beta = jax.nn.sigmoid(b.astype(jnp.float32))
        g = -jnp.exp(self.A_log[...]) * jax.nn.softplus(a.astype(jnp.float32) + self.dt_bias[...])
        ratio = c.linear_num_value_heads // c.linear_num_key_heads
        if ratio > 1:
            q = jnp.repeat(q, ratio, axis=2)
            k = jnp.repeat(k, ratio, axis=2)
        q = l2norm(q.astype(jnp.float32)) * c.linear_key_head_dim**-0.5
        k = l2norm(k.astype(jnp.float32))
        with jax.named_scope("gdn_rule"):
            out = gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                segment_ids,
                impl=self.compute.gdn_impl,
                chunk=self.compute.gdn_chunk,
                precision=self.compute.gdn_precision,
            )
        return out.astype(qkv.dtype)

    def __call__(self, x: jax.Array, segment_ids: jax.Array) -> jax.Array:
        c = self.config
        batch, length, _ = x.shape
        valid = (segment_ids > 0)[..., None]
        x = jnp.where(valid, x, 0).astype(x.dtype)
        with jax.named_scope("gdn_proj"):
            qkv, z, b, a = grouped(
                x, self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a,
                names=("gdn_qkv", "gdn_z", "gdn_ba", "gdn_ba"),
            )  # fmt: skip
            z = z.reshape(batch, length, c.linear_num_value_heads, -1)
        core: Callable[..., jax.Array] = self.core
        if self.compute.remat != "none":
            core = jax.checkpoint(core, prevent_cse=False)
        with jax.named_scope("gdn_core"):
            out = checkpoint_name(core(qkv, b, a, segment_ids), "gdn_core_out")
        with jax.named_scope("gdn_out"):
            out = self.norm(out, z)
            out = jnp.where(valid, out.reshape(batch, length, -1), 0).astype(x.dtype)
            return self.out_proj(out, name="mixer_out")
