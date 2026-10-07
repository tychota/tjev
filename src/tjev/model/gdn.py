"""Gated DeltaNet layer: projections, segmented causal conv, gates, the delta rule.

Packed rows: ``segment_ids`` [B,T] (0 = padding) isolate segments, so a packed row computes
exactly what each segment would compute alone. The delta rule and the conv are ops from
:mod:`tjev.kernels`, chosen by ``ComputeSpec``.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name

from tjev.config import ComputeSpec
from tjev.kernels import gated_delta_rule, l2norm, segmented_causal_conv1d
from tjev.kernels.gated_delta_rule import MEGA, gdn_core_mega
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
        if self.compute.gdn_impl in MEGA:
            return self.core_mega(qkv, b, a, segment_ids)
        c = self.config
        batch, length, _ = qkv.shape
        valid = (segment_ids > 0)[..., None]
        with jax.named_scope("gdn_conv"):
            # XLA:TPU miscompiles the conv fused into the rest of this core: its shifted reads
            # (t-1..t-3) see corrupted rows near 512-row tiles (v6e, jax 0.11.2, libtpu 0.0.48;
            # the conv output 3% off the official model, every layer after it wrong, while
            # the conv alone is exact; the in-place aliasing hazard of openxla/xla#38597,
            # jax-ml/jax#35409). Optimization barriers around it keep it out of the fusion:
            # the TPU then matches the official Qwen3.5 to 1e-6 per layer. =0 reproduces it.
            barrier = os.environ.get("TJEV_GDN_CONV_BARRIER", "1") != "0"
            if barrier:
                qkv = jax.lax.optimization_barrier(qkv)
            conv = segmented_causal_conv1d(qkv, self.conv[...], segment_ids)
            if barrier:
                conv = jax.lax.optimization_barrier(conv)
            qkv = jax.nn.silu(conv)
            qkv = jnp.where(valid, qkv, 0).astype(qkv.dtype)
        q, k, v = jnp.split(qkv, [c.linear_key_dim, 2 * c.linear_key_dim], axis=-1)
        q = q.reshape(batch, length, c.linear_num_key_heads, c.linear_key_head_dim)
        k = k.reshape(batch, length, c.linear_num_key_heads, c.linear_key_head_dim)
        v = v.reshape(batch, length, c.linear_num_value_heads, c.linear_value_head_dim)
        with jax.named_scope("gdn_gates"):
            beta = jax.nn.sigmoid(b.astype(jnp.float32))
            g = -jnp.exp(self.A_log[...]) * jax.nn.softplus(
                a.astype(jnp.float32) + self.dt_bias[...]
            )
        with jax.named_scope("gdn_qk_norm"):  # on the key heads: the rule maps GQA groups
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
                inverse=self.compute.gdn_inverse,
                heads_per_step=self.compute.gdn_heads_per_step,
                vmem_fraction=self.compute.gdn_vmem_fraction,
            )
        return out.astype(qkv.dtype)

    def core_mega(self, qkv: jax.Array, b: jax.Array, a: jax.Array, segment_ids: jax.Array):
        """The core in the GDN megakernels: mega2 from the raw projection (conv + SiLU in
        the kernel, so the XLA conv and its barriers are gone); mega1 after the XLA conv."""
        c, compute = self.config, self.compute
        conv_weight = self.conv[...]
        if compute.gdn_impl == "pallas_tpu_mega1":
            valid = (segment_ids > 0)[..., None]
            with jax.named_scope("gdn_conv"):  # the XLA path's conv, barriers included
                barrier = os.environ.get("TJEV_GDN_CONV_BARRIER", "1") != "0"
                if barrier:
                    qkv = jax.lax.optimization_barrier(qkv)
                conv = segmented_causal_conv1d(qkv, conv_weight, segment_ids)
                if barrier:
                    conv = jax.lax.optimization_barrier(conv)
                qkv = jnp.where(valid, jax.nn.silu(conv), 0).astype(qkv.dtype)
            conv_weight = None
        with jax.named_scope("gdn_rule"):
            out = gdn_core_mega(
                qkv, a, b, self.A_log[...], self.dt_bias[...], conv_weight, segment_ids,
                key_heads=c.linear_num_key_heads, key_dim=c.linear_key_head_dim,
                value_heads=c.linear_num_value_heads, value_dim=c.linear_value_head_dim,
                chunk=compute.gdn_chunk, precision=compute.gdn_precision,
                inverse=compute.gdn_inverse, heads_per_step=compute.gdn_heads_per_step,
                vmem_fraction=compute.gdn_vmem_fraction,
            )  # fmt: skip
        return out.astype(qkv.dtype)

    def __call__(self, x: jax.Array, segment_ids: jax.Array, remat: str | None = None
                 ) -> jax.Array:  # fmt: skip
        """``remat``: the decoder layer's policy (None: ``compute.remat``)."""
        c = self.config
        remat = self.compute.remat if remat is None else remat
        batch, length, _ = x.shape
        valid = (segment_ids > 0)[..., None]
        flat = self.compute.gdn_norm == "flat"
        # Padding rows (segment 0) of x reach nothing valid: the conv taps stay in their
        # segment, q / k / v are zeroed after it, the state restarts at a segment change
        # and the output is zeroed below. compute.gdn_input_mask=false skips this select
        # (fused into the in-projection GEMMs' prologue).
        if self.compute.gdn_input_mask:
            x = jnp.where(valid, x, 0).astype(x.dtype)
        with jax.named_scope("gdn_proj"):
            qkv, z, b, a = grouped(
                x, self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a,
                names=("gdn_qkv", "gdn_z", "gdn_ba", "gdn_ba"),
            )  # fmt: skip
            if not flat:
                z = z.reshape(batch, length, c.linear_num_value_heads, -1)
        core: Callable[..., jax.Array] = self.core
        # own remat unit (fp32 internals live only during its backward), except under
        # remat=none, core_states and minimal_states (whose layer policy saves the kernel
        # residuals)
        if remat not in ("none", "core_states", "minimal_states"):
            core = jax.checkpoint(core, prevent_cse=False)
        with jax.named_scope("gdn_core"):
            out = core(qkv, b, a, segment_ids)
            if flat:  # the heads side by side [B,T,H·Dv] from here (the saved residual too)
                out = out.reshape(batch, length, -1)
            out = checkpoint_name(out, "gdn_core_out")
        with jax.named_scope("gdn_out"):
            if flat:  # GatedRMSNorm.flat: no [B,T,H,Dv] view
                out = self.norm.flat(out, z, c.linear_num_value_heads)
            else:
                out = self.norm(out, z).reshape(batch, length, -1)
            out = jnp.where(valid, out, 0).astype(x.dtype)
            return self.out_proj(out, name="mixer_out")
