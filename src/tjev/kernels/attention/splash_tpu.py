"""Causal, segment-masked GQA attention on TPU: JAX's Pallas splash attention kernels.

``compute.attention=splash``. Splash computes softmax(q kᵀ) v with an online softmax, never
materialising [T, T], and skips key blocks that the causal and segment masks rule out;
its backward (dq, dkv kernels) comes with it. GQA runs one MQA kernel per KV head, over
that head's group of query heads (``make_splash_mqa_single_device``, vmapped over batch
and KV heads), so K/V are never repeated. Qwen3.5's output gate and q/k norms stay outside.

Shapes: q [B,T,H,D], k/v [B,T,KV,D], segment_ids [B,T] (0 = padding, which attends only
to padding: no empty softmax rows). T is padded to a multiple of 128 (the lane width).
On a non-TPU backend the kernels run in interpret mode (tests).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as masks

# MaxText's default block (sa_block_* = 512); smaller rows use the whole row.
BLOCK = 512
LANES = 128


def _block(length: int) -> int:
    for size in (BLOCK, 256, LANES):
        if length % size == 0:
            return size
    raise ValueError(f"length {length} is not a multiple of {LANES}")


@functools.lru_cache(maxsize=32)
def _kernel(length: int, group: int, interpret: bool, mesh: str = ""):
    """mesh: the abstract mesh at trace time (its repr), part of the cache key: the
    kernel's mask tables are built under it and cannot be reused under another one."""
    block = _block(length)
    sizes = splash.BlockSizes(
        block_q=block,
        block_kv=block,
        block_kv_compute=block,
        block_q_dkv=block,
        block_kv_dkv=block,
        block_kv_dkv_compute=block,
        block_q_dq=block,
        block_kv_dq=block,
    )
    mask = masks.MultiHeadMask([masks.CausalMask((length, length)) for _ in range(group)])
    # Built eagerly (its mask tables are concrete arrays) so the cached kernel can be reused
    # across traces. "attn_context": the name remat policies (``minimal``) save.
    with jax.ensure_compile_time_eval():
        return splash.make_splash_mqa_single_device(
            mask, block_sizes=sizes, residual_checkpoint_name="attn_context", interpret=interpret
        )


def splash_attention(q, k, v, segment_ids, scale: float | None = None):
    batch, length, heads, dim = q.shape
    kv_heads = k.shape[2]
    group = heads // kv_heads
    scale = dim**-0.5 if scale is None else scale
    pad = (-length) % LANES
    if pad:
        width = lambda x: ((0, 0), (0, pad)) + ((0, 0),) * (x.ndim - 2)  # noqa: E731
        q, k, v, segment_ids = (jnp.pad(x, width(x)) for x in (q, k, v, segment_ids))
    total = length + pad
    kernel = _kernel(
        total, group, jax.default_backend() != "tpu", repr(jax.sharding.get_abstract_mesh())
    )
    # [B, KV, G, T, D] queries (head h = kv·G + g, the GQA grouping of jax.nn attention)
    qh = (q * jnp.asarray(scale, q.dtype)).reshape(batch, total, kv_heads, group, dim)
    qh = jnp.transpose(qh, (0, 2, 3, 1, 4))
    kh, vh = (jnp.transpose(x, (0, 2, 1, 3)) for x in (k, v))  # [B, KV, T, D]

    def one_row(q_row, k_row, v_row, seg):
        ids = splash.SegmentIds(q=seg, kv=seg)
        return jax.vmap(lambda qg, kg, vg: kernel(qg, kg, vg, segment_ids=ids))(q_row, k_row, v_row)

    out = jax.vmap(one_row)(qh, kh, vh, segment_ids.astype(jnp.int32))  # [B, KV, G, T, D]
    out = jnp.transpose(out, (0, 3, 1, 2, 4)).reshape(batch, total, heads, dim)
    return out[:, :length]
