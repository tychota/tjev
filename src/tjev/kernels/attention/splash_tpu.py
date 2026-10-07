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


def _block(length: int, preferred: int = BLOCK) -> int:
    """The largest of ``preferred``, 512, 256, 128 (≤ preferred) that divides the row."""
    for size in (preferred, 512, 256, LANES):
        if size <= preferred and length % size == 0:
            return size
    raise ValueError(f"length {length} is not a multiple of {LANES}")


def block_sizes(length: int, block: int = BLOCK, compute: int = 0, fused_bwd: bool = False):
    """Splash tile sizes: q and kv blocks of ``block`` rows (fewer when the row is not a
    multiple), the kv compute sub-block ``compute`` (0: the whole block), and the backward
    as separate dq + dkv kernels or, ``fused_bwd``, one dkv kernel that also emits dq
    (summed in XLA over the kv blocks: no second QKᵀ / softmax recompute)."""
    size = _block(length, block)
    inner = min(compute or size, size)
    if size % inner:
        raise ValueError(f"splash compute block {inner} does not divide the block {size}")
    dq = None if fused_bwd else size
    return splash.BlockSizes(
        block_q=size,
        block_kv=size,
        block_kv_compute=inner,
        block_q_dkv=size,
        block_kv_dkv=size,
        block_kv_dkv_compute=inner,
        block_q_dq=dq,
        block_kv_dq=dq,
        use_fused_bwd_kernel=fused_bwd,
    )


@functools.lru_cache(maxsize=32)
def _kernel(
    length: int, group: int, interpret: bool, mesh: str = "", block: int = BLOCK,
    compute: int = 0, fused_bwd: bool = False,
):  # fmt: skip
    """mesh: the abstract mesh at trace time (its repr), part of the cache key: the
    kernel's mask tables are built under it and cannot be reused under another one."""
    sizes = block_sizes(length, block, compute, fused_bwd)
    mask = masks.MultiHeadMask([masks.CausalMask((length, length)) for _ in range(group)])
    # Built eagerly (its mask tables are concrete arrays) so the cached kernel can be reused
    # across traces. "attn_context": the name remat policies (``minimal``) save.
    with jax.ensure_compile_time_eval():
        return splash.make_splash_mqa_single_device(
            mask, block_sizes=sizes, residual_checkpoint_name="attn_context", interpret=interpret
        )


def splash_attention(
    q, k, v, segment_ids, scale: float | None = None, *, block: int = BLOCK, compute: int = 0,
    fused_bwd: bool = False,
):  # fmt: skip
    """``block``, ``compute``, ``fused_bwd``: the tile sizes and backward form
    (:func:`block_sizes`); ``compute.attention_splash_*``."""
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
        total, group, jax.default_backend() != "tpu", repr(jax.sharding.get_abstract_mesh()),
        block, compute, fused_bwd,
    )  # fmt: skip
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
