"""Analytic FLOPs for MFU (MaxText-style ``perf/tflops_per_device``).

"Model FLOPs" count useful work only: matmuls on real tokens, causal attention within
segments, the GDN delta rule, and — for LoRA — forward + activation-gradient passes
(no weight gradients for the frozen base). Rematerialisation is not counted.
"""

from __future__ import annotations

import os

import numpy as np

from tjev.model import ModelConfig

# Dense bf16 peak per chip (FLOP/s) with fp32 accumulation. Override with TJEV_PEAK_FLOPS
# (e.g. on a GPU, where the XLA paths run too).
PEAK_FLOPS = {
    "TPU v5 lite": 197e12,
    "TPU v5e": 197e12,
    "TPU v5p": 459e12,
    "TPU v6 lite": 918e12,
    "TPU v6e": 918e12,
}


def peak_flops(device_kind: str) -> float | None:
    if os.environ.get("TJEV_PEAK_FLOPS"):
        return float(os.environ["TJEV_PEAK_FLOPS"])
    return PEAK_FLOPS.get(device_kind)


def matmul_params(c: ModelConfig) -> dict[str, int]:
    d = c.hidden_size
    gdn = (
        d * (c.conv_dim + c.linear_value_dim + 2 * c.linear_num_value_heads)
        + c.linear_value_dim * d
    )
    attn = (
        d * (2 * c.num_heads * c.head_dim + 2 * c.num_kv_heads * c.head_dim)
        + c.num_heads * c.head_dim * d
    )
    mlp = 3 * d * c.intermediate_size
    n_attn = c.num_super_blocks
    n_gdn = c.num_layers - n_attn
    return {"gdn": n_gdn * gdn, "attention": n_attn * attn, "mlp": c.num_layers * mlp}


def forward_flops(
    c: ModelConfig, segment_lengths: np.ndarray, chunk: int = 64, label_count: int = 0
) -> float:
    """Forward FLOPs for a set of segments (lengths), plus K-way label readout."""
    tokens = float(np.sum(segment_lengths))
    linear = 2.0 * sum(matmul_params(c).values()) * tokens
    # Causal attention: QKᵀ and AV, 2 FLOPs per MAC, n(n+1)/2 pairs per segment.
    pairs = float(np.sum(segment_lengths * (segment_lengths + 1) / 2))
    attn = c.num_super_blocks * 2 * 2 * pairs * c.num_heads * c.head_dim
    h, dk, dv = c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
    gdn = (c.num_layers - c.num_super_blocks) * tokens * 2 * h * (chunk * (dk + dv) + 3 * dk * dv)
    readout = 2.0 * c.hidden_size * label_count
    return linear + attn + gdn + readout


def train_flops(
    c: ModelConfig, segment_lengths: np.ndarray, chunk: int = 64, labels: int = 0
) -> float:
    """LoRA training: forward + activation-gradient pass (≈ 2 × forward)."""
    return 2.0 * forward_flops(c, segment_lengths, chunk, labels)


def segment_lengths(segment_ids: np.ndarray) -> np.ndarray:
    """Lengths of all segments in a (possibly stacked) [.., T] segment-id array."""
    rows = segment_ids.reshape(-1, segment_ids.shape[-1])
    out = []
    for row in rows:
        _, counts = np.unique(row[row > 0], return_counts=True)
        out.extend(counts.tolist())
    return np.asarray(out, np.float64)
