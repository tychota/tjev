"""Sequence-mixing ops of the Qwen3.5 decoder, with XLA references and Pallas TPU kernels.

One package per op (ejkernel-style): ``xla`` holds the pure-JAX reference that every other
implementation is tested against; ``*_tpu`` modules hold the Pallas TPU kernels, which run
in interpret mode on other backends (tests) and under ``shard_map`` on a mesh
(:mod:`.sharding`). The model picks an implementation per op from ``ComputeSpec``.
"""

from .attention import attention
from .causal_conv1d import segmented_causal_conv1d
from .gated_delta_rule import gated_delta_rule, l2norm

__all__ = ["attention", "gated_delta_rule", "l2norm", "segmented_causal_conv1d"]
