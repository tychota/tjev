"""The Qwen3.5 text decoder: architecture config, HF weight import, NNX modules."""

from .config import ModelConfig
from .qwen35 import REMAT_POLICIES, Qwen35, build_model
from .weights import expected_shapes, load_hf, read_config, snapshot_identity, stack_layers

__all__ = [
    "REMAT_POLICIES",
    "ModelConfig",
    "Qwen35",
    "build_model",
    "expected_shapes",
    "load_hf",
    "read_config",
    "snapshot_identity",
    "stack_layers",
]
