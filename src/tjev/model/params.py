"""Parameter kinds and weight dictionaries shared by the modules of the decoder.

Pretrained weights are :class:`Frozen` (a non-``Param`` variable, so the optimizer never
sees them); LoRA adapters are ``nnx.LoRAParam``. Every module accepts arrays with extra
*leading* axes (layers stacked over super-blocks): the decoder scans over those axes, so
inside the forward pass each module sees single-layer arrays.
"""

from __future__ import annotations

import jax
from flax import nnx

# HF text-model weight names (prefix stripped) → arrays, per layer or stacked
Weights = dict[str, jax.Array]


class Frozen(nnx.Variable):
    """Pretrained weight: never differentiated, never touched by the optimizer."""


def sub_weights(w: Weights, prefix: str) -> Weights:
    return {k[len(prefix) :]: v for k, v in w.items() if k.startswith(prefix)}
