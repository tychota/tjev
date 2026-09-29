"""SwiGLU MLP with LoRA on gate, up and down projections."""

from __future__ import annotations

import jax
from flax import nnx

from tjev.model.lora import LinearFactory, grouped
from tjev.model.params import Weights


class MLP(nnx.Module):
    def __init__(self, w: Weights, linear: LinearFactory):
        self.gate_proj = linear("gate_proj", w["gate_proj.weight"])
        self.up_proj = linear("up_proj", w["up_proj.weight"])
        self.down_proj = linear("down_proj", w["down_proj.weight"])

    def __call__(self, x: jax.Array) -> jax.Array:
        with jax.named_scope("mlp"):
            gate, up = grouped(x, self.gate_proj, self.up_proj, names=("mlp_gate", "mlp_up"))
            return self.down_proj(jax.nn.silu(gate) * up)
