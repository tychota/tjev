"""Jitted train and eval steps over the split (graphdef, LoRA, frozen) model.

The model is split once; each step is a plain ``jax.jit`` of a pure function (the fast
NNX pattern: no graph traversal per call). The frozen base is an *argument*, never a
closure constant, and LoRA and optimizer state are donated.

Every step holds microbatches of a single sequence bucket, and a 4096-token step holds far
fewer answer slots than a 1024-token one. The gradient is therefore divided by the
*expected* slots per step (a constant measured once on the packer), not by the slots of
the step: every item weighs the same whatever the bucket it landed in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from tjev.data.pack import Batch
from tjev.model import Qwen35
from tjev.train.objective import loss_sums

TrainStep = Callable[[nnx.State, Any, nnx.State, Batch], tuple[nnx.State, Any, dict]]
EvalStep = Callable[[nnx.State, nnx.State, Batch], jax.Array]


class Split(NamedTuple):
    graphdef: nnx.GraphDef
    lora: nnx.State
    frozen: nnx.State


def split_model(model: Qwen35) -> Split:
    graphdef, lora, frozen = nnx.split(model, nnx.LoRAParam, ...)
    return Split(graphdef, lora, frozen)


def forward_logits(graphdef: nnx.GraphDef, lora: nnx.State, frozen: nnx.State, batch: Batch):
    model = nnx.merge(graphdef, lora, frozen)
    hidden = model.hidden(batch.tokens, batch.segment_ids, batch.positions)
    return model.label_logits(hidden, batch.slots, batch.label_ids)


def make_train_step(
    graphdef: nnx.GraphDef,
    tx: optax.GradientTransformation,
    *,
    brier_weight: float,
    slots_per_step: float,
) -> TrainStep:
    """``batch`` leaves are [A, R, ...]: A microbatches, accumulated with ``lax.scan``."""

    def micro_loss(lora, frozen, batch):
        logits = forward_logits(graphdef, lora, frozen, batch)
        return loss_sums(logits, batch, brier_weight=brier_weight)

    grad_fn = jax.value_and_grad(micro_loss, has_aux=True)

    def step(lora, opt_state, frozen, batch):
        zeros_g = jax.tree.map(jnp.zeros_like, lora)
        first = jax.tree.map(lambda x: x[0], batch)
        zeros_s = jax.tree.map(
            jnp.zeros_like, jax.eval_shape(lambda: micro_loss(lora, frozen, first)[1])
        )

        def accumulate(carry, microbatch):
            g_sum, s_sum = carry
            (_, sums), grads = grad_fn(lora, frozen, microbatch)
            return (jax.tree.map(jnp.add, g_sum, grads), jax.tree.map(jnp.add, s_sum, sums)), None

        (grads, sums), _ = jax.lax.scan(accumulate, (zeros_g, zeros_s), batch)
        denom = jnp.maximum(sums["weight"], 1.0)  # per-slot means for the metrics
        grads = jax.tree.map(lambda g: g / slots_per_step, grads)
        grad_norm = optax.tree.norm(grads)
        updates, new_opt = tx.update(grads, opt_state, lora)
        new_lora = optax.apply_updates(lora, updates)
        ok = jnp.isfinite(grad_norm) & jnp.isfinite(sums["loss"])
        # Reject a non-finite update on device: parameters and moments stay unchanged.
        new_lora = jax.tree.map(lambda n, o: jnp.where(ok, n, o), new_lora, lora)
        new_opt = jax.tree.map(lambda n, o: jnp.where(ok, n, o), new_opt, opt_state)
        metrics = {
            "loss": sums["loss"] / denom,
            "nll": sums["nll"] / denom,
            "brier": sums["brier"] / denom,
            "accuracy": sums["correct"] / denom,
            "slots": sums["weight"],
            "tokens": sums["tokens"],
            "grad_norm": grad_norm,
            "update_rejected": 1.0 - ok.astype(jnp.float32),
        }
        return new_lora, new_opt, metrics

    return step


def make_eval_step(graphdef: nnx.GraphDef) -> EvalStep:
    def step(lora, frozen, batch):
        return forward_logits(graphdef, lora, frozen, batch)

    return step
