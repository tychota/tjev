"""Jitted train and eval steps over the split (graphdef, LoRA, frozen) model.

The model is split once; each step is a plain ``jax.jit`` of a pure function (the fast
NNX pattern: no graph traversal per call). The frozen base is an *argument*, never a
closure constant, and LoRA and optimizer state are donated.

Every step holds microbatches of a single sequence bucket, and a 4096-token step holds far
fewer answer slots than a 1024-token one. The gradient is therefore divided by the
*expected* slots per step (a constant measured once on the packer), not by the slots of
the step: every item weighs the same whatever the bucket it landed in.

Options (all off by default, and then the step is today's program): the ``loss.*``
regularisers (:mod:`tjev.train.objective`), the KL to the base needing one more forward per
microbatch with the adapters off (outside the gradient: no backward, no remat), and
the adapter-input dropout (``lora.dropout``), for which the step takes a ``key`` (one
fold per microbatch). The EMA of ``optim.ema_decay`` lives in the optimizer state, and so
does the running CogCalib threshold (``loss.cogcalib_quantile``): every microbatch of a step
gates on the threshold of the previous steps, then the step's quantile of its slot NLLs
(collected by the scan) is folded into it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from tjev.config import LossSpec, OptimSpec
from tjev.data.pack import Batch
from tjev.model import Qwen35
from tjev.model.lora import Linear
from tjev.train.health import device_metrics
from tjev.train.objective import loss_sums
from tjev.train.optim import cogcalib_state, cogcalib_update, with_cogcalib

TrainStep = Callable[..., tuple[nnx.State, Any, dict]]
EvalStep = Callable[[nnx.State, nnx.State, Batch], jax.Array]


class Split(NamedTuple):
    graphdef: nnx.GraphDef
    lora: nnx.State
    frozen: nnx.State


def split_model(model: Qwen35) -> Split:
    graphdef, lora, frozen = nnx.split(model, nnx.LoRAParam, ...)
    return Split(graphdef, lora, frozen)


def forward_logits(graphdef: nnx.GraphDef, lora: nnx.State, frozen: nnx.State, batch: Batch,
                   *, base: bool = False, dropout: tuple[jax.Array, float] | None = None):  # fmt: skip
    """Letter logits [R,S,K]. ``base``: the base model, every adapter switched off (the
    LoRA scaled to 0, without computing it: the kernel W is the base's whatever the init,
    an offset init's s·x(AB − A0B0) included; a ``gate_full`` kernel trains the base weight
    itself and is refused). ``dropout``: (key, rate)."""
    model = nnx.merge(graphdef, lora, frozen)
    if base:
        for _, m in nnx.iter_modules(model):
            if isinstance(m, Linear):
                if m.full:
                    raise ValueError("the base of a gate_full model is not recoverable")
                m.rank = 0  # trace time: this merged copy's projections are plain xW
    hidden = model.hidden(batch.tokens, batch.segment_ids, batch.positions, dropout)
    return model.label_logits(hidden, batch.slots, batch.label_ids)


def slot_table(table: np.ndarray | None, batch: Batch) -> jax.Array | None:
    """A per-source 0/1 table (tjev.train.objective.source_table) at each slot [R,S]."""
    return None if table is None else jnp.asarray(table)[batch.source_id]


def named(fn: Callable, name: str) -> Callable:
    """``fn`` under another ``__name__``: ``jax.jit`` names the HLO module ``jit_<__name__>``
    (``pxla.lower_sharding_computation``: ``wrap_name("jit", fun_name(fun))``) and the root
    of every op's ``op_name`` ``jit(<name>)``, which xprof shows as the program name."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper.__name__ = wrapper.__qualname__ = name
    return wrapper


def train_step_name(batch: Batch) -> str:
    """``train_step_T<length>_mb<microbatches>`` of a stacked [A, R, T] batch."""
    a, _, t = batch.tokens.shape
    return f"train_step_T{t}_mb{a}"


def eval_step_name(batch: Batch) -> str:
    """``eval_step_T<length>_b<rows>`` of an [R, T] batch."""
    r, t = batch.tokens.shape
    return f"eval_step_T{t}_b{r}"


class NamedJit:
    """``jax.jit(fn, **jit_kwargs)`` with one jitted copy per program name, the name made
    from the call's batch (``namer(batch)``, the argument at ``batch_arg``): each shape
    compiles to its own, readable program (``jit_train_step_T1024_mb32`` in xprof, the HLO
    dumps and the compile logs) instead of five ``jit_step(<hash>)``. One compile per shape
    as before; the outputs are the plain jit's (names are metadata only)."""

    def __init__(self, fn: Callable, namer: Callable[[Batch], str], batch_arg: int,
                 **jit_kwargs: Any):  # fmt: skip
        self.fn, self.namer, self.batch_arg, self.jit_kwargs = fn, namer, batch_arg, jit_kwargs
        self.jits: dict[str, Any] = {}

    def jitted(self, *args: Any) -> Any:
        name = self.namer(args[self.batch_arg])
        if name not in self.jits:
            self.jits[name] = jax.jit(named(self.fn, name), **self.jit_kwargs)
        return self.jits[name]

    def __call__(self, *args: Any) -> Any:
        return self.jitted(*args)(*args)

    def lower(self, *args: Any) -> Any:
        return self.jitted(*args).lower(*args)


def jit_train_step(step: TrainStep, **jit_kwargs: Any) -> NamedJit:
    """The jitted train step, one program per bucket shape (``train_step_T…_mb…``)."""
    return NamedJit(step, train_step_name, 3, **jit_kwargs)


def jit_eval_step(step: EvalStep, **jit_kwargs: Any) -> NamedJit:
    """The jitted eval step, one program per batch shape (``eval_step_T…_b…``)."""
    return NamedJit(step, eval_step_name, 2, **jit_kwargs)


def make_train_step(
    graphdef: nnx.GraphDef,
    tx: optax.GradientTransformation,
    *,
    brier_weight: float,
    slots_per_step: float,
    optim: OptimSpec | None = None,
    lr_schedule: optax.Schedule | None = None,
    health: bool = True,
    loss: LossSpec | None = None,
    smooth_table: np.ndarray | None = None,
    kl_table: np.ndarray | None = None,
    dropout: float = 0.0,
) -> TrainStep:
    """``batch`` leaves are [A, R, ...]: A microbatches, accumulated with ``lax.scan``.
    ``health``: also the ``health/*`` diagnostics (tjev.train.health; ``optim`` adds the
    clipping and AdamW statistics, ``lr_schedule`` the learning rate actually used).
    ``loss``: the regularisers; ``smooth_table`` / ``kl_table``: their per-source 0/1
    selection (None: every source). ``dropout`` > 0: the step is
    ``step(lora, opt_state, frozen, batch, key)``."""
    spec = loss or LossSpec()
    kl = bool(spec.kl_base_weight)
    adaptive = spec.cogcalib_adaptive

    def micro_loss(lora, frozen, batch, key=None, base_logits=None, threshold=None):
        drop = (key, dropout) if dropout else None
        logits = forward_logits(graphdef, lora, frozen, batch, dropout=drop)
        with jax.named_scope("loss"):
            return loss_sums(logits, batch, brier_weight=brier_weight, loss=loss,
                             smooth_on=slot_table(smooth_table, batch),
                             kl_on=slot_table(kl_table, batch), base_logits=base_logits,
                             cogcalib_threshold=threshold, slot_nll=adaptive)  # fmt: skip

    grad_fn = jax.value_and_grad(micro_loss, has_aux=True)

    def step(lora, opt_state, frozen, batch, key=None):
        if dropout and key is None:
            raise ValueError("lora.dropout: the train step needs a key")
        zeros_g = jax.tree.map(jnp.zeros_like, lora)
        first = jax.tree.map(lambda x: x[0], batch)
        base0 = (jax.eval_shape(lambda: forward_logits(graphdef, lora, frozen, first, base=True))
                 if kl else None)  # fmt: skip
        calib = cogcalib_state(opt_state) if adaptive else None
        if adaptive and calib is None:
            raise ValueError("loss.cogcalib_quantile: the optimizer has no CogCalibState "
                             "(make_optimizer(..., loss=cfg.loss))")  # fmt: skip
        # the running threshold of the previous steps (none yet: nothing is gated)
        threshold = (jnp.where(calib.count > 0, calib.threshold, -jnp.inf)
                     if adaptive else None)  # fmt: skip
        shapes = jax.eval_shape(lambda b: micro_loss(lora, frozen, first, key, b, threshold)[1],
                                base0)  # fmt: skip
        shapes.pop("slot_nll", None)
        zeros_s = jax.tree.map(jnp.zeros_like, shapes)

        def accumulate(carry, xs):
            g_sum, s_sum = carry
            microbatch, i = xs
            base_logits = None
            if kl:  # the frozen model's letters: a forward only, never differentiated
                with jax.named_scope("kl_base"):
                    base_logits = jax.lax.stop_gradient(
                        forward_logits(graphdef, lora, frozen, microbatch, base=True)
                    )
            mkey = None if key is None else jax.random.fold_in(key, i)
            (_, sums), grads = grad_fn(lora, frozen, microbatch, mkey, base_logits, threshold)
            slot_nll = sums.pop("slot_nll", None)  # [R,S] per slot, not a sum
            return ((jax.tree.map(jnp.add, g_sum, grads), jax.tree.map(jnp.add, s_sum, sums)),
                    slot_nll)  # fmt: skip

        n_micro = jax.tree.leaves(batch)[0].shape[0]
        with jax.named_scope("grad_accum"):
            (grads, sums), slot_nll = jax.lax.scan(accumulate, (zeros_g, zeros_s),
                                                   (batch, jnp.arange(n_micro)))  # fmt: skip
        denom = jnp.maximum(sums["weight"], 1.0)  # per-slot means for the metrics
        grads = jax.tree.map(lambda g: g / slots_per_step, grads)
        with jax.named_scope("optimizer"):
            grad_norm = optax.tree.norm(grads)
            updates, new_opt = tx.update(grads, opt_state, lora)
            new_lora = optax.apply_updates(lora, updates)
        if adaptive:  # the step's quantile of its slot NLLs (empty slots are NaN)
            quantile = jnp.nanquantile(slot_nll, spec.cogcalib_quantile)
            new_opt = with_cogcalib(new_opt, cogcalib_update(calib, quantile, spec.cogcalib_ema))
        # Reject a non-finite update on device: parameters and moments stay unchanged.
        ok = jnp.isfinite(grad_norm) & jnp.isfinite(sums["loss"])
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
        if "kl_base" in sums:
            metrics["kl_base"] = sums["kl_base"] / denom
        if "cogcalib_on" in sums:  # the fraction of slots gated and the penalty per slot
            metrics["cogcalib_frac"] = sums["cogcalib_on"] / denom
            metrics["cogcalib_term"] = sums["cogcalib"] / denom
            metrics["cogcalib_threshold"] = (  # 0 until the first quantile is in
                jnp.where(jnp.isfinite(threshold), threshold, 0.0)
                if adaptive else jnp.asarray(spec.cogcalib_threshold, jnp.float32)
            )  # fmt: skip
        if "teacher_on" in sums:  # the fraction of slots trained on the teacher mixture
            metrics["teacher_frac"] = sums["teacher_on"] / denom
        if health:
            with jax.named_scope("health"):
                metrics["health"] = device_metrics(
                    grads, updates, lora, new_lora, opt_state, new_opt, grad_norm=grad_norm,
                    optim=optim, lr_schedule=lr_schedule,
                )  # fmt: skip
        new_lora = jax.tree.map(lambda n, o: jnp.where(ok, n, o), new_lora, lora)
        new_opt = jax.tree.map(lambda n, o: jnp.where(ok, n, o), new_opt, opt_state)
        return new_lora, new_opt, metrics

    return step


def make_eval_step(graphdef: nnx.GraphDef) -> EvalStep:
    def step(lora, frozen, batch):
        return forward_logits(graphdef, lora, frozen, batch)

    return step
