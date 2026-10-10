"""Checkpoints of adapters + optimizer state + data stream state (Orbax, async).

Only trainable state is saved (the frozen base is identified by its snapshot hash).
Pytrees are saved as flat leaf lists and restored against a freshly built template,
so NNX/optax container types never need to be serialisable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import jax
import orbax.checkpoint as ocp


def _flat(tree) -> dict[str, jax.Array]:
    return {f"{i:05d}": leaf for i, leaf in enumerate(jax.tree.leaves(tree))}


def _unflat(template, flat: dict):
    leaves, treedef = jax.tree.flatten(template)
    restored = [flat[f"{i:05d}"] for i in range(len(leaves))]
    for got, want in zip(restored, leaves, strict=True):
        if tuple(got.shape) != tuple(want.shape):
            raise ValueError(f"checkpoint leaf shape {got.shape} != {want.shape}")
    return jax.tree.unflatten(treedef, restored)


class Checkpoints:
    """Keeps the ``keep`` latest planned steps plus protected ones (the best eval step, the
    planned branch points of ``train.keep_steps``); a resume-only save (time-based,
    SIGTERM / SIGINT, the session deadline) is deleted as soon as a newer save exists.

    Retention is done here, not by Orbax's ``max_to_keep``, which would rotate the best
    step away once ``keep`` newer checkpoints exist."""

    def __init__(self, directory: str | Path, keep: int = 3):
        self.directory = Path(directory).resolve()
        self.keep = keep
        self.manager = ocp.CheckpointManager(
            self.directory, options=ocp.CheckpointManagerOptions(max_to_keep=None, create=True)
        )

    def latest(self) -> int | None:
        return self.manager.latest_step()

    def all_steps(self) -> list[int]:
        return sorted(self.manager.all_steps())

    def prune_plan(self, step: int, keep: int | None = None, protect: Iterable[int] = (),
                   resume_only: Callable[[int], bool] | None = None) -> list[int]:  # fmt: skip
        """The steps a save of ``step`` deletes: every older resume-only step, and the
        planned ones beyond the latest ``self.keep`` (counting ``step``), never ``keep`` or
        ``protect``."""
        safe = set(protect) | ({keep} if keep is not None else set())
        older = [s for s in self.all_steps() if s != step]
        temporary = [s for s in older if resume_only is not None and resume_only(s) and s < step]
        planned = [s for s in older if s not in temporary]
        doomed = temporary + sorted(planned)[: max(0, len(planned) - (self.keep - 1))]
        return sorted(s for s in doomed if s not in safe)

    def save(self, step: int, lora, opt_state, meta: dict, keep: int | None = None,
             protect: Iterable[int] = (), resume_only: Callable[[int], bool] | None = None,
             ema=None) -> None:  # fmt: skip
        """Async save of ``step`` after deleting :meth:`prune_plan`'s steps (``keep``: the
        best step; ``protect``: more steps never deleted; ``resume_only(s)``: whether step
        s was a resume-only save). ``ema``: the EMA adapters (``optim.ema_decay``), saved as
        their own item ``ema/`` with the adapters' tree, so a checkpoint backed up without
        its optimizer state still has them (the EMA inside ``opt/`` is the one resumed)."""
        self.manager.wait_until_finished()  # the previous save, before pruning around it
        for s in self.prune_plan(step, keep, protect, resume_only):
            self.manager.delete(s)
        items = {
            "lora": ocp.args.StandardSave(_flat(lora)),
            "opt": ocp.args.StandardSave(_flat(opt_state)),
            "meta": ocp.args.JsonSave(meta),
        }
        if ema is not None:
            items["ema"] = ocp.args.StandardSave(_flat(ema))
        self.manager.save(step, args=ocp.args.Composite(**items))

    def save_adapters(self, step: int, lora, meta: dict) -> None:
        """The adapters and meta of ``step`` only (no optimizer state; e.g. a weight-space
        soup, :mod:`tjev.eval.soup`): what :meth:`restore_lora` reads back."""
        self.manager.save(step, args=ocp.args.Composite(
            lora=ocp.args.StandardSave(_flat(lora)), meta=ocp.args.JsonSave(meta)))  # fmt: skip
        self.manager.wait_until_finished()

    def restore(self, step: int, lora_template, opt_template):
        def abstract(tree):
            return {
                k: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=getattr(v, "sharding", None))
                for k, v in _flat(tree).items()
            }

        out = self.manager.restore(
            step,
            args=ocp.args.Composite(
                lora=ocp.args.StandardRestore(abstract(lora_template)),
                opt=ocp.args.StandardRestore(abstract(opt_template)),
                meta=ocp.args.JsonRestore(),
            ),
        )
        return _unflat(lora_template, out["lora"]), _unflat(opt_template, out["opt"]), out["meta"]

    def _restore_item(self, step: int, item: str, template):
        flat = _flat(template)
        out = self.manager.restore(
            step,
            args=ocp.args.Composite(**{
                item: ocp.args.StandardRestore({
                    k: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=getattr(v, "sharding", None))
                    for k, v in flat.items()
                }),
                "meta": ocp.args.JsonRestore(),
            }),
        )  # fmt: skip
        return _unflat(template, out[item]), out["meta"]

    def restore_lora(self, step: int, lora_template):
        """The adapters and meta of ``step`` only: an evaluation needs no optimizer state, and
        a backed-up checkpoint other than a run's last one has none."""
        return self._restore_item(step, "lora", lora_template)

    def has_ema(self, step: int) -> bool:
        """Whether ``step`` saved the EMA adapters as their own item."""
        return (self.directory / str(step) / "ema").is_dir()

    def restore_ema(self, step: int, lora_template):
        """The EMA adapters of ``step`` (the ``ema/`` item; the adapters' tree and dtype), or
        None when the checkpoint has none (a run without ``optim.ema_decay``, or one saved
        before the item existed: its EMA is in the optimizer state, :func:`ema_params`)."""
        if not self.has_ema(step):
            return None
        return self._restore_item(step, "ema", lora_template)[0]

    def wait(self) -> None:
        self.manager.wait_until_finished()

    def close(self) -> None:
        self.manager.close()
