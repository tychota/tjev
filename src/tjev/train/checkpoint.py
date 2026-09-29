"""Checkpoints of adapters + optimizer state + data stream state (Orbax, async).

Only trainable state is saved (the frozen base is identified by its snapshot hash).
Pytrees are saved as flat leaf lists and restored against a freshly built template,
so NNX/optax container types never need to be serialisable.
"""

from __future__ import annotations

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
    """Keeps the ``keep`` latest steps plus one protected step (the best eval step).

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

    def save(self, step: int, lora, opt_state, meta: dict, keep: int | None = None) -> None:
        """Async save of ``step``; older steps beyond the latest ``self.keep`` are deleted,
        except ``keep`` (the protected step)."""
        self.manager.wait_until_finished()  # the previous save, before pruning around it
        older = [s for s in self.manager.all_steps() if s != step]
        for s in sorted(older)[: max(0, len(older) - (self.keep - 1))]:
            if s != keep:
                self.manager.delete(s)
        self.manager.save(
            step,
            args=ocp.args.Composite(
                lora=ocp.args.StandardSave(_flat(lora)),
                opt=ocp.args.StandardSave(_flat(opt_state)),
                meta=ocp.args.JsonSave(meta),
            ),
        )

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

    def wait(self) -> None:
        self.manager.wait_until_finished()

    def close(self) -> None:
        self.manager.close()
