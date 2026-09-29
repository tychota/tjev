"""Ahead-of-time compile of the train step per bucket: memory before spending chip hours.

Builds abstract inputs only (no weights loaded, no data read) and reports XLA's memory
analysis per sequence bucket (MaxText ``train_compile`` idea): ``tjev compile-check``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tjev.config import RunConfig
from tjev.data.pack import Batch, rows_for
from tjev.model import Qwen35, expected_shapes, read_config, stack_layers
from tjev.sharding import configure_runtime, install_mesh, make_mesh
from tjev.train.optim import make_optimizer
from tjev.train.step import make_train_step, split_model

# A step needing more than this share of a device's memory leaves no room for fragmentation
# and the other buckets' executables: lower train.microbatch_tokens or the largest bucket.
SAFE_FRACTION = 0.9


def abstract_model(cfg: RunConfig) -> Qwen35:
    config = read_config(cfg.model.path)

    def build() -> Qwen35:
        tensors = {k: jnp.zeros(v, jnp.float32) for k, v in expected_shapes(config).items()}
        return Qwen35(
            stack_layers(tensors, config),
            config,
            cfg.compute,
            cfg.lora,
            dtype=getattr(jnp, cfg.model.dtype),
            rngs=nnx.Rngs(0),
        )

    return nnx.eval_shape(build)


def abstract_batch(accumulation: int, rows: int, length: int, slots: int, labels: int) -> Batch:
    def shape(*s: int, dtype: Any = np.int32) -> Any:
        return jax.ShapeDtypeStruct((accumulation, rows, *s), dtype)

    return Batch(
        tokens=shape(length),
        segment_ids=shape(length),
        positions=shape(length),
        slots=shape(slots),
        label_ids=shape(slots, labels),
        label_mask=shape(slots, labels, dtype=np.bool_),
        target=shape(slots, labels, dtype=np.float32),
        weight=shape(slots, dtype=np.float32),
        type_id=shape(slots),
        source_id=shape(slots),
        lang_id=shape(slots),
        family_id=shape(slots),
        index=shape(slots),
    )


def compile_check(cfg: RunConfig) -> dict:
    configure_runtime(cfg.compute)
    mesh = make_mesh(cfg.mesh)
    install_mesh(cfg.compute, mesh)
    graphdef, lora, frozen = split_model(abstract_model(cfg))
    tx = make_optimizer(cfg.optim, cfg.train.steps)
    opt_state = jax.eval_shape(tx.init, lora)
    step = make_train_step(graphdef, tx, brier_weight=cfg.train.brier_weight, slots_per_step=1.0)
    per_step = cfg.train.microbatch_tokens * mesh.size
    accumulation = max(1, cfg.train.tokens_per_step // per_step)
    report: dict[str, Any] = {"devices": mesh.size, "accumulation": accumulation, "buckets": {}}
    device_bytes = (jax.devices()[0].memory_stats() or {}).get("bytes_limit")
    for length in cfg.train.seq_buckets:
        rows = rows_for(length, per_step)
        batch = abstract_batch(
            accumulation, rows, length, cfg.train.max_segments, cfg.train.max_labels
        )
        compiled = (
            jax.jit(step, donate_argnums=(0, 1)).lower(lora, opt_state, frozen, batch).compile()
        )
        mem = compiled.memory_analysis()
        entry: dict[str, Any] = {"rows_per_microbatch": rows}
        if mem is not None:
            need = mem.argument_size_in_bytes + mem.temp_size_in_bytes + mem.output_size_in_bytes
            entry |= {
                "argument_gb": mem.argument_size_in_bytes / 1e9,
                "output_gb": mem.output_size_in_bytes / 1e9,
                "temp_gb": mem.temp_size_in_bytes / 1e9,
                "total_gb": need / 1e9,
            }
            if device_bytes:
                entry["device_fraction"] = round(need / device_bytes, 3)
                if need > SAFE_FRACTION * device_bytes:
                    entry["warning"] = (
                        f"needs {need / 2**30:.1f} of {device_bytes / 2**30:.1f} GiB: lower "
                        "train.microbatch_tokens or the largest bucket"
                    )
        report["buckets"][length] = entry
    return report
