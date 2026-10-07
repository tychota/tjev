"""CPU estimate of a train step's device memory per (size, bucket, remat policy).

    python scripts/remat_memory.py CONFIGS 9B:2048:minimal_states 9B:4096:core_states …

``CONFIGS/<size>/config.json``: the Qwen3.5 configs (no weights: every input is abstract).
Each spec is compiled on the CPU like ``tjev compile-check`` (the v6e1 configuration: mega2
chunk 128 blocked, splash, rank 16 without the gate adapters, merged LoRA ≤ 4B, the v6e1
microbatch per size, 65,536 tokens per step, one bucket) and reports XLA's memory analysis,
with the saved residuals of one super-block's layer checkpoints from the jaxpr (all
super-blocks, by checkpoint name; activations only: 2·(R(T) − R(T/2))). One JSON line per
spec. The CPU's arguments equal the TPU's; its temporaries are within −2 / +18% of the
TPU's at the bench points (docs/training.md, "Remat per bucket").
"""

from __future__ import annotations

import json
import sys
import time

import jax
import jax.numpy as jnp
from flax import nnx
from jax._src.ad_checkpoint import saved_residuals

from tjev.config import load_config
from tjev.train.compile_check import abstract_model, compile_check

MICROBATCH = {"0.8B": 8192, "2B": 8192, "4B": 4096, "9B": 2048}  # campaign.plan v6e1
TARGETS = "[q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj]"


def config(root: str, size: str, bucket: int, remat: str):
    overrides = [
        f"model.path={root}/{size}", f"compute.remat={remat}",
        "compute.gdn_impl=pallas_tpu_mega2", "compute.gdn_chunk=128",
        "compute.gdn_inverse=blocked", "lora.rank=16", f"lora.targets={TARGETS}",
        f"train.microbatch_tokens={MICROBATCH[size]}", f"train.seq_buckets=[{bucket}]",
        *(["compute.lora_impl=merged"] if size != "9B" else []),
    ]  # fmt: skip
    return load_config("tpu-v6e", overrides=overrides)


def residuals(cfg, rows: int, length: int) -> dict[str, int]:
    """Bytes of the residuals one super-block saves for its backward, by checkpoint name."""
    model = abstract_model(cfg)
    graphdef, lora, frozen = nnx.split(model.blocks, nnx.LoRAParam, ...)

    def first(state):  # the stacked super-block weights → one super-block's
        return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), state)

    x = jax.ShapeDtypeStruct((rows, length, model.config.hidden_size), jnp.bfloat16)
    seg = jnp.ones((rows, length), jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(length), (rows, length))

    def loss(lo, fr, h):
        block = nnx.merge(graphdef, lo, fr)
        return jnp.sum(block(h, seg, pos).astype(jnp.float32) ** 2)

    out: dict[str, int] = {}
    for aval, where in saved_residuals(loss, first(lora), first(frozen), x):
        if hasattr(aval, "shape"):
            name = where.split("named '")[1].split("'")[0] if "named '" in where else "other"
            out[name] = out.get(name, 0) + aval.size * aval.dtype.itemsize
    return out


def main() -> None:
    root, *specs = sys.argv[1:]
    for spec in specs:
        size, bucket_s, remat = spec.split(":")
        bucket = int(bucket_s)
        cfg = config(root, size, bucket, remat)
        t0 = time.time()
        rows = max(1, MICROBATCH[size] // bucket)
        full, half = residuals(cfg, rows, bucket), residuals(cfg, rows, bucket // 2)
        blocks = abstract_model(cfg).config.num_super_blocks
        by_name = {k: 2 * (v - half.get(k, 0)) * blocks for k, v in full.items()}
        report = compile_check(cfg)["buckets"][bucket]
        print(json.dumps({
            "size": size, "bucket": bucket, "remat": report["remat"], "rows": rows,
            "residual_gb": sum(by_name.values()) / 1e9,
            "residual_by_name_gb": {k: round(v / 1e9, 3) for k, v in
                                    sorted(by_name.items(), key=lambda kv: -kv[1])},
            **{k: report.get(k) for k in ("argument_gb", "temp_gb", "output_gb", "total_gb")},
            "seconds": round(time.time() - t0, 1),
        }), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
