"""Time the fused RMSNorm → projection kernel against XLA on the decoder's shapes (TPU VM).

    python experiments/tpu/bench_norm_matmul.py [--tokens 8192]

For each model size and projection (GDN qkv, MLP gate, attention q): forward, and forward +
backward, XLA (``reference`` under jit) vs the Pallas kernel, in ms (median of 20). Worth
adopting only if the forward gain survives in a full train step (``tjev campaign bench``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "norm_matmul", Path(__file__).with_name("norm_matmul.py")
)
if _spec is None or _spec.loader is None:
    raise ImportError("experiments/tpu/norm_matmul.py not found")
norm_matmul = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(norm_matmul)

SHAPES = {  # (hidden K, output N) of the projections that read a normalised input
    "0.8B": {"gdn_qkv": (1024, 6144), "mlp_gate": (1024, 3584), "attn_q": (1024, 4096)},
    "2B": {"gdn_qkv": (2048, 6144), "mlp_gate": (2048, 6144), "attn_q": (2048, 4096)},
    "4B": {"gdn_qkv": (2560, 8192), "mlp_gate": (2560, 9216), "attn_q": (2560, 8192)},
}


def _time(fn, *args, reps: int = 20) -> float:
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report: dict = {"device": jax.devices()[0].device_kind, "tokens": args.tokens, "ms": {}}
    key = jax.random.key(0)
    for size, projections in SHAPES.items():
        for name, (k, n) in projections.items():
            x = jax.random.normal(key, (args.tokens, k), jnp.bfloat16)
            gamma = jnp.zeros((k,), jnp.float32)
            w = (jax.random.normal(key, (k, n), jnp.float32) / np.sqrt(k)).astype(jnp.bfloat16)
            row = {}
            for label, fn in (
                ("xla", norm_matmul.reference),
                ("pallas", norm_matmul.fused_rmsnorm_matmul),
            ):
                forward = jax.jit(fn)
                backward = jax.jit(jax.grad(lambda a, g, b, f=fn: jnp.sum(f(a, g, b).astype(jnp.float32)),
                                            argnums=(0, 1)))  # fmt: skip
                row[label] = {
                    "fwd": _time(forward, x, gamma, w),
                    "fwd_bwd": _time(backward, x, gamma, w),
                }
            report["ms"][f"{size} {name}"] = row
            print(size, name, json.dumps(row), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
