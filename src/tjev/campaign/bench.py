"""TPU benchmarks: the Pallas kernels against XLA, then the real train step per model size.

    tjev campaign bench --models MODELS [--sizes 0.8B,2B,4B] --out bench.json

1. GDN op (fwd + bwd) at 8k tokens per chip: chunked XLA ("high") vs pallas_tpu
   (high / bf16, chunk 64 / 128); splash vs XLA attention at T 1024 and 4096.
2. Train step on one chip, synthetic packed batches (2048-token rows, 2 microbatches):
   the hardware preset (kernels on), selective remat, the split GDN kernel, and the XLA
   paths. Writes ``tokens_per_second_per_chip`` and ``best_variant`` (the fastest variant
   that leaves room for the long buckets) per size, read by ``tjev campaign cost --bench``.
Everything runs on the first chip; a variant that fails (compile error, OOM) is recorded
with its error and skipped.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tjev.campaign.plan import HARDWARE


def hardware() -> str:
    """HARDWARE key of this host's chips ("TPU v5 lite" = v5e, else v6e)."""
    import jax

    return "v5e" if "v5" in jax.devices()[0].device_kind else "v6e"


def _time(fn, *args, reps=5):
    import jax

    jax.block_until_ready(fn(*args))  # compile + warm up
    jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps


def bench_kernels() -> dict:
    import jax
    import jax.numpy as jnp

    from tjev.kernels.attention.splash_tpu import splash_attention
    from tjev.kernels.gated_delta_rule.pallas_tpu import chunk_gated_delta_rule_tpu
    from tjev.kernels.gated_delta_rule.xla import chunk_gated_delta_rule, l2norm

    out = {}
    for heads in (16, 32):  # 0.8B/2B, 4B/9B value heads
        b, t = 2, 4096
        keys = jax.random.split(jax.random.key(0), 5)
        q = l2norm(jax.random.normal(keys[0], (b, t, heads, 128))) * 128**-0.5
        k = l2norm(jax.random.normal(keys[1], (b, t, heads, 128)))
        v = jax.random.normal(keys[2], (b, t, heads, 128))
        g = -jax.nn.softplus(jax.random.normal(keys[3], (b, t, heads)) - 1.0)
        beta = jax.nn.sigmoid(jax.random.normal(keys[4], (b, t, heads)))
        seg = jnp.repeat(jnp.arange(1, 17), t // 16)[None].repeat(b, 0).astype(jnp.int32)
        variants = {
            "chunked/high": lambda *a: chunk_gated_delta_rule(*a, precision="high")[0],
            "pallas/high": lambda *a: chunk_gated_delta_rule_tpu(*a, precision="high"),
            "pallas/bf16": lambda *a: chunk_gated_delta_rule_tpu(*a, precision="bf16"),
            "pallas/high/C128": lambda *a: chunk_gated_delta_rule_tpu(
                *a, chunk=128, precision="high"
            ),
        }
        for name, fn in variants.items():

            def step(q, k, v, g, beta, fn=fn, seg=seg):
                loss = lambda *x: jnp.sum(fn(*x, seg))  # noqa: E731
                return jax.grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)

            key = f"gdn H{heads} {name}"
            try:
                out[key] = {"ms": 1e3 * _time(jax.jit(step), q, k, v, g, beta)}
            except Exception as e:
                out[key] = {"error": f"{type(e).__name__}: {e}"[:500]}
            print(key, out[key], flush=True)
    for length, heads, kv in ((1024, 8, 2), (4096, 8, 2), (4096, 16, 4)):
        rows = 8192 // length
        keys = jax.random.split(jax.random.key(1), 3)
        q = jax.random.normal(keys[0], (rows, length, heads, 256), jnp.bfloat16)
        k, v = (jax.random.normal(kk, (rows, length, kv, 256), jnp.bfloat16) for kk in keys[1:])
        seg = jnp.repeat(jnp.arange(1, 5), length // 4)[None].repeat(rows, 0).astype(jnp.int32)
        idx = jnp.arange(length)
        mask = (seg[:, :, None] == seg[:, None, :]) & (idx[None, :] <= idx[:, None])
        variants = {
            "splash": lambda q, k, v, seg=seg: splash_attention(q, k, v, seg),
            "xla": lambda q, k, v, m=mask: jax.nn.dot_product_attention(q, k, v, mask=m[:, None]),
        }
        for name, fn in variants.items():

            def step(q, k, v, fn=fn):
                loss = lambda *x: jnp.sum(fn(*x).astype(jnp.float32))  # noqa: E731
                return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

            key = f"attention T{length} H{heads}/{kv} {name}"
            try:
                out[key] = {"ms": 1e3 * _time(jax.jit(step), q, k, v)}
            except Exception as e:
                out[key] = {"error": f"{type(e).__name__}: {e}"[:500]}
            print(key, out[key], flush=True)
    return out


def bench_step(size: str, model_dir: Path, overrides: list[str], bucket=2048, accum=2) -> dict:
    import jax

    from tjev.config import MeshSpec, load_config
    from tjev.model import build_model
    from tjev.sharding import install_mesh, make_mesh
    from tjev.testing import synthetic_batch
    from tjev.train.optim import make_optimizer
    from tjev.train.step import make_train_step, split_model

    hw = HARDWARE[hardware()]
    cfg = load_config(
        hw["preset"],
        f"qwen35-{size.lower()}",
        overrides=[
            f"model.path={model_dir}",
            f"train.microbatch_tokens={hw['microbatch'][size]}",
            *overrides,
        ],
    )
    install_mesh(cfg.compute, make_mesh(MeshSpec(data=1), jax.devices()[:1]))
    _, model = build_model(str(model_dir), cfg.compute, cfg.lora, dtype=cfg.model.dtype)
    graphdef, lora, frozen = split_model(model)
    del model
    tx = make_optimizer(cfg.optim, 100)
    opt = tx.init(lora)
    step = jax.jit(make_train_step(graphdef, tx, brier_weight=0.1, slots_per_step=1.0),
                   donate_argnums=(0, 1))  # fmt: skip
    rows = max(1, cfg.train.microbatch_tokens // bucket)
    batch = synthetic_batch(rows, bucket, accum)
    t0 = time.perf_counter()
    compiled = step.lower(lora, opt, frozen, batch).compile()
    compile_s = time.perf_counter() - t0
    for _ in range(2):
        lora, opt, metrics = compiled(lora, opt, frozen, batch)
    jax.block_until_ready(metrics)
    t0, reps = time.perf_counter(), 4
    for _ in range(reps):
        lora, opt, metrics = compiled(lora, opt, frozen, batch)
    jax.block_until_ready(metrics)
    seconds = (time.perf_counter() - t0) / reps
    stats = jax.devices()[0].memory_stats() or {}
    return {
        "step_s": seconds,
        "tokens_per_second": rows * bucket * accum / seconds,
        "compile_s": compile_s,
        "hbm_peak_gb": stats.get("peak_bytes_in_use", 0) / 1e9,
        "finite": bool(np.isfinite(float(metrics["loss"]))),
    }


VARIANTS = {  # the preset (fused GDN forward, splash, full remat) and alternatives
    "kernels": [],
    "kernels+core": ["compute.remat=core"],
    "kernels+minimal": ["compute.remat=minimal"],
    "split": ["compute.gdn_impl=pallas_tpu_split"],
    "xla": ["compute.attention=xla", "compute.gdn_impl=chunked"],
}
HBM_FRACTION = 0.8  # a variant must leave room for the 4096-token buckets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tjev campaign bench")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--sizes", default="0.8B,2B,4B,9B")
    parser.add_argument("--skip-kernels", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("bench.json"))
    args = parser.parse_args(argv)
    import jax

    report: dict[str, Any] = {"device": jax.devices()[0].device_kind, "kernels": {}, "steps": {}}
    if not args.skip_kernels:
        report["kernels"] = bench_kernels()
    best: dict[str, tuple[float, str]] = {}
    for size in args.sizes.split(","):
        folder = args.models / f"Qwen3.5-{size}"
        if not folder.exists():
            continue
        for name, overrides in VARIANTS.items():
            key = f"{size} {name}"
            try:
                result: dict[str, Any] = bench_step(size, folder, overrides)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"[:500]}
                traceback.print_exc()
            report["steps"][key] = result
            print(key, result, flush=True)
            hbm = (jax.devices()[0].memory_stats() or {}).get("bytes_limit", 32e9) / 1e9
            fits = result.get("hbm_peak_gb", 0) < HBM_FRACTION * hbm
            if (
                result.get("finite")
                and fits
                and result["tokens_per_second"] > best.get(size, (0,))[0]
            ):
                best[size] = (result["tokens_per_second"], name)
            jax.clear_caches()
    report["tokens_per_second_per_chip"] = {s: v[0] for s, v in best.items()}
    report["best_variant"] = {s: v[1] for s, v in best.items()}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"tokens_per_second_per_chip": report["tokens_per_second_per_chip"],
                      "best_variant": report["best_variant"]}, indent=2))  # fmt: skip
    return 0
