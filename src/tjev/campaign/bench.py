"""TPU benchmarks: the Pallas kernels against XLA, then the real train step per model size.

    tjev campaign bench --models MODELS [--sizes 0.8B,2B,4B] --out bench.json

1. GDN op (fwd + bwd) at 8k tokens per chip: chunked XLA ("high") vs pallas_tpu
   (high / bf16, chunk 64 / 128); splash vs XLA attention at T 1024 and 4096.
2. Train step on one chip, synthetic packed batches (2048-token rows, 2 microbatches):
   the hardware preset (kernels on), selective remat, the split GDN kernel, and the XLA
   paths. Writes ``tokens_per_second_per_chip`` and ``best_variant`` (the fastest variant
   that leaves room for the long buckets) per size, read by ``tjev campaign cost --bench``.
3. With ``--profile DIR``: per size and variant, an xprof trace of 3 timed train steps in
   ``DIR/<size>-<variant>`` and one untimed detail step (TRACE_COMPUTE_AND_SYNC: the sync
   and DMA waits) in ``DIR/<size>-<variant>/detail`` (view them with ``xprof --logdir``).
4. With ``--debug-dir DIR`` (a probe session only; off by default): for the variants named
   by ``--debug-variants``, the compile log with cache-miss explanations
   (``DIR/compiles-<size>-<variant>.log``) and the XLA HLO dump
   (``DIR/xla/<size>-<variant>/``, ``*after_optimizations*.txt`` holds the real fusions and
   op_name metadata); for the whole process, the Mosaic (Pallas) MLIR dump in
   ``DIR/mosaic`` (a libtpu flag, read once at backend start). Timings of a dumped variant
   include the dump: run it apart from the measured bench.
Everything runs on the first chip; a variant that fails (compile error, OOM) is recorded
with its error and skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import time
import traceback
from collections.abc import Iterator
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
    from tjev.kernels.gated_delta_rule.pallas_tpu_v2 import chunk_gated_delta_rule_v2
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
            "split/high/solve": lambda *a: chunk_gated_delta_rule_tpu(
                *a, precision="high", fused=False, inverse="solve"
            ),
            "split/high/blocked": lambda *a: chunk_gated_delta_rule_tpu(
                *a, precision="high", fused=False, inverse="blocked"
            ),
            "v2/high": lambda *a: chunk_gated_delta_rule_v2(*a, precision="high"),
            "v2/mixed": lambda *a: chunk_gated_delta_rule_v2(*a, precision="mixed"),
            "v2/bf16": lambda *a: chunk_gated_delta_rule_v2(*a, precision="bf16"),
            "v2/mixed/C128": lambda *a: chunk_gated_delta_rule_v2(*a, chunk=128, precision="mixed"),
            "v2/high/blocked": lambda *a: chunk_gated_delta_rule_v2(
                *a, precision="high", inverse="blocked"
            ),
            "v2/high/hb16": lambda *a: chunk_gated_delta_rule_v2(
                *a, precision="high", heads_per_step=16
            ),
            "v2/high/hb32": lambda *a: chunk_gated_delta_rule_v2(
                *a, precision="high", heads_per_step=32
            ),
            "v2/high/C128": lambda *a: chunk_gated_delta_rule_v2(*a, chunk=128, precision="high"),
            "v2/high/C128/blocked": lambda *a: chunk_gated_delta_rule_v2(
                *a, chunk=128, precision="high", inverse="blocked"
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
    out |= bench_gdn_core()
    out |= bench_lora()
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
            "splash/fusedbwd": lambda q, k, v, seg=seg: splash_attention(
                q, k, v, seg, fused_bwd=True
            ),
            "splash/1024c512": lambda q, k, v, seg=seg: splash_attention(
                q, k, v, seg, block=1024, compute=512
            ),
            "splash/1024c512/fusedbwd": lambda q, k, v, seg=seg: splash_attention(
                q, k, v, seg, block=1024, compute=512, fused_bwd=True
            ),
            "splash/512c256": lambda q, k, v, seg=seg: splash_attention(q, k, v, seg, compute=256),
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


def enable_mosaic_dump(debug_dir: Path) -> None:
    """Append the Mosaic MLIR dump to LIBTPU_INIT_ARGS: call before JAX starts (libtpu
    reads it once). The flags already in the environment (cloud/tpu_env.sh) are kept."""
    flag = f"--xla_mosaic_dump_to={debug_dir / 'mosaic'}"
    current = os.environ.get("LIBTPU_INIT_ARGS", "")
    if flag not in current:
        os.environ["LIBTPU_INIT_ARGS"] = f"{current} {flag}".strip()


@contextlib.contextmanager
def debug_compiles(debug_dir: Path | None, tag: str) -> Iterator[dict[str, Any]]:
    """Within the block: JAX compile logging with cache-miss explanations, written to
    ``debug_dir/compiles-<tag>.log``; yields the ``compiler_options`` that dump the XLA HLO
    of the compiled step to ``debug_dir/xla/<tag>``. A no-op (empty options) without a dir."""
    if debug_dir is None:
        yield {}
        return
    import jax

    debug_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(debug_dir / f"compiles-{tag}.log")
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("jax")
    logger.addHandler(handler)
    saved = jax.config.jax_log_compiles, jax.config.jax_explain_cache_misses
    jax.config.update("jax_log_compiles", True)
    jax.config.update("jax_explain_cache_misses", True)
    try:
        yield {"xla_dump_to": str(debug_dir / "xla" / tag), "xla_dump_hlo_as_text": True}
    finally:
        jax.config.update("jax_log_compiles", saved[0])
        jax.config.update("jax_explain_cache_misses", saved[1])
        logger.removeHandler(handler)
        handler.close()


def bench_step(
    size: str,
    model_dir: Path,
    overrides: list[str],
    bucket=2048,
    accum=2,
    profile_dir: Path | None = None,
    profile_steps: int = 3,
    compiler_options: dict[str, Any] | None = None,
) -> dict:
    import jax

    from tjev.config import MeshSpec, load_config
    from tjev.model import build_model
    from tjev.sharding import configure_runtime, install_mesh, make_mesh
    from tjev.testing import synthetic_batch
    from tjev.train import flops as flops_mod
    from tjev.train.optim import make_optimizer
    from tjev.train.step import make_train_step, named, split_model, train_step_name

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
    configure_runtime(cfg.compute)  # before the first device query below
    install_mesh(cfg.compute, make_mesh(MeshSpec(data=1), jax.devices()[:1]))
    config, model = build_model(str(model_dir), cfg.compute, cfg.lora, dtype=cfg.model.dtype)
    graphdef, lora, frozen = split_model(model)
    del model
    tx = make_optimizer(cfg.optim, 100)
    opt = tx.init(lora)
    rows = max(1, cfg.train.microbatch_tokens // bucket)
    batch = synthetic_batch(rows, bucket, accum)
    # the program is named after its role and shape (jit_train_step_T…_mb…), as in training
    step = jax.jit(named(make_train_step(graphdef, tx, brier_weight=0.1, slots_per_step=1.0),
                         train_step_name(batch)), donate_argnums=(0, 1))  # fmt: skip
    t0 = time.perf_counter()
    compiled = step.lower(lora, opt, frozen, batch).compile(compiler_options or None)
    compile_s = time.perf_counter() - t0
    # the step's own footprint (the device's peak_bytes_in_use is process-wide: every
    # variant benched before this one would count)
    mem = compiled.memory_analysis()  # None where the backend has no analysis
    mem_bytes = lambda field: getattr(mem, field, 0) if mem is not None else 0  # noqa: E731
    step_bytes = (
        mem_bytes("argument_size_in_bytes") + mem_bytes("output_size_in_bytes")
        + mem_bytes("temp_size_in_bytes") - mem_bytes("alias_size_in_bytes")
    )  # fmt: skip
    model_flops = flops_mod.train_flops(
        config,
        flops_mod.segment_lengths(np.asarray(batch.segment_ids)),
        cfg.compute.gdn_chunk,
        int(np.asarray(batch.label_mask).sum()),
    )
    for _ in range(2):
        lora, opt, metrics = compiled(lora, opt, frozen, batch)
    jax.block_until_ready(metrics)
    t0, reps = time.perf_counter(), 4
    for _ in range(reps):
        lora, opt, metrics = compiled(lora, opt, frozen, batch)
    jax.block_until_ready(metrics)
    seconds = (time.perf_counter() - t0) / reps
    stats = jax.devices()[0].memory_stats() or {}  # before the trace, which may allocate
    profile_error = None
    if profile_dir is not None:  # xprof traces after the timing: never fatal to the timing
        from tjev.campaign.xprof_summary import profile_options

        try:
            with jax.profiler.trace(str(profile_dir), profiler_options=profile_options()):
                for i in range(profile_steps):  # step markers: per-step tools need them
                    with jax.profiler.StepTraceAnnotation("train", step_num=i):
                        lora, opt, metrics = compiled(lora, opt, frozen, batch)
                        jax.block_until_ready(metrics)
            # what is live on the device now (the step's buffers): xprof's memory profile
            jax.profiler.save_device_memory_profile(str(Path(profile_dir) / "memory.prof"))
            # the detail capture: one untimed step with the sync / DMA waits (Pallas stalls)
            with (
                jax.profiler.trace(
                    str(Path(profile_dir) / "detail"),
                    profiler_options=profile_options(detail=True),
                ),
                jax.profiler.StepTraceAnnotation("train", step_num=0),
            ):
                lora, opt, metrics = compiled(lora, opt, frozen, batch)
                jax.block_until_ready(metrics)
        except Exception as e:
            profile_error = f"{type(e).__name__}: {e}"[:300]
    peak = flops_mod.peak_flops(jax.devices()[0].device_kind)
    return {
        "step_s": seconds,
        "tokens_per_second": rows * bucket * accum / seconds,
        "tokens_per_step": rows * bucket * accum,
        "mfu": model_flops / seconds / peak if peak else None,
        "compile_s": compile_s,
        "hbm_step_gb": step_bytes / 1e9,
        "hbm_temp_gb": mem_bytes("temp_size_in_bytes") / 1e9,
        "profile": (
            {"error": profile_error}
            if profile_error
            else _profile_summary(profile_dir, profile_steps)
        ),
        "hbm_process_peak_gb": stats.get("peak_bytes_in_use", 0) / 1e9,
        "finite": bool(np.isfinite(float(metrics["loss"]))),
    }


def _gdn_core_variants(hk: int, heads: int, b: int = 2, t: int = 4096, d: int = 128):
    """Inputs and the core functions of bench_gdn_core at one model layout."""
    import jax
    import jax.numpy as jnp

    from tjev.kernels.causal_conv1d import segmented_causal_conv1d
    from tjev.kernels.gated_delta_rule.pallas_tpu_mega import gdn_megakernel
    from tjev.kernels.gated_delta_rule.pallas_tpu_v2 import chunk_gated_delta_rule_v2
    from tjev.kernels.gated_delta_rule.xla import l2norm

    width = 2 * hk * d + heads * d
    keys = jax.random.split(jax.random.key(0), 6)
    qkv = jax.random.normal(keys[0], (b, t, width), jnp.bfloat16)
    a = jax.random.normal(keys[1], (b, t, heads), jnp.bfloat16)
    bb = jax.random.normal(keys[2], (b, t, heads), jnp.bfloat16)
    a_log = jax.random.normal(keys[3], (heads,)) * 0.5
    dt_bias = jax.random.normal(keys[4], (heads,)) * 0.5
    w = (jax.random.normal(keys[5], (4, width)) * 0.3).astype(jnp.bfloat16)
    seg = jnp.repeat(jnp.arange(1, 17), t // 16)[None].repeat(b, 0).astype(jnp.int32)

    def glue(qkv, a, bb, conv=True):
        if conv:
            x = segmented_causal_conv1d(qkv, w, seg)
            qkv = jnp.where((seg > 0)[..., None], jax.nn.silu(x), 0).astype(qkv.dtype)
        q, k, v = jnp.split(qkv, [hk * d, 2 * hk * d], axis=-1)
        q = l2norm(q.reshape(b, t, hk, d).astype(jnp.float32)) * d**-0.5
        k = l2norm(k.reshape(b, t, hk, d).astype(jnp.float32))
        beta = jax.nn.sigmoid(bb.astype(jnp.float32))
        g = -jnp.exp(a_log) * jax.nn.softplus(a.astype(jnp.float32) + dt_bias)
        v = v.reshape(b, t, heads, d)
        return chunk_gated_delta_rule_v2(q, k, v, g, beta, seg, precision="high")

    def mega(qkv, a, bb, conv=True, **kw):
        return gdn_megakernel(
            qkv, a, bb, a_log, dt_bias, w if conv else None, seg, key_heads=hk, key_dim=d,
            value_heads=heads, value_dim=d, precision="high", **kw,
        )  # fmt: skip

    variants = {
        "xla-glue+v2": glue,
        "mega2": mega,
        "mega2/blocked": lambda *x: mega(*x, inverse="blocked"),
        "mega2/hb16": lambda *x: mega(*x, heads_per_step=16),
        "post-conv: xla-glue+v2": lambda *x: glue(*x, conv=False),
        "post-conv: mega1": lambda *x: mega(*x, conv=False),
    }
    return (qkv, a, bb), variants


def bench_gdn_core() -> dict:
    """The GDN core from the projection outputs (raw qkv, a, b → o, fwd + bwd): the XLA glue
    (conv + SiLU, l2norm, gates) around v2 against the megakernels mega1 and mega2."""
    import jax
    import jax.numpy as jnp

    out = {}
    for hk, heads in ((16, 16), (16, 32)):  # 0.8B / 2B, 9B
        inputs, variants = _gdn_core_variants(hk, heads)
        for name, fn in variants.items():

            def step(qkv, a, bb, fn=fn):
                loss = lambda *x: jnp.sum(fn(*x).astype(jnp.float32))  # noqa: E731
                return jax.grad(loss, argnums=(0, 1, 2))(qkv, a, bb)

            key = f"gdn-core H{hk}/{heads} {name}"
            try:
                out[key] = {"ms": 1e3 * _time(jax.jit(step), *inputs)}
            except Exception as e:
                out[key] = {"error": f"{type(e).__name__}: {e}"[:500]}
            print(key, out[key], flush=True)
    return out


# the projection groups (fan-in, fan-outs) of a decoder layer: GDN in_proj_qkv + in_proj_z,
# gate + up, down; rows = the planner's microbatch on v6e-1 (4B 4096, 9B 2048 tokens)
LORA_LAYOUTS = {
    "4B": (4096, {"gdn": (2560, (8192, 4096)), "mlp": (2560, (9216, 9216)),
                  "down": (9216, (2560,))}),
    "9B": (2048, {"gdn": (4096, (8192, 4096)), "mlp": (4096, (12288, 12288)),
                  "down": (12288, (4096,))}),
}  # fmt: skip


def _lora_variants(fan_in: int, fan_outs: tuple[int, ...], rank: int = 16):
    """The LoRA forms on stacked bf16 kernels [8, K, N] read at a traced layer index (as in
    the scan): the frozen base alone (the floor), unmerged (XLA), merged (W' per call)."""
    import jax
    import jax.numpy as jnp

    from tjev.model.lora import _merged_matmuls

    bf16 = jnp.bfloat16
    keys = iter(jax.random.split(jax.random.key(0), 3 * len(fan_outs)))
    ws = tuple(0.02 * jax.random.normal(next(keys), (8, fan_in, n), bf16) for n in fan_outs)
    a = tuple(jax.random.normal(next(keys), (fan_in, rank)) / fan_in**0.5 for _ in fan_outs)
    b = tuple(1e-3 * jax.random.normal(next(keys), (rank, n)) for n in fan_outs)
    scales = (4.0,) * len(fan_outs)

    def layer(ws, i):
        return tuple(jax.lax.dynamic_index_in_dim(w, i, 0, keepdims=False) for w in ws)

    def base(x, ws, a, b, i):
        return [x @ w for w in layer(ws, i)]

    def unmerged(x, ws, a, b, i):
        xa = x @ jnp.concatenate([ai.astype(bf16) for ai in a], axis=-1)
        return [x @ w + (xa[..., j * rank : (j + 1) * rank] @ bj.astype(bf16)) * s
                for j, (w, bj, s) in enumerate(zip(layer(ws, i), b, scales, strict=True))]  # fmt: skip

    def merged(x, ws, a, b, i):
        labels = tuple(f"proj{j}" for j in range(len(ws)))  # named scopes only
        return list(_merged_matmuls(x, layer(ws, i), a, b, (None,) * len(ws), scales, "bfloat16",
                                    labels))  # fmt: skip

    variants = {"base": base, "unmerged": unmerged, "merged": merged}
    return (ws, a, b), variants


def bench_lora() -> dict:
    """One layer's projection groups, fwd + bwd (gradients of x, A, B): the frozen base
    alone against unmerged and merged LoRA (rank 16, docs/kernels.md "Merged LoRA")."""
    import jax
    import jax.numpy as jnp

    out = {}
    for size, (rows, groups) in LORA_LAYOUTS.items():
        for group, (fan_in, fan_outs) in groups.items():
            (ws, a, b), variants = _lora_variants(fan_in, fan_outs)
            x = jax.random.normal(jax.random.key(1), (1, rows, fan_in), jnp.bfloat16)
            for name, fn in variants.items():

                def step(x, a, b, ws, i, fn=fn):
                    loss = lambda x, a, b: sum(  # noqa: E731
                        jnp.sum(y.astype(jnp.float32)) for y in fn(x, ws, a, b, i)
                    )
                    return jax.grad(loss, argnums=(0, 1, 2))(x, a, b)

                key = f"lora {size} {group} {name}"
                try:
                    ms = 1e3 * _time(jax.jit(step), x, a, b, ws, jnp.int32(3))
                    out[key] = {"ms": ms}
                except Exception as e:
                    out[key] = {"error": f"{type(e).__name__}: {e}"[:500]}
                print(key, out[key], flush=True)
    return out


SPLIT = "compute.gdn_impl=pallas_tpu_split"
V2 = "compute.gdn_impl=pallas_tpu_v2"
FUSED_BWD = "compute.attention_fused_bwd=true"
SPLASH_1024 = ("compute.attention_splash_block=1024", "compute.attention_splash_compute=512")
STATES = "compute.remat=core_states"
MEGA1 = ("compute.gdn_impl=pallas_tpu_mega1", STATES)
MEGA2 = ("compute.gdn_impl=pallas_tpu_mega2", STATES)
BLOCKED = "compute.gdn_inverse=blocked"
MERGED = "compute.lora_impl=merged"
HB = lambda n: (f"compute.gdn_heads_per_step={n}", "compute.gdn_vmem_fraction=0.85")  # noqa: E731
# cycle 3 (docs/kernels.md): every new option on the fastest correct path (v2 + core_states)
CYCLE3 = {
    # lever 2: v2 tuning (heads per step with a larger VMEM limit, blocked inverse, C=128)
    "v2-hb16": [V2, STATES, *HB(16)],
    "v2-hb32": [V2, STATES, *HB(32)],  # 0.8B has 16 value heads: = v2-hb16
    "v2-blocked": [V2, STATES, BLOCKED],
    "v2-c128": [V2, STATES, "compute.gdn_chunk=128"],
    "v2-c128-blocked": [V2, STATES, "compute.gdn_chunk=128", BLOCKED],
    # lever 1: the megakernels (mega1: gates, l2norm, GQA in the kernels; mega2: + conv)
    "mega1": [*MEGA1],
    "mega2": [*MEGA2],
    "mega2-blocked": [*MEGA2, BLOCKED],
    "mega2-hb16": [*MEGA2, *HB(16)],
    "mega2-c128-blocked": [*MEGA2, "compute.gdn_chunk=128", BLOCKED],
    # lever 3: merged LoRA weights
    "lora-merged": [V2, STATES, MERGED],
    "mega2+lora-merged": [*MEGA2, MERGED],
    # lever 4 on the megakernel, and everything together
    "mega2+splash-1024-fusedbwd": [*MEGA2, *SPLASH_1024, FUSED_BWD],
    "mega2-blocked+lora-merged+splash-1024-fusedbwd": [
        *MEGA2,
        BLOCKED,
        MERGED,
        *SPLASH_1024,
        FUSED_BWD,
    ],
}
# cycle 4: on the sweep's path (mega2, chunk 128, blocked inverse), the two speed-ups the
# sweep found free (one-pass bf16 GDN matmuls, merged LoRA), bf16 saved states, chunk 256
SWEEP_PATH = [*MEGA2, "compute.gdn_chunk=128", BLOCKED]
GDN_BF16 = "compute.gdn_precision=bf16"
CYCLE4 = {
    "mega2-c128-gbf16": [*SWEEP_PATH, GDN_BF16],
    "mega2-c128-merged": [*SWEEP_PATH, MERGED],
    "mega2-c128-gbf16-merged": [*SWEEP_PATH, GDN_BF16, MERGED],
    "mega2-c128-mixed": [*SWEEP_PATH, "compute.gdn_precision=mixed"],
    "mega2-c256-blocked": [*MEGA2, "compute.gdn_chunk=256", BLOCKED],
}
# the integration's validation arms on the sweep's path (docs/training.md, "Integration on
# v6e1"): each feature alone on the planner's v6e1 configuration of the size (≤ 4B: merged
# LoRA, mega2-c128-merged; 9B: unmerged, mega2-c128-blocked). Remat: minimal_states for
# ≤ 4B, per bucket at 9B (its 4096 bucket keeps core_states). The all-in arms with the fused
# LoRA kernel (allin, allin-9b) went with that kernel: it measured slower and was removed
FLAT = "compute.gdn_norm=flat"
LETTERS = "compute.head_rows=letters"  # 9B's untied head (the tied heads refuse nothing)
BEST_FIT = "train.packing=best_fit"
MSTATES = "compute.remat=minimal_states"
BY_BUCKET = 'compute.remat_by_bucket={"1024":"minimal_states","2048":"minimal_states"}'
MIXED = "compute.gdn_precision=mixed"
INTEGRATION = {
    # ≤ 4B, against mega2-c128-merged
    "c128-merged-flat": [*SWEEP_PATH, MERGED, FLAT],
    "c128-merged-packing": [*SWEEP_PATH, MERGED, BEST_FIT],
    "c128-merged-mstates": [*SWEEP_PATH, MERGED, MSTATES],
    # 9B, against mega2-c128-blocked
    "c128-flat-letters": [*SWEEP_PATH, FLAT, LETTERS],
    "c128-packing": [*SWEEP_PATH, BEST_FIT],
    "c128-mstates-bucket": [*SWEEP_PATH, BY_BUCKET],
}
VARIANTS = {  # the preset (fused GDN forward, splash, full remat) and alternatives
    "kernels": [],
    "kernels+core": ["compute.remat=core"],
    "kernels+minimal": ["compute.remat=minimal"],
    "split": [SPLIT],
    "xla": ["compute.attention=xla", "compute.gdn_impl=chunked"],
    # the XLA paths without the optimization barriers around the GDN conv: XLA:TPU then
    # miscompiles it (docs/results.md, TPU correctness): fast, but wrong (VARIANT_ENV)
    "xla-nobarrier": ["compute.attention=xla", "compute.gdn_impl=chunked"],
    # the GDN forward kernel once per microbatch, not 2-3 times (docs/kernels.md, remat)
    "split+core": [SPLIT, "compute.remat=core"],
    "split+states": [SPLIT, "compute.remat=core_states"],
    "kernels+states": ["compute.remat=core_states"],
    # the XLA terms' UT solve (split forward + backward; the fused backward's recompute)
    "split+solve": [SPLIT, "compute.gdn_inverse=solve"],
    "split+blocked": [SPLIT, "compute.gdn_inverse=blocked"],
    "kernels+solve": ["compute.gdn_inverse=solve"],
    # bf16_3x intra-chunk, one bf16 pass on the state path (xla.part)
    "mixed": ["compute.gdn_precision=mixed"],
    "split+mixed": [SPLIT, "compute.gdn_precision=mixed"],
    # fused Pallas forward and backward (pallas_tpu_v2), with the residual-saving remat
    "fused-bwd": [V2],
    "fused-bwd+states": [V2, "compute.remat=core_states"],
    "fused-bwd+mixed": [V2, "compute.gdn_precision=mixed"],
    "fused-bwd+mixed+states": [V2, "compute.gdn_precision=mixed", "compute.remat=core_states"],
    # splash tiles and backward form (research: 1024-class tiles at T=4096, fused dkv+dq)
    "splash-fusedbwd": [FUSED_BWD],
    "splash-1024": [*SPLASH_1024],
    "splash-1024-fusedbwd": [*SPLASH_1024, FUSED_BWD],
    # the splash options on the fastest correct GDN path (v2 + core_states remat)
    "v2+splash-fusedbwd": [V2, "compute.remat=core_states", FUSED_BWD],
    "v2+splash-1024-fusedbwd": [V2, "compute.remat=core_states", *SPLASH_1024, FUSED_BWD],
    "splash-512-c256": ["compute.attention_splash_compute=256"],
    **CYCLE3,
    **CYCLE4,
    **INTEGRATION,
}
# what a variant sets in the environment rather than the config (read at trace time)
VARIANT_ENV: dict[str, dict[str, str]] = {"xla-nobarrier": {"TJEV_GDN_CONV_BARRIER": "0"}}


@contextlib.contextmanager
def variant_env(name: str) -> Iterator[dict[str, str]]:
    """Within the block: the variant's environment (VARIANT_ENV), restored after."""
    env = VARIANT_ENV.get(name, {})
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield env
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


HBM_FRACTION = 0.8  # a variant must leave room for the 4096-token buckets


def _profile_summary(profile_dir: Path | None, steps: int = 3) -> dict | None:
    """Where the device time of the traced steps goes (by model scope, phase, HLO category,
    op): the xplane through xprof when installed (tjev[profile]), else the trace JSON."""
    if profile_dir is None:
        return None
    from tjev.campaign.xprof_tools import summarize_any

    try:
        return summarize_any(profile_dir, steps=steps)
    except Exception as e:  # a summary is a convenience: the raw trace is the record
        return {"error": f"{type(e).__name__}: {e}"[:300]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tjev campaign bench")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--sizes", default="0.8B,2B,4B,9B")
    parser.add_argument("--skip-kernels", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("bench.json"))
    parser.add_argument(
        "--profile", type=Path,
        help="xprof traces of every benched variant: PROFILE/<size>-<variant> (the timed "
             "capture, and one untimed TRACE_COMPUTE_AND_SYNC step under detail/)",
    )  # fmt: skip
    parser.add_argument("--variants", default=",".join(VARIANTS), help="train-step variants")
    parser.add_argument("--profile-steps", type=int, default=3, help="steps per xprof trace")
    parser.add_argument(
        "--debug-dir", type=Path, help="probe only: compile logs, XLA and Mosaic dumps (docstring)"
    )
    parser.add_argument(
        "--debug-variants", default="split", help="the variants whose compiles are logged/dumped"
    )
    args = parser.parse_args(argv)
    names = args.variants.split(",")
    if unknown := set(names) - set(VARIANTS):
        parser.error(f"unknown variants {sorted(unknown)}; known: {', '.join(VARIANTS)}")
    debugged = set(args.debug_variants.split(",")) if args.debug_dir else set()
    if args.debug_dir:
        enable_mosaic_dump(args.debug_dir)  # before JAX starts
    import jax

    from tjev.campaign.experiments import Experiment
    from tjev.campaign.xprof_tools import log_profile

    device = jax.devices()[0].device_kind
    report: dict[str, Any] = {"device": device, "kernels": {}, "steps": {}}
    project = os.environ.get("TJEV_WANDB_BENCHMARKS_PROJECT", "tjev-benchmarks")
    exp = Experiment("bench", f"bench-{hardware()}", project=project, config={
        "device": device, "jax": jax.__version__, "sizes": args.sizes,
        "variants": {n: VARIANTS[n] for n in names},
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
    })  # fmt: skip
    if not args.skip_kernels:
        report["kernels"] = bench_kernels()
    best: dict[str, tuple[float, str]] = {}
    for size in args.sizes.split(","):
        folder = args.models / f"Qwen3.5-{size}"
        if not folder.exists():
            continue
        for name in names:
            key = f"{size} {name}"
            debug_dir = args.debug_dir if name in debugged else None
            try:
                profile = args.profile / f"{size}-{name}" if args.profile else None
                with variant_env(name), debug_compiles(debug_dir, f"{size}-{name}") as options:
                    result: dict[str, Any] = bench_step(
                        size, folder, VARIANTS[name], profile_dir=profile,
                        profile_steps=args.profile_steps,
                        compiler_options=options,
                    )  # fmt: skip
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"[:500]}
                traceback.print_exc()
            report["steps"][key] = result
            print(key, result, flush=True)
            hbm = (jax.devices()[0].memory_stats() or {}).get("bytes_limit", 32e9) / 1e9
            fits = result.get("hbm_step_gb", 0) < HBM_FRACTION * hbm
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
    exp.table("kernels", [{"op": k, **v} for k, v in report["kernels"].items()])
    exp.table("steps", [{"size": k.split()[0], "variant": k.split()[1], **v}
                        for k, v in report["steps"].items()])  # fmt: skip
    exp.summary({f"{k.replace(' ', '/')}/{m}": v[m] for k, v in report["steps"].items()
                 for m in ("tokens_per_second", "mfu", "hbm_step_gb") if v.get(m) is not None})  # fmt: skip
    exp.summary({f"best/{s}": n for s, n in report["best_variant"].items()})
    for size in args.sizes.split(","):
        runs = {k.split()[1]: v for k, v in report["steps"].items() if k.split()[0] == size}
        for metric in ("tokens_per_second", "mfu"):
            values = {n: v[metric] for n, v in runs.items() if v.get(metric) is not None}
            if values:
                exp.bar(f"{size}/{metric}", values, metric)
        profiled = {n: v["profile"] for n, v in runs.items()
                    if isinstance(v.get("profile"), dict) and "by_scope" in v["profile"]}  # fmt: skip
        for n, prof in profiled.items():
            exp.bar(f"{size}/{n}/time_by_scope", prof["by_scope"], "percent")
            exp.bar(f"{size}/{n}/time_by_category", prof["by_category"], "percent")
            log_profile(exp, f"{size}/{n}/", prof)  # scope × phase, roofline, peak buffers
            exp.summary({f"{size}/{n}/profile_step_ms": prof["step_ms"],
                         f"{size}/{n}/busy_fraction": prof["busy_fraction"]})  # fmt: skip
        # the raw traces (~1 GB per traced variant) come back with the session's reports; W&B
        # (free plan: 5 GB) keeps the numbers and charts, the traces only if asked
        if args.profile and os.environ.get("TJEV_WANDB_TRACES") == "1":
            dirs = [args.profile / f"{size}-{n}" for n in profiled]
            if dirs:
                exp.artifact(f"xprof-{size}", "xprof", dirs)
        exp.table(f"{size}/profile_by_scope", [
            {"variant": n, **v["profile"]["by_scope"]} for n, v in runs.items()
            if isinstance(v.get("profile"), dict) and "by_scope" in v["profile"]
        ])  # fmt: skip
    exp.artifact("bench", "bench", [args.out])
    exp.finish()
    print(json.dumps({"tokens_per_second_per_chip": report["tokens_per_second_per_chip"],
                      "best_variant": report["best_variant"]}, indent=2))  # fmt: skip
    return 0
