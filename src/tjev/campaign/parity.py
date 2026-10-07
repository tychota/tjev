"""Kernels against the XLA reference on real weights and real data: loss, logits, gradients.

    tjev campaign parity --models MODELS --mix MIXDIR [--sizes 0.8B,9B] --out parity.json

The kernel tests check each op on random inputs; this checks the whole train step the way
training runs it. Per size:

1. Real batches from the training stream of the mix (``SegmentAt`` → packer), one per
   sequence bucket, so the long 4096-token rows are covered.
2. One LoRA state for every variant: the model's initial adapters with ``lora_b`` set to
   small random values (at init ``lora_b`` is zero, and so is every ``lora_a`` gradient).
3. The reference: the XLA paths with fp32 delta-rule matmuls (``gdn_precision=highest``).
   Each variant then computes loss, label logits and LoRA gradients on the same batches.
4. Per variant: loss and logit differences, and the gradient against the reference
   (relative L2 error and cosine, overall and per LoRA leaf), plus fwd+bwd time.

A variant passes when its gradient error is at most twice the production XLA path's own
(``xla``, default precision), with a floor of 3e-2, its gradient cosine is ≥ 0.9995 overall
and ≥ 0.998 on every LoRA leaf, and its loss is within 5e-3 (relative). The floor is the
bf16 scale: bf16 operands round at 2⁻⁸ ≈ 0.4% and the error compounds over the layers;
``xla-bf16`` (the XLA paths with one-pass bf16 delta-rule matmuls) measures it.

The bf16 model is also a noise amplifier: its activations round to bf16 after every op, so
*any* change of an op's fp32 result (even 1e-7, a different but exact algorithm) moves some
roundings and the whole gradient by ~2%. Measured on CPU with the realistic tiny model (real
head dims): the XLA reference with only its triangular solve swapped for the doubling
inverse is 1.8e-2 off in gradient (loss 3.7e-3); with ``--model-dtype float32`` the same
swap is 2.8e-6 off, and the fused / split Pallas GDN 8.6e-6 / 3.1e-6. On CPU every GDN
kernel is fp32-exact (Mosaic's HIGH/HIGHEST have no CPU meaning): the ~2% of ``gdn`` and
``gdn-split`` in a bf16 model is this noise, not bf16 casts in the kernels. To resolve kernel
errors below it, run with ``--model-dtype float32`` (every variant and the reference).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tjev.campaign.bench import CYCLE3, CYCLE4, HARDWARE, INTEGRATION, hardware, variant_env

HIGHEST = "@highest"  # not a config override: trace and run under default precision "highest"
REFERENCE = [
    "compute.attention=xla",
    "compute.gdn_impl=chunked",
    "compute.gdn_precision=highest",
]
V2 = "compute.gdn_impl=pallas_tpu_v2"
VARIANTS = {  # the production XLA path first: the others are judged against its error
    "xla": ["compute.attention=xla", "compute.gdn_impl=chunked"],
    # the conv barriers off (bench.VARIANT_ENV): the XLA:TPU miscompile, on purpose
    "xla-nobarrier": ["compute.attention=xla", "compute.gdn_impl=chunked"],
    "kernels": [],
    "split": ["compute.gdn_impl=pallas_tpu_split"],
    "kernels+minimal": ["compute.remat=minimal"],
    "kernels+bf16": ["compute.gdn_precision=bf16"],
    "xla-bf16": ["compute.attention=xla", "compute.gdn_impl=chunked", "compute.gdn_precision=bf16"],
    # one kernel at a time, to locate a mismatch: the fused GDN, the split GDN, splash
    "gdn": ["compute.attention=xla"],
    "gdn-split": ["compute.attention=xla", "compute.gdn_impl=pallas_tpu_split"],
    "splash": ["compute.gdn_impl=chunked"],
    # recompute changes only (remat): the gradients of their base variant, bit for bit or
    # within float reassociation
    "kernels+core": ["compute.remat=core"],
    "split+core": ["compute.gdn_impl=pallas_tpu_split", "compute.remat=core"],
    "split+states": ["compute.gdn_impl=pallas_tpu_split", "compute.remat=core_states"],
    "kernels+states": ["compute.remat=core_states"],
    # the same function, another UT solve in the XLA terms
    "split+solve": ["compute.gdn_impl=pallas_tpu_split", "compute.gdn_inverse=solve"],
    "split+blocked": ["compute.gdn_impl=pallas_tpu_split", "compute.gdn_inverse=blocked"],
    "kernels+solve": ["compute.gdn_inverse=solve"],
    # the precision split: its gate gradients (A_log, dt_bias leaves) are the thing to watch
    "mixed": ["compute.gdn_precision=mixed"],
    "split+mixed": ["compute.gdn_impl=pallas_tpu_split", "compute.gdn_precision=mixed"],
    "xla-mixed": [
        "compute.attention=xla",
        "compute.gdn_impl=chunked",
        "compute.gdn_precision=mixed",
    ],
    # the fused Pallas forward + backward (pallas_tpu_v2); gdn-v2: that kernel alone
    "fused-bwd": [V2],
    "fused-bwd+states": [V2, "compute.remat=core_states"],
    "fused-bwd+mixed": [V2, "compute.gdn_precision=mixed"],
    "fused-bwd+mixed+states": [V2, "compute.gdn_precision=mixed", "compute.remat=core_states"],
    "gdn-v2": ["compute.attention=xla", V2],
    # splash tiles and backward form: float reassociation only
    "splash-fusedbwd": ["compute.attention_fused_bwd=true"],
    "splash-1024": ["compute.attention_splash_block=1024", "compute.attention_splash_compute=512"],
    "splash-1024-fusedbwd": [
        "compute.attention_splash_block=1024",
        "compute.attention_splash_compute=512",
        "compute.attention_fused_bwd=true",
    ],
    "v2+splash-fusedbwd": [V2, "compute.remat=core_states", "compute.attention_fused_bwd=true"],
    "v2+splash-1024-fusedbwd": [
        V2,
        "compute.remat=core_states",
        "compute.attention_splash_block=1024",
        "compute.attention_splash_compute=512",
        "compute.attention_fused_bwd=true",
    ],
    "splash-512-c256": ["compute.attention_splash_compute=256"],
    # cycle 3 (the bench's names): v2 tuning, the megakernels, merged LoRA, splash on top
    **CYCLE3,
    **CYCLE4,
    # perf/integration's validation arms (packing changes no parity batch)
    **INTEGRATION,
    # the megakernel alone (XLA attention), to locate a mismatch
    "gdn-mega1": ["compute.attention=xla", "compute.gdn_impl=pallas_tpu_mega1"],
    "gdn-mega2": ["compute.attention=xla", "compute.gdn_impl=pallas_tpu_mega2"],
    # determinism: the reference itself, rebuilt and recompiled (exactly 0 unless the
    # backend is nondeterministic or a compile is not reproducible)
    "reference": list(REFERENCE),
    # the reference with EVERY f32 dot at fp32: on TPU an f32 dot without an explicit
    # precision is one bf16 pass (XLA:CPU ignores precision), so the plain reference only
    # pins the delta-rule matmuls
    "reference-highest": [*REFERENCE, HIGHEST],
}
# the adapters' size: a LoRA output is ≈ s·√r·σ_B·rms(x) = 32·σ_B·rms(x) (rank 32, rsLoRA
# scale 5.66) against ≈ 0.6·rms(x) for a Qwen3.5 projection, so σ_B = 2e-4 changes each
# projection by ~1% (a trained adapter's order). 0.01 made it ~50%: an unstable 24-layer model
# in which bf16 rounding grows ~1.2× per layer, until two correct paths decorrelate (TPU:
# xla vs reference 3.6e-3 after layer 0, 62% in the loss) — a test artefact, not a bug.
LORA_B_STD = 2e-4
GRAD_FLOOR = 3e-2  # the bf16 scale (module docstring)
BRIER = 0.1


def _config(size: str, model_dir: Path, mix: Path, overrides: list[str]):
    from tjev.config import load_config

    overrides = [o for o in overrides if o != HIGHEST]  # a run mode, not a config override
    hw = HARDWARE[hardware()]
    return load_config(
        hw["preset"],
        f"qwen35-{size.lower()}",
        str(mix / "mix.yaml"),
        overrides=[
            f"model.path={model_dir}",
            f"train.microbatch_tokens={hw['microbatch'][size]}",
            *overrides,
        ],
    )


def real_batches(cfg, count: int, lengths: tuple[int, ...] = ()) -> list[Any]:
    """The first training step of each sequence bucket (one microbatch), then more steps
    up to ``count``: host numpy Batches ``[1, R, ...]``. ``lengths``: only these buckets
    (e.g. the 4096-token one: a check at one compile per variant)."""
    from tjev.data.item import read_jsonl
    from tjev.data.mixture import Mixture
    from tjev.data.pack import Vocab
    from tjev.data.pipeline import train_iterator
    from tjev.data.segments import SegmentAt
    from tjev.data.tokenize import PromptTokenizer
    from tjev.train.loop import load_sources

    sources = load_sources(cfg.data.train)
    extra = [list(read_jsonl(p)) for p in (cfg.data.validation, cfg.data.heldout) if p]
    vocab = Vocab.from_items(*sources.values(), *extra)
    vocab.frozen = True
    mixture = Mixture(sources, cfg.data.mixture or None, cfg.data.mixture_alpha)
    stream = train_iterator(
        SegmentAt(mixture, PromptTokenizer(cfg.model.path), vocab, seed=cfg.train.seed),
        buckets=cfg.train.seq_buckets,
        microbatch_tokens=cfg.train.microbatch_tokens,
        accumulation=1,
        slots=cfg.train.max_segments,
        labels=cfg.train.max_labels,
        prefetch=0,
        bins=cfg.train.packing_bins,
        packing=cfg.train.packing,
    )
    by_length: dict[int, Any] = {}
    more: list[Any] = []
    wanted = set(lengths or cfg.train.seq_buckets)
    every = len(wanted)
    for _ in range(256):  # the long bucket is rare: look a few hundred steps ahead
        batch = next(stream)
        length = batch.tokens.shape[-1]
        if length not in wanted:
            continue
        if length not in by_length:
            by_length[length] = batch
        elif len(by_length) == every and every + len(more) < count:
            more.append(batch)
        if len(by_length) == every and every + len(more) >= count:
            break
    getattr(stream, "close", lambda: None)()
    return [by_length[k] for k in sorted(by_length)] + more


def _named(tree) -> dict[str, np.ndarray]:
    import jax

    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): np.asarray(x, np.float32) for path, x in flat}


def shared_lora(lora, seed: int = 0):
    """The adapters every variant uses: lora_b ~ N(0, LORA_B_STD²), lora_a as initialised."""
    import jax

    flat, treedef = jax.tree_util.tree_flatten_with_path(lora)
    keys = jax.random.split(jax.random.key(seed), len(flat))
    out = []
    for (path, x), key in zip(flat, keys, strict=True):
        if "lora_b" in jax.tree_util.keystr(path):
            x = (LORA_B_STD * jax.random.normal(key, x.shape, np.float32)).astype(x.dtype)
        out.append(np.asarray(x))
    return jax.tree_util.tree_unflatten(treedef, out)


def run_variant(size, model_dir, mix, overrides, batches, lora_values=None) -> dict[str, Any]:
    """Loss, label logits and LoRA gradients of each batch under one compute config."""
    import jax
    import jax.numpy as jnp

    from tjev.config import MeshSpec
    from tjev.model import build_model
    from tjev.sharding import configure_runtime, install_mesh, make_mesh
    from tjev.train.objective import loss_sums
    from tjev.train.step import forward_logits, split_model

    highest = HIGHEST in overrides
    overrides = [o for o in overrides if o != HIGHEST]
    # part of jit's cache key: every call runs under it, or the timing would recompile
    precision = lambda: jax.default_matmul_precision("highest" if highest else None)  # noqa: E731
    cfg = _config(size, model_dir, mix, overrides)
    configure_runtime(cfg.compute)
    install_mesh(cfg.compute, make_mesh(MeshSpec(data=1), jax.devices()[:1]))
    _, model = build_model(
        str(model_dir), cfg.compute, cfg.lora, dtype=cfg.model.dtype, seed=cfg.train.seed
    )
    graphdef, lora, frozen = split_model(model)
    del model
    if lora_values is None:
        lora_values = shared_lora(lora)
    if jax.tree.structure(lora_values) != jax.tree.structure(lora):
        raise ValueError("the variant's LoRA tree differs from the reference's")
    lora = jax.tree.map(lambda v, x: jnp.asarray(v, x.dtype), lora_values, lora)

    def loss_fn(lora, frozen, micro):
        logits = forward_logits(graphdef, lora, frozen, micro)
        total, sums = loss_sums(logits, micro, brier_weight=BRIER)
        return total, (sums, logits)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    out: dict[str, Any] = {"batches": [], "lora": lora_values}
    for batch in batches:
        micro = jax.tree.map(lambda x: jnp.asarray(x[0]), batch)
        t0 = time.perf_counter()
        with precision():
            (total, (sums, logits)), grads = grad_fn(lora, frozen, micro)
        jax.block_until_ready(grads)
        first_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        with precision():
            for _ in range(3):
                res = grad_fn(lora, frozen, micro)
        jax.block_until_ready(res)
        out["batches"].append({
            "length": int(micro.tokens.shape[-1]),
            "rows": int(micro.tokens.shape[0]),
            "tokens": float(sums["tokens"]),
            "slots": float(sums["weight"]),
            "loss_sum": float(total),
            "logits": np.asarray(logits, np.float32),
            "label_mask": np.asarray(micro.label_mask),
            "grads": _named(grads),
            "compile_and_first_s": first_s,
            "fwd_bwd_s": (time.perf_counter() - t0) / 3,
        })  # fmt: skip
    del frozen, grad_fn
    jax.clear_caches()
    return out


def trace_layers(size, model_dir, mix, overrides, batch, lora_values) -> dict[str, np.ndarray]:
    """Forward only, unrolled (no layer scan, no remat), on one batch: the residual stream
    after every decoder layer (``layer<i>``), every sequence-mixer core output (``gdn<i>``,
    ``attn<i>``, in call order) and the label logits, fetched by host callbacks. Comparing
    two variants' traces locates the first layer and op where they part."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    import tjev.model.attention as attention_mod
    import tjev.model.gdn as gdn_mod
    from tjev.config import MeshSpec
    from tjev.model import build_model
    from tjev.sharding import configure_runtime, install_mesh, make_mesh
    from tjev.train.step import split_model

    cfg = _config(size, model_dir, mix, overrides)
    configure_runtime(cfg.compute)
    install_mesh(cfg.compute, make_mesh(MeshSpec(data=1), jax.devices()[:1]))
    config, model = build_model(
        str(model_dir), cfg.compute, cfg.lora, dtype=cfg.model.dtype, seed=cfg.train.seed
    )
    graphdef, lora, frozen = split_model(model)
    del model
    lora = jax.tree.map(lambda v, x: jnp.asarray(v, x.dtype), lora_values, lora)
    records: dict[str, np.ndarray] = {}
    count = {"gdn": 0, "attn": 0}

    def tap(name, x):
        def put(value, name=name):
            records[name] = np.asarray(value, np.float32)

        jax.debug.callback(put, x)

    def tapped(kind, fn):
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            tap(f"{kind}{count[kind]}", out)
            count[kind] += 1
            return out

        return wrapper

    def forward(lora, frozen, micro):
        m = nnx.merge(graphdef, lora, frozen)
        x = m.embed[...][micro.tokens]
        block_def, state = nnx.split(m.blocks)
        for s in range(config.num_super_blocks):
            block = nnx.merge(block_def, jax.tree.map(lambda a, s=s: a[s], state))
            for i in range(len(block.mlps)):
                x = block.layer(i, x, micro.segment_ids, micro.positions)
                tap(f"layer{s * len(block.mlps) + i}", x)
        tap("logits", m.label_logits(m.norm(x), micro.slots, micro.label_ids))

    saved = gdn_mod.gated_delta_rule, attention_mod.attention
    gdn_mod.gated_delta_rule = tapped("gdn", saved[0])
    attention_mod.attention = tapped("attn", saved[1])
    try:
        micro = jax.tree.map(lambda x: jnp.asarray(x[0]), batch)
        with jax.default_matmul_precision("highest" if HIGHEST in overrides else None):
            jax.jit(forward)(lora, frozen, micro)
            jax.effects_barrier()
    finally:
        gdn_mod.gated_delta_rule, attention_mod.attention = saved
    jax.clear_caches()
    return records


def trace_divergence(got: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> dict[str, float]:
    """Relative L2 difference per traced tensor, in forward order."""
    return {
        name: float(np.linalg.norm(got[name] - r) / max(float(np.linalg.norm(r)), 1e-30))
        for name, r in ref.items()
        if name in got
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a.ravel(), b.ravel()) / den) if den > 0 else float("nan")


def compare(got: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    per_batch, leaves = [], {}
    for g, r in zip(got["batches"], ref["batches"], strict=True):
        mask = r["label_mask"] & np.isfinite(r["logits"])
        dlogit = np.abs(np.where(mask, g["logits"] - r["logits"], 0.0))
        names = sorted(r["grads"])
        ga = np.concatenate([g["grads"][n].ravel() for n in names])
        ra = np.concatenate([r["grads"][n].ravel() for n in names])
        per_batch.append({
            "length": r["length"],
            "loss_per_slot": g["loss_sum"] / max(g.get("slots", 1.0), 1.0),
            "ref_loss_per_slot": r["loss_sum"] / max(r.get("slots", 1.0), 1.0),
            "loss_rel": abs(g["loss_sum"] - r["loss_sum"]) / max(abs(r["loss_sum"]), 1e-30),
            "logit_max_abs": float(dlogit.max()),
            "grad_rel_l2": float(np.linalg.norm(ga - ra) / max(np.linalg.norm(ra), 1e-30)),
            "grad_cosine": _cosine(ga, ra),
            "grad_finite": bool(np.all(np.isfinite(ga))),
            "fwd_bwd_s": g["fwd_bwd_s"],
        })  # fmt: skip
        for n in names:
            a, b = g["grads"][n], r["grads"][n]
            norm = float(np.linalg.norm(b))
            if norm == 0:
                continue
            rel = float(np.linalg.norm(a - b) / norm)
            worst = leaves.setdefault(n, {"leaf": n, "rel_l2": 0.0, "cosine": 1.0})
            worst["rel_l2"] = max(worst["rel_l2"], rel)
            worst["cosine"] = min(worst["cosine"], _cosine(a, b))
    worst_leaves = sorted(leaves.values(), key=lambda x: -x["rel_l2"])
    return {
        "loss_rel": max(b["loss_rel"] for b in per_batch),
        "logit_max_abs": max(b["logit_max_abs"] for b in per_batch),
        "grad_rel_l2": max(b["grad_rel_l2"] for b in per_batch),
        "grad_cosine": min(b["grad_cosine"] for b in per_batch),
        "grad_finite": all(b["grad_finite"] for b in per_batch),
        "worst_leaf_cosine": min((x["cosine"] for x in worst_leaves), default=1.0),
        "fwd_bwd_s": sum(b["fwd_bwd_s"] for b in per_batch),
        "batches": per_batch,
        "worst_leaves": worst_leaves[:20],
    }


def verdict(results: dict[str, dict[str, Any]]) -> None:
    """Add ``pass`` (and the bound used) to every variant of one size, in place."""
    base = results.get("xla", {}).get("grad_rel_l2")
    bound = max(2 * base, GRAD_FLOOR) if base is not None else GRAD_FLOOR
    for r in results.values():
        if "error" in r:
            r["pass"] = False
            continue
        r["grad_bound"] = bound
        r["pass"] = bool(
            r["grad_finite"]
            and r["grad_rel_l2"] <= bound
            and r["grad_cosine"] >= 0.9995
            and r.get("worst_leaf_cosine", 1.0) >= 0.998
            and r["loss_rel"] <= 5e-3
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tjev campaign parity")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--sizes", default="0.8B,9B")
    parser.add_argument("--batches", type=int, default=3, help="at least one per bucket")
    parser.add_argument(
        "--lengths", default="", help="only these sequence buckets (comma list, e.g. 4096)"
    )
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--save-grads", type=Path, help="NPZ of every variant's gradients")
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float32"), help="override model.dtype everywhere"
    )
    parser.add_argument(
        "--trace-layers", action="store_true",
        help="also per-layer / per-op forward divergence on the first batch (trace_layers)",
    )  # fmt: skip
    parser.add_argument("--out", type=Path, default=Path("parity.json"))
    args = parser.parse_args(argv)
    import jax

    from tjev.campaign.experiments import Experiment

    names = args.variants.split(",")
    if unknown := set(names) - set(VARIANTS):
        parser.error(f"unknown variants {sorted(unknown)}; known: {', '.join(VARIANTS)}")
    dtype = [f"model.dtype={args.model_dtype}"] if args.model_dtype else []
    reference = REFERENCE + dtype
    exp = Experiment("parity", f"parity-{hardware()}", {
        "device": jax.devices()[0].device_kind, "jax": jax.__version__, "sizes": args.sizes,
        "reference": reference, "variants": {n: VARIANTS[n] + dtype for n in names},
        "lora_b_std": LORA_B_STD, "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
    })  # fmt: skip
    report: dict[str, Any] = {"device": jax.devices()[0].device_kind, "sizes": {}}
    for size in args.sizes.split(","):
        folder = args.models / f"Qwen3.5-{size}"
        if not folder.exists():
            continue
        cfg = _config(size, folder, args.mix, reference)
        lengths = tuple(int(x) for x in args.lengths.split(",") if x)
        batches = real_batches(cfg, args.batches, lengths)
        print(f"parity {size}: {[b.tokens.shape for b in batches]}", flush=True)
        ref = run_variant(size, folder, args.mix, reference, batches)
        ref_trace = (
            trace_layers(size, folder, args.mix, reference, batches[0], ref["lora"])
            if args.trace_layers
            else {}
        )
        results: dict[str, Any] = {}
        saved = {f"reference/{n}": v for n, v in ref["batches"][0]["grads"].items()}
        for name in names:
            try:
                overrides = VARIANTS[name] + dtype
                with variant_env(name):
                    got = run_variant(size, folder, args.mix, overrides, batches, ref["lora"])
                results[name] = compare(got, ref)
                saved |= {f"{name}/{n}": v for n, v in got["batches"][0]["grads"].items()}
                if args.trace_layers:
                    with variant_env(name):
                        trace = trace_layers(size, folder, args.mix, overrides, batches[0],
                                             ref["lora"])  # fmt: skip
                    diverged = trace_divergence(trace, ref_trace)
                    results[name]["trace"] = [
                        {"tensor": k, "rel_l2": v} for k, v in diverged.items()
                    ]
                    print(f"trace {size} {name}: " + " ".join(
                        f"{k}={v:.1e}" for k, v in diverged.items()), flush=True)  # fmt: skip
            except Exception as e:
                results[name] = {"error": f"{type(e).__name__}: {e}"[:500]}
                traceback.print_exc()
            summary = {k: v for k, v in results[name].items() if not isinstance(v, list)}
            print(f"parity {size} {name}: {json.dumps(summary)}", flush=True)
            # logged as soon as it is measured: a run stopped later keeps it
            exp.summary({f"{size}/{name}/{k}": v for k, v in summary.items()
                         if isinstance(v, int | float | bool)})  # fmt: skip
        verdict(results)
        report["sizes"][size] = {
            "reference_fwd_bwd_s": sum(b["fwd_bwd_s"] for b in ref["batches"]),
            "batches": [
                {k: b[k] for k in ("length", "rows", "tokens", "slots", "loss_sum")}
                for b in ref["batches"]
            ],
            "variants": results,
        }
        if args.save_grads:
            args.save_grads.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save_grads / f"grads-{size}.npz", **saved)
        exp.table(f"{size}/variants", [
            {"variant": n, **{k: v for k, v in r.items() if not isinstance(v, list)}}
            for n, r in results.items()
        ])  # fmt: skip
        exp.table(f"{size}/worst_leaves", [
            {"variant": n, **leaf} for n, r in results.items() for leaf in r.get("worst_leaves", [])
        ])  # fmt: skip
        exp.summary({f"{size}/{n}/{k}": r[k] for n, r in results.items()
                     for k in ("pass", "grad_rel_l2", "grad_cosine", "loss_rel", "fwd_bwd_s")
                     if k in r})  # fmt: skip
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    exp.artifact("parity", "parity", [args.out])
    exp.finish()
    failed = [f"{s}/{n}" for s, r in report["sizes"].items()
              for n, v in r["variants"].items() if not v["pass"]]  # fmt: skip
    print(json.dumps({"parity_failed": failed}), flush=True)
    return 1 if failed else 0
