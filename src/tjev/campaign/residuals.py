"""What one decoder layer saves for its backward, per remat policy: the scan residual stacks.

The forward scan over the super-blocks stacks each layer's saved residuals over the
super-blocks: XLA allocates every stack up front (an ``AllocateBuffer`` custom call, with no
op name: the ``custom-call.119 … 133`` at the top of the memory viewer) and the backward
scan reads it back. Their names exist only at trace time (``checkpoint_name``; the
``name`` primitive lowers to nothing), so this module lists them the way
``jax.ad_checkpoint.print_saved_residuals`` does, on an abstract model (no weights), and
``xprof_tools`` labels the stacks by shape:

    python -m tjev.campaign.residuals tpu-v6e qwen35-9b MIX/mix.yaml model.path=… \\
        train.microbatch_tokens=2048 --length 1024 [--out residuals.json]

Each entry: the layer (its kind and position in the super-block), the residual's shape and
dtype, its description (``named 'gdn_states'``, ``from the argument h``: the layer input,
``output of …``), its bytes and its stacked bytes (× the super-blocks).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _label(desc: str, ndim: int = 0) -> str:
    """A short label for a saved residual's description."""
    if m := re.search(r"named '([^']+)'", desc):
        return m[1]
    if "_splash_attention" in desc:  # its residuals, named inside the kernel's own jit
        return "attn_context" if ndim >= 5 else "attn_logsumexp"
    if "from the argument h" in desc or "argument 'h'" in desc:
        return "layer_input"
    if m := re.search(r"output of (\S+)", desc):
        return f"output of {m[1]}"
    return desc.split(" from ")[0]


def _saved(f, *args, **kwargs) -> list[tuple[Any, str]]:
    """(aval, description) per saved residual: the function behind
    ``jax.ad_checkpoint.print_saved_residuals`` (``jax._src.ad_checkpoint``, jax 0.11)."""
    from jax._src.ad_checkpoint import saved_residuals

    return saved_residuals(f, *args, **kwargs)


def layer_residuals(cfg: Any, length: int, rows: int | None = None) -> dict[str, Any]:
    """The saved residuals of each layer of one super-block at [rows, length] tokens per
    microbatch, under the bucket's remat policy (``cfg.compute.remat_at(length)``:
    compute.remat_by_bucket, else compute.remat; abstract: shapes only, any backend)."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from tjev.data.pack import rows_for
    from tjev.model.qwen35 import REMAT_POLICIES
    from tjev.sharding import configure_runtime
    from tjev.train.compile_check import abstract_model

    configure_runtime(cfg.compute)
    rows = rows or rows_for(length, cfg.train.microbatch_tokens, 1)
    model = abstract_model(cfg)
    graphdef, lora, frozen = nnx.split(model.blocks, nnx.LoRAParam, ...)
    one = lambda t: jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), t)  # noqa: E731
    lora, frozen = one(lora), one(frozen)
    supers = model.config.num_super_blocks
    seg = jnp.ones((rows, length), jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (rows, length))
    h = jax.ShapeDtypeStruct((rows, length, model.config.hidden_size), jnp.dtype(cfg.model.dtype))
    remat = cfg.compute.remat_at(length)
    layers = []
    for i in range(model.config.full_attention_interval):
        kind = f"layer{i}_gdn" if i < model.config.full_attention_interval - 1 else f"layer{i}_attn"

        def run(adapters, h, weights, i=i):
            block = nnx.merge(graphdef, adapters, jax.lax.stop_gradient(weights))
            fn = lambda h: block.layer(i, h, seg, pos, remat)  # noqa: E731
            if remat != "none":
                fn = jax.checkpoint(fn, policy=REMAT_POLICIES[remat], prevent_cse=False)
            return fn(h)

        saved = []
        for aval, desc in _saved(run, lora, h, weights=frozen):
            if (
                "argument adapters" in desc
                or "argument weights" in desc
                or "stop_gradient" in desc
                or "from a constant" in desc
            ):
                continue  # the weights and masks: the scan's own inputs, not new buffers
            shape = tuple(int(d) for d in aval.shape)
            nbytes = int(np.prod(shape, dtype=np.int64)) * aval.dtype.itemsize
            saved.append({
                "label": _label(desc, len(shape)), "shape": list(shape), "dtype": str(aval.dtype),
                "bytes": nbytes, "stacked_bytes": nbytes * supers, "description": desc[:200],
            })  # fmt: skip
        layers.append({"position": i, "kind": kind, "saved": saved,
                       "bytes": sum(s["bytes"] for s in saved)})  # fmt: skip
    return {
        "remat": remat,
        "rows": rows,
        "length": length,
        "super_blocks": supers,
        "layers": layers,
        "stacked_gib": round(sum(ly["bytes"] for ly in layers) * supers / 2**30, 3),
    }


def stack_labels(report: dict[str, Any]) -> dict[tuple[tuple[int, ...], str], list[str]]:
    """(stacked shape, dtype) → the residual labels with that stack shape: [S, *shape]."""
    out: dict[tuple[tuple[int, ...], str], list[str]] = {}
    for layer in report.get("layers", []):
        for s in layer["saved"]:
            key = ((report["super_blocks"], *s["shape"]), s["dtype"])
            out.setdefault(key, []).append(f"{layer['kind']}[{layer['position']}] {s['label']}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tjev.campaign.residuals")
    parser.add_argument("items", nargs="*", help="presets, mix.yaml and overrides (tjev train)")
    parser.add_argument("--length", type=int, action="append", help="bucket(s); default all")
    parser.add_argument(
        "--policies", default="", help="also these remat policies (comma list, or 'all')"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    from dataclasses import replace

    from tjev.config import load_config, split_args
    from tjev.model.qwen35 import REMAT_POLICIES

    files, overrides = split_args(args.items)
    cfg = load_config(*files, overrides=overrides)
    lengths = args.length or list(cfg.train.seq_buckets)
    out: dict[str, Any] = {"remat": cfg.compute.remat, "buckets": {}, "policies": {}}
    for length in lengths:
        report = layer_residuals(cfg, length)
        out["buckets"][str(length)] = report
        print(f"T={length} rows={report['rows']} remat={report['remat']}: "
              f"{report['stacked_gib']} GiB stacked")  # fmt: skip
        for layer in report["layers"]:
            for s in layer["saved"]:
                print(f"  {layer['kind']}[{layer['position']}] {s['label']:<24} {s['dtype']}"
                      f"{s['shape']}  {s['stacked_bytes'] / 2**20:8.1f} MiB stacked")  # fmt: skip
    policies = list(REMAT_POLICIES) if args.policies == "all" else [
        p for p in args.policies.split(",") if p]  # fmt: skip
    for policy in policies:  # what each remat policy would save (the same buckets)
        other = replace(cfg, compute=replace(cfg.compute, remat=policy, remat_by_bucket={}))
        out["policies"][policy] = {}
        for length in lengths:
            r = layer_residuals(other, length)
            out["policies"][policy][str(length)] = {
                "stacked_gib": r["stacked_gib"],
                "saved": sorted({s["label"] for ly in r["layers"] for s in ly["saved"]}),
            }
            print(f"policy {policy} T={length}: {r['stacked_gib']} GiB stacked")
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
