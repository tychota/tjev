"""Compile the train step for a TPU topology on a CPU host (no TPU, no weights) and inspect it.

    pip install 'jax[tpu]'   # libtpu: its compile-only client needs no TPU attached
    JAX_PLATFORMS='' python scripts/aot_tpu.py RUN_CONFIG.json MODEL_DIR \\
        --bucket 2048 --rows 1 [key=value overrides ...] [--dump out/]

RUN_CONFIG.json: a run's saved ``config.json`` (``{"config": ...}``) or a bare RunConfig
dict; MODEL_DIR holds the HF ``config.json`` only. The step is the run's own
(:func:`tjev.train.step.make_train_step`) at one bucket, one microbatch of ``--rows`` rows
(``--micro`` microbatches), compiled by libtpu for ``--topology`` (MaxText
``train_compile``). Reports XLA's memory analysis, its cost analysis, and the layout copies
and transposes of the optimised HLO (count and bytes, by named scope): what layout
assignment did, which the CPU backend cannot show. Timing still needs the chip.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

BYTES = {"bf16": 2, "f32": 4, "s32": 4, "u32": 4, "pred": 1, "s8": 1, "u8": 1, "f16": 2,
         "s64": 8, "f64": 8}  # fmt: skip
SHAPE = re.compile(r"\b(bf16|f32|s32|u32|pred|s8|u8|f16|s64|f64)\[([\d,]*)\](\{[^}]*\})?")
INSTR = re.compile(r"^\s*(?:ROOT )?%(\S+) = (.+?) (\w[\w-]*)\((.*)$")
SCOPES = ("gdn_out", "gdn_proj", "gdn_core", "gdn_rule", "attn_proj", "attn_out",
          "attention_core", "mlp", "norm", "lora", "readout", "embed", "optimizer")  # fmt: skip


def nbytes(text: str) -> int:
    total = 0
    for dtype, dims, _ in SHAPE.findall(text):
        n = int(np.prod([int(d) for d in dims.split(",") if d])) if dims else 1
        total += n * BYTES[dtype]
    return total


def scope_of(line: str) -> str:
    m = re.search(r'op_name="([^"]*)"', line)
    name = m.group(1) if m else ""
    for s in SCOPES:
        if f"/{s}/" in name or name.endswith(f"/{s}"):
            return s
    return "other" if name else "?"


def layout_ops(hlo: str) -> list[dict]:
    """Copies and transposes that move data (not copy-start/-done, the async memory-space
    moves), at the top level or as the root of a fusion, with their output bytes."""
    fused_roots: dict[str, str] = {}
    current, comps = None, {}
    for line in hlo.splitlines():
        head = re.match(r"^(?:ENTRY )?%(\S+) .*\{$", line)
        if head:
            current = head.group(1)
            continue
        if current and line.strip().startswith("ROOT"):
            m = INSTR.match(line)
            if m:
                fused_roots[current] = m.group(3)
        if current:
            comps.setdefault(current, []).append(line)
    called = set()
    for line in hlo.splitlines():
        for c in re.findall(r"calls=%([\w.\-]+)", line):
            called.add(c)
    lines_by_name = {}
    for lines in comps.values():
        for line in lines:
            m = INSTR.match(line)
            if m:
                lines_by_name[m.group(1)] = line

    def scope(line: str, depth: int = 0) -> str:
        """The named scope of an op, or of its first operand's (layout assignment's copies
        carry no metadata)."""
        found = scope_of(line)
        if found != "?" or depth > 6:
            return found
        m = INSTR.match(line)
        operand = re.search(r"%([\w.\-]+)", m.group(4)) if m else None
        source = lines_by_name.get(operand.group(1)) if operand else None
        return scope(source, depth + 1) if source else found

    out = []
    for comp, lines in comps.items():
        # fusion bodies: their ops are counted through the fusion's root
        if comp in called and ("fused" in comp or "fusion" in comp):
            continue
        for line in lines:
            m = INSTR.match(line)
            if not m:
                continue
            name, shape, op, rest = m.groups()
            kind = None
            if op in ("copy", "transpose"):
                kind = op
            elif op == "fusion":
                callee = re.search(r"calls=%([\w.\-]+)", rest)
                root = fused_roots.get(callee.group(1)) if callee else None
                if root in ("copy", "transpose"):
                    kind = f"fusion:{root}"
            if kind:
                out.append({"name": name, "kind": kind, "shape": shape.strip(),
                            "bytes": nbytes(shape), "scope": scope(line)})  # fmt: skip
    return out


def kernel_calls(hlo: str) -> dict[str, int]:
    """The Pallas (Mosaic) custom calls of the optimised HLO by kernel name, as written in
    the program (a call in the layer scan's body counts once): e.g. how many GDN forward
    kernels the backward still recomputes."""
    counts: collections.Counter = collections.Counter()
    for line in hlo.splitlines():
        if "custom-call(" not in line or "tpu_custom_call" not in line:
            continue
        # the instruction is named after the kernel: %gdn_mega2_fwd.271 = (...) custom-call(
        m = re.match(r"^\s*(?:ROOT )?%([\w\-]+?)(?:\.\d+)* = ", line)
        counts[m.group(1) if m else "?"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_config")
    p.add_argument("model_dir")
    p.add_argument("overrides", nargs="*")
    p.add_argument("--topology", default="v6e:1x1")
    p.add_argument("--bucket", type=int, default=2048)
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--micro", type=int, default=1)
    p.add_argument("--dump", default="")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from tjev.config.loader import apply_override, from_dict
    from tjev.sharding import BATCH_AXES, install_mesh
    from tjev.train.compile_check import abstract_batch, abstract_model
    from tjev.train.optim import make_optimizer
    from tjev.train.step import make_train_step, named, split_model, train_step_name

    raw = json.loads(Path(args.run_config).read_text())
    raw = raw.get("config", raw)
    raw["model"]["path"] = args.model_dir
    for item in args.overrides:
        apply_override(raw, item)
    cfg = from_dict(raw)
    topo = topologies.get_topology_desc(
        platform="tpu", topology_name=args.topology, chip_config_name="default",
        chips_per_host_bounds=(1, 1, 1), num_slices=1, wrap=(False, False, False),
    )  # fmt: skip
    devices = np.asarray(topo.devices)
    mesh = Mesh(devices.reshape(devices.size, 1), BATCH_AXES)
    if devices.size > 1:
        install_mesh(cfg.compute, mesh)
    graphdef, lora, frozen = split_model(abstract_model(cfg))
    tx = make_optimizer(cfg.optim, cfg.train.steps, cfg.loss)
    opt_state = jax.eval_shape(tx.init, lora)
    step = make_train_step(graphdef, tx, brier_weight=cfg.train.brier_weight,
                           slots_per_step=1.0, loss=cfg.loss, dropout=cfg.lora.dropout)  # fmt: skip
    rep = NamedSharding(mesh, P())

    def place(tree, sharding):
        return jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), tree
        )

    batch = abstract_batch(args.micro, args.rows, args.bucket, cfg.train.max_segments,
                           cfg.train.max_labels)  # fmt: skip
    batch = place(batch, NamedSharding(mesh, P(None, BATCH_AXES)))
    # named like the training program of this bucket (train_step_T…_mb…); its remat is
    # the bucket's (compute.remat_by_bucket, else compute.remat)
    lowered = jax.jit(named(step, train_step_name(batch)), donate_argnums=(0, 1)).lower(
        place(lora, rep), place(opt_state, rep), place(frozen, rep), batch
    )
    compiled = lowered.compile()
    hlo = compiled.as_text()
    mem = compiled.memory_analysis()
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    ops = layout_ops(hlo)
    by_scope: dict[str, list] = collections.defaultdict(lambda: [0, 0])
    for o in ops:
        by_scope[o["scope"]][0] += 1
        by_scope[o["scope"]][1] += o["bytes"]
    report = {
        "tag": args.tag,
        "overrides": args.overrides,
        "bucket": args.bucket,
        "rows": args.rows,
        "micro": args.micro,
        "remat": cfg.compute.remat_at(args.bucket),
        "memory_gib": {
            k: round(getattr(mem, k) / 2**30, 3)
            for k in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
                "generated_code_size_in_bytes",
            )
        },
        "cost": {k: cost.get(k) for k in ("flops", "bytes accessed", "transcendentals")},
        "kernel_calls": kernel_calls(hlo),
        "layout_ops": {"count": len(ops), "mib": round(sum(o["bytes"] for o in ops) / 2**20, 1)},
        "layout_ops_by_scope": {
            k: {"count": v[0], "mib": round(v[1] / 2**20, 1)} for k, v in sorted(by_scope.items())
        },
        "largest_layout_ops": sorted(ops, key=lambda o: -o["bytes"])[:25],
    }
    print(json.dumps(report, indent=1))
    if args.dump:
        out = Path(args.dump)
        out.mkdir(parents=True, exist_ok=True)
        tag = args.tag or "step"
        (out / f"{tag}.hlo.txt").write_text(hlo)
        (out / f"{tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    # eager constants (splash's mask tables) on the host; the kernels lower for Mosaic, not
    # the interpreter: they ask jax.default_backend(), here "tpu" (the shardings' devices)
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    jax.default_backend = lambda: "tpu"
    jnp.zeros(())
    main()
