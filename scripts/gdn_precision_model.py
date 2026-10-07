"""``compute.gdn_precision`` end to end: the LoRA gradients of a whole real model.

    JAX_PLATFORMS=cpu python scripts/gdn_precision_model.py models/Qwen3.5-0.8B --length 2048

What training sees: a LoRA-adapted real snapshot (rank 32 on the default targets, B
seeded non-zero so every adapter has a gradient) on real text packed in one row; the loss
is the next-token cross-entropy at every ``--stride``-th position. The truth is the fp32
model on the XLA path at ``highest``. Against it: the bf16 model on the sweep's GDN core
(pallas_tpu_mega2, chunk 128, blocked inverse; Pallas interpret mode, ``high`` emulated as
bf16_3x) at each precision, and the bf16 model at ``highest`` (the bf16 model's own noise,
the floor). Reports the loss error, the LoRA gradient's relative L2 error and cosine, and
the worst projection group's (by target name).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "kernels"))
sys.path.insert(0, str(ROOT / "scripts"))
jax.config.update("jax_default_matmul_precision", "highest")


def run(folder, tokens, seg, pos, compute, dtype, stride):
    from flax import nnx

    from tjev.config import ComputeSpec, LoRASpec
    from tjev.model import build_model

    _, model = build_model(folder, ComputeSpec(**compute), LoRASpec(rank=32), dtype=dtype)
    g, lora, frozen = nnx.split(model, nnx.LoRAParam, ...)
    paths, leaves = zip(*jax.tree_util.tree_flatten_with_path(lora)[0], strict=True)
    treedef = jax.tree.structure(lora)
    offsets = [1e-4 * jax.random.normal(jax.random.fold_in(jax.random.key(7), i), x.shape)
               for i, x in enumerate(leaves)]  # fmt: skip
    lora = jax.tree.unflatten(
        treedef, [x + o.astype(x.dtype) for x, o in zip(leaves, offsets, strict=True)]
    )
    at = jnp.arange(0, tokens.shape[1] - 1, stride)
    same = seg[0, at + 1] == seg[0, at]

    def loss(lo):
        m = nnx.merge(g, lo, frozen)
        hidden = m.hidden(tokens, seg, pos)[:, at]
        logits = m.logits(hidden)[0]
        nll = -jnp.take_along_axis(jax.nn.log_softmax(logits), tokens[0, at + 1][:, None], 1)[:, 0]
        return jnp.sum(jnp.where(same, nll, 0.0)) / jnp.sum(same)

    value, grads = jax.jit(jax.value_and_grad(loss))(lora)
    names = [jax.tree_util.keystr(p) for p in paths]
    return float(value), names, [np.asarray(x, np.float64) for x in jax.tree.leaves(grads)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--length", type=int, default=2048)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--precisions", default="highest,high,mixed,bf16")
    p.add_argument("--out", default="")
    args = p.parse_args()
    from gdn_precision_study import packed_tokens
    from test_gdn_long import emulated_high

    folder = Path(args.model)
    tokens, seg = packed_tokens(folder, args.length)
    pos = np.concatenate([np.arange(n) for n in np.bincount(seg[0])[1:]])[None]
    tokens, seg, pos = (jnp.asarray(x) for x in (tokens, seg, pos))
    t0 = time.time()
    truth_loss, names, truth = run(folder, tokens, seg, pos,
                                   {"gdn_impl": "chunked", "gdn_precision": "highest"},
                                   "float32", args.stride)  # fmt: skip
    print(f"truth (fp32, XLA, highest) loss {truth_loss:.5f} ({time.time() - t0:.0f}s)", flush=True)
    flat_truth = np.concatenate([x.ravel() for x in truth])
    # the projection of each leaf: the key before lora_a / lora_b
    group = [(lambda keys: keys[-2] if len(keys) > 1 else n)(re.findall(r"'(\w+)'", n))
             for n in names]  # fmt: skip
    report = {"model": folder.name, "length": args.length, "truth_loss": truth_loss, "rows": {}}
    for precision in args.precisions.split(","):
        t1 = time.time()
        compute = {"gdn_impl": "pallas_tpu_mega2", "gdn_chunk": 128, "gdn_inverse": "blocked",
                   "gdn_precision": precision}  # fmt: skip
        with emulated_high():
            got_loss, _, got = run(folder, tokens, seg, pos, compute, "bfloat16", args.stride)
        flat = np.concatenate([x.ravel() for x in got])
        by_group = collections.defaultdict(lambda: [0.0, 0.0])
        for gname, a, b in zip(group, got, truth, strict=True):
            by_group[gname][0] += float(np.sum((a - b) ** 2))
            by_group[gname][1] += float(np.sum(b**2))
        groups = {k: (v[0] / v[1]) ** 0.5 for k, v in by_group.items()}
        row = {
            "loss": got_loss,
            "loss_err": abs(got_loss - truth_loss) / abs(truth_loss),
            "grad_l2": float(np.linalg.norm(flat - flat_truth) / np.linalg.norm(flat_truth)),
            "grad_cos": float(
                flat @ flat_truth / (np.linalg.norm(flat) * np.linalg.norm(flat_truth))
            ),
            "groups": groups,
        }
        report["rows"][precision] = row
        worst = max(groups, key=groups.get)
        print(f"bf16 model, mega2 {precision} ({time.time() - t1:.0f}s): loss err {row['loss_err']:.1e}, "
              f"LoRA grad L2 {row['grad_l2']:.2e} cos {row['grad_cos']:.6f}, worst group {worst} "
              f"{groups[worst]:.2e}", flush=True)  # fmt: skip
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
