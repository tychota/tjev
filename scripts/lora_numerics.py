"""The LoRA forms against the fp32 unmerged reference: loss, gradient rel L2 and cosine.

    python scripts/lora_numerics.py [--model models/Qwen3.5-0.8B --layers 4] [--rank 32]

On real weights (``--model``, truncated to the first ``--layers`` decoder layers so the 9B
fits a CPU host) or, without one, on a random realistic-width fixture (hidden 1024, head
dims 128, 4 layers, Qwen-like projection scale). One packed row of ``--tokens`` tokens in
4 segments plus padding (real text when the model ships a tokenizer), next-token
cross-entropy over the vocabulary. One LoRA state for every path: A as initialised, B
~ N(0, σ_B) with σ_B = 2e-4 (rank 32 rsLoRA: each projection moves by ~1%, a trained
adapter's order; campaign/parity.py). Every path runs the same model (XLA attention,
chunked GDN, full remat); only ``compute.lora_impl`` and the model dtype change.
Reported per path: loss rel, gradient rel L2 and cosine over all LoRA leaves, the worst
leaf's.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx

from tjev.config import ComputeSpec, LoRASpec
from tjev.model import Qwen35, load_hf, stack_layers

TEXT = (
    "The committee reviewed the refund policy and decided that customers who return an item "
    "within thirty days receive the full price, while later returns earn store credit only. "
    "Exceptions apply to damaged goods, which are refunded at any time with a receipt. "
)
PATHS = ("merged:float32", "unmerged:bfloat16", "merged:bfloat16")


def fixture(layers: int):
    from tjev.testing import make_tiny_snapshot

    folder = Path(tempfile.mkdtemp()) / "model"
    make_tiny_snapshot(folder, hidden_size=1024, intermediate_size=3584, num_layers=layers,
                       num_heads=8, num_kv_heads=2, head_dim=128, linear_num_key_heads=4,
                       linear_num_value_heads=8, linear_key_head_dim=128,
                       linear_value_head_dim=128)  # fmt: skip
    config, tensors = load_hf(folder, "float32")
    for k, v in tensors.items():  # Qwen-like projection scale (~0.02), bf16-exact base
        if v.ndim == 2 and "embed" not in k and "norm" not in k:
            v = v * 0.4
        tensors[k] = v.astype(ml_dtypes.bfloat16).astype(np.float32)
    return config, tensors, None


def real(folder: str, layers: int):
    config, tensors = load_hf(folder, "bfloat16")
    keep = {}
    for k, v in tensors.items():
        if k.startswith("layers.") and int(k.split(".")[1]) >= layers:
            continue
        keep[k] = np.asarray(v, np.float32)
    config = dataclasses.replace(config, num_layers=layers)
    tokenizer = None
    if (Path(folder) / "tokenizer.json").exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(Path(folder) / "tokenizer.json"))
    return config, keep, tokenizer


def batch(tokens: int, vocab: int, tokenizer, seed: int = 0):
    rng = np.random.default_rng(seed)
    pad = tokens // 16
    lens = np.diff(np.linspace(0, tokens - pad, 5).astype(int))
    if tokenizer is not None:
        ids = np.asarray(tokenizer.encode(TEXT * (tokens // 30 + 1)).ids[:tokens])
    else:
        ids = rng.integers(0, vocab, tokens)
    seg = np.concatenate([np.full(n, i + 1) for i, n in enumerate(lens)] + [np.zeros(pad)])
    pos = np.concatenate([np.arange(n) for n in lens] + [np.zeros(pad)])
    return tuple(jnp.asarray(a[None], jnp.int32) for a in (ids, seg, pos))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="a Qwen3.5 snapshot (default: the random fixture)")
    parser.add_argument("--layers", type=int, default=4, help="decoder layers kept (×4)")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--sigma-b", type=float, default=2e-4)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--paths", default=",".join(PATHS), help="IMPL:DTYPE,…")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    config, tensors, tok = real(args.model, args.layers) if args.model else fixture(args.layers)
    ids, seg, pos = batch(args.tokens, config.vocab_size, tok)
    lora_spec = LoRASpec(rank=args.rank, alpha=32.0)
    state = {}

    def run(impl: str, dtype: str):
        model = Qwen35(stack_layers(dict(tensors), config), config,
                       ComputeSpec(gdn_chunk=64, lora_impl=impl), lora_spec,
                       dtype=getattr(jnp, dtype), rngs=nnx.Rngs(0))  # fmt: skip
        graphdef, lora, frozen = nnx.split(model, nnx.LoRAParam, ...)
        if "lora" not in state:  # A as initialised, B ~ N(0, σ_B), the same for every path
            flat, treedef = jax.tree_util.tree_flatten_with_path(lora)
            state["lora"] = jax.tree_util.tree_unflatten(treedef, [
                args.sigma_b * jax.random.normal(jax.random.fold_in(jax.random.key(3), i), x.shape)
                if "lora_b" in jax.tree_util.keystr(p) else x for i, (p, x) in enumerate(flat)
            ])  # fmt: skip

        def loss(lo, frozen):  # frozen as an argument: not baked into the program
            m = nnx.merge(graphdef, lo, frozen)
            logits = m.logits(m.hidden(ids, seg, pos))[:, :-1].astype(jnp.float32)
            valid = (seg[:, 1:] == seg[:, :-1]) & (seg[:, 1:] > 0)
            nll = -jnp.take_along_axis(jax.nn.log_softmax(logits), ids[:, 1:, None], -1)[..., 0]
            return jnp.sum(nll * valid) / jnp.sum(valid)

        t0 = time.time()
        value, grads = jax.jit(jax.value_and_grad(loss))(state["lora"], frozen)
        leaves = [np.asarray(x, np.float64).ravel() for x in jax.tree.leaves(grads)]
        return float(value), leaves, time.time() - t0

    ref_loss, ref, sec = run("unmerged", "float32")
    flat_ref = np.concatenate(ref)
    print(f"reference (unmerged fp32) loss {ref_loss:.6f} ({sec:.0f}s)", flush=True)
    rows = []
    for item in args.paths.split(","):
        impl, dtype = item.split(":")
        value, leaves, sec = run(impl, dtype)
        flat = np.concatenate(leaves)
        cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))  # noqa: E731
        row = {
            "path": f"{dtype} {impl}", "loss_rel": abs(value - ref_loss) / abs(ref_loss),
            "grad_rel_l2": float(np.linalg.norm(flat - flat_ref) / np.linalg.norm(flat_ref)),
            "cosine": cos(flat, flat_ref),
            "worst_leaf_rel_l2": max(float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
                                     for a, b in zip(leaves, ref, strict=True)),
            "worst_leaf_cosine": min(cos(a, b) for a, b in zip(leaves, ref, strict=True)),
            "seconds": sec,
        }  # fmt: skip
        rows.append(row)
        print(f"{row['path']}: loss rel {row['loss_rel']:.2e}  grad rel L2 {row['grad_rel_l2']:.2e}"
              f"  cosine {row['cosine']:.6f}  worst leaf {row['worst_leaf_rel_l2']:.2e} /"
              f" {row['worst_leaf_cosine']:.5f}  ({sec:.0f}s)", flush=True)  # fmt: skip
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "reference_loss": ref_loss,
                                              "paths": rows}, indent=2))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
