"""GDN delta-rule precision (``compute.gdn_precision``) on real Qwen3.5 weights and text.

    JAX_PLATFORMS=cpu python scripts/gdn_precision_study.py models/Qwen3.5-0.8B \\
        --layers 0,1,2 --length 4096 --out gdn-precision-0.8b.json

The first GDN layers of a real snapshot (only their tensors are read), run in fp32 on
real text packed in one row (segments of different documents, boundaries inside chunks),
give each layer's real core inputs: the raw in-projection qkv, a, b (rounded to bf16, as
the bf16 model computes them) and the layer's A_log, dt_bias and conv weight. On those, the
sweep's GDN core (pallas_tpu_mega2, chunk 128, blocked inverse; Pallas interpret mode) at
``highest`` / ``high`` / ``mixed`` / ``bf16`` against the fp32 XLA reference (segmented
conv, chunked delta rule at ``highest``): the output, the loss Σ out·c (a fixed random
cotangent c, scaled like the output), and the gradients d(q), d(k), d(v) (the qkv
gradient's three parts), d(a), d(b) (the gate projections' inputs), d(A_log), d(dt_bias).
``high`` is emulated as bf16_3x and ``bf16`` casts its operands, as on the MXU (the CPU
ignores a dot's precision: tests/kernels/test_gdn_long.py, emulated_high). Errors: relative
L2, max over max, cosine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "kernels"))
jax.config.update("jax_default_matmul_precision", "highest")

TEXTS = ("docs/training.md", "docs/fr-data.md", "docs/kernels.md", "docs/fr-samples.md",
         "docs/calibration.md", "README.md")  # fmt: skip


def layer_tensors(folder: Path, layers: list[int]) -> dict[str, np.ndarray]:
    """embed_tokens and layers.0..max(layers) (fp32), read from the safetensors alone."""
    from safetensors import safe_open

    from tjev.model.weights import TEXT_PREFIXES, tensor_files

    last = max(layers)
    keep = ["embed_tokens."] + [f"layers.{i}." for i in range(last + 1)]
    files = tensor_files(folder)
    index = folder / "model.safetensors.index.json"
    if index.exists():  # only the shards holding these tensors (they alone may be present)
        names = {
            v for k, v in json.loads(index.read_text())["weight_map"].items()
            if any(k.startswith(p + w) for p in TEXT_PREFIXES for w in keep)
        }  # fmt: skip
        files = [f for f in files if f.name in names]
    out = {}
    for file in files:
        with safe_open(str(file), framework="numpy") as f:
            for name in f.keys():  # noqa: SIM118
                prefix = next((p for p in TEXT_PREFIXES if name.startswith(p)), None)
                if prefix is None or name.startswith(("model.visual.", "mtp.")):
                    continue
                key = name[len(prefix) :]
                if key.startswith(tuple(keep)):
                    out[key] = np.asarray(f.get_tensor(name), np.float32)
    return out


def packed_tokens(folder: Path, length: int) -> tuple[np.ndarray, np.ndarray]:
    """One row of ``length`` real tokens: the texts' starts as segments of uneven lengths
    (each boundary inside a chunk), no padding."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
    shares = np.array([0.31, 0.23, 0.19, 0.13, 0.09, 0.05])
    sizes = np.floor(shares * length).astype(int)
    sizes[-1] = length - sizes[:-1].sum()
    tokens, seg = [], []
    for i, (path, n) in enumerate(zip(TEXTS, sizes, strict=True)):
        ids = tok.encode((ROOT / path).read_text(encoding="utf-8")).ids[:n]
        assert len(ids) == n, (path, len(ids), n)
        tokens += ids
        seg += [i + 1] * n
    return np.asarray([tokens], np.int32), np.asarray([seg], np.int32)


def core_inputs(folder: Path, layers: list[int], length: int):
    """Per GDN layer: (qkv, a, b, A_log, dt_bias, conv weight [K, C]) and segment ids."""
    from flax import nnx

    from tjev.config import ComputeSpec
    from tjev.model.gdn import GatedDeltaNet
    from tjev.model.lora import LinearFactory
    from tjev.model.mlp import MLP
    from tjev.model.norms import RMSNorm
    from tjev.model.params import sub_weights
    from tjev.model.weights import read_config

    config = read_config(folder)
    w = layer_tensors(folder, layers)
    tokens, seg = packed_tokens(folder, length)
    compute = ComputeSpec(gdn_impl="chunked", gdn_precision="highest")
    linear = LinearFactory(None, jnp.float32, nnx.Rngs(0))
    seg_j = jnp.asarray(seg)
    x = jnp.asarray(w["embed_tokens.weight"])[jnp.asarray(tokens)]
    out = {}
    for i in range(max(layers) + 1):
        if (i + 1) % config.full_attention_interval == 0:
            raise ValueError(f"layer {i} is an attention layer: GDN layers only")
        lw = sub_weights(w, f"layers.{i}.")
        gdn = GatedDeltaNet(sub_weights(lw, "linear_attn."), config, compute, linear)
        h = RMSNorm(lw["input_layernorm.weight"], config.rms_norm_eps)(x)
        if i in layers:
            proj = [m(h) for m in (gdn.in_proj_qkv, gdn.in_proj_a, gdn.in_proj_b)]
            out[i] = (*proj, gdn.A_log[...], gdn.dt_bias[...], gdn.conv[...])
        x = x + gdn(h, seg_j)
        h = RMSNorm(lw["post_attention_layernorm.weight"], config.rms_norm_eps)(x)
        x = x + MLP(sub_weights(lw, "mlp."), linear)(h)
    return config, out, seg_j


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--layers", default="0,1,2")
    p.add_argument("--length", type=int, default=4096)
    p.add_argument("--precisions", default="highest,high,mixed,bf16")
    p.add_argument("--dtypes", default="bfloat16,float32")
    p.add_argument("--out", default="")
    args = p.parse_args()
    from test_gdn_long import emulated_high
    from test_gdn_mega import _mega, _reference

    folder = Path(args.model)
    layers = [int(x) for x in args.layers.split(",")]
    t0 = time.time()
    config, inputs, seg = core_inputs(folder, layers, args.length)
    hk, h = config.linear_num_key_heads, config.linear_num_value_heads
    d = config.linear_key_head_dim
    assert config.linear_value_head_dim == d
    print(f"inputs {time.time() - t0:.0f}s: Hk={hk} H={h} D={d} T={args.length}", flush=True)
    report: dict = {"model": folder.name, "length": args.length, "layers": {}}
    for (layer, (qkv, a, b, a_log, dt_bias, w)), dtype in (
        (item, dt) for item in inputs.items() for dt in args.dtypes.split(",")
    ):
        # bfloat16: the bf16 model's core inputs (the projections rounded to bf16; the
        # reference reads the same rounded values in fp32; the kernel's outputs and input
        # gradients in bf16). float32: the algorithm's own error, nothing rounded around it
        lo = jnp.dtype(dtype)
        xs = (qkv.astype(lo), a.astype(lo), b.astype(lo), a_log, dt_bias, w.astype(lo), seg)
        xs32 = tuple(x.astype(jnp.float32) if i < 6 else x for i, x in enumerate(xs))
        ref = jax.tree_util.Partial(_reference, hk=hk, h=h, d=d, conv=True)
        out_ref = jax.jit(ref)(*xs32)
        cot = jax.random.normal(jax.random.key(9), out_ref.shape, jnp.float32)
        cot = cot * jnp.std(out_ref) / jnp.std(cot)

        def grads(fn, xs, cot=cot):
            def loss(*diff):  # qkv, a, b, A_log, dt_bias
                return jnp.sum(fn(*diff, xs[5], xs[6]).astype(jnp.float32) * cot)

            value, g = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4)))(*xs[:5])
            out = jax.jit(fn)(*xs)
            gq, gk, gv = np.split(np.asarray(g[0], np.float64), [hk * d, 2 * hk * d], axis=-1)
            terms = {"out": out, "q": gq, "k": gk, "v": gv, "a": g[1], "b": g[2],
                     "A_log": g[3], "dt_bias": g[4]}  # fmt: skip
            return float(value), {k: np.asarray(v, np.float64) for k, v in terms.items()}

        want_loss, want = grads(ref, xs32)
        rows = {}
        for precision in args.precisions.split(","):
            t1 = time.time()
            fn = _mega(hk=hk, h=h, d=d, conv=True, chunk=128, inverse="blocked",
                       precision=precision)  # fmt: skip
            with emulated_high():
                got_loss, got = grads(fn, xs)
            row = {"loss": abs(got_loss - want_loss) / abs(want_loss)}
            for name, q in want.items():
                pv, qv = got[name].ravel(), q.ravel()
                row[name] = {
                    "l2": float(np.linalg.norm(pv - qv) / (np.linalg.norm(qv) + 1e-30)),
                    "max": float(np.max(np.abs(pv - qv)) / (np.max(np.abs(qv)) + 1e-30)),
                    "cos": float(pv @ qv / (np.linalg.norm(pv) * np.linalg.norm(qv) + 1e-30)),
                }
            rows[precision] = row
            print(f"layer {layer} {dtype} {precision} ({time.time() - t1:.0f}s): loss {row['loss']:.1e} "
                  + " ".join(f"{k} {v['l2']:.1e}/{v['cos']:.6f}" for k, v in row.items()
                             if k != "loss"), flush=True)  # fmt: skip
        report["layers"][f"{layer}/{dtype}"] = rows
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
