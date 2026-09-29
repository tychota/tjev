"""Compare tjev's Qwen3.5 against Hugging Face transformers on a real checkpoint.

    uv run python scripts/check_parity.py models/Qwen3.5-0.8B [--dtype bfloat16]
    python scripts/check_parity.py models/Qwen3.5-2B --attention splash --gdn-impl pallas_tpu

HF runs on CPU (fp32, or bf16 for >= 4B to fit RAM); tjev on the default JAX device, with
all prompts packed in one row (segment isolation on real weights). Reports max |Δlogit|,
top-1 / top-10 agreement and KL(HF || tjev) per prompt. Needs the ``parity`` extra.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

PROMPTS = [
    "The capital of France is Paris. The capital of Germany is",
    "Politique de remboursement : un client peut obtenir un remboursement sous 30 jours si",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
]


def compare(want: np.ndarray, got: np.ndarray) -> dict[str, float]:
    lp_w = want - np.logaddexp.reduce(want, axis=-1, keepdims=True)
    lp_g = got - np.logaddexp.reduce(got, axis=-1, keepdims=True)
    kl = np.sum(np.exp(lp_w) * (lp_w - lp_g), axis=-1)
    top10 = [
        len(set(np.argsort(-want[i])[:10]) & set(np.argsort(-got[i])[:10])) / 10
        for i in range(len(want))
    ]
    return {
        "tokens": len(want),
        "max_abs_logit_diff": float(np.max(np.abs(want - got))),
        "top1_agreement": float(np.mean(want.argmax(-1) == got.argmax(-1))),
        "top10_overlap": float(np.mean(top10)),
        "kl_max": float(kl.max()),
        "kl_mean": float(kl.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--hf-dtype", default="float32", choices=["bfloat16", "float32"])
    parser.add_argument("--attention", default="xla", choices=["xla", "splash"])
    parser.add_argument(
        "--gdn-impl", default="chunked", choices=["chunked", "pallas_tpu", "pallas_tpu_split"]
    )
    args = parser.parse_args()

    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    from tjev.config import ComputeSpec
    from tjev.model import build_model

    tok = Tokenizer.from_file(f"{args.folder}/tokenizer.json")
    ids = [tok.encode(p, add_special_tokens=False).ids for p in PROMPTS]
    tokens = np.concatenate([np.asarray(x) for x in ids])[None]
    seg = np.concatenate([np.full(len(x), i + 1) for i, x in enumerate(ids)])[None]
    pos = np.concatenate([np.arange(len(x)) for x in ids])[None]

    t0 = time.time()
    hf = AutoModelForCausalLM.from_pretrained(args.folder, dtype=getattr(torch, args.hf_dtype))
    hf.eval()
    with torch.no_grad():
        want = [hf(torch.tensor([x])).logits[0].float().numpy() for x in ids]
    del hf
    t_hf = time.time() - t0

    t0 = time.time()
    compute = ComputeSpec(remat="none", attention=args.attention, gdn_impl=args.gdn_impl)
    _, model = build_model(args.folder, compute, dtype=args.dtype)
    hidden = jax.jit(lambda m, t, s, p: m.hidden(t, s, p))(
        model, jnp.asarray(tokens), jnp.asarray(seg), jnp.asarray(pos)
    )
    got = np.asarray(model.logits(hidden))[0]
    t_tjev = time.time() - t0

    starts = np.cumsum([0, *map(len, ids)])
    report = {
        "folder": args.folder,
        "dtype": args.dtype,
        "compute": {"attention": args.attention, "gdn_impl": args.gdn_impl},
        "device": str(jax.devices()[0]),
        "seconds_hf": round(t_hf, 1),
        "seconds_tjev": round(t_tjev, 1),
        "prompts": [
            compare(w, got[a:b]) for w, a, b in zip(want, starts[:-1], starts[1:], strict=True)
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
