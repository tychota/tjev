"""Compare tjev's Qwen3.5 against Hugging Face transformers on a real checkpoint.

    uv run python scripts/check_parity.py models/Qwen3.5-0.8B [--dtype bfloat16]
    python scripts/check_parity.py models/Qwen3.5-2B --attention splash --gdn-impl pallas_tpu
    python scripts/check_parity.py MODEL --texts MIX/train/gen_long_refund_policy.jsonl \
        --count 3 --max-tokens 8192 --configs xla:chunked:high,xla:chunked:highest:highest

HF runs on CPU (fp32, or bf16 for >= 4B to fit RAM); tjev on the default JAX device, with
all prompts packed in one row (segment isolation on real weights). ``--texts`` takes the
prompts from the ``state`` of real JSONL items instead (long ones cross many 64-token GDN
chunks: the inter-chunk state carry and the segment resets). ``--configs`` compares several
tjev compute configs against one HF run: ATTENTION:GDN_IMPL:GDN_PRECISION[:highest] (the
last field runs tjev under default matmul precision "highest": every f32 dot in fp32). Reports max |Δlogit|,
top-1 / top-10 agreement and KL(HF || tjev) per prompt. Needs the ``parity`` extra.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, get_args, get_type_hints

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
    parser.add_argument("--gdn-impl", default="chunked")
    parser.add_argument("--gdn-precision", default="high")
    parser.add_argument("--configs", help="comma list of ATTENTION:GDN_IMPL:PRECISION[:highest]")
    parser.add_argument("--texts", help="JSONL items: their `state` texts are the prompts")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--out", help="also write the report here")
    args = parser.parse_args()

    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    from tjev.config import ComputeSpec
    from tjev.model import build_model

    tok = Tokenizer.from_file(f"{args.folder}/tokenizer.json")
    prompts = PROMPTS
    if args.texts:
        lines = Path(args.texts).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        states = [
            r["state"] if isinstance(r["state"], str) else json.dumps(r["state"]) for r in rows
        ]
        prompts = sorted(states, key=len, reverse=True)[: args.count]  # the longest
    ids = [tok.encode(p, add_special_tokens=False).ids for p in prompts]
    budget, kept = args.max_tokens, []
    for x in ids:  # packed into one row of at most max_tokens
        if len(x) <= budget:
            kept.append(x)
            budget -= len(x)
    ids = kept
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

    configs = (args.configs or f"{args.attention}:{args.gdn_impl}:{args.gdn_precision}").split(",")
    starts = np.cumsum([0, *map(len, ids)])
    report: dict = {
        "folder": args.folder,
        "dtype": args.dtype,
        "device": str(jax.devices()[0]),
        "tokens": [len(x) for x in ids],
        "seconds_hf": round(t_hf, 1),
        "configs": {},
    }
    for spec in configs:
        attention, gdn_impl, precision, *extra = spec.split(":")
        t0 = time.time()
        fields: dict[str, Any] = {
            "remat": "none", "attention": attention, "gdn_impl": gdn_impl,
            "gdn_precision": precision,
        }  # fmt: skip
        for name in ("attention", "gdn_impl", "gdn_precision"):  # a typo fails loudly
            allowed = get_args(get_type_hints(ComputeSpec)[name])
            if fields[name] not in allowed:
                raise SystemExit(f"{spec}: {name} must be one of {allowed}")
        compute = ComputeSpec(**fields)
        _, model = build_model(args.folder, compute, dtype=args.dtype)
        with jax.default_matmul_precision("highest" if "highest" in extra else None):
            hidden = jax.jit(lambda m, t, s, p: m.hidden(t, s, p))(
                model, jnp.asarray(tokens), jnp.asarray(seg), jnp.asarray(pos)
            )
            got = np.asarray(model.logits(hidden))[0]
        per_prompt = [
            compare(w, got[a:b]) for w, a, b in zip(want, starts[:-1], starts[1:], strict=True)
        ]
        report["configs"][spec] = {
            "seconds": round(time.time() - t0, 1),
            "max_abs_logit_diff": max(p["max_abs_logit_diff"] for p in per_prompt),
            "top1_agreement": min(p["top1_agreement"] for p in per_prompt),
            "prompts": per_prompt,
        }
        del model
        print(json.dumps({"config": spec} | {k: v for k, v in report["configs"][spec].items()
                                              if k != "prompts"}), flush=True)  # fmt: skip
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
