"""The ground truth: tjev against the official Qwen3.5 (HF transformers), forward and backward.

    python scripts/ground_truth.py MODEL --texts MIX/train/gen_long_refund_policy.jsonl \\
        --configs xla:chunked:highest:highest,xla:chunked:high --out ground-truth.json

The official model (fp32, on CPU) is the gold. On long real texts packed in one row (every
64-token GDN chunk boundary and every segment reset is crossed), both models compute the
same loss, next-token cross-entropy summed over every ``--stride``-th position of each
segment, and its gradient. HF runs each text on its own; tjev runs the packed row.

Compared, per tjev config (ATTENTION:GDN_IMPL:GDN_PRECISION[:highest], as check_parity.py):

- the loss, and the logits at the scored positions (max |Δ|, KL);
- the gradient of every parameter that maps one to one between the two models, per layer:
  the GDN gates (A_log, dt_bias) and output norm, the input and post-attention RMSNorms,
  the attention q/k norms, and the final norm. The gradient of layer 0 goes through the
  backward of every later layer, so this checks the whole backward pass.

HF in bf16 against HF in fp32 is also reported: the noise a bf16 model has anyway, the
bound a bf16 tjev may be held to. Needs the ``parity`` extra (torch, transformers).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path
from typing import Any, get_args, get_type_hints

import jax
import jax.numpy as jnp
import numpy as np

INTERVAL = 4  # 3 GDN layers then 1 attention layer per super-block


def texts(path: str, count: int, max_tokens: int, tok) -> list[list[int]]:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
    states = [r["state"] if isinstance(r["state"], str) else json.dumps(r["state"]) for r in rows]
    share = max_tokens // count  # each text gets its share, truncated if longer
    longest = sorted(states, key=len, reverse=True)[:count]
    return [tok.encode(text, add_special_tokens=False).ids[:share] for text in longest]


def positions(lengths: list[int], stride: int) -> list[np.ndarray]:
    """Scored positions of each segment: every stride-th one with a next token."""
    return [np.arange(0, n - 1, stride) for n in lengths]


def hf_name(key: str) -> str | None:
    """tjev frozen-state path → HF text-model parameter name (layers.N.…), or None."""
    m = re.fullmatch(r"\['blocks'\]\['gdn'\]\[(\d)\]\['(A_log|dt_bias)'\]\.value", key)
    if m:
        return f"layers.{{s{m[1]}}}.linear_attn.{m[2]}"
    m = re.fullmatch(r"\['blocks'\]\['gdn'\]\[(\d)\]\['norm'\]\['weight'\]\.value", key)
    if m:
        return f"layers.{{s{m[1]}}}.linear_attn.norm.weight"
    m = re.fullmatch(r"\['blocks'\]\['(input|post)_norms'\]\[(\d)\]\['weight'\]\.value", key)
    if m:
        name = "input_layernorm" if m[1] == "input" else "post_attention_layernorm"
        return f"layers.{{s{m[2]}}}.{name}.weight"
    m = re.fullmatch(r"\['blocks'\]\['attention'\]\['(q|k)_norm'\]\['weight'\]\.value", key)
    if m:
        return f"layers.{{s{INTERVAL - 1}}}.self_attn.{m[1]}_norm.weight"
    if key == "['norm']['weight'].value":
        return "norm.weight"
    return None


def unstack(key: str, value: np.ndarray) -> dict[str, np.ndarray]:
    """One stacked tjev leaf [S, …] → per-layer HF names."""
    name = hf_name(key)
    if name is None:
        return {}
    m = re.search(r"\{s(\d)\}", name)
    if m is None:
        return {name: value}
    j = int(m[1])
    return {name.replace(m[0], str(INTERVAL * s + j)): value[s] for s in range(value.shape[0])}


def run_hf(folder: str, ids: list[list[int]], scored: list[np.ndarray], dtype: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(folder, dtype=getattr(torch, dtype))
    hf.train(False)
    for p in hf.parameters():  # loaded frozen: the gold needs their gradients
        p.requires_grad_(True)
    params = {re.sub(r"^model\.(language_model\.)?", "", n): p for n, p in hf.named_parameters()}
    loss, logits = 0.0, []
    for x, pos in zip(ids, scored, strict=True):
        # the LM head only at the scored positions, and one backward per text (gradients
        # accumulate): the full-vocabulary logits and all graphs at once do not fit in RAM
        hidden = hf.model(torch.tensor([x])).last_hidden_state[0]
        sel = hf.lm_head(hidden[torch.as_tensor(pos)]).float()
        targets = torch.as_tensor(np.asarray(x)[pos + 1])
        seg_loss = torch.nn.functional.cross_entropy(sel, targets, reduction="sum")
        seg_loss.backward()
        loss += float(seg_loss.detach())
        logits.append(sel.detach().numpy())
        del hidden, sel, seg_loss
    grads = {n: p.grad.float().numpy() for n, p in params.items() if p.grad is not None}
    return {"loss": loss, "logits": np.concatenate(logits), "grads": grads}


def _key(ids, scored) -> str:
    """What the gold was computed on: the texts' tokens and the scored positions."""
    import hashlib

    h = hashlib.sha256()
    for x, p in zip(ids, scored, strict=True):
        h.update(np.asarray(x, np.int64).tobytes() + b"|" + np.asarray(p, np.int64).tobytes())
    return h.hexdigest()


def save_gold(path: str, ids, scored, gold: dict) -> None:
    grads = {f"grad/{k}": v for k, v in gold["grads"].items()}
    np.savez(path, key=_key(ids, scored), loss=gold["loss"], logits=gold["logits"], **grads)


def load_gold(path: str, ids, scored) -> dict | None:
    if not Path(path).exists():
        return None
    z = np.load(path)
    if str(z["key"]) != _key(ids, scored):
        raise SystemExit(f"{path}: computed on other texts or positions")
    grads = {k.removeprefix("grad/"): z[k] for k in z.files if k.startswith("grad/")}
    return {"loss": float(z["loss"]), "logits": z["logits"], "grads": grads}


def _compute(spec: str):
    """ATTENTION:GDN_IMPL:GDN_PRECISION[:highest][:block=N][:splash_block=N][:chunk=N]
    [:inverse=blocked][:heads_per_step=N][:lora=merged] → a ComputeSpec (full remat: without
    it XLA keeps every layer's intermediates for the backward; recomputing changes no
    number)."""
    from tjev.config import ComputeSpec

    attention, gdn_impl, precision, *extra = spec.split(":")
    fields: dict[str, Any] = {
        "remat": "full", "attention": attention, "gdn_impl": gdn_impl,
        "gdn_precision": precision,
    }  # fmt: skip
    for item in extra:  # block=N (XLA attention query block), splash_block=N, and the rest
        name, _, value = item.partition("=")
        if name in ("block", "splash_block"):
            fields["attention_" + name] = int(value)
        elif name in ("chunk", "heads_per_step"):  # gdn_chunk, gdn_heads_per_step
            fields["gdn_" + name] = int(value)
        elif name == "inverse":  # gdn_inverse
            fields["gdn_inverse"] = value
        elif name == "lora":  # lora_impl
            fields["lora_impl"] = value
        elif name not in ("highest", ""):
            raise SystemExit(f"{spec}: unknown option {item!r}")
    for name in ("attention", "gdn_impl", "gdn_precision", "gdn_inverse", "lora_impl"):
        if name not in fields:  # a typo fails loudly
            continue
        allowed = get_args(get_type_hints(ComputeSpec)[name])
        if fields[name] not in allowed:
            raise SystemExit(f"{spec}: {name} must be one of {allowed}")
    return ComputeSpec(**fields)


def run_tjev(folder: str, spec: str, dtype: str, ids, scored) -> dict:
    from tjev.model import build_model
    from tjev.train.step import split_model

    extra = spec.split(":")[3:]
    _, model = build_model(folder, _compute(spec), dtype=dtype)
    graphdef, lora, frozen = split_model(model)
    del model
    tokens = np.concatenate([np.asarray(x) for x in ids])[None]
    seg = np.concatenate([np.full(len(x), i + 1) for i, x in enumerate(ids)])[None]
    pos = np.concatenate([np.arange(len(x)) for x in ids])[None]
    starts = np.cumsum([0, *map(len, ids)])[:-1]
    flat = np.concatenate([a + p for a, p in zip(starts, scored, strict=True)])
    targets = tokens[0][flat + 1]

    from flax import nnx

    def loss_fn(frozen):
        m = nnx.merge(graphdef, lora, frozen)
        hidden = m.hidden(jnp.asarray(tokens), jnp.asarray(seg), jnp.asarray(pos))
        logits = m.logits(hidden[:, flat]).astype(jnp.float32)[0]  # [N, V]
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(jnp.take_along_axis(logp, jnp.asarray(targets)[:, None], 1)), logits

    t0 = time.time()
    with jax.default_matmul_precision("highest" if "highest" in extra else None):
        (loss, logits), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(frozen)
        jax.block_until_ready(grads)
    per_layer: dict[str, np.ndarray] = {}
    for path, g in jax.tree_util.tree_flatten_with_path(grads)[0]:
        per_layer |= unstack(jax.tree_util.keystr(path), np.asarray(g, np.float32))
    return {"loss": float(loss), "logits": np.asarray(logits), "grads": per_layer,
            "seconds": round(time.time() - t0, 1)}  # fmt: skip


def trace_hf(folder: str, ids: list[list[int]]) -> list[np.ndarray]:
    """The official residual stream, packed like tjev's row: entry i is the output of layer
    i (HF's hidden_states[i + 1]; the last one after the final norm)."""
    import torch
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)
    hf.train(False)
    per_text = []
    with torch.no_grad():
        for x in ids:
            hs = hf.model(torch.tensor([x]), output_hidden_states=True).hidden_states
            per_text.append([h[0].float().numpy() for h in hs[1:]])
    return [np.concatenate([t[i] for t in per_text]) for i in range(len(per_text[0]))]


def trace_tjev(folder: str, spec: str, dtype: str, ids) -> list[np.ndarray]:
    """tjev's residual stream after every layer (the last after the final norm), unrolled
    and fetched with jax.debug.callback taps (the jit + scan hides it otherwise)."""
    from flax import nnx

    from tjev.model import build_model

    compute = _compute(spec)
    _, model = build_model(folder, compute, dtype=dtype)
    tokens = jnp.asarray(np.concatenate([np.asarray(x) for x in ids])[None])
    seg = jnp.asarray(np.concatenate([np.full(len(x), i + 1) for i, x in enumerate(ids)])[None])
    pos = jnp.asarray(np.concatenate([np.arange(len(x)) for x in ids])[None])
    records: dict[int, np.ndarray] = {}

    def tap(i, x):
        jax.debug.callback(lambda v, i=i: records.__setitem__(i, np.asarray(v[0], np.float32)), x)

    def forward(m):
        x = m.embed[...][tokens]
        block_def, state = nnx.split(m.blocks)
        n = 0
        for s in range(m.config.num_super_blocks):
            block = nnx.merge(block_def, jax.tree.map(lambda a, s=s: a[s], state))
            for i in range(len(block.mlps)):
                x = block.layer(i, x, seg, pos)
                tap(n, x)
                n += 1
        tap(n, m.norm(x))

    extra = spec.split(":")[3:]
    with jax.default_matmul_precision("highest" if "highest" in extra else None):
        jax.jit(forward)(model)
        jax.effects_barrier()
    n = max(records)
    return [records[i] for i in range(n - 1)] + [records[n]]  # the last layer: normed


def trace_ops_hf(folder: str, ids: list[list[int]]) -> dict[str, np.ndarray]:
    """The first GDN layer's internals in the official model, packed like tjev's row: conv
    output (after SiLU), q and k as the rule uses them (l2-normed, q scaled), v, g, beta,
    the core output. Captured by wrapping HF's module-level GDN functions."""
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as hfm
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)
    hf.train(False)
    per_text: list[dict[str, np.ndarray]] = []
    conv_fn, rule_fn = hfm.causal_conv1d_fn, hfm.torch_chunk_gated_delta_rule

    def conv(x, *a, **kw):
        out = conv_fn(x, *a, **kw)
        if "conv" not in per_text[-1]:
            per_text[-1]["conv_in"] = x[0].transpose(0, 1).float().numpy()  # [T, C]
            per_text[-1]["conv"] = out[0].transpose(0, 1).float().numpy()
        return out

    def rule(q, k, v, *a, g=None, beta=None, **kw):
        out = rule_fn(q, k, v, *a, g=g, beta=beta, **kw)
        rec = per_text[-1]
        if "core" not in rec:
            assert g is not None, "HF passes g by keyword"
            assert beta is not None, "HF passes beta by keyword"
            l2 = lambda x: x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)  # noqa: E731
            rec["q"] = (l2(q.float()) * q.shape[-1] ** -0.5)[0].numpy()
            rec["k"] = l2(k.float())[0].numpy()
            rec["v"], rec["g"], rec["beta"] = (t[0].float().numpy() for t in (v, g, beta))
            rec["core"] = out[0][0].float().numpy()
        return out

    hfm.causal_conv1d_fn, hfm.torch_chunk_gated_delta_rule = conv, rule
    try:
        with torch.no_grad():
            for x in ids:
                per_text.append({})
                hf.model(torch.tensor([x]))
    finally:
        hfm.causal_conv1d_fn, hfm.torch_chunk_gated_delta_rule = conv_fn, rule_fn
    return {n: np.concatenate([t[n] for t in per_text]) for n in per_text[0]}


def trace_ops_tjev(folder: str, spec: str, dtype: str, ids) -> dict[str, np.ndarray]:
    """The same internals of tjev's first GDN layer (unrolled, jax.debug.callback taps)."""
    import tjev.model.gdn as gdn_mod

    records: dict[str, np.ndarray] = {}
    conv_fn, rule_fn = gdn_mod.segmented_causal_conv1d, gdn_mod.gated_delta_rule
    seen = {"conv": False, "rule": False}

    def tap(name, x):
        jax.debug.callback(
            lambda v, n=name: records.__setitem__(n, np.asarray(v[0], np.float32)), x
        )

    def conv(x, *a, **kw):
        out = conv_fn(x, *a, **kw)
        if not seen["conv"]:
            seen["conv"] = True
            tap("conv_in", x)
            tap("conv", jax.nn.silu(out))
        return out

    def rule(q, k, v, g, beta, *a, **kw):
        out = rule_fn(q, k, v, g, beta, *a, **kw)
        if not seen["rule"]:
            seen["rule"] = True
            for n, t in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("core", out)):
                tap(n, t)
        return out

    setattr(gdn_mod, "segmented_causal_conv1d", conv)  # noqa: B010  (a test-time wrapper)
    setattr(gdn_mod, "gated_delta_rule", rule)  # noqa: B010
    try:
        trace_tjev(folder, spec, dtype, ids)  # the unrolled forward; its taps are ignored
    finally:
        gdn_mod.segmented_causal_conv1d, gdn_mod.gated_delta_rule = conv_fn, rule_fn
    return records


def compare_ops(got: dict[str, np.ndarray], want: dict[str, np.ndarray], lengths) -> dict:
    """Per internal tensor: relative L2, and the token where it is worst."""
    starts = np.cumsum([0, *lengths])
    out = {}
    for name, w in want.items():
        g = got.get(name)
        if g is not None and w.ndim == 3 and g.ndim == 3 and w.shape[1] != g.shape[1]:
            w = w[:, :: w.shape[1] // g.shape[1]]  # HF repeats the key heads (GQA); tjev does not
        if g is None or g.shape != w.shape:
            out[name] = {"shape": [list(w.shape), None if g is None else list(g.shape)]}
            continue
        err = np.abs(g - w).reshape(len(w), -1).max(-1)
        worst = int(np.argmax(err))
        text = int(np.searchsorted(starts, worst, side="right") - 1)
        out[name] = {
            "rel_l2": float(np.linalg.norm(g - w) / max(float(np.linalg.norm(w)), 1e-30)),
            "max_abs": float(err[worst]),
            "worst_token": {"text": text, "position": worst - int(starts[text])},
        }
    return out


def compare_layers(got: list[np.ndarray], want: list[np.ndarray], lengths: list[int]) -> list[dict]:
    """Per layer: relative L2 over all tokens, per text, and the worst token."""
    starts = np.cumsum([0, *lengths])
    out = []
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        err = np.linalg.norm(g - w, axis=-1)  # per token
        ref = np.linalg.norm(w, axis=-1)
        worst = int(np.argmax(err / np.maximum(ref, 1e-30)))
        text = int(np.searchsorted(starts, worst, side="right") - 1)
        out.append({
            "layer": i,
            "rel_l2": float(np.linalg.norm(g - w) / np.linalg.norm(w)),
            "rel_l2_by_text": [float(np.linalg.norm(err[a:b]) / np.linalg.norm(ref[a:b]))
                               for a, b in itertools.pairwise(starts)],
            "worst_token": {"text": text, "position": worst - int(starts[text]),
                            "rel": float(err[worst] / max(ref[worst], 1e-30))},
        })  # fmt: skip
    return out


def _cos(a, b) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a.ravel(), b.ravel()) / den) if den else float("nan")


def compare(got: dict, gold: dict, where: list[tuple[int, int, int]] | None = None) -> dict:
    """``where``: (text, position, text length) of each scored row, for the per-position
    breakdown (which texts and how deep into them the logits go wrong)."""
    lp_g = got["logits"] - np.logaddexp.reduce(got["logits"], axis=-1, keepdims=True)
    lp_w = gold["logits"] - np.logaddexp.reduce(gold["logits"], axis=-1, keepdims=True)
    kl = np.sum(np.exp(lp_w) * (lp_w - lp_g), axis=-1)
    groups: dict[str, list[tuple[float, float]]] = {}
    per_param = {}
    for name, want in gold["grads"].items():
        if name not in got["grads"]:
            continue
        have = got["grads"][name]
        norm = float(np.linalg.norm(want))
        rel = float(np.linalg.norm(have - want) / norm) if norm else float("nan")
        per_param[name] = {"rel_l2": rel, "cosine": _cos(have, want)}
        group = re.sub(r"layers\.\d+\.", "layers.*.", name)
        groups.setdefault(group, []).append((rel, per_param[name]["cosine"]))
    diff = np.max(np.abs(got["logits"] - gold["logits"]), axis=-1)
    positions: dict[str, Any] = {}
    if where is not None:
        order = np.argsort(-kl)[:20]
        positions["worst"] = [
            {"text": where[i][0], "position": where[i][1], "length": where[i][2],
             "kl": float(kl[i]), "logit_max_abs": float(diff[i]),
             "top1_same": bool(got["logits"][i].argmax() == gold["logits"][i].argmax())}
            for i in order
        ]  # fmt: skip
        # by depth into the text (quarters): does the error grow along a sequence?
        depth = np.array([w[1] / max(w[2], 1) for w in where])
        positions["kl_by_depth_quarter"] = [
            float(kl[(depth >= q / 4) & (depth < (q + 1) / 4)].mean()) for q in range(4)
        ]
        positions["kl_by_text"] = [float(kl[[w[0] == t for w in where]].mean())
                                   for t in sorted({w[0] for w in where})]  # fmt: skip
    return {
        "positions": positions,
        "loss_rel": abs(got["loss"] - gold["loss"]) / abs(gold["loss"]),
        "logit_max_abs": float(np.max(np.abs(got["logits"] - gold["logits"]))),
        "kl_max": float(kl.max()),
        "top1_agreement": float(np.mean(got["logits"].argmax(-1) == gold["logits"].argmax(-1))),
        "grad_params": len(per_param),
        "grad_groups": {
            g: {"worst_rel_l2": max(r for r, _ in v), "worst_cosine": min(c for _, c in v)}
            for g, v in sorted(groups.items())
        },
        "grad_worst": sorted(
            ({"param": n} | v for n, v in per_param.items()), key=lambda x: -x["rel_l2"]
        )[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder")
    parser.add_argument("--texts", required=True, help="JSONL items: their `state` texts")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=8, help="score every stride-th position")
    parser.add_argument("--configs", default="xla:chunked:highest:highest,xla:chunked:high")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--hf-bf16", action="store_true", help="also HF in bf16 (the noise floor)")
    parser.add_argument(
        "--trace-layers",
        action="store_true",
        help="also the residual stream after every layer, vs HF's (forward)",
    )
    parser.add_argument(
        "--trace-ops",
        action="store_true",
        help="also the first GDN layer's internals (conv, q, k, v, g, beta, core)",
    )
    parser.add_argument("--out")
    parser.add_argument(
        "--gold-cache", help="NPZ of the HF fp32 gold: loaded if it exists, else written"
    )
    args = parser.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(f"{args.folder}/tokenizer.json")
    ids = texts(args.texts, args.count, args.max_tokens, tok)
    scored = positions([len(x) for x in ids], args.stride)
    where = [(t, int(p), len(ids[t])) for t, pos in enumerate(scored) for p in pos]
    print(
        json.dumps({"tokens": [len(x) for x in ids], "scored": int(sum(map(len, scored)))}),
        flush=True,
    )
    t0 = time.time()
    gold = load_gold(args.gold_cache, ids, scored) if args.gold_cache else None
    if gold is None:
        gold = run_hf(args.folder, ids, scored, "float32")
        if args.gold_cache:
            save_gold(args.gold_cache, ids, scored, gold)
    report: dict[str, Any] = {
        "folder": args.folder, "device": str(jax.devices()[0]), "tjev_dtype": args.dtype,
        "tokens": [len(x) for x in ids], "gold_loss": gold["loss"],
        "gold_seconds": round(time.time() - t0, 1), "configs": {},
    }  # fmt: skip
    if args.hf_bf16:
        report["hf_bf16_vs_fp32"] = compare(run_hf(args.folder, ids, scored, "bfloat16"), gold)
        print(json.dumps({"config": "hf-bf16"} | _brief(report["hf_bf16_vs_fp32"])), flush=True)
    hf_trace = trace_hf(args.folder, ids) if args.trace_layers else None
    hf_ops = trace_ops_hf(args.folder, ids) if args.trace_ops else None
    for spec in args.configs.split(","):
        got = run_tjev(args.folder, spec, args.dtype, ids, scored)
        report["configs"][spec] = compare(got, gold, where) | {"seconds": got["seconds"]}
        if hf_ops is not None:
            ops = compare_ops(trace_ops_tjev(args.folder, spec, args.dtype, ids), hf_ops,
                              [len(x) for x in ids])  # fmt: skip
            report["configs"][spec]["ops_layer0"] = ops
            print(json.dumps({"config": spec, "ops_layer0": {
                n: round(v["rel_l2"], 7) if "rel_l2" in v else v for n, v in ops.items()}}), flush=True)  # fmt: skip
        if hf_trace is not None:
            layers = compare_layers(trace_tjev(args.folder, spec, args.dtype, ids), hf_trace,
                                    [len(x) for x in ids])  # fmt: skip
            report["configs"][spec]["layers"] = layers
            print(
                json.dumps(
                    {"config": spec, "layer_rel_l2": [round(x["rel_l2"], 6) for x in layers]}
                ),
                flush=True,
            )
        print(json.dumps({"config": spec} | _brief(report["configs"][spec])), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _brief(r: dict) -> dict:
    worst = max(r["grad_groups"].values(), key=lambda g: g["worst_rel_l2"], default={})
    return {k: r[k] for k in ("loss_rel", "logit_max_abs", "kl_max", "top1_agreement", "grad_params")} | {
        "grad_worst_rel_l2": worst.get("worst_rel_l2"), "grad_worst_cosine": min(
            (g["worst_cosine"] for g in r["grad_groups"].values()), default=None)}  # fmt: skip


if __name__ == "__main__":
    main()
