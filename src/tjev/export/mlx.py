"""MLX on Apple silicon: convert an export, check it against JAX, calibrate and evaluate it.

The prompt (:mod:`tjev.data.render`) and the letter readout are tjev's; only the backend
differs. mlx-lm layout (``mlx_lm/models/qwen3_5.py``): ``Model.model`` is the text decoder
(embed_tokens, layers, final norm; returns normed hidden states) and the output head is the
tied embedding (``embed_tokens.as_linear``), or ``lm_head`` for untied checkpoints (9B).
MLX is imported lazily and JAX never, so this runs in a small venv on the Mac.

Quantizations (``QUANTS``), measured on an M3 Pro 18 GB with the base models: 2B → bf16
(as fast as 4-bit: prefill is compute-bound, and 4-bit costs accuracy), 4B → 8bit (bf16
swaps); ``4bit-emb`` keeps the tied embedding, i.e. the answer readout, in bf16.
Quantization shifts the logits: temperatures are refitted on the converted model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from tjev.data.item import Item, parse_item
from tjev.data.render import LETTERS, TEMPLATE_VERSION, RenderSpec, render
from tjev.eval import calibrate as cal
from tjev.eval.metrics import summarize
from tjev.utils import fingerprint

QUANTS = ("bf16", "8bit", "4bit", "4bit-emb")
DECISION_META = "tjev_decision.json"


def convert(export: str | Path, out: str | Path, quant: str) -> Path:
    """``mlx_lm`` conversion of a merged export; records the quantization in the
    decision metadata (calibration artifacts bind to checkpoint *and* quantization)."""
    from mlx_lm import convert as mlx_convert

    if quant not in QUANTS:
        raise ValueError(f"quant must be one of {QUANTS}")
    export, out = Path(export), Path(out)
    if quant == "bf16":
        mlx_convert(str(export), mlx_path=str(out), dtype="bfloat16")
    elif quant == "4bit-emb":

        def keep_readout(path: str, module: Any, *_: Any) -> bool:
            del module
            return "embed_tokens" not in path  # the tied embedding is the letter readout

        mlx_convert(str(export), mlx_path=str(out), quantize=True, q_bits=4, q_group_size=64,
                    dtype="bfloat16", quant_predicate=keep_readout)  # fmt: skip
    else:
        bits = int(quant.removesuffix("bit"))
        mlx_convert(str(export), mlx_path=str(out), quantize=True, q_bits=bits, q_group_size=64)
    meta = json.loads((export / DECISION_META).read_text(encoding="utf-8"))
    (out / DECISION_META).write_text(json.dumps(meta | {"quant": quant}, indent=2))
    return out


def mlx_identity(path: str | Path) -> str:
    """Checkpoint identity of an MLX model folder: the exported checkpoint and its
    quantization; for a plain base model, its path."""
    meta = Path(path) / DECISION_META
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        return f"{m['checkpoint']}/mlx-{m.get('quant', 'unknown')}"
    return str(path)


def _text_model(model: Any) -> tuple[Any, Any]:
    """The decoder stack returning final-norm hidden states, and its embedding."""
    inner = getattr(model, "model", None)  # qwen3_5.Model.model -> Qwen3_5TextModel
    if inner is not None and hasattr(inner, "embed_tokens") and hasattr(inner, "norm"):
        return inner, inner.embed_tokens
    candidates = [model, getattr(model, "language_model", None), inner]
    for c in list(candidates):
        if c is not None:
            candidates += [getattr(c, "model", None), getattr(c, "language_model", None)]
    for c in candidates:
        if c is not None and hasattr(c, "embed_tokens") and hasattr(c, "layers"):
            return c, c.embed_tokens
    raise RuntimeError("could not locate the text decoder inside the mlx-lm model")


def _output_head(model: Any, embed: Any) -> Any:
    for c in (model, getattr(model, "language_model", None)):
        head = getattr(c, "lm_head", None) if c is not None else None
        if head is not None:
            return head
    return embed.as_linear


class Scorer:
    """Letter logits of one prompt at a time (the serving path) on an mlx-lm model."""

    def __init__(self, path: str | Path):
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        self.path = str(path)
        self.model, self.tokenizer = load(self.path)
        self.decoder, self.embed = _text_model(self.model)
        self.head = _output_head(self.model, self.embed)
        ids = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(i) != 1 for i in ids):
            raise RuntimeError("letters are not single tokens")
        self.letter_ids = mx.array([i[0] for i in ids])

    def logits(self, text: str, k: int) -> tuple[np.ndarray, int]:
        mx = self.mx
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        hidden = self.decoder(mx.array(tokens)[None])  # [1,T,D], final norm applied
        full = self.head(hidden[:, -1, :])[0]  # as_linear / lm_head handle quantized weights
        out = full[self.letter_ids[:k]].astype(mx.float32)
        mx.eval(out)
        return np.asarray(out), len(tokens)


def score_items(scorer: Any, items: list[Item], temperatures: dict | None = None) -> list[dict]:
    """Per item: letter logits in display order, probabilities, target, latency."""
    spec = RenderSpec(train=False)
    rows = []
    for item in items:
        text, order = render(item, spec)
        t0 = time.perf_counter()
        z, n_tokens = scorer.logits(text, len(order))
        t = (temperatures or {}).get(item.question.type, 1.0)
        p = np.exp((z - z.max()) / t)
        p /= p.sum()
        rows.append({"id": item.id, "z": z, "p": p, "tokens": n_tokens,
                     "seconds": time.perf_counter() - t0,
                     "target": np.asarray([item.target[j] for j in order]),
                     "type": item.question.type, "lang": item.lang})  # fmt: skip
    return rows


def _padded(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(r[key]) for r in rows)
    values = np.zeros((len(rows), width))
    mask = np.zeros((len(rows), width), bool)
    for i, r in enumerate(rows):
        values[i, : len(r[key])], mask[i, : len(r[key])] = r[key], True
    return values, mask


def fit_calibration(scorer: Any, items: list[Item], out: str | Path) -> dict:
    """Per-type temperatures fitted on *this* backend, as a provenance-bound artifact."""
    rows = score_items(scorer, items)
    logits, mask = _padded(rows, "z")
    target, _ = _padded(rows, "target")
    fit = cal.fit_per_type(logits, mask, target, [r["type"] for r in rows])
    art = cal.artifact(
        fit,
        checkpoint=mlx_identity(scorer.path),
        data=fingerprint({"n": len(rows), "items": [r["id"] for r in rows]}),
        template=TEMPLATE_VERSION,
        backend="mlx",
    )
    Path(out).write_text(json.dumps(art, indent=2))
    return art


def evaluate(scorer: Any, items: list[Item], temperatures: dict | None = None) -> dict:
    """Quality (accuracy, NLL, ECE, … by type and language) and latency (p50 / p95)."""
    rows = score_items(scorer, items, temperatures)

    def group(selected: list[dict]) -> dict:
        probs, mask = _padded(selected, "p")
        target, _ = _padded(selected, "target")
        seconds = np.asarray([r["seconds"] for r in selected])
        return summarize(probs, target, mask) | {
            "p50_s": float(np.percentile(seconds, 50)),
            "p95_s": float(np.percentile(seconds, 95)),
            "median_tokens": float(np.median([r["tokens"] for r in selected])),
        }

    report: dict = {"all": group(rows)}
    for key in ("type", "lang"):
        report[key] = {
            v: group([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})
        }
    return report


def parity(scorer: Any, reference: dict) -> dict:
    """MLX against the fp32 JAX reference logits (``reference.json`` of an export)."""
    diffs, agree, kls = [], [], []
    for raw, ref in zip(reference["items"], reference["logits"], strict=True):
        text, order = render(parse_item(raw), RenderSpec(train=False))
        got, _ = scorer.logits(text, len(order))
        want = np.asarray(ref)
        diffs.append(float(np.max(np.abs(got - want))))
        agree.append(float(got.argmax() == want.argmax()))
        lp, lq = want - np.logaddexp.reduce(want), got - np.logaddexp.reduce(got)
        kls.append(float(np.sum(np.exp(lp) * (lp - lq))))
    return {"items": len(diffs), "max_abs_logit_diff": max(diffs),
            "top1_agreement": float(np.mean(agree)), "kl_mean": float(np.mean(kls)),
            "kl_max": float(np.max(kls))}  # fmt: skip
