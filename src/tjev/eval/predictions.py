"""Per-item predictions of an eval pass, for studying calibration offline (gzipped JSONL).

One line per scored item: ``id``, ``source``, ``lang``, ``lang_guess`` (the server's
:func:`tjev.data.lang.guess_lang` on the item's text, ``None`` when unsure), ``family``,
``type``, ``labels`` (the label names), ``target`` (the gold distribution), and per view
(displayed ordering) ``logits`` (the raw slot logits in the *label* order, T = 1) and
``orders`` (``orders[v][shown position] = label index``, for position-bias studies).
``tokens``: the prompt length of the first view. ``probs`` is the reported distribution
(the order-averaged softmax at the temperatures the report applied, or T = 1). Raw
logits, not probabilities, so any temperature or grouping can be refitted on them
(:func:`arrays`).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np


def write(path: str | Path, records: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)
    return path


def read(path: str | Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def records(arrays: dict, probs: np.ndarray, items_orders: list[list[list[int]]]) -> list[dict]:
    """``arrays``: :meth:`tjev.eval.evalset.EvalSet` slot arrays (logits / mask [N,V,K],
    target [N,K] and the per-item metadata); ``items_orders``: each kept item's view orders."""
    out = []
    for i in range(len(arrays["logits"])):
        n = len(arrays["labels"][i])
        present = np.any(arrays["mask"][i], axis=-1)
        out.append({
            "id": arrays["id"][i],
            "source": arrays["source"][i],
            "lang": arrays["lang"][i],
            "lang_guess": arrays["lang_guess"][i],
            "family": arrays["family"][i],
            "type": str(arrays["type"][i]),
            "labels": list(arrays["labels"][i]),
            "target": [round(float(x), 6) for x in arrays["target"][i, :n]],
            "logits": [[round(float(x), 4) for x in arrays["logits"][i, v, :n]]
                       for v in range(len(present)) if present[v]],
            "orders": [list(map(int, o)) for o in items_orders[i]],
            "probs": [round(float(x), 6) for x in probs[i, :n]],
            **({"tokens": int(arrays["tokens"][i])} if "tokens" in arrays else {}),
        })  # fmt: skip
    return out


def arrays(rows: list[dict]) -> dict:
    """The dump back as padded arrays: logits / mask [N,V,K] (absent views all False),
    target [N,K], and per-item lists (type, lang, lang_guess, source, family, id, n)."""
    views = max(len(r["logits"]) for r in rows)
    width = max(len(r["labels"]) for r in rows)
    n = len(rows)
    logits = np.zeros((n, views, width))
    mask = np.zeros((n, views, width), bool)
    target = np.zeros((n, width))
    for i, r in enumerate(rows):
        k = len(r["labels"])
        target[i, :k] = r["target"]
        for v, z in enumerate(r["logits"]):
            logits[i, v, :k], mask[i, v, :k] = z, True
    meta = {key: [r.get(key) for r in rows]
            for key in ("type", "lang", "lang_guess", "source", "family", "id", "orders", "labels")}  # fmt: skip
    return {"logits": logits, "mask": mask, "target": target,
            "n": np.asarray([len(r["labels"]) for r in rows]),
            "tokens": np.asarray([r.get("tokens", 0) for r in rows])} | meta  # fmt: skip
