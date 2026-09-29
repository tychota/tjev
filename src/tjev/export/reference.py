"""Reference letter logits from the fp32 JAX model (XLA kernels), to check a converted model.

The items are held-out EN/FR generator items plus, if given, short JevBench public items;
both render with the eval template. ``tjev mlx check`` recomputes the same logits with MLX
and reports max |Δ|, top-1 agreement and KL.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tjev.config import ComputeSpec
from tjev.data.item import Item, item_to_json, parse_item, read_jsonl
from tjev.data.render import RenderSpec, render
from tjev.data.sources import generate
from tjev.data.tokenize import PromptTokenizer
from tjev.eval.runs import LoadedRun
from tjev.model import Qwen35

REFERENCE_GENERATORS = ("gen_ticket_triage", "gen_refund_policy", "gen_register", "gen_base_rate")


def reference_items(jevbench: str | Path | None = None) -> list[Item]:
    items: list[Item] = []
    if jevbench is not None:
        public = [i for i in read_jsonl(jevbench) if i.source != "jevbench_hard"]
        items += public[::6][:12]
    for name in REFERENCE_GENERATORS:
        rows = generate(name, 3, seed=777, split="heldout", fr_share=0.5)
        items += [parse_item(r) for r in rows]
    return items


def letter_logits(model: Qwen35, tok: PromptTokenizer, items: list[Item]) -> list[list[float]]:
    rows = []
    for item in items:
        text, order = render(item, RenderSpec(train=False))
        ids = jnp.asarray(tok.encode(text))[None]
        hidden = model.hidden(ids, jnp.ones_like(ids), jnp.arange(ids.shape[1])[None])
        labels = jnp.asarray(tok.letter_ids[: len(order)])[None, None]
        z = model.label_logits(hidden, jnp.asarray([[ids.shape[1] - 1]]), labels)[0, 0]
        rows.append([float(x) for x in np.asarray(z)])
    return rows


def write_reference(
    run_dir: str | Path, step: int, out: str | Path, *, jevbench: str | Path | None = None
) -> None:
    """A run checkpoint in fp32 with the reference (XLA) kernels, on the fixed items. It
    holds a full fp32 copy of the base: on a small accelerator, run it with JAX_PLATFORMS=cpu."""
    items = reference_items(jevbench)
    run = LoadedRun(run_dir, step, compute=ComputeSpec(remat="none"), dtype="float32")
    ref = {
        "items": [item_to_json(i) for i in items],
        "logits": letter_logits(run.model(), run.tok, items),
        "checkpoint": run.checkpoint_identity,
        "step": run.step,
    }
    Path(out).write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
