"""Prompt rendering: one answer slot per question, options as letters.

Every question type becomes a lettered option list and the model's next token after
the assistant header is the letter. Choice and noul options are shuffled (training
debiasing; the eval order is a fixed hash of the item id); score levels keep their
order. The chat framing is written out explicitly (it equals the Qwen3.5 template with
thinking disabled; ``tests/data/test_real_template.py`` checks this against the HF template).

``TEMPLATE_VERSION`` is part of every run and calibration identity: bump it whenever
the rendered text changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

from tjev.data.item import Item

TEMPLATE_VERSION = 1
LETTERS = tuple(chr(ord("A") + i) for i in range(26))
Layout = Literal["plain", "json"]

SYSTEM = {
    "plain": (
        "You are a decision engine. Read the state and the question, then answer with the "
        "letter of the single best option. Answer with one letter only."
    ),
    "json": (
        "You are a decision engine. The user message is a JSON decision request. Answer "
        "with the letter of the single best option, one letter only."
    ),
}
TYPE_HINT = {"noul": "yes/no", "choice": "choose one", "score": "rate on the scale"}


def chat(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def user_text(item: Item, order: list[int], layout: Layout) -> str:
    q = item.question
    options = []
    for letter, index in zip(LETTERS, order, strict=False):
        label, description = q.labels[index], q.descriptions[index]
        shown = f"{label}: {description}" if description and description != label else label
        options.append((letter, shown))
    if layout == "json":
        return json.dumps(
            {
                "state": item.state,
                "question": q.instructions,
                "type": TYPE_HINT[q.type],
                "options": dict(options),
            },
            ensure_ascii=False,
            indent=1,
        )
    listing = "\n".join(f"{letter}. {text}" for letter, text in options)
    return (
        f"State:\n{item.state}\n\n"
        f"Question ({TYPE_HINT[q.type]}): {q.instructions}\n"
        f"Options:\n{listing}"
    )


def _stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(parts).encode()).digest()[:8], "little")


@dataclass(frozen=True)
class RenderSpec:
    train: bool = True
    shuffle: bool = True
    max_options: int = 26
    min_options: int = 4  # distractor sub-sampling keeps at least this many (if available)
    subsample_prob: float = 0.3  # training: probability of dropping some distractors
    json_prob: float = 0.5  # training: probability of the JSON layout
    layout: Layout = "plain"  # eval layout


def choose_order(
    item: Item, spec: RenderSpec, rng: np.random.Generator | None
) -> tuple[list[int], Layout]:
    """Displayed option order (indices into labels) and the layout."""
    n = len(item.question.labels)
    if item.question.type == "score":
        order = list(range(n))
    elif not spec.train:
        eval_rng = np.random.default_rng(_stable_seed(item.id, "order"))
        order = list(eval_rng.permutation(n)) if spec.shuffle else list(range(n))
    else:
        assert rng is not None
        keep = list(range(n))
        if n > spec.min_options and rng.random() < spec.subsample_prob:
            distractors = [i for i in keep if i not in item.positives]
            low = max(spec.min_options - len(item.positives), 1)
            if low <= len(distractors):  # soft targets over (nearly) every label: keep all
                size = int(rng.integers(low, len(distractors) + 1))
                chosen = set(rng.choice(distractors, size=size, replace=False).tolist())
                keep = [i for i in keep if i in item.positives or i in chosen]
        order = [keep[i] for i in rng.permutation(len(keep))] if spec.shuffle else keep
    if len(order) > spec.max_options:
        raise ValueError(f"{item.id}: {len(order)} options exceed {spec.max_options}")
    if spec.train:
        assert rng is not None
        layout: Layout = "json" if rng.random() < spec.json_prob else "plain"
    else:
        layout = spec.layout
    return [int(i) for i in order], layout


def render(
    item: Item, spec: RenderSpec, rng: np.random.Generator | None = None
) -> tuple[str, list[int]]:
    order, layout = choose_order(item, spec, rng)
    return chat(SYSTEM[layout], user_text(item, order, layout)), order
