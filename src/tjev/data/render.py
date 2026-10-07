"""Prompt rendering: one answer slot per question, options as letters.

Every question type becomes a lettered option list and the model's next token after
the assistant header is the letter. Choice and noul options are shuffled (training
debiasing; the eval order is a fixed hash of the item id); score levels keep their
order. The chat framing is written out explicitly (it equals the Qwen3.5 template with
thinking disabled; ``tests/data/test_real_template.py`` checks this against the HF template).

At eval and serving, ``RenderSpec(noul_order="fixed")`` shows noul as A = no, B = yes for
every item (Cygnet's "false first"); training is unaffected. :func:`eval_specs` lists the K
option rotations order averaging scores (:mod:`tjev.eval.evalset`, :mod:`tjev.serve`).

``TEMPLATE_VERSION`` is part of every run and calibration identity: bump it whenever
the rendered text changes.

``RenderSpec(prompt="semif")`` (eval only) is the field's frozen-model control prompt, the
one SemIf, Jobe and local-jev score a frozen Qwen3.5-4B with (0.805–0.810 on the JevBench
public 231 under the official harness; github.com/TheoLeeCJ/openjev
``src/semif_phase1/core.py``, ``benchmarks/build_typesafe.py``): a one-line JSON user message
``{"evidence", "criterion", "options": [{"letter", "description": "<id>: <text>"}]}``, its
own system line, no type hint, options in the declared order (noul: ``true`` then
``false``), the same chat framing and letter readout. docs/official-eval.md compares it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from tjev.data.item import Item

TEMPLATE_VERSION = 1
LETTERS = tuple(chr(ord("A") + i) for i in range(26))
Layout = Literal["plain", "json"]
NoulOrder = Literal["shuffled", "fixed"]
NOUL_ORDERS: tuple[NoulOrder, ...] = ("shuffled", "fixed")

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
Prompt = Literal["tjev", "semif"]
SEMIF_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


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


def semif_user(item: Item, order: list[int]) -> str:
    """SemIf's user message (``semif_phase1.core``; option mapping of ``build_typesafe``)."""
    q = item.question
    options = []
    for letter, index in zip(LETTERS, order, strict=False):
        label, text = q.labels[index], q.descriptions[index]
        if q.type == "noul":
            label = "true" if label == "yes" else "false"
            text = text or f"The proposition is {label}."
        options.append({"letter": letter, "description": f"{label}: {text or label}"})
    state: object = item.state
    if state.lstrip().startswith(("{", "[")):  # a structured state (JevBench hard): the object
        with contextlib.suppress(json.JSONDecodeError):
            state = json.loads(state)
    payload = {"evidence": state, "criterion": q.instructions, "options": options}
    return json.dumps(payload, ensure_ascii=False)


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
    # eval only: rotate the displayed choice / noul options by this many places (order
    # averaging: each rotation puts every option at another position; eval_specs)
    rotation: int = 0
    prompt: Prompt = "tjev"  # eval only: "semif", the frozen-model control prompt
    # eval / serving only, tjev prompt: "shuffled" (the hash of the item id, like choice) or
    # "fixed" (no first, yes second for every item). Training always shuffles noul.
    noul_order: NoulOrder = "shuffled"


def choose_order(
    item: Item, spec: RenderSpec, rng: np.random.Generator | None
) -> tuple[list[int], Layout]:
    """Displayed option order (indices into labels) and the layout."""
    n = len(item.question.labels)
    if item.question.type == "score":
        order = list(range(n))
    elif not spec.train:
        if spec.prompt == "semif":  # the declared order; noul true first
            order = [1, 0] if item.question.type == "noul" else list(range(n))
        elif item.question.type == "noul" and spec.noul_order == "fixed":
            order = list(range(n))  # NOUL_LABELS = (no, yes): false first
        else:
            eval_rng = np.random.default_rng(_stable_seed(item.id, "order"))
            order = list(eval_rng.permutation(n)) if spec.shuffle else list(range(n))
        if spec.rotation % n:
            order = order[spec.rotation % n :] + order[: spec.rotation % n]
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
        # "options per choice": the Decision Index kit's capacity marker (a refusal, not an error)
        raise ValueError(
            f"{item.id}: {len(order)} options exceed {spec.max_options} options per choice"
        )
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
    if spec.prompt == "semif":
        if spec.train:
            raise ValueError("the semif prompt is an eval-only control")
        return chat(SEMIF_SYSTEM, semif_user(item, order)), order
    return chat(SYSTEM[layout], user_text(item, order, layout)), order


def eval_specs(item: Item, spec: RenderSpec, k: int) -> list[RenderSpec]:
    """The render specs order averaging scores an item under: ``k`` evenly spread
    rotations of its eval order (rotation ``r·n // k'`` for ``r < k' = min(k, n)``: all
    distinct, never the same ordering twice; noul at 2 = both orders; k >= n = every
    rotation, each option once at each position). The first is the plain eval order, so
    ``k = 1`` is the plain eval. Score levels keep their order (one spec): training only
    ever shows them in order."""
    if k < 1:
        raise ValueError(f"orders must be >= 1, got {k}")
    if spec.train:
        raise ValueError("order averaging is for eval / serving (RenderSpec(train=False))")
    n = len(item.question.labels)
    views = 1 if item.question.type == "score" else min(k, n)
    return [replace(spec, rotation=spec.rotation + (r * n) // views) for r in range(views)]


def eval_orders(item: Item, spec: RenderSpec, k: int) -> list[list[int]]:
    """The displayed orders of :func:`eval_specs`."""
    return [choose_order(item, s, None)[0] for s in eval_specs(item, spec, k)]
