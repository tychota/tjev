"""The ``POST /v1/systemone`` wire format (the JevBench TypeSafe adapter's), as pydantic models.

Request::  {"state": str | object, "model": str?, "questions": {name: question}}
where ``question`` is ``{"type", "instructions", "criteria", "labels"?}`` (JevBench).

Response:: {"answers": {name: {"type", "probabilities": {label: p}, "choice" | "noul"}},
            "usage": {"prompt_tokens", "seconds", "batch_items"}, "model": str}

Probabilities are in each question's own label order and sum to 1 (renormalised in
float64). ``noul`` is P(yes); ``choice`` is the most probable label (choice and score).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from tjev.data.item import Item, parse_item


class Question(BaseModel):
    type: Literal["noul", "choice", "score"]
    instructions: str = ""
    criteria: dict[str, str] | list[str] | None = None
    labels: list[str] | None = None


class DecisionRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1)
    model: str | None = None


class Answer(BaseModel):
    type: Literal["noul", "choice", "score"]
    probabilities: dict[str, float]
    choice: str | None = None
    noul: float | None = None


class Usage(BaseModel):
    prompt_tokens: int
    seconds: float
    batch_items: int  # items scored together with this request's (micro-batching)


class DecisionResponse(BaseModel):
    answers: dict[str, Answer]
    usage: Usage
    model: str


def to_items(request: DecisionRequest) -> list[tuple[str, Item]]:
    """One :class:`Item` per question (fail-closed: ``ValueError`` on an invalid rubric)."""
    out = []
    for name, q in request.questions.items():
        raw: dict[str, Any] = {
            "id": f"request:{name}",
            "state": request.state,
            "question": {
                "type": q.type,
                "instructions": q.instructions,
                "criteria": q.criteria or {},
            },
            "source": "request",
        }
        if q.type != "noul" and q.labels:
            raw["labels"] = q.labels
        if q.type == "noul":
            labels = ["no", "yes"]
        elif q.labels:
            labels = q.labels
        elif isinstance(q.criteria, list):
            labels = [str(i) for i in range(len(q.criteria))]
        else:
            labels = list(q.criteria or {})
        if not labels:
            raise ValueError(f"question {name!r} has no labels or criteria")
        raw["target"] = {label: 1.0 / len(labels) for label in labels}  # unused at inference
        out.append((name, parse_item(raw, f"question {name!r}")))
    return out


def answer(item: Item, probs: np.ndarray) -> Answer:
    labels = item.question.labels
    p = np.asarray(probs, np.float64)
    dist = {label: float(v) for label, v in zip(labels, p / p.sum(), strict=True)}
    if item.question.type == "noul":
        return Answer(type="noul", probabilities=dist, noul=dist["yes"])
    return Answer(
        type=item.question.type, probabilities=dist, choice=max(dist, key=dist.__getitem__)
    )
