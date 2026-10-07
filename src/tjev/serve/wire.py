"""The ``POST /v1/systemone`` wire format (the JevBench TypeSafe adapter's), as pydantic models.

Request::  {"state": str | object, "model": str?, "questions": {name: question}}
where ``question`` is ``{"type", "instructions", "criteria", "labels"?}`` (JevBench).

Response:: {"answers": {name: {"type", "probabilities": {label: p}, "choice" | "noul"}},
            "usage": {"prompt_tokens", "input_tokens", "output_tokens", "seconds",
                      "batch_items"}, "model": str}

Probabilities are in each question's own label order and sum to 1 (renormalised in
float64). ``noul`` is P(yes); ``choice`` is the most probable label (choice and score).
``input_tokens`` / ``output_tokens`` are the names the JevBench harness prices
(``input_tokens`` = ``prompt_tokens``; a logit readout generates nothing, so 0 output).

A question's displayed option order is a fixed hash of its content (state, type,
instructions, labels), as the eval order is a hash of the item id: the same question always
gets the same prompt, and different questions do not all share one permutation.

The JevBench TypeSafe adapter reads ``answers.<name>.noul`` for a noul question (as
``{"yes": p, "no": 1 - p}``; JevBench v1.5 then calls P(yes) <= 0.2 No, >= 0.8 Yes and
anything between an abstention, counted wrong) and ``probabilities`` (plus ``choice``,
which must be a label) for choice and score. It never reads ``probabilities`` of a noul.

An optional :class:`ServingPolicy` (:mod:`tjev.serve.policy`; off by default, and then the
response is exactly as above) is applied after calibration:

- per-type serving temperatures: ``probabilities`` (and ``noul``, ``choice``) are the
  calibrated distribution sharpened at that type's T;
- a noul :class:`AbstentionPolicy` adds ``committed`` (P(yes) moved out of the abstention
  band to an edge, or kept in it within the commit margin) and ``decision`` ("yes" / "no"
  / "abstain"). With ``into_noul`` the committed value is also what ``noul`` reports, i.e.
  what the TypeSafe adapter scores (its accuracy and its noul ECE), so ``noul`` !=
  ``probabilities["yes"]`` for a committed in-band answer.

Whenever a policy changes a value the adapter scores, the answer also carries
``calibrated``: the calibrated distribution, unchanged. Honest by construction: the
calibrated probabilities are always in the response.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_serializer

from tjev.data.item import Item, parse_item
from tjev.serve.policy import AbstentionPolicy, ServingPolicy, as_policy

__all__ = ["AbstentionPolicy", "ServingPolicy"]  # re-exported (tjev.serve.policy)


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
    # serving policy only (absent from the JSON when the policy is off)
    committed: float | None = None
    decision: str | None = None
    calibrated: dict[str, float] | None = None

    @model_serializer(mode="wrap")
    def _drop_policy_fields(self, handler):
        out = handler(self)
        for key in ("committed", "decision", "calibrated"):
            if out.get(key) is None:
                out.pop(key, None)
        return out


class Usage(BaseModel):
    prompt_tokens: int
    input_tokens: int = 0  # = prompt_tokens (the JevBench harness's name)
    output_tokens: int = 0  # a logit readout generates no tokens
    seconds: float
    batch_items: int  # items scored together with this request's (micro-batching)


class DecisionResponse(BaseModel):
    answers: dict[str, Answer]
    usage: Usage
    model: str


EMPTY_STATE = "(no state: the question holds all the information)"


def to_items(request: DecisionRequest) -> list[tuple[str, Item]]:
    """One :class:`Item` per question (fail-closed: ``ValueError`` on an invalid rubric)."""
    out = []
    for name, q in request.questions.items():
        content = json.dumps([request.state, q.type, q.instructions, q.criteria, q.labels],
                             sort_keys=True, ensure_ascii=False, default=str)  # fmt: skip
        raw: dict[str, Any] = {
            "id": "request:" + hashlib.sha256(content.encode()).hexdigest()[:16],
            # an empty state (all the evidence in the question, as some Decision Index
            # benchmarks send) is answered, not refused
            "state": request.state if request.state not in ("", {}, []) else EMPTY_STATE,
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


def answer(
    item: Item, probs: np.ndarray, policy: ServingPolicy | AbstentionPolicy | None = None
) -> Answer:
    """The answer to one question from its calibrated label probabilities, under the
    serving ``policy`` (None: the calibrated distribution as is)."""
    labels = item.question.labels
    kind = item.question.type
    p = np.asarray(probs, np.float64)
    p = p / p.sum()
    policy = as_policy(policy)
    if policy is None:
        dist = {label: float(v) for label, v in zip(labels, p, strict=True)}
        if kind == "noul":
            return Answer(type="noul", probabilities=dist, noul=dist["yes"])
        return Answer(type=kind, probabilities=dist, choice=max(dist, key=dist.__getitem__))
    served = policy.serve(kind, p)
    dist = {label: float(v) for label, v in zip(labels, served, strict=True)}
    extra: dict = {}
    if policy.alters(kind):  # the calibrated distribution stays available
        extra["calibrated"] = {label: float(v) for label, v in zip(labels, p, strict=True)}
    if kind != "noul":
        return Answer(type=kind, probabilities=dist, choice=max(dist, key=dist.__getitem__),
                      **extra)  # fmt: skip
    abstention = policy.abstention
    if abstention is None:
        return Answer(type="noul", probabilities=dist, noul=dist["yes"], **extra)
    committed = abstention.commit(dist["yes"])
    if abstention.low < committed < abstention.high:
        decision = "abstain"  # kept in the band (the commit margin)
    else:
        decision = "yes" if committed >= abstention.high else "no"
    noul = committed if abstention.into_noul else dist["yes"]
    return Answer(type="noul", probabilities=dist, noul=noul, committed=committed,
                  decision=decision, **extra)  # fmt: skip
