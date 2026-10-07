"""Serving policies: what is served on top of the calibrated distribution (all off by default).

A policy is applied *after* calibration and never replaces it: the response keeps the
calibrated distribution (``calibrated``, :mod:`tjev.serve.wire`) whenever a policy changes
a value a consumer scores. Two kinds:

- :class:`AbstentionPolicy` (noul only): JevBench v1.5 reads P(yes) <= 0.2 as No, >= 0.8
  as Yes and anything between as an abstention, counted wrong. The policy moves an
  in-band P(yes) to a band edge (``eps`` beyond it), optionally only when it is at least
  ``margin`` away from 0.5 (a near-coin-flip stays an abstention). Plumb v5.2's
  ``--noul-commit`` is ``margin=0``.
- serving temperatures (:attr:`ServingPolicy.temperatures`, per type): the calibrated
  distribution raised to 1/T and renormalised (T < 1 sharpens). The argmax is unchanged;
  for a score item the expected level moves towards the mode, which is what v1.5's MAE of
  the expected level rewards. On a single ordering it equals serving at the calibrated
  temperature × T; on an order-averaged distribution it sharpens the mixture.

Pure numpy: the server (:mod:`tjev.serve.wire`) and the offline v1.5 scorer
(:mod:`tjev.eval.v15`) apply the very same functions. docs/serve.md#serving-policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Snap = Literal["nearest", "argmax"]
KINDS = ("choice", "noul", "score")


@dataclass(frozen=True)
class AbstentionPolicy:
    """Commit a noul P(yes) inside the open band (low, high) to an edge of it.

    ``snap="nearest"``: to the nearer edge (ties: high); ``"argmax"``: to ``high`` when
    P(yes) >= 0.5, else ``low`` (the same for a band symmetric about 0.5). Values outside
    the band, and the edges themselves, are kept (JevBench v1.5 reads <= 0.2 / >= 0.8 as
    committed). ``margin`` > 0: an in-band P(yes) with |P(yes) − 0.5| <= margin is kept
    (an abstention); ``eps``: commit to ``low − eps`` / ``high + eps`` (clipped to [0, 1])
    instead of the edge itself, robust to a consumer that rounds or compares strictly."""

    low: float = 0.2
    high: float = 0.8
    snap: Snap = "nearest"
    into_noul: bool = False  # also report the committed value as ``noul`` (what is scored)
    margin: float = 0.0
    eps: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.low < self.high <= 1.0:
            raise ValueError(f"abstention band must satisfy 0 <= low < high <= 1, got "
                             f"({self.low}, {self.high})")  # fmt: skip
        if self.snap not in ("nearest", "argmax"):
            raise ValueError(f"snap must be nearest or argmax, got {self.snap!r}")
        if not 0.0 <= self.margin <= 0.5:
            raise ValueError(f"commit margin must be in [0, 0.5], got {self.margin}")
        if not 0.0 <= self.eps < 0.1:
            raise ValueError(f"commit eps must be in [0, 0.1), got {self.eps}")

    def commit(self, p_yes: float) -> float:
        if not self.low < p_yes < self.high:
            return p_yes
        if self.margin > 0 and abs(p_yes - 0.5) <= self.margin:
            return p_yes  # too close to a coin flip: stays an abstention
        pivot = 0.5 if self.snap == "argmax" else (self.low + self.high) / 2
        if p_yes >= pivot:
            return min(1.0, self.high + self.eps)
        return max(0.0, self.low - self.eps)

    def commit_array(self, p_yes: np.ndarray) -> np.ndarray:
        return np.asarray([self.commit(float(p)) for p in np.asarray(p_yes).ravel()]).reshape(
            np.shape(p_yes)
        )

    def describe(self) -> dict:
        return {"band": [self.low, self.high], "snap": self.snap, "into_noul": self.into_noul,
                "margin": self.margin, "eps": self.eps}  # fmt: skip


def sharpen(probs: np.ndarray, temperature: float) -> np.ndarray:
    """``probs`` ([K] or [N,K]) raised to 1/T, renormalised; zeros stay zero."""
    p = np.asarray(probs, np.float64)
    if temperature == 1.0:
        return p / p.sum(-1, keepdims=True)
    logp = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), -np.inf) / temperature
    logp = logp - logp.max(-1, keepdims=True)
    e = np.where(p > 0, np.exp(logp), 0.0)
    return e / e.sum(-1, keepdims=True)


@dataclass(frozen=True)
class ServingPolicy:
    """The serving policy of a server or an offline scoring: an optional noul abstention
    policy and per-type serving temperatures (absent type: T = 1, the calibrated one)."""

    abstention: AbstentionPolicy | None = None
    temperatures: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for kind, t in self.temperatures.items():
            if kind not in KINDS:
                raise ValueError(f"serving temperature for an unknown type {kind!r}")
            if not 0.05 <= t <= 20.0:
                raise ValueError(f"serving temperature {kind}={t} outside [0.05, 20]")

    def temperature(self, kind: str) -> float:
        return float(self.temperatures.get(kind, 1.0))

    def serve(self, kind: str, probs: np.ndarray) -> np.ndarray:
        """The served distribution of one item ([K]) or several ([N,K]) of type ``kind``."""
        return sharpen(probs, self.temperature(kind))

    def alters(self, kind: str) -> bool:
        """Whether a scored value of a ``kind`` answer can differ from the calibrated one."""
        noul = kind == "noul" and self.abstention is not None and self.abstention.into_noul
        return self.temperature(kind) != 1.0 or noul

    def describe(self) -> dict:
        return {"abstention": None if self.abstention is None else self.abstention.describe(),
                "temperatures": {k: self.temperature(k) for k in KINDS}}  # fmt: skip

    @property
    def off(self) -> bool:
        return self.abstention is None and all(t == 1.0 for t in self.temperatures.values())


def as_policy(policy: ServingPolicy | AbstentionPolicy | None) -> ServingPolicy | None:
    """A bare abstention policy as a serving policy; an all-off policy as None."""
    if isinstance(policy, AbstentionPolicy):
        policy = ServingPolicy(abstention=policy)
    return None if policy is None or policy.off else policy
