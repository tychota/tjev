"""From letter logits (in the order the options were displayed) to label probabilities."""

from __future__ import annotations

import numpy as np


def label_probs(logits, order, n_labels: int, temperature: float = 1.0) -> np.ndarray:
    """Softmax of the displayed options' logits at ``temperature``, returned in the item's
    own label order (``order[shown] = label index``, from tjev.data.render.render)."""
    z = np.asarray(logits, np.float64) / temperature
    p = np.exp(z - z.max())
    out = np.zeros(n_labels)
    out[np.asarray(order)] = p / p.sum()
    return out
