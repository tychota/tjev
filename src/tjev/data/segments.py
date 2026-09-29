"""Segment ``i`` of the training stream as a pure, picklable function of (seed, i).

Its source, item, option order, distractor sub-sample and layout all come from
``default_rng([seed, i])``, so any process can compute any segment independently: this
is the random-access stage that Grain may run in worker processes (tjev.data.pipeline).
"""

from __future__ import annotations

import numpy as np

from tjev.data.pack import Segment, Vocab, make_segment
from tjev.data.render import RenderSpec
from tjev.data.tokenize import PromptTokenizer


class SegmentAt:
    def __init__(self, mixture, tok: PromptTokenizer, vocab: Vocab, *, seed: int, spec=None):
        self.mixture, self.tok, self.vocab, self.seed = mixture, tok, vocab, seed
        self.spec = spec or RenderSpec()

    def __call__(self, i: int) -> Segment:
        rng = np.random.default_rng([self.seed, int(i)])
        name, j = self.mixture.draw(rng)
        item = self.mixture.sources[name][j]
        return make_segment(item, self.tok, self.spec, self.vocab, int(i), rng)

    def pair(self, i: int) -> tuple[int, Segment]:
        """(i, segment): the index travels with the segment so consumers can check order."""
        return int(i), self(i)
