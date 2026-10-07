"""Fixed evaluation sets: packed once, scored by a jitted eval step, grouped reports.

Order averaging (``orders`` = K > 1): every item is rendered under min(K, n) evenly spread
rotations of its options (score levels: one; :func:`tjev.data.render.eval_specs`, the ones
the server averages), each a full segment of its own: the state is re-encoded per rotation
(up to K× the tokens and forward passes of K = 1; the packed rows hold independent
segments, so there is no shared-prefix reuse). Each rotation's slot logits are un-permuted
to the item's label order, so the arrays are ``logits [N,V,L]`` and ``mask [N,V,L]``
(V = K; a view is absent when an item has fewer rotations) and ``target [N,L]``; the
metrics average the views' softmaxes (:func:`tjev.eval.metrics.view_probs`). K = 1 gives
the same numbers as before (one view; the label order only permutes each slot's labels).
"""

from __future__ import annotations

import jax
import numpy as np

from tjev.config import RunConfig
from tjev.data.item import Item
from tjev.data.lang import guess_lang, item_text
from tjev.data.pack import Batch, Vocab, make_segment, pack_eval
from tjev.data.render import NoulOrder, RenderSpec, choose_order, eval_specs
from tjev.eval import predictions
from tjev.eval.metrics import grouped_report, slot_temperatures, view_probs
from tjev.sharding import batch_sharding


class EvalSet:
    def __init__(
        self,
        items: list[Item],
        tok,
        vocab,
        cfg: RunConfig,
        mesh,
        *,
        orders: int = 1,
        noul_order: NoulOrder = "shuffled",
    ):
        spec = RenderSpec(train=False, noul_order=noul_order)
        self.orders, self.noul_order = orders, noul_order
        self.labels = cfg.train.max_labels
        # one view per (item, ordering); a segment's index is its view's position here
        self.views: list[tuple[int, list[int], int]] = []  # (item, order, view slot)
        self.item_tokens: list[int] = []  # prompt tokens of each item's first view
        segments = []
        for i, it in enumerate(items):
            for v, view in enumerate(eval_specs(it, spec, orders)):
                segments.append(make_segment(it, tok, view, vocab, len(self.views)))
                self.views.append((i, choose_order(it, view, None)[0], v))
                if v == 0:
                    self.item_tokens.append(len(segments[-1].tokens))
        n_dev = mesh.size
        microbatch = cfg.train.microbatch_tokens * n_dev
        self.batches, dropped_views = pack_eval(
            segments,
            cfg.train.seq_buckets,
            microbatch,
            cfg.train.max_segments,
            cfg.train.max_labels,
            devices=n_dev,
        )
        # an item is scored only if every one of its orderings fits
        self.dropped = sorted({self.views[j][0] for j in dropped_views})
        # Rows must divide evenly across devices: pad each batch with empty rows.
        self.batches = [pad_rows(b, n_dev) for b in self.batches]
        self.items = items
        self.segments = len(segments)
        self.prompt_tokens = int(sum(len(s.tokens) for s in segments))

    def _arrays(self, eval_step, lora, frozen, mesh) -> dict:
        n, v, k = len(self.items), self.orders, self.labels
        logits = np.zeros((n, v, k), np.float32)
        mask = np.zeros((n, v, k), bool)
        sharding = batch_sharding(mesh, stacked=False)
        for b in self.batches:
            out = np.asarray(eval_step(lora, frozen, jax.device_put(b, sharding)))
            for r, s in zip(*np.nonzero(b.index >= 0), strict=True):
                i, order, view = self.views[int(b.index[r, s])]
                shown = len(order)
                logits[i, view, order] = out[r, s, :shown]  # canonical label order
                mask[i, view, order] = True
        keep = np.ones(n, bool)
        keep[self.dropped] = False
        target = np.zeros((n, k), np.float32)
        for i, it in enumerate(self.items):
            target[i, : len(it.target)] = np.asarray(it.target, np.float32)
        target = target / np.maximum(target.sum(-1, keepdims=True), 1e-12)
        kept = [it for it, ok in zip(self.items, keep, strict=True) if ok]
        by_item: list[list[list[int]]] = [[] for _ in self.items]
        for j, o, _ in self.views:
            by_item[j].append(o)
        self.kept_orders = [by_item[i] for i in np.flatnonzero(keep)]
        return {
            "logits": logits[keep],
            "mask": mask[keep],
            "target": target[keep],
            "type": np.asarray([it.question.type for it in kept]),
            "source": [it.source for it in kept],
            "lang": [it.lang for it in kept],
            "family": [it.family for it in kept],
            "id": [it.id for it in kept],
            "labels": [it.question.labels for it in kept],
            "lang_guess": [guess_lang(item_text(it)) for it in kept],
            "tokens": [t for t, ok in zip(self.item_tokens, keep, strict=True) if ok],
        }

    def run(self, eval_step, lora, frozen, mesh, vocab: Vocab, temperatures=None) -> dict:
        a = self._arrays(eval_step, lora, frozen, mesh)
        # the raw slot arrays of this pass, for temperature-scaled selection (train loop)
        self.last = (a["logits"], a["mask"], a["target"])
        self.last_arrays, self.last_temperatures = a, temperatures
        self.last_lang = list(a["lang"])  # the per-language selection signal
        meta = {"type": list(a["type"]), **{g: a[g] for g in ("source", "lang", "family")}}
        report = grouped_report(*self.last, meta, temperatures)
        report["dropped_too_long"] = len(self.dropped)
        report["orders"] = self.orders
        report["noul_order"] = self.noul_order
        report["cost"] = {"segments": self.segments, "prompt_tokens": self.prompt_tokens,
                          "batches": len(self.batches)}  # fmt: skip
        return report

    def predictions(self) -> list[dict]:
        """The last pass's per-item predictions (:mod:`tjev.eval.predictions`), with the
        distribution at the temperatures that pass applied."""
        a = self.last_arrays
        temps = slot_temperatures(self.last_temperatures, list(a["type"]), a["lang"]) \
            if self.last_temperatures else 1.0  # fmt: skip
        probs = view_probs(a["logits"], a["mask"], temps)
        return predictions.records(a, probs, self.kept_orders)

    def arrays(self, eval_step, lora, frozen, mesh, vocab):
        """Raw per-item arrays (for calibration fitting): logits / mask [N,V,L] in the
        canonical label order, target [N,L], type, lang, source [N]."""
        a = self._arrays(eval_step, lora, frozen, mesh)
        self.last_arrays, self.last_temperatures = a, None
        return {key: a[key] for key in ("logits", "mask", "target", "type", "lang", "source")}


def pad_rows(b: Batch, multiple: int) -> Batch:
    rows = b.tokens.shape[0]
    extra = (-rows) % multiple
    if not extra:
        return b
    fields = {}
    for name, value in vars(b).items():
        if value is None:  # Batch.teacher: train streams only
            fields[name] = None
            continue
        pad = np.zeros((extra, *value.shape[1:]), value.dtype)
        if name == "index":
            pad[...] = -1
        fields[name] = np.concatenate([value, pad])
    return Batch(**fields)
