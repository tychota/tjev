"""Fixed evaluation sets: packed once, scored by a jitted eval step, grouped reports."""

from __future__ import annotations

from collections import defaultdict

import jax
import numpy as np

from tjev.config import RunConfig
from tjev.data.item import TYPES, Item
from tjev.data.pack import Batch, Vocab, make_segment, pack_eval
from tjev.data.render import RenderSpec
from tjev.eval.metrics import grouped_report
from tjev.sharding import batch_sharding


class EvalSet:
    def __init__(self, items: list[Item], tok, vocab, cfg: RunConfig, mesh):
        spec = RenderSpec(train=False)
        segments = [make_segment(it, tok, spec, vocab, i) for i, it in enumerate(items)]
        n_dev = mesh.size
        microbatch = cfg.train.microbatch_tokens * n_dev
        self.batches, self.dropped = pack_eval(
            segments,
            cfg.train.seq_buckets,
            microbatch,
            cfg.train.max_segments,
            cfg.train.max_labels,
        )
        # Rows must divide evenly across devices: pad each batch with empty rows.
        self.batches = [pad_rows(b, n_dev) for b in self.batches]
        self.items = items

    def run(self, eval_step, lora, frozen, mesh, vocab: Vocab, temperatures=None) -> dict:
        logits, masks, targets, meta = [], [], [], defaultdict(list)
        sharding = batch_sharding(mesh, stacked=False)
        for b in self.batches:
            out = np.asarray(eval_step(lora, frozen, jax.device_put(b, sharding)))
            sel = b.weight > 0
            logits.append(out[sel])
            masks.append(b.label_mask[sel])
            targets.append(b.target[sel])
            meta["type"] += [TYPES[i] for i in b.type_id[sel]]
            meta["source"] += [vocab.names["source"][i] for i in b.source_id[sel]]
            meta["lang"] += [vocab.names["lang"][i] for i in b.lang_id[sel]]
            meta["family"] += [vocab.names["family"][i] for i in b.family_id[sel]]
        # the raw slot arrays of this pass, for temperature-scaled selection (train loop)
        self.last = (np.concatenate(logits), np.concatenate(masks), np.concatenate(targets))
        report = grouped_report(*self.last, dict(meta), temperatures)
        report["dropped_too_long"] = len(self.dropped)
        return report

    def arrays(self, eval_step, lora, frozen, mesh, vocab):
        """Raw per-slot arrays (for calibration fitting)."""
        out = defaultdict(list)
        sharding = batch_sharding(mesh, stacked=False)
        for b in self.batches:
            logits = np.asarray(eval_step(lora, frozen, jax.device_put(b, sharding)))
            sel = b.weight > 0
            out["logits"].append(logits[sel])
            out["mask"].append(b.label_mask[sel])
            out["target"].append(b.target[sel])
            out["type"].append(np.asarray([TYPES[i] for i in b.type_id[sel]]))
        return {k: np.concatenate(v) for k, v in out.items()}


def pad_rows(b: Batch, multiple: int) -> Batch:
    rows = b.tokens.shape[0]
    extra = (-rows) % multiple
    if not extra:
        return b
    fields = {}
    for name, value in vars(b).items():
        pad = np.zeros((extra, *value.shape[1:]), value.dtype)
        if name == "index":
            pad[...] = -1
        fields[name] = np.concatenate([value, pad])
    return Batch(**fields)
