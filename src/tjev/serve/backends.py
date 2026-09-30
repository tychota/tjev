"""Scoring backends: the same prompt, readout and temperatures on JAX or on MLX.

A backend maps a list of items to per-item probabilities over each item's labels, in the
item's *own* label order (the displayed option order is undone here). ``score`` is
blocking and not thread-safe: the server calls it from one worker at a time.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from tjev.data.item import TYPES, Item, parse_item
from tjev.data.render import TEMPLATE_VERSION, RenderSpec, render
from tjev.eval import calibrate as cal


class Backend(Protocol):
    name: str

    def score(self, items: list[Item]) -> tuple[list[np.ndarray], dict]: ...


def label_probs(
    logits: np.ndarray, order: list[int], n_labels: int, temperature: float
) -> np.ndarray:
    """Softmax of the displayed options' logits at ``temperature``, in the item's label
    order (``order[shown] = label index``)."""
    z = np.asarray(logits, np.float64) / temperature
    p = np.exp(z - z.max())
    out = np.zeros(n_labels)
    out[np.asarray(order)] = p / p.sum()
    return out


def _temperatures(calibration: str | Path | None, checkpoint: str, backend: str) -> dict:
    if not calibration:
        return {}
    art = json.loads(Path(calibration).read_text(encoding="utf-8"))
    return cal.validate(art, checkpoint=checkpoint, template=TEMPLATE_VERSION, backend=backend)


class JaxBackend:
    """A trained run (``run_dir``, its selected step by default) or an untrained base model
    (``model_path``, the zero-shot control) on the default JAX devices."""

    def __init__(
        self,
        *,
        run_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        calibration: str | Path | None = None,
        step: int | None = None,
        buckets: tuple[int, ...] = (1024, 2048, 4096, 8192),
        warmup: bool = True,
    ):
        import jax

        from tjev.config import RunConfig
        from tjev.data.pack import Vocab
        from tjev.data.tokenize import PromptTokenizer
        from tjev.eval.runs import LoadedRun, base_identity
        from tjev.model import build_model
        from tjev.sharding import make_mesh, place_frozen
        from tjev.train.step import make_eval_step, split_model

        if run_dir is not None:
            run = LoadedRun(run_dir, step)
            self.cfg, self.tok, self.eval_fn, self.mesh = run.cfg, run.tok, run.eval_fn, run.mesh
            self.lora, self.frozen = run.lora, run.frozen
            self.name = f"tjev:{Path(run_dir).name}@{run.step}"
            identity = run.checkpoint_identity
        elif model_path is not None:
            self.cfg = RunConfig()
            _, model = build_model(model_path, self.cfg.compute, None)
            self.mesh = make_mesh(self.cfg.mesh)
            graphdef, self.lora, frozen = split_model(model)
            del model
            self.frozen = place_frozen(frozen, self.mesh)
            self.eval_fn = jax.jit(make_eval_step(graphdef))
            self.tok = PromptTokenizer(model_path)
            self.name = f"base:{Path(model_path).name}"
            identity = base_identity(model_path)
        else:
            raise ValueError("JaxBackend needs run_dir or model_path")
        self.temperatures = _temperatures(calibration, identity, "jax")
        self.buckets = tuple(sorted(buckets))
        self.vocab = Vocab({"source": ["request"], "lang": ["en"], "family": ["unknown"]})
        if warmup:
            self.warmup()

    def warmup(self) -> None:
        """Compile every bucket once, so no request pays a compile."""
        for length in self.buckets:
            text = "x " * max(1, length * 3 // 4 - 150)  # lands in this bucket
            item = parse_item({"state": text, "expected": "yes", "question": {
                "type": "noul", "instructions": "?", "criteria": {"false": "no", "true": "yes"}}})  # fmt: skip
            # a filler that does not fit leaves that bucket to compile on first use
            with contextlib.suppress(ValueError):
                self.score([item])

    def score(self, items: list[Item]) -> tuple[list[np.ndarray], dict]:
        import jax

        from tjev.data.pack import make_segment, pack_eval
        from tjev.eval.evalset import pad_rows
        from tjev.sharding import batch_sharding

        t0 = time.perf_counter()
        spec = RenderSpec(train=False)
        segs = [make_segment(item, self.tok, spec, self.vocab, i) for i, item in enumerate(items)]
        orders = [render(item, spec)[1] for item in items]
        longest = max(len(s.tokens) for s in segs)
        # pack into the smallest bucket that fits: no fixed padding for small requests
        batches, dropped = pack_eval(segs, self.buckets, max(longest, self.buckets[0]),
                                     self.cfg.train.max_segments, self.cfg.train.max_labels)  # fmt: skip
        if dropped:
            raise ValueError(f"state too long for the largest bucket ({self.buckets[-1]} tokens)")
        out: list[np.ndarray | None] = [None] * len(items)
        sharding = batch_sharding(self.mesh, stacked=False)
        for b in (pad_rows(b, self.mesh.size) for b in batches):
            logits = np.asarray(self.eval_fn(self.lora, self.frozen, jax.device_put(b, sharding)))
            for r, s in zip(*np.nonzero(b.index >= 0), strict=True):
                i, k = int(b.index[r, s]), int(b.label_mask[r, s].sum())
                t = self.temperatures.get(TYPES[int(b.type_id[r, s])], 1.0)
                out[i] = label_probs(logits[r, s, :k], orders[i], len(items[i].question.labels), t)
        usage = {"prompt_tokens": int(sum(len(s.tokens) for s in segs)),
                 "seconds": time.perf_counter() - t0}  # fmt: skip
        return [p for p in out if p is not None], usage


class MlxBackend:
    """Apple silicon: an MLX model from ``tjev mlx convert`` (calibrated on MLX)."""

    def __init__(self, *, model_path: str | Path, calibration: str | Path | None = None):
        from tjev.export.mlx import Scorer, mlx_identity

        self.scorer = Scorer(model_path)
        self.name = f"mlx:{Path(model_path).name}"
        self.temperatures = _temperatures(calibration, mlx_identity(model_path), "mlx")

    def score(self, items: list[Item]) -> tuple[list[np.ndarray], dict]:
        t0 = time.perf_counter()
        spec = RenderSpec(train=False)
        out, tokens = [], 0
        for item in items:  # one prompt at a time: MLX prefill is compute-bound anyway
            text, order = render(item, spec)
            z, n = self.scorer.logits(text, len(order))
            tokens += n
            t = self.temperatures.get(item.question.type, 1.0)
            out.append(label_probs(z, order, len(item.question.labels), t))
        return out, {"prompt_tokens": tokens, "seconds": time.perf_counter() - t0}
