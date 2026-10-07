"""Scoring backends: the same prompt, readout and temperatures on JAX or on MLX.

A backend maps a list of items to per-item probabilities over each item's labels, in the
item's *own* label order (the displayed option order is undone here). ``score`` is
blocking and not thread-safe: the server calls it from one worker at a time.

Order averaging (``orders`` = K > 1): each item is scored under the option rotations of
:func:`tjev.data.render.eval_specs` (min(K, n) of them; score levels: one), each at the
calibrated temperature, and the distributions (un-permuted to the label order) are
averaged, exactly as :mod:`tjev.eval.evalset` does at eval. The JAX backend packs every
rotation of every item of a micro-batch into the same call; each is a full prompt (about
K× the prompt tokens, no shared-state prefix reuse). The calibration artifact must have
been fitted with the same K and noul order, or the backend refuses to start.

Per-language temperatures (a ``type_lang`` artifact, :mod:`tjev.eval.calibrate`): the
wire format carries no language, so a request's is guessed from its text (state,
instructions, option descriptions) with :func:`tjev.data.lang.guess_lang`, the data
report's stop-word heuristic; an unsure guess (``None``: short, mixed, code) or a
language the fit never saw gets the type's temperature.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from tjev.data.item import Item, parse_item
from tjev.data.lang import guess_lang, item_text
from tjev.data.render import (
    TEMPLATE_VERSION,
    NoulOrder,
    Prompt,
    RenderSpec,
    choose_order,
    eval_specs,
    render,
)
from tjev.eval import calibrate as cal

# a noul item: the bucket warm-up filler, and the probe that validates ``orders``
_FILLER = {"state": "x", "expected": "yes", "question": {
    "type": "noul", "instructions": "?", "criteria": {"false": "no", "true": "yes"}}}  # fmt: skip

# Request lengths the JAX backend compiles (one executable each, at warmup): up to 32k
# tokens, a single long question as Jev serves it. ``max_tokens`` caps them (warmup time).
SERVE_BUCKETS = (1024, 2048, 4096, 8192, 16384, 32768)


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


def temperature(temperatures: dict, item: Item) -> float:
    """A served item's temperature: its (type, guessed language) cell's, else its type's."""
    if not any("/" in key for key in temperatures):  # per type: no guess needed
        return cal.lookup(temperatures, item.question.type)
    return cal.lookup(temperatures, item.question.type, guess_lang(item_text(item)))


def _temperatures(
    calibration: str | Path | None, checkpoint: str, backend: str, orders: int, noul_order: str
) -> dict:
    if not calibration:
        return {}
    art = json.loads(Path(calibration).read_text(encoding="utf-8"))
    return cal.validate(art, checkpoint=checkpoint, template=TEMPLATE_VERSION, backend=backend,
                        orders=orders, noul_order=noul_order)  # fmt: skip


class JaxBackend:
    """A trained run (``run_dir``, its selected step by default) or an untrained base model
    (``model_path``, the zero-shot control) on the default JAX devices.

    ``orders`` > 1: order averaging. Each choice / noul item is scored under ``orders``
    rotations of its displayed options (``min(orders, n)`` distinct ones, spread evenly:
    noul at 2 = both orders) and the calibrated label probabilities are averaged; score
    levels keep their order (one pass). ``prompt="semif"``: the field's frozen-model control
    prompt (:mod:`tjev.data.render`), uncalibrated (temperatures bind tjev's template)."""

    def __init__(
        self,
        *,
        run_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        calibration: str | Path | None = None,
        step: int | None = None,
        buckets: tuple[int, ...] = SERVE_BUCKETS,
        max_tokens: int | None = None,
        warmup: bool = True,
        orders: int = 1,
        prompt: Prompt = "tjev",
        noul_order: NoulOrder = "shuffled",
    ):

        from tjev.config import RunConfig
        from tjev.data.pack import Vocab
        from tjev.data.tokenize import PromptTokenizer
        from tjev.eval.runs import LoadedRun, base_identity
        from tjev.model import build_model
        from tjev.sharding import make_mesh, place_frozen
        from tjev.train.step import jit_eval_step, make_eval_step, split_model

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
            self.eval_fn = jit_eval_step(make_eval_step(graphdef))
            self.tok = PromptTokenizer(model_path)
            self.name = f"base:{Path(model_path).name}"
            identity = base_identity(model_path)
        else:
            raise ValueError("JaxBackend needs run_dir or model_path")
        if prompt not in ("tjev", "semif"):
            raise ValueError(f"unknown prompt {prompt!r}")
        if prompt != "tjev" and calibration:
            raise ValueError("calibration artifacts are fitted with tjev's prompt template")
        if prompt != "tjev" and noul_order != "shuffled":
            raise ValueError("noul_order applies to tjev's prompt (semif has its own order)")
        self.prompt, self.orders, self.noul_order = prompt, orders, noul_order
        self.spec = RenderSpec(train=False, prompt=prompt, noul_order=noul_order)
        eval_specs(parse_item(_FILLER), self.spec, orders)  # validates orders
        self.temperatures = _temperatures(calibration, identity, "jax", orders, noul_order)
        self.name += (
            ("" if prompt == "tjev" else f"+{prompt}")
            + (f"+orders{orders}" if orders > 1 else "")
            + ("+noul-fixed" if noul_order == "fixed" else "")
        )
        self.buckets = tuple(sorted(b for b in buckets if max_tokens is None or b <= max_tokens))
        if not self.buckets:
            raise ValueError(f"no serving bucket within max_tokens={max_tokens}")
        self.vocab = Vocab({"source": ["request"], "lang": ["en"], "family": ["unknown"]})
        if warmup:
            self.warmup()

    def warmup(self) -> None:
        """Compile every bucket once, so no request pays a compile."""
        for length in self.buckets:
            text = "x " * max(1, length * 3 // 4 - 150)  # lands in this bucket
            item = parse_item({**_FILLER, "state": text})
            # a filler that does not fit leaves that bucket to compile on first use
            with contextlib.suppress(ValueError):
                self.score([item])

    def score(self, items: list[Item]) -> tuple[list[np.ndarray], dict]:
        import jax

        from tjev.data.pack import make_segment, pack_eval
        from tjev.eval.evalset import pad_rows
        from tjev.sharding import batch_sharding

        t0 = time.perf_counter()
        segs, owner, orders = [], [], []
        for i, item in enumerate(items):
            for spec in self.specs(item):
                segs.append(make_segment(item, self.tok, spec, self.vocab, len(segs)))
                orders.append(choose_order(item, spec, None)[0])
                owner.append(i)
        longest = max(len(s.tokens) for s in segs)
        # pack into the smallest bucket that fits: no fixed padding for small requests
        batches, dropped = pack_eval(segs, self.buckets, max(longest, self.buckets[0]),
                                     self.cfg.train.max_segments, self.cfg.train.max_labels)  # fmt: skip
        if dropped:
            raise ValueError(f"state too long for the largest bucket ({self.buckets[-1]} tokens): "
                             "too many tokens")  # fmt: skip
        total = [np.zeros(len(item.question.labels)) for item in items]
        temps = [temperature(self.temperatures, item) for item in items]
        passes = [0] * len(items)
        sharding = batch_sharding(self.mesh, stacked=False)
        for b in (pad_rows(b, self.mesh.size) for b in batches):
            logits = np.asarray(self.eval_fn(self.lora, self.frozen, jax.device_put(b, sharding)))
            for r, s in zip(*np.nonzero(b.index >= 0), strict=True):
                j, k = int(b.index[r, s]), int(b.label_mask[r, s].sum())
                i = owner[j]
                total[i] += label_probs(
                    logits[r, s, :k], orders[j], len(items[i].question.labels), temps[i]
                )
                passes[i] += 1
        usage = {"prompt_tokens": int(sum(len(s.tokens) for s in segs)),
                 "seconds": time.perf_counter() - t0}  # fmt: skip
        if not all(passes):
            raise RuntimeError("an item was not scored")
        return [p / n for p, n in zip(total, passes, strict=True)], usage

    def specs(self, item: Item) -> list[RenderSpec]:
        """The render specs an item is scored under: one, or its option rotations
        (:func:`tjev.data.render.eval_specs`, the same ones the eval averages)."""
        return eval_specs(item, self.spec, self.orders)


class MlxBackend:
    """Apple silicon: an MLX model from ``tjev mlx convert`` (calibrated on MLX)."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        calibration: str | Path | None = None,
        orders: int = 1,
        noul_order: NoulOrder = "shuffled",
    ):
        from tjev.export.mlx import Scorer, mlx_identity

        self.scorer = Scorer(model_path)
        self.name = f"mlx:{Path(model_path).name}"
        self.orders, self.noul_order = orders, noul_order
        self.spec = RenderSpec(train=False, noul_order=noul_order)
        eval_specs(parse_item(_FILLER), self.spec, orders)  # validates orders
        self.temperatures = _temperatures(calibration, mlx_identity(model_path), "mlx",
                                          orders, noul_order)  # fmt: skip

    def score(self, items: list[Item]) -> tuple[list[np.ndarray], dict]:
        t0 = time.perf_counter()
        out, tokens = [], 0
        for item in items:  # one prompt at a time: MLX prefill is compute-bound anyway
            t = temperature(self.temperatures, item)
            views = eval_specs(item, self.spec, self.orders)
            total = np.zeros(len(item.question.labels))
            for spec in views:
                text, order = render(item, spec)
                z, n = self.scorer.logits(text, len(order))
                tokens += n
                total += label_probs(z, order, len(item.question.labels), t)
            out.append(total / len(views))
        return out, {"prompt_tokens": tokens, "seconds": time.perf_counter() - t0}
