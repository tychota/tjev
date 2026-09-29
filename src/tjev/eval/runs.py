"""Finished runs: restore a checkpoint, evaluate, calibrate; the zero-shot base-model control."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
from flax import nnx

from tjev.config import ComputeSpec, RunConfig
from tjev.config.loader import from_dict
from tjev.data.item import Item, read_jsonl
from tjev.data.pack import Vocab
from tjev.data.render import TEMPLATE_VERSION
from tjev.data.tokenize import PromptTokenizer
from tjev.eval import calibrate as cal
from tjev.eval.evalset import EvalSet
from tjev.model import Qwen35, build_model, snapshot_identity
from tjev.sharding import (
    configure_runtime,
    install_mesh,
    make_mesh,
    place_frozen,
    place_replicated,
)
from tjev.train.checkpoint import Checkpoints
from tjev.train.optim import make_optimizer
from tjev.train.step import make_eval_step, split_model
from tjev.utils import file_hash, fingerprint, write_json


class LoadedRun:
    """A run's config, adapters (selected step by default) and a jitted eval step.

    ``compute`` / ``dtype`` override the run's (e.g. fp32 XLA reference kernels for parity
    references); the adapters and their placement are unchanged."""

    def __init__(
        self,
        run_dir: str | Path,
        step: int | None = None,
        *,
        compute: ComputeSpec | None = None,
        dtype: str | None = None,
    ):
        self.dir = Path(run_dir)
        saved = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        cfg = from_dict(saved["config"])
        if compute is not None:
            cfg = replace(cfg, compute=compute)
        if dtype is not None:
            cfg = replace(cfg, model=replace(cfg.model, dtype=dtype))
        self.cfg: RunConfig = cfg
        configure_runtime(cfg.compute)
        self.identity = saved["identity"]
        self.vocab = Vocab(json.loads((self.dir / "vocab.json").read_text(encoding="utf-8")))
        if step is None:
            selection = self.dir / "selection.json"
            step = json.loads(selection.read_text())["step"] if selection.exists() else None
        ckpt = Checkpoints(self.dir / "checkpoints")
        latest = ckpt.latest()
        self.step: int = step if step is not None else (latest if latest is not None else -1)
        if self.step < 0:
            raise FileNotFoundError(f"{self.dir} has no checkpoint")
        self.config, model = build_model(
            cfg.model.path, cfg.compute, cfg.lora, dtype=cfg.model.dtype
        )
        self.mesh = make_mesh(cfg.mesh)
        install_mesh(cfg.compute, self.mesh)
        self.graphdef, lora, frozen = split_model(model)
        del model
        tx = make_optimizer(cfg.optim, cfg.train.steps)
        self.lora, _, _ = ckpt.restore(self.step, lora, tx.init(lora))
        ckpt.close()
        self.frozen = place_frozen(frozen, self.mesh)
        self.lora = place_replicated(self.lora, self.mesh)
        self.eval_fn = jax.jit(make_eval_step(self.graphdef))
        self.tok = PromptTokenizer(cfg.model.path)

    @property
    def checkpoint_identity(self) -> str:
        """What calibration artifacts bind to: the run's identity and the step."""
        return fingerprint({"run": self.identity, "step": self.step})

    def evalset(self, path: str | Path) -> EvalSet:
        items = list(read_jsonl(path))
        for item in items:  # unseen sources / languages get fresh ids (appended, stable)
            self.vocab.id("source", item.source)
            self.vocab.id("lang", item.lang)
            self.vocab.id("family", item.family)
        return EvalSet(items, self.tok, self.vocab, self.cfg, self.mesh)

    def model(self) -> Qwen35:
        return nnx.merge(self.graphdef, self.lora, self.frozen)


def evaluate(
    run_dir: str | Path,
    data: str | Path,
    *,
    step: int | None = None,
    calibration: str | Path | None = None,
) -> dict:
    """Grouped report of a run on ``data`` (raw, or with a calibration artifact's
    temperatures, which must have been fitted on this very checkpoint)."""
    run = LoadedRun(run_dir, step)
    temperatures = None
    if calibration:
        art = json.loads(Path(calibration).read_text(encoding="utf-8"))
        temperatures = cal.validate(
            art, checkpoint=run.checkpoint_identity, template=TEMPLATE_VERSION, backend="jax"
        )
    report = run.evalset(data).run(
        run.eval_fn, run.lora, run.frozen, run.mesh, run.vocab, temperatures
    )
    return report | {"step": run.step, "temperatures": temperatures}


def calibrate(
    run_dir: str | Path,
    data: str | Path,
    *,
    step: int | None = None,
    out: str | Path | None = None,
) -> dict:
    """Fit per-type temperatures on ``data`` (the mix calibration split, never JevBench)."""
    run = LoadedRun(run_dir, step)
    arrays = run.evalset(data).arrays(run.eval_fn, run.lora, run.frozen, run.mesh, run.vocab)
    fit = cal.fit_per_type(arrays["logits"], arrays["mask"], arrays["target"], list(arrays["type"]))
    art = cal.artifact(
        fit,
        checkpoint=run.checkpoint_identity,
        data=fingerprint({"file": file_hash(data), "n": len(arrays["logits"])}),
        template=TEMPLATE_VERSION,
        backend="jax",
    )
    write_json(out or run.dir / f"calibration-step-{run.step}.json", art)
    return art


def base_identity(model_path: str | Path) -> str:
    """Checkpoint identity of an untrained base snapshot."""
    return "base:" + fingerprint(snapshot_identity(model_path))


def first_per_source(items: list[Item], n: int) -> list[Item]:
    if not n:
        return items
    seen: dict[str, int] = {}
    kept = []
    for item in items:
        if seen.get(item.source, 0) < n:
            seen[item.source] = seen.get(item.source, 0) + 1
            kept.append(item)
    return kept


def evaluate_base(
    model_path: str | Path,
    data: str | Path,
    *,
    calibrate_on: str | Path | None = None,
    cfg: RunConfig | None = None,
    per_source: int = 0,
) -> dict:
    """Zero-shot control: the untrained base model with the same prompt and readout.

    If ``calibrate_on`` is given, per-type temperatures are fitted there (never on
    ``data``) and applied to ``data``, as the JevBench raw-logit controls do."""
    cfg = cfg or RunConfig()
    cfg = replace(cfg, model=replace(cfg.model, path=str(model_path)))
    configure_runtime(cfg.compute)
    _, model = build_model(cfg.model.path, cfg.compute, None, dtype=cfg.model.dtype)
    mesh = make_mesh(cfg.mesh)
    install_mesh(cfg.compute, mesh)
    graphdef, lora, frozen = split_model(model)
    del model
    frozen = place_frozen(frozen, mesh)
    eval_fn = jax.jit(make_eval_step(graphdef))
    tok = PromptTokenizer(cfg.model.path)
    items = first_per_source(list(read_jsonl(data)), per_source)
    cal_items = first_per_source(list(read_jsonl(calibrate_on)), per_source) if calibrate_on else []
    vocab = Vocab.from_items(items, cal_items)
    evalset = EvalSet(items, tok, vocab, cfg, mesh)
    out: dict = {"model": str(model_path), "raw": evalset.run(eval_fn, lora, frozen, mesh, vocab)}
    if calibrate_on:
        arrays = EvalSet(cal_items, tok, vocab, cfg, mesh).arrays(
            eval_fn, lora, frozen, mesh, vocab
        )
        fit = cal.fit_per_type(
            arrays["logits"], arrays["mask"], arrays["target"], list(arrays["type"])
        )
        out["calibrated"] = evalset.run(eval_fn, lora, frozen, mesh, vocab, fit["temperatures"])
        out["temperatures"] = fit["temperatures"]
        out["calibration_fit"] = fit
        out["calibration_artifact"] = cal.artifact(
            fit,
            checkpoint=base_identity(model_path),
            data=fingerprint({"file": file_hash(calibrate_on), "per_source": per_source}),
            template=TEMPLATE_VERSION,
            backend="jax",
        )
    return out
