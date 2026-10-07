"""Finished runs: restore a checkpoint, evaluate, calibrate; the zero-shot base-model control."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import jax
from flax import nnx

from tjev.config import ComputeSpec, RunConfig
from tjev.config.loader import from_dict
from tjev.data.item import Item, read_jsonl
from tjev.data.pack import Vocab
from tjev.data.render import TEMPLATE_VERSION, NoulOrder
from tjev.data.tokenize import PromptTokenizer
from tjev.eval import calibrate as cal
from tjev.eval import predictions as dump
from tjev.eval.evalset import EvalSet
from tjev.model import Qwen35, build_model, snapshot_identity
from tjev.model.lora_init import INIT_FILE, load_init, restore_init
from tjev.sharding import (
    configure_runtime,
    install_mesh,
    make_mesh,
    place_frozen,
    place_replicated,
)
from tjev.train.checkpoint import Checkpoints
from tjev.train.optim import ema_params, make_optimizer
from tjev.train.step import jit_eval_step, make_eval_step, split_model
from tjev.utils import file_hash, fingerprint, write_json

Data = str | Path | Sequence[str | Path]


def pool_files(data: Data) -> list[Path]:
    """The files of a data argument: a JSONL file, a directory (its ``*.jsonl``, sorted),
    or a list of either (a calibration pool)."""
    paths = [data] if isinstance(data, str | Path) else list(data)
    out: list[Path] = []
    for p in map(Path, paths):
        out += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    if not out:
        raise FileNotFoundError(f"no JSONL file in {data}")
    return out


def read_pool(data: Data) -> list[Item]:
    """The items of every file of ``data`` (:func:`pool_files`); across files, an id is
    kept once (by source, id and type; one file is read as is)."""
    files = pool_files(data)
    if len(files) == 1:
        return list(read_jsonl(files[0]))
    seen: set[tuple[str, str, str]] = set()
    items = []
    for path in files:
        for item in read_jsonl(path):
            key = (item.source, item.id, item.question.type)
            if key not in seen:
                seen.add(key)
                items.append(item)
    return items


def data_fingerprint(data: Data, n: int) -> str:
    """What a calibration was fitted on: one file's hash (as before pools), or every
    pool file's."""
    files = pool_files(data)
    if isinstance(data, str | Path) and Path(data).is_file():
        return fingerprint({"file": file_hash(data), "n": n})
    return fingerprint({"files": {p.name: file_hash(p) for p in files}, "n": n})


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
        # the selected step's weights: the adapters, or their EMA (train.select_ema)
        self.weights = "raw"
        if step is None:
            selection = self.dir / "selection.json"
            chosen = json.loads(selection.read_text()) if selection.exists() else {}
            step = chosen.get("step")
            self.weights = chosen.get("weights", "raw")
        ckpt = Checkpoints(self.dir / "checkpoints")
        latest = ckpt.latest()
        self.step: int = step if step is not None else (latest if latest is not None else -1)
        if self.step < 0:
            raise FileNotFoundError(f"{self.dir} has no checkpoint")
        self.config, model = build_model(
            cfg.model.path, cfg.compute, cfg.lora, dtype=cfg.model.dtype
        )
        # lora.init: A0, B0 (and the frozen offset A0, B0 of pissa / lora-ga)
        self.init: str = "default"
        self.init_adapters: dict | None = None
        if (init_path := self.dir / INIT_FILE).exists():
            self.init = restore_init(model, init_path)
            self.init_adapters = load_init(init_path)[1]
        self.mesh = make_mesh(cfg.mesh)
        install_mesh(cfg.compute, self.mesh)
        self.graphdef, lora, frozen = split_model(model)
        del model
        if self.weights == "ema":  # the EMA adapters live in the optimizer state
            tx = make_optimizer(cfg.optim, cfg.train.steps, cfg.loss)
            self.lora, opt, _ = ckpt.restore(self.step, lora, tx.init(lora))
            ema = ema_params(opt)
            if ema is None:
                raise ValueError(f"{self.dir}: selection.json names EMA weights, none saved")
            self.lora = jax.tree.map(lambda e, p: e.astype(p.dtype), ema, self.lora)
            del opt
        else:  # the adapters only (a backed-up intermediate checkpoint has no optimizer state)
            self.lora, _ = ckpt.restore_lora(self.step, lora)
        ckpt.close()
        self.frozen = place_frozen(frozen, self.mesh)
        self.lora = place_replicated(self.lora, self.mesh)
        self.eval_fn = jit_eval_step(make_eval_step(self.graphdef))
        self.tok = PromptTokenizer(cfg.model.path)

    @property
    def checkpoint_identity(self) -> str:
        """What calibration artifacts bind to: the run's identity and the step."""
        body = {"run": self.identity, "step": self.step}
        return fingerprint(body | ({"weights": "ema"} if self.weights == "ema" else {}))

    def evalset(
        self, path: Data, *, orders: int = 1, noul_order: NoulOrder = "shuffled"
    ) -> EvalSet:
        """``path``: a JSONL file, a directory of them or a list (:func:`read_pool`)."""
        items = read_pool(path)
        for item in items:  # unseen sources / languages get fresh ids (appended, stable)
            self.vocab.id("source", item.source)
            self.vocab.id("lang", item.lang)
            self.vocab.id("family", item.family)
        return EvalSet(items, self.tok, self.vocab, self.cfg, self.mesh, orders=orders,
                       noul_order=noul_order)  # fmt: skip

    def model(self) -> Qwen35:
        return nnx.merge(self.graphdef, self.lora, self.frozen)


def evaluate(
    run_dir: str | Path,
    data: str | Path,
    *,
    step: int | None = None,
    calibration: str | Path | None = None,
    orders: int = 1,
    noul_order: NoulOrder = "shuffled",
    predictions: str | Path | None = None,
) -> dict:
    """Grouped report of a run on ``data`` (raw, or with a calibration artifact's
    temperatures, which must have been fitted on this very checkpoint and readout:
    the same ``orders`` and ``noul_order``). ``predictions``: also dump the per-item
    predictions there (:mod:`tjev.eval.predictions`)."""
    run = LoadedRun(run_dir, step)
    temperatures = None
    if calibration:
        art = json.loads(Path(calibration).read_text(encoding="utf-8"))
        temperatures = cal.validate(
            art, checkpoint=run.checkpoint_identity, template=TEMPLATE_VERSION, backend="jax",
            orders=orders, noul_order=noul_order,
        )  # fmt: skip
    evalset = run.evalset(data, orders=orders, noul_order=noul_order)
    report = evalset.run(run.eval_fn, run.lora, run.frozen, run.mesh, run.vocab, temperatures)
    if predictions:
        dump.write(predictions, evalset.predictions())
    return report | {"step": run.step, "temperatures": temperatures}


def calibrate(
    run_dir: str | Path,
    data: Data,
    *,
    step: int | None = None,
    out: str | Path | None = None,
    orders: int = 1,
    noul_order: NoulOrder = "shuffled",
    grouping: str = cal.DEFAULT_GROUPING,
    predictions: str | Path | None = None,
) -> dict:
    """Fit temperatures (``grouping``: :data:`tjev.eval.calibrate.GROUPINGS`) on ``data``
    (the mix calibration split, never JevBench; or a *calibration pool*: several files or
    a directory of held-out sources, :func:`read_pool`, recorded in ``fit.pool``), on the
    order-averaged distributions of ``orders`` orderings (the artifact records the readout
    and the grouping; it is valid only for the same readout). ``predictions``: also dump
    the split's per-item raw predictions there."""
    run = LoadedRun(run_dir, step)
    evalset = run.evalset(data, orders=orders, noul_order=noul_order)
    arrays = evalset.arrays(run.eval_fn, run.lora, run.frozen, run.mesh, run.vocab)
    if predictions:
        dump.write(predictions, evalset.predictions())
    fit = cal.fit_grouped(arrays["logits"], arrays["mask"], arrays["target"],
                          list(arrays["type"]), list(arrays["lang"]), grouping=grouping)  # fmt: skip
    if not (isinstance(data, str | Path) and Path(data).is_file()):
        fit["pool"] = {"files": [p.name for p in pool_files(data)],
                       "sources": dict(Counter(arrays["source"]))}  # fmt: skip
    art = cal.artifact(
        fit,
        checkpoint=run.checkpoint_identity,
        data=data_fingerprint(data, len(arrays["logits"])),
        template=TEMPLATE_VERSION,
        backend="jax",
        orders=orders,
        noul_order=noul_order,
    )
    name = f"calibration-step-{run.step}{readout_tag(orders, noul_order)}.json"
    write_json(out or run.dir / name, art)
    return art


def predict(
    run_dir: str | Path,
    data: str | Path,
    out: str | Path,
    *,
    step: int | None = None,
    orders: int = 1,
    noul_order: NoulOrder = "shuffled",
) -> Path:
    """Only the per-item raw predictions (T = 1) of a run on ``data``."""
    run = LoadedRun(run_dir, step)
    evalset = run.evalset(data, orders=orders, noul_order=noul_order)
    evalset.run(run.eval_fn, run.lora, run.frozen, run.mesh, run.vocab)
    return dump.write(out, evalset.predictions())


def readout_tag(orders: int, noul_order: str) -> str:
    """File-name suffix of a non-default readout ("" for K = 1, shuffled noul)."""
    tag = f"-k{orders}" if orders != 1 else ""
    return tag + ("-noul-fixed" if noul_order == "fixed" else "")


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
    orders: int = 1,
    noul_order: NoulOrder = "shuffled",
) -> dict:
    """Zero-shot control: the untrained base model with the same prompt and readout.

    If ``calibrate_on`` is given, temperatures (the default grouping) are fitted there (never on
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
    eval_fn = jit_eval_step(make_eval_step(graphdef))
    tok = PromptTokenizer(cfg.model.path)
    items = first_per_source(list(read_jsonl(data)), per_source)
    cal_items = first_per_source(list(read_jsonl(calibrate_on)), per_source) if calibrate_on else []
    vocab = Vocab.from_items(items, cal_items)
    evalset = EvalSet(items, tok, vocab, cfg, mesh, orders=orders, noul_order=noul_order)
    out: dict = {"model": str(model_path), "raw": evalset.run(eval_fn, lora, frozen, mesh, vocab)}
    if calibrate_on:
        cal_set = EvalSet(cal_items, tok, vocab, cfg, mesh, orders=orders, noul_order=noul_order)
        arrays = cal_set.arrays(eval_fn, lora, frozen, mesh, vocab)
        fit = cal.fit_grouped(arrays["logits"], arrays["mask"], arrays["target"],
                              list(arrays["type"]), list(arrays["lang"]))  # fmt: skip
        out["calibrated"] = evalset.run(eval_fn, lora, frozen, mesh, vocab, fit["temperatures"])
        out["temperatures"] = fit["temperatures"]
        out["calibration_fit"] = fit
        out["calibration_artifact"] = cal.artifact(
            fit,
            checkpoint=base_identity(model_path),
            data=fingerprint({"file": file_hash(calibrate_on), "per_source": per_source}),
            template=TEMPLATE_VERSION,
            backend="jax",
            orders=orders,
            noul_order=noul_order,
        )
    return out
