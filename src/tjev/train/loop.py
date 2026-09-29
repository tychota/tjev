"""The training loop: data stream → jitted step → async metrics, evals, checkpoints.

Run directory layout (``<output>/<name>/``)::

    config.json      resolved RunConfig + identity (model snapshot, data files, template)
    vocab.json       metadata id tables;  mixture.json  source probabilities
    metrics.jsonl    learning/* perf/* data/* eval*/*;  tb/  TensorBoard
    checkpoints/     Orbax (adapters, optimizer, data-stream state)
    eval/step-N.json grouped validation reports (step-N-heldout.json: data.heldout)
    selection.json   the selected (best) step and its scores

Checkpoint selection: NLL after one temperature fitted on the even validation slots,
scored on the odd ones and averaged with the held-out generators (temperatures are applied
after training anyway, and ECE is not a proper score: arXiv 2501.19195, 2309.12236).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import numpy as np

from tjev.config import RunConfig
from tjev.config.loader import from_dict
from tjev.data.item import Item, read_jsonl
from tjev.data.mixture import Mixture
from tjev.data.pack import Batch, Vocab, rows_for
from tjev.data.pipeline import train_iterator
from tjev.data.render import TEMPLATE_VERSION
from tjev.data.segments import SegmentAt
from tjev.data.tokenize import PromptTokenizer
from tjev.eval.calibrate import fit_temperature, nll
from tjev.eval.evalset import EvalSet
from tjev.model import build_model, snapshot_identity
from tjev.sharding import (
    batch_sharding,
    install_mesh,
    make_mesh,
    place_frozen,
    place_replicated,
)
from tjev.train import flops as flops_mod
from tjev.train.checkpoint import Checkpoints
from tjev.train.log import MetricLogger
from tjev.train.optim import make_optimizer, schedule
from tjev.train.step import make_eval_step, make_train_step, split_model
from tjev.utils import file_hash, fingerprint, write_json

EXPECTED_SLOTS_STEPS = 64
# Fields that change how a run is logged, evaluated or checkpointed, never what it trains:
# they may differ across a resume (or between a run and its cooldown branch).
NOT_IDENTITY = {
    "train": ("steps", "log_every", "quick_eval_every", "checkpoint_every",
              "checkpoint_secs", "keep_checkpoints", "profile_start", "profile_steps",
              "branch_from"),
    "data": ("workers", "worker_buffer", "validation", "validation_per_source", "heldout",
             "quick_validation_per_source"),
}  # fmt: skip


def load_sources(paths: tuple[str, ...]) -> dict[str, list[Item]]:
    sources: dict[str, list[Item]] = defaultdict(list)
    for path in paths:
        for item in read_jsonl(path):
            sources[item.source].append(item)
    return dict(sources)


def expected_slots_per_step(stream: Iterator[Batch], steps: int = EXPECTED_SLOTS_STEPS) -> float:
    """Mean filled slots per step over the first ``steps`` steps of a fresh stream (the
    run's own seed, mixture and packing, so it is deterministic and resume-safe)."""
    try:
        total = sum(float(np.asarray(next(stream).weight).sum()) for _ in range(steps))
    finally:
        getattr(stream, "close", lambda: None)()
    return round(total / steps, 1)


def run_identity(cfg: RunConfig, model_identity: str) -> dict:
    """What must not change across a resume: the training itself."""
    raw = cfg.to_dict()
    for section, keys in NOT_IDENTITY.items():
        for key in keys:
            raw[section].pop(key)
    for key in ("name", "output", "log"):
        raw.pop(key)
    return {
        "config": fingerprint(raw),
        "model": model_identity,
        "data": fingerprint({p: file_hash(p) for p in cfg.data.train}),
        "template": TEMPLATE_VERSION,
    }


def first_per_source(items: list[Item], n: int) -> list[Item]:
    seen: dict[str, int] = defaultdict(int)
    out = []
    for item in items:
        if seen[item.source] < n:
            seen[item.source] += 1
            out.append(item)
    return out


Slots = tuple[np.ndarray, np.ndarray, np.ndarray]


def calibrated_nll(validation: Slots, heldout: Slots | None = None) -> dict[str, float]:
    """NLL after one temperature fitted on the even validation slots: on the odd ones and
    (if given) on the held-out slots; ``score`` is their mean. Slots: (logits, mask, target)."""
    logits, mask, target = validation
    fit = np.arange(len(logits)) % 2 == 0
    temperature, *_ = fit_temperature(logits[fit], mask[fit], target[fit])
    out = {"temperature": temperature,
           "val_nll": nll(logits[~fit], mask[~fit], target[~fit], temperature)}  # fmt: skip
    parts = [out["val_nll"]]
    if heldout is not None:
        out["heldout_nll"] = nll(*heldout, temperature)
        parts.append(out["heldout_nll"])
    out["score"] = float(np.mean(parts))
    return out


def branch_source(cfg: RunConfig, identity: dict) -> tuple[Path, int]:
    """Parse and check ``train.branch_from``: the source run must be this run's training
    (same identity) with the same learning rates up to the branch step."""
    path, _, step_text = cfg.train.branch_from.rpartition("@")
    source, step = Path(path), int(step_text)
    saved = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if saved["identity"] != identity:
        raise ValueError(f"branch_from {source}: a different training (identity mismatch)")
    theirs_cfg = from_dict(saved["config"])
    theirs = schedule(theirs_cfg.optim, theirs_cfg.train.steps)
    ours = schedule(cfg.optim, cfg.train.steps)
    at = np.arange(step + 1)
    if not np.allclose(np.asarray(theirs(at)), np.asarray(ours(at)), rtol=1e-6, atol=0):
        raise ValueError(f"branch_from {source}@{step}: learning rates differ before the branch")
    return source, step


def code_version() -> str:
    """git HEAD of the source tree (hosts that get a ``git archive`` set CODE_VERSION)."""
    if os.environ.get("CODE_VERSION"):
        return os.environ["CODE_VERSION"]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=Path(__file__).parent, timeout=5, check=False)  # fmt: skip
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def eval_scalars(report: dict) -> dict:
    return {"all": report["all"], **{g: report.get(g, {}) for g in ("type", "lang")}}


def train(cfg: RunConfig) -> dict:
    run_dir = Path(cfg.output) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    mesh = make_mesh(cfg.mesh)
    install_mesh(cfg.compute, mesh)
    n_dev = mesh.size
    tok = PromptTokenizer(cfg.model.path)
    sources = load_sources(cfg.data.train)
    validation = list(read_jsonl(cfg.data.validation)) if cfg.data.validation else []
    if cfg.data.validation_per_source:
        validation = first_per_source(validation, cfg.data.validation_per_source)
    heldout = list(read_jsonl(cfg.data.heldout)) if cfg.data.heldout else []
    vocab = Vocab.from_items(*sources.values(), validation, heldout)
    mixture = Mixture(sources, cfg.data.mixture or None, cfg.data.mixture_alpha)

    identity = run_identity(cfg, snapshot_identity(cfg.model.path)["identity"])
    config_path = run_dir / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError(f"{run_dir} belongs to a different run (identity mismatch)")
    write_json(config_path, {"config": cfg.to_dict(), "identity": identity})
    write_json(run_dir / "vocab.json", vocab.names)
    write_json(run_dir / "mixture.json", mixture.describe())

    config, model = build_model(
        cfg.model.path, cfg.compute, cfg.lora, dtype=cfg.model.dtype, seed=cfg.train.seed
    )
    graphdef, lora, frozen = split_model(model)
    del model
    frozen = place_frozen(frozen, mesh)
    lora = place_replicated(lora, mesh)

    # Accumulate microbatches (per device) until the global token budget is met.
    per_step = cfg.train.microbatch_tokens * n_dev
    accumulation = max(1, cfg.train.tokens_per_step // per_step)
    for length in cfg.train.seq_buckets:
        if rows_for(length, per_step) % n_dev:
            raise ValueError(f"bucket {length}: rows per microbatch not divisible by {n_dev}")
    tx = make_optimizer(cfg.optim, cfg.train.steps)
    opt_state = place_replicated(tx.init(lora), mesh)
    lr_at = schedule(cfg.optim, cfg.train.steps)
    vocab.frozen = True

    def make_stream(workers: int):
        return train_iterator(
            SegmentAt(mixture, tok, vocab, seed=cfg.train.seed),
            buckets=cfg.train.seq_buckets,
            microbatch_tokens=per_step,
            accumulation=accumulation,
            slots=cfg.train.max_segments,
            labels=cfg.train.max_labels,
            workers=workers,
            worker_buffer=cfg.data.worker_buffer,
            bins=cfg.train.packing_bins,
        )

    slots_per_step = expected_slots_per_step(make_stream(0))
    print(json.dumps({"event": "expected_slots_per_step", "value": slots_per_step}), flush=True)
    stream = make_stream(cfg.data.workers)

    ckpt = Checkpoints(run_dir / "checkpoints", cfg.train.keep_checkpoints)
    start = 0
    best: dict[str, Any] = {"step": None, "score": float("inf")}
    if (latest := ckpt.latest()) is not None:
        lora, opt_state, meta = ckpt.restore(latest, lora, opt_state)
        stream.set_state(meta["stream"])
        best = meta.get("best", best)
        start = latest
        print(json.dumps({"event": "resumed", "step": start}), flush=True)
    elif cfg.train.branch_from:
        source, start = branch_source(cfg, identity)
        other = Checkpoints(source / "checkpoints")
        lora, opt_state, meta = other.restore(start, lora, opt_state)
        other.close()
        stream.set_state(meta["stream"])  # best stays fresh: this run selects its own
        print(json.dumps({"event": "branched", "from": str(source), "step": start}), flush=True)

    step_fn = jax.jit(
        make_train_step(
            graphdef, tx, brier_weight=cfg.train.brier_weight, slots_per_step=slots_per_step
        ),
        donate_argnums=(0, 1),
    )
    eval_fn = jax.jit(make_eval_step(graphdef))
    evalset = EvalSet(validation, tok, vocab, cfg, mesh) if validation else None
    heldset = EvalSet(heldout, tok, vocab, cfg, mesh) if heldout else None
    quickset = None
    if validation and cfg.train.quick_eval_every:
        quick = first_per_source(validation, cfg.data.quick_validation_per_source)
        quickset = EvalSet(quick, tok, vocab, cfg, mesh)
    device_kind = jax.devices()[0].device_kind
    system = {
        "device_kind": device_kind,
        "devices": n_dev,
        "jax": jax.__version__,
        "code": code_version(),
        "tpu_visible_chips": os.environ.get("TPU_VISIBLE_CHIPS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
    }
    log = cfg.log
    logger = MetricLogger(
        run_dir,
        tensorboard=log.tensorboard,
        wandb=log.wandb,
        project=log.wandb_project,
        config=cfg.to_dict() | {"identity": identity, "system": system},
        name=cfg.name,
        resume_step=start,
        entity=log.wandb_entity,
        group=log.wandb_group,
        tags=log.wandb_tags,
        mode=log.wandb_mode,
    )
    window = MetricWindow(logger, cfg, n_dev, flops_mod.peak_flops(device_kind), lr_at)
    in_sharding = batch_sharding(mesh, stacked=True)
    profiling = False
    stream_state = stream.get_state()
    stop = threading.Event()  # SIGTERM (preemption, queue shutdown): save, then exit 143
    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: stop.set())
    t_saved = time.monotonic()
    try:
        for step in range(start + 1, cfg.train.steps + 1):
            if cfg.train.profile_start and step == cfg.train.profile_start:
                jax.profiler.start_trace(str(run_dir / "profile"))
                profiling = True
            t_wait = time.perf_counter()
            batch = next(stream)
            wait = time.perf_counter() - t_wait
            stream_state = stream.get_state()  # the state that goes with this step
            lengths = flops_mod.segment_lengths(batch.segment_ids)
            step_flops = flops_mod.train_flops(
                config, lengths, cfg.compute.gdn_chunk, int(batch.label_mask.sum())
            )
            lora, opt_state, metrics = step_fn(
                lora, opt_state, frozen, jax.device_put(batch, in_sharding)
            )
            window.add(step, metrics, batch, step_flops, wait, stream_state["next_index"])
            if profiling and step >= cfg.train.profile_start + cfg.train.profile_steps:
                jax.block_until_ready(metrics)
                jax.profiler.stop_trace()
                profiling = False

            due = step % cfg.train.checkpoint_every == 0 or step == cfg.train.steps
            full_eval = bool(evalset) and due
            quick_eval = bool(quickset) and not due and step % cfg.train.quick_eval_every == 0
            synced = due or quick_eval or stop.is_set()
            window.flush(everything=synced)

            if quickset and quick_eval:
                report = quickset.run(eval_fn, lora, frozen, mesh, vocab)
                logger.log(step, {"eval_quick": eval_scalars(report)})
            new_best = False
            if evalset and full_eval:
                report = evalset.run(eval_fn, lora, frozen, mesh, vocab)
                write_json(run_dir / "eval" / f"step-{step}.json", report)
                logger.log(step, {"eval": eval_scalars(report)})
                held = None
                if heldset:
                    held = heldset.run(eval_fn, lora, frozen, mesh, vocab)
                    write_json(run_dir / "eval" / f"step-{step}-heldout.json", held)
                    logger.log(step, {"eval_heldout": eval_scalars(held)})
                cal = calibrated_nll(evalset.last, heldset.last if heldset else None)
                logger.log(step, {"eval_calibrated": cal})
                if cal["score"] < best["score"]:
                    new_best = True
                    best = {"step": step, "score": cal["score"], "all": report["all"],
                            "calibrated": cal}  # fmt: skip
                    if held is not None:
                        best["heldout"] = held["all"]

            # Full evals happen at checkpoint steps, so the best step is always checkpointed; it
            # is never pruned, and selection.json names it only once its checkpoint is written.
            # Time-based and SIGTERM saves are for resuming only (no eval).
            timed = cfg.train.checkpoint_secs and (
                time.monotonic() - t_saved > cfg.train.checkpoint_secs
            )
            if due or timed or stop.is_set():
                meta = {"stream": stream_state, "best": best}
                ckpt.save(step, lora, opt_state, meta, keep=best["step"])
                t_saved = time.monotonic()
            if new_best:
                ckpt.wait()
                write_json(run_dir / "selection.json", best)
                logger.summary({"best": best})
            if stop.is_set() and step < cfg.train.steps:
                ckpt.wait()
                print(json.dumps({"event": "interrupted", "step": step}), flush=True)
                raise SystemExit(143)  # resumes exactly from this checkpoint
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
        stream.close()
        ckpt.wait()
        ckpt.close()
        logger.close()
    return {"run": str(run_dir), "best": best, "steps": cfg.train.steps}


class MetricWindow:
    """Device metrics fetched one step late (while the next step runs), so logging every
    step does not stall the device on the host; evals, checkpoints and the last step flush
    everything."""

    def __init__(self, logger: MetricLogger, cfg: RunConfig, devices: int, peak, lr_at):
        self.logger, self.cfg, self.devices, self.peak, self.lr_at = (
            logger,
            cfg,
            devices,
            peak,
            lr_at,
        )
        self.pending: list[dict] = []
        self.shapes: set[tuple[int, ...]] = set()
        self.compiled = False
        self.t_last = time.perf_counter()

    def add(self, step: int, metrics: dict, batch: Batch, flops: float, wait: float, segments: int):
        self.compiled |= batch.shape_key not in self.shapes
        self.shapes.add(batch.shape_key)
        self.pending.append(
            {
                "step": step,
                "metrics": metrics,
                "flops": flops,
                "pad": 1.0 - float((batch.segment_ids > 0).mean()),
                "wait": wait,
                "segments": segments,
            }
        )

    def flush(self, *, everything: bool) -> None:
        if not everything and len(self.pending) <= self.cfg.train.log_every:
            return
        ready = self.pending if everything else self.pending[:-1]
        if not ready:
            return
        fetched = jax.device_get([p["metrics"] for p in ready])
        now = time.perf_counter()
        per_step = (now - self.t_last) / len(ready)
        self.t_last = now
        every = self.cfg.train.log_every
        for i in range(0, len(ready), every):
            self._row(ready[i : i + every], fetched[i : i + every], per_step)
            self.compiled = False
        self.pending = self.pending[len(ready) :]

    def _row(self, group: list[dict], values: list[dict], per_step: float) -> None:
        step = group[-1]["step"]
        elapsed = per_step * len(group)
        total_flops = sum(p["flops"] for p in group)
        tokens = sum(float(m["tokens"]) for m in values)
        learning = {
            k: float(np.mean([m[k] for m in values]))
            for k in ("loss", "nll", "brier", "accuracy", "grad_norm")
        }
        learning["lr"] = float(self.lr_at(step))
        learning["update_rejected"] = float(sum(m["update_rejected"] for m in values))
        perf = {
            "step_seconds": per_step,
            "tokens_per_second_per_device": tokens / elapsed / self.devices,
            "tflops_per_device": total_flops / elapsed / self.devices / 1e12,
            "compiled_shapes": len(self.shapes),
            "includes_compile": float(self.compiled),  # 1: not steady-state timing
        }
        if self.peak:
            perf["mfu"] = total_flops / elapsed / self.devices / self.peak
        stats = jax.local_devices()[0].memory_stats() or {}
        if "peak_bytes_in_use" in stats:
            perf["hbm_peak_gb"] = stats["peak_bytes_in_use"] / 1e9
        record = {
            "learning": learning,
            "perf": perf,
            "data": {
                "pad_fraction": float(np.mean([p["pad"] for p in group])),
                "slots_per_step": float(values[-1]["slots"]),
                "segments_consumed": group[-1]["segments"],
                # host time blocked on the stream (> ~5% of step_seconds: add data.workers)
                "wait_seconds": float(np.mean([p["wait"] for p in group])),
            },
        }
        self.logger.log(step, record)
        print(json.dumps({"step": step, **learning, **perf}), flush=True)
