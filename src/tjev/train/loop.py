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
import shutil
import signal
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx

from tjev.config import RunConfig
from tjev.config.loader import from_dict
from tjev.config.schema import GATES
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
from tjev.model.lora_init import INIT_FILE, initialize, restore_init, save_init
from tjev.sharding import (
    batch_sharding,
    configure_runtime,
    install_mesh,
    libtpu_settings,
    make_mesh,
    place_frozen,
    place_replicated,
)
from tjev.train import flops as flops_mod
from tjev.train.checkpoint import Checkpoints
from tjev.train.health import HealthMonitor
from tjev.train.log import MetricLogger
from tjev.train.objective import loss_sums, source_table
from tjev.train.optim import ema_params, make_optimizer, schedule, unit_schedule
from tjev.train.step import (
    jit_eval_step,
    jit_train_step,
    make_eval_step,
    make_train_step,
    split_model,
    train_step_name,
)
from tjev.utils import file_hash, fingerprint, write_json

EXPECTED_SLOTS_STEPS = 64
STOP_FILE = "STOP"  # touch <run>/STOP: the run ends at its next checkpoint, as completed
# Fields that change how a run is logged, evaluated or checkpointed, never what it trains:
# they may differ across a resume (or between a run and its cooldown branch).
NOT_IDENTITY = {
    "train": ("steps", "log_every", "quick_eval_every", "checkpoint_every",
              "checkpoint_secs", "keep_checkpoints", "profile_start", "profile_steps",
              "branch_from", "health_metrics", "health_report", "health_kill",
              "precompile_buckets", "tail_fraction", "checkpoint_tail_every",
              "full_eval_every", "full_eval_tail_every", "quick_eval_tail_every",
              "keep_steps", "keep_every", "keep_from", "select_on", "select_ema"),
    "compute": ("libtpu_flags",),
    "data": ("workers", "worker_buffer", "validation", "validation_per_source", "heldout",
             "quick_validation_per_source"),
}  # fmt: skip
# Fields added after runs existed: at their default they are left out of the identity, so
# the runs from before them keep theirs (and resume).
IDENTITY_NEW_DEFAULTS = {
    "compute": {
        "gdn_inverse": "default",
        "attention_splash_block": 512,
        "attention_splash_compute": 0,
        "attention_fused_bwd": False,
        "gdn_heads_per_step": 0,
        "gdn_vmem_fraction": 0.0,
        "lora_impl": "unmerged",
        "gdn_norm": "heads",
        "gdn_input_mask": True,
        "flat_rows": False,
        "kernel_out_in": [],
        "head_rows": "full",
        "remat_by_bucket": {},
    },
    "lora": {
        "init": "default",
        "init_niter": 4,
        "init_batches": 8,
        "init_gamma": 16.0,
        "gate_rank": 0,
        "gate_full": False,
        "dropout": 0.0,
    },
    "optim": {"ema_decay": 0.0},
    "train": {"packing": "bucket"},
    # a section added later: left out of the identity whole while every field is default
    "loss": {"label_smoothing": 0.0, "smooth_sources": [], "kl_base_weight": 0.0,
             "kl_sources": [], "hard_targets": False, "cogcalib_weight": 0.0,
             "cogcalib_threshold": 0.0, "cogcalib_quantile": 0.0, "cogcalib_ema": 0.9,
             "teacher_mix": 0.0, "teacher_gate": 0.6},
}  # fmt: skip


def lora_init_step(cfg: RunConfig, run_dir: Path, mesh, graphdef, lora, frozen,
                   make_stream) -> tuple:  # fmt: skip
    """``lora.init`` (tjev.model.lora_init) on the placed model: computed once on the first
    microbatches of a fresh stream (the ones step 1 trains on) and saved to the run as
    A0, B0; a resumed run (or a branch, from its source) restores it instead, so residual
    kernels are bitwise those of the original start. Returns the re-placed split."""
    path = run_dir / INIT_FILE
    if not path.exists() and cfg.train.branch_from:
        source = Path(cfg.train.branch_from.rpartition("@")[0]) / INIT_FILE
        if source.exists():
            shutil.copy2(source, path)
    model = nnx.merge(graphdef, lora, frozen)
    del lora, frozen
    t0 = time.perf_counter()
    if path.exists():
        restore_init(model, path)
        event = "lora_init_restored"
    else:
        cache: list[Batch] = []

        def batches() -> list[Batch]:
            if not cache:
                cache.extend(init_microbatches(make_stream(), cfg.lora.init_batches, mesh))
            return cache

        def loss(model, batch: Batch):
            hidden = model.hidden(batch.tokens, batch.segment_ids, batch.positions)
            logits = model.label_logits(hidden, batch.slots, batch.label_ids)
            return loss_sums(logits, batch, brier_weight=cfg.train.brier_weight)[0]

        initialize(model, cfg.lora, seed=cfg.train.seed, batches=batches, loss=loss)
        save_init(path, model, cfg.lora.init)
        event = "lora_init"
    seconds = round(time.perf_counter() - t0, 1)
    print(json.dumps({"event": event, "init": cfg.lora.init, "seconds": seconds}), flush=True)
    graphdef, lora, frozen = split_model(model)
    del model
    return graphdef, place_replicated(lora, mesh), place_frozen(frozen, mesh)


def init_microbatches(stream: Iterator[Batch], n: int, mesh) -> list[Batch]:
    """The first ``n`` microbatches of a stream ([R, ...] each), placed like a step's."""
    out: list[Batch] = []
    try:
        while len(out) < n:
            step = next(stream)
            out += [jax.tree.map(lambda x, i=i: x[i], step) for i in range(len(step.tokens))]
    finally:
        getattr(stream, "close", lambda: None)()
    sharding = batch_sharding(mesh, stacked=False)
    return [jax.device_put(b, sharding) for b in out[:n]]


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
    for section, defaults in IDENTITY_NEW_DEFAULTS.items():
        for key, default in defaults.items():
            if section in raw and _same(raw[section].get(key), default):
                raw[section].pop(key)
        if section == "loss" and not raw.get(section):
            raw.pop(section, None)
    for key in ("name", "output", "log"):
        raw.pop(key)
    return {
        "config": fingerprint(raw),
        "model": model_identity,
        "data": fingerprint({p: file_hash(p) for p in cfg.data.train}),
        "template": TEMPLATE_VERSION,
    }


def _same(value: Any, default: Any) -> bool:
    return list(value) == default if isinstance(value, tuple) else value == default


def first_per_source(items: list[Item], n: int) -> list[Item]:
    seen: dict[str, int] = defaultdict(int)
    out = []
    for item in items:
        if seen[item.source] < n:
            seen[item.source] += 1
            out.append(item)
    return out


Slots = tuple[np.ndarray, np.ndarray, np.ndarray]


def calibrated_nll(validation: Slots, heldout: Slots | None = None,
                   langs: tuple[Sequence[str], Sequence[str] | None] | None = None) -> dict:  # fmt: skip
    """NLL after one temperature fitted on the even validation slots: on the odd ones and
    (if given) on the held-out slots; ``score`` is their mean. Slots: (logits, mask, target).
    ``langs``: the validation and held-out slots' languages; adds ``lang/<lang>`` (the
    calibrated NLL of that language's odd validation and held-out slots), ``lang_mean`` and
    ``lang_max`` over the languages (the global NLL is ~3/4 English)."""
    logits, mask, target = validation
    fit = np.arange(len(logits)) % 2 == 0
    temperature, *_ = fit_temperature(logits[fit], mask[fit], target[fit])
    out: dict[str, Any] = {"temperature": temperature,
                           "val_nll": nll(logits[~fit], mask[~fit], target[~fit], temperature)}  # fmt: skip
    parts = [out["val_nll"]]
    if heldout is not None:
        out["heldout_nll"] = nll(*heldout, temperature)
        parts.append(out["heldout_nll"])
    out["score"] = float(np.mean(parts))
    if langs is not None:
        scored = [(logits[~fit], mask[~fit], target[~fit], np.asarray(langs[0])[~fit])]
        if heldout is not None and langs[1] is not None:
            scored.append((*heldout, np.asarray(langs[1])))
        cat = [np.concatenate([part[i] for part in scored]) for i in range(4)]
        per = {str(lang): nll(cat[0][cat[3] == lang], cat[1][cat[3] == lang],
                              cat[2][cat[3] == lang], temperature)
               for lang in sorted(set(cat[3].tolist()))}  # fmt: skip
        out["lang"] = per
        out["lang_mean"] = float(np.mean(list(per.values())))
        out["lang_max"] = float(np.max(list(per.values())))
    return out


def selection_score(cal: dict, select_on: str) -> float:
    """The score ``train.select_on`` selects on (lower is better)."""
    return float(cal["score"] if select_on == "score" else cal[select_on])


class Cadence:
    """When a step checkpoints, runs the full eval (and selects) or the quick eval: the main
    cadences, and the tail's (the last ``tail_fraction`` of the steps, 0: the cooldown).
    Every full-eval step is a checkpoint; the last step always is both. Steps that are
    neither (time-based, signal, deadline saves) are resume-only (pruned once superseded)."""

    def __init__(self, cfg: RunConfig):
        t = cfg.train
        self.steps = t.steps
        frac = t.tail_fraction or cfg.optim.decay_fraction
        self.tail_start = t.steps - round(frac * t.steps)
        self.enabled = t.checkpoint_every > 0  # <= 0: a benchmark, no checkpoints or evals
        self.ckpt = (t.checkpoint_every, t.checkpoint_tail_every)
        # full evals follow the checkpoints unless given their own cadence
        own = bool(t.full_eval_every or t.full_eval_tail_every)
        self.full = ((t.full_eval_every or t.checkpoint_every, t.full_eval_tail_every)
                     if own else self.ckpt)  # fmt: skip
        self.quick = (t.quick_eval_every, t.quick_eval_tail_every)
        self.keep_steps = frozenset(t.keep_steps)
        self.keep_every, self.keep_from = t.keep_every, t.keep_from

    def _every(self, step: int, cadence: tuple[int, int]) -> bool:
        main, tail = cadence
        n = tail if tail and step > self.tail_start else main
        return n > 0 and step % n == 0

    def full_eval(self, step: int) -> bool:
        return self.enabled and (step == self.steps or self._every(step, self.full))

    def checkpoint(self, step: int) -> bool:
        return self.enabled and (
            step == self.steps or self._every(step, self.ckpt) or self.full_eval(step)
            or self.kept(step)
        )  # fmt: skip

    def kept(self, step: int) -> bool:
        """A step never pruned: train.keep_steps, and every keep_every-th from keep_from."""
        return step in self.keep_steps or (
            self.keep_every > 0 and step >= self.keep_from and step % self.keep_every == 0
        )

    def quick_eval(self, step: int) -> bool:
        return self._every(step, self.quick) and not self.full_eval(step)

    def resume_only(self, step: int) -> bool:
        return not self.checkpoint(step)


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


HEALTH_CODES = {"ok": 0, "warn": 1, "kill": 2, "killed": 3}


def hyperparameters(cfg: RunConfig) -> dict:
    """Flat W&B config keys for the sweep views (parallel coordinates, grouping): what a
    campaign varies, under one prefix (``hp.lr``, ``hp.arm``, …)."""
    arm = next((t.split(":", 1)[1] for t in cfg.log.wandb_tags if t.startswith("arm:")), "")
    return {
        "size": cfg.model.path.rstrip("/").rsplit("Qwen3.5-", 1)[-1],
        "arm": arm,
        "optimizer": cfg.optim.name,
        "lr": cfg.optim.lr,
        "rank": cfg.lora.rank,
        "init": getattr(cfg.lora, "init", "default"),
        "dtype": cfg.model.dtype,
        "gdn_precision": cfg.compute.gdn_precision,
        "clip": cfg.optim.clip_mode,
        "b2": cfg.optim.b2,
        "tokens_per_step": cfg.train.tokens_per_step,
        "steps": cfg.train.steps,
        "seed": cfg.train.seed,
        # what else defines a run: the data mix, the GDN kernel, merged LoRA, the gates
        "mix": _mix_name(cfg.data.train),
        "gdn_impl": cfg.compute.gdn_impl,
        "gdn_chunk": cfg.compute.gdn_chunk,
        "lora_impl": cfg.compute.lora_impl,
        "gates": (
            "full"
            if cfg.lora.gate_full
            else "frozen"
            if not set(GATES) & set(cfg.lora.targets)
            else f"r{cfg.lora.gate_rank}"
            if cfg.lora.gate_rank
            else "lora"
        ),
        "branch_from": cfg.train.branch_from or "",
        # the regularisers (tjev.train.objective, optim, lora)
        "weight_decay": cfg.optim.weight_decay,
        "final_lr_fraction": cfg.optim.final_lr_fraction,
        "decay_fraction": cfg.optim.decay_fraction,
        "ema_decay": cfg.optim.ema_decay,
        "lora_dropout": cfg.lora.dropout,
        "label_smoothing": cfg.loss.label_smoothing,
        "kl_base_weight": cfg.loss.kl_base_weight,
        "hard_targets": cfg.loss.hard_targets,
        "cogcalib_weight": cfg.loss.cogcalib_weight,
        "cogcalib_threshold": cfg.loss.cogcalib_threshold,
        "cogcalib_quantile": cfg.loss.cogcalib_quantile,
        "teacher_mix": cfg.loss.teacher_mix,
        "teacher_gate": cfg.loss.teacher_gate,
        "select_on": cfg.train.select_on,
    }


def _mix_name(train: Sequence[str] | str) -> str:
    """The mix directory of the training files (``…/<mix>/train/<source>.jsonl``)."""
    first = train[0] if isinstance(train, (list, tuple)) and train else str(train)
    parts = Path(str(first)).parts
    return parts[parts.index("train") - 1] if "train" in parts[1:] else ""


def eval_scalars(report: dict) -> dict:
    return {"all": report["all"], **{g: report.get(g, {}) for g in ("type", "lang")}}


def train(cfg: RunConfig) -> dict:
    # the libtpu flags the run trains with (the session's probed flags, the preset's, or
    # none) and the scoped VMEM limit they set: in stdout, config.json and the W&B config
    libtpu = libtpu_settings(configure_runtime(cfg.compute))
    print(json.dumps({"event": "libtpu", **libtpu}), flush=True)
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
    write_json(config_path, {"config": cfg.to_dict(), "identity": identity, "libtpu": libtpu})
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
    # A bucket longer than microbatch_tokens takes one row per device (rows_for): its steps
    # hold more tokens, at the same number of microbatches (gradients are per answer slot).
    accumulation = max(1, cfg.train.tokens_per_step // per_step)
    tx = make_optimizer(cfg.optim, cfg.train.steps, cfg.loss)
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
            devices=n_dev,
            teacher=bool(cfg.loss.teacher_mix),
            packing=cfg.train.packing,
        )

    slots_per_step = expected_slots_per_step(make_stream(0))
    print(json.dumps({"event": "expected_slots_per_step", "value": slots_per_step}), flush=True)
    if cfg.lora.init != "default" and cfg.lora.rank:
        graphdef, lora, frozen = lora_init_step(cfg, run_dir, mesh, graphdef, lora, frozen,
                                                lambda: make_stream(0))  # fmt: skip
        opt_state = place_replicated(tx.init(lora), mesh)
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

    if cfg.loss.kl_base_weight and cfg.lora.gate_full:
        raise ValueError("loss.kl_base_weight: the base of a lora.gate_full model is not kept")
    if cfg.lora.dropout and cfg.compute.lora_impl == "merged":
        print(json.dumps({"event": "lora_dropout_unmerged",
                          "note": "merged LoRA cannot drop the adapter input"}), flush=True)  # fmt: skip
    sources = vocab.names["source"]
    step_fn = jit_train_step(
        make_train_step(
            graphdef,
            tx,
            brier_weight=cfg.train.brier_weight,
            slots_per_step=slots_per_step,
            optim=cfg.optim,
            lr_schedule=unit_schedule(cfg.optim, cfg.train.steps),  # × the state's peak
            health=cfg.train.health_metrics,
            loss=cfg.loss,
            smooth_table=source_table(cfg.loss.smooth_sources, sources)
            if cfg.loss.label_smoothing
            else None,
            kl_table=source_table(cfg.loss.kl_sources, sources)
            if cfg.loss.kl_base_weight
            else None,
            dropout=cfg.lora.dropout,
        ),
        donate_argnums=(0, 1),
    )
    # lora.dropout: the step's key, a function of (seed, step) (resume-exact)
    dropout_root = jax.random.key(cfg.train.seed) if cfg.lora.dropout else None

    def step_key(step: int) -> tuple:
        return () if dropout_root is None else (jax.random.fold_in(dropout_root, step),)

    eval_fn = jit_eval_step(make_eval_step(graphdef))
    # checkpoint_every <= 0: no checkpoints and no full evals (a benchmark run; SIGTERM
    # still saves), so the full eval sets are never built
    cadence = Cadence(cfg)
    checkpoints = cadence.enabled
    evalset = EvalSet(validation, tok, vocab, cfg, mesh) if validation and checkpoints else None
    heldset = EvalSet(heldout, tok, vocab, cfg, mesh) if heldout and checkpoints else None
    quickset = None
    if validation and (cfg.train.quick_eval_every or cfg.train.quick_eval_tail_every):
        quick = first_per_source(validation, cfg.data.quick_validation_per_source)
        quickset = EvalSet(quick, tok, vocab, cfg, mesh)
    device_kind = jax.devices()[0].device_kind
    system = {
        "device_kind": device_kind,
        "devices": n_dev,
        "jax": jax.__version__,
        "code": code_version(),
        "tpu_visible_chips": os.environ.get("TPU_VISIBLE_CHIPS", ""),
        **libtpu,
    }
    log = cfg.log
    logger = MetricLogger(
        run_dir,
        tensorboard=log.tensorboard,
        wandb=log.wandb,
        project=log.wandb_project,
        config=cfg.to_dict() | {"identity": identity, "system": system, "hp": hyperparameters(cfg)},
        name=cfg.name,
        resume_step=start,
        entity=log.wandb_entity,
        group=log.wandb_group,
        tags=log.wandb_tags,
        mode=log.wandb_mode,
    )
    # numbers only in the summary (0: libtpu's / Mosaic's default); the flags in config.json
    logger.summary({"libtpu": {k: libtpu[k] or 0 for k in ("scoped_vmem_limit_kib",
                                                            "kernel_vmem_limit_bytes")}})  # fmt: skip
    monitor = HealthMonitor(warmup=cfg.optim.warmup_steps, clip_mode=cfg.optim.clip_mode,
                            optimizer=cfg.optim.name)  # fmt: skip
    window = MetricWindow(logger, cfg, n_dev, flops_mod.peak_flops(device_kind), lr_at, monitor)
    killed = False
    in_sharding = batch_sharding(mesh, stacked=True)
    if cfg.train.precompile_buckets:
        precompile(step_fn, lora, opt_state, frozen, in_sharding, window, cfg, accumulation,
                   per_step, n_dev, step_key(0))  # fmt: skip
    profiling = False
    stream_state = stream.get_state()
    # SIGTERM (preemption, the queue's shutdown and session deadline), SIGINT (^C), or
    # TJEV_DEADLINE (unix seconds) passed: save the current step, then exit 143 (130: SIGINT)
    stop = threading.Event()
    stop_code = [143]

    def on_signal(signum, _frame):
        stop_code[0] = 130 if signum == signal.SIGINT else 143
        stop.set()

    previous_handlers: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, on_signal)
    deadline = float(os.environ.get("TJEV_DEADLINE") or 0)
    t_saved = time.monotonic()
    try:
        for step in range(start + 1, cfg.train.steps + 1):
            if cfg.train.profile_start and step == cfg.train.profile_start:
                from tjev.campaign.xprof_summary import profile_options

                # the previous step done first: the window then holds whole executions only
                # (its tail would otherwise land, truncated, at the start of the trace)
                jax.block_until_ready((lora, opt_state))
                jax.profiler.start_trace(
                    str(run_dir / "profile"), profiler_options=profile_options()
                )
                profiling = True
            t_wait = time.perf_counter()
            batch = next(stream)
            wait = time.perf_counter() - t_wait
            stream_state = stream.get_state()  # the state that goes with this step
            lengths = flops_mod.segment_lengths(batch.segment_ids)
            step_flops = flops_mod.train_flops(
                config, lengths, cfg.compute.gdn_chunk, int(batch.label_mask.sum())
            )
            new_shape = batch.shape_key not in window.shapes
            t_call = time.perf_counter()
            # a step marker: xprof's per-step tools (overview, step time) need it; ~free
            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                lora, opt_state, metrics = step_fn(
                    lora, opt_state, frozen, jax.device_put(batch, in_sharding), *step_key(step)
                )
            if new_shape:  # the call traced and compiled this shape (dispatch is async)
                print(
                    json.dumps(
                        {
                            "event": "compiled",
                            "step": step,
                            "shape": list(batch.shape_key),
                            "program": train_step_name(batch),  # its name in xprof
                            "seconds": round(time.perf_counter() - t_call, 2),
                        }
                    ),
                    flush=True,
                )
            window.add(step, metrics, batch, step_flops, wait, stream_state["next_index"])
            if profiling and step >= cfg.train.profile_start + cfg.train.profile_steps:
                jax.block_until_ready(metrics)
                jax.profiler.stop_trace()
                profiling = False
                jax.profiler.save_device_memory_profile(str(run_dir / "profile" / "memory.prof"))

            if deadline and time.time() > deadline:
                stop.set()
            # <run>/STOP, checked at every step: this step gets the full eval and a
            # checkpoint, then the run finishes as completed (its best checkpoint selected;
            # branches already queued from its earlier steps proceed). Removing STOP and
            # running again resumes it exactly from here.
            stopping = step < cfg.train.steps and (run_dir / STOP_FILE).exists()
            full_eval = bool(evalset) and (cadence.full_eval(step) or stopping)
            quick_eval = bool(quickset) and not full_eval and cadence.quick_eval(step)
            save = cadence.checkpoint(step) or stopping
            synced = save or full_eval or quick_eval or stop.is_set()
            window.flush(everything=synced)
            if cfg.train.health_kill and monitor.status == "kill":
                killed = True  # the verdict lags the device by one step (metrics fetched late)
                print(json.dumps({"event": "health_kill", "step": step,
                                  "rules": sorted(monitor.fired)}), flush=True)  # fmt: skip
                break
            if cfg.train.health_report and synced:
                write_json(run_dir / "health.json", monitor.verdict())

            if quickset and quick_eval:
                with jax.profiler.TraceAnnotation("eval"):  # a profile window shows it as such
                    report = quickset.run(eval_fn, lora, frozen, mesh, vocab)
                logger.log(step, {"eval_quick": eval_scalars(report)})
            new_best = False
            if evalset and full_eval:
                with jax.profiler.TraceAnnotation("eval"):
                    scores = full_evals(step, eval_fn, lora, opt_state, frozen, mesh, vocab,
                                        evalset, heldset, run_dir, logger, cfg)  # fmt: skip
                pick = scores["ema" if cfg.train.select_ema and "ema" in scores else "raw"]
                if pick["score"] < best["score"]:
                    new_best = True
                    best = {"step": step, **pick}

            # Full evals happen at checkpoint steps, so the best step is always checkpointed; it
            # is never pruned, and selection.json names it only once its checkpoint is written.
            # Time-based, signal and deadline saves are for resuming only (no eval; pruned
            # once a newer save exists), as are planned steps beyond keep_checkpoints, except
            # the kept steps (train.keep_steps, the planned branch points, and keep_every).
            timed = cfg.train.checkpoint_secs and (
                time.monotonic() - t_saved > cfg.train.checkpoint_secs
            )
            if save or timed or stop.is_set():
                meta = {"stream": stream_state, "best": best}
                ckpt.save(step, lora, opt_state, meta, keep=best["step"],
                          protect=filter(cadence.kept, ckpt.all_steps()), resume_only=cadence.resume_only)  # fmt: skip
                t_saved = time.monotonic()
            if new_best:
                ckpt.wait()
                write_json(run_dir / "selection.json", best)
                logger.summary({"best": best})
            if synced:  # evals and checkpoint writes are not step time
                window.restart_clock()
            if stop.is_set() and step < cfg.train.steps:
                ckpt.wait()  # the async save is on disk before the process exits
                print(json.dumps({"event": "interrupted", "step": step}), flush=True)
                raise SystemExit(stop_code[0])  # resumes exactly from this checkpoint
            if stopping:
                ckpt.wait()
                print(json.dumps({"event": "stopped_early", "step": step}), flush=True)
                logger.summary({"stopped_early": step})
                break
        window.flush(everything=True)
        verdict = monitor.verdict(killed=killed)
        logger.summary({"best": best, "health": {
            "status": HEALTH_CODES[verdict["status"]], "spikes": verdict["stats"]["spikes"],
            "rejected": verdict["stats"]["rejected"]}})  # fmt: skip
    finally:
        if cfg.train.health_report or killed:
            write_json(run_dir / "health.json", monitor.verdict(killed=killed))
        if profiling:  # a run that stops inside the profile window still writes it
            jax.profiler.stop_trace()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        stream.close()
        ckpt.wait()
        ckpt.close()
        logger.close()
    return {"run": str(run_dir), "best": best, "steps": cfg.train.steps,
            "health": monitor.status if not killed else "killed"}  # fmt: skip


def full_evals(step: int, eval_fn, lora, opt_state, frozen, mesh, vocab: Vocab,
               evalset: EvalSet, heldset: EvalSet | None, run_dir: Path, logger: MetricLogger,
               cfg: RunConfig) -> dict[str, dict]:  # fmt: skip
    """The full eval of ``step`` (validation, held-out, the calibrated selection scores,
    per language too), of the adapters and, with ``optim.ema_decay``, of their EMA
    (``eval_ema/*``, ``eval_heldout_ema/*``, ``eval_calibrated_ema/*``, ``step-N-ema.json``).
    Returns {"raw": …, "ema": …}: each the selection entry (``score``: ``train.select_on``'s)."""
    weights = {"raw": lora}
    if (ema := ema_params(opt_state)) is not None:
        weights["ema"] = jax.tree.map(lambda e, p: e.astype(p.dtype), ema, lora)
    out = {}
    for kind, adapters in weights.items():
        tag = "" if kind == "raw" else "-ema"
        suffix = "" if kind == "raw" else "_ema"
        report = evalset.run(eval_fn, adapters, frozen, mesh, vocab)
        write_json(run_dir / "eval" / f"step-{step}{tag}.json", report)
        logger.log(step, {f"eval{suffix}": eval_scalars(report)})
        held = None
        if heldset:
            held = heldset.run(eval_fn, adapters, frozen, mesh, vocab)
            write_json(run_dir / "eval" / f"step-{step}-heldout{tag}.json", held)
            logger.log(step, {f"eval_heldout{suffix}": eval_scalars(held)})
        cal = calibrated_nll(evalset.last, heldset.last if heldset else None,
                             (evalset.last_lang, heldset.last_lang if heldset else None))  # fmt: skip
        logger.log(step, {f"eval_calibrated{suffix}": cal})
        entry = {"score": selection_score(cal, cfg.train.select_on), "all": report["all"],
                 "calibrated": cal}  # fmt: skip
        if cfg.train.select_on != "score":
            entry["select_on"] = cfg.train.select_on
        if kind == "ema":
            entry["weights"] = "ema"
        if held is not None:
            entry["heldout"] = held["all"]
        out[kind] = entry
    return out


def precompile(step_fn, lora, opt_state, frozen, in_sharding, window: MetricWindow,
               cfg: RunConfig, accumulation: int, per_step: int,
               devices: int = 1, extra: tuple = ()) -> None:  # fmt: skip
    """``train.precompile_buckets``: compile every bucket's train step before step 1, so all
    the timed steps of a short benchmark are steady-state.

    Each shape's step runs once on an all-padding batch (segment ids 0, no label, weight 0:
    the masked loss is 0 and its outputs are never read) built and placed like a real batch (the
    pack's dtypes, ``jax.device_put`` to ``in_sharding``), on copies of the adapters and the
    optimizer state with their own shardings: the jit cache key of the real steps, which
    then never compile. The copies are donated and the outputs dropped, so training is
    untouched. The "compiled" event (step 0, ``precompiled``) has the dispatch time (trace +
    compile, as for a step that compiles) and ``run_seconds`` (+ the dummy step's run)."""
    from tjev.train.compile_check import abstract_batch

    def copy(tree):
        return jax.tree.map(
            lambda x: jax.device_put(x, x.sharding, may_alias=False)
            if isinstance(x, jax.Array) else x, tree)  # fmt: skip

    for length in cfg.train.seq_buckets:
        spec = abstract_batch(accumulation, rows_for(length, per_step, devices), length,
                              cfg.train.max_segments, cfg.train.max_labels)  # fmt: skip
        dummy = jax.tree.map(lambda x: np.zeros(x.shape, x.dtype), spec)
        if dummy.shape_key in window.shapes:
            continue
        t0 = time.perf_counter()
        out = step_fn(copy(lora), copy(opt_state), frozen, jax.device_put(dummy, in_sharding),
                      *extra)  # fmt: skip
        dispatch = time.perf_counter() - t0
        jax.block_until_ready(out)
        del out
        window.shapes.add(dummy.shape_key)
        print(json.dumps({"event": "compiled", "step": 0, "shape": list(dummy.shape_key),
                          "program": train_step_name(dummy),
                          "seconds": round(dispatch, 2), "precompiled": True,
                          "run_seconds": round(time.perf_counter() - t0, 2)}),
              flush=True)  # fmt: skip
    window.restart_clock()  # the first timed step's clock starts after the compiles


class MetricWindow:
    """Device metrics fetched one step late (while the next step runs), so logging every
    step does not stall the device on the host; evals, checkpoints and the last step flush
    everything."""

    def __init__(self, logger: MetricLogger, cfg: RunConfig, devices: int, peak, lr_at,
                 monitor: HealthMonitor | None = None):  # fmt: skip
        self.logger, self.cfg, self.devices, self.peak, self.lr_at = (
            logger,
            cfg,
            devices,
            peak,
            lr_at,
        )
        self.monitor = monitor
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
                "bucket": batch.shape_key,
                "padded_tokens": int(np.prod(batch.shape_key)),
            }
        )

    def restart_clock(self) -> None:
        """After a full flush and the evals / checkpoint that follow it: the next window's
        time starts here, so eval and checkpoint time never count as step time."""
        if not self.pending:
            self.t_last = time.perf_counter()

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
        tokens = sum(float(m["tokens"]) for m in values)  # non-padding tokens
        padded = sum(p["padded_tokens"] for p in group)
        optional = ("kl_base", "cogcalib_frac", "cogcalib_term", "cogcalib_threshold",
                    "teacher_frac")  # fmt: skip
        keys = ("loss", "nll", "brier", "accuracy", "grad_norm",
                *(k for k in optional if k in values[0]))  # fmt: skip
        learning = {k: float(np.mean([m[k] for m in values])) for k in keys}
        health = [m.get("health", {}) for m in values]
        # the rate the optimizer used (its own step count), else the schedule at this step
        lr = [h["lr"] for h in health if "lr" in h]
        learning["lr"] = float(np.mean(lr)) if lr else float(self.lr_at(step - 1))
        learning["update_rejected"] = float(sum(m["update_rejected"] for m in values))
        record_health: dict[str, float] = {}
        if health and health[0]:
            keys = [k for k in health[0] if k != "lr"]
            record_health = {k: float(np.mean([h[k] for h in health])) for k in keys}
            if "clip_active" in record_health:  # the fraction of the row's steps
                record_health["clip_fraction"] = record_health.pop("clip_active")
            for k in ("grad_norm_clipped", "update_ratio_a", "update_ratio_b"):
                if k in record_health:
                    record_health[f"{k}_max"] = float(np.max([h[k] for h in health]))
        if self.monitor is not None:
            for p, m in zip(group, values, strict=True):
                flat = {k: float(m[k]) for k in ("loss", "grad_norm", "update_rejected")}
                self.monitor.add(
                    p["step"],
                    flat | {k: float(v) for k, v in m.get("health", {}).items()},
                    bucket=p.get("bucket"),
                )
            host = self.monitor.row()
            learning["spikes"] = host.pop("spikes")
            record_health |= host
        perf = {
            "step_seconds": per_step,
            "tokens_per_second_per_device": tokens / elapsed / self.devices,
            "padded_tokens_per_second_per_device": padded / elapsed / self.devices,
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
            **({"health": record_health} if record_health else {}),
            "perf": perf,
            "data": {
                "pad_fraction": float(np.mean([p["pad"] for p in group])),
                # the sequence bucket of the row's last step (log_every=1: its own)
                "seq_len": int(group[-1]["bucket"][-1]),
                "slots_per_step": float(values[-1]["slots"]),
                "segments_consumed": group[-1]["segments"],
                # host time blocked on the stream (> ~5% of step_seconds: add data.workers)
                "wait_seconds": float(np.mean([p["wait"] for p in group])),
            },
        }
        self.logger.log(step, record)
        print(json.dumps({"step": step, **learning, **perf}), flush=True)
