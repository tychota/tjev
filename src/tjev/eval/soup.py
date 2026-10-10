"""ΔW soups: the weight-space average of LoRA runs, written as a run directory.

Each member is a run (a step, its raw or EMA adapters: ``RUN[@STEP][:ema]``) whose every
adapted projection adds ΔW_i = s_i·A_i B_i to the shared frozen kernel W (an offset init's
s_i·(A_i B_i − A0_i B0_i), as the rank-2r product of :meth:`Linear.factors`). The soup adds
the weighted mean Σ_i w_i·ΔW_i (w_i = 1/n by default). A and B factors are never averaged
across runs: two runs' factors live in unrelated bases, only their products are comparable.

The mean is exact at rank Σ r_i, by concatenating the scaled factors::

    Σ_i w_i·s_i·A_i B_i = s·[A_1 | … | A_n]·[c_1·B_1 ; … ; c_n·B_n],   c_i = w_i·s_i / s

with s the soup's own LoRA scale (rsLoRA, α = √R at rank R: s = 1 on the main projections).
A projection whose kernel trains directly (``lora.gate_full``) gets the mean kernel
Σ_i w_i·W_i = W + Σ_i w_i·ΔW_i.

The output is an ordinary run directory (``config.json`` with the soup's LoRA spec,
``vocab.json``, ``checkpoints/0/lora``, ``soup.json`` naming the members and weights):
:class:`~tjev.eval.runs.LoadedRun` loads it, so ``tjev post``, ``tjev eval`` and
``tjev export`` (PEFT adapters at rank R, or the merged snapshot) take it as they take a
trained run. Its checkpoint identity is a fingerprint of the members' identities and
weights: a calibration fitted on a member never validates on the soup.

The greedy soup (Wortsman et al., arXiv 2203.05482): members ranked by their own held-out
metric, each added (uniform weights over the kept members) only if the soup's metric on
the held-out data improves. The held-out metric is the per-type temperature-scaled NLL
(``nll-cal``: temperatures fitted on the held-out data itself, so a member is not judged
on a miscalibration one scalar per type removes), the raw NLL or the accuracy. All the
candidate soups are evaluated in one model at rank Σ r_i over every member (a dropped
member's B block is zero), one compiled eval step; the written soup has only the kept
members.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tjev.config import LoRASpec, RunConfig
from tjev.config.schema import GATES
from tjev.data.pack import Vocab
from tjev.data.render import NoulOrder
from tjev.model.lora import Linear
from tjev.model.lora_init import RESIDUAL
from tjev.utils import fingerprint, write_json

METRICS = ("nll-cal", "nll", "accuracy")
SOUP_FILE = "soup.json"
SOUP_STEP = 0


@dataclass(frozen=True)
class MemberSpec:
    """``RUN[@STEP][:ema]``: a run directory, a step (default: the selected one) and the
    adapters (raw; ``:ema`` the EMA adapters, ``checkpoints/<step>/ema``)."""

    run: Path
    step: int | None = None
    weights: str | None = None  # None: what selection.json names

    @classmethod
    def parse(cls, text: str) -> MemberSpec:
        weights = None
        if text.endswith(":ema"):
            text, weights = text[: -len(":ema")], "ema"
        elif text.endswith(":raw"):
            text, weights = text[: -len(":raw")], "raw"
        step = None
        if m := re.fullmatch(r"(.+)@(\d+)", text):
            text, step = m.group(1), int(m.group(2))
        return cls(Path(text), step, weights)

    def __str__(self) -> str:
        at = f"@{self.step}" if self.step is not None else ""
        return f"{self.run}{at}" + (f":{self.weights}" if self.weights else "")


@dataclass
class Member:
    """One member's ΔW on the host: per adapted projection (graph path) the factors (A, B)
    with ΔW = A·B (the LoRA scale folded into B), per directly trained kernel the kernel."""

    spec: MemberSpec
    cfg: RunConfig
    identity: dict
    checkpoint: str  # the member's checkpoint identity (LoadedRun.checkpoint_identity)
    step: int
    weights: str
    vocab: dict[str, list[str]]
    labels: dict[str, str] = field(default_factory=dict)  # path → projection name
    factors: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    full: dict[str, np.ndarray] = field(default_factory=dict)

    def rank(self, path: str) -> int:
        return self.factors[path][0].shape[-1]


def linears(model: nnx.Module) -> dict[str, Linear]:
    """Graph path ("blocks/mlps/0/gate_proj") → every (stacked) projection, graph order."""
    return {"/".join(map(str, p)): n for p, n in nnx.iter_graph(model) if isinstance(n, Linear)}


def load_member(spec: MemberSpec) -> Member:
    """Restore ``spec`` (:class:`~tjev.eval.runs.LoadedRun`) and keep its ΔW factors."""
    from tjev.eval.runs import LoadedRun

    run = LoadedRun(spec.run, spec.step, weights=spec.weights)
    member = Member(spec, run.cfg, run.identity, run.checkpoint_identity, run.step,
                    run.weights, run.vocab.names)  # fmt: skip
    for path, layer in linears(run.model()).items():
        if layer.full:
            member.full[path] = np.asarray(jax.device_get(layer.kernel[...]), np.float32)
            member.labels[path] = layer.label
        elif layer.rank:
            if run.init in RESIDUAL and not layer.offset:
                raise ValueError(f"{spec}: a legacy residual-init run (kernel W − s·A0B0) "
                                 "does not share the base kernel; it cannot join a soup")  # fmt: skip
            a, b = (np.asarray(jax.device_get(t), np.float32) for t in layer.factors(jnp.float32))
            member.factors[path] = (a, b * np.float32(layer.scale))
            member.labels[path] = layer.label
    del run
    return member


def check_compatible(members: list[Member]) -> None:
    """Same base model (snapshot identity), prompt template, LoRA targets and adapted
    projections (shapes up to the rank); else ValueError."""
    if not members:
        raise ValueError("a soup needs at least one member")
    first = members[0]
    for m in members[1:]:
        problems = []
        if m.identity.get("model") != first.identity.get("model"):
            problems.append("base model")
        if m.identity.get("template") != first.identity.get("template"):
            problems.append("prompt template")
        if tuple(m.cfg.lora.targets) != tuple(first.cfg.lora.targets):
            problems.append(f"LoRA targets {m.cfg.lora.targets} vs {first.cfg.lora.targets}")
        if m.cfg.lora.gate_full != first.cfg.lora.gate_full:
            problems.append("lora.gate_full")
        if set(m.factors) != set(first.factors) or set(m.full) != set(first.full):
            problems.append("adapted projections")
        else:
            for path, (a, b) in m.factors.items():
                a0, b0 = first.factors[path]
                if a.shape[:-1] != a0.shape[:-1] or b.shape[:-2] + b.shape[-1:] != \
                        b0.shape[:-2] + b0.shape[-1:]:  # fmt: skip
                    problems.append(f"{path} shape")
                    break
        if problems:
            raise ValueError(f"{m.spec} cannot join a soup with {first.spec}: "
                             f"different {', '.join(problems)}")  # fmt: skip


def soup_lora(members: list[Member], weights: list[float]) -> LoRASpec:
    """The soup's LoRA spec: rank R = Σ r_i over the members with a weight (the gate
    projections their own Σ as ``gate_rank`` when it differs), α = √R (rsLoRA, s = 1)."""
    kept = [m for m, w in zip(members, weights, strict=True) if w]
    first = kept[0]
    sums = {p: sum(m.rank(p) for m in kept) for p in first.factors}
    main = {r for p, r in sums.items() if first.labels[p] not in GATES}
    gate = {r for p, r in sums.items() if first.labels[p] in GATES}
    if len(main) > 1 or len(gate) > 1:
        raise ValueError(f"members have per-projection ranks a LoRA spec cannot express: {sums}")
    rank = main.pop() if main else gate.copy().pop()
    gate_rank = gate.pop() if gate else 0
    return replace(first.cfg.lora, rank=rank, alpha=float(rank) ** 0.5, rslora=True,
                   init="default", gate_rank=0 if gate_rank == rank else gate_rank,
                   dropout=0.0)  # fmt: skip


def fill(model: nnx.Module, members: list[Member], weights: list[float],
         keep_zero: bool = False) -> None:  # fmt: skip
    """Set ``model``'s adapters to Σ w_i·ΔW_i (built at :func:`soup_lora`'s spec of the same
    members, or of every member with ``keep_zero``: a zero weight is a zero B block)."""
    used = [(m, w) for m, w in zip(members, weights, strict=True) if w or keep_zero]
    for path, layer in linears(model).items():
        if layer.full:
            layer.kernel[...] = jnp.asarray(sum(w * m.full[path] for m, w in used), jnp.float32)
        elif layer.rank:
            a = np.concatenate([m.factors[path][0] for m, _ in used], axis=-1)
            b = np.concatenate([np.float32(w / layer.scale) * m.factors[path][1]
                                for m, w in used], axis=-2)  # fmt: skip
            if a.shape != layer.lora_a.shape or b.shape != layer.lora_b.shape:
                raise ValueError(f"{path}: soup factors {a.shape}, {b.shape} do not fit the "
                                 f"model's {layer.lora_a.shape}, {layer.lora_b.shape}")  # fmt: skip
            layer.lora_a[...] = jnp.asarray(a, jnp.float32)
            layer.lora_b[...] = jnp.asarray(b, jnp.float32)


def build(members: list[Member], weights: list[float], keep_zero: bool = False):
    """(cfg, model): the base model with the soup's adapters."""
    from tjev.model import build_model

    pick = [1.0] * len(members) if keep_zero else weights
    cfg = replace(members[0].cfg, lora=soup_lora(members, pick))
    _, model = build_model(cfg.model.path, cfg.compute, cfg.lora, dtype=cfg.model.dtype)
    fill(model, members, weights, keep_zero)
    return cfg, model


def merged_vocab(members: list[Member]) -> dict[str, list[str]]:
    vocab = Vocab(members[0].vocab)
    for m in members[1:]:
        for kind, names in m.vocab.items():
            vocab.names.setdefault(kind, [])
            for name in names:
                vocab.id(kind, name)
    return vocab.names


def write(out: str | Path, members: list[Member], weights: list[float],
          extra: dict | None = None) -> Path:  # fmt: skip
    """Write the soup of ``members`` (``weights``; zero-weight members left out) as a run
    directory ``out`` (refuses a non-empty one)."""
    from tjev.train.checkpoint import Checkpoints
    from tjev.train.step import split_model

    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    kept = [(m, w) for m, w in zip(members, weights, strict=True) if w]
    cfg, model = build([m for m, _ in kept], [w for _, w in kept])
    cfg = replace(cfg, name=out.name, output=str(out.parent))
    _, lora, _ = split_model(model)
    rows = [{"member": str(m.spec), "run": str(m.spec.run.resolve()), "step": m.step,
             "weights": m.weights, "checkpoint": m.checkpoint, "weight": w}
            for m, w in zip(members, weights, strict=True)]  # fmt: skip
    first = kept[0][0]
    identity = {
        "soup": fingerprint([(r["checkpoint"], r["weight"]) for r in rows if r["weight"]]),
        "model": first.identity.get("model"),
        "template": first.identity.get("template"),
    }
    write_json(out / "config.json", {"config": cfg.to_dict(), "identity": identity})
    write_json(out / "vocab.json", merged_vocab([m for m, _ in kept]))
    write_json(out / SOUP_FILE, {"members": rows, "rank": cfg.lora.rank,
                                 "gate_rank": cfg.lora.gate_rank, **(extra or {})})  # fmt: skip
    ckpt = Checkpoints(out / "checkpoints")
    ckpt.save_adapters(SOUP_STEP, lora, {"step": SOUP_STEP, "soup": True})
    ckpt.close()
    return out


class Scorer:
    """The held-out metric of soups of ``members`` on ``data`` (lower is better), every
    candidate in one model at rank Σ r_i (one compiled eval step)."""

    def __init__(self, members: list[Member], data, metric: str = "nll-cal", *,
                 orders: int = 1, noul_order: NoulOrder = "shuffled"):  # fmt: skip
        from tjev.data.tokenize import PromptTokenizer
        from tjev.eval.evalset import EvalSet
        from tjev.eval.runs import read_pool
        from tjev.sharding import (
            configure_runtime,
            install_mesh,
            make_mesh,
            place_frozen,
        )
        from tjev.train.step import jit_eval_step, make_eval_step, split_model

        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        self.members, self.metric = members, metric
        n = len(members)
        cfg, model = build(members, [1.0 / n] * n, keep_zero=True)
        configure_runtime(cfg.compute)
        self.mesh = make_mesh(cfg.mesh)
        install_mesh(cfg.compute, self.mesh)
        self.graphdef, lora, frozen = split_model(model)
        del model
        self.frozen = place_frozen(frozen, self.mesh)
        self.template = lora
        self.eval_fn = jit_eval_step(make_eval_step(self.graphdef))
        self.vocab = Vocab(merged_vocab(members))
        items = read_pool(data)
        for item in items:
            for kind in ("source", "lang", "family"):
                self.vocab.id(kind, getattr(item, kind))
        self.evalset = EvalSet(items, PromptTokenizer(cfg.model.path), self.vocab, cfg,
                               self.mesh, orders=orders, noul_order=noul_order)  # fmt: skip

    def __call__(self, weights: list[float]) -> float:
        from tjev.eval import calibrate as cal
        from tjev.eval.metrics import label_mask, slot_temperatures, summarize, view_probs
        from tjev.sharding import place_replicated

        model = nnx.merge(self.graphdef, self.template, self.frozen)
        fill(model, self.members, weights, keep_zero=True)
        lora = place_replicated(nnx.state(model, nnx.LoRAParam), self.mesh)
        a = self.evalset.arrays(self.eval_fn, lora, self.frozen, self.mesh, self.vocab)
        temps = 1.0
        if self.metric == "nll-cal":
            fit = cal.fit_grouped(a["logits"], a["mask"], a["target"], list(a["type"]),
                                  list(a["lang"]))  # fmt: skip
            temps = slot_temperatures(fit["temperatures"], list(a["type"]))
        probs = view_probs(a["logits"], a["mask"], temps)
        s = summarize(probs, a["target"], label_mask(a["mask"]))
        return -s["accuracy"] if self.metric == "accuracy" else s["nll"]


def greedy(scorer: Scorer) -> tuple[list[float], list[dict]]:
    """Greedy soup: the members ranked by their own score, each kept only if the uniform
    soup with it scores strictly better. Returns (weights, log)."""
    n = len(scorer.members)
    alone = [scorer([float(i == j) for j in range(n)]) for i in range(n)]
    order = sorted(range(n), key=lambda i: alone[i])
    kept = [order[0]]
    best = alone[order[0]]
    log = [{"member": str(scorer.members[i].spec), "alone": alone[i]} for i in range(n)]
    log[order[0]] |= {"soup": best, "kept": True}

    def uniform(ids: list[int]) -> list[float]:
        return [1.0 / len(ids) if j in ids else 0.0 for j in range(n)]

    for i in order[1:]:
        score = scorer(uniform([*kept, i]))
        log[i] |= {"soup": score, "kept": score < best}
        if score < best:
            kept.append(i)
            best = score
    return uniform(kept), log


def soup(members: list[str | MemberSpec], out: str | Path, *, greedy_on=None,
         metric: str = "nll-cal", orders: int = 1,
         noul_order: NoulOrder = "shuffled") -> dict:  # fmt: skip
    """Load the members, check them, and write the uniform (or, with ``greedy_on``: held-out
    data, a JSONL file / directory / list, greedy) soup to ``out``. Returns ``soup.json``."""
    specs = [m if isinstance(m, MemberSpec) else MemberSpec.parse(m) for m in members]
    loaded = [load_member(s) for s in specs]
    check_compatible(loaded)
    n = len(loaded)
    weights, extra = [1.0 / n] * n, {"method": "uniform"}
    if greedy_on is not None:
        scorer = Scorer(loaded, greedy_on, metric, orders=orders, noul_order=noul_order)
        weights, log = greedy(scorer)
        del scorer
        on = [str(p) for p in greedy_on] if isinstance(greedy_on, list | tuple) else str(greedy_on)
        extra = {"method": "greedy", "select_on": on, "metric": metric, "orders": orders,
                 "noul_order": noul_order, "greedy": log}  # fmt: skip
    path = write(out, loaded, weights, extra)
    return json.loads((path / SOUP_FILE).read_text(encoding="utf-8"))
