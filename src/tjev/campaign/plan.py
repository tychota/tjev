"""TPU campaign planner: which runs to do, what they settle, the final recipes and their cost.

    tjev campaign plan PHASE --runs RUNS --mix MIX --models MODELS [--out QUEUE]
    tjev campaign fit | select --runs RUNS
    tjev campaign cost [--bench BENCH.json] [--chips 8] [--rate USD_PER_CHIP_HOUR]
    budget knobs: --seeds --arm-seeds --sweep-steps --arms --no-muon

Three phases, each planned from the results of the ones before (docs/tpu.md):

  sweep     the proxy (2B) at 600 steps: an AdamW rate bracket √2 apart around the prior
            (``seeds`` seeds; the centre rate one more), a Muon Polar Express bracket, and
            paired arms at the centre rate (ARMS: clipping, batch, rank, Adam β2, fp32
            storage).
  transfer  the proxy's horizon at its fitted rate and at LOW_LR × it (each one 2400-step
            run with WSD cooldown branches at 600 and 1200 steps: the best horizon and the
            horizon exponent γ), and the other sizes at the predicted rate × {½, 1, 2}.
  final     every final size at its recipe: one long run and cooldown branches at ¼ and ½
            of its horizon; ``select`` picks the best of the three per size.

How ``fit`` decides (priors: local mix-v3 runs, docs/training.md):
  * Runs with the same seed see the same batches (segment i is a function of (seed, i)), so
    the *paired online loss* (training NLL over the last 20% of steps, window by window)
    compares them with SE ~0.01 from one seed, against a seed sd of ~0.026 for the eval
    score. Batch arms are compared at matching token counts (the same segments). The rate
    curve is a quadratic in log2(lr) with seed fixed effects on the online loss (the eval
    score if it brackets no optimum).
  * An arm, or Muon (its best grid run against AdamW's), is adopted when its paired online
    difference is below −max(FLOOR, 2 SE) and its eval score is not worse by more than FLOOR.
    ``report`` arms (fp32 storage) are measured, never adopted.
  * Rank: rank 64 on the proxy (r64 or r64-samelr better) moves the small sizes to rank 64;
    the rank rate rule is lr/√2 per rank doubling (rsLoRA: the optimum moves as 1/√r with
    α/√r scaling; "LoRA Without Regret", thinkingmachines.ai/blog/lora) unless r64-samelr
    beats r64. Across widths lr*(d) = lr*(d_proxy)·(d/d_proxy)^-P_WIDTH (arXiv 2609.01244
    fits ~0 for Qwen LoRA, Tinker 0.08). Halving the batch: lr × 0.85 (arXiv 2609.01244;
    2507.07101: √batch is too steep).
  * Horizon: lr* ∝ steps^-γ. The paired online differences of the two transfer rates at 600
    and 2400 steps locate each horizon's optimum on the sweep curve; their shift gives γ,
    shrunk to the prior N(0.15, 0.15²) (fine-tuning evidence is near zero; 0.32 is a
    pretraining fit, Bjorck et al. arXiv 2409.19913).
  * WSD: a cooldown branch off the stable phase trains exactly what a shorter run would
    (Hägele et al., arXiv 2405.18392), so one long run prices several horizons.

``--no-kernels`` (set by the campaign when the TPU kernel tests fail on the VM) runs every
job on the XLA paths (compute.attention=xla, compute.gdn_impl=chunked).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tjev.model import ModelConfig
from tjev.train.flops import train_flops

SIZES: dict[str, dict[str, Any]] = {  # Qwen3.5 text configs (HF config.json); 9B has an untied lm_head
    "0.8B": {"hidden_size": 1024, "intermediate_size": 3584, "num_layers": 24, "heads": (8, 2), "gdn": 16},
    "2B": {"hidden_size": 2048, "intermediate_size": 6144, "num_layers": 24, "heads": (8, 2), "gdn": 16},
    "4B": {"hidden_size": 2560, "intermediate_size": 9216, "num_layers": 32, "heads": (16, 4), "gdn": 32},
    "9B": {"hidden_size": 4096, "intermediate_size": 12288, "num_layers": 32, "heads": (16, 4), "gdn": 32},
}  # fmt: skip
# Per accelerator: preset, bf16 peak per chip, USD per chip-hour (flex-start; Kaggle is
# free), chips per run, tokens per chip per microbatch, extra overrides (v5e: 16 GB, so 4B
# shards its base), and model-FLOP-utilisation priors (replaced by the measured tokens/s
# of `tjev campaign bench`).
HARDWARE: dict[str, dict[str, Any]] = {
    "v6e": {
        "preset": "tpu-v6e", "peak": 918e12, "rate": 1.35,
        "chips": {"0.8B": 1, "2B": 1, "4B": 2, "9B": 4},
        "microbatch": {"0.8B": 8192, "2B": 8192, "4B": 4096, "9B": 2048},
        "extra": {}, "mfu": {"0.8B": 0.09, "2B": 0.18, "4B": 0.19, "9B": 0.28},
    },
    "v5e": {
        "preset": "tpu-v5e", "peak": 197e12, "rate": 0.0,
        "chips": {"0.8B": 1, "2B": 1, "4B": 2, "9B": 8},
        "microbatch": {"0.8B": 8192, "2B": 4096, "4B": 2048, "9B": 1024},
        "extra": {"4B": ["mesh.fsdp=2"], "9B": ["mesh.fsdp=8"]},
        "mfu": {"0.8B": 0.15, "2B": 0.30, "4B": 0.32, "9B": 0.35},
    },
}  # fmt: skip
RANK = {"0.8B": 32, "2B": 32, "4B": 64, "9B": 64}
PRIOR_LR = 4e-5  # AdamW, 2B, 600 steps (local mix-v3 paired runs: optimum ≤ 4.5e-5)
SWEEP_GRID = (2.5e-5, 3.54e-5, 5e-5, 7.07e-5)  # AdamW, √2 apart
MUON_GRID = (2.5e-5, 3.54e-5, 5e-5)  # Muon Polar Express (local: 3.5e-5 < 5e-5 < 7e-5)
MUON = ("optim.name=muon",)
CENTRE = SWEEP_GRID[1]  # the arms pair with the sweep runs at this rate
BATCH_LR = 0.85  # lr factor per halving of the batch
ROOT_HALF = math.sqrt(0.5)


@dataclass(frozen=True)
class Arm:
    """A paired sweep arm: overrides on top of the centre run, its rate factor, its batch
    halvings (steps × 2^halvings, compared at matching tokens), and whether ``fit`` may
    adopt it (``report`` arms only measure)."""

    overrides: tuple[str, ...]
    lr_factor: float = 1.0
    halvings: int = 0
    report: bool = False


ARMS = {
    "zclip": Arm(("optim.clip_mode=zscore",)),
    "noclip": Arm(("optim.clip_mode=none",)),
    "b32k": Arm(("train.tokens_per_step=32768",), BATCH_LR, 1),
    "b16k": Arm(("train.tokens_per_step=16384",), BATCH_LR**2, 2),
    "r64": Arm(("lora.rank=64",), ROOT_HALF),  # the rsLoRA 1/√r rule
    "r64-samelr": Arm(("lora.rank=64",)),  # against r64: does the rate move with the rank?
    "b2-0.99": Arm(("optim.b2=0.99",)),
    # fp32 weights and activations with one-pass bf16 matmuls (a true fp32 control needs
    # 6-pass HIGHEST matmuls, ~4-6× the cost): does bf16 storage cost quality?
    "fp32-storage": Arm(("model.dtype=float32",), report=True),
}
P_WIDTH = 0.05  # lr*(d) ∝ d^-P_WIDTH
GAMMA = (0.15, 0.15)  # prior on γ in lr* ∝ steps^-γ: mean, sd
LOW_LR = 0.64  # the second horizon run's rate factor (4^-0.32: a bracket)
TOKENS_PER_STEP = 65536
HORIZONS = (600, 1200, 2400)
CHECKPOINT_EVERY = 50  # the schema default: a checkpoint and a full eval
FLOOR = 0.007  # smallest effect acted on: a same-seed rerun differs by this much
ONLINE_TAIL = 0.2  # the online loss: training NLL over the last 20% of steps
WINDOW = 10


@dataclass(frozen=True)
class Campaign:
    """Where the campaign's inputs are, what it trains, and how much it spends."""

    runs: Path
    mix: str
    models: str
    hardware: str = "v6e"
    proxy: str = "2B"
    sweep_steps: int = 600
    seeds: int = 1  # seeds of the AdamW grid (the centre rate gets one more)
    arm_seeds: int = 1  # seeds of every arm and of the Muon grid
    arms: tuple[str, ...] = tuple(ARMS)
    muon: bool = True
    final_sizes: tuple[str, ...] = ("4B", "2B", "0.8B")  # longest first: the queue fills chips
    kernels: bool = True
    wandb: dict[str, str] = field(default_factory=dict)  # project, entity, mode, campaign

    @property
    def hw(self) -> dict[str, Any]:
        return HARDWARE[self.hardware]

    def others(self) -> list[str]:
        return [s for s in ("0.8B", "4B") if s != self.proxy]


# ------------------------------------------------------------------------------------------
# Queue lines: "<name> chips=N post=quick|full|none [after=RUN@STEP] <tjev train args>"


def _common(c: Campaign, size: str) -> list[str]:
    return [
        c.hw["preset"],
        f"qwen35-{size.lower()}",
        f"{c.mix}/mix.yaml",
        f"model.path={c.models}/Qwen3.5-{size}",
        f"train.microbatch_tokens={c.hw['microbatch'][size]}",
        *c.hw["extra"].get(size, []),
        f"data.heldout={c.mix}/heldout-select.jsonl",
    ]


def _wandb(c: Campaign, name: str, size: str, chips: int) -> list[str]:
    if not c.wandb:
        return []
    phase = name.split("-", 1)[0]
    tags = [phase, size, f"{c.hardware}-x{chips}"]
    out = [
        "log.wandb=true",
        f"log.wandb_project={c.wandb.get('project', 'tjev')}",
        f"log.wandb_group={c.wandb.get('campaign', 'tpu')}/{phase}",
        f"log.wandb_tags=[{','.join(tags)}]",
        f"log.wandb_mode={c.wandb.get('mode', 'online')}",
    ]
    if c.wandb.get("entity"):
        out.append(f"log.wandb_entity={c.wandb['entity']}")
    return out


def job(c: Campaign, name: str, size: str, overrides: list[str], *, steps: int,
        chips: int | None = None, after: str | None = None, post: str = "quick") -> str:  # fmt: skip
    chips = chips or c.hw["chips"][size]
    fields = [name, f"chips={chips}", f"post={post}", *([f"after={after}"] if after else [])]
    args = [
        *_common(c, size),
        f"train.steps={steps}",  # checkpoint + full eval every 50 steps (the default)
        *overrides,
        *_wandb(c, name, size, chips),
    ]
    if not c.kernels:
        args += ["compute.attention=xla", "compute.gdn_impl=chunked"]
    return " ".join(fields + args)


def horizon_jobs(c: Campaign, prefix: str, size: str, overrides: list[str], horizon: int,
                 chips: int | None = None, post: str = "quick") -> list[str]:  # fmt: skip
    """One run to ``horizon`` plus WSD cooldown branches at horizon/4 and horizon/2 (each
    branch keeps the 20% cooldown fraction)."""
    long = f"{prefix}-h{horizon}"
    every = horizon // 10  # the branch points 0.2·H and 0.4·H are multiples
    out = [job(c, long, size, [*overrides, f"train.checkpoint_every={every}",
                               "train.keep_checkpoints=12"], steps=horizon, chips=chips,
               post=post)]  # fmt: skip
    for short in (horizon // 4, horizon // 2):
        src = f"{long}@{int(0.8 * short)}"
        out.append(job(c, f"{prefix}-h{short}", size, [*overrides, f"train.branch_from=$RUNS/{src}"],
                       steps=short, chips=chips, after=src, post=post))  # fmt: skip
    return out


def _sweep(c: Campaign) -> list[str]:
    steps, lines = c.sweep_steps, []
    for seed in range(c.seeds):  # all of one seed first: a first read after a fraction
        lines += [job(c, f"sweep-lr{lr:.3g}-s{seed}", c.proxy, [f"optim.lr={lr}", f"train.seed={seed}"],
                      steps=steps) for lr in SWEEP_GRID]  # fmt: skip
    lines.append(job(c, f"sweep-lr{CENTRE:.3g}-s{c.seeds}", c.proxy,
                     [f"optim.lr={CENTRE}", f"train.seed={c.seeds}"], steps=steps))  # fmt: skip
    for seed in range(c.arm_seeds):
        if c.muon:
            lines += [job(c, f"sweep-muon-lr{lr:.3g}-s{seed}", c.proxy,
                          [*MUON, f"optim.lr={lr}", f"train.seed={seed}"], steps=steps)
                      for lr in MUON_GRID]  # fmt: skip
        for name in c.arms:
            arm = ARMS[name]
            lines.append(job(c, f"sweep-{name}-s{seed}", c.proxy,
                             [f"optim.lr={CENTRE * arm.lr_factor:.3g}", f"train.seed={seed}",
                              *arm.overrides], steps=steps * 2**arm.halvings))  # fmt: skip
    return lines


def plan(phase: str, c: Campaign) -> list[str]:
    if phase == "sweep":
        return _sweep(c)
    report = fit(c)
    recipe = report["recipe"]
    if phase == "transfer":
        proxy = recipe[c.proxy]
        rest = [o for o in proxy["overrides"] if not o.startswith("optim.lr=")]
        chips = 4 if c.hardware == "v6e" else None  # the long pole: data-parallel on 4 chips
        lines = []
        for tag, lr in (("hi", proxy["sweep_lr"]), ("lo", LOW_LR * proxy["sweep_lr"])):
            lines += horizon_jobs(c, f"transfer-{c.proxy.lower()}-{tag}", c.proxy,
                                  [*rest, f"optim.lr={lr:.3g}", "train.seed=0"], max(HORIZONS),
                                  chips=chips)  # fmt: skip
        for size in c.others():
            rest = [o for o in recipe[size]["overrides"] if not o.startswith("optim.lr=")]
            lr = recipe[size]["sweep_lr"]
            lines += [job(c, f"transfer-{size.lower()}-lr{lr * f:.3g}", size,
                          [*rest, f"optim.lr={lr * f:.3g}", "train.seed=0"], steps=c.sweep_steps)
                      for f in (0.5, 1.0, 2.0)]  # fmt: skip
        return lines
    if phase == "final":
        return [line for size in c.final_sizes
                for line in horizon_jobs(c, f"final-{size.lower()}", size,
                                         recipe[size]["overrides"], recipe[size]["horizon"])]  # fmt: skip
    raise ValueError(f"unknown phase {phase!r} (sweep, transfer, final)")


# ------------------------------------------------------------------------------------------
# Results


def online_loss(run: Path, steps: int) -> dict:
    """Training NLL in WINDOW-step windows (last step → mean) and its mean over the last
    ONLINE_TAIL of the steps (``online``)."""
    path = run / "metrics.jsonl"
    by: dict[int, float] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line) if line.strip() else {}
            if "learning/nll" in row:
                by[row["step"]] = row["learning/nll"]
    windows = {}
    for end in range(WINDOW, steps + 1, WINDOW):
        vals = [by[s] for s in range(end - WINDOW + 1, end + 1) if s in by]
        if len(vals) == WINDOW:
            windows[end] = float(np.mean(vals))
    tail = [v for k, v in windows.items() if k > (1 - ONLINE_TAIL) * steps]
    return {"windows": windows, "online": float(np.mean(tail)) if tail else None}


def load_runs(runs: Path) -> list[dict]:
    """Finished runs (``queue.done``) with their parameters and selection scores."""
    out = []
    for run in sorted(p for p in runs.iterdir() if (p / "queue.done").exists()):
        if not (run / "selection.json").exists():
            continue
        cfg = json.loads((run / "config.json").read_text())["config"]
        sel = json.loads((run / "selection.json").read_text())
        steps = cfg["train"]["steps"]
        out.append({
            "name": run.name,
            "phase": run.name.split("-", 1)[0],
            "size": cfg["model"]["path"].rstrip("/").rsplit("Qwen3.5-", 1)[-1],
            "lr": cfg["optim"]["lr"],
            "seed": cfg["train"]["seed"],
            "tokens_per_step": cfg["train"]["tokens_per_step"],
            "steps": steps,
            "score": sel["score"],
            **online_loss(run, steps),
        })  # fmt: skip
    return out


def online_delta(run: dict, base: dict, halvings: int = 0) -> tuple[float, float] | None:
    """Mean window difference (run − base) over the online tail, and its SE; only for runs
    on the same batches. ``halvings``: ``run`` has 2^h × the steps at 1/2^h of the batch, so
    its window at step k·2^h covers the same segments as the base's window at step k."""
    scale = 2**halvings
    if run["seed"] != base["seed"] or run["steps"] != base["steps"] * scale:
        return None
    if run["tokens_per_step"] * scale != base["tokens_per_step"]:
        return None
    windows = {k // scale: v for k, v in run["windows"].items() if k % (scale * WINDOW) == 0}
    tail = [k for k in windows if k > (1 - ONLINE_TAIL) * base["steps"] and k in base["windows"]]
    if len(tail) < 3:
        return None
    d = np.asarray([windows[k] - base["windows"][k] for k in tail])
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def combine(pairs: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Mean of per-seed paired differences and the SE of that mean."""
    if not pairs:
        return None
    return float(np.mean([p[0] for p in pairs])), math.sqrt(sum(p[1] ** 2 for p in pairs)) / len(
        pairs
    )


def adopted(online: tuple[float, float] | None, score: float) -> bool:
    return bool(online and online[0] < -max(FLOOR, 2 * online[1]) and score <= FLOOR)


def lr_curve(points: list[tuple[float, float, int]]) -> dict:
    """y = a_seed + b·x + c·x² with x = log2(lr): the optimum (clipped half an octave past
    the grid), whether the grid brackets it, and the curvature c. ``points``: (lr, value,
    seed)."""
    lrs = np.asarray([p[0] for p in points])
    ys = np.asarray([p[1] for p in points])
    seeds = np.asarray([p[2] for p in points])
    means = {float(lr): float(ys[lrs == lr].mean()) for lr in np.unique(lrs)}
    best = min(means, key=means.__getitem__)
    out: dict[str, Any] = {"opt": best, "best_grid": best, "bracketed": False,
                           "curvature": None, "means": means}  # fmt: skip
    if len(means) < 3:
        return out
    x = np.log2(lrs)
    levels = sorted(set(seeds.tolist()))
    design = np.column_stack([*(seeds == s for s in levels), x, x**2]).astype(float)
    coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
    b, c = float(coef[-2]), float(coef[-1])
    out["curvature"] = c
    if c <= 0:
        return out
    lo, hi = x.min(), x.max()
    vertex = -b / (2 * c)
    return out | {"opt": 2 ** float(np.clip(vertex, lo - 0.5, hi + 0.5)),
                  "bracketed": bool(lo - 0.5 <= vertex <= hi + 0.5)}  # fmt: skip


def rate_curve(records: list[dict], prefix: str) -> dict | None:
    """The rate curve of the sweep runs named ``prefix``-lr…: on the online loss, else on
    the eval score when that brackets the optimum and the online loss does not."""
    runs = [r for r in records if r["name"].startswith(prefix)]
    online = [(r["lr"], r["online"], r["seed"]) for r in runs if r["online"] is not None]
    curve = lr_curve(online) | {"source": "online"} if len({p[0] for p in online}) >= 3 else None
    if curve is None or not curve["bracketed"]:
        scores = [(r["lr"], r["score"], r["seed"]) for r in runs]
        if len({p[0] for p in scores}) >= 3:
            by_score = lr_curve(scores) | {"source": "score"}
            if by_score["bracketed"] or curve is None:
                curve = by_score
    return curve


def _named(records: list[dict], name: str) -> dict | None:
    return next((r for r in records if r["name"] == name), None)


def arm_effects(records: list[dict], arm_seeds: int = 1) -> dict[str, dict]:
    """Per sweep arm, the paired online difference (seeds combined) and the eval score
    difference against the sweep run at the centre rate with the same seed."""
    out = {}
    for name, arm in ARMS.items():
        pairs, scores = [], []
        for seed in range(arm_seeds):
            base = _named(records, f"sweep-lr{CENTRE:.3g}-s{seed}")
            run = _named(records, f"sweep-{name}-s{seed}")
            if base is None or run is None:
                continue
            scores.append(run["score"] - base["score"])
            if (d := online_delta(run, base, arm.halvings)) is not None:
                pairs.append(d)
        if scores:
            online, score = combine(pairs), float(np.mean(scores))
            out[name] = {"online": online, "score": score,
                         "better": not arm.report and adopted(online, score)}  # fmt: skip
    return out


def rank_rule(records: list[dict], effects: dict, arm_seeds: int) -> tuple[bool, float]:
    """(rank 64 for the small sizes, the rate factor per rank doubling). r64-samelr against
    r64 (same seed, same batches) says whether the rate should move with the rank."""
    big = bool(effects.get("r64", {}).get("better") or effects.get("r64-samelr", {}).get("better"))
    pairs, scores = [], []
    for seed in range(arm_seeds):
        same, scaled = (
            _named(records, f"sweep-r64-samelr-s{seed}"),
            _named(records, f"sweep-r64-s{seed}"),
        )
        if same and scaled:
            scores.append(same["score"] - scaled["score"])
            if (d := online_delta(same, scaled)) is not None:
                pairs.append(d)
    keep_rate = bool(scores) and adopted(combine(pairs), float(np.mean(scores)))
    return big, 1.0 if keep_rate else ROOT_HALF


def muon_choice(records: list[dict], adamw: dict | None, muon: dict | None) -> dict:
    """Muon replaces AdamW only when its best grid run beats AdamW's (seed 0, paired)."""
    if not adamw or not muon:
        return {"use": False}
    base = _named(records, f"sweep-lr{adamw['best_grid']:.3g}-s0")
    run = _named(records, f"sweep-muon-lr{muon['best_grid']:.3g}-s0")
    if base is None or run is None:
        return {"use": False}
    online, score = online_delta(run, base), run["score"] - base["score"]
    return {"use": adopted(online, score), "online": online, "score": score}


def horizon_gamma(records: list[dict], proxy: str, curvature: float | None) -> dict:
    """γ from the transfer runs: at H = 600 and 2400 steps the paired online difference
    d = lo − hi (LOW_LR × the rate against the rate) places that horizon's optimum on the
    sweep quadratic, x*_H − x_hi = dx/2 − d/(2·c·dx) with dx = log2(LOW_LR); γ is the
    optimum's shift per log2 of steps, shrunk to the prior."""
    prior, prior_sd = GAMMA
    short, long = min(HORIZONS), max(HORIZONS)
    shifts = {}
    for steps in (short, long):
        hi = _named(records, f"transfer-{proxy.lower()}-hi-h{steps}")
        lo = _named(records, f"transfer-{proxy.lower()}-lo-h{steps}")
        if hi and lo and (d := online_delta(lo, hi)) is not None:
            shifts[steps] = d
    if len(shifts) < 2 or not curvature or curvature <= 0:
        return {"gamma": prior, "measured": None}
    dx = math.log2(LOW_LR)

    def shift(d: float) -> float:
        return dx / 2 - d / (2 * curvature * dx)

    span = math.log2(long / short)
    measured = -(shift(shifts[long][0]) - shift(shifts[short][0])) / span
    se = math.hypot(shifts[long][1], shifts[short][1]) / (2 * curvature * abs(dx) * span)
    w_m, w_p = 1 / max(se, 1e-6) ** 2, 1 / prior_sd**2
    gamma = float(np.clip((measured * w_m + prior * w_p) / (w_m + w_p), 0.0, 0.6))
    return {"gamma": gamma, "measured": measured, "se": se}


def fit(c: Campaign) -> dict:
    records = load_runs(c.runs) if c.runs.exists() else []
    adamw = rate_curve(records, "sweep-lr")
    muon = rate_curve(records, "sweep-muon-lr")
    use_muon = muon_choice(records, adamw, muon)
    anchor = muon if use_muon["use"] else adamw
    proxy_lr = anchor["opt"] if anchor else PRIOR_LR
    effects = arm_effects(records, c.arm_seeds)
    clip = next((list(ARMS[a].overrides) for a in ("zclip", "noclip")
                 if effects.get(a, {}).get("better")), [])  # fmt: skip
    batch = min((a for a in ("b32k", "b16k") if effects.get(a, {}).get("better")),
                key=lambda a: effects[a]["online"][0], default=None)  # fmt: skip
    halvings = ARMS[batch].halvings if batch else 0
    tokens = TOKENS_PER_STEP // 2**halvings
    rank_big, rank_scale = rank_rule(records, effects, c.arm_seeds)
    optimizer = list(MUON) if use_muon["use"] else []
    if effects.get("b2-0.99", {}).get("better"):
        optimizer.append("optim.b2=0.99")
    # horizon: the best of the proxy's hi runs (default 1200), γ from the hi/lo pairs
    horizon = {r["steps"]: r["score"] for r in records
               if r["name"].startswith(f"transfer-{c.proxy.lower()}-hi-h")}  # fmt: skip
    best_h = min(horizon, key=horizon.__getitem__) if horizon else 1200
    gamma = horizon_gamma(records, c.proxy, anchor.get("curvature") if anchor else None)
    final_h = best_h * 2**halvings
    horizon_factor = (best_h / c.sweep_steps) ** -gamma["gamma"]
    recipe = {}
    for size, dims in SIZES.items():
        rank = 64 if rank_big else RANK[size]
        rows = [r for r in records if r["name"].startswith(f"transfer-{size.lower()}-lr")]
        measured = lr_curve([(r["lr"], r["score"], r["seed"]) for r in rows]) if rows else None
        if measured and measured["bracketed"]:
            lr = measured["opt"]  # already at this size's rank and batch
        else:
            lr = proxy_lr * (dims["hidden_size"] / SIZES[c.proxy]["hidden_size"]) ** -P_WIDTH
            proxy_rank = 64 if rank_big else RANK[c.proxy]
            lr *= rank_scale ** math.log2(rank / proxy_rank)
            lr *= BATCH_LR**halvings
        overrides = [*optimizer, *clip, f"lora.rank={rank}", f"train.tokens_per_step={tokens}"]
        final = float(f"{lr * horizon_factor:.2g}")
        recipe[size] = {"lr": final, "sweep_lr": float(f"{lr:.3g}"), "rank": rank,
                        "horizon": final_h, "overrides": [*overrides, f"optim.lr={final}"]}  # fmt: skip
    return {
        "runs": len(records),
        "curves": {"adamw": adamw, "muon": muon},
        "effects": effects,
        "muon": use_muon,
        "horizon": gamma,
        "decisions": {
            "optimizer": " ".join(optimizer) or "adamw",
            "proxy_lr": proxy_lr,
            "clipping": clip or "global",
            "tokens_per_step": tokens,
            "rank_small_sizes": 64 if rank_big else 32,
            "rank_lr_factor_per_doubling": rank_scale,
            "best_horizon": best_h,
            "final_horizon": final_h,
            "horizon_gamma": gamma["gamma"],
            "horizon_lr_factor": horizon_factor,
        },
        "recipe": recipe,
    }


def select(runs: Path) -> dict[str, dict]:
    """Per size, the best final run (long run or cooldown branch) by selection score."""
    best: dict[str, dict] = {}
    for r in load_runs(runs):
        if r["phase"] == "final" and (
            r["size"] not in best or r["score"] < best[r["size"]]["score"]
        ):
            best[r["size"]] = r
    return best


# ------------------------------------------------------------------------------------------
# Cost


def model_config(size: str) -> ModelConfig:
    d = SIZES[size]
    return ModelConfig(
        vocab_size=248320, hidden_size=d["hidden_size"], intermediate_size=d["intermediate_size"],
        num_layers=d["num_layers"], num_heads=d["heads"][0], num_kv_heads=d["heads"][1],
        head_dim=256, linear_num_key_heads=16, linear_num_value_heads=d["gdn"],
        linear_key_head_dim=128, linear_value_head_dim=128,
    )  # fmt: skip


def tokens_per_second(c: Campaign, size: str, bench: dict | None = None) -> float:
    """Per chip: measured (``tjev campaign bench`` on this hardware) or the MFU prior."""
    if bench and size in bench.get("tokens_per_second_per_chip", {}):
        return bench["tokens_per_second_per_chip"][size]
    per_token = train_flops(model_config(size), np.asarray([400.0])) / 400  # ~400-token items
    return c.hw["mfu"][size] * c.hw["peak"] / per_token


EVAL_ITEMS = 3400  # validation (64 per source) + held-out generators, ~400 tokens each


def cost(c: Campaign, phases: tuple[str, ...] = ("sweep", "transfer", "final"),
         bench: dict | None = None, vm_chips: int = 8, rate: float | None = None) -> dict:  # fmt: skip
    """Chip-hours per phase (training tokens at the measured or prior speed, a full eval at
    every checkpoint, compile and post-training), the wall time on one VM if its chips stay
    busy, and USD."""
    rate = c.hw["rate"] if rate is None else rate
    out = {}
    for phase in phases:
        chip_s = 0.0
        lines = plan(phase, c)
        for line in lines:
            fields = dict(f.split("=", 1) for f in line.split() if "=" in f)
            size = next(t for t in line.split() if t.startswith("qwen35-"))[7:].upper()
            steps = int(fields["train.steps"])
            if "train.branch_from" in fields:
                steps -= int(fields["train.branch_from"].rsplit("@", 1)[1])
            tokens = steps * int(fields.get("train.tokens_per_step", TOKENS_PER_STEP))
            chips = min(int(fields["chips"]), vm_chips)
            speed = tokens_per_second(c, size, bench) * chips
            slow = 1.6 if fields.get("model.dtype") == "float32" else 1.0  # fp32 storage
            n_evals = math.ceil(steps / int(fields.get("train.checkpoint_every", CHECKPOINT_EVERY)))
            evals = n_evals * EVAL_ITEMS * 400 / (3 * speed)  # forward ≈ ⅓ of a train token
            chip_s += chips * (slow * tokens / speed + evals + 300)
        wall_h = chip_s / 3600 / vm_chips
        out[phase] = {"jobs": len(lines), "chip_hours": chip_s / 3600, "wall_h": wall_h,
                      "usd": wall_h * vm_chips * rate}  # fmt: skip
    out["total"] = {
        k: sum(v[k] for v in out.values()) for k in ("jobs", "chip_hours", "wall_h", "usd")
    }
    return out


def campaign_from_env(runs: Path, mix: str, models: str, **kw: Any) -> Campaign:
    """W&B logging when TJEV_WANDB=1 (project, entity, mode, campaign from TJEV_WANDB_*)."""
    wandb = {}
    if os.environ.get("TJEV_WANDB") == "1":
        wandb = {k: os.environ[f"TJEV_WANDB_{k.upper()}"] for k in ("project", "entity", "mode", "campaign")
                 if os.environ.get(f"TJEV_WANDB_{k.upper()}")}  # fmt: skip
        wandb.setdefault("project", "tjev")
    return Campaign(runs=runs, mix=mix, models=models, wandb=wandb, **kw)
