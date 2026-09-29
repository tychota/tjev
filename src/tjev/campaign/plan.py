"""TPU campaign planner: which runs to do, what they settle, the final recipes and their cost.

    tjev campaign plan PHASE --runs RUNS --mix MIX --models MODELS [--out QUEUE]
    tjev campaign fit | select --runs RUNS
    tjev campaign cost [--bench BENCH.json] [--chips 8] [--rate USD_PER_CHIP_HOUR]

Three phases, each planned from the results of the ones before (docs/tpu.md):

  sweep     the proxy (2B) at 600 steps: an AdamW rate bracket √2 apart around the prior
            (seed 0, the centre also seed 1), and paired arms at the centre rate on seed 0
            (z-score clipping, no clipping, half the batch at lr × 0.85 for twice the steps).
  transfer  the proxy's horizon (one 2400-step run with WSD cooldown branches at 600 and
            1200 steps) and the other sizes at the predicted rate × {½, 1, 2}.
  final     every final size at its recipe: one long run and cooldown branches at ¼ and ½
            of its horizon; ``select`` picks the best of the three per size.

How ``fit`` decides (priors: local mix-v3 runs, docs/training.md):
  * Runs with the same seed see the same batches, so the *paired online loss* (training
    NLL over the last 20% of steps, window by window) compares them with SE ~0.01 from one
    seed, against a seed sd of ~0.026 for the eval score. The rate curve is a quadratic in
    log2(lr) with seed fixed effects on the online loss (the eval score if it brackets no
    optimum).
  * An arm is adopted when its paired online difference is below −max(FLOOR, 2 SE) and its
    eval score is not worse by more than FLOOR.
  * Rank 64 (4B, 9B) takes lr/√2 (rsLoRA: with α/√r scaling the optimum moves as 1/√r;
    "LoRA Without Regret", thinkingmachines.ai/blog/lora). Across widths
    lr*(d) = lr*(d_proxy)·(d/d_proxy)^-P_WIDTH (arXiv 2609.01244 fits ~0 for Qwen LoRA,
    Tinker 0.08). Halving the batch: lr × 0.85 (arXiv 2609.01244; 2507.07101: √batch is
    too steep). Horizon: lr ∝ steps^-GAMMA from the sweep length (fine-tuning evidence is
    near zero; 0.32 is a pretraining fit, Bjorck et al. arXiv 2409.19913).
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
SWEEP_GRID = (2.5e-5, 3.54e-5, 5e-5, 7.07e-5)
CENTRE = SWEEP_GRID[1]  # the arms pair with the sweep runs at this rate
ARMS = {
    "zclip": ["optim.clip_mode=zscore"],
    "noclip": ["optim.clip_mode=none"],
    "b32k": ["train.tokens_per_step=32768"],  # at lr × BATCH_LR, for twice the steps
}
BATCH_LR = 0.85  # lr factor per halving of the batch
P_WIDTH = 0.05  # lr*(d) ∝ d^-P_WIDTH
GAMMA = 0.15  # lr* ∝ steps^-GAMMA
TOKENS_PER_STEP = 65536
HORIZONS = (600, 1200, 2400)
FLOOR = 0.007  # smallest effect acted on: a same-seed rerun differs by this much
ONLINE_TAIL = 0.2  # the online loss: training NLL over the last 20% of steps
WINDOW = 10


@dataclass(frozen=True)
class Campaign:
    """Where the campaign's inputs are and what it trains."""

    runs: Path
    mix: str
    models: str
    hardware: str = "v6e"
    proxy: str = "2B"
    sweep_steps: int = 600
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
    evals = max(1, steps // 10)  # a full eval and a checkpoint at every tenth of the run
    fields = [name, f"chips={chips}", f"post={post}", *([f"after={after}"] if after else [])]
    args = [
        *_common(c, size),
        f"train.steps={steps}",
        f"train.eval_every={evals}",
        f"train.checkpoint_every={evals}",
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


def plan(phase: str, c: Campaign) -> list[str]:
    if phase == "sweep":
        steps = c.sweep_steps
        lines = [job(c, f"sweep-lr{lr:.3g}-s0", c.proxy, [f"optim.lr={lr}", "train.seed=0"],
                     steps=steps) for lr in SWEEP_GRID]  # fmt: skip
        lines.append(job(c, f"sweep-lr{CENTRE:.3g}-s1", c.proxy,
                         [f"optim.lr={CENTRE}", "train.seed=1"], steps=steps))  # fmt: skip
        for arm, overrides in ARMS.items():
            batch = arm == "b32k"
            lr = CENTRE * BATCH_LR if batch else CENTRE
            lines.append(job(c, f"sweep-{arm}-s0", c.proxy,
                             [f"optim.lr={lr:.3g}", "train.seed=0", *overrides],
                             steps=2 * steps if batch else steps))  # fmt: skip
        return lines
    recipe = fit(c)["recipe"]
    if phase == "transfer":
        rest = [o for o in recipe[c.proxy]["overrides"] if not o.startswith("optim.lr=")]
        lr = recipe[c.proxy]["sweep_lr"]
        lines = horizon_jobs(c, f"transfer-{c.proxy.lower()}", c.proxy,
                             [*rest, f"optim.lr={lr:.3g}", "train.seed=0"], max(HORIZONS),
                             chips=4 if c.hardware == "v6e" else None)  # fmt: skip
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


def online_delta(run: dict, base: dict) -> tuple[float, float] | None:
    """Mean window difference (run − base) over the run's online tail, and its SE; only for
    runs on the same batches (same seed, batch size and steps)."""
    if any(run[k] != base[k] for k in ("seed", "tokens_per_step", "steps")):
        return None
    tail = [
        k for k in run["windows"] if k > (1 - ONLINE_TAIL) * run["steps"] and k in base["windows"]
    ]
    if len(tail) < 3:
        return None
    d = np.asarray([run["windows"][k] - base["windows"][k] for k in tail])
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def lr_curve(points: list[tuple[float, float, int]]) -> dict:
    """y = a_seed + b·x + c·x² with x = log2(lr): the optimum (clipped half an octave past
    the grid) and whether the grid brackets it. ``points``: (lr, value, seed)."""
    lrs = np.asarray([p[0] for p in points])
    ys = np.asarray([p[1] for p in points])
    seeds = np.asarray([p[2] for p in points])
    means = {float(lr): float(ys[lrs == lr].mean()) for lr in np.unique(lrs)}
    best = min(means, key=means.__getitem__)
    out = {"opt": best, "bracketed": False, "means": means}
    if len(means) < 3:
        return out
    x = np.log2(lrs)
    levels = sorted(set(seeds.tolist()))
    design = np.column_stack([*(seeds == s for s in levels), x, x**2]).astype(float)
    coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
    b, c = float(coef[-2]), float(coef[-1])
    if c <= 0:
        return out
    lo, hi = x.min(), x.max()
    vertex = -b / (2 * c)
    return out | {"opt": 2 ** float(np.clip(vertex, lo - 0.5, hi + 0.5)),
                  "bracketed": bool(lo - 0.5 <= vertex <= hi + 0.5)}  # fmt: skip


def arm_effects(records: list[dict]) -> dict[str, dict]:
    """Per sweep arm, the paired online difference and the eval score difference against
    the seed-0 sweep run at the centre rate, and whether the arm is adopted."""
    base = next((r for r in records if r["name"] == f"sweep-lr{CENTRE:.3g}-s0"), None)
    out = {}
    for arm in ARMS:
        run = next((r for r in records if r["name"] == f"sweep-{arm}-s0"), None)
        if base is None or run is None:
            continue
        online = online_delta(run, base) if arm != "b32k" else _batch_delta(run, base)
        score = run["score"] - base["score"]
        better = bool(online and online[0] < -max(FLOOR, 2 * online[1]) and score <= FLOOR)
        out[arm] = {"online": online, "score": score, "better": better}
    return out


def _batch_delta(run: dict, base: dict) -> tuple[float, float] | None:
    """Half-batch arm: same tokens at twice the steps, compared at matching token counts."""
    scaled = {**run, "steps": base["steps"], "tokens_per_step": base["tokens_per_step"],
              "windows": {k // 2: v for k, v in run["windows"].items() if k % (2 * WINDOW) == 0}}  # fmt: skip
    return online_delta(scaled, base)


def fit(c: Campaign) -> dict:
    records = load_runs(c.runs) if c.runs.exists() else []
    sweep = [r for r in records if r["name"].startswith("sweep-lr")]
    online = [(r["lr"], r["online"], r["seed"]) for r in sweep if r["online"] is not None]
    curve = lr_curve(online) if len({p[0] for p in online}) >= 3 else None
    if curve is None or not curve["bracketed"]:
        scores = [(r["lr"], r["score"], r["seed"]) for r in sweep]
        by_score = lr_curve(scores) if scores else None
        curve = by_score if by_score and (by_score["bracketed"] or curve is None) else curve
    proxy_lr = curve["opt"] if curve else PRIOR_LR
    effects = arm_effects(records)
    clip = next((ARMS[a] for a in ("zclip", "noclip") if effects.get(a, {}).get("better")), [])
    half_batch = bool(effects.get("b32k", {}).get("better"))
    tokens = TOKENS_PER_STEP // 2 if half_batch else TOKENS_PER_STEP
    # the proxy's horizon: the best of its long run and cooldown branches (default 1200)
    horizon = {
        r["steps"]: r["score"]
        for r in records
        if r["name"].startswith(f"transfer-{c.proxy.lower()}-h")
    }
    best_h = min(horizon, key=horizon.__getitem__) if horizon else 1200
    final_h = best_h * (TOKENS_PER_STEP // tokens)
    horizon_factor = (final_h * tokens / (c.sweep_steps * TOKENS_PER_STEP)) ** -GAMMA
    recipe = {}
    for size, dims in SIZES.items():
        rows = [r for r in records if r["name"].startswith(f"transfer-{size.lower()}-lr")]
        measured = lr_curve([(r["lr"], r["score"], r["seed"]) for r in rows]) if rows else None
        if measured and measured["bracketed"]:
            lr = measured["opt"]
        else:
            lr = proxy_lr * (dims["hidden_size"] / SIZES[c.proxy]["hidden_size"]) ** -P_WIDTH
            lr *= math.sqrt(RANK[c.proxy] / RANK[size])
        lr *= BATCH_LR if half_batch else 1.0
        overrides = [*clip, f"lora.rank={RANK[size]}", f"train.tokens_per_step={tokens}"]
        final = float(f"{lr * horizon_factor:.2g}")
        recipe[size] = {"lr": final, "sweep_lr": float(f"{lr:.3g}"), "rank": RANK[size],
                        "horizon": final_h, "overrides": [*overrides, f"optim.lr={final}"]}  # fmt: skip
    return {
        "runs": len(records),
        "curve": curve,
        "effects": effects,
        "decisions": {
            "proxy_lr": proxy_lr,
            "clipping": clip or "global",
            "tokens_per_step": tokens,
            "best_horizon": best_h,
            "final_horizon": final_h,
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


def cost(c: Campaign, phases: tuple[str, ...] = ("sweep", "transfer", "final"),
         bench: dict | None = None, vm_chips: int = 8, rate: float | None = None) -> dict:  # fmt: skip
    """Chip-hours per phase (training tokens at the measured or prior speed, plus compile,
    evals and post-training), the wall time on one VM if its chips stay busy, and USD."""
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
            evals = 10 * 3400 * 400 / (3 * speed)  # ~3.4k eval items forward, ⅓ of a train token
            chip_s += chips * (tokens / speed + evals + 300)
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
