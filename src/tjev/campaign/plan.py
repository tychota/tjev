"""TPU campaign planner: which runs to do, what they settle, the final recipes and their cost.

    tjev campaign plan PHASE --runs RUNS --mix MIX --models MODELS [--out QUEUE]
    tjev campaign fit | select --runs RUNS
    tjev campaign cost [--bench BENCH.json] [--chips 8] [--rate USD_PER_CHIP_HOUR]
    budget knobs: --seeds --arm-seeds --sweep-steps --arms --no-muon --proxy --centre
    sweep stages: --stage lr (the rate brackets) | arms (paired at --centre) | all
    W&B: project tjev-sweep (TJEV_WANDB_PROJECT), group CAMPAIGN/PHASE[/lr|/arms], tags
    arm:… opt:… lr:… rank:… init:… dtype:… seed:… (docs/sweep-plan.md)

Three phases, each planned from the results of the ones before (docs/tpu.md):

  sweep     the proxy (2B) at 600 steps: an AdamW rate bracket √2 apart around the prior
            (``seeds`` seeds; the centre rate one more), a Muon Polar Express bracket, and
            paired arms at the centre rate (ARMS: clipping, batch, rank, Adam β2, fp32
            storage, GDN precision, LoRA init: PiSSA, EVA, LoRA-GA). Runs stopped by
            their health rules (health.json "killed") are left out of every fit.
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
  * Rank: the rank axis r16 / r64 / r128 (scaled and same-rate arms): the best adopted rank
    goes to the small sizes (4B / 9B keep at least 64);
    the rank rate rule is lr/√2 per rank doubling (rsLoRA: the optimum moves as 1/√r with
    α/√r scaling; "LoRA Without Regret", thinkingmachines.ai/blog/lora) unless a same-rate
    arm beats its scaled twin. Across widths lr*(d) = lr*(d_proxy)·(d/d_proxy)^-P_WIDTH (arXiv 2609.01244
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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from tjev.config.schema import GATES, LORA_TARGETS
from tjev.model import ModelConfig
from tjev.train.flops import train_flops

SIZES: dict[str, dict[str, Any]] = {  # Qwen3.5 text configs (HF config.json); 9B has an untied lm_head
    "0.8B": {"hidden_size": 1024, "intermediate_size": 3584, "num_layers": 24, "heads": (8, 2), "gdn": 16},
    "2B": {"hidden_size": 2048, "intermediate_size": 6144, "num_layers": 24, "heads": (8, 2), "gdn": 16},
    "4B": {"hidden_size": 2560, "intermediate_size": 9216, "num_layers": 32, "heads": (16, 4), "gdn": 32},
    "9B": {"hidden_size": 4096, "intermediate_size": 12288, "num_layers": 32, "heads": (16, 4), "gdn": 32},
}  # fmt: skip
# Per accelerator: preset, bf16 peak per chip, USD per chip-hour (GCP list price; Colab
# charges compute units), chips per run, tokens per chip per microbatch, extra overrides (v5e: 16 GB, so 4B
# shards its base), and model-FLOP-utilisation priors (replaced by the measured tokens/s
# of `tjev campaign bench`).
# v6e1's measured speed-ups on top of the sweep's kernels (HARDWARE["v6e1"]["extra"])
V6E1_ALL = ["train.packing=best_fit", "compute.gdn_norm=flat"]
# tune-jax picks (tjev-tune1, docs on perf/tune), step time checked on v6e1 (tjev-tune-verify:
# +5-9% tok/s at 0.8B / 2B / 4B, HBM unchanged) and kernel parity (interpret mode, fwd + grads
# against the reference, rel err <= 2e-5); 9B was not tuned. The splash tiles only: the GDN pick
# (gdn_heads_per_step 32 at 4B) overflows the 32 MiB scoped VMEM of mega2's forward with the
# flat norm (39.2 MiB; tjev-v75b s1, 2026-10-06), which tune-verify's arms did not have
V6E1_TUNED = {
    s: ["compute.attention_splash_block=1024", "compute.attention_fused_bwd=true"]
    for s in ("0.8B", "2B", "4B")
}
V6E1_9B = [
    "compute.head_rows=letters",
    'compute.remat_by_bucket={"1024":"minimal_states","2048":"minimal_states"}',
]
HARDWARE: dict[str, dict[str, Any]] = {
    "v6e": {
        "preset": "tpu-v6e", "peak": 918e12, "rate": 1.35,
        "chips": {"0.8B": 1, "2B": 1, "4B": 2, "9B": 4},
        "microbatch": {"0.8B": 8192, "2B": 8192, "4B": 4096, "9B": 2048},
        "extra": {}, "mfu": {"0.8B": 0.09, "2B": 0.18, "4B": 0.19, "9B": 0.28},
    },
    # Colab v6e-1: one chip per run whatever the size, the kernels the bench measured (fused
    # Pallas GDN forward + backward, core_states remat), the rate in compute units per hour,
    # and the measured tokens/s per chip (2B / 4B interpolated, not benched): docs/sweep-plan.md
    "v6e1": {
        "preset": "tpu-v6e", "peak": 918e12, "rate": 4.08,
        "chips": {"0.8B": 1, "2B": 1, "4B": 1, "9B": 1},
        "microbatch": {"0.8B": 8192, "2B": 8192, "4B": 4096, "9B": 2048},
        # the GDN megakernel with the conv inside (validated against the official model in
        # fp32 and bf16, tjev-megaval4), chunk 128, the blocked in-kernel inverse; merged
        # LoRA where it pays (tjev-bench4: +26% 0.8B, +16% 2B, +9% 4B, −4% 9B; no quality
        # cost on two sweep seeds)
        # perf/integration (tjev-integ-4b / -9b, real tokens/s against that configuration):
        # train.packing=best_fit (padding 15.6% -> 2.9%: +13.8% at 4B, +16% at 9B), the
        # flat GDN output norm (+3.8% at 4B); 9B also the letters-only head (untied: +5.2%
        # with flat) and minimal_states remat on its 1024 / 2048 buckets (+16.9%; 4096 keeps
        # core_states). Not taken: minimal_states at <= 4B (-0.6% at 4B), the fused LoRA
        # kernel (-16% at 4B, -7% at 9B), gdn_precision=mixed (not validated on its own).
        "extra": {s: ["compute.gdn_impl=pallas_tpu_mega2", "compute.gdn_chunk=128",
                      "compute.gdn_inverse=blocked", "compute.remat=core_states",
                      *(["compute.lora_impl=merged"] if s != "9B" else V6E1_9B),
                      *V6E1_ALL, *V6E1_TUNED.get(s, [])]
                  for s in ("0.8B", "2B", "4B", "9B")},
        "mfu": {"0.8B": 0.12, "2B": 0.22, "4B": 0.22, "9B": 0.28},
        # measured (tjev-bench4, the configuration above, 16k / 8k-token steps)
        "tokens": {"0.8B": 52650.0, "2B": 36250.0, "4B": 14120.0, "9B": 9220.0},
        # the long buckets (Campaign.max_bucket > 4096): per size the largest bucket that
        # fits a chip by step_hbm_gb, its microbatch and overrides; unmeasured on the TPU
        # (docs/training.md, "Long buckets": run `tjev compile-check` before training).
        # 4B at 8192 fits only without the core_states residuals (remat=full); 9B's base
        # leaves no room past 4096 tokens per microbatch on one chip.
        "long": {
            "0.8B": {"max_bucket": 16384, "microbatch": 16384, "extra": []},
            "2B": {"max_bucket": 16384, "microbatch": 16384, "extra": []},
            "4B": {"max_bucket": 8192, "microbatch": 8192, "extra": ["compute.remat=full"]},
            "9B": {"max_bucket": 4096, "microbatch": 2048, "extra": []},
        },
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
# One train step's device memory on a v6e chip (tjev-bench4, `compiled.memory_analysis()`,
# the v6e1 configuration with merged LoRA except 9B, the preset ranks, the 2048 bucket):
# arguments (the bf16 base, the adapters and Adam's state) and temporaries at ``tokens``
# per microbatch. The process's ``perf/hbm_peak_gb`` (peak_bytes_in_use) counts only the
# arguments on TPU (10.1 GB for 4B), not the step's temporaries.
STEP_HBM = {
    "0.8B": {"arguments": 1.77, "temp": 7.30, "tokens": 8192},
    "2B": {"arguments": 4.17, "temp": 9.23, "tokens": 8192},
    "4B": {"arguments": 9.97, "temp": 11.60, "tokens": 4096},
    "9B": {"arguments": 19.99, "temp": 8.17, "tokens": 2048},
}
# the GDN layers' saved kernel residuals under remat=core_states, fp32 per token (chunk 128):
# states [Dk, Dv] per chunk (512 B per value head) and v_new [Dv] (512 B per value head)
CORE_STATES_BYTES_PER_TOKEN = {"0.8B": 294912, "2B": 294912, "4B": 786432, "9B": 786432}
HBM_GB = {"v6e": 32.0, "v6e1": 32.0, "v5e": 16.0}
HBM_SAFE = 0.9  # tjev.train.compile_check.SAFE_FRACTION


def step_hbm_gb(size: str, tokens: int, remat: str = "core_states") -> float:
    """Estimated device memory of a train step at ``tokens`` per microbatch (one row of the
    largest bucket, or the microbatch budget): the measured temporaries scaled linearly in
    tokens (an upper bound: a part of them, the gradient accumulators, does not grow), less
    the core_states residuals under ``remat="full"`` (the layer recompute keeps none)."""
    m = STEP_HBM[size]
    temp = m["temp"] * tokens / m["tokens"]
    if remat == "full":
        temp -= CORE_STATES_BYTES_PER_TOKEN[size] * tokens / 1e9
    return m["arguments"] + temp


LONG_BUCKETS = (1024, 2048, 4096, 8192, 16384)
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


# Decided by hand from the 0.8B sweep (docs/results.md), on top of fit's recipe: the GDN gate
# projections frozen (no adapter on in_proj_a / in_proj_b; two seeds: online −0.008 mean,
# eval better on both, against the rank-32 gate adapters)
FINAL_OVERRIDES = (f"lora.targets=[{','.join(t for t in LORA_TARGETS if t not in GATES)}]",)
LONG = (*FINAL_OVERRIDES, "train.checkpoint_every=300", "train.quick_eval_every=0")

ARMS = {
    "zclip": Arm(("optim.clip_mode=zscore",)),
    "noclip": Arm(("optim.clip_mode=none",)),
    "b32k": Arm(("train.tokens_per_step=32768",), BATCH_LR, 1),
    "b16k": Arm(("train.tokens_per_step=16384",), BATCH_LR**2, 2),
    # the rank axis 16 / 32 (the centre) / 64 / 128 at the rsLoRA 1/√r rate rule, and same-rate
    # controls: against its scaled twin, does the rate move with the rank?
    "r16": Arm(("lora.rank=16",), math.sqrt(2)),
    "r16-samelr": Arm(("lora.rank=16",)),
    "r64": Arm(("lora.rank=64",), ROOT_HALF),
    "r64-samelr": Arm(("lora.rank=64",)),
    "r128": Arm(("lora.rank=128",), 0.5),
    "r128-samelr": Arm(("lora.rank=128",)),  # an octave from r128's rate: the strongest test
    "b2-0.99": Arm(("optim.b2=0.99",)),
    # fp32 weights and activations with one-pass bf16 matmuls (a true fp32 control needs
    # 6-pass HIGHEST matmuls, ~4-6× the cost): does bf16 storage cost quality?
    "fp32-storage": Arm(("model.dtype=float32",), report=True),
    # merged LoRA (x·(W + s·AB), one matmul): +20% at 0.8B, but the bf16 W' rounds a small
    # s·AB at W's scale: measured, adopted only by hand after its paired loss
    "lora-merged": Arm(("compute.lora_impl=merged",), report=True),
    # stage 3 (after tjev-sweep1): Muon's bracket below the AdamW centre, the data-free
    # inits at lower rates (both lost at the centre rate: their adapters start large),
    # and a second seed of the two free speed-ups (paired with the stage-1 centre run of
    # seed 1: the arm's own train.seed overrides the planned one)
    "muon": Arm(MUON),
    "muon-half": Arm(MUON, 0.5),
    "pissa-half": Arm(("lora.init=pissa",), 0.5),
    "pissa-quarter": Arm(("lora.init=pissa",), 0.25),
    "eva-half": Arm(("lora.init=eva",), 0.5),
    "eva-quarter": Arm(("lora.init=eva",), 0.25),
    "lora-merged-seed1": Arm(("compute.lora_impl=merged", "train.seed=1"), report=True),
    "gdn-bf16-seed1": Arm(("compute.gdn_precision=bf16", "train.seed=1"), report=True),
    # the GDN gate projections (in_proj_a / in_proj_b: one output per value head, so rank
    # 32 is already full rank there): a rank below their width, frozen, trained directly
    "gate-r8": Arm(("lora.gate_rank=8",)),
    "gate-frozen": Arm((f"lora.targets=[{','.join(t for t in LORA_TARGETS if t not in GATES)}]",)),
    "gate-full": Arm(("lora.gate_full=true",)),
    # second seed of the two gate arms that moved (stage 4: frozen −0.013 ± 0.006, trained
    # directly +0.017 ± 0.007), paired with the stage-1 centre run of seed 1
    # the long-horizon study (1200 steps at the proxy, the final runs' config: frozen gates;
    # evals every 300 steps): the rate fitted at 300 steps checked at 4× the horizon, and
    # decoupled weight decay against the over-confidence the 4B run shows after ~900 steps
    "long-centre": Arm((*LONG,)),
    "long-lr-low": Arm((*LONG,), ROOT_HALF),
    "long-lr-high": Arm((*LONG,), 2**0.5),
    "long-wd": Arm((*LONG, "optim.weight_decay=0.05")),
    # rank 16 (the leaders' choice, JevK5 / Plumb / metask; the 300-step sweep had 16 = 32): less
    # capacity to over-fit; the rsLoRA rate rule gives it lr·√2
    "long-r16": Arm((*LONG, "lora.rank=16"), 2**0.5),
    "gate-frozen-seed1": Arm(
        (f"lora.targets=[{','.join(t for t in LORA_TARGETS if t not in GATES)}]", "train.seed=1")
    ),
    "gate-full-seed1": Arm(("lora.gate_full=true", "train.seed=1")),
    # the inits again after fix/lora-ga-init (pissa / lora-ga adapt by s·(AB − A0B0): the
    # model is the base bit for bit at init, where the bf16 residual kernel was 12× the
    # base's own noise for lora-ga; eva centres its activations), at the rates the
    # step-size analysis suggests (eva ~1/4, lora-ga ~1/3, pissa 1/2 to 1)
    "pissa-v2": Arm(("lora.init=pissa",)),
    "pissa-half-v2": Arm(("lora.init=pissa",), 0.5),
    "lora-ga-third": Arm(("lora.init=lora-ga",), 0.3),
    "eva-quarter-v2": Arm(("lora.init=eva",), 0.25),
    # LoRA initialisations (lora.init, docs/training.md), paired at the centre rate. PiSSA's
    # principal adapters start with large A and B: its papers train at a rate similar to or
    # lower than default LoRA's, so a loss here may be a rate effect (the factor stays 1.0,
    # the pair isolates the init at equal rate). EVA keeps B = 0 (the model starts as the
    # base); LoRA-GA's first step approximates full fine-tuning's.
    "pissa": Arm(("lora.init=pissa",)),
    "eva": Arm(("lora.init=eva",)),
    "lora-ga": Arm(("lora.init=lora-ga",)),
    # one bf16 pass on the delta rule's matmuls instead of bf16_3x: faster if it costs
    # nothing (MaxText saw broken gate gradients at 8k tokens; not reproduced at 4k)
    "gdn-bf16": Arm(("compute.gdn_precision=bf16",), report=True),
}
# planned only when named (--arms): the follow-ups of a first sweep
LATER_ARMS = frozenset({"muon", "muon-half", "pissa-half", "pissa-quarter", "eva-half",
                        "eva-quarter", "lora-merged-seed1", "gdn-bf16-seed1",
                        "gate-r8", "gate-frozen", "gate-full", "pissa-v2", "pissa-half-v2",
                        "lora-ga-third", "eva-quarter-v2", "gate-frozen-seed1",
                        "gate-full-seed1", "long-centre", "long-lr-low", "long-lr-high",
                        "long-wd", "long-r16"})  # fmt: skip
INITS = ("pissa", "eva", "lora-ga")  # adopted: the best clear win of these
INIT_ARMS = {name: ARMS[name] for name in INITS}
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
    arms: tuple[str, ...] = tuple(a for a in ARMS if a not in LATER_ARMS)
    muon: bool = True
    final_sizes: tuple[str, ...] = ("4B", "2B", "0.8B")  # longest first: the queue fills chips
    kernels: bool = True
    wandb: dict[str, str] = field(default_factory=dict)  # project, entity, mode, campaign
    # the AdamW grid rate the arms pair with (stage 2 after a stage-1 fit: its best grid rate)
    centre: float = CENTRE
    # which part of the sweep to plan: "lr" (stage 1: the AdamW and Muon brackets), "arms"
    # (stage 2: the paired arms at the centre), "all"; stage 3 re-plans with more arm seeds;
    # "reg" the regulariser arms on 4B and 2B (REG_PLAN, each size paired with its own centre)
    stage: str = "all"
    # sweep economy (docs/sweep-plan.md): no quick eval, a full eval every 100 steps, no
    # post-training, and runs whose health verdict is "kill" stop (train.health_kill)
    lean: bool = False
    # >0: the final horizon (steps at the full batch) instead of the best transfer horizon
    # (a budget without the transfer phase: docs/sweep-plan.md)
    horizon: int = 0
    # the largest training bucket: > 4096 adds the long buckets (LONG_BUCKETS), each size
    # capped by its hardware's "long" settings (the mix must be built with
    # --max-prompt-tokens at least this)
    max_bucket: int = 4096

    def __post_init__(self) -> None:
        if not any(math.isclose(self.centre, lr, rel_tol=1e-3) for lr in SWEEP_GRID):
            raise ValueError(f"centre {self.centre} is not a sweep grid rate {SWEEP_GRID}")
        if self.stage not in ("lr", "arms", "all", "reg"):
            raise ValueError(f"stage {self.stage!r}: lr, arms, all or reg")
        if self.max_bucket not in LONG_BUCKETS[2:]:
            raise ValueError(f"max_bucket {self.max_bucket}: one of {LONG_BUCKETS[2:]}")
        if self.max_bucket > 4096 and "long" not in self.hw:
            raise ValueError(f"hardware {self.hardware!r} has no long-bucket settings")

    @property
    def hw(self) -> dict[str, Any]:
        return HARDWARE[self.hardware]

    def others(self) -> list[str]:
        return [s for s in ("0.8B", "4B") if s != self.proxy]


# ------------------------------------------------------------------------------------------
# Queue lines: "<name> chips=N post=quick|full|none [after=RUN@STEP] <tjev train args>"


def long_settings(c: Campaign, size: str) -> tuple[tuple[int, ...], int, list[str]]:
    """(buckets, microbatch tokens, extra overrides) of ``size`` at ``c.max_bucket``."""
    if c.max_bucket <= 4096:
        return LONG_BUCKETS[:3], c.hw["microbatch"][size], []
    long = c.hw["long"][size]
    top = min(c.max_bucket, long["max_bucket"])
    return tuple(b for b in LONG_BUCKETS if b <= top), long["microbatch"], list(long["extra"])


def _common(c: Campaign, size: str) -> list[str]:
    buckets, microbatch, extra = long_settings(c, size)
    return [
        c.hw["preset"],
        f"qwen35-{size.lower()}",
        f"{c.mix}/mix.yaml",
        f"model.path={c.models}/Qwen3.5-{size}",
        f"train.microbatch_tokens={microbatch}",
        *c.hw["extra"].get(size, []),
        *extra,  # after the hardware's: e.g. remat=full over core_states
        *([f"train.seq_buckets=[{','.join(map(str, buckets))}]"] if c.max_bucket > 4096 else []),
        f"data.heldout={c.mix}/heldout-select.jsonl",
    ]


# the campaign's training runs (tjev-benchmarks: speed; tjev-experiments: kernels, parity)
WANDB_PROJECT = "tjev-sweep"
FINAL_PROJECT = "tjev-final"  # the final models apart from the sweep (TJEV_WANDB_FINAL_PROJECT)


def _value(overrides: list[str], key: str, default: str) -> str:
    found = [o.split("=", 1)[1] for o in overrides if o.startswith(f"{key}=")]
    return found[-1] if found else default


def _arm(name: str) -> str:
    """The arm a sweep job measures: ``grid`` / ``muon-grid`` for the rate brackets."""
    body = name.split("-", 1)[1] if "-" in name else name
    body = body.rsplit("-s", 1)[0]
    if body.startswith("lr"):
        return "grid"
    if body.startswith("muon-lr"):
        return "muon-grid"
    return body


def _project(c: Campaign, phase: str) -> str:
    """The final runs go to their own project (the models), the rest to the sweep's."""
    if phase == "final":
        return c.wandb.get(
            "final_project", os.environ.get("TJEV_WANDB_FINAL_PROJECT", FINAL_PROJECT)
        )
    return c.wandb.get("project", WANDB_PROJECT)


def _wandb(c: Campaign, name: str, size: str, chips: int, overrides: list[str]) -> list[str]:
    """W&B project, group (campaign/phase[/stage]) and tags: what the run varies, so the
    runs table filters and the parallel-coordinates view group by arm, optimizer, rate, …"""
    if not c.wandb:
        return []
    phase = name.split("-", 1)[0]
    arm = _arm(name) if phase == "sweep" else phase
    group = f"{c.wandb.get('campaign', 'tpu')}/{phase}"
    if phase == "sweep":
        group += ("/lr" if arm in ("grid", "muon-grid")
                  else "/reg" if arm.startswith("reg-") else "/arms")  # fmt: skip
    tags = [
        phase, size, f"{c.hardware}-x{chips}", f"arm:{arm}",
        f"opt:{_value(overrides, 'optim.name', 'adamw')}",
        f"lr:{_value(overrides, 'optim.lr', 'preset')}",
        f"rank:{_value(overrides, 'lora.rank', str(RANK[size]))}",
        f"init:{_value(overrides, 'lora.init', 'default')}",
        f"dtype:{'fp32' if _value(overrides, 'model.dtype', 'bfloat16') == 'float32' else 'bf16'}",
        f"seed:{_value(overrides, 'train.seed', '0')}",
    ]  # fmt: skip
    out = [
        "log.wandb=true",
        f"log.wandb_project={_project(c, phase)}",
        f"log.wandb_group={group}",
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
        *_wandb(c, name, size, chips, overrides),
    ]
    if not c.kernels:
        args += ["compute.attention=xla", "compute.gdn_impl=chunked"]
    return " ".join(fields + args)


def tail_every(steps: int) -> int:
    """The long run's checkpoint + full-eval cadence in its cooldown: ~30 over the run's
    length, a multiple of 50 (3000 steps: every 100)."""
    return max(50, steps // 30 // 50 * 50)


def horizon_jobs(c: Campaign, prefix: str, size: str, overrides: list[str], horizon: int,
                 chips: int | None = None, post: str = "quick") -> list[str]:  # fmt: skip
    """One run to ``horizon`` plus WSD cooldown branches at horizon/4 and horizon/2 (each
    branch keeps the 20% cooldown fraction). The long run keeps its branch points whatever
    keep_checkpoints prunes (train.keep_steps: a sequential queue starts the branches only
    after the long run, when an unprotected branch point was long rotated away), and
    checkpoints (with a full eval) densely in its own cooldown."""
    long = f"{prefix}-h{horizon}"
    every = horizon // 10  # the branch points 0.2·H and 0.4·H are multiples
    points = [int(0.8 * short) for short in (horizon // 4, horizon // 2)]
    out = [job(c, long, size, [*overrides, f"train.checkpoint_every={every}",
                               "train.keep_checkpoints=12",
                               f"train.keep_steps=[{','.join(map(str, points))}]",
                               f"train.checkpoint_tail_every={tail_every(horizon)}"],
               steps=horizon, chips=chips, post=post)]  # fmt: skip
    for short in (horizon // 4, horizon // 2):
        src = f"{long}@{int(0.8 * short)}"
        out.append(job(c, f"{prefix}-h{short}", size, [*overrides, f"train.branch_from=$RUNS/{src}"],
                       steps=short, chips=chips, after=src, post=post))  # fmt: skip
    return out


LEAN = ("train.quick_eval_every=0", "train.checkpoint_every=100", "train.health_kill=true")


def _sweep(c: Campaign) -> list[str]:
    steps, lines, centre = c.sweep_steps, [], c.centre
    extra, post = (list(LEAN), "none") if c.lean else ([], "quick")

    def sweep_job(name: str, overrides: list[str], run_steps: int) -> str:
        # the arm's own overrides last: an arm may set what --lean sets (e.g. its eval cadence)
        return job(c, name, c.proxy, [*extra, *overrides], steps=run_steps, post=post)

    if c.stage == "reg":
        return _reg(c)
    rates, arms = c.stage in ("lr", "all"), c.stage in ("arms", "all")
    if rates:
        for seed in range(c.seeds):  # all of one seed first: a first read after a fraction
            lines += [sweep_job(f"sweep-lr{lr:.3g}-s{seed}", [f"optim.lr={lr}", f"train.seed={seed}"],
                                steps) for lr in SWEEP_GRID]  # fmt: skip
        lines.append(sweep_job(f"sweep-lr{centre:.3g}-s{c.seeds}",
                               [f"optim.lr={centre}", f"train.seed={c.seeds}"], steps))  # fmt: skip
    for seed in range(c.arm_seeds):
        if c.muon and rates:
            lines += [sweep_job(f"sweep-muon-lr{lr:.3g}-s{seed}",
                                [*MUON, f"optim.lr={lr}", f"train.seed={seed}"], steps)
                      for lr in MUON_GRID]  # fmt: skip
        if arms and seed >= c.seeds and seed > 0:  # the arms' base run at this seed
            lines.append(sweep_job(f"sweep-lr{centre:.3g}-s{seed}",
                                   [f"optim.lr={centre}", f"train.seed={seed}"], steps))  # fmt: skip
        for name in c.arms if arms else ():
            arm = ARMS[name]
            lines.append(sweep_job(f"sweep-{name}-s{seed}",
                                   [f"optim.lr={centre * arm.lr_factor:.3g}", f"train.seed={seed}",
                                    *arm.overrides], steps * 2**arm.halvings))  # fmt: skip
    return list(dict.fromkeys(lines))  # a base run planned twice (stage "all") runs once


# ------------------------------------------------------------------------------------------
# The "reg" stage: regularisers against the over-confidence of long runs, each arm paired with
# its own size's centre run (same seed, same batches: the paired online loss). Every run is
# a final run of its size (fit's recipe at a REG_STEPS horizon, its predicted rate, plus the
# frozen gates) cut to REG_STEPS steps. The centres also track the EMA of their adapters
# (optim.ema_decay=0.999: it does not change the training, so a centre's raw and EMA evals
# are themselves the EMA pair). 4B gets the levers aimed at its own over-confidence, where it
# happens; 2B, ~2.6x faster, screens the rest.

REG_STEPS = 1500
# keep_every: the trajectory's checkpoints stay (backed up adapters-only by PROBE_SWEEP_BACKUP=1)
# for JevBench on more than the selected and the last step
REG_BASE = (*FINAL_OVERRIDES, "train.checkpoint_every=100", "train.quick_eval_every=0",
            "train.keep_checkpoints=2", "train.keep_every=300", "train.health_kill=true")  # fmt: skip
REG_CENTRE = ("optim.ema_decay=0.999",)
REG_ARMS = {
    # label smoothing toward uniform over the displayed letters (all sources; the generators
    # only: their rule labels are exact but over-trained)
    "ls0.05": Arm(("loss.label_smoothing=0.05",)),
    "ls0.1": Arm(("loss.label_smoothing=0.1",)),
    "ls0.1-gen": Arm(("loss.label_smoothing=0.1", "loss.smooth_sources=[block:generators]")),
    # KL(p_base ‖ p) over the letters, every row / the replay rows (forgetting); one extra
    # forward per microbatch (~1/3 of a step)
    "kl0.1": Arm(("loss.kl_base_weight=0.1",)),
    "kl0.1-replay": Arm(("loss.kl_base_weight=0.1", "loss.kl_sources=[block:replay]")),
    # the soft-target CE's control: train on the argmax of the soft targets (stsb,
    # go_emotions, pavlick_formality, mlma, gen_base_rate, the jev teachers' posteriors)
    "hard": Arm(("loss.hard_targets=true",)),
    # decoupled AdamW weight decay on the LoRA factors
    "wd0.05": Arm(("optim.weight_decay=0.05",)),
    "wd0.1": Arm(("optim.weight_decay=0.1",)),
    # decoupled decay is lr·wd per step: at the 2e-5 rates wd 0.1 shrinks the adapters by
    # ≤ 0.4% over 1500 steps (‖B‖ −3% on 4B), so wd 1.0 is the first value that bites
    "wd1.0": Arm(("optim.weight_decay=1.0",)),
    # measured: wd 1.0 moves ‖B‖ by −1% on 2B (Adam's ~lr steps outweigh lr·wd·w at these
    # norms); 10 decays by up to (1 − lr·wd)^steps ≈ e^-0.5 over 1500 steps at the 2B rate:
    # a decay that bites without erasing the adapter
    "wd10": Arm(("optim.weight_decay=10.0",)),
    # measured: 1 is a no-op and 10 hurts the cooldown (A −15 %, B −6 % by step 500): the middle
    "wd5": Arm(("optim.weight_decay=5.0",)),
    # the end of the schedule: a higher floor (10% of the peak), a shorter cooldown (10%)
    "floor0.1": Arm(("optim.final_lr_fraction=0.1",)),
    "cool0.1": Arm(("optim.decay_fraction=0.1",)),
    # rank 16 at fit's rank rate rule (lr/√2 per doubling: 4B 64 → 16 lr × 2, 2B 32 → 16
    # lr × √2): the factor is set per size (reg_rate)
    "rank16": Arm(("lora.rank=16",)),
    "rank32": Arm(("lora.rank=32",)),
    # rank 16 around its rank-rule rate (never tuned at r16: LoRA Without Regret finds the rate
    # nearly rank-free under α/r, our rsLoRA rule doubles it from r64), and at twice the horizon
    # (the horizon rule steps^-0.15)
    "rank16-lr0.7": Arm(("lora.rank=16",), 0.7),
    "rank16-lr1.4": Arm(("lora.rank=16",), 1.4),
    "rank16-lr0.5": Arm(("lora.rank=16",), 0.5),
    "rank16-lr0.85": Arm(("lora.rank=16",), 0.85),
    "rank16-h3000": Arm(("lora.rank=16", "train.steps=3000"), 2**-0.15),
    # repetition: r16 on the sibling mix <mix>-rep1 (the same held-out files; the core sources
    # capped at 1 epoch, fresh generator rows and more unique rows of the other sources in
    # place of their second epoch), paired with rank16 on <mix> (core sources up to 2 epochs)
    "rank16-rep1": Arm(("lora.rank=16", "@rep1")),
    "rank16-cogcalib-rep1": Arm(
        ("lora.rank=16", "loss.cogcalib_weight=0.1", "loss.cogcalib_quantile=0.3", "@rep1")
    ),
    # CogCalib's gated entropy penalty (CoECP, arXiv 2505.20903: β = 0.1 for multiple choice)
    # on the slots whose NLL is in the lowest 30% (a running quantile), at r16
    "rank16-cogcalib": Arm(
        ("lora.rank=16", "loss.cogcalib_weight=0.1", "loss.cogcalib_quantile=0.3")
    ),
    # Kev's gated teacher targets: (1 − 0.5)·gold + 0.5·teacher where a teacher puts ≥ 0.6 on
    # another letter (plumb / decider rows that carry a teacher), at r16. No mix-v6 row passes
    # the gate (decider drops its disagreeing questions): it needs a mix that keeps them
    # CogCalib strength (its term is ~0.1% of the loss at weight 0.1 on mix-v7.5 4B: the gated
    # rows' NLL threshold is ~0.005): 0.1 vs 10x, paired, without the cooldown branch
    # rep1 at a 2500-step horizon with the 5000-step rate (2^-0.15: the horizon rule), paired with
    # rank16-cogcalib's 1-epoch cooldown branch (2000 -> 2500 at that rate)
    "rank16-cogcalib-rep1-lr5k": Arm(
        ("lora.rank=16", "loss.cogcalib_weight=0.1", "loss.cogcalib_quantile=0.3", "@rep1"),
        2**-0.15,
    ),
    "rank16-cc0.1": Arm(("lora.rank=16", "loss.cogcalib_weight=0.1", "loss.cogcalib_quantile=0.3")),
    "rank16-cc1.0": Arm(("lora.rank=16", "loss.cogcalib_weight=1.0", "loss.cogcalib_quantile=0.3")),
    "rank16-teacher": Arm(("lora.rank=16", "loss.teacher_mix=0.5", "loss.teacher_gate=0.6")),
    # no LoRA on the GDN layers (in_proj_qkv / in_proj_z / out_proj; the gates stay frozen):
    # attention + MLP only, arXiv 2604.22127's advice for hybrids (its GDN arm also adapted
    # the gates and was scored on generative GSM8K; LoRA Without Regret says adapt all layers)
    "nogdn": Arm(("lora.targets=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]",)),
    # LoRA dropout on the adapter input (unmerged LoRA: 9-16% slower than merged)
    "drop0.05": Arm(("lora.dropout=0.05",)),
}
# the arms per size (--sizes picks the sizes, --arms restricts every size to those arms)
REG_PLAN = {
    "4B": ("wd0.1", "floor0.1"),
    "2B": ("ls0.1", "ls0.1-gen", "kl0.1", "hard", "drop0.05", "rank16", "cool0.1"),
}


def reg_steps(c: Campaign) -> int:
    """The reg stage's horizon: --horizon, else REG_STEPS."""
    return c.horizon or REG_STEPS


def reg_long(c: Campaign, steps: int) -> list[str]:
    """Past REG_STEPS (a final-length test): a checkpoint + full eval every 250 steps (every
    100 in the cooldown), and every one kept from half the run on (the core sources' second
    epoch), for JevBench on the late trajectory."""
    if steps <= REG_STEPS:
        return []
    return ["train.checkpoint_every=250", "train.checkpoint_tail_every=100",
            "train.keep_every=250", f"train.keep_from={steps // 2}"]  # fmt: skip


def _mix_token(c: Campaign, o: str) -> str:
    """``@NAME`` in an arm's overrides: the sibling mix ``<mix>-NAME``'s mix.yaml."""
    return f"{c.mix}-{o[1:]}/mix.yaml" if o.startswith("@") else o


def reg_recipe(c: Campaign) -> dict[str, dict]:
    """fit's recipe at the reg horizon (its rate rules: the proxy's fitted or prior rate,
    width, rank, batch and horizon), and its rank rate factor per doubling."""
    report = fit(replace(c, horizon=reg_steps(c)))
    factor = report["decisions"]["rank_lr_factor_per_doubling"]
    return {size: r | {"rank_factor": factor} for size, r in report["recipe"].items()}


def reg_rate(recipe: dict, name: str) -> float:
    """An arm's rate: the recipe's, the rank arms moved by the rank rule (and their own factor)."""
    if not name.startswith("rank"):
        return recipe["lr"] * REG_ARMS[name].lr_factor
    rank = int(name.removeprefix("rank").split("-")[0])
    rule = recipe["rank_factor"] ** math.log2(rank / recipe["rank"])
    return recipe["lr"] * rule * REG_ARMS[name].lr_factor


def _reg(c: Campaign) -> list[str]:
    """Per size of REG_PLAN in ``c.final_sizes`` (--sizes): its centre and its arms (only
    those of ``c.arms`` when some are REG_ARMS names), seed 0 (more with arm_seeds)."""
    named = [a for a in c.arms if a in REG_ARMS]
    recipes = reg_recipe(c)
    steps = reg_steps(c)
    lines = []
    # a final-length reg run (and its cooldown branch) is a proper run: post=full (the
    # selected checkpoint calibrated, JevBench, per-item dumps)
    post = "full" if steps > REG_STEPS else "none"
    for size, arms in REG_PLAN.items():
        if size not in c.final_sizes:
            continue
        recipe = recipes[size]
        common = [o for o in recipe["overrides"] if not o.startswith("optim.lr=")]
        tag = size.lower()
        for seed in range(c.arm_seeds):
            base = [*common, *REG_BASE, *reg_long(c, steps), f"train.seed={seed}"]
            if "nocentre" not in c.arms:  # --arms …,nocentre: the arms only (a centre already ran)
                lines.append(job(c, f"sweep-reg-{tag}-centre-s{seed}", size,
                                 [*base, f"optim.lr={recipe['lr']:.3g}", *REG_CENTRE],
                                 steps=steps, post=post))  # fmt: skip
            for name in named or arms:
                run = f"sweep-reg-{tag}-{name}-s{seed}"
                over = [*base, f"optim.lr={reg_rate(recipe, name):.3g}",
                        *(_mix_token(c, o) for o in REG_ARMS[name].overrides)]  # fmt: skip
                # a final-length rank16 (and rank16-cogcalib) also gets the 1-epoch run of its
                # own stable phase: a WSD cooldown branch to steps/2 off it (the core sources'
                # first epoch)
                branched = name in ("rank16", "rank16-cogcalib") and steps > REG_STEPS
                point = int(0.8 * (steps // 2)) if branched else 0
                lines.append(job(c, run, size, [*over, *([f"train.keep_steps=[{point}]"] if point else [])],
                                 steps=steps, post=post))  # fmt: skip
                if point:
                    lines.append(job(c, f"sweep-reg-{tag}-{name}-h{steps // 2}-s{seed}", size,
                                     [*over, f"train.branch_from=$RUNS/{run}@{point}",
                                      # a branch starts at its parent's loss and hovers there:
                                      # the health rule loss_above_start killed one (2043)
                                      "train.health_kill=false"],
                                     steps=steps // 2, after=f"{run}@{point}", post=post))  # fmt: skip
    return lines


def reg_effects(records: list[dict], arm_seeds: int = 1) -> dict[str, dict]:
    """Per size and reg arm (``"4B/wd0.1"``): the paired online difference and the selection
    score difference against that size's reg centre of the same seed (measured; adopted by
    hand)."""
    out = {}
    for size in SIZES:
        tag = size.lower()
        for name in REG_ARMS:
            pairs, scores = [], []
            for seed in range(arm_seeds):
                base = _named(records, f"sweep-reg-{tag}-centre-s{seed}")
                run = _named(records, f"sweep-reg-{tag}-{name}-s{seed}")
                if base is None or run is None:
                    continue
                scores.append(run["score"] - base["score"])
                if (d := online_delta(run, base)) is not None:
                    pairs.append(d)
            if scores:
                online, score = combine(pairs), float(np.mean(scores))
                out[f"{size}/{name}"] = {"online": online, "score": score,
                                         "better": adopted(online, score)}  # fmt: skip
    return out


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
                                         [*recipe[size]["overrides"], *FINAL_OVERRIDES],
                                         recipe[size]["horizon"])]  # fmt: skip
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


def killed(run: Path) -> bool:
    """Stopped by its health rules (train.health_kill): a divergent run, never a data point
    of a rate curve or an arm (its online tail does not exist)."""
    path = run / "health.json"
    return path.exists() and json.loads(path.read_text()).get("status") == "killed"


def load_runs(runs: Path) -> list[dict]:
    """Finished runs (``queue.done``) with their parameters and selection scores."""
    out = []
    for run in sorted(p for p in runs.iterdir() if (p / "queue.done").exists()):
        if not (run / "selection.json").exists() or killed(run):
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


def arm_effects(records: list[dict], arm_seeds: int = 1,
                centre: float = CENTRE) -> dict[str, dict]:  # fmt: skip
    """Per sweep arm, the paired online difference (seeds combined) and the eval score
    difference against the sweep run at the centre rate with the same seed."""
    out = {}
    for name, arm in ARMS.items():
        pairs, scores = [], []
        for seed in range(arm_seeds):
            base = _named(records, f"sweep-lr{centre:.3g}-s{seed}")
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


RANKS = (16, 64, 128)  # the rank arms (32: the centre run)


def rank_rule(records: list[dict], effects: dict, arm_seeds: int) -> tuple[int, float]:
    """(the proxy's rank, the rate factor per rank doubling).

    The rank: the adopted rank arm (scaled or same-rate) with the largest paired win, else
    32. The rate: each rN-samelr against rN (same seed, same batches) asks whether the
    optimum moves as 1/√r; one adopted same-rate win keeps the rate across ranks."""
    best, best_d = 32, 0.0
    for rank in RANKS:
        for name in (f"r{rank}", f"r{rank}-samelr"):
            e = effects.get(name, {})
            if e.get("better") and e["online"][0] < best_d:
                best, best_d = rank, e["online"][0]
    keep_rate = False
    for rank in RANKS:
        pairs, scores = [], []
        for seed in range(arm_seeds):
            same = _named(records, f"sweep-r{rank}-samelr-s{seed}")
            scaled = _named(records, f"sweep-r{rank}-s{seed}")
            if same and scaled:
                scores.append(same["score"] - scaled["score"])
                if (d := online_delta(same, scaled)) is not None:
                    pairs.append(d)
        keep_rate |= bool(scores) and adopted(combine(pairs), float(np.mean(scores)))
    return best, 1.0 if keep_rate else ROOT_HALF


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
    effects = arm_effects(records, c.arm_seeds, c.centre)
    clip = next((list(ARMS[a].overrides) for a in ("zclip", "noclip")
                 if effects.get(a, {}).get("better")), [])  # fmt: skip
    batch = min((a for a in ("b32k", "b16k") if effects.get(a, {}).get("better")),
                key=lambda a: effects[a]["online"][0], default=None)  # fmt: skip
    halvings = ARMS[batch].halvings if batch else 0
    init = min((a for a in INITS if effects.get(a, {}).get("better")),
               key=lambda a: effects[a]["online"][0], default=None)  # fmt: skip
    tokens = TOKENS_PER_STEP // 2**halvings
    proxy_rank, rank_scale = rank_rule(records, effects, c.arm_seeds)
    optimizer = list(MUON) if use_muon["use"] else []
    if effects.get("b2-0.99", {}).get("better"):
        optimizer.append("optim.b2=0.99")
    init_args = [f"lora.init={init}"] if init else []
    # horizon: the best of the proxy's hi runs (default 1200), γ from the hi/lo pairs
    horizon = {r["steps"]: r["score"] for r in records
               if r["name"].startswith(f"transfer-{c.proxy.lower()}-hi-h")}  # fmt: skip
    best_h = c.horizon or (min(horizon, key=horizon.__getitem__) if horizon else 1200)
    gamma = horizon_gamma(records, c.proxy, anchor.get("curvature") if anchor else None)
    final_h = best_h * 2**halvings
    horizon_factor = (best_h / c.sweep_steps) ** -gamma["gamma"]
    recipe = {}
    for size, dims in SIZES.items():
        # small sizes take the proxy's rank; 4B / 9B never go below their 64
        rank = proxy_rank if RANK[size] <= RANK[c.proxy] else max(proxy_rank, RANK[size])
        rows = [r for r in records if r["name"].startswith(f"transfer-{size.lower()}-lr")]
        measured = lr_curve([(r["lr"], r["score"], r["seed"]) for r in rows]) if rows else None
        if measured and measured["bracketed"]:
            lr = measured["opt"]  # already at this size's rank and batch
        else:
            lr = proxy_lr * (dims["hidden_size"] / SIZES[c.proxy]["hidden_size"]) ** -P_WIDTH
            # the sweep's rates are at rank 32 (the centre run's rank)
            lr *= rank_scale ** math.log2(rank / RANK[c.proxy])
            lr *= BATCH_LR**halvings
        overrides = [*optimizer, *clip, *init_args, f"lora.rank={rank}",
                     f"train.tokens_per_step={tokens}"]  # fmt: skip
        final = float(f"{lr * horizon_factor:.2g}")
        recipe[size] = {"lr": final, "sweep_lr": float(f"{lr:.3g}"), "rank": rank,
                        "horizon": final_h, "overrides": [*overrides, f"optim.lr={final}"]}  # fmt: skip
    return {
        "runs": len(records),
        "killed": sorted(p.name for p in c.runs.iterdir() if killed(p)) if c.runs.exists() else [],
        "curves": {"adamw": adamw, "muon": muon},
        "effects": effects,
        "reg_effects": reg_effects(records, c.arm_seeds),
        "muon": use_muon,
        "horizon": gamma,
        "decisions": {
            "optimizer": " ".join(o for o in optimizer if not o.startswith("lora.")) or "adamw",
            "proxy_lr": proxy_lr,
            "clipping": clip or "global",
            "lora_init": init or "default",
            "tokens_per_step": tokens,
            "rank_small_sizes": proxy_rank,
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
    if size in c.hw.get("tokens", {}):
        return c.hw["tokens"][size]
    per_token = train_flops(model_config(size), np.asarray([400.0])) / 400  # ~400-token items
    return c.hw["mfu"][size] * c.hw["peak"] / per_token


# LoRA's share of the device time at rank 32 (xprof: 0.8B 13%, 9B 16%; 2B / 4B interpolated).
# It scales ~linearly with the rank (an upper bound: small-rank matmuls are latency-bound).
LORA_SHARE = {"0.8B": 0.13, "2B": 0.14, "4B": 0.15, "9B": 0.16}


def rank_slowdown(size: str, rank: int) -> float:
    """Step time at ``rank`` over the step time at rank 32 (the benched rank)."""
    return 1.0 + LORA_SHARE[size] * (rank / 32 - 1)


EVAL_ITEMS = 3400  # validation (64 per source) + held-out generators, ~400 tokens each
KL_SLOWDOWN = 4 / 3  # + one forward of the base per microbatch (forward ≈ ⅓ of a train step)
MERGED_GAIN = {"0.8B": 1.26, "2B": 1.16, "4B": 1.09}  # merged LoRA's speed-up (tjev-bench4)


def full_evals(fields: dict[str, str], steps: int) -> int:
    """Full evals of a job's ``steps`` (after a branch point: its own), the cooldown's tail
    cadence included (tjev.train.loop.Cadence; approximate for branches)."""
    every = int(fields.get("train.full_eval_every", 0)
                or fields.get("train.checkpoint_every", CHECKPOINT_EVERY))  # fmt: skip
    tail = int(fields.get("train.full_eval_tail_every", 0)
               or fields.get("train.checkpoint_tail_every", 0))  # fmt: skip
    if every <= 0:
        return 0
    total = int(fields["train.steps"])
    frac = float(fields.get("train.tail_fraction", 0) or fields.get("optim.decay_fraction", 0.2))
    start = total - round(frac * total)
    first = total - steps
    return sum(1 for k in range(first + 1, total + 1)
               if k == total or k % (tail if tail and k > start else every) == 0)  # fmt: skip


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
            slow *= rank_slowdown(size, int(fields.get("lora.rank", RANK[size])))
            if float(fields.get("loss.kl_base_weight", 0)):
                slow *= KL_SLOWDOWN  # the base's forward per microbatch
            if float(fields.get("lora.dropout", 0)) and fields.get("compute.lora_impl") == "merged":
                slow *= MERGED_GAIN.get(size, 1.0)  # dropout runs the unmerged LoRA
            n_evals = full_evals(fields, steps)
            if float(fields.get("optim.ema_decay", 0)):
                n_evals *= 2  # the EMA adapters are evaluated too
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
        wandb.setdefault("project", WANDB_PROJECT)
    return Campaign(runs=runs, mix=mix, models=models, wandb=wandb, **kw)
