"""The run configuration: frozen dataclasses, one per section, with the recommended defaults.

The dataclass defaults are the single source of truth. Preset files (``configs/``) and
``key=value`` overrides only change what they name. Values are coerced and checked against
the annotations, and unknown keys are errors (see :mod:`tjev.config.loader`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

from tjev.utils import fingerprint

LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class ModelSpec:
    path: str = ""  # local Hugging Face snapshot directory of a Qwen3.5 model
    dtype: Literal["bfloat16", "float32"] = "bfloat16"  # frozen base and activations


@dataclass(frozen=True)
class LoRASpec:
    """Low-rank adapters on every projection of the decoder (docs/training.md)."""

    rank: int = 32
    alpha: float = 32.0
    rslora: bool = True  # scale alpha/sqrt(rank) (rank-stable), else alpha/rank
    targets: tuple[str, ...] = LORA_TARGETS

    @property
    def scale(self) -> float:
        return self.alpha / (self.rank**0.5 if self.rslora else self.rank)


Remat = Literal["none", "full", "core", "minimal"]


@dataclass(frozen=True)
class ComputeSpec:
    """How the forward and backward passes run; never what they compute."""

    # Per decoder layer: "full" recomputes everything; "core" keeps the GDN core and
    # attention outputs; "minimal" also keeps the projections and MLP activations.
    remat: Remat = "full"
    # xla: query-blocked jax.nn.dot_product_attention; splash: Pallas TPU splash kernels
    attention: Literal["xla", "splash"] = "xla"
    attention_block: int = 512  # xla: query block for O(block·T) memory (0: one block)
    # chunked: the XLA chunked delta rule (the reference); recurrent: token by token (tests);
    # pallas_tpu: the fused Pallas TPU forward; pallas_tpu_split: the chunk-local terms in
    # XLA around the Pallas recurrence (the fallback of the fused kernel)
    gdn_impl: Literal["chunked", "recurrent", "pallas_tpu", "pallas_tpu_split"] = "chunked"
    gdn_chunk: int = 64
    # Delta-rule matmuls: "high" = bf16_3x on TPU / TF32 on GPU (MaxText's choice for the
    # gate gradients), "highest" = fp32 (the parity reference), "bf16" = one bf16 pass.
    gdn_precision: Literal["highest", "high", "bf16"] = "high"

    def __post_init__(self) -> None:
        if self.gdn_impl.startswith("pallas_tpu") and self.gdn_chunk & (self.gdn_chunk - 1):
            raise ValueError("gdn_impl=pallas_tpu needs a power-of-two gdn_chunk")

    @property
    def uses_tpu_kernels(self) -> bool:
        return self.attention == "splash" or self.gdn_impl.startswith("pallas_tpu")


@dataclass(frozen=True)
class MeshSpec:
    """Devices as a (data, fsdp) mesh; the batch is split over both axes."""

    data: int = -1  # -1: all devices not used by fsdp
    fsdp: int = 1  # >1 shards the frozen base (chips whose HBM cannot hold a replica)


@dataclass(frozen=True)
class OptimSpec:
    """AdamW (default) or Muon on the LoRA factors, a WSD schedule and gradient clipping."""

    name: Literal["adamw", "muon"] = "adamw"
    lr: float = 4e-5
    warmup_steps: int = 50  # independent of the horizon: cooldown branches match their parent
    decay_fraction: float = 0.2  # WSD: the last fraction of steps decays
    final_lr_fraction: float = 0.02
    decay_shape: Literal["sqrt", "linear"] = "sqrt"  # sqrt: 1 - sqrt(t) (Hägele et al. 2024)
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    # "global": clip_by_global_norm(clip_norm); "zscore": clip outliers only, above
    # mean + clip_z·std of an EMA (clip_ema) of past norms (ZClip); "none"
    clip_mode: Literal["global", "zscore", "none"] = "global"
    clip_norm: float = 1.0
    clip_z: float = 2.5
    clip_ema: float = 0.97
    # Muon: Polar Express Newton-Schulz (8 steps), Nesterov momentum, and the update RMS of
    # AdamW (0.2), so AdamW learning rates and schedules carry over (docs/research/muon.md)
    muon_beta: float = 0.95
    muon_rms: float = 0.2


@dataclass(frozen=True)
class TrainSpec:
    steps: int = 1200
    seed: int = 0  # LoRA init and the data stream
    tokens_per_step: int = 65536  # global batch: microbatch tokens × devices × accumulation
    seq_buckets: tuple[int, ...] = (1024, 2048, 4096)
    microbatch_tokens: int = 8192  # per device per microbatch
    max_segments: int = 32  # answer slots per packed row
    packing_bins: int = 16  # open microbatches per bucket (first-fit over the bins)
    max_labels: int = 26  # answer letters A..Z
    brier_weight: float = 0.1
    # "<run dir>@<step>": start from that run's checkpoint (adapters, optimizer, data stream),
    # e.g. a WSD cooldown branch off a longer run's stable phase (Hägele et al. 2024). The
    # source must be this run's training (its identity) with the same schedule up to <step>.
    branch_from: str = ""
    log_every: int = 1  # metrics are fetched one step late: logging every step is free
    quick_eval_every: int = 10  # a small eval for the curves only (0: off)
    # Every checkpoint_every steps (and the last step): a checkpoint *and* the full eval
    # (validation + held-out) that drives selection, so every evaluated step can be selected
    checkpoint_every: int = 50
    # >0: also checkpoint when this many seconds passed since the last save (preemptible and
    # time-limited VMs); SIGTERM always saves and stops
    checkpoint_secs: float = 900.0
    keep_checkpoints: int = 3  # latest ones, plus the selected (best) step
    profile_start: int = 0  # >0: a jax.profiler trace of profile_steps steps from this one
    profile_steps: int = 5


@dataclass(frozen=True)
class DataSpec:
    train: tuple[str, ...] = ()  # JSONL files of decision items (a built mix: mix.yaml)
    validation: str = ""
    validation_per_source: int = 64  # >0: the first N validation rows of each source
    heldout: str = ""  # held-out generators: a second eval set, part of the selection score
    quick_validation_per_source: int = 24  # the quick eval set
    # Grain worker processes rendering and tokenizing segments (0: in the trainer process).
    # They cannot change the data, so they are not part of a run's identity.
    workers: int = 0
    worker_buffer: int = 32
    mixture: dict[str, float] = field(default_factory=dict)  # source -> weight (else n^alpha)
    mixture_alpha: float = 0.5


@dataclass(frozen=True)
class LogSpec:
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "tjev"
    wandb_entity: str = ""  # "": the API key's default entity
    wandb_group: str = ""  # e.g. the campaign and phase: "tpu-20261001/sweep"
    wandb_tags: tuple[str, ...] = ()
    # offline: log to <run>/wandb and `wandb sync` later (no key or no network on the host)
    wandb_mode: Literal["online", "offline"] = "online"


@dataclass(frozen=True)
class RunConfig:
    name: str = "run"
    output: str = "runs"
    model: ModelSpec = field(default_factory=ModelSpec)
    lora: LoRASpec = field(default_factory=LoRASpec)
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    mesh: MeshSpec = field(default_factory=MeshSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    data: DataSpec = field(default_factory=DataSpec)
    log: LogSpec = field(default_factory=LogSpec)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())
