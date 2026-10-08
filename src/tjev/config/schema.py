"""The run configuration: frozen dataclasses, one per section, with the recommended defaults.

The dataclass defaults are the single source of truth. Preset files (``configs/``) and
``key=value`` overrides only change what they name. Values are coerced and checked against
the annotations, and unknown keys are errors (see :mod:`tjev.config.loader`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, get_args

from tjev.utils import fingerprint

GATES = ("in_proj_a", "in_proj_b")  # the GDN gate projections (one output per value head)
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
    # Adapter initialisation (tjev.model.lora_init, docs/training.md "LoRA initialisations"):
    # "default" A ~ N(0, 1/fan_in), B = 0; "pissa" the top-rank SVD of W in s·AB, the
    # frozen kernel the residual W - s·AB (Meng et al., arXiv 2404.02948); "eva" A = the
    # top right singular vectors of each projection's input activations, B = 0 (Paischer
    # et al., arXiv 2410.07170; fixed rank); "lora-ga" A, B from the SVD of the full-weight
    # gradient, the kernel W - s·AB (Wang et al., arXiv 2407.05000).
    init: Literal["default", "pissa", "eva", "lora-ga"] = "default"
    init_niter: int = 4  # randomized SVD power iterations (pissa, eva, lora-ga)
    init_batches: int = 8  # microbatches of training data (eva, lora-ga)
    init_gamma: float = 16.0  # lora-ga's stable scale: A, B × d_out^(1/4) / √γ
    # The GDN gate projections in_proj_a / in_proj_b map the hidden state to one value per
    # value head (16 at 0.8B / 2B, 32 at 4B / 9B): any rank ≥ that is already full rank.
    # gate_rank > 0: their adapters' own rank (and rsLoRA scale); gate_full: no adapter,
    # their kernels train directly (fp32, ~16-130k parameters per layer). Leaving them out
    # of ``targets`` freezes them.
    gate_rank: int = 0
    gate_full: bool = False
    # dropout on the adapter's input (y = xW + s·drop(x)·AB; one mask per projection group
    # sharing an input), training steps only: eval, calibration and export never drop. > 0
    # needs the unmerged adapter (compute.lora_impl=merged falls back to it, see Qwen35)
    dropout: float = 0.0

    @property
    def scale(self) -> float:
        return self.alpha / (self.rank**0.5 if self.rslora else self.rank)

    def rank_of(self, name: str) -> int:
        """The adapter rank of projection ``name`` (0: not adapted)."""
        if name not in self.targets or (self.gate_full and name in GATES):
            return 0
        return self.gate_rank if self.gate_rank and name in GATES else self.rank

    def scale_of(self, name: str) -> float:
        r = self.rank_of(name)
        return self.alpha / (r**0.5 if self.rslora else r) if r else 1.0

    def full(self, name: str) -> bool:
        """Whether projection ``name`` trains its kernel directly."""
        return self.gate_full and name in GATES and name in self.targets


Remat = Literal["none", "full", "core", "core_states", "minimal", "minimal_states"]


@dataclass(frozen=True)
class ComputeSpec:
    """How the forward and backward passes run; never what they compute."""

    # Per decoder layer: "full" recomputes everything; "core" keeps the GDN core and
    # attention outputs; "core_states" also the Pallas GDN kernel's residuals (its forward
    # kernel then runs once, not 2-3 times); "minimal" keeps the core outputs, projections
    # and MLP activations; "minimal_states" both: minimal's names and the GDN kernel's
    # residuals (the least recompute, the most memory).
    remat: Remat = "full"
    # Per sequence bucket, a policy over ``remat`` (keys: bucket lengths, e.g.
    # {"1024": "minimal_states", "2048": "minimal_states"}; a bucket not named keeps
    # ``remat``). Each bucket's train step is its own compiled program, so each can trade
    # recompute for memory on its own (docs/training.md, "Remat per bucket").
    remat_by_bucket: dict[str, Remat] = field(default_factory=dict)
    # xla: query-blocked jax.nn.dot_product_attention; splash: Pallas TPU splash kernels
    attention: Literal["xla", "splash"] = "xla"
    attention_block: int = 512  # xla: query block for O(block·T) memory (0: one block)
    # splash: q/kv tile rows (fewer when a row is not a multiple), the kv compute sub-block
    # (0: the whole tile), and one fused dkv+dq backward kernel instead of dq + dkv
    attention_splash_block: int = 512
    attention_splash_compute: int = 0
    attention_fused_bwd: bool = False
    # chunked: the XLA chunked delta rule (the reference); recurrent: token by token (tests);
    # pallas_tpu: the fused Pallas TPU forward; pallas_tpu_split: the chunk-local terms in
    # XLA around the Pallas recurrence (the fallback of the fused kernel)
    # pallas_tpu_v2: fused Pallas forward and backward (heads batched, closed-form inverse
    # and gate gradients; saves (I+L)⁻¹ and the per-chunk states)
    # pallas_tpu_mega1: v2 with the gates, l2norm(q, k), q scale and GQA in the kernels (the
    # conv + SiLU in XLA); pallas_tpu_mega2: also the conv + SiLU (the core from the raw
    # projections, no post-conv qkv in HBM)
    gdn_impl: Literal[
        "chunked", "recurrent", "pallas_tpu", "pallas_tpu_split", "pallas_tpu_v2",
        "pallas_tpu_mega1", "pallas_tpu_mega2",
    ] = "chunked"  # fmt: skip
    gdn_chunk: int = 64
    # Delta-rule matmuls: "high" = bf16_3x on TPU / TF32 on GPU (MaxText's choice for the
    # gate gradients), "highest" = fp32 (the parity reference), "bf16" = one bf16 pass,
    # "mixed" = bf16_3x intra-chunk, one bf16 pass on the state path (MaxText's split).
    gdn_precision: Literal["highest", "high", "bf16", "mixed"] = "high"
    # How the XLA side of the delta rule solves the chunk's UT transform (I+L)⁻¹[vβ, kβe^γ]:
    # "default" = each implementation's own (chunked: solve; the Pallas paths' XLA terms,
    # i.e. the split forward and every Pallas backward: doubling); "solve" = XLA's triangular
    # solve; "doubling" = the dense inverse by matmuls; "blocked" = blocked forward
    # substitution (16-row blocks). The fused forward kernel always uses doubling.
    # pallas_tpu_v2 / mega: the forward kernel's inverse, "default"/"doubling" = log2(C)
    # doubling levels, "blocked" = exact 16×16 diagonal blocks by substitution, then
    # log2(C/16) doubling levels ("solve" is refused).
    gdn_inverse: Literal["default", "solve", "doubling", "blocked"] = "default"
    # pallas_tpu_v2 / mega: value heads per grid step (0: 8; the largest multiple of the GQA
    # ratio dividing H up to this), and the kernels' VMEM limit as a fraction of the chip's
    # VMEM (0: 90% of the libtpu --xla_tpu_scoped_vmem_limit_kib; never above that flag)
    gdn_heads_per_step: int = 0
    gdn_vmem_fraction: float = 0.0
    # LoRA: "unmerged" = y = xW + s·(xA)B; "merged" = y = x(W + s·AB) with W' formed per
    # call in the compute dtype, under a custom VJP (dx = dy W'ᵀ, dA and dB by the low-rank
    # formulas, never the full dW)
    lora_impl: Literal["unmerged", "merged"] = "unmerged"
    # The GDN output's gated RMSNorm: "heads" = over a [B,T,H,Dv] view; "flat" = on the
    # heads side by side [B,T,H·Dv], the per-head sums as fp32 matmuls with a 0/1 head
    # matrix (no [.., H, Dv] relayout on TPU; GatedRMSNorm.flat)
    gdn_norm: Literal["heads", "flat"] = "heads"
    # Zero the padding rows of the GDN input before its in-projections (redundant: padding
    # reaches nothing valid; false drops the select from the GEMMs' prologue)
    gdn_input_mask: bool = True
    # Projections on [B·T, D] rows (reshaped back after): every bucket's GEMMs the same 2-D
    # shapes (the 2048-token bucket's [1, 2048] rows are already [2048] to XLA)
    flat_rows: bool = False
    # Projections (LoRA target names) whose frozen kernel is stored [out, in], the matmuls
    # reading its transpose: XLA:TPU copies some stacked kernels to that layout once per
    # step (merged LoRA: in_proj_z, out_proj, o_proj at 9B, 1.75 GiB). lora.init default
    kernel_out_in: tuple[str, ...] = ()
    # Training's output head: "full" = every row; "letters" = an untied head (9B) keeps only
    # the answer letters' rows (the readout reads nothing else; Qwen35.head_rows_of). The
    # full-vocabulary logits (parity tools) need "full"
    head_rows: Literal["full", "letters"] = "full"
    # libtpu flags (LIBTPU_INIT_ARGS) set before JAX starts, unless the environment already
    # sets LIBTPU_INIT_ARGS (cloud/tpu_env.sh exports the subset libtpu accepted on the host:
    # an unknown flag aborts libtpu). They change speed, never results: not in the identity.
    libtpu_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.gdn_impl.startswith("pallas_tpu") and self.gdn_chunk & (self.gdn_chunk - 1):
            raise ValueError("gdn_impl=pallas_tpu needs a power-of-two gdn_chunk")
        by_bucket = {}
        for key, policy in self.remat_by_bucket.items():
            if not str(key).isdigit() or int(key) <= 0:
                raise ValueError(f"compute.remat_by_bucket: {key!r} is not a bucket length")
            if policy not in get_args(Remat):
                raise ValueError(f"compute.remat_by_bucket: {policy!r} is not one of "
                                 f"{get_args(Remat)}")  # fmt: skip
            by_bucket[str(int(key))] = policy
        object.__setattr__(self, "remat_by_bucket", by_bucket)  # keys as the loader's: "1024"

    def __hash__(self) -> int:  # the model's static attributes (jit static args) hash it
        return hash(tuple(
            tuple(sorted(v.items())) if isinstance(v, dict) else v
            for v in (getattr(self, f.name) for f in dataclasses.fields(self))
        ))  # fmt: skip

    def remat_at(self, length: int) -> str:
        """The remat policy of the train step of bucket ``length``."""
        return self.remat_by_bucket.get(str(length), self.remat)

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
    # > 0: an fp32 exponential moving average of the adapters, in the optimizer state (so
    # checkpointed and restored with it): ema ← d·ema + (1 − d)·params after each accepted
    # update, d = min(ema_decay, (1 + t)/(10 + t)) (the usual warmup: early steps would
    # otherwise keep the init's weight). The full evals score it too (eval_ema/*)
    ema_decay: float = 0.0


@dataclass(frozen=True)
class LossSpec:
    """Regularisers of the objective (tjev.train.objective); all off by default.

    Source selectors (``*_sources``): fnmatch patterns on the item's source name, or
    ``block:<name>`` for a mix block (tjev.data.mix.block_of_source: jev, generators,
    replay, analysis, writing); empty: every source."""

    # target ← (1 − ε)·target + ε·uniform over the displayed letters, in the CE and Brier
    # terms (the logged nll / brier / accuracy stay against the data's target)
    label_smoothing: float = 0.0
    smooth_sources: tuple[str, ...] = ()
    # + w·KL(p_base ‖ p) over the displayed letters, p_base the same model with the
    # adapters scaled to 0 (the base; one extra forward per microbatch, no backward)
    kl_base_weight: float = 0.0
    kl_sources: tuple[str, ...] = ()
    # train on the argmax of a soft target (the control of the soft-target CE + Brier)
    hard_targets: bool = False
    # CogCalib's gated confidence penalty (CoECP, arXiv 2505.20903): + w·(−H(p)) over the
    # displayed letters on the slots the model already "knows", NLL ≤ t (the NLL against the
    # data's target). t is cogcalib_threshold (nats), or with cogcalib_quantile q > 0 an EMA
    # (decay cogcalib_ema) of each step's q-quantile of its slot NLLs, held in the optimizer
    # state (no extra pass; the first step gates nothing)
    cogcalib_weight: float = 0.0
    cogcalib_threshold: float = 0.0
    cogcalib_quantile: float = 0.0
    cogcalib_ema: float = 0.9
    # Gated teacher soft targets: on a row whose teacher (Item.teacher: plumb teacher_probs,
    # decider teacher_p) puts ≥ teacher_gate on a non-gold letter, the training target is
    # (1 − λ)·t + λ·teacher, λ = teacher_mix (t one-hot: (1 − λ)·one-hot(gold) + λ·teacher);
    # every other row keeps t
    teacher_mix: float = 0.0
    teacher_gate: float = 0.6

    def __post_init__(self) -> None:
        if self.cogcalib_weight < 0:
            raise ValueError("loss.cogcalib_weight must be ≥ 0")
        if not 0 <= self.cogcalib_quantile < 1:
            raise ValueError("loss.cogcalib_quantile must be in [0, 1)")
        if not 0 <= self.cogcalib_ema < 1:
            raise ValueError("loss.cogcalib_ema must be in [0, 1)")
        if self.cogcalib_weight and not (self.cogcalib_quantile or self.cogcalib_threshold > 0):
            raise ValueError(
                "loss.cogcalib_weight needs cogcalib_threshold > 0 or cogcalib_quantile"
            )
        if not 0 <= self.teacher_mix <= 1:
            raise ValueError("loss.teacher_mix must be in [0, 1]")
        if not 0 < self.teacher_gate <= 1:
            raise ValueError("loss.teacher_gate must be in (0, 1]")

    @property
    def cogcalib_adaptive(self) -> bool:
        """The threshold is the running quantile (optimizer state), not a constant."""
        return bool(self.cogcalib_weight and self.cogcalib_quantile)


# Bucket lengths are multiples of the TPU lane width: the GDN chunk (64 / 128) and the splash
# blocks tile them without padding.
BUCKET_MULTIPLE = 128


@dataclass(frozen=True)
class TrainSpec:
    steps: int = 1200
    seed: int = 0  # LoRA init and the data stream
    tokens_per_step: int = 65536  # global batch: microbatch tokens × devices × accumulation
    # Row lengths, ascending; a segment goes to the smallest that holds it (longer ones are
    # dropped). Long buckets (8192, 16384) are for long documents (docs/training.md).
    seq_buckets: tuple[int, ...] = (1024, 2048, 4096)
    # Tokens per device per microbatch: microbatch_tokens // T rows of a bucket T. A bucket
    # longer than this takes one row per device, so a microbatch holds
    # max(microbatch_tokens, T) tokens: the largest bucket sets the activation memory.
    microbatch_tokens: int = 8192
    max_segments: int = 32  # answer slots per packed row
    packing_bins: int = 16  # open microbatches per bucket (first-fit over the bins)
    # Where a segment goes (tjev.data.pack.BucketPacker, docs/data.md "Packing"): "bucket"
    # the smallest bucket that holds it, first-fit (a segment over half a bucket is alone in
    # its row); "best_fit" the fullest open row of any bucket that holds it, so short
    # segments fill the tails of the long rows
    packing: Literal["bucket", "best_fit"] = "bucket"
    max_labels: int = 26  # answer letters A..Z
    brier_weight: float = 0.1
    # "<run dir>@<step>": start from that run's checkpoint (adapters, optimizer, data stream),
    # e.g. a WSD cooldown branch off a longer run's stable phase (Hägele et al. 2024). The
    # source must be this run's training (its identity) with the same schedule up to <step>.
    branch_from: str = ""
    log_every: int = 1  # metrics are fetched one step late: logging every step is free
    quick_eval_every: int = 10  # a small eval for the curves only (0: off)
    # Every checkpoint_every steps (and the last step): a checkpoint *and* the full eval
    # (validation + held-out) that drives selection, so every evaluated step can be selected;
    # <= 0: neither (a benchmark run: tjev.campaign.journey)
    checkpoint_every: int = 50
    # The tail (the last ``tail_fraction`` of the steps; 0: the WSD cooldown,
    # optim.decay_fraction): checkpoints and quick / full evals at their own, usually denser
    # cadence there (0: the main cadence). Every full eval step is also a checkpoint (it can
    # be selected), and the last step and a STOP step always get the full eval.
    tail_fraction: float = 0.0
    checkpoint_tail_every: int = 0
    full_eval_every: int = 0  # 0: at every checkpoint_every step (and the last)
    full_eval_tail_every: int = 0
    quick_eval_tail_every: int = 0
    # Steps never pruned by keep_checkpoints: e.g. the branch points of planned cooldown
    # branches (tjev.campaign.plan sets them)
    keep_steps: tuple[int, ...] = ()
    # and every keep_every-th step from keep_from on (0: off), e.g. dense checkpoints once
    # the mix's first epoch is done, for evaluating late checkpoints (JevBench) after the run;
    # keep_every should be a multiple of checkpoint_every (a kept step is a planned save)
    keep_every: int = 0
    keep_from: int = 0
    # every checkpoint of the tail (the cooldown) is kept too: a cooldown branch's decay, or
    # the long run's, evaluated step by step after the run
    keep_tail: bool = False
    # The selection score: "score" the mean of the calibrated validation (odd slots) and
    # held-out NLLs; "lang_mean" / "lang_max" the mean / max over languages of the
    # calibrated NLL per language (validation odd slots and held-out together)
    select_on: Literal["score", "lang_mean", "lang_max"] = "score"
    # select on (and use downstream) the EMA adapters (optim.ema_decay > 0)
    select_ema: bool = False
    # >0: also checkpoint when this many seconds passed since the last save (preemptible and
    # time-limited VMs); SIGTERM always saves and stops
    checkpoint_secs: float = 900.0
    keep_checkpoints: int = 3  # latest ones, plus the selected (best) step
    # compile (and run once on an all-padding batch, discarded) the train step of every
    # sequence bucket before step 1, so a short benchmark's steps are all steady-state
    precompile_buckets: bool = False
    profile_start: int = 0  # >0: a jax.profiler trace of profile_steps steps from this one
    profile_steps: int = 5
    # Run health (tjev.train.health, docs/sweep-plan.md): the on-device health/* metrics,
    # the health.json verdict (kill rules evaluated, never acted on), and acting on it: a
    # run whose verdict is "kill" stops (exit 0, health.json status "killed")
    health_metrics: bool = True
    health_report: bool = True
    health_kill: bool = False

    def __post_init__(self) -> None:
        buckets = self.seq_buckets
        if not buckets or list(buckets) != sorted(set(buckets)):
            raise ValueError(f"train.seq_buckets must be distinct and ascending: {buckets}")
        if any(b <= 0 or b % BUCKET_MULTIPLE for b in buckets):
            raise ValueError(f"train.seq_buckets must be multiples of {BUCKET_MULTIPLE}: {buckets}")
        if self.microbatch_tokens <= 0:
            raise ValueError("train.microbatch_tokens must be positive")
        # rows of the buckets that fit fill the budget exactly; the longer ones take one row
        uneven = [b for b in buckets if b <= self.microbatch_tokens and self.microbatch_tokens % b]
        if uneven:
            raise ValueError(
                f"train.microbatch_tokens={self.microbatch_tokens} is not a multiple of the "
                f"buckets {uneven} (rows of a bucket must fill the microbatch)"
            )

    def microbatch_tokens_at(self, length: int) -> int:
        """Tokens per device in a microbatch of bucket ``length``: one row of a bucket longer
        than ``microbatch_tokens``."""
        return max(self.microbatch_tokens, length)


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
    loss: LossSpec = field(default_factory=LossSpec)
    data: DataSpec = field(default_factory=DataSpec)
    log: LogSpec = field(default_factory=LogSpec)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())
