# Kernels

The sequence-mixing ops of the Qwen3.5 decoder, their pure-JAX (XLA) references and their
Pallas TPU kernels. The code is the source of truth: `src/tjev/kernels/`. Nothing in this
document has been measured on a real TPU yet. The kernels are tested for parity in Pallas
interpret mode and lowered for TPU on CPU (see [Testing](#testing)). Every performance figure
here is a prediction.

An earlier GPU version had Mosaic GPU kernels; see the jev repo history.

## Overview

Three ops mix along the sequence. Everything else in the model (projections, norms, RoPE,
gates, MLP, LoRA) is plain XLA.

| Op | Entry point | Implementations |
|---|---|---|
| Gated delta rule (GDN core) | `tjev.kernels.gated_delta_rule` | `chunked`, `recurrent` (`xla.py`); `pallas_tpu`, `pallas_tpu_split` (`pallas_tpu.py`) |
| Segmented causal conv1d (GDN input conv) | `tjev.kernels.segmented_causal_conv1d` | XLA only, with a custom VJP |
| Causal segment-masked GQA attention | `tjev.kernels.attention` | `xla` (`xla.py`), `splash` (`splash_tpu.py`) |

Layout: one package per op. `xla.py` holds the reference every other implementation is tested
against; `pallas_tpu.py` / `splash_tpu.py` hold the TPU kernels. On a non-TPU backend the
kernels run in Pallas interpret mode. The package `__init__.py` is a dispatcher that takes
`impl=` and, for the TPU kernels, wraps the call in `batch_parallel`
(`src/tjev/kernels/sharding.py`).

The model (`src/tjev/model/mixers.py`) picks the implementation from `ComputeSpec`
(`src/tjev/config/schema.py`):

| Field | Values (default first) | Effect |
|---|---|---|
| `gdn_impl` | `chunked`, `recurrent`, `pallas_tpu`, `pallas_tpu_split` | GDN core implementation |
| `gdn_chunk` | `64` | chunk size C; must be a power of two for `pallas_tpu*` (checked in `__post_init__`) |
| `gdn_precision` | `high`, `highest`, `bf16` | matmul precision inside the delta rule |
| `attention` | `xla`, `splash` | attention implementation |
| `attention_block` | `512` | XLA path only: query block (0 = one block) |
| `remat` | `full`, `none`, `core`, `minimal` | per-decoder-layer remat policy |

`ComputeSpec.uses_tpu_kernels` is true when `attention == "splash"` or `gdn_impl` starts with
`pallas_tpu`. The defaults are the portable XLA paths. The hardware presets
(`src/tjev/config/configs/hardware/tpu-v6e.yaml`, and `tpu-v5e.yaml` which extends it) set
`attention: splash`, `gdn_impl: pallas_tpu`, `gdn_precision: high`, `remat: full`.

## The gated delta rule

### Notation

Per row: q, k `[B,T,H,Dk]`, v `[B,T,H,Dv]`, g, β `[B,T,H]`, `segment_ids` `[B,T]` (0 =
padding). For Qwen3.5, Dk = Dv = 128; H is the number of value heads (key heads are repeated
up to it in `GatedDeltaNet.core`). C is the chunk size, N = T/C the number of chunks. The
model prepares the inputs: q and k are L2-normalised (`l2norm`), q is scaled by Dk^-1/2,
β = σ(b), g = −exp(A_log)·softplus(a + dt_bias) ≤ 0. The output is `[B,T,H,Dv]` fp32.

### Recurrence

Per head, with state S ∈ R^{Dk×Dv}:

```
S_t = exp(g_t) S_{t-1}
S_t = S_t + k_t ⊗ β_t (v_t − S_tᵀ k_t)
o_t = S_tᵀ q_t
```

`recurrent_gated_delta_rule` is this, token by token under `lax.scan` (tests and T = 1 decode).
It also takes an `initial_state` and returns the final state.

### Chunked WY/UT form

`chunk_gated_delta_rule` (the reference, following HF `torch_chunk_gated_delta_rule` and
MaxText `jax_chunk_gated_delta_rule`). Inside a chunk let γ_i = Σ_{j≤i} g_j (log decay since
the chunk start) and D_ij = exp(γ_i − γ_j) for j ≤ i in the same segment, 0 otherwise.

```
L      = strict_lower((kβ kᵀ) ⊙ D)                  [C,C]
T      = (I + L)⁻¹                                  UT transform of the in-chunk updates
u      = T (vβ)                  w = T (kβ ⊙ e^γ ⊙ cont)
intra  = (q kᵀ) ⊙ D                                 causal, diagonal included
q_read = q ⊙ e^γ ⊙ cont          k_write = k ⊙ e^{γ_C − γ} ⊙ in_last
carry  = e^{γ_C} · cont_C
```

and the scan over chunks carries only S:

```
v_new = u − w S
o     = q_read S + intra v_new
S     = carry · S + k_writeᵀ v_new
```

This is O(T·C) chunk-local work plus N sequential steps of [C,Dk]×[Dk,Dv] matmuls.

### Segment resets for packed rows

Packed rows carry several documents. The state restarts at every segment boundary, exactly as
if each segment ran alone (MaxText PR #5351 semantics). Three masks do it, all derived from
`segment_ids`:

- `cont` (`continues`): the token is in the same segment as the token just before the chunk,
  so it may read the incoming state. For chunk 0 the "previous segment" is −2, so nothing reads
  a state (no initial state in training).
- `in_last`: the token is in the chunk's last segment, so it writes the outgoing state.
- D is zeroed across segments, so L and `intra` never mix segments.

The carry survives the chunk only when its last token continues the incoming segment. Padding
to a multiple of C appends zero q/k/v/β with g = 0, which are exact no-ops on the state.

### The doubling inverse

`unit_lower_inverse` computes T = (I + L)⁻¹ by matmuls only (recursive doubling). With T_1 = I
and M_s the mask of the lower-left s×s block inside every 2s×2s diagonal block
(`doubling_masks`):

```
T_2s = T_s − T_s (L ⊙ M_s) T_s,     s = 1, 2, 4, …, C/2
```

This is exact block inversion, [[A,0],[B,D]]⁻¹ = [[A⁻¹,0],[−D⁻¹BA⁻¹,D⁻¹]]: 2·log2(C) matmuls,
C a power of two. It exists because Mosaic TPU has no triangular solve. The XLA reference uses
`solve_triangular` by default (`inverse="solve"`); `inverse="doubling"` runs the kernel's
algorithm in XLA for testing.

The power series (I − L)(I + L²)(I + L⁴)… is also exact for a nilpotent L, but its partial
products overflow when keys are correlated, which real keys are: L ≈ β on the whole strict
lower triangle. Every doubling intermediate is a sub-block of the true inverse, so it stays
bounded. `test_matmul_inverse_is_stable_for_correlated_keys` pins this.

### Precision modes

`gdn_precision` sets the delta-rule matmuls (`xla.mm`, `pallas_tpu._dot`):

| Mode | TPU meaning | Use |
|---|---|---|
| `highest` | fp32 (6 bf16 MXU passes) | the parity reference |
| `high` | bf16_3x | default; MaxText's choice (PR #5399) |
| `bf16` | bf16 operands, fp32 accumulation, 1 pass | fastest; loses accuracy |

Gates, exponentials, the carried state and every accumulator are fp32 in all modes. In the
fused kernel the cumsum and segment-id transposes always run at `highest`. The reason `high` is
the default: MaxText found that the gate adjoint sums products that nearly cancel, and 1-pass
bf16 operands leave the A_log / dt_bias gradients pointing the wrong way (cosine ~0.3 against
fp32).

## Pallas TPU GDN

`src/tjev/kernels/gated_delta_rule/pallas_tpu.py`, entry point `chunk_gated_delta_rule_tpu`:
the chunked reference's contract, without initial or final state. T is padded to a multiple of
`chunk` with segment-0 tokens. The design is adapted from MaxText (PR #5364, commit `61ef5db`,
and the `atwigg/mlperf` branch at `a51bcf5` with PR #5399): the reverse chunk order through
the index maps, the state gradient carried in VMEM scratch, the saved per-chunk states instead
of a recompute, and the bf16_3x precision policy. The file header and `NOTICE` carry the
attribution.

### Fused forward (`gdn_impl=pallas_tpu`)

`fused_rule` (custom VJP) → `_fused_forward` → one `pallas_call` named `gdn_fused_fwd`.

- Grid `(B, H/heads, N)`, `dimension_semantics=("parallel", "parallel", "arbitrary")`: rows
  and head groups are independent, chunks run in order.
- Inputs: q, k, v read token-major as `[B,T,H·D]` with blocks `(C, heads·D)`, so one block
  holds `heads` heads side by side on the lanes; g and β as `[B,H/heads,T,heads]` blocks
  `(C, heads)`; segment ids as a float `[B,T,1]` column (exact for small integers); the segment
  id of the token before each chunk broadcast to 128 lanes; the doubling masks
  `[log2 C, C, C]`, one block for the whole grid.
- VMEM scratch `(heads, Dk, Dv)` fp32 holds the state, zeroed when `program_id(2) == 0`.
- Per chunk and head, everything chunk-local stays in VMEM: the in-chunk cumsum (a triangular
  matmul), the segment masks (the id column transposed through the MXU with an identity
  matmul), D, L, T by recursive doubling, u, w, intra, q_read, k_write and carry, then the
  recurrence step. None of these reach HBM; in the unfused design they were about 80% of the
  GDN core's HBM traffic.
- Outputs: `out` `[B,T,H·Dv]`, `states` `[B,H,N,Dk,Dv]` (S before each chunk) and `v_new`
  `[B,H,N,C,Dv]`, the residuals of the backward.

### Reverse-recurrence backward

`_fused_rule_bwd` recomputes `prepare` and `local` (the chunk-local terms) in XLA under
`jax.vjp`, runs the Pallas kernel `gdn_recurrence_bwd` (`_backward`), and pulls its gradients
back through the XLA terms by autodiff.

- Grid `(B·H/heads, N)`, `("parallel", "arbitrary")`. The index map returns chunk `N−1−j`, so
  chunks run in reverse.
- VMEM scratch `(heads, Dk, Dv)` fp32 holds G = ∂L/∂S after the chunk, zeroed at the first
  (last) chunk.
- Per chunk and head, with S the saved state before the chunk:

```
d_vnew  = intraᵀ do + k_write G
d_u     = d_vnew               d_w      = −d_vnew Sᵀ
d_intra = do v_newᵀ            d_qread  = do Sᵀ
d_kwrite = v_new Gᵀ            d_carry  = Σ S ⊙ G
G       = carry · G + q_readᵀ do − wᵀ d_vnew
```

### Split variant (`gdn_impl=pallas_tpu_split`)

The fallback of the fused forward, and the previous design. `prepare` and `local` run in XLA
(differentiable, batched over `[B·H, N]`), and only the recurrence is a kernel: `recurrence`
(custom VJP) with forward `gdn_recurrence_fwd` and the same `gdn_recurrence_bwd`. Inputs are
laid out `[B·H, N, C, *]` fp32, grid `(B·H/heads, N)`, blocks `(heads, C, *)` per chunk. The
per-chunk scalar `carry` is broadcast to `[1, Dv]` so its block is lane-shaped.

### Heads per program

`heads_per_block(lead, preferred=8)`: 8 heads per program, halved until it divides `lead`
(16 → 8, 12 → 4, 6 → 2, 3 → 1). Fewer, larger grid steps: 8 × 64 rows per MXU pass. The fused
kernel divides H (heads per row); the split kernel divides B·H. A C = 64 chunk alone
([64×128]@[128×64]) underfills a 256×256 v6e MXU; several heads per program are the cheap way
to fill it. Off TPU, `_interpret()` selects the generic Pallas interpreter, not
`pltpu.InterpretParams`, whose effects remat cannot differentiate.

## Splash attention wrapper

`src/tjev/kernels/attention/splash_tpu.py`, `splash_attention(q, k, v, segment_ids)`: JAX's
Pallas TPU splash kernels (`jax.experimental.pallas.ops.tpu.splash_attention`, used as a
library). Splash computes softmax(q kᵀ) v with an online softmax, never materialises [T,T],
skips key blocks ruled out by the static causal mask, and ships its own backward (dq and dkv
kernels).

- GQA without repeating K/V: one MQA kernel per KV head over its group of G = H/KV query heads
  (`make_splash_mqa_single_device`, mask `MultiHeadMask` of G `CausalMask`s), vmapped over
  batch and KV heads. Head h = kv·G + g, the grouping of `jax.nn.dot_product_attention`.
- Segments: `SegmentIds(q=seg, kv=seg)`. Padding (segment 0) attends to earlier padding and
  itself, so no softmax row is empty. Splash skips blocks from the static mask only, never from
  segment ids.
- The softmax scale (default D^-1/2) is applied to q before the kernel. Qwen3.5's output gate
  and q/k norms stay outside.
- T is padded to a multiple of 128 (the lane width). All block sizes (`block_q`, `block_kv`,
  `block_kv_compute` and the dq/dkv ones) are 512 (MaxText's default), or 256 or 128 when T is
  not a multiple of 512.
- The kernel is built eagerly under `jax.ensure_compile_time_eval` (its mask tables are
  concrete arrays) and cached by `_kernel` (`lru_cache`, 32 entries) keyed by padded length,
  group size, interpret flag and the `repr` of the abstract mesh at trace time: mask tables
  built under one mesh cannot be reused under another.
- `residual_checkpoint_name="attn_context"`: the name the `core` and `minimal` remat policies
  save.

The XLA path (`blocked_attention`) runs `jax.nn.dot_product_attention` one query block at a
time, each block reading keys up to its own end and, under remat, wrapped in its own
`jax.checkpoint`: O(block·T) peak memory.

## Running under data parallelism

XLA cannot partition a Pallas custom call; on a multi-device mesh it would replicate it and
every device would process every row. `batch_parallel(fn, *args)` therefore runs the kernel
under `jax.shard_map` over the batch axis, split over all mesh axes (`("data", "fsdp")`,
`tjev.sharding.BATCH_AXES`), with `check_vma=False`. On an empty or one-device mesh it calls
`fn` directly. All arguments and outputs are batch-leading.

The mesh comes from `jax.sharding.get_abstract_mesh()`. `tjev.sharding.install_mesh(compute,
mesh)` sets it with `jax.set_mesh` when `compute.uses_tpu_kernels`, and clears it otherwise.
Training, evaluation (`tjev.eval.runs`), `tjev compile-check` and the bench call it. With LoRA
only the adapter gradients are all-reduced, so the kernels need no collectives.

## Remat policies

`ComputeSpec.remat` selects a per-decoder-layer `jax.checkpoint` policy
(`save_only_these_names`) from `REMAT_POLICIES` (`src/tjev/model/qwen35.py`); `none` disables
remat.

| Policy | Saved names |
|---|---|
| `full` | nothing (policy `None`): only the layer input |
| `core` | `gdn_core_out`, `attn_context` |
| `minimal` | `core` + `mlp_gate`, `mlp_up`, `mixer_out`, `lora_xa`, `gdn_qkv`, `gdn_z`, `gdn_ba`, `attn_qkv` |

`attn_context` exists only on the splash path. Whenever remat is on, the GDN core
(conv → gates → l2norm → delta rule) is also wrapped in its own `jax.checkpoint`, so its fp32
intermediates live only during its own backward.

Facts that shape these choices (checked on CPU with toy layers in the jev repo):

- A `custom_vjp` forward rule is rematerialised like any other code; its residuals are not kept
  automatically.
- A `checkpoint_name` inside a `custom_vjp` forward rule is honoured by the outer policy. That
  is how splash's `attn_context` is saved.
- Recompute is dead-code-eliminated per value, but a Pallas kernel is one opaque call. Saving
  `gdn_core_out` spares the layer recompute of the core, while the core's own backward still
  reruns the forward kernel for its residuals.
- With a frozen base, the backward of a base GEMM is dX only, so full remat spends about a
  third of the GEMM time recomputing. `core` and `minimal` trade that for memory.

Do not save the GDN core internals: the residual set (`states`, `v_new`, the chunk-local
terms) is tens of KB per token per layer, and the kernel has to rerun anyway.

## Testing

`tests/kernels/`:

| File | What it checks |
|---|---|
| `test_gated_delta_rule.py` | chunked vs recurrent for C ∈ {4, 8, 16, 64}; packed segments equal independent runs; initial-state continuation; gradients; doubling inverse vs solve; `high`/`bf16`/doubling variants; stability on correlated keys; the dispatcher |
| `test_causal_conv1d.py` | segments never mix; the custom VJP equals autodiff |
| `test_attention.py` | blocked equals full attention (with and without remat); segment isolation; unknown impl fails |
| `test_pallas_tpu.py` | the Pallas kernels, three ways (below) |

`test_pallas_tpu.py`:

1. Interpret mode on CPU. Fused and split GDN at `highest` against `chunk_gated_delta_rule`,
   shapes (1,128,2) C64, (2,200,4) C64 (padding) and (1,256,8) C128, with segment boundaries
   anywhere inside chunks: forward and all five gradients within 1e-5 relative. Splash against
   a dense masked reference with GQA, segments and trailing padding (T = 200 padded to 256):
   within 1e-4, gradients included.
2. TPU lowering on CPU. `jax.export(..., platforms=["tpu"])` of the gradient function, with
   interpret mode forced off, runs the real Pallas → Mosaic lowering. GDN (fused and split, all
   three precisions) must contain exactly 2 `tpu_custom_call`s (forward and backward); splash
   exactly 3 (forward, dq, dkv). Lowering errors show up here before any TPU time is paid for.
3. Compiled on a TPU VM, `@pytest.mark.tpu`. GDN at shapes up to (1, 4096, 32), C64 and C128,
   forward tolerance 1e-4 / 1e-3 / 3e-2 for `highest` / `high` / `bf16`, gradients finite and
   within twice that; splash at T 1024 and 4096 in bf16 within 3e-2 absolute.

The default `pytest` run deselects `tpu` (and `real`) tests. On a TPU VM:

```
JAX_PLATFORMS=tpu pytest -m tpu tests/kernels/test_pallas_tpu.py -q -s
```

Campaign phase `s0` (`cloud/tpu_campaign.sh`) runs exactly this first. If it fails, it writes
`reports/tpu-kernels.off`, and every later job falls back to the XLA paths (`--no-kernels`;
the zero-shot evals get `compute.attention=xla compute.gdn_impl=chunked`). It then runs
`tjev campaign bench` (`src/tjev/campaign/bench.py`):

- Kernel micro-benchmarks, forward + backward. GDN at 8k tokens (B = 2, T = 4096, H ∈ {16, 32}):
  `chunked/high`, `pallas/high`, `pallas/bf16`, `pallas/high/C128`. Attention at 8192 tokens
  (T 1024 H8/2, T 4096 H8/2, T 4096 H16/4): `splash` against dense XLA.
- The real train step per model size on one chip, variants `kernels` (the preset),
  `kernels+core`, `kernels+minimal`, `split` and `xla`. `best_variant` is the fastest finite
  variant whose HBM peak stays under 80% of the chip; `tjev campaign cost --bench` reads it.
  A variant that fails to compile or runs out of memory is recorded with its error and
  skipped.

## Performance model and roadmap

All numbers in this section are predictions from a per-op roofline analysis done in the jev
repo. None has been measured on a TPU. The s0 bench and an xprof trace are the first
measurements.

### Rooflines

| Chip | bf16 peak | HBM | Critical intensity | MXU |
|---|---|---|---|---|
| v6e | 918 TFLOP/s | 32 GB, 1.6 TB/s | ~575 FLOP/B | 2 × 256×256 |
| v5e | 197 TFLOP/s | 16 GB, 0.82 TB/s | ~240 FLOP/B | 4 × 128×128 |

A bf16 matmul X[B,D]·W[D,F] has arithmetic intensity ≈ B for B ≪ D, F: compute-bound only
above ~575 tokens per matmul on v6e (the presets use ≥ 1k per chip). LoRA adapters (intensity
≈ rank), elementwise ops and the unfused fp32 GDN core are HBM-bound; v5e, with its lower
critical intensity, loses less to them than v6e.

### Where the time goes (2B, v6e, one super-block, 8192 tokens per chip, full remat)

Time per op = max(FLOPs × passes / (918 TF × MXU fill), bytes / 1.64 TB/s), with MXU fill
assumed (K/256)·(N/256) for dots narrower than 256. That fill is the main uncertainty.

Before the fused forward: 53.5 ms per super-block against an 8.2 ms matmul floor, about 15%
model MFU; 77% of the step is non-MXU work.

| Share of step | Sink |
|---|---|
| 49% | GDN core on the XLA path: doubling inverse 16.5% (12 small dots, each writing a `[BH,N,C,C]` fp32 tensor), `prepare` 7.8%, recurrence 6.7%, u/w 6.4%, kβkᵀ + intra 5.3% |
| 14% | LoRA expand + add |
| 5% | silu · up |
| ~3% each | conv1d, l2norm |
| 3–4% | attention (splash) |

The fused forward is predicted to save about 15% of the step; a fused backward too, about 36%
in total. After the full fusion ladder below, predicted model MFU is 40% for 2B and 44% for 4B.

### Roadmap

1. Fused GDN backward (`gdn_chunk_bwd`). The backward currently recomputes the chunk-local
   terms in XLA. Spec: same grid as the forward, chunks reversed, G in a `(heads, Dk, Dv)`
   fp32 scratch; reads q, k, v, the gates, the saved `states` (bf16 is enough, since the
   state-path dots are 1-pass) and T (fp32, saved by the forward); recomputes D, u/w and intra
   in VMEM. With X = Tᵀ·d_vnew, the identities (checked in fp64 in the jev repo)

   ```
   d_vβ   = X
   d_kbe  = −X Sᵀ
   dL     = strict_lower(−X v_newᵀ)     (replaces dT and the two T sandwiches)
   −wᵀ d_vnew = −kbeᵀ X                 (the backward needs neither w nor u)
   ```

   leave five Dv contractions (do·Sᵀ, X·Sᵀ, v_new·Gᵀ, do·v_newᵀ, X·v_newᵀ) plus the chunk-local
   q/k part; d_γ comes from row and column sums of dD ⊙ D plus the e^γ and carry terms
   (MaxText's closed form). Precision: `high` for the intra-chunk products, 1-pass bf16 for the
   state path. Predicted 0.44 ms forward and 0.88 ms backward per layer, against 2.40 / 3.99 ms
   unfused.
2. Replace the dense doubling inverse by blocked-16 forward substitution (VPU on the 16×16
   diagonal blocks, MXU off-diagonal, as MaxText does): at C = 64 the 64-wide doubling dots fill
   1/16 of a v6e MXU.
3. A fused LoRA linear: x@W + (x@A)@(sB) in one Pallas matmul, predicted about −15%.
4. Smaller fusions: swiglu into the `down_proj` prologue, gated norm into the `out_proj`
   prologue, conv1d + silu + mask inside the GDN kernel (a few percent each).

Measure first on the VM: the MXU fill of 64- and 128-wide Pallas dots, whether XLA fuses the
LoRA add into the dot epilogue, whether the layer and core recomputes of the GDN core are
deduplicated, and the HLO op stats per kernel name (`gdn_fused_fwd`, `gdn_recurrence_fwd`,
`gdn_recurrence_bwd`, the splash kernels) and named scope (`gdn_conv`, `gdn_rule`, `gdn_core`,
`attention_core`, `mlp`).
