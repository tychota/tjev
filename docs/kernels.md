# Kernels

The sequence-mixing ops of the Qwen3.5 decoder, their pure-JAX (XLA) references and their
Pallas TPU kernels. The code is the source of truth: `src/tjev/kernels/`. The kernels are
tested for parity in Pallas interpret mode, lowered for TPU on CPU (see [Testing](#testing))
and validated against the official model on a v6e-1; their measured correctness and
throughput are in [results.md](results.md#tpu-correctness-v6e-1-measured), the step
profile in [profiling.md](profiling.md). The training path on v6e-1 is the GDN megakernel
`pallas_tpu_mega2` at chunk 128 with the blocked inverse, `core_states` remat, splash
attention (1024 tiles, fused backward up to 4B) and merged LoRA up to 4B
(`plan.HARDWARE["v6e1"]`). Numbers labelled as predictions below are the design-time
roofline estimates, kept as the record of why each kernel was built.

An earlier, GPU-only development version also had Mosaic GPU kernels; they are not part of tjev.

## Overview

Three ops mix along the sequence. Everything else in the model (projections, norms, RoPE,
gates, MLP, LoRA) is plain XLA.

| Op | Entry point | Implementations |
|---|---|---|
| Gated delta rule (GDN core) | `tjev.kernels.gated_delta_rule` | `chunked`, `recurrent` (`xla.py`); `pallas_tpu`, `pallas_tpu_split` (`pallas_tpu.py`); `pallas_tpu_v2` (`pallas_tpu_v2.py`); `pallas_tpu_mega1`, `pallas_tpu_mega2` (`pallas_tpu_mega.py`: the whole core, conv included) |
| Segmented causal conv1d (GDN input conv) | `tjev.kernels.segmented_causal_conv1d` | XLA only, with a custom VJP |
| Causal segment-masked GQA attention | `tjev.kernels.attention` | `xla` (`xla.py`), `splash` (`splash_tpu.py`) |

Layout: one package per op. `xla.py` holds the reference every other implementation is tested
against; `pallas_tpu.py` / `splash_tpu.py` hold the TPU kernels. On a non-TPU backend the
kernels run in Pallas interpret mode. The package `__init__.py` is a dispatcher that takes
`impl=` and, for the TPU kernels, wraps the call in `batch_parallel`
(`src/tjev/kernels/sharding.py`).

The model (`src/tjev/model/gdn.py`, `attention.py`) picks the implementation from `ComputeSpec`
(`src/tjev/config/schema.py`):

| Field | Values (default first) | Effect |
|---|---|---|
| `gdn_impl` | `chunked`, `recurrent`, `pallas_tpu`, `pallas_tpu_split`, `pallas_tpu_v2`, `pallas_tpu_mega1`, `pallas_tpu_mega2` | GDN core implementation |
| `gdn_chunk` | `64` | chunk size C; must be a power of two for `pallas_tpu*` (checked in `__post_init__`); the v6e1 path uses 128 |
| `gdn_precision` | `high`, `highest`, `bf16`, `mixed` | matmul precision inside the delta rule |
| `gdn_inverse` | `default`, `solve`, `doubling`, `blocked` | how the UT transform is solved: on the XLA side for the split paths, inside the kernels for v2 and the megakernels (below) |
| `gdn_heads_per_step`, `gdn_vmem_fraction` | `0`, `0.0` | v2 / megakernels: value heads per grid step, the kernels' VMEM limit ([v2 options](#v2-options-blocked-inverse-heads-per-step-vmem)) |
| `attention` | `xla`, `splash` | attention implementation |
| `attention_block` | `512` | XLA path only: query block (0 = one block) |
| `attention_splash_block`, `attention_splash_compute`, `attention_fused_bwd` | `512`, `0`, `false` | splash tiles, kv compute sub-block, fused dkv+dq backward |
| `remat`, `remat_by_bucket` | `full`, `none`, `core`, `core_states`, `minimal`, `minimal_states`; `{}` | per-decoder-layer remat policy ([Remat policies](#remat-policies)), optionally per sequence bucket |
| `lora_impl` | `unmerged`, `merged` | LoRA form ([Merged LoRA](#merged-lora-computelora_implmerged)) |
| `gdn_norm` | `heads`, `flat` | GDN output gated RMSNorm over a `[B,T,H,Dv]` view, or on `[B,T,H·Dv]` with the per-head sums as fp32 matmuls (no TPU relayout copies; below) |
| `gdn_input_mask` | `true`, `false` | zero the padding rows of the GDN input before its in-projections (redundant: padding reaches nothing valid) |
| `flat_rows` | `false`, `true` | projections on `[B·T, D]` rows |
| `kernel_out_in` | `[]` | projections whose frozen kernel is stored `[out, in]` (lora.init default only) |
| `head_rows` | `full`, `letters` | an untied head (9B) keeps only the answer letters' rows (no full-vocabulary logits) |

### XLA-side flags (AOT v6e evidence)

`scripts/aot_tpu.py` compiles the run's train step for a v6e topology on a CPU host
(libtpu's compile-only client, `pip install 'jax[tpu]'`) and reports XLA's memory analysis,
its cost analysis and the layout copies of the optimised HLO. On the 9B smoke config
(mega2 C128 blocked, core_states, r16, one 2048-token row, two microbatches):

| Variant | temp GiB | args GiB | bytes accessed | layout copies |
|---|---|---|---|---|
| defaults | 5.21 | 17.13 | 43.7 GB | 682 MiB (f32[256,8,32,128] gated-norm relayouts) |
| `gdn_norm=flat` | 5.02 | 17.13 | 41.2 GB (−5.6%) | 343 MiB (none in gdn_out / gdn_core) |
| `head_rows=letters` | 5.21 | 15.23 (−1.89) | 43.8 GB | 682 MiB |
| `lora_impl=merged` | 7.76 | 17.13 | 47.5 GB | 3148 MiB (7 stacked bf16[8,4096,4096] kernels copied to {1,2,0}) |
| merged + `gdn_norm=flat` | 5.98 | 17.13 | 42.3 GB | 460 MiB (no stacked-kernel copy) |
| merged + `kernel_out_in=[in_proj_z]` | 6.07 | 17.13 | 44.2 GB | 1228 MiB (no stacked-kernel copy) |

At the 1024 bucket `kernel_out_in=[k_proj,v_proj]` removes the two bf16[8,4096,1024] copies
(temp −0.13 GiB); `flat_rows` is worse there (bytes +9%, temp +0.45 GiB). Measured on v6e-1
(tjev-integ-4b / -9b, [training.md](training.md#integration-on-v6e1-what-was-adopted)):
`gdn_norm=flat` +3.8% real tokens/s at 4B, `head_rows=letters` +5.2% at 9B with it; both
are v6e1 defaults, `flat_rows` and `kernel_out_in` are not.

`ComputeSpec.uses_tpu_kernels` is true when `attention == "splash"`, `gdn_impl` starts with
`pallas_tpu` or `lora_impl == "fused"`. The defaults are the portable XLA paths. The hardware presets
(`src/tjev/config/configs/hardware/tpu-v6e.yaml`, and `tpu-v5e.yaml` which extends it) set
`attention: splash`, `gdn_impl: pallas_tpu`, `gdn_precision: high`, `remat: full`. The
planner's `--hardware v6e1` line overrides them per size with the measured best path
(`pallas_tpu_mega2`, chunk 128, `blocked`, `core_states`, merged LoRA ≤ 4B, best-fit
packing, `gdn_norm=flat`; 9B: `head_rows=letters`, `minimal_states` on the 1024 / 2048
buckets, unmerged LoRA).

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

`gdn_inverse` picks the solve on the XLA side (`xla.solve_unit_lower`): the chunked path
(default `solve`), and the Pallas paths' XLA terms (default `doubling`): the split forward and
backward, and the fused backward's recompute of `local`. `blocked` is blocked forward
substitution on the right-hand side [vβ, kβe^γ] (`xla.blocked_solve`): the 16×16 diagonal
blocks inverted by doubling (batched), then C/16 block rows solved in order, one
[16, 16i]×[16i, 256] matmul each. At C = 64 that is 4 small dependent matmuls instead of 12
dense 64³ ones with a [C,C] fp32 HBM round trip each (the dense doubling is 35% of the
forward FLOPs of `local` and dominated "xla.py" in the split profile). The fused forward
kernel always uses doubling. Variants: `split+solve`, `split+blocked`, `kernels+solve`.

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
| `mixed` | `high` for the intra-chunk products, `bf16` for the dots with the state S (or its gradient G) as an operand | MaxText's split (PR #5154/#5386/#5399) |

`mixed` (`xla.part(precision, kind)`): intra-chunk = kβkᵀ, qkᵀ, the UT solve, T·[vβ, kbe],
intra·v_new, the state update k_writeᵀ·v_new and, backward, intraᵀ·do, do·v_newᵀ and the
dS terms q_readᵀ·do, wᵀ·d_vnew; state path = w·S, q_read·S and, backward, k_write·G,
d_vnew·Sᵀ, do·Sᵀ, v_new·Gᵀ. Variants `mixed`, `split+mixed`, `xla-mixed` (parity).

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

### Fused forward and backward (`gdn_impl=pallas_tpu_v2`)

`src/tjev/kernels/gated_delta_rule/pallas_tpu_v2.py`, entry point
`chunk_gated_delta_rule_v2`: two kernels, `gdn_v2_fwd` and `gdn_v2_bwd`, and nothing
chunk-local in XLA. The design follows MaxText's `atwigg/mlperf` GDN kernels (PRs #5153,
#5273, #5274, #5399) and flash-linear-attention's closed forms; the code is written for tjev's
layouts and segment masks (header, `NOTICE`).

- XLA side: γ, the in-chunk cumulative log decay (`inchunk_cumsum`: one [C,C] HIGHEST
  matmul, exact, whatever XLA does with a cumsum); per-token masks `aux` [B,T,3] (segment
  id, reads the incoming state, writes the outgoing one) and the segment ids as rows
  [B,N,1,C] (`token_masks`); head-major [B,H,T,D] layouts. q and k stay at the key heads:
  value head h reads key head h // (H/Hk) through the BlockSpec index map (the model no
  longer repeats them; the other implementations repeat in the dispatcher).
- Grid (B, H/hb, N), `("parallel", "parallel", "arbitrary")`; hb value heads per program
  (`heads_per_block`: the largest multiple of the GQA ratio dividing H, up to 8) are the
  **batch dimension of every dot** (`lax.dot_general` batch dims (0, 0)), not a Python loop.
- Forward per chunk: D from γ columns and rows (no MXU transposes), L, T = (I+L)⁻¹ by batched
  doubling with iota masks, `T @ [vβ | kβe^γ]` as one 256-lane matmul, `[w ; q_read] @ S` as
  one 2C-row matmul, v_new, o, the state update. Outputs o (v's dtype, bf16 in the model),
  T [B,H,N,C,C] and S before each chunk [B,H,N,Dk,Dv] (named `gdn_tinv`, `gdn_states`).
- Backward per chunk, in reverse, G in scratch: v_new = T(vβ − kbe S) (no saved v_new, u, w),
  X = Tᵀ d_vnew, dL = strict(−X v_newᵀ) (no differentiation through the inverse),
  `[do ; X] @ Sᵀ` and `[q_read ; −kbe]ᵀ @ [do ; X]` as one matmul each, dq/dk/dv, dβ, and dγ
  in closed form: the e^{±γ} factors' derivatives plus the decay's row sums minus column sums,
  rowsum((dL⊙D)⊙(kβkᵀ)) = Σ kβ⊙((dL⊙D)k) and the like, from matmuls the gradients need anyway.
  GQA: dq, dk summed over each group in the kernel.
- Precision: all four modes; `mixed` stores S in bf16 (`bf16` also T). VMEM:
  `vmem_limit_bytes` = 90% of `--xla_tpu_scoped_vmem_limit_kib` when LIBTPU_INIT_ARGS sets it.
- Remat: under `core_states` the layer saves `gdn_tinv` + `gdn_states` (~1.3 KB per token and
  value head per layer in fp32 at C=64, T 256 B + S 1 KB; S in bf16 under `mixed`) and the
  forward kernel runs once per microbatch.
- Shapes on TPU: head dims multiples of 128, chunk a power of two ≥ 8 (`check_tpu_shapes`).

Variants: `fused-bwd`, `fused-bwd+states`, `fused-bwd+mixed`, `fused-bwd+mixed+states`
(bench and parity), `gdn-v2` (parity, the kernel alone); bench kernels `v2/high`, `v2/mixed`,
`v2/bf16`, `v2/mixed/C128`.

### v2 options: blocked inverse, heads per step, VMEM

- `compute.gdn_inverse=blocked` (v2 and the megakernels): the forward kernel's (I+L)⁻¹ starts
  from the exact 16×16 diagonal-block inverse, computed by forward substitution in fp32 on the
  VPU for every block and head at once (`_diagonal_block_inverse`: at step r, row b·16+r of X
  is final and is eliminated from the rows below it in its block; MaxText's
  `invert_triangular_matrix`, FLA's `solve_tril`), then log2(C/16) doubling levels: 2
  dependent dot pairs at C = 64 instead of 6, 3 at C = 128. Every intermediate is still a
  sub-block of the true inverse (no power series beyond the diagonal blocks). `default` and
  `doubling` keep the 6-level doubling; `solve` is refused.
- `compute.gdn_heads_per_step` (0: 8): value heads per grid step, the batch dimension of
  every dot (0.8B: 16 = all heads, grid (B, 1, N); 9B: up to 32).
- `compute.gdn_vmem_fraction` (0: 90% of `--xla_tpu_scoped_vmem_limit_kib` when set): the
  kernels' `vmem_limit_bytes` as a fraction of `pltpu.get_tpu_info().vmem_capacity_bytes`,
  clamped to 90% of the scoped flag (a kernel limit above XLA's scoped one is an error). With
  the v6e preset's 96 MiB flag the default is already 86 MiB.
- Chunk 128: `compute.gdn_chunk=128` (one more doubling level, T [B,H,N,128,128]).

Variants (bench and parity): `v2-hb16`, `v2-hb32`, `v2-blocked`, `v2-c128`,
`v2-c128-blocked`.

### Megakernels (`gdn_impl=pallas_tpu_mega1`, `pallas_tpu_mega2`)

`src/tjev/kernels/gated_delta_rule/pallas_tpu_mega.py`, entry point `gdn_megakernel`, called
from `GatedDeltaNet.core_mega` through `gated_delta_rule.gdn_core_mega` (shard_map over the
batch; A_log, dt_bias and the conv weight replicated). The GDN core from the projection
outputs in one forward and one backward kernel around v2's chunk step (`fwd_step`,
`bwd_step`). Design: MaxText `atwigg/mlperf` `kernels/gdn/` (forward: conv halo in VMEM
scratch, in-kernel gates / l2norm / GQA) and `gdn_bwd/` (in-kernel l2norm and gate
backward); the conv backward follows FLA's `causal_conv1d_bwd_kernel` and Dao's
`causal_conv1d_bwd.cu` (per-token sequence masks on every tap, SiLU' on the recomputed
pre-activation). No code copied (header, `NOTICE`).

- mega1: input the post-conv, post-SiLU, masked qkv (the XLA conv with its barriers stays).
  mega2: input the raw `in_proj_qkv` output; the conv, SiLU and padding mask run in the
  kernels, so the XLA conv (and the fenced miscompile) is gone and no post-conv qkv reaches
  HBM, forward or backward.
- Layouts: qkv token-major `[B, T, 2·Hk·Dk + H·Dv]`, q, k and v read by three BlockSpecs of the
  same array (lane blocks hb/r·Dk and hb·Dv; three XLA slices only if the offsets are not whole
  blocks), split per head by 128-lane slices in the kernel; a and b packed per head group into
  one lane-dense `[B, G, T, 128]` fp32 tile (lanes [0, hb) a, [hb, 2hb) b), A_log and dt_bias in
  `[G, 2, 128]`; token masks `aux [B, T, 8]` (segment id, reads / writes the carried state,
  keep_1..keep_{K−1} = "token t may read t − L", valid) and the segment row `[B, N, 1, C]`;
  the output token-major `[B, T, H·Dv]`.
- Forward per chunk: [mega2: z = Σ_L keep_L ⊙ x[t−L] ⊙ w[K−1−L] with the previous chunk's
  last 8 raw rows in a VMEM halo (`pltpu.roll` on the [8 + C] rows), SiLU, valid] → per-head
  split → l2norm(q), l2norm(k) with the zero-row guard, q·Dk^-½ → β = σ(b),
  g = −e^{A_log}·softplus(a + dt_bias) on the lane tile, γ = causal·g (one HIGHEST dot for all
  heads), γ as rows by an identity TN dot → v2's `fwd_step`. Residuals: T and S (named
  `gdn_tinv`, `gdn_states`: `core_states` saves them) and the inputs.
- Backward per chunk (reverse): the same recompute (mega2's halo from an XLA-built
  `[B, N, 8, W]` array of the 8 rows before each chunk: the previous chunk is not visited yet)
  → v2's `bwd_step` → l2norm backward r·(dy − u Σ u·dy) → GQA group sum → [mega2: dz =
  d(act)·SiLU'(z)·valid, d(x)[t] = Σ_L w[K−1−L]·(keep_L ⊙ dz)[t+L], the rows past the chunk end
  from the next chunk's first 8 rows of keep_L ⊙ dz carried in scratch]; gates: dγ, dβ into a
  zeroed [C, 128] scratch by one-lane stores, d(g) = causalᵀ·dγ (one TN dot), d(a) =
  d(g)·(−e^{A_log}σ(a + dt_bias)), d(b) = dβ·β(1−β), d(A_log) per token d(g)·g (lanes
  [2hb, 3hb), `pltpu.roll`). Outputs d(qkv) (three token-major arrays, concatenated), the
  gradient tile; XLA sums A_log / dt_bias (dead code in training: frozen).
- Conv weight gradient: only when the weight is differentiated (the custom VJP uses symbolic
  zeros; the backward kernel then also writes dz, `gdn_mega2_bwd_dw`, and XLA forms
  Σ_t dz·keep·x[t−L]): the ground truth, never training.
- Precision: all four modes (v2's split); conv, SiLU, l2norm and gates fp32 on the VPU, the
  cumsum and transposes at HIGHEST. hb ≤ 42 (three lanes per head in the gate tile).
- Tests (`tests/kernels/test_gdn_mega.py`): interpret mode against the XLA path at fp32,
  forward and the gradient of qkv, a, b, A_log, dt_bias, conv weight (segment starts at
  chunk offsets 0-3 and C−1, T = 1100 across 512-row offsets, GQA 1 and 2, blocked inverse,
  two head groups, `high`, `mixed`); packed rows = segments alone; the TPU interpreter
  (NaN-filled VMEM, races); TPU lowering at the 0.8B and 9B layouts; `-m tpu`: the compiled
  kernels at head dim 128. Model: `tests/model/test_parity.py` (logits vs HF, packed rows),
  `tests/model/test_remat.py` (policies, one forward kernel per layer under `core_states`).

Variants (bench and parity, all with `remat=core_states`): `mega1`, `mega2`,
`mega2-blocked`, `mega2-hb16`, `mega2-c128-blocked`, `mega2+lora-merged`,
`mega2+splash-1024-fusedbwd`, `mega2-blocked+lora-merged+splash-1024-fusedbwd`; parity also
`gdn-mega1`, `gdn-mega2` (XLA attention: the kernel alone). Bench kernels (`gdn-core
H16/16`, `H16/32`): `xla-glue+v2`, `mega2`, `mega2/blocked`, `mega2/hb16`, and post-conv
`xla-glue+v2` / `mega1`, fwd + bwd from the projection outputs.

Model level on CPU (fp32, 1024-wide 4-layer fixture with head dims 128, 512 tokens in 4
segments + padding, rank-32 LoRA, σ_B = 2e-4; LoRA gradients vs the XLA reference at
`highest`): v2 7.3e-6 (worst leaf 1.6e-5), v2-blocked 7.3e-6, v2-c128-blocked 1.1e-5, mega1
7.3e-6, mega2 7.4e-6, mega2-blocked 7.3e-6, mega2-c128-blocked 1.1e-5, (mega2+)lora-merged
1.0e-5; loss ≤ 6.4e-8.

## Merged LoRA (`compute.lora_impl=merged`)

`src/tjev/model/lora.py` (`_merged_matmuls`, a custom VJP over the projections that share an
input). Forward y_i = x·W'_i with W'_i = W_i + s_i·A_i B_i formed per call (A, B rounded to the
compute dtype, the delta accumulated in fp32, W' rounded once): one matmul at the base cost,
no LoRA pass. Backward: dx = Σ dy_i·W'_iᵀ; t_i = s_i·dy_i B_iᵀ; dA = xᵀ[t_1 | t_2 | …] and
x·[A_1 | A_2 | …] each in one pass over x (as `grouped`); dB_i = s_i·(xA_i)ᵀ dy_i; dA, dB in
fp32. The full dW = xᵀdy only when the frozen base is differentiated (symbolic zeros: the
ground truth, never training). Residuals: x, W', A, B (W' is recomputed under remat, it is
not a named residual).

Numerics (CPU, hidden 1024 / 4-layer realistic-width fixture, rank 32 rsLoRA,
σ_B = 2e-4 as in parity, 512 packed tokens, LoRA gradients vs the fp32 unmerged reference):

| path | loss rel | grad rel L2 | cosine |
|---|---|---|---|
| fp32 merged | 1.3e-7 | 1.0e-5 | 1.000000 |
| bf16 unmerged (today) | 2.2e-4 | 5.2e-2 | 0.99865 |
| bf16 merged | 3.3e-4 | 4.9e-2 | 0.99879 |

In bf16, W' quantises the delta at W's scale (δ ≈ 2e-4 against a W ulp of ~2e-4 at
|W| ≈ 0.05): the forward resolves the adapter no better than one ulp of W. The gradient error
is the bf16 model's own floor either way; the adapter's effect on the loss is below bf16 noise
for both paths at this σ_B. The loss-curve A/B ran as a sweep arm on v6e-1: paired online
loss −0.005 ± 0.004 and −0.009 on two seeds, eval −0.005 / −0.008, i.e. no quality cost
([results.md](results.md#tpu-sweep-v6e-1-08b-proxy-measured)). Speed (tjev-bench4,
against unmerged at the planner's microbatches): +26% at 0.8B, +16% at 2B, +9% at 4B,
−4% at 9B, where 2048-token microbatches cannot amortise the W' pass. Merged LoRA is the
v6e1 default up to 4B; 9B trains unmerged. Variants: `lora-merged`, `mega2+lora-merged`.

## Fused LoRA (removed)

A Pallas TPU kernel pair that computed the unmerged function exactly, y_i = x W_i +
s_i·(x_d A_i) B_i with fp32 accumulation and no W' (a custom VJP over the projections
sharing an input: one forward kernel per projection, one backward kernel per group, the
frozen kernels scanned unsliced and indexed by a scalar-prefetch layer index). It was
`compute.lora_impl=fused` from cycle 3 to the integration runs; after those measured it
slower than both production forms it was removed, with its tiles (`lora_fused_block_*`),
its bench and parity variants (`lora-fused`, `mega2-c128-fused`, `allin`) and its tests.
The design, its numerics and the measurement stay here as the record.

Why. Measured on v6e-1 (9B, rank 16, microbatch 2048, xprof hlo_stats with perf counters):
the `lora` scope is 14.2% of device time at 21 TFLOP/s and 1.15 TB/s, HBM-bound 96% of its
time, ~20% of HBM traffic; the base matmuls are 54%, compute-bound at ~701 TFLOP/s (76% of
peak). The narrow r-column matmuls (x·A, xᵀ·t, dy·Bᵀ, xaᵀ·dy) each re-read x or dy
(e.g. bf16[2,1024,16] from bf16[2,1024,4096]: ~41 µs against ~11 µs to stream x). Merged
LoRA removes them but forms W' = W + s·AB per microbatch, one weight read and write per
projection: tjev-bench4 measured it against unmerged at the planner microbatches (0.8B /
2B 8192, 4B 4096, 9B 2048 tokens) +26% / +16% / +9% / −4% (9B slower: 2048 tokens cannot
amortise the W' pass), so 0.8B-4B train merged and 9B unmerged (`plan.py`, v6e1). The
target of the fused kernels: beat merged at 4B and unmerged at 9B, with no W' at all.
Per 65k-token step (profile review): 9B unmerged r16 spends 1,083 ms in LoRA (16%: forward
238, backward 674, remat 171); 4B merged 62.5 ms (10.8%, merge included), 2B 52, 0.8B 34;
merged at 9B also pays ~17 ms per microbatch for the merge and 1.75 GiB of transposed
copies of the stacked frozen kernels per step. The estimated floor of a fused kernel at
9B is ~60-100 ms per step. The kernels read W in its stored [L, K, N] layout (dx = dy·Wᵀ
as an NT dot inside the kernel, no XLA transpose) and fuse both dy consumers of the
backward (t = s·dy·Bᵀ and dB = xaᵀ·dy) into the dx pass.

### Design

Forward: grid (M/tm, N/tn, K/tk), the base matmul accumulated in an fp32 VMEM scratch, the
group's xa = x_d·[A_1 | A_2 | …] computed on the n = 0 sweep from the x tile already in
VMEM and written once; y = acc + s·xa·B_pad at the last k step. Backward: one call per
group, grid (M/tm, K/tk, J) over the concatenated output tiles of the group, each
projection's dy_i / W_i / B_i blocks reached once per (m, k) through clamped index maps;
dx for the whole group, t = ∂L/∂xa and dB accumulated in VMEM, dA from the x tile at the
finalize, so x is read once in the backward and dy once per K tile. Narrow projections
(GDN `in_proj_a` / `in_proj_b`, 16-32 outputs) stayed in XLA on their columns of xa;
ragged rows masked in every reduction; rows halved while the VMEM estimate exceeded 90%
of the scoped limit. Roofline intensity on paper: 1024 FLOP/B forward, 683 backward
(v6e's critical intensity ~575).

### Numerics

CPU, interpret mode, LoRA gradients against the fp32 unmerged reference
(`scripts/lora_numerics.py`; fixture: hidden 1024, head dims 128, 4 layers, projections
at Qwen scale, rank 32 rsLoRA, σ_B = 2e-4, 512 packed tokens in 4 segments + padding,
next-token CE over the vocabulary):

| path | loss rel | grad rel L2 | cosine | worst leaf rel L2 / cosine |
|---|---|---|---|---|
| fp32 fused | 6.2e-8 | 1.6e-6 | 1.000000 | 8.1e-6 / 1.00000 |
| fp32 merged | 6.2e-8 | 3.7e-6 | 1.000000 | 1.0e-5 / 1.00000 |
| bf16 unmerged | 1.2e-4 | 3.0e-2 | 0.999552 | 4.5e-2 / 0.99900 |
| bf16 merged | 1.5e-4 | 2.3e-2 | 0.999728 | 3.8e-2 / 0.99930 |
| bf16 fused | 6.7e-5 | 2.2e-2 | 0.999770 | 4.3e-2 / 0.99909 |
| bf16 unmerged, rank 16 | 9.4e-5 | 2.9e-2 | 0.999593 | 4.6e-2 / 0.99899 |
| bf16 merged, rank 16 | 2.1e-5 | 2.3e-2 | 0.999729 | 3.8e-2 / 0.99929 |
| bf16 fused, rank 16 | 4.8e-5 | 2.1e-2 | 0.999777 | 4.2e-2 / 0.99913 |

The bf16 loss differences (1e-5 to 2e-4) move between runs of the same path by ~1e-4
(the XLA:CPU program changes with what is a constant): they are noise at this size.

Real weights (`--model`, the official Qwen3.5 snapshots truncated to their first 4 layers =
one super-block, rank 32 rsLoRA, σ_B = 2e-4, 512 tokens of English text in 4 segments +
padding; Colab CPU, interpret mode):

| size | path | loss rel | grad rel L2 | cosine | worst leaf rel L2 / cosine |
|---|---|---|---|---|---|
| 0.8B | fp32 fused | 2.1e-7 | 9.6e-7 | 1.000000 | 3.7e-6 / 1.00000 |
| 0.8B | bf16 unmerged | 7.0e-5 | 1.4e-2 | 0.999901 | 4.5e-2 / 0.99933 |
| 0.8B | bf16 merged | 2.1e-4 | 7.2e-3 | 0.999974 | 2.3e-2 / 0.99974 |
| 0.8B | bf16 fused | 8.6e-5 | 5.9e-3 | 0.999983 | 4.2e-2 / 0.99945 |
| 2B | fp32 fused | 0 | 8.9e-7 | 1.000000 | 4.9e-6 / 1.00000 |
| 2B | bf16 unmerged | 5.7e-5 | 1.4e-2 | 0.999902 | 3.6e-2 / 0.99949 |
| 2B | bf16 merged | 1.9e-4 | 7.5e-3 | 0.999972 | 2.3e-2 / 0.99974 |
| 2B | bf16 fused | 7.4e-5 | 6.0e-3 | 0.999982 | 3.2e-2 / 0.99959 |
| 4B | fp32 fused | 8.1e-8 | 9.7e-7 | 1.000000 | 4.0e-6 / 1.00000 |
| 4B | bf16 unmerged | 1.1e-4 | 1.7e-2 | 0.999869 | 3.1e-2 / 0.99954 |
| 4B | bf16 merged | 7.7e-5 | 7.1e-3 | 0.999975 | 1.5e-2 / 0.99989 |
| 4B | bf16 fused | 5.3e-5 | 6.0e-3 | 0.999982 | 4.0e-2 / 0.99925 |
| 9B | fp32 fused | 0 | 7.8e-7 | 1.000000 | 2.1e-6 / 1.00000 |
| 9B | bf16 unmerged | 8.6e-5 | 2.1e-2 | 0.999792 | 9.0e-2 / 0.99960 |
| 9B | bf16 merged | 5.4e-6 | 7.1e-3 | 0.999975 | 1.6e-2 / 0.99987 |
| 9B | bf16 fused | 4.3e-5 | 7.5e-3 | 0.999972 | 9.2e-2 / 0.99959 |
| 9B, rank 16 | bf16 unmerged | 1.1e-5 | 1.6e-2 | 0.999876 | 8.3e-2 / 0.99957 |
| 9B, rank 16 | bf16 merged | 1.1e-6 | 7.5e-3 | 0.999972 | 1.7e-2 / 0.99986 |
| 9B, rank 16 | bf16 fused | 2.3e-5 | 9.0e-3 | 0.999960 | 8.6e-2 / 0.99962 |

(fp32 merged: ≤ 1.4e-6 at every size.) On real weights the fused path's overall gradient
error is 0.4-0.6× unmerged's, level with merged's (below it at 0.8B-4B, 1.1-1.2× it at
9B); its worst single leaf stays at unmerged's (3-9e-2: the same leaf, a near-zero
gradient). Every bf16 path is at the bf16 model's own floor; fp32 fused is exact to
reordering (≤ 1e-6).

In bf16 the fused path rounded each projection once (fp32 accumulation of the base and the
adapter, one cast), where unmerged rounds x·W, xa, xa·B and the sum separately: its
gradient error was at or below the other two on this fixture, all three at the bf16
model's own floor (~2-3%). Its unit tests (interpret mode, fp32) held the forward and
every gradient within 1e-5 of the reference at ragged rows, groups of 1-3 projections,
rank 16 / 32 / 64, dropout and the stacked weights; the model matched the unmerged loss
and LoRA gradients (2e-5) under every remat policy and HF's official logits with a fresh
adapter (2e-4).

### Measured on TPU: slower, removed

The kernels compiled and ran on v6e-1. In the train step they lost to the production
paths: `mega2-c128-fused` against `mega2-c128-merged` at 4B −16% real tokens/s, against
unmerged at 9B −7% (tjev-integ-4b / -9b,
[training.md](training.md#integration-on-v6e1-what-was-adopted)), and they added
~2–2.4 GiB of temporaries per bucket. The design had the base matmul's intensity on
paper, but XLA's base matmuls already ran at ~76% of peak, and the kernel's structural
costs were what the roofline left out: the per-projection forward re-read x once per
output tile, the backward's every-axis-"arbitrary" grid kept the dy_i and W_i blocks of
every projection of the group double-buffered and gave no megacore split, and under
`full` / `core_states` the forward kernels reran in the recompute as the XLA base matmuls
do. Tile tuning (larger tn, a deeper `pl.Buffered` W stream as tokamax `gmm_v2`) was tried
without a gain. Merged LoRA (above) took the kernel's place up to 4B; 9B trains unmerged.

### Prior art consulted

| Source (license) | What it does | Taken |
|---|---|---|
| JAX Pallas TPU `ops/tpu/matmul.py` and the [matmul guide](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html) (Apache-2.0) | tiled matmul, fp32 VMEM accumulator over the "arbitrary" k axis | the accumulator pattern; the guide's v5e tiles (bm 512, bk/bn 1024: ~91% of peak) as the scale of the defaults |
| JAX `ops/tpu/megablox` `gmm` / `tgmm` (Apache-2.0) | grouped matmul over stacked [G, K, N] weights, its backward | the scalar-prefetch group index in the weight's index map (here the layer); tgmm's resident accumulator and zeroing both operands of a ragged row reduction |
| [tokamax](https://github.com/openxla/tokamax) `ragged_dot` / `gmm_v2` (Apache-2.0) | tile heuristic halving tiles to fit 0.75× scoped VMEM; `pl.Buffered(3)` weight stream | the fit-to-VMEM loop (rows halved); Buffered(3) left as a tuning step |
| MaxText / [qwix](https://github.com/google/qwix) LoRA (Apache-2.0) | LoRA as an extra low-rank `dot_general` (no fused kernel) | confirms nothing fused exists in that stack |
| [LoRAFusion](https://arxiv.org/abs/2510.00206) (EuroSys'26, CentML/lorafusion; no LICENSE file found) | GPU Triton: x·A separate, base GEMM accumulator seeded with s·xa·B; backward dy read once for dB and ds = dy·B (split-K atomics), dx seeded with ds·A | the ordering of the backward (t·Aᵀ into dx's accumulator); rejected: a separate x·A pass (on TPU it re-reads x at ~574/N of the GEMM time) and split-K atomics (none on TPU: resident VMEM accumulators instead) |
| [unsloth](https://github.com/unslothai/unsloth) `kernels/fast_lora.py` (Apache-2.0) | LoRA_QKV / LoRA_MLP autograd: dX = Σ dY_i·W_iᵀ + (dY_i·B_iᵀ)·A_iᵀ in place, xa recomputed | the grouped dx sum; rejected: recomputing xa (it is Rp·4 bytes per token) |
| vLLM / Punica / S-LoRA LoRA kernels (Apache-2.0) | multi-adapter inference (shrink / expand, bgmv / sgmv) | nothing (inference, many adapters) |
| torchtune `LoRALinear` | unfused | nothing |

No code was copied; the kernels are written for tjev's layouts.

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
- T is padded to a multiple of 128 (the lane width). Block sizes (`splash_tpu.block_sizes`):
  q and kv tiles of `compute.attention_splash_block` rows (default 512, MaxText's), or the
  largest of 512, 256, 128 dividing T; the kv compute sub-block
  `compute.attention_splash_compute` (0: the whole tile); `compute.attention_fused_bwd`
  replaces the dq and dkv kernels by one dkv kernel that also emits dq (per kv block, summed
  in XLA): one QKᵀ/softmax recompute less, a (T/block)× dq buffer more. Variants
  `splash-fusedbwd`, `splash-1024` (1024 tiles, 512 compute), `splash-1024-fusedbwd`,
  `splash-512-c256`; bench kernels `splash/fusedbwd`, `splash/1024c512`, … .
- The kernel is built eagerly under `jax.ensure_compile_time_eval` (its mask tables are
  concrete arrays) and cached by `_kernel` (`lru_cache`, 32 entries) keyed by padded length,
  group size, interpret flag and the `repr` of the abstract mesh at trace time: mask tables
  built under one mesh cannot be reused under another.
- `residual_checkpoint_name="attn_context"`: the name the `core` and `minimal` remat policies
  save.

The XLA path (`blocked_attention`) runs `jax.nn.dot_product_attention` one query block at a
time, each block reading keys up to its own end and, under remat, wrapped in its own
`jax.checkpoint`: O(block·T) peak memory.

### Tuned tiles (tune-jax, not merged)

A tune-jax autotuner (branch `perf/tune`, `tjev-tune1` on v6e-1) searched the kernel tiles
per size. Its picks: splash tiles 1024 with the fused backward, 1.57× the attention kernel
alone; GDN `gdn_heads_per_step` 32 at 4B, 1.03–1.05× the GDN kernel; LoRA tiles for the
fused LoRA kernel, 1.12–1.15×. The verification runs (`tjev-tune-verify`: +5–9% tokens/s at
0.8B / 2B / 4B, HBM unchanged, interpret-mode parity at rel err ≤ 2e-5) timed the picks on
the fused all-in setup, which no longer exists. Only the splash picks are in the planner
(`V6E1_TUNED`): the GDN pick overflows the 32 MiB scoped VMEM of mega2's forward with the
flat norm (39.2 MiB, `tjev-v75b` seed 1), which the verification arms did not have, and the
fused LoRA kernel was removed. 9B was not tuned. The autotuner itself was not merged; the
picks that remain are plain config values.

Whether the 32 MiB limit the failure reported is the session's (the preset asks for
`--xla_tpu_scoped_vmem_limit_kib=98304`, 96 MiB) or the kernel's own `vmem_limit_bytes` is
open; every run now logs its effective libtpu flags and scoped-VMEM limit at start
(`tjev.sharding.libtpu_settings`: stdout, `config.json`, W&B, `summary.json`), and the next
final run's line settles it.

**The rule for any future tuning.** A pick is adopted only if it is faster *and* fits the
memory budget of the exact production configuration: compiled ahead of time for v6e
(`scripts/aot_tpu.py`) on every sequence bucket, under the session's libtpu flags, with the
step's HBM peak reported next to its time. The verification arm must be the planner's
`v6e1` line itself (`tjev campaign plan final --hardware v6e1`, one job per size), not a
bench variant: the removed fused all-in arm is how a +5–9% was measured on a setup that
never shipped. No code from `perf/tune` is merged; a port runs its search on that line.

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
| `core_states` | `core` + `gdn_states`, `gdn_vnew` (the Pallas GDN kernel's residuals); the GDN core is not its own remat unit |
| `minimal` | `core` + `mlp_gate`, `mlp_up`, `mixer_out`, `lora_xa`, `gdn_qkv`, `gdn_z`, `gdn_ba`, `attn_qkv` |
| `minimal_states` | `minimal` + `gdn_states`, `gdn_vnew`, `gdn_tinv`; per bucket with `compute.remat_by_bucket` ([training.md](training.md#remat-per-bucket-computeremat_by_bucket)) |

`attn_context` exists only on the splash path. Whenever remat is on (except `core_states`),
the GDN core (conv → gates → l2norm → delta rule) is also wrapped in its own
`jax.checkpoint`, so its fp32 intermediates live only during its own backward.

How often the GDN forward kernel runs per microbatch (custom calls in the TPU-lowered
gradient of a super-block; `test_gdn_forward_kernel_runs_once_under_core_states`): 3 per
layer under `full` (the primal, the layer recompute, the core's own recompute for its
residuals), 2 under `core` and `minimal`, 1 under `core_states` and `none`. Under
`core_states` the layer policy saves the kernel residuals (`states` [B,H,N,Dk,Dv] and
`v_new` [B,H,N,C,Dv], fp32: ~1.5 KB per token and value head per GDN layer, e.g. 3.6 GB for
0.8B at 8192 tokens) and the XLA terms (conv, gates, `prepare`, `local`) are recomputed once
in the layer backward. XLA may still CSE some of the repeats of `full`/`core` at compile
time: the xprof occurrence count of `gdn_fused_fwd` / `gdn_recurrence_fwd` settles it.

Facts that shape these choices (checked on CPU with toy layers during development):

- A `custom_vjp` forward rule is rematerialised like any other code; its residuals are not kept
  automatically.
- A `checkpoint_name` inside a `custom_vjp` forward rule is honoured by the outer policy. That
  is how splash's `attn_context` is saved.
- Recompute is dead-code-eliminated per value, but a Pallas kernel is one opaque call. Saving
  `gdn_core_out` spares the layer recompute of the core, while the core's own backward still
  reruns the forward kernel for its residuals.
- With a frozen base, the backward of a base GEMM is dX only, so full remat spends about a
  third of the GEMM time recomputing. `core` and `minimal` trade that for memory.

Do not save the XLA chunk-local terms of the GDN core: tens of KB per token per layer.

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

Shapes: on TPU, `check_tpu_shapes` rejects (a `ValueError` naming the fallback) head dims
that are not multiples of 128 for the fused kernel (it slices heads out of lane-dense
`[C, heads·D]` blocks) and chunks that are not multiples of 8, before Mosaic sees them.
Interpret mode runs any shape exactly (the tiny test model's head dim 16 included).

Numerics on CPU: every GDN kernel is fp32-exact there (Mosaic's `high`/`highest` mean
nothing on CPU; `mixed` and `bf16` do round). In the parity tool's bf16 model any change of
an op's fp32 result, even an exact algorithm swap at 1e-7, moves bf16 roundings and the
whole gradient by ~2% (realistic tiny model: the XLA reference with the doubling inverse
instead of its triangular solve is 1.8e-2 off; the Pallas GDN paths 1.8e-2 to 2.1e-2). With
`tjev campaign parity --model-dtype float32` the same comparisons are 2.8e-6 (doubling),
3e-6 (split) and 9e-6 (fused): the ~2% is the bf16 model's own noise, not casts in the
kernels. `--trace-layers` prints the forward divergence per layer and per mixer output, to
find where two variants part.

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
  `kernels+core`, `kernels+minimal`, `split`, `xla` and the candidates of the kernel
  iteration (`bench.VARIANTS`, each also a `parity.VARIANTS` entry, so the probe measures
  speed and real-weights parity alike; `--variants` picks some). `best_variant` is the fastest finite
  variant whose HBM peak stays under 80% of the chip; `tjev campaign cost --bench` reads it.
  A variant that fails to compile or runs out of memory is recorded with its error and
  skipped.

## Performance model and roadmap

All numbers in this section are predictions from a per-op roofline analysis made before the
first TPU run, kept as the record of what each kernel was expected to buy. The measurements
that replaced them: throughput and MFU per variant in
[results.md](results.md#tpu-throughput-v6e-1-measured) (XLA chunked 13.0k → mega2 c128
blocked 41.8k → with merged LoRA and splash 1024 51.4k tokens/s at 0.8B; 9B 3.6k → 9.3k,
28% MFU), the per-scope profile in [profiling.md](profiling.md#the-9b-capture-tjev-smoke9b-v6e-1-rank-16-2048-token-microbatches),
and the per-iteration journey in [kernel-journey.md](kernel-journey.md).

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

The fused forward was predicted to save about 15% of the step; a fused backward too, about
36% in total. After the full fusion ladder below, predicted model MFU was 40% for 2B and 44%
for 4B. Measured: the v2 fused forward + backward is 2.2× the XLA path at 0.8B and 9B, the
megakernels add another 1.5× at 0.8B (1.2× at 9B), and the 9B step reaches 28% MFU; the
remaining time is the frozen base matmuls (54%, compute-bound), the GDN kernels, the
`core_states` residual traffic and LoRA.

### Roadmap, and what became of it

| Step | Status |
|---|---|
| 1. Fused GDN backward | done: `pallas_tpu_v2` (+2.2× over XLA chunked), then the megakernels `mega1` / `mega2` (gates, l2norm, GQA, the conv and SiLU inside; the XLA conv and its miscompile barriers gone) |
| 2. Blocked-16 inverse | done on the XLA side (`gdn_inverse=blocked`) and inside the v2 / mega kernels (`_diagonal_block_inverse`): +2% at chunk 128 |
| 3. Fused LoRA linear | built and measured, exact but slower than merged / unmerged on v6e-1, then removed ([Fused LoRA (removed)](#fused-lora-removed)); merged LoRA took its place up to 4B (+26 / +16 / +9%) |
| 4. Small fusions | conv1d + SiLU + mask inside the GDN kernel: done (mega2). The gated norm's relayout copies: removed by `gdn_norm=flat` (+3.8% at 4B) instead of a prologue fusion. swiglu into `down_proj`: not done |

The original spec of step 1, kept because the kernels implement it:

1. Fused GDN backward (`gdn_chunk_bwd`), as `gdn_impl=pallas_tpu_v2` (above). Same grid as
   the forward, chunks reversed, G in a `(heads, Dk, Dv)` fp32 scratch; reads q, k, v, the
   gates, the saved `states` (bf16 is enough, since the state-path dots are 1-pass) and T
   (fp32, saved by the forward); recomputes D, u/w and intra in VMEM. With X = Tᵀ·d_vnew,
   the identities (checked in fp64 during development)

   ```
   d_vβ   = X
   d_kbe  = −X Sᵀ
   dL     = strict_lower(−X v_newᵀ)     (replaces dT and the two T sandwiches)
   −wᵀ d_vnew = −kbeᵀ X                 (the backward needs neither w nor u)
   ```

   leave five Dv contractions (do·Sᵀ, X·Sᵀ, v_new·Gᵀ, do·v_newᵀ, X·v_newᵀ) plus the chunk-local
   q/k part; d_γ comes from row and column sums of dD ⊙ D plus the e^γ and carry terms
   (MaxText's closed form). Precision: `high` for the intra-chunk products, 1-pass bf16 for the
   state path (`mixed`; the default stays `high`, since `mixed` was never validated on its
   own). Predicted 0.44 ms forward and 0.88 ms backward per layer, against 2.40 / 3.99 ms
   unfused. The 64-wide doubling dots at C = 64 fill 1/16 of a v6e MXU, which is why the
   blocked inverse and chunk 128 followed.

## Profiling

Named scopes (`jax.named_scope`) put every op of the step under a name that xprof's op
profile and framework op stats group by. They survive remat (the recompute runs under
`…/rematted_computation/…`) and autodiff (backward ops under `transpose(jvp(…))`);
`tests/model/test_scopes.py` checks both on the compiled HLO. The capture protocol, the
headless xprof tools and how to read their tables (scope × phase, roofline, estimated vs
measured): [profiling.md](profiling.md).

| Scope | Where |
|---|---|
| `embed`, `final_norm`, `readout`, `loss` | token embedding, final RMSNorm, label logits, the slot losses |
| `layer0_gdn` … `layer2_gdn`, `layer3_attn` | a whole decoder layer, by its position in the super-block and its mixer (norms and MLP included) |
| `norm`, `mlp` | the pre-mixer / pre-MLP RMSNorms, the SwiGLU MLP |
| `base`, `lora` | the frozen projection matmul (`Linear.base`), the adapters |
| `<proj>/base`, `<proj>/lora/{A,B,merge}` | per projection (`q_proj` … `o_proj`, `in_proj_qkv` … `out_proj`, `gate_proj`, `up_proj`, `down_proj`): its base matmul, the adapter's x·A (shared by a group: `gate_proj+up_proj/lora/A`), ·B, and merged LoRA's W + s·AB |
| `residual` | the residual-stream adds (the next layer's input, which its remat saves) |
| `L00` … `L31` | the layer's number, only with `TJEV_UNROLL_LAYERS=1` (a profiling capture with the layer scan unrolled) |
| `gdn_proj`, `gdn_core`, `gdn_out` | GDN in-projections, the core (its own remat unit), gated norm + out_proj |
| `gdn_conv`, `gdn_gates`, `gdn_qk_norm`, `gdn_rule` | inside the core: conv + silu, β and g, GQA repeat + l2norm, the delta rule |
| `gdn_prepare`, `gdn_local`, `gdn_inverse`, `gdn_scan` | inside the rule: XLA terms of the Pallas paths, the (I+L)⁻¹ step, the chunked XLA scan |
| `attn_proj`, `attention_core`, `attn_out` | attention projections + q/k norm + RoPE, the attention op, the output gate + o_proj |

Debug dumps for a probe session, off by default: `tjev campaign bench --debug-dir DIR
--debug-variants split` logs the compiles of those variants with cache-miss explanations
(`DIR/compiles-<size>-<variant>.log`, `jax_log_compiles` + `jax_explain_cache_misses`),
dumps their XLA HLO (`DIR/xla/<size>-<variant>/`, per-compile `xla_dump_to`) and, for the
whole process, the Mosaic MLIR of every Pallas kernel (`DIR/mosaic`, the libtpu flag
`--xla_mosaic_dump_to` appended to `LIBTPU_INIT_ARGS` before JAX starts). The probe phase
runs it as a separate bench process when `PROBE_DEBUG=<variants>` is set, into
`reports/probe/debug`.
