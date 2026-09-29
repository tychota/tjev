# Muon-style optimizers for LoRA: a literature survey

Researched 2026-09-27. The settings and measured outcome for per-factor Muon in `tjev` are in [`muon.md`](muon.md): AdamW is the default, Muon (Polar Express) an option.

Tags: **[paper]** means the authors claim it. **[our inference]** means we reasoned it out and have not measured it.
Notation: our LoRA is `W_eff = W0 + s·A@B` with `A: [..., in, r]` and `B: [..., r, out]`. `B` starts at zero. `s = α/√r` (rsLoRA). The leading axes stack layers. `G = ∂L/∂W_eff`, so `g_A = s·G Bᵀ` and `g_B = s·Aᵀ G`. `msign(X) = U Vᵀ` is the polar factor, computed with Newton–Schulz (NS).

The central problem, in one line. Plain Muon applied to each factor, which is what `optax.contrib.muon` on `lora_a`/`lora_b` does, orthogonalises `A` and `B` separately. That ignores the fact that `(A R, R⁻¹ B)` gives the same `W` for every invertible `R`. The resulting step in `W` therefore depends on the arbitrary scale and basis of the factors, and it is not a spectral-norm steepest-descent step on `W`. Every paper below fixes this by orthogonalising in terms of the product `AB`, meaning the tangent space of the rank-r manifold.

---

## 1. Can Muon Fine-tune Adam-Pretrained Models? (ICML 2026 poster 64467)

- **Authors / venue.** Xingyu Qu, Peigeng Huang, Samuel Horváth (MBZUAI, Nanjing U.). ICML 2026. arXiv [2605.10468](https://arxiv.org/abs/2605.10468), [OpenReview NKKwTEYdAm](https://openreview.net/forum?id=NKKwTEYdAm). Code: [github.com/XingyuQu/muon-finetune](https://github.com/XingyuQu/muon-finetune).
- **Idea.** An empirical and theoretical study, not a new optimizer. Adam's implicit bias is toward minimum max-norm solutions, Muon's toward minimum spectral-norm solutions, so Muon-trained weights have a higher stable rank. Fine-tuning an Adam-pretrained model with Muon therefore creates an "optimizer mismatch". The harm grows with update strength. [paper]
- **Method used.** Plain Muon (NS on the momentum), applied per factor on both LoRA A and B. Adam handles the non-matrix parameters. The LR is shape-scaled, β = 0.95, r = 8, α = 16. The LR is swept separately for each method. [paper]
- **Results** [paper]:

  | Task / model | Full-Adam | Full-Muon | LoRA-Adam | LoRA-Muon |
  |---|---|---|---|---|
  | GLUE avg, T5-Base | 89.14 | 88.77 | 88.93 | 88.97 |
  | GSM8K, Llama-2-7B | 61.66 | 57.37 | 59.64 | 59.57 |
  | HumanEval, Llama-2-7B | 35.57 | 34.35 | 27.85 | 29.47 |
  | 6-task vision, CLIP ViT-B/32 | 86.55 | 86.05 | 84.17 | 84.48 |

  In full fine-tuning, Muon's distance from the pretrained weights is 5.6–7.4× Adam's. Under LoRA it is 0.2–0.8× Adam's. LoRA-Muon beats LoRA-Adam on math only at low rank (r ≤ 16), and it gets worse as rank grows. Variants tuned for Adam (rsLoRA, LoRA-One, PiSSA) do not carry over to Muon.
- **Cost.** Steps are 1.1–1.2× slower than Adam. Optimizer state is half of Adam's. [paper]
- **Limitations.** Muon-pretrained models were tested only up to 561M parameters. The theory uses linear regression. The size of the mismatch varies by task, and the causes are unclear. [paper]
- **For us.** This paper is the main reason AdamW stays the default. Under LoRA, Muon is roughly neutral against Adam. We use r = 32–64 with rsLoRA, which is the regime where the paper's gains fade. [our inference]

## 2. LoRA meets Riemannion: Muon Optimizer for Parametrization-independent Low-Rank Adapters (arXiv 2507.12142)

- **Authors / venue.** V. Bogachev, V. Aletov, A. Molozhavenko, D. Bobkov, V. Soboleva, A. Alanov, M. Rakhuba (HSE, MSU). **ICLR 2026**. [arXiv](https://arxiv.org/abs/2507.12142). Code: [github.com/Bogachevv/RiemanianFinetune](https://github.com/Bogachevv/RiemanianFinetune) (PyTorch).
- **Math** [paper]:
  - The adapter is a point on the fixed-rank manifold `M_r = {X : rank X = r}`, stored as `X = A_L G B_Rᵀ` where `A_L` and `B_R` have orthonormal columns and `G` is r×r.
  - The Riemannian gradient is `P_T(∇F)`. A tangent vector `ξ = [Ȧ A_L][B_R Ḃ]ᵀ`, with `ȦᵀA_L = 0`, has rank ≤ 2r.
  - Update: momentum `M_t = β·M̂_{t−1} + P_T(G_t)`, where the previous momentum is transported to the new tangent space. Then `M̃_t = P_T(Ortho_r(M_t))`, where `Ortho_r` sets the first 2r singular values to 1. This is Muon inside the tangent space.
  - Retraction: `X ← truncSVD_r(X − η M̃_t)`. Each step costs O((m+n)r² + r³) through QRs and a small SVD, and never forms the m×n matrix.
- **LOI init.** Choose the initial ΔW that maximises the norm of the projected gradient. The answer comes from the top-2r singular vectors of the full gradient `∇_W L`: `ΔW⁰ = α·U_{1:r} V_{r+1:2r}ᵀ`. It is computed by randomized SVD with backprop tricks, costs 2(q+1) backward passes, and takes about 0.25% of training time. [paper]
- **Results** [paper]: Llama-3-8B, r = 16, commonsense-170k-style training (~22k steps), average over 8 tasks. Adam-LoRA 87.1±0.6, DoRA 86.6±0.3, per-factor Muon 83.0±0.6, RPrecAdamW 86.8±0.4, **Riemannion 88.1±0.2**. Riemannion wins on all 8 tasks and has the lowest variance across seeds. On SD-2 DreamBooth it converges faster and scores better on CLIP-T/DINO. Ranks tested: 4, 8, 16.
- **Cost.** The optimizer step is O((m+n)r²+r³), but it needs QR, a small SVD and a truncated-SVD retraction on every layer at every step. The independent sMuon paper measured it at **~33% of step time at r = 64** (≈700 ms vs ≈12 ms for sMuon's optimizer step). [sMuon paper]
- **Limitations.** No convergence theory. `Ortho_r` is approximate: the singular values come out at 0.9–1.1. LOI needs its scale α tuned. The per-factor Muon baseline looks weakly tuned (−4 pts against Adam). [paper] / [our inference]
- **Fit for us.** Poor. It changes the adapter parametrisation to three orthonormal factors, needs SVD/QR every step (slow when batched, and awkward on TPU), and needs a gradient-SVD init pass. That init replaces our zero-B init, which would change the exported PEFT adapters.

## 3. LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold (arXiv 2606.12921)

- **Authors / venue.** Franz Louis Cesista, Katherine Crowson, Cédric Simal, Stella Biderman. arXiv preprint, 2026-06-11, 20 pp. [arXiv](https://arxiv.org/abs/2606.12921). **No code link.** Algorithm 1 and baselines (Spectron, simplified LoRA-RITE) are in App. B.5–B.6.
- **Math** [paper]. Their notation is `W = A Bᵀ` with A m×r and B n×r. They solve spectral-norm steepest descent on `M_r` by splitting the tangent step `ΔA Bᵀ + A ΔBᵀ` into two half-radius subproblems:

  `ΔA* = −(η/2)·msign(∇_A f · S_B^{-1/2}) · S_B^{-1/2}`, with `S_B = BᵀB`,
  `ΔB* = −(η/2)·msign(∇_B f · S_A^{-1/2}) · S_A^{-1/2}`, with `S_A = AᵀA`.

  Each half moves `W` by exactly `(η/2)·msign(G Q_B) Q_Bᵀ`, where `Q_B` is an orthonormal basis of B's column space. That makes the step on `W` invariant under `(A,B) → (AR, BR⁻ᵀ)` for any `R ∈ GL(r)`.
  - msign uses NS with Polar Express coefficients (~8 steps). `S^{-1/2}` uses a coupled NS inverse-root iteration (~7 steps). There is **no QR and no SVD**, only matmuls.
  - **Split weight decay:** `A ← √(1−λη)·A + ΔA/√(1−λη)`, and the same for B. This gives `W ← (1−λη)W + tangent step + O(η²)`, whereas naive per-factor weight decay decays W by (1−λη)².
  - No retraction beyond the factor update itself.
- **Cost.** (6+4T_o)(m+n)r² + (16T_r+4T_o)r³ FLOPs per pair. Persistent state is only the first moments, (m+n)r, which is half of LoRA-RITE's. [paper]
- **Results** [paper]. All experiments are toy-scale: TinyShakespeare character LM with a 2-layer, d=128 transformer, ~1M tokens, 6 seeds. The best dense LR (η = 0.1) carries over across rank (≥2), width, depth and factor rescaling ×1…×27, where the loss curves coincide. Rank-32 LoRA-Muon gets 1.776 val loss against 1.789 for dense Muon. Under gauge rebalancing LoRA-Muon's loss shifts by 5e-5, against 2e-2 for Spectron.
- **Hyperparameters.** β = 0.9, λ = 0.01, T_o = 8, T_r = 7, LR swept 0.01–1.0. [paper]
- **Limitations.** No downstream or LLM fine-tuning evaluation. Transfer of the other hyperparameters is assumed, not measured. [paper] In the independent sMuon benchmark, LoRA-Muon was the *weakest* optimizer on Qwen2.5-3B SFT (see §4). [sMuon paper]

## 4. Related work (context only)

- **sMuon, "Approximate Muon with Low-Rank Adapters"** (Anson, Houghton, Milsom; arXiv [2608.14492](https://arxiv.org/abs/2608.14492); no code link). It linearises Muon's problem and solves a least-squares projection onto the updates LoRA can express: `δA = −η B†msign(H)` and `δB = −η(I−BB†)msign(H)A†`, with the 2r-rank core msign computed at 2r×2r. It uses only matmuls, damped inverse roots, momentum transport and split weight decay. Overhead is ~0.8% at r = 64. [paper] **This is the evaluation closest to our setup:** 2¹⁶ tokens/batch (≈ our 65k), r = 16, α = 32, clip 1, LR swept and chosen by validation loss, and Muon LR scaled by `0.2·√(d_in·d_out/r)` to match AdamW RMS. Average of 7 commonsense tasks [paper]:

  | Model (AdamW-pretrained) | AdamW | per-factor Muon | LoRA-Muon | Riemannion | sMuon |
  |---|---|---|---|---|---|
  | Qwen2.5-3B | **74.5** | 73.6 | 72.6 | 72.7 | 73.2 |
  | Llama-3.2-3B | 62.6 | 62.8 | 61.6 | **63.9** | 62.4 |
  | DeepSeek-V2-Lite | **67.7** | 66.3 | 66.3 | 64.1 | 67.3 |

  Muon variants win only on the Muon-pretrained Moonlight-16B, taking 6 of 11 tasks. The authors say the results depend on model and eval. [paper]
- **PoLoRA** (Ghosh, Parshakova, Gower; arXiv [2607.17620](https://arxiv.org/abs/2607.17620); code [github.com/nikhilgsh/polora](https://github.com/nikhilgsh/polora)). It orthogonalises in terms of the product, with diagonal Kronecker curvature preconditioners `P` (out-side) and `Q` (in-side). These are EMAs of the second moments of the factor gradients (β₂ = 0.99). The steps are `D_A = C_B^{-1/2} msign(C_B^{-1/2} M̂_A Q^{-1/2}) Q^{-1/2}` and the analogue for B. Step size is `ρ = η/(‖A‖₂+‖B‖₂)`, using power iteration. NS uses 8 Gram-NS steps. [paper] It reaches tuned Adam's final **held-out loss in 1.2–1.7× fewer steps** on OLMo-2-1B, Llama-3.2-1B, Qwen2.5-1.5B and Llama-3-8B, training on code, math and Bengali Aya at **rank 256**. Overhead is ≤ 3% per step, and the optimal LR is stable across rank. Curvature and magnitude control each account for about half the gain over "Product Muon". It reports held-out loss only, with no downstream benchmarks. [paper]
- **LoRA-TSD** (Andriianov, Veprikov, Beznosikov; arXiv [2609.02734](https://arxiv.org/abs/2609.02734)). It takes a Muon step inside the tangent space, computed from the factor gradients, and uses a retraction up to 2.8× cheaper than truncated SVD. It gives the first global convergence guarantees for LoRA-Pro and LoRA-TSD, and claims wins on 6 benchmarks with Llama and Qwen. Code is said to be on GitHub. *We could read only the abstract, so these claims are unverified.* [paper, abstract only]

---

## 5. Comparison

| Method | What is optimised | Extra cost vs AdamW-LoRA | Reported gain vs AdamW-LoRA | Fits optax? | Risk for us |
|---|---|---|---|---|---|
| Per-factor Muon (**in `src/tjev/train/optim.py`**) | msign of each factor's momentum, not gauge-invariant | ~1.1–1.2× step, ½ state | ≈0 (Qu), −4 pts (Riemannion paper), −0.9 on Qwen2.5-3B (sMuon) | Yes (`optax.contrib.muon`) | Low cost, likely neutral to slightly negative |
| Riemannion | Tangent-space Muon on `M_r` plus SVD retraction plus LOI init | QR/SVD every step, ~33% step time at r=64 | +1.0 avg (Llama-3-8B CSR, r=16); −1.8 on Qwen2.5-3B (sMuon) | Poorly (new parametrisation, SVD, init pass) | High engineering cost, inconsistent gains |
| LoRA-Muon | Gauge-invariant spectral steepest descent on `AB` through Gram inverse roots; split WD | Matmul-only, r×r inverse roots, ½ state | Toy only; −1.9 on Qwen2.5-3B (sMuon) | Yes (needs `params`, pairs a/b) | Unvalidated at scale |
| sMuon | Least-squares projection of Muon(H) onto the expressible subspace | ~0.8% | −1.3…+0.1 (AdamW-pretrained); wins on Muon-pretrained | Yes (same shape as LoRA-Muon) | Neutral on AdamW-pretrained Qwen |
| PoLoRA | Product-aware msign plus diagonal Kronecker preconditioner plus spectral magnitude rule | ≤3% step, O(in+out) extra state/layer | 1.2–1.7× fewer steps to Adam's final held-out loss (r=256) | Yes (more state, power iteration) | Gains shown at r=256 and for loss only; unknown at r=32–64 and 1k steps |
| LoRA-TSD | Tangent-space Muon plus cheap retraction | "cheaper than SVD" | Claims wins on 6 benchmarks (unverified) | Probably | Unverified |

## 6. Recommendation

**Bottom line [our inference].** Keep AdamW as the default. None of these methods gives a reliable gain for an **AdamW-pretrained Qwen at batch 2¹⁶ and r = 16–64**. The one independent benchmark in exactly that regime (sMuon, Qwen2.5-3B) has AdamW winning. The cheap per-factor Muon option in `src/tjev/train/optim.py` is enough to "check Muon". Drop its priority below LR, α and λ(Brier) sweeps.

If we want one geometry-aware ablation, implement **one gauge-invariant product-space update**: the LoRA-Muon core, with PoLoRA's diagonal preconditioner as an optional flag. Reasons [our inference]:
- It is matmul-only, with r×r inverse roots. That jits cleanly on TPU, batched over the stacked layer axis.
- It keeps our A/B parametrisation and the PEFT export unchanged.
- It is the shared core of LoRA-Muon, sMuon and PoLoRA, so one implementation covers the family.
- Its most useful property for us is **LR transfer across r and s**, not accuracy. It could let a single LR sweep on 2B carry over to 4B, where r changes from 32 to 64.

Skip Riemannion: it needs SVD, QR and a new init.

### Sketch (optax `GradientTransformation`, stacked leaves)

```python
# a: [..., in, r], b: [..., r, out]; leading axes = stacked layers (batch axes).
# Pair leaves by parent path (same prefix, keys "lora_a"/"lora_b"); others go to adamw via optax.partition.


def msign(x, steps=5):  # polar factor, batched over leading axes
    x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + 1e-7)
    tr = x.shape[-2] > x.shape[-1]
    if tr:
        x = jnp.swapaxes(x, -1, -2)
    for a, b, c in NS_COEFFS[:steps]:  # e.g. tjev.train.optim.POLAR_EXPRESS_8
        g = x @ jnp.swapaxes(x, -1, -2)
        x = a * x + (b * g + c * g @ g) @ x
    return jnp.swapaxes(x, -1, -2) if tr else x


def inv_sqrt_psd(S, rel=1e-4, abs_=1e-12):  # [..., r, r], r <= 64: eigh is fine (or coupled NS)
    r = S.shape[-1]
    d = rel * jnp.trace(S, axis1=-2, axis2=-1)[..., None, None] / r + abs_
    w, V = jnp.linalg.eigh(S + d * jnp.eye(r))
    return (V * jax.lax.rsqrt(jnp.maximum(w, abs_))[..., None, :]) @ jnp.swapaxes(V, -1, -2)


def lora_muon(lr_sched, s, beta=0.9, wd=0.0, nesterov=True, precond=False):
    # momentum only (+ optional diag EMAs of size in/out for PoLoRA-style precond)
    def init(params):
        return dict(count=0, mu=jax.tree.map(jnp.zeros_like, params))

    def update(grads, state, params):  # params REQUIRED (Gram matrices of current A, B)
        mu = jax.tree.map(lambda m, g: beta * m + g, state["mu"], grads)
        eta = lr_sched(state["count"])
        out = {}
        for path in lora_pairs(params):  # static Python loop over module paths
            A, B = params[path]["lora_a"], params[path]["lora_b"]
            Ma, Mb = mu[path]["lora_a"], mu[path]["lora_b"]
            if nesterov:
                Ma, Mb = beta * Ma + grads[path]["lora_a"], beta * Mb + grads[path]["lora_b"]
            Ra = inv_sqrt_psd(jnp.swapaxes(A, -1, -2) @ A)  # (AᵀA)^-1/2   [..., r, r]
            Rb = inv_sqrt_psd(B @ jnp.swapaxes(B, -1, -2))  # (BBᵀ)^-1/2   [..., r, r]
            dA = msign(Ma @ Rb) @ Rb  # [..., in, r]
            dB = Ra @ msign(Ra @ Mb)  # [..., r, out]
            # W_eff = W0 + s·AB → divide by s so eta is a W-space spectral step
            step = -(eta / 2) / s
            k = jnp.sqrt(1.0 - wd * eta)  # split weight decay (LoRA-Muon eq. 19)
            out[path] = dict(lora_a=(k - 1) * A + step * dA / k, lora_b=(k - 1) * B + step * dB / k)
        return out, dict(count=state["count"] + 1, mu=mu)

    return optax.GradientTransformation(init, update)
```

Implementation notes [our inference]:
1. **Zero-init B.** At step 0, `g_A = s·G Bᵀ = 0`, so `dA = msign(0)=0` and only B moves. That is correct. From then on `Rb ~ 1/‖B‖` is large, so A takes large factor-space steps while `ΔW` stays at spectral norm η/2. W-space is fine, but the gauge drifts toward a large A and a tiny B, and our forward pass casts A and B to bf16. Use relative plus absolute damping as above, and monitor `‖A‖₂/‖B‖₂`. Optionally rebalance every ~50 steps with `(A,B) → (A·Ra·Σ^{1/2}…)`, which leaves W unchanged, and transform `mu` with the same R (`M_A R^{-T}`, `Rᵀ M_B`).
2. **LR scale.** A unit-spectral rank-r step has entry RMS ≈ √(r/(in·out)). To start near AdamW, set η ≈ `lr_adamw · 0.2·√(in·out/r)` per matrix, as sMuon does [paper]. Then sweep ×{0.5, 1, 2}. Compute this per leaf from the shapes, because GDN `in_proj_*` and MLP shapes differ a lot.
3. Clip (`optim.clip_mode`) *before* this transform, as `make_optimizer` already does. msign ignores gradient scale, so clipping matters only through momentum mixing.
4. Cost: for 2B, r = 32, the work is O(L·(in+out)·r²) per step, negligible against the ~65k-token forward and backward pass. Check this with a step profile.
5. Test `W_eff`-level invariance. For a random `R`, the update on `A@B` from `(A,B)` and from `(AR, R⁻¹B)` must match to about 1e-4.

### Run plan and metrics
- **Arms:** AdamW (baseline) / per-factor Muon (existing) / LoRA-Muon (above) / optionally LoRA-Muon+precond. Use 2B, r = 32, the same WSD, and 1000 steps. Sweep the LR at 3 points per arm and use **≥ 2 seeds** (3 preferred, since expected differences are about the size of the noise).
- **Primary:** held-out NLL on answer-letter logits and ECE (15-bin plus adaptive). Brier.
- **Secondary:** accuracy per question type and per language. Steps needed to reach AdamW's final held-out NLL (PoLoRA's metric). Held-out NLL on general or zero-shot control text, to measure forgetting. Per-layer-type (attn / GDN / MLP) `‖s·AB‖_F` and stable rank. Step time and memory.
- **Adoption rule.** Adopt only if held-out NLL **and** ECE both improve across seeds (the rule in force is in [`muon.md`](muon.md#decision-rule)). For the gauge-invariant arm, also treat "the 2B-optimal LR also works for 4B" as a reason to keep it even if the result is a tie.

### Expected effect and interaction with calibration and small data [our inference]
- **Accuracy/NLL:** expect a tie within ±0.5 pt or ±1% NLL. PoLoRA's 1.2–1.7× faster steps might show up in our short 500–1200-step runs as a slightly lower NLL at fixed steps. Those gains were measured at r = 256, so they may not hold at r = 32–64.
- **Calibration:** msign flattens the singular spectrum of each step. That puts more update mass on weak directions than Adam does, which can sharpen logits on rare patterns and increase over-confidence. On the other hand, Qu et al. find LoRA-Muon moves *less* far from the pretrained weights than LoRA-Adam, which should preserve calibration. The sign is unknown. This is why ECE is a co-primary metric, and why post-hoc temperature scaling should be run on each arm (compare ECE before and after temperature).
- **Small data / noisy gradients:** our loss depends on a few answer-letter positions per sample, so each step's gradient is low-rank and noisy. msign sets the weak directions of the momentum's top-2r subspace to the same size as the strong ones, which amplifies noise. Momentum β ≥ 0.9 matters, and weight decay (split) acts as the regulariser. Expect Muon arms to prefer a *lower* LR than the RMS-matched starting point. Qu et al. make the same point. [paper]
- **rsLoRA:** Qu et al. find that rsLoRA and similar variants tuned for Adam do not transfer to Muon [paper]. With the gauge-invariant update, `s` only rescales η, and the `/s` fix above cancels that. With per-factor Muon, `s = α/√r` changes the effective W-step between r = 32 and r = 64, so sweep the LR separately for each size.
