# Sweep and training plan under the Colab v6e-1 budget

The sweep design: what to run with a ~20 TPU-hour budget, in what order, how to read the
runs, and how to spend the rest on final training. The planner it drives is
`src/tjev/campaign/plan.py` (its docstring has the decision rules); the run-health metrics
are `src/tjev/train/health.py`.

> **Status.** Stages 0–3 ran as designed (`tjev-sweep1`, W&B `tychota/tjev-sweep`): the
> rate brackets, the paired arms, the confirmation seeds, then the GDN-gate and fixed-init
> follow-ups. The numbers and what was adopted are in [results.md](results.md). The
> regulariser stage (§4, stage 4) then ran on 4B and 2B, including final-length 4B runs;
> what their checkpoints showed about calibration is in [calibration.md](calibration.md). This document keeps the design and
> the budget arithmetic; where a stage's outcome changed a rule, the rule below says so.

## 1. The budget

- **Compute.** About 80 compute units (CU) are left. A v6e-1 costs 4.08 CU/h, so the
  budget is **19.6 TPU-hours for everything**: sweep, checks, final training, post-training.
- **Throughput.** The plan was priced on v6e-1 with `gdn_impl=pallas_tpu_v2` (fused
  forward and backward) and `remat=core_states`. The planner's `v6e1` line has since moved
  to the mega2 kernel, merged LoRA ≤ 4B, best-fit packing and the flat norm
  ([training.md](training.md#integration-on-v6e1-what-was-adopted)), with every size
  measured (tjev-bench4); the hours in this document are the planning-time ones.

  | Size | Tokens/s at planning | Tokens/s now (`HARDWARE["v6e1"]["tokens"]`) | Hidden size | Layers |
  |---|---|---|---|---|
  | 0.8B | 28.3k (measured) | 52.7k | 1024 | 24 |
  | 2B | ~19k (interpolated) | 36.3k | 2048 | 24 |
  | 4B | ~11k (interpolated) | 14.1k | 2560 | 32 |
  | 9B | 7.8k (measured) | 9.2k | 4096 | 32 |

- **Pricing.** Use `--hardware v6e1` (one chip per run, these kernels, these speeds, the
  rate in CU):

  ```bash
  tjev campaign cost --hardware v6e1 --chips 1 --proxy 0.8B --sweep-steps 300 --lean ...
  ```

**What one run costs.** A run's cost is S·65,536 tokens at the size's speed, plus a full
eval per checkpoint (3,400 items × ~400 tokens, forward only, ≈ ⅓ of a training token) and
~300 s fixed. The `--lean` sweep economy drops the quick evals, evaluates every 100 steps,
skips post-training and kills unhealthy runs (§4).

| Run | Tokens | 0.8B | 2B | 4B | 9B |
|---|---|---|---|---|---|
| 200 steps, lean | 13.1M | 0.22 h | 0.29 h | 0.44 h | 0.58 h |
| 300 steps, lean | 19.7M | **0.29 h** (1.2 CU) | 0.39 h (1.6 CU) | 0.61 h | 0.83 h |
| 600 steps, lean | 39.3M | 0.50 h (2.0 CU) | 0.70 h (2.9 CU) | 1.14 h | 1.58 h |
| final, H = 1200 + branches at 300 / 600 | 90.4M | 1.3 h | 1.8 h | 3.2 h | 4.4 h |
| final, H = 600 + branches at 150 / 300 | 45.2M | 0.8 h | 1.05 h | 1.8 h | 2.5 h |

The sweep rows are at rank 32. The final rows are at each size's rank (4B and 9B: 64,
which slows the step by ~15%, see the rank axis in §4) and include ~0.1–0.2 h of
post-training each.

**The default plan does not fit.** At the planner's defaults (2B proxy, 600 steps, all arms,
plus the 2400-step transfer phase) it costs:

| Phase | Cost |
|---|---|
| Sweep (21 jobs, with the whole rank axis) | 16.7 h |
| Transfer | 11.7 h |
| Final (4B, 2B, 0.8B) | 5.9 h |
| **Total** | **34 h** (140 CU) |

The full "big sweep" (every arm, LoRA inits, two seeds per arm, at 2B) is ~30 h for the
sweep alone. Both are out of reach. Three cuts make the plan fit, and they are the
recommendation below:

- the proxy is 0.8B instead of 2B;
- sweep runs are 300 steps;
- there is no 2400-step transfer phase: the horizon comes from the final runs' own WSD
  cooldown branches.

## 2. What the statistics can resolve

**The noise.** Every quantity below comes from `plan.py` and the local mix-v3 runs
([results.md](results.md)):

- **Paired online loss.** Runs with the same seed see the same batches. Their difference
  in training NLL, over the last 20% of steps in 10-step windows, has an SE of **~0.010 for
  one seed at 600 steps** (12 windows).
  - At **300 steps** the tail is 6 windows, and the GPU runs measured **~0.014** on their
    cooldown windows (0.010 on 17 stable-phase windows).
- **Eval score.** Its seed sd is **0.026**. Unpaired, it needs (2.8 × 0.026 × √2 / δ)²
  seeds per arm: **~100 seeds to see 0.01**. It can only be a guard, never the decision.
- **FLOOR = 0.007.** A same-seed rerun differs by this much, so smaller effects are not
  actionable.

**The power.** The adoption rule is d < −max(FLOOR, 2·SE): one-sided, a false-positive
rate of ~2.3% per arm. Detecting a true effect δ with 80% power needs
n ≈ ((2 + 0.84)·σ / δ)² seed-pairs:

| Effect δ | 300 steps (σ ≈ 0.014) | 600 steps (σ ≈ 0.010) |
|---|---|---|
| 0.01 | 16 pairs = 4.6 h at 0.8B | 8 pairs = 4.0 h |
| 0.02 | 4 pairs = 1.2 h | 2 pairs = 1.0 h |
| 0.03 | 2 pairs = 0.6 h | 1 pair = 0.5 h |
| 0.04 | 1 pair = 0.3 h | 1 pair |

A 0.01 effect costs 4–5 TPU-hours per arm, a quarter of the whole budget, and sits barely
above FLOOR. The sweep is therefore designed to resolve **δ ≳ 0.03–0.04 with one seed**
and **δ ≳ 0.02–0.025 after a confirmation seed**. Effects of 0.01 are "not adopted; cannot
tell", and the defaults stay.

- **The rate.** The rate effects are large enough: a half-octave rate step changed the
  paired loss by 0.036 on GPU, so one seed brackets the rate.
- **Per CU, 600 steps resolve ~12% more than 300.** They are also closer to the final
  horizon. But they double the per-run cost and halve the number of arms that fit, so the
  plan uses 300 steps and spends the confirmation seeds on the winners only.
- **The eval guard is a coin flip at one seed.** The guard rule is "score not worse by more
  than FLOOR". If the paired eval difference had the unpaired sd (0.037), a truly neutral
  arm would fail it ~42% of the time. A same-seed pair is surely less noisy, but that is
  unmeasured. Read a guard veto at one seed as *inconclusive*, and send the arm to stage 3.
- **Multiple arms.** About 11 arms at a ~2.3% false-positive rate each gives a ~23% chance
  that at least one is a false win. Hence stage 3: a winner is adopted only once a second
  seed agrees (`combine` over the seeds).

## 3. Is 0.8B a good enough proxy?

Yes for the learning rate and the cheap arms, with one 4B check.

- **Width only.** 0.8B and 2B differ only in width: both have 24 layers, (8, 2) attention
  heads and 16 GDN heads. 0.8B is the cleanest width-only proxy for 2B.
- **The width rule moves the rate very little.** With lr ∝ d^−0.05, the rate changes by
  (2048/1024)^−0.05 = **0.966** from 0.8B to 2B, 0.955 to 4B and 0.933 to 9B. That is
  3–7%, against the grid's √2 = 41% spacing. The prior's sd of 0.1 on the exponent gives
  ±7% to 2B and ±14% to 9B, which is still well inside one grid step. O'Neill et al. fit
  p = 0.00 [0, 0.03] for Qwen LoRA. The proxy size barely matters for the rate.
- **It costs 1.5× less.** 28.3k against ~19k tokens/s: a 300-step run is 0.29 h instead of
  0.39 h.
- **4B and 9B differ in depth anyway.** They have 32 layers and twice the heads, which
  neither proxy covers, so 2B is not a better proxy for them. The 4B check in stage 3
  covers that gap.
- **The risks:**
  - 0.8B is the weakest model at zero-shot (0.53 against 0.78 for 4B), so it has more to
    learn. Effects that depend on how far the adapter must move may not transfer: the
    inits especially (PiSSA, LoRA-GA), and clipping.
  - A 0.8B sweep at 150 steps on mix-v1 was monotone in the rate. That run was too short,
    on an older mix, before the loss normalisation, so it settles nothing either way.
- **The fallback.** If the stage-3 4B check disagrees with the 0.8B curve by more than
  half an octave, the 4B and 9B rates follow the check, not the proxy.

## 4. The recommended reduced sweep (~9.1 h, ~37 CU)

Every sweep queue is planned with
`--hardware v6e1 --proxy 0.8B --sweep-steps 300 --lean`, and `fit` needs the same knobs.

> **`--lean`** adds `train.quick_eval_every=0 train.checkpoint_every=100
> train.health_kill=true` and `post=none` to every sweep job. That saves ~15% of a
> 300-step run. A divergent high-rate run stops at the kill rule (§6) instead of spending
> its remaining steps.

### Stage 0: before any sweep run (~0.5 h, 2 CU)

- **Kernel validation** on the session's libtpu: `PROBE_ONLY=validate`, the kernel tests
  and the fp32 and bf16 ground truth ([results.md](results.md), *TPU correctness*).
- **Zero-shot controls.** Re-measure 4B on TPU (expected 0.779 ± 0.02); 9B was unmeasured
  at planning time.
- **2B and 4B speed.** `tjev campaign bench`, so the interpolated speeds are replaced
  before the final queue is priced (done: the table in §1).

### Stage 1: the rate brackets (8 runs, 2.3 h, 9.5 CU)

```bash
tjev campaign plan sweep --stage lr --hardware v6e1 --proxy 0.8B --sweep-steps 300 --lean ...
```

- **AdamW** at 2.5e-5, 3.54e-5, 5e-5 and 7.07e-5 (seed 0), plus the centre rate on seed 1.
  The second seed gives the seed fixed effect and a first seed-sd estimate.
- **Muon** (Polar Express) at 2.5e-5, 3.54e-5 and 5e-5.

Then run `tjev campaign fit ...` and read §7.

*Outcome.* AdamW bracketed at 3.24e-5 (3.54e-5 the best grid run); Muon at 2.5e-5 tied
AdamW's best with its optimum at or below the grid's edge, so stage 3 extended it
downward. Two Muon runs were stopped near their end by the pooled spike rule, since fixed
(spikes are judged per sequence bucket).

- **The optimum at an edge.** If either curve is not bracketed (the optimum is at an edge
  of the grid, or the curvature is ≤ 0), add one run √2 beyond that edge before stage 2.
  This is the 0.6 h contingency.
- **The stage-2 centre** (`--centre`) is the grid rate nearest the fitted AdamW optimum.
  The arms must pair with a near-optimal run, or every arm that effectively changes the
  step size (rank, init, β2, clipping) looks better just by moving toward the optimum.

### Stage 2: paired arms at the centre (13 runs, ~4.2 h, ~17 CU)

```bash
tjev campaign plan sweep --stage arms --centre <stage-1 rate> --arms \
  zclip,pissa,eva,lora-ga,r16,r64,r128,r128-samelr,b2-0.99,noclip,b32k,gdn-bf16,fp32-storage ...
```

| Priority | Arm | Decides | Cost |
|---|---|---|---|
| 1 | `zclip` | z-score clipping: the known bump source (clip_norm 1 clips every step) | 0.29 h |
| 2 | `pissa`, `lora-ga`, `eva` | the LoRA init (`lora.init`, branch feat/lora-init) | ~1.0 h (3 runs + init passes) |
| 3 | `r16`, `r64`, `r128`, `r128-samelr` | the rank axis (below) | 1.32 h |
| 4 | `b2-0.99` | Adam β2 | 0.29 h |
| 5 | `noclip` | no clipping | 0.29 h |
| 6 | `b32k` | half batch, 2× steps at lr × 0.85 (same tokens) | 0.29 h |
| 7 | `gdn-bf16` (report) | one-pass bf16 delta rule: a speed option if neutral | 0.27 h |
| 8 | `fp32-storage` (report) | bf16 against fp32 storage | 0.41 h (1.6×) |

- **Init arms and rates.** The inits (`plan.INIT_ARMS`) run at the centre rate. The
  literature says PiSSA and LoRA-GA "want lower LRs" (arXiv 2602.04998), so an init that
  loses narrowly at the centre gets one more run at centre/√2 in stage 3 before it is
  rejected.
- *Outcome.* Clipping does not matter (`zclip`, `noclip` within ±0.0003); smaller batches
  are worse; the rank rule holds (same-rate controls worse by 0.02 at r64 and 0.07 at
  r128) and no rank beat 32 at the rsLoRA rate; β2 0.99, fp32 storage, merged LoRA and
  one-pass bf16 GDN neutral; PiSSA worse at the centre; EVA and LoRA-GA were killed
  (`loss_above_start`), which exposed a bf16 residual-kernel defect in the offset inits,
  since fixed ([training.md](training.md#lora-initialisations)).
- **What to drop first if the budget slips:** the report arms (fp32-storage, then
  gdn-bf16), then `noclip`, then `b32k`, then `r128-samelr`. Dropping the first four
  brings the sweep back to ~7.9 h.

#### The rank axis: 16, 32, 64, 128

The arms are `r16` (lr × √2), `r64` (lr/√2) and `r128` (lr/2), each at the rsLoRA 1/√r
rule, paired with the rank-32 centre. Each has a same-rate control: `r16-samelr`,
`r64-samelr` and `r128-samelr`.

- **The rank choice.** `fit` takes the adopted rank arm (scaled or same-rate) with the
  largest paired win; otherwise the rank stays 32 (`plan.rank_rule`).
  - The small sizes (0.8B, 2B) take that rank.
  - 4B and 9B keep at least 64: a lower rank is not extrapolated to the deeper models.
- **The rate rule.** If any same-rate control beats its scaled twin, the rate stays put
  across ranks; otherwise it is lr/√2 per doubling.
- **One control is enough: `r128-samelr`.** Against `r128` it is a full octave of rate, the
  largest lever the axis has. At the measured curvature (~0.036 per half-octave) the rule's
  prediction differs by roughly 0.05–0.1, resolvable at one seed. `r64-samelr` (half an octave,
  ~0.02) and `r16-samelr` (half an octave the other way) are below one-seed resolution, so
  both are deferred. A same-rate win at r128 would move the whole rule.
- **What to expect.** Rank barely matters in the literature: r32 is 0.003–0.006 nats worse
  than r64, and r128 gains < 0.001 (2609.01244). That is far below the ~0.03–0.04
  resolution (§2). The likely outcome is "no rank adopted", so the axis is mostly a test of
  the rate rule, plus a speed option at r16: a neutral r16 is reported, not adopted, since
  `fit` adopts only wins.

**The cost of a rank.** LoRA is 13% (0.8B) and 16% (9B) of the device time at rank 32
(xprof). Scaled linearly, step time ≈ 1 + share·(r/32 − 1), which `plan.rank_slowdown`
prices. That is an upper bound for small ranks, whose narrow matmuls are latency-bound.

| Rank | 0.8B step | 0.8B, 300-step run | 9B step |
|---|---|---|---|
| 16 | 0.935× | 0.28 h | 0.92× |
| 32 | 1× | 0.29 h | 1× |
| 64 | 1.13× | 0.32 h | 1.16× |
| 128 | 1.39× | 0.37 h | 1.48× |

**The memory of a rank.** The 9B adapters hold 2.70M parameters per unit of rank:

- 24 GDN layers × 86,080 plus 8 attention layers × 79,872, summed over (in + out) of every
  target.
- r128 is 346M parameters. In fp32, parameters, gradients and Adam's m and v take
  16 B/parameter: **5.5 GB**, plus ~1.4 GB of update temporaries, on top of the ~18 GB
  bf16 base.
- That fits the 32 GB of a v6e chip with less room for activations. r64 takes 2.8 GB and
  r32 1.4 GB.
- A 9B at r128 is not planned unless r128 wins on two seeds. Even then, 9B at r128 is
  1.48× the step time.

#### Rank × init: do not cross them

A full cross is 4 ranks × 4 inits = 16 runs (~5 h), which does not fit.

- **Stage 2** runs the inits at rank 32 and the ranks with the default init.
- **Stage 3.** If both a rank and an init win, one combined run (winning rank × winning
  init, seed 0, plus seed 1 if it wins) checks that they add up.

There is one interaction to expect. PiSSA puts the top-r singular directions of W into the
adapter, so its effect grows with r. EVA's rank allocation depends on r. LoRA-GA's
gradient SVD is rank-r. An init that loses at r32 can still win at a higher rank, so the
stage-3 combined run uses the higher of the winning rank and 64 when an init is adopted.

### Stage 3: confirmations and the 4B check (~2.0 h, ~8.3 CU)

**Confirmation seeds** (4 runs, 1.2 h). Each adopted or borderline arm gets seed 1. Rank
the arms by |d|/SE and take at most 3, plus Muon if its best run is within 2 SE of
AdamW's.

```bash
tjev campaign plan sweep --stage arms --arm-seeds 2 --arms <winners> --centre <rate> ...
```

The queue skips finished seed-0 runs. If the centre moved away from 3.54e-5, the planner
adds the seed-1 base run. `fit --arm-seeds 2` combines the seeds.

**The 4B rate check** (2 runs, 0.9 h). Two 4B runs, 200 steps each, seed 0, at the 4B
rate the fit predicts × 2^∓½ (a one-octave pair, x = log2 lr). Their paired online
difference d = lo − hi gives the 4B vertex on the 0.8B curvature c:

x* = x_mid − d / (2·c·Δx), with Δx = 1.

- **Inside the rule.** If x* is within ±½ octave of the predicted rate, keep the rule.
- **Outside it.** Otherwise move the 4B and 9B rates halfway toward x*, a shrunk update.
- **Naming.** Name these runs `check-4b-…` so `fit` does not mistake them for transfer
  runs.

*Outcome.* Muon's optimum is ~2e-5, tied with AdamW's best; the inits at ½ and ¼ of the
rate gained nothing (eval worse); merged LoRA and one-pass bf16 GDN neutral on two seeds
(merged adopted for 0.8B / 2B for speed; one-pass bf16 not, its gate gradient is 5% off).
A fourth round on seed 0 tested the GDN gates and the fixed inits: frozen gates were the
one win (−0.013 ± 0.006), now `plan.FINAL_OVERRIDES`; no initialisation beats the default.

### Stage 4: the regularisers (the `reg` stage), 4B and 2B

```bash
tjev campaign plan sweep --stage reg --hardware v6e1 --mix data/MIX ...        # REG_PLAN: 4B + 2B arms
tjev campaign plan sweep --stage reg --hardware v6e1 --sizes 4B ...            # 4B only
tjev campaign plan sweep --stage reg --hardware v6e1 --sizes 2B --horizon 2500 \
    --arms rank16-cc0,rank16-cc0.1-q0.3,rank16-cc0.3-q0.6,rank16-cc1.0-q0.6,nocentre ...  # CogCalib, final length
```

Against the over-confidence the 4B long runs show after ~900 steps
([calibration.md](calibration.md)). Every run is a final run of its size (fit's recipe at
the reg horizon, the frozen gates) cut to `REG_STEPS` = 1500 steps (`--horizon` overrides
it), a full eval every 100 steps, every 300th checkpoint kept (`train.keep_every=300`, for
JevBench on more than the selected and last steps), health kill on, no post-training
(W&B group `CAMPAIGN/sweep/reg`). Every arm is paired with its own size's centre
(`sweep-reg-<size>-centre-s0`: seed 0, the same batches, the paired online loss;
`--arms …,nocentre` skips the centre when it already ran). The centres also track the EMA
of their adapters (`optim.ema_decay=0.999`): the EMA does not change the training, so each
centre's raw and EMA evals are the EMA pair, for the price of a second eval. Past 1500
steps (the final-length tests below) every arm tracks it too.

**Final-length tests** (`--horizon` past 1500): the run is a proper final, with
`post=full` at the end (calibration, JevBench, the per-item dumps), checkpoints and full
evals every 200 steps and every 50 in the cooldown (quick evals every 25), every one kept
from half the run on (`train.keep_every=200 train.keep_from=H/2`, the core sources' second
epoch) and through the cooldown (`train.keep_tail`). The
`rank16` and `rank16-cogcalib` arms also get a 1-epoch WSD cooldown branch to H/2 off
their own stable phase (branch point 0.8·H/2). A branch starts at its parent's loss and
hovers there, so branches run with `train.health_kill=false` (`loss_above_start` killed
one at its first steps). The `@rep1` arms train on the sibling mix `<mix>-rep1` (the same
held-out files, the core sources capped at one epoch, fresh generator rows in place of the
second epoch), paired with the same arm on `<mix>`.

- **4B** gets the levers aimed at its own over-confidence, measured where it happens:
  `wd0.1`, `floor0.1`.
- **2B** (2.6× faster) screens the rest: `ls0.1`, `ls0.1-gen`, `kl0.1`, `hard`, `drop0.05`,
  `rank16`, `cool0.1`.
- **Rates.** fit's own prediction at 1500 steps (the proxy's rate, then the width, rank,
  batch and horizon rules): before any sweep run 4B 2.4e-5 and 2B 3.5e-5 (the 2B preset's
  3.6e-5 at 1200 steps × 1.25^-0.15). `rank16` moves by the rank rule (lr/√2 per doubling):
  2B 32 → 16, 4.95e-5. A fitted sweep moves them all.

| Arm | Overrides |
|---|---|
| `ls0.1` (`ls0.05`) | `loss.label_smoothing` |
| `ls0.1-gen` | the same, on the generators only (`loss.smooth_sources=[block:generators]`) |
| `kl0.1` (`kl0.1-replay`) | `loss.kl_base_weight=0.1`, every row (the replay rows: `loss.kl_sources=[block:replay]`); +1/3 step |
| `hard` | `loss.hard_targets=true`: the control of the soft-target CE |
| `wd0.1` (`wd0.05`) | `optim.weight_decay` (decoupled AdamW, the LoRA factors) |
| `floor0.1` | `optim.final_lr_fraction=0.1` (vs 0.02) |
| `cool0.1` | `optim.decay_fraction=0.1` (vs 0.2) |
| `rank16` (`rank32`) | `lora.rank=16` at the rank rule's rate (lr/√2 per doubling: 4B 64 → 16 lr × 2, 2B 32 → 16 lr × √2) |
| `rank16-lr0.5`, `-lr0.7`, `-lr0.85`, `-lr1.4` | rank 16 around its rank-rule rate (never tuned at r16: the rsLoRA rule doubles it from r64, LoRA Without Regret finds the rate nearly rank-free under α/r) |
| `rank16-h3000` | rank 16 at twice the horizon, the rate × 2^−0.15 (the horizon rule) |
| `rank16-rep1` | rank 16 on the `-rep1` sibling mix (one epoch of the core sources), paired with `rank16` |
| `rank16-cogcalib` | CogCalib's gated entropy penalty at w = 0.1, q = 0.3, at rank 16 ([training.md](training.md#calibration-losses)); its term is ~0.1% of the loss on mix-v7.5 4B; at a final length it also gets the 1-epoch cooldown branch |
| `rank16-cc0`, `rank16-cc0.1-q0.3`, `rank16-cc0.3-q0.6`, `rank16-cc1.0-q0.6` | CogCalib's strength and gate: off, the decided 0.1 / q 0.3, 0.3 and 1.0 on the lowest 60% of NLL; all rank 16 at the same rate, the EMA tracked, run together with `nocentre` (paired with each other), no branch |
| `rank16-cc0.1-q0.6`, `rank16-cc0.2-q0.6`, `rank16-cc0.3-q0.3`, `rank16-cc0.5-q0.6`, `rank16-cc0.3-q0.8`, `rank16-cc0.3-q0.6-seed1` | the CogCalib grid around the 2B sweep's best arm (0.3 / q 0.6): each knob alone, its neighbours, and a second seed |
| `rank16-cc0.1-q0.6-seed1`, `rank16-cc0.2-q0.6-seed1`, `rank16-cc0.1-q0.6-np` | the grid's two finalists on seed 1, and the first on the natural-prior sibling mix `<mix>-np` ([data.md](data.md#the-current-mix-mix-v75)) |
| `rank16-cc0.1-q0.6-b2cool`, `rank16-cc0.1-q0.6-loraplus8` | optimizer levers on that arm: Adam's β2 raised to 0.9999 in the cooldown (`optim.cooldown_b2`), LoRA+ with the B factors at 8× the rate (`optim.loraplus_ratio`) |
| `rank16-cc0.3-q0.6-base` | CogCalib's paper gate (`loss.cogcalib_gate=base`): the base model's NLL decides the known rows, one extra base forward per microbatch |
| `rank16-cogcalib-rep1`, `rank16-cogcalib-rep1-lr5k` | the same on `-rep1`; `-lr5k` at a 2500-step horizon with the 5000-step rate, paired with `rank16-cogcalib`'s 1-epoch cooldown branch |
| `rank16-teacher` | Kev's gated teacher targets, λ = 0.5, gate 0.6 (no mix-v6 row passes the gate: it needs a mix that keeps decider's disagreeing questions) |
| `wd1.0`, `wd5`, `wd10` | decoupled decay that bites: at the 2e-5 rates wd 0.1 shrinks the adapters by ≤ 0.4% over 1500 steps (‖B‖ −3% on 4B) and wd 1.0 moves ‖B‖ by −1% on 2B; wd 10 decays by up to e^−0.5 and hurt the cooldown (A −15%, B −6% by step 500); 5 is the middle |
| `nogdn` | no LoRA on the GDN projections (attention + MLP only, arXiv 2604.22127's advice for hybrids) |
| `drop0.05` | `lora.dropout=0.05` (unmerged LoRA: 9% slower at 4B, 16% at 2B) |

`REG_PLAN` puts `wd0.1`, `floor0.1` on 4B and `ls0.1`, `ls0.1-gen`, `kl0.1`, `hard`,
`drop0.05`, `rank16`, `cool0.1` on 2B; `--arms` puts any catalogue arm on every planned
size.

Cost (the planner's model: 4B 14,120 and 2B 36,250 tokens/s, 4.08 CU/h, one chip):

| Run | Size | Steps | lr | Hours | CU |
|---|---|---|---|---|---|
| sweep-reg-4b-centre-s0 (+EMA) | 4B | 1500 | 2.4e-5 | 2.57 | 10.5 |
| sweep-reg-4b-wd0.1-s0 | 4B | 1500 | 2.4e-5 | 2.44 | 10.0 |
| sweep-reg-4b-floor0.1-s0 | 4B | 1500 | 2.4e-5 | 2.44 | 10.0 |
| sweep-reg-2b-centre-s0 (+EMA) | 2B | 1500 | 3.5e-5 | 0.94 | 3.8 |
| sweep-reg-2b-ls0.1-s0, -ls0.1-gen, -hard, -cool0.1 | 2B | 1500 | 3.5e-5 | 0.89 each | 3.6 each |
| sweep-reg-2b-kl0.1-s0 | 2B | 1500 | 3.5e-5 | 1.14 | 4.7 |
| sweep-reg-2b-drop0.05-s0 | 2B | 1500 | 3.5e-5 | 1.01 | 4.1 |
| sweep-reg-2b-rank16-s0 | 2B | 1500 | 4.95e-5 | 0.84 | 3.4 |
| **4B / 2B / total** | | | | 7.5 / 7.5 / 14.9 | 30.4 / 30.5 / 60.9 |

`fit` reports each arm's paired online and score differences against its size's centre
(`reg_effects`, keys `4B/wd0.1`, `2B/ls0.1`, …); adoption is by hand. The cost table above
is the planning-time estimate at the `REG_PLAN` arms; the final-length 4B runs
(`final-4b-h1500`, `final-4b-h3000`) and what their checkpoints showed about calibration
are in [calibration.md](calibration.md).

### Deferred (not in the ~9.1 h)

| Deferred | Why |
|---|---|
| Data mix (`data.mixture`, `mixture_alpha`) | Another mix means other batches, so the paired online loss is invalid. Only the eval score could judge it (~100 seeds for 0.01). Look at the per-source eval breakdowns of the finals instead |
| `b16k`, `r16-samelr`, `r64-samelr`, the rank × init cross, LoRA+, DoRA, weight decay, Muon β / RMS | Small expected effects, or below one-seed resolution (the half-octave same-rate controls), or already lost locally (LoRA+) |
| A true fp32 control (6-pass HIGHEST) | 4–6× the cost; fp32-storage is the cheap proxy |
| More seeds for 0.01 effects | 4–5 h per arm (§2) |
| The 2400-step transfer phase (γ) | 11 h; the final runs' WSD branches choose the horizon directly |
| A 2B rate check | 0.8B to 2B is a 3.5% rule; the 4B check is the one that matters |

### What would cut the cost further

- **Make the rate a traced argument.** The schedule is a closure today, so every grid run
  compiles its own executable: a few minutes × ~25 runs ≈ 1–2 h of the budget.
  `optax.inject_hyperparams` would put the rate in the optimizer state. Runs that differ
  only in rate would then share the persistent compile cache.
- **Measure the fixed cost** on the first stage-1 run (`perf/includes_compile` rows and the
  wall time before step 1), and re-price with it.

## 5. W&B layout

| Project | What | Who writes it |
|---|---|---|
| `tychota/tjev-sweep` | **every campaign training run**: sweep stages, checks, finals | `tjev campaign plan` (new default; `TJEV_WANDB_PROJECT` overrides) |
| `tychota/tjev-benchmarks` | speed (tokens/s, MFU, memory) | `tjev campaign bench` |
| `tychota/tjev-experiments` | correctness: kernel tests, parity, ground truth | the probes (`src/tjev/campaign/experiments.py`, `TJEV_WANDB_EXPERIMENTS_PROJECT`) |

- **Entity.** Set `WANDB_ENTITY=tychota` for `cloud/colab.sh`, which passes it on as
  `TJEV_WANDB_ENTITY`. Unset, the API key's default entity is used.
- **Groups.** One group per stage, so a group is one comparison:
  - `CAMPAIGN/sweep/lr`: stage 1, both rate brackets.
  - `CAMPAIGN/sweep/arms`: stages 2 and 3.
  - `CAMPAIGN/transfer`, `CAMPAIGN/final`.

  `CAMPAIGN` is `TJEV_WANDB_CAMPAIGN`, by default `tpu-YYYYMMDD`.
- **Tags** (`log.wandb_tags`, filterable in the runs table): the phase, the size, the
  hardware and chips, and `arm:<name|grid|muon-grid>`, `opt:<adamw|muon>`, `lr:<rate>`,
  `rank:<r>`, `init:<default|pissa|eva|lora-ga>`, `dtype:<bf16|fp32>`, `seed:<n>`.
  `job_type` is the phase.
- **Config for the sweep views.** Besides the full nested config, every run logs a flat
  `hp` block: `hp.size`, `hp.arm`, `hp.optimizer`, `hp.lr`, `hp.rank`, `hp.init`, `hp.dtype`,
  `hp.gdn_precision`, `hp.clip`, `hp.b2`, `hp.tokens_per_step`, `hp.steps` and `hp.seed`.
  - A parallel-coordinates panel on `hp.lr`, `hp.optimizer`, `hp.init`, `hp.rank` →
    `best.score` (the summary) shows the sweep at a glance.
  - Group the runs table by `hp.arm`.
- **Summary per run:** `best.*` (the selected step, its calibrated score, accuracy and
  ECE), plus `health.status` (0 ok, 1 warn, 2 kill, 3 killed), `health.spikes` and
  `health.rejected`.
- **Panels.** Chart these against `step`:
  - `learning/nll` and `learning/lr`;
  - `health/grad_norm` and `health/grad_norm_clipped`;
  - `health/clip_fraction`;
  - `health/update_ratio_a` and `health/update_ratio_b`;
  - `health/momentum_cos`;
  - `learning/spikes`.
- **Storage** (the free plan has 5 GB). Only scalars, config, summary and the console log
  are sent. Checkpoints, xprof traces and eval reports stay on disk and in the bucket or
  Drive mirror: the logger never calls `wandb.save` or logs artifacts, and
  `sync_tensorboard` is off.
  - A 300-step run with the health keys is ~60 scalars × 300 rows, well under 1 MB.
  - Check the usage page after stage 1 anyway.

## 6. How to tell a good run from a bad run

### The run-health metrics (logged every step)

They are computed inside the jitted step and reduced to scalars, under the `health/`
prefix in `metrics.jsonl` and W&B. The host-side loss-spike detector logs `learning/spikes`.

| Key | What |
|---|---|
| `grad_norm` (learning/), `grad_norm_clipped`, `clip_scale`, `clip_fraction` | the global norm before and after clipping, the scale applied (1 = inactive), the fraction of the row's steps that clipped |
| `zclip_mean`, `zclip_std`, `zclip_limit` | ZClip's EMA statistics and threshold (zscore mode) |
| `grad_norm_{gdn,attention,mlp}`, `grad_norm_{a,b}` | the gradient norm per module group and per factor |
| `update_norm_{a,b}`, `update_rms_{a,b}`, `param_norm_{a,b}`, `param_rms_{a,b}` | per factor |
| `update_ratio_{a,b}` (and `_max`) | ‖ΔW‖/‖W‖ per factor after the update: the relative step actually applied |
| `momentum_norm`, `momentum_cos` | the first moment (Adam's m, Muon's momentum) and its cosine with the incoming gradient |
| `adam_sqrt_v_mean`, `adam_eps_fraction` | AdamW: mean of √v̂, and the fraction of entries where √v̂ < ε (ε sets the step there) |
| `muon_ortho_norm` | Muon: the norm of the orthogonalised update (update / lr) |
| `lr` (learning/) | the rate the optimizer used, from its own step count. A rejected update does not advance it; the old host value was off by one step |
| `loss_median`, `loss_mad`, `status` | the spike detector's running median / σ̂ (1.4826·MAD over the last 50 losses), the verdict code |

**The overhead is negligible.** The diagnostics read the LoRA factors, gradients, updates
and moments once: about 6 × 134 MB for the 2B rank-32 adapters (33.6M parameters).
- **On TPU** that is ~0.5 ms at 1.6 TB/s, against a 3.4 s step (65k tokens at 19k
  tokens/s): **<0.02%**.
- **On CPU** the diagnostics alone take 18 ms (AdamW) or 9 ms (Muon) on that 2B-sized tree.
  The optimizer update itself takes 12 ms or 156 ms.
- **On the tiny CPU model**, the jitted step on one batch, timed 40 times with the
  diagnostics on and off interleaved, gives medians of AdamW 704.6 → 699.0 ms (−0.8%) and
  Muon 808.1 → 817.9 ms (+1.2%). The minima move by +1.3% and +0.9%. Both are inside the
  machine's noise: shorter repeats ranged from −8% to +8%.

`train.health_metrics=false` turns them off.

### Kill rules (`health.json`, `train.health_kill`)

The host-side monitor (`tjev.train.health.HealthMonitor`) evaluates these rules on every
logged step.

- **`health.json`.** The verdict is written at every eval or checkpoint and at the end
  (`train.health_report=true`, the default): `ok`, `warn`, `kill` or `killed`, with each
  fired rule's step, value and threshold, and summary statistics.
- **Kill.** With `train.health_kill=true` (on in `--lean` sweeps, off by default), a run
  whose verdict reaches `kill` stops at once. It exits 0, so the queue marks it done and
  does not retry it, and its status is `killed`.
- **Killed runs in the fit.** `fit` leaves killed runs out of every curve and arm and lists
  them under `killed`. For the rate curve, a killed run means "too high". If it was the top
  of the grid, the curve still has three rates.

| Rule | Level | Condition (window W = 20 steps) | Why |
|---|---|---|---|
| `non_finite` | kill | ≥ 3 rejected updates (non-finite loss or gradient) | the update is rejected on device; repeated rejections mean the run cannot continue |
| `loss_above_start` | kill | after warmup + W, the median of the last W losses > the median of the first W | training made the model worse than the base |
| `loss_regression` | warn / kill | gave back > 50% / > 75% of its best progress (rolling-median best against start), and by more than one per-step σ̂ | divergence after learning (the "loss bumps" of the local runs, at their worst) |
| `loss_spikes` | warn / kill | ≥ 3 spikes / spikes on > 5% of ≥ 100 steps | a spike is loss > median + 5·σ̂ of the last 50 (a spike-prone run; Wortsman et al. 2023) |
| `grad_blowup` | warn / kill | median grad norm of the last W > 3× / 10× the median of the first 2W post-warmup steps | the instability precursor in Wortsman et al.; ZClip's premise |
| `update_ratio_a` | warn / kill | median ‖ΔA‖/‖A‖ over W > 1e-2 / 5e-2 (after warmup) | the effective rate. Healthy AdamW at these rates is ~1e-3 (lr·O(1) per entry against an A RMS of 1/√fan_in ≈ 0.03); the ~1e-3 rule of thumb for pretraining |
| `update_ratio_a_dead` | warn | the same, < 1e-5, outside the cooldown | the rate does nothing |
| `update_ratio_b` | warn | > 0.2 after 100 steps | B grows from zero, so only late and only a warning |
| `clip_saturated` | warn | zscore clipping active on > 20% of the last 50 steps | ZClip should clip only outliers. Global clip_norm = 1 clips every step by design and is not judged |
| `momentum_against_gradient` | warn | cos(momentum, gradient) < 0 for ≥ 50 consecutive steps | oscillation, or a stale momentum (Muon β too high for the rate) |

- **Calibrate on stage 1.** Check that the centre runs (both seeds) end `ok`, and read
  their `health.json` stats against the thresholds. If a warning fires on a run whose
  paired loss is fine, loosen that rule (`HealthRules` in `tjev.train.health`) before
  stage 2. Kills are deliberately far from healthy values (5–10×), so a false kill should
  not happen. Two false kills did happen and changed the rules: the pooled spike rule
  stopped two healthy Muon runs near their end (spikes are now judged per sequence
  bucket), and `loss_above_start` killed a cooldown branch at its first steps (a branch
  starts at its parent's loss; branches now run with `train.health_kill=false`). The
  EVA / LoRA-GA kills were real.
- **After a resume** the monitor restarts with the resumed process: `health.json` then
  covers the steps since the resume.

**Not implemented.**

- **Logit growth** (Wortsman et al.: output-logit divergence, attention-logit growth). The
  base is frozen and the adapters start at zero, so it can only enter through the
  adapters. `update_ratio_b` and `grad_norm_attention` are the cheap stand-ins. A
  `max |label logit|` metric would be the direct one.
- **The per-projection ‖s·AB‖.** Too expensive per step; the global ‖A‖ and ‖B‖ RMS stand
  in for it.

### Ranking finished runs

1. **Exclude** killed runs. Look at `warn` runs before trusting them.
2. **The paired online loss against the centre run of the same seed** (`fit`'s
   `effects`). An arm is adopted when d < −max(FLOOR, 2·SE) and the eval score is not worse
   by more than FLOOR.
   - At one seed, a guard veto or 2 SE < |d| < 3 SE is *inconclusive*, and the arm goes to
     stage 3.
   - Report arms (fp32 storage, gdn-bf16) are judged by |d| < max(FLOOR, 2·SE):
     **neutral** means bf16 storage costs nothing, and the one-pass delta rule can be
     adopted for speed.
3. **The rate curve:** a quadratic in log2(lr) with seed fixed effects, on the online
   loss.
   - **Bracketed** means a curvature > 0 and a vertex within the grid ± ½ octave. Use the
     vertex.
   - **At the edge** (vertex outside, best grid rate at an endpoint, or curvature ≤ 0):
     extend the grid by one √2 step at that edge before deciding anything else.
   - **Flatness.** A curvature below ~0.02 per octave² means the rate barely matters at
     that resolution; prefer the lower rate (smoother, fewer bumps).
4. **Muon** replaces AdamW only on a clear paired win of its best grid run over AdamW's,
   confirmed on a second seed. The literature expects a tie for LoRA on an AdamW-pretrained
   model (sMuon, PoLoRA in [research/muon.md](research/muon.md)).
5. **Tie-breakers, reported but never deciding:** calibrated eval NLL (`eval_calibrated`
   score), ECE raw and calibrated, accuracy, the held-out generators, and fewer spikes or a
   lower clip fraction.

## 7. The full-training plan

The design below priced the finals. What ran: 4B finals at horizons 1500 and 3000
(`final-4b-h1500`, `final-4b-h3000`, with `--horizon`; their selected checkpoints and
calibration in [calibration.md](calibration.md)), then the final-length reg arms (§4).

### Transfer rules (`plan.fit`, from the 0.8B, 300-step sweep)

lr(size) = lr*₀.₈ · (d/1024)^−0.05 · rank_factor · (H/300)^−γ · 0.85^halvings

| Term | Rule |
|---|---|
| Width | P_WIDTH = 0.05: × 0.966 (2B), 0.955 (4B), 0.933 (9B) |
| Rank | The small sizes take the proxy's adopted rank (16 / 32 / 64 / 128); 4B / 9B take at least 64. lr/√2 per doubling from rank 32 (rsLoRA), unless a same-rate control beats its scaled twin |
| Horizon | γ = 0.15 (the prior: no transfer phase): × 0.901 at H = 600, × 0.812 at H = 1200 |
| Batch | × 0.85 per halving, if `b32k` won |
| Override | `--horizon H` sets the final horizon instead of the fitted one; `--sizes` picks the sizes |

For example, with lr*₀.₈ = 4.4e-5 the 4B at H = 1200 gets 4.4e-5 × 0.955 × 0.707 × 0.812
≈ **2.4e-5**, which matches its current preset (2.5e-5). The 4B check (stage 3) corrects
4B and 9B if it disagrees.

**WSD cooldown branches.** Each final is one long run to H with checkpoints every H/10.
Two branches start from its stable phase at 0.8·H/4 and 0.8·H/2 and cool down over the
last 20% of their own horizon. One run thus prices three horizons, and
`tjev campaign select` picks the best by the selection score (the held-out generators
included). That replaces γ and the 2400-step transfer phase.

**Epochs.** At H = 1200 mix-v3 is ~1.3 epochs. Past ~2 epochs LoRA overfits (2609.01244).
H = 2400 is not planned.

### What each size costs, and what fits

| Size | Horizon | Tokens | Cost | CU | Notes |
|---|---|---|---|---|---|
| 4B | 1200 (+ branches at 300 / 600) | 90M | 3.2 h | 13.1 | priority 1: the best zero-shot (0.779) |
| 9B | 600 (+ 150 / 300); 1200 if 4B selects h1200 and the budget allows | 45–90M | 2.5–4.4 h | 10.2–18.0 | priority 2: **go only if its zero-shot beats 4B by ≥ 0.03** |
| 2B | 600 (+ 150 / 300) | 45M | 1.05 h | 4.3 | priority 3 |
| 0.8B | 600 (+ 150 / 300) | 45M | 0.8 h | 3.3 | priority 4: the cheapest model, the weakest |

**The whole budget:**

| Item | TPU-hours | CU |
|---|---|---|
| Stage 0 (validation, zero-shot, bench) | 0.5 | 2.0 |
| Reduced sweep (stages 1–3 + contingency) | 9.1 | 37.1 |
| 4B final, H = 1200 (rank 64) | 3.2 | 13.1 |
| 9B final, H = 600 (rank 64) | 2.5 | 10.2 |
| 2B final, H = 600 | 1.05 | 4.3 |
| 0.8B final, H = 600 | 0.8 | 3.3 |
| **Subtotal** | **17.15** | **70.0** |
| Reserve: failures, reruns (9B at H = 1200 is +1.9 h: only if the sweep came in under) | 2.45 | 10.0 |
| **Total** | **19.6** | **80** |

- **9B's horizon follows 4B's.** If 4B selects its h300 or h600 branch, 9B runs at
  H = 600. If 4B selects h1200 and ≥ 2.5 h remain (the sweep came in under budget), 9B
  runs at H = 1200.
- **Queue order: 4B first, then 9B.** If 9B is a no-go (zero-shot not better), its 2.5 h
  go to the reserve or to a second 4B seed. Then 2B and 0.8B.
- **Colab sessions end at ~12 h.** The queue checkpoints and resumes, but keep the runs
  mirrored (`cloud/colab.sh` pulls them back between sessions).

### Go / no-go checklist before the final queue

1. **Kernels.** `PROBE_ONLY=validate` passes on this session's libtpu: the kernel tests and
   the fp32 / bf16 ground truth against the official model.
2. **Speed and memory.** `tjev campaign bench` has measured 2B and 4B, and
   `tjev campaign cost --hardware v6e1 --chips 1 --bench BENCH.json` prices the final queue.
   9B at microbatch 2048 with `core_states` shows `perf/hbm_peak_gb` < 30.
3. **Health calibration.** Both stage-1 centre runs end `ok`, and no kill rule fired on a
   run whose paired loss was fine.
4. **The fit is sane.**
   - The AdamW curve is bracketed (curvature > 0).
   - The two centre seeds differ by about the seed sd (≤ 0.03 on the score).
   - The paired SE is ~0.014 or better at 300 steps.
5. **Every adopted arm was confirmed on two seeds.** (No init won; the one adopted change
   is the frozen GDN gates, confirmed on seed 0 only and taken as `FINAL_OVERRIDES`.)
6. **The 4B check agrees** with the rule within ½ octave, or the 4B and 9B rates were
   moved.
7. **The 9B zero-shot** is measured and beats 4B by ≥ 0.03 accuracy, or 9B is dropped.
8. **Budget.** The remaining CU ≥ 1.15 × the priced final queue. If not, drop sizes in
   reverse priority.
9. **Data and identity.** The same mix (`MIX`, its `heldout-select.jsonl`) and the same
   template as the sweep. The finals' `train.health_kill` stays off: a final run is
   watched, not killed.
10. **W&B.** Project `tjev-sweep` (entity tychota), group `CAMPAIGN/final`, storage below
    4 GB, no artifacts.

## References

- ZClip: adaptive spike mitigation for LLM pre-training (arXiv 2504.02507). z-score
  clipping on an EMA of gradient norms, the `zscore` mode and the `clip_saturated` rule.
- Wortsman et al. 2023, "Small-scale proxies for large-scale Transformer training
  instabilities" (arXiv 2309.14322). Output-logit divergence and attention-logit growth at
  high learning rates, visible in small models. The ratio of update to parameter norms and
  the gradient-RMS signals behind `update_ratio_*` and `grad_blowup`; the √v̂ against ε
  effect behind `adam_eps_fraction`.
- Muon: Jordan et al. 2024 (kellerjordan.github.io/posts/muon); Moonlight, Liu et al. 2025
  (arXiv 2502.16982), the 0.2 update RMS; Polar Express, Amsel et al. 2025
  (arXiv 2505.16932); sMuon (arXiv 2608.14492) and PoLoRA (arXiv 2607.17620) on Muon for
  LoRA ([research/muon.md](research/muon.md),
  [research/lora_optimizers.md](research/lora_optimizers.md)).
- "LoRA Without Regret" (Thinking Machines 2025, thinkingmachines.ai/blog/lora): LoRA's
  optimal rate is ~10× full fine-tuning, and its rank behaviour. O'Neill et al.
  (arXiv 2609.01244): LoRA learning-rate transfer across Qwen3 sizes (p ≈ 0) and batch.
  rsLoRA (Kalajdzievski 2023, arXiv 2312.03732).
- LoRA inits: PiSSA (arXiv 2404.02948), EVA (arXiv 2410.07170), LoRA-GA
  (arXiv 2407.05000); within 1–2% once tuned and wanting lower rates (arXiv 2602.04998).
- WSD and cooldown branches: Hägele et al. 2024 (arXiv 2405.18392). Horizon exponent:
  Bjorck et al. (arXiv 2409.19913), Schaipp et al. (arXiv 2501.18965).
