# TPU kernel experiments

Candidate kernels that are not part of the training path yet. Each one must first be timed
on a TPU VM, and then pay off in a full train step (`tjev campaign bench`), before it moves
into `src/tjev/kernels/`. Tests live in `tests/experiments/`: interpret mode on CPU, and
lowering for TPU with `jax.export`.

| Candidate | State | Expected gain | Notes |
|---|---|---|---|
| Fused RMSNorm → projection (`norm_matmul.py`) | Written; tested in interpret mode and lowered for TPU; untimed | Small: saves one [M, K] bf16 round trip to HBM per norm | XLA often fuses the norm already. Measure with `bench_norm_matmul.py` before wiring it into `model/` |
| Fused GDN backward | Spec only (docs/kernels.md, *Roadmap*) | Large: about half of a 2B step is the GDN core, and the backward holds most of it | Chunk-local WY terms in VMEM with X = Tᵀ·d_vnew (no dw, dT or T sandwiches); multi-day kernel work |
| GatedRMSNorm (GDN output) fused into `out_proj` | Idea | Small, as above | The same shape of fusion, with the silu(z) gate |
| LoRA A/B fused into the base projection | Rejected on GPU (no gain), untested on TPU | Unclear | The LoRA products are narrow (rank 32–64): XLA overlaps them with the base matmul |

An earlier GPU-only experiment (tokamax's Triton RMSNorm) was slower than XLA; it says
nothing about TPU.

```bash
# on a TPU VM
python experiments/tpu/bench_norm_matmul.py --tokens 8192 --out reports/norm-matmul.json
JAX_PLATFORMS=tpu pytest tests/experiments -q
```
