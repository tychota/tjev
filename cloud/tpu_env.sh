# Process environment for tjev on a TPU host; sourced by cloud/tpu_campaign.sh (and handy in
# an ssh shell: `source ~/tjev-work/src/cloud/tpu_env.sh`).
#
# libtpu flags: MaxText's v6e dense-model set (benchmarks/xla_flags_library.py): the
# latency-hiding scheduler, data-parallel all-reduce optimisation (our LoRA gradients),
# async collective fusion, and a larger scoped VMEM for Pallas/XLA fusions. Unknown flags
# abort libtpu at startup, so phase s0 probes them one by one (probe_libtpu_flags) and keeps
# only the accepted ones in reports/libtpu.flags, which is what this file exports.
ROOT=${TJEV_ROOT:-$HOME/tjev-work}
export JAX_PLATFORMS=tpu
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/tjev/jax-compile  # shared by every run on the host
export TPU_STDERR_LOG_LEVEL=1 TPU_MIN_LOG_LEVEL=1 TF_CPP_MIN_LOG_LEVEL=1  # libtpu: warnings up
TJEV_LIBTPU_CANDIDATES=(
  --xla_tpu_enable_latency_hiding_scheduler=true
  --xla_tpu_enable_data_parallel_all_reduce_opt=true
  --xla_tpu_data_parallel_opt_different_sized_ops=true
  --xla_tpu_enable_async_collective_fusion=true
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true
  --xla_tpu_overlap_compute_collective_tc=true
  --xla_enable_async_all_gather=true
  --xla_tpu_scoped_vmem_limit_kib=98304
)
if [[ -s $ROOT/reports/libtpu.flags ]]; then
  LIBTPU_INIT_ARGS=$(cat "$ROOT/reports/libtpu.flags")
  export LIBTPU_INIT_ARGS
fi
if [[ -s $ROOT/src/CODE_VERSION ]]; then
  CODE_VERSION=$(cat "$ROOT/src/CODE_VERSION")
  export CODE_VERSION
fi
# W&B (the planner adds log.wandb=true to every job): online with the key uploaded next to
# the data (never on a command line), else offline (<run>/wandb, `wandb sync` after pulling)
export TJEV_WANDB=1 WANDB_SILENT=true
if [[ -s $ROOT/.wandb_key ]]; then
  WANDB_API_KEY=$(cat "$ROOT/.wandb_key")
  export WANDB_API_KEY TJEV_WANDB_MODE=online
else
  export TJEV_WANDB_MODE=offline
fi

probe_libtpu_flags() {  # keep each candidate libtpu accepts; writes reports/libtpu.flags
  local ok=() flag
  local test='import jax, jax.numpy as jnp; print(float(jax.jit(lambda x: (x @ x).sum())(jnp.ones((512, 512)))))'
  for flag in "${TJEV_LIBTPU_CANDIDATES[@]}"; do
    if LIBTPU_INIT_ARGS="${ok[*]} $flag" timeout 300 python -c "$test" > /dev/null 2>&1; then
      ok+=("$flag")
    else
      echo "[tpu_env] libtpu rejected $flag"
    fi
  done
  mkdir -p "$ROOT/reports"
  echo "${ok[*]}" > "$ROOT/reports/libtpu.flags"
  export LIBTPU_INIT_ARGS="${ok[*]}"
  echo "[tpu_env] LIBTPU_INIT_ARGS=$LIBTPU_INIT_ARGS"
}
