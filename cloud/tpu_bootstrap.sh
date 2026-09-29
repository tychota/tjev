#!/usr/bin/env bash
# On the TPU host, after the code and data are unpacked under $TJEV_ROOT (cloud/tpu.sh push,
# or cloud/session_run.py on Kaggle / Colab): a Python 3.12 venv with jax[tpu] and tjev, the
# Qwen3.5 snapshots, the held-out selection set. Idempotent: re-running skips what exists.
#
#   MODELS="0.8B 2B 4B" MIX=mix-v3 bash ~/tjev-work/src/cloud/tpu_bootstrap.sh
set -euo pipefail
: "${MODELS:=0.8B 2B 4B}" "${MIX:=mix-v3}" "${TJEV_ROOT:=$HOME/tjev-work}" "${TJEV_VENV:=$HOME/tjev-venv}"
SRC=$TJEV_ROOT/src
# TPU runtime images ship Python 3.10; tjev needs >= 3.12. uv brings its own Python.
if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=$HOME/.local/bin:$PATH
[[ -d $TJEV_VENV ]] || uv venv --python 3.12 "$TJEV_VENV"
# shellcheck disable=SC1091
source "$TJEV_VENV/bin/activate"
uv pip install -e "$SRC[tpu,log]" pytest
python -c "import jax; print(jax.devices())"

mkdir -p "$TJEV_ROOT/models" "$TJEV_ROOT/logs" "$TJEV_ROOT/reports"
for m in $MODELS; do  # GCP -> HF is fast; ~34 GB for 0.8B, 2B, 4B and 9B
  dir=$TJEV_ROOT/models/Qwen3.5-$m
  [[ -f $dir/config.json ]] || hf download "Qwen/Qwen3.5-$m" --local-dir "$dir"
done
held=$TJEV_ROOT/data/$MIX/heldout-select.jsonl
[[ -s $held ]] || tjev data heldout "$held"
df -h "$HOME" | tail -1
echo "[bootstrap] ok"
