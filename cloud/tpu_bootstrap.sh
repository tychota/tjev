#!/usr/bin/env bash
# On the TPU host, after the code and data are unpacked under $TJEV_ROOT (cloud/session_run.py
# on Colab): a Python 3.12 venv with jax[tpu] and tjev, the
# Qwen3.5 snapshots, the held-out selection set. Idempotent: re-running skips what exists.
#
#   MODELS="0.8B 2B 4B" MIX=mix-v7.5 bash ~/tjev-work/src/cloud/tpu_bootstrap.sh
set -euo pipefail
: "${MODELS:=0.8B 2B 4B}" "${MIX:=mix-v7.5}" "${TJEV_ROOT:=$HOME/tjev-work}" "${TJEV_VENV:=$HOME/tjev-venv}"
SRC=$TJEV_ROOT/src
# TPU runtime images ship Python 3.10; tjev needs >= 3.12. uv brings its own Python.
if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=$HOME/.local/bin:$PATH
[[ -d $TJEV_VENV ]] || uv venv --python 3.12 "$TJEV_VENV"
# shellcheck disable=SC1091
source "$TJEV_VENV/bin/activate"
# the venv by name: Colab sets UV_SYSTEM_PYTHON=1, which sends `uv pip` to /usr even with
# the venv activated
uv pip install --python "$TJEV_VENV/bin/python" -e "$SRC[tpu,log]" pytest
python -c "import jax; print(jax.devices())"

mkdir -p "$TJEV_ROOT/models" "$TJEV_ROOT/logs" "$TJEV_ROOT/reports"
for m in $MODELS; do  # ~34 GB for 0.8B, 2B, 4B and 9B
  dir=$TJEV_ROOT/models/Qwen3.5-$m
  [[ -f $dir/config.json ]] || hf download "Qwen/Qwen3.5-$m" --local-dir "$dir"
done
held=$TJEV_ROOT/data/$MIX/heldout-select.jsonl
[[ -s $held ]] || tjev data heldout "$held"
df -h "$HOME" | tail -1
echo "[bootstrap] ok"
