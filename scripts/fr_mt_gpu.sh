#!/usr/bin/env bash
# jev_fr_mt on a Colab GPU runtime (docs/data.md, mix-v7.5): the 200-row sample again (pipes
# and code spans now protected), then every train row of plumb, decider_teacher and
# mghafiri_scenarios, then their held-out rows (validation / calibration / test, kept apart),
# all through CTranslate2 float16 + round trip + CometKiwi at the default thresholds.
# Run from the worktree root on the runtime, detached:
#
#   nohup bash scripts/fr_mt_gpu.sh > /content/frmt.log 2>&1 &
#
# Inputs come from the private HF repo ($HF_REPO, data/$MIX); outputs go to $OUT:
# sample/, train/shard-NN/, held/{validation,calibration,test}/ (each: translated.jsonl,
# rejected.jsonl, stats.json, tables.json). A shard with stats.json is done (resumable).
# STAGES (default "setup sample train held") selects the steps.
set -euo pipefail
: "${HF_REPO:=tychot/tjev-data}" "${MIX:=mix-v7}" "${OUT:=/content/frmt}" "${SHARD:=4000}"
: "${STAGES:=setup sample train held}" "${BATCH:=64}" "${KIWI_BATCH:=64}"
want() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo "[fr-mt-gpu] $(date +%H:%M:%S) $*"; }
V=/content/mt-venv
PY=$V/bin/python
SRC=(plumb decider_teacher mghafiri_scenarios)
mkdir -p "$OUT" /content/frmt-in
if want setup; then
  log "venv (unbabel-comet needs numpy < 2: not the tjev venv)"
  command -v uv > /dev/null || pip install -q uv
  [[ -x $PY ]] || uv venv -q --python 3.11 "$V"
  uv pip install -q --python "$PY" "unbabel-comet>=2.2" "numpy<2" ctranslate2 sentencepiece \
    sacremoses "transformers>=4.40,<5" huggingface_hub "setuptools<81" nvidia-cublas-cu12
  log "inputs from $HF_REPO data/$MIX"
  "$PY" - <<EOF
from huggingface_hub import hf_hub_download
import shutil
names = ["validation.jsonl", "calibration.jsonl"]
names += [f"train/{s}.jsonl" for s in "${SRC[*]}".split()]
names += [f"test/{s}.jsonl" for s in "${SRC[*]}".split()]
for n in names:
    p = hf_hub_download("$HF_REPO", f"data/$MIX/{n}", repo_type="dataset")
    shutil.copyfile(p, "/content/frmt-in/" + n.replace("/", "__"))
print("inputs ok", flush=True)
EOF
fi
# CTranslate2 4.x is built for CUDA 12 (cuBLAS 12 from the pip nvidia package; torch may be
# on CUDA 13): its libraries are found in the pip nvidia packages of the venv
NV=$("$PY" -c "import glob, os, site; print(':'.join(sorted(glob.glob(os.path.join(site.getsitepackages()[0], 'nvidia', '*', 'lib')))))")
export LD_LIBRARY_PATH="$NV:${LD_LIBRARY_PATH:-}"
run() {  # IN OUT
  [[ -f $2/stats.json ]] && { log "done already: $2"; return; }
  "$PY" scripts/fr_mt_sample.py run --in "$1" --out "$2" --backend ct2 --device cuda \
    --compute-type float16 --batch "$BATCH" --kiwi Unbabel/wmt22-cometkiwi-da \
    --kiwi-batch "$KIWI_BATCH" --gpus 1 > "$2.log" 2>&1 || { log "FAILED $2"; tail -n 30 "$2.log"; return 1; }
  "$PY" -c "import json; s = json.load(open('$2/stats.json')); print('[fr-mt-gpu]', '$2', s['accepted'], '/', s['rows'], s['by_base_source'], 'wall', s['wall_seconds'])"
}
TRAIN=(); for s in "${SRC[@]}"; do TRAIN+=("/content/frmt-in/train__$s.jsonl"); done
if want sample; then
  log "sample (200 rows, seed 0: the gen-v7 sample)"
  mkdir -p "$OUT"
  "$PY" scripts/fr_mt_sample.py sample --in "${TRAIN[@]}" --n 200 --seed 0 --out "$OUT/sample-in.jsonl"
  run "$OUT/sample-in.jsonl" "$OUT/sample"
fi
if want train; then
  log "train rows, shards of $SHARD"
  mkdir -p "$OUT/train"
  cat "${TRAIN[@]}" > "$OUT/train-all.jsonl"
  split -d -a 2 -l "$SHARD" "$OUT/train-all.jsonl" "$OUT/train/in-"
  for f in "$OUT"/train/in-*; do
    run "$f" "$OUT/train/shard-${f##*-}"
  done
fi
if want held; then
  log "held-out rows (validation / calibration / test of the three sources)"
  mkdir -p "$OUT/held"
  for part in validation calibration; do
    "$PY" - <<EOF
import json
src = set("${SRC[*]}".split())
with open("/content/frmt-in/$part.jsonl") as f, open("$OUT/held/$part-in.jsonl", "w") as g:
    for line in f:
        if json.loads(line).get("source") in src:
            g.write(line)
EOF
    run "$OUT/held/$part-in.jsonl" "$OUT/held/$part"
  done
  cat /content/frmt-in/test__*.jsonl > "$OUT/held/test-in.jsonl"
  run "$OUT/held/test-in.jsonl" "$OUT/held/test"
fi
log "done"
