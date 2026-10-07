#!/usr/bin/env bash
# Build mix-v7.5, mix-v7.5-rep1 and mix-v7.5c on a Colab CPU runtime (docs/data.md, mix-v7.5), then upload
# them to the private HF dataset repo. Run from the worktree root on the runtime, detached:
#
#   nohup bash scripts/build_mix_v75.sh > /content/v75.log 2>&1 &
#
# Inputs from $HF_REPO: data/mix-v7 (the base), data/jevbench, data/jev-fr-mt (the jev_fr_mt
# mix folder of scripts/fr_mt_mix.py, translated on a GPU by scripts/fr_mt_gpu.sh). Steps
# (STAGES, default all): seeds (gen_v7 real-text seeds), extend (`tjev data extend-v75` →
# mix-v7.5-pre, untiered, not uploaded), reshare (JEV_FR_MT_SHARE again on the pre mix, no
# rebuild), variants (derives of the pre mix: mix-v7.5 with the epoch tiers, mix-v7.5-rep1
# with tier A at 1 epoch and fresh generator rows, mix-v7.5c commercial-only), report
# (sampled data reports at 5,000 steps best_fit), upload (only if the repo is private;
# every mix.yaml after every file, mix-v7.5's last). Outputs: $W/data/mix-v7.5{,-rep1,c}.
set -euo pipefail
: "${HF_REPO:=tychot/tjev-data}" "${W:=/root/tjev-work}" "${STAGES:=all}" "${STEPS:=5000}"
: "${JEV_FR_MT_SHARE:=2.4}" "${REPORT_SAMPLE:=4000}"
want() { [[ $STAGES == all || " $STAGES " == *" $1 "* ]]; }
log() { echo "[build-v75] $(date +%H:%M:%S) $*"; }
mkdir -p "$W/models/Qwen3.5-0.8B" "$W/data" "$W/reports"
TOK=$W/models/Qwen3.5-0.8B
[[ -f $TOK/tokenizer.json ]] || hf download Qwen/Qwen3.5-0.8B tokenizer.json --local-dir "$TOK" > /dev/null
if [[ ! -f $W/data/mix-v7/manifest.json || ! -f $W/data/jev-fr-mt/manifest.json || -n ${REFRESH:-} ]]; then
  log "download mix-v7, jevbench and jev-fr-mt from $HF_REPO"
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$HF_REPO", repo_type="dataset", local_dir="$W",
                  allow_patterns=["data/mix-v7/*", "data/mix-v7/**", "data/jevbench/*",
                                  "data/jev-fr-mt/*", "data/jev-fr-mt/train/*",
                                  "data/jev-fr-mt/test/*", "data/jev-fr-mt/meta/*"])
PY
fi
PUB=$W/data/jevbench/public.jsonl
CAP=(--max-epochs 2 --steps "$STEPS" --packing best_fit --also-steps 2500,3000 --tokenizer "$TOK")
TIERS=(--tiers --min-epochs 1)
if want seeds; then
  log "gen_v7 real-text seeds (KALI, COLD, ShARC)"
  python -m tjev.data.sources.gen_v7.realtext build-seeds
fi
if want extend; then  # the untiered build every variant derives from (not uploaded)
  log "extend-v75 mix-v7 → mix-v7.5-pre (jev_fr_mt share $JEV_FR_MT_SHARE)"
  rm -rf "$W/data/mix-v7.5-pre"
  frmt=(); [[ -f $W/data/jev-fr-mt/manifest.json ]] && frmt=(--fr-mt "$W/data/jev-fr-mt")
  [[ ${#frmt[@]} -gt 0 ]] || log "WARNING: no data/jev-fr-mt: built without jev_fr_mt"
  tjev data extend-v75 "$W/data/mix-v7" "$W/data/mix-v7.5-pre" "${frmt[@]}" \
    --jevbench-public "$PUB" --jev-fr-mt-share "$JEV_FR_MT_SHARE" "${CAP[@]}"
fi
if want reshare; then  # other jev_fr_mt share on the built pre mix (no rebuild)
  log "reshare mix-v7.5-pre (jev_fr_mt share $JEV_FR_MT_SHARE)"
  rm -rf "$W/data/mix-v7.5-pre0" && mv "$W/data/mix-v7.5-pre" "$W/data/mix-v7.5-pre0"
  tjev data derive "$W/data/mix-v7.5-pre0" "$W/data/mix-v7.5-pre" --v75-shares "$JEV_FR_MT_SHARE" "${CAP[@]}"
  rm -rf "$W/data/mix-v7.5-pre0"
fi
if want variants; then
  log "derive mix-v7.5 (tier A ≤ 2 epochs, tier B ≤ 1, every source ≥ 1)"
  rm -rf "$W/data/mix-v7.5"
  tjev data derive "$W/data/mix-v7.5-pre" "$W/data/mix-v7.5" "${TIERS[@]}" "${CAP[@]}"
  log "derive mix-v7.5-rep1 (tier A ≤ 1 epoch, fresh generator rows)"
  rm -rf "$W/data/mix-v7.5-rep1"
  tjev data derive "$W/data/mix-v7.5-pre" "$W/data/mix-v7.5-rep1" "${TIERS[@]}" --tier-a-cap 1 \
    --fresh-generators --jevbench-public "$PUB" "${CAP[@]}"
  log "derive mix-v7.5c (commercial-only, tiered)"
  rm -rf "$W/data/mix-v7.5c"
  tjev data derive "$W/data/mix-v7.5-pre" "$W/data/mix-v7.5c" --drop-license-use non-commercial \
    "${TIERS[@]}" "${CAP[@]}"
fi
if want report; then
  for m in mix-v7.5 mix-v7.5c; do
    log "data report $m"
    python scripts/data_report.py --mix "$W/data/$m" --jevbench "$PUB" --tokenizer "$TOK" \
      --out "$W/reports/$m" --steps "$STEPS" --packing best_fit --sample "$REPORT_SAMPLE" --no-plots \
      || log "report $m FAILED"
    cp "$W/reports/$m/report.md" "$W/data/$m/report.md" 2> /dev/null || true
  done
fi
if want upload; then
  log "upload to $HF_REPO"
  python - <<PY
from huggingface_hub import HfApi
api = HfApi()
if not api.dataset_info("$HF_REPO").private:
    raise SystemExit("$HF_REPO is not private: not uploading")
# every file first, then the mix.yaml files: mix-v7.5c, mix-v7.5-rep1, and mix-v7.5's last
# (a TPU run waits for data/mix-v7.5/mix.yaml; rep1 must be complete before it appears)
mixes = ("mix-v7.5c", "mix-v7.5-rep1", "mix-v7.5")
for m in mixes:
    api.upload_folder(repo_id="$HF_REPO", repo_type="dataset", folder_path="$W/data/" + m,
                      path_in_repo="data/" + m, ignore_patterns=["mix.yaml"],
                      commit_message="data: " + m + " (files)")
    print("uploaded", m, "files", flush=True)
for m in mixes:
    api.upload_file(repo_id="$HF_REPO", repo_type="dataset",
                    path_or_fileobj="$W/data/" + m + "/mix.yaml",
                    path_in_repo="data/" + m + "/mix.yaml", commit_message="data: " + m)
    print("uploaded", m, "mix.yaml", flush=True)
PY
fi
log "done"
