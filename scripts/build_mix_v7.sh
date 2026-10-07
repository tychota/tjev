#!/usr/bin/env bash
# Build mix-v7 and mix-v7c on a Colab CPU runtime (docs/data.md, mix-v7), then upload both to
# the private HF dataset repo. Run from the worktree root on the runtime, detached:
#
#   nohup bash scripts/build_mix_v7.sh > /content/v7.log 2>&1 &
#
# Steps: the Qwen3.5 tokenizer, mix-v6 + the JevBench public items from $HF_REPO, `tjev data
# extend` (mix-v7), `tjev data derive --drop-license-use non-commercial` (mix-v7c), the data
# reports (REPORT_STEPS 1500 under the default bucket packing: the paired TPU run), the
# upload (only if the repo is private; each mix.yaml last). STAGES (default "all") can skip work:
# "extend derive report upload" or a subset ("reshare": re-apply the shares of the code
# to a built mix-v7). Outputs: $W/data/mix-v7{,c}, $W/reports/.
set -euo pipefail
: "${HF_REPO:=tychot/tjev-data}" "${W:=/root/tjev-work}" "${STAGES:=all}"
: "${STEPS:=5000}" "${SOURCES:=}" "${REPORT_STEPS:=1500}" "${REPORT_PACKING:=bucket}" "${REPORT_SAMPLE:=4000}"
want() { [[ $STAGES == all || " $STAGES " == *" $1 "* ]]; }
log() { echo "[build-v7] $(date +%H:%M:%S) $*"; }
mkdir -p "$W/models/Qwen3.5-0.8B" "$W/data" "$W/reports"
TOK=$W/models/Qwen3.5-0.8B
[[ -f $TOK/tokenizer.json ]] || hf download Qwen/Qwen3.5-0.8B tokenizer.json --local-dir "$TOK" > /dev/null
if [[ ! -f $W/data/mix-v6/manifest.json ]]; then
  log "download mix-v6 and jevbench from $HF_REPO"
  python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("$HF_REPO", repo_type="dataset", local_dir="$W",
                  allow_patterns=["data/mix-v6/*", "data/mix-v6/**", "data/jevbench/*"])
EOF
fi
PUB=$W/data/jevbench/public.jsonl
if want extend; then
  log "extend mix-v6 → mix-v7"
  rm -rf "$W/data/mix-v7"
  tjev data extend "$W/data/mix-v6" "$W/data/mix-v7" --jevbench-public "$PUB" \
    --tokenizer "$TOK" --steps "$STEPS" --packing best_fit --also-steps 3000 \
    ${SOURCES:+--sources "$SOURCES"}
fi
if want reshare; then  # new shares on the built mix (no rebuild): mix-v7 → mix-v7-pre → mix-v7
  log "reshare mix-v7"
  rm -rf "$W/data/mix-v7-pre" && mv "$W/data/mix-v7" "$W/data/mix-v7-pre"
  tjev data derive "$W/data/mix-v7-pre" "$W/data/mix-v7" --refresh-shares --max-epochs 2 \
    --steps "$STEPS" --packing best_fit --also-steps 3000 --tokenizer "$TOK"
fi
if want derive; then
  log "derive mix-v7c (commercial-only: non-commercial rows dropped)"
  rm -rf "$W/data/mix-v7c"
  tjev data derive "$W/data/mix-v7" "$W/data/mix-v7c" --drop-license-use non-commercial \
    --max-epochs 2 --steps "$STEPS" --packing best_fit --also-steps 3000 --tokenizer "$TOK"
fi
if want report; then
  for m in mix-v7 mix-v7c; do
    log "data report $m"
    python scripts/data_report.py --mix "$W/data/$m" --jevbench "$PUB" --tokenizer "$TOK" \
      --out "$W/reports/$m" --steps "$REPORT_STEPS" --packing "$REPORT_PACKING" --sample "$REPORT_SAMPLE" --no-plots \
      || log "report $m FAILED"
  done
fi
if want upload; then
  log "upload to $HF_REPO"
  python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
if not api.dataset_info("$HF_REPO").private:
    raise SystemExit("$HF_REPO is not private: not uploading")
# mix-v7c first; mix-v7's mix.yaml last (a TPU run starts when it appears)
for m in ("mix-v7c", "mix-v7"):
    api.upload_folder(repo_id="$HF_REPO", repo_type="dataset", folder_path="$W/data/" + m,
                      path_in_repo="data/" + m, ignore_patterns=["mix.yaml"],
                      commit_message="data: " + m + " (files)")
    print("uploaded", m, "files", flush=True)
for m in ("mix-v7c", "mix-v7"):
    api.upload_file(repo_id="$HF_REPO", repo_type="dataset",
                    path_or_fileobj="$W/data/" + m + "/mix.yaml",
                    path_in_repo="data/" + m + "/mix.yaml", commit_message="data: " + m)
    print("uploaded", m, "mix.yaml", flush=True)
EOF
fi
log "done"
