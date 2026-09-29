#!/usr/bin/env bash
# The campaign on Kaggle's free TPU v5e-8 (~20 TPU hours a week, 9 h per session, one TPU
# session at a time), as a chain of headless script kernels. Each session restores the
# previous one's output, runs until 0.4 h before its limit, then saves. Run from the machine
# with the data (WSL / Linux / macOS), with the Kaggle CLI set up (~/.kaggle/kaggle.json)
# and a phone-verified account (TPU access).
#
#   bash cloud/kaggle.sh push-data           # code + data as private datasets
#   bash cloud/kaggle.sh run N PHASES…       # session N (N > 1 restores session N-1)
#   bash cloud/kaggle.sh wait N | pull N     # poll until done; download its output
#   bash cloud/kaggle.sh all PHASES…         # push-data, then sessions 1, 2, … until done
#
# e.g.  bash cloud/kaggle.sh all s0 sweep transfer final
#
# Settings: CAMPAIGN (tjev-<date>: kernel slugs <campaign>-s<N>, W&B group), SIZES
# ("4B,2B,0.8B"), MODELS ("0.8B 2B 4B"), TJEV_ROOT (~/tjev-work), MIX (mix-v3), SESSION_HOURS (9), MAX_SESSIONS
# (4), OUT ($TJEV_ROOT/kaggle-results/<campaign>). W&B logs offline (<run>/wandb, `wandb sync`
# after pull) unless WANDB_INLINE=1, which writes $WANDB_API_KEY into the private kernel.
#
# Kaggle-cli #1197: a pushed kernel may silently get a CPU image; session_run.py then stops
# at once ("no TPU on this host"): open the kernel page and pick "TPU v5e-8" by hand.
set -euo pipefail
: "${TJEV_ROOT:=$HOME/tjev-work}" "${MIX:=mix-v3}" "${MODELS:=0.8B 2B 4B}" "${SESSION_HOURS:=9}"
: "${CAMPAIGN:=tjev-$(date +%Y%m%d)}" "${SIZES:=4B,2B,0.8B}"
: "${MAX_SESSIONS:=4}" "${OUT:=$TJEV_ROOT/kaggle-results/$CAMPAIGN}"
REPO=$(cd "$(dirname "$0")/.." && pwd)
KUSER=${KAGGLE_USERNAME:-$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")}
WORK=$OUT/.kaggle
log() { echo "[kaggle] $(date +%H:%M:%S) $*"; }
slug() { echo "$CAMPAIGN-s$1"; }

dataset() {  # dataset NAME DIR: create it the first time, then new versions
  local name=$1 dir=$2
  printf '{"title": "%s", "id": "%s/%s", "licenses": [{"name": "other"}]}\n' \
    "$name" "$KUSER" "$name" > "$dir/dataset-metadata.json"
  if kaggle datasets status "$KUSER/$name" > /dev/null 2>&1; then
    kaggle datasets version -p "$dir" -m "$(git -C "$REPO" rev-parse --short HEAD)" -q
  else
    kaggle datasets create -p "$dir" -q  # private unless --public
  fi
}

push_data() {
  mkdir -p "$WORK/tjev-code" "$WORK/tjev-data"
  git -C "$REPO" archive --format=tar.gz -o "$WORK/tjev-code/src.tgz" HEAD
  tar czf "$WORK/tjev-data/data.tgz" -C "$TJEV_ROOT" "data/$MIX" data/jevbench
  log "code $(du -h "$WORK/tjev-code/src.tgz" | cut -f1), data $(du -h "$WORK/tjev-data/data.tgz" | cut -f1)"
  dataset tjev-code "$WORK/tjev-code"
  dataset tjev-data "$WORK/tjev-data"
  log "waiting for the datasets to be ready"
  for name in tjev-code tjev-data; do
    until kaggle datasets status "$KUSER/$name" 2>/dev/null | grep -qi ready; do sleep 15; done
  done
}

run_session() {  # run_session N PHASES…
  local n=$1; shift
  local dir=$WORK/kernel-s$n prev="" key=""
  mkdir -p "$dir"
  [[ $n -gt 1 ]] && prev="\"$KUSER/$(slug $((n - 1)))\""
  [[ ${WANDB_INLINE:-0} == 1 ]] && key=${WANDB_API_KEY:?WANDB_INLINE=1 needs WANDB_API_KEY}
  python3 - "$REPO/cloud/session_run.py" "$dir/run.py" <<EOF
import json, sys
job = {
    "home": "auto",
    "code": "/kaggle/input/tjev-code", "data": "/kaggle/input/tjev-data",
    "data_root": "$TJEV_ROOT", "mix": "$MIX", "models": "$MODELS",
    "restore": ["/kaggle/input/$( [[ $n -gt 1 ]] && slug $((n - 1)) )"] if $n > 1 else [],
    "save": "/kaggle/working", "session_hours": float("$SESSION_HOURS"), "chips": 8,
    "phases": "$*", "hardware": "v5e", "sizes": "$SIZES", "campaign": "$CAMPAIGN",
    "code_version": "$(git -C "$REPO" rev-parse --short HEAD)", "wandb_key": "$key",
}
src = open(sys.argv[1]).read().replace("JOB: dict = {}", "JOB: dict = " + repr(job), 1)
open(sys.argv[2], "w").write(src)
EOF
  cat > "$dir/kernel-metadata.json" <<EOF
{"id": "$KUSER/$(slug "$n")", "title": "$(slug "$n")", "code_file": "run.py",
 "language": "python", "kernel_type": "script", "is_private": "true",
 "enable_gpu": "false", "enable_tpu": "true", "enable_internet": "true",
 "machine_shape": "TpuV5E8",
 "dataset_sources": ["$KUSER/tjev-code", "$KUSER/tjev-data"],
 "kernel_sources": [${prev}], "competition_sources": [], "model_sources": []}
EOF
  kaggle kernels push -p "$dir" --accelerator TpuV5E8 2>/dev/null || kaggle kernels push -p "$dir"
  log "session $n pushed: https://www.kaggle.com/code/$KUSER/$(slug "$n")"
}

wait_session() {
  local state
  while :; do
    state=$(kaggle kernels status "$KUSER/$(slug "$1")" 2>&1 | tr 'A-Z' 'a-z')
    case "$state" in
      *complete*) log "session $1 complete"; return 0 ;;
      *error*|*cancel*) log "session $1: $state"; return 1 ;;
      *) sleep 120 ;;
    esac
  done
}

pull_session() {
  mkdir -p "$OUT/s$1"
  kaggle kernels output "$KUSER/$(slug "$1")" -p "$OUT/s$1" -q
  log "session $1 output in $OUT/s$1"
  ls "$OUT/s$1/tjev-work/logs/" 2>/dev/null | grep campaign || true
}

cmd=${1:-}; shift || true
case "$cmd" in
  push-data) push_data ;;
  run) run_session "$@" ;;
  wait) wait_session "$1" ;;
  pull) pull_session "$1" ;;
  all)
    push_data
    for n in $(seq 1 "$MAX_SESSIONS"); do
      run_session "$n" "$@"
      wait_session "$n" || true
      pull_session "$n"
      logs=$OUT/s$n/tjev-work/logs
      if [[ -f $logs/campaign.done ]]; then log "campaign done after $n session(s)"; exit 0; fi
      if [[ -f $logs/campaign.failed || ! -d $logs ]]; then log "session $n failed: see its log"; exit 1; fi
      log "paused: starting session $((n + 1))"
    done
    log "not done after $MAX_SESSIONS sessions (the weekly quota is ~20 h)"; exit 1 ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
