#!/usr/bin/env bash
# Cloud TPU lifecycle, run from the machine that holds the data (Linux / macOS / WSL with
# gcloud authenticated): create -> upload code + data -> bootstrap -> campaign -> download
# -> delete. Costs money; nothing here runs unless you call it (docs/tpu.md).
#
#   bash cloud/tpu.sh up                     # queued resource (flex-start by default), wait ACTIVE
#   bash cloud/tpu.sh push                   # git HEAD + the mix + JevBench public
#   bash cloud/tpu.sh setup                  # cloud/tpu_bootstrap.sh on the VM (venv, models)
#   bash cloud/tpu.sh run s0 sweep transfer  # cloud/tpu_campaign.sh PHASES, detached on the VM
#   bash cloud/tpu.sh status | wait | ssh
#   bash cloud/tpu.sh pull                   # runs (selected checkpoint each), reports, logs
#   bash cloud/tpu.sh down                   # delete the queued resource and the VM
#   bash cloud/tpu.sh all PHASES…            # up push setup run wait pull, then ALWAYS down
#
# Settings (environment):
#   TPU_NAME=tjev-tpu  ZONE=us-east5-b  TYPE=v6e-8  RUNTIME=v2-alpha-tpuv6e  PROJECT (gcloud's)
#   MODE=flex|spot|on-demand   flex-start: never preempted, deleted by GCP at MAX_HOURS
#   MAX_HOURS=4 (flex hard cap)  WAIT_HOURS=2 (give up provisioning after)
#   BUCKET=gs://…/runs (optional; spot: runs mirrored every 10 min. On a new VM, restore
#     with `gcloud storage rsync -r $BUCKET ~/tjev-work/runs` before `run`: runs resume)
#   TJEV_ROOT=~/tjev-work (local: data/<MIX>/, data/jevbench/public.jsonl)  MIX=mix-v3
#   MODELS="0.8B 2B 4B"  SIZES=4B,2B,0.8B  CAMPAIGN=tpu-<date> (W&B group prefix)
#   WANDB_API_KEY (else ~/.netrc's api.wandb.ai entry; else W&B logs offline)
#   OUT=$TJEV_ROOT/tpu-results/<date>
set -euo pipefail
: "${TPU_NAME:=tjev-tpu}" "${ZONE:=us-east5-b}" "${TYPE:=v6e-8}" "${RUNTIME:=v2-alpha-tpuv6e}"
: "${MODE:=flex}" "${MAX_HOURS:=4}" "${WAIT_HOURS:=2}"
: "${TJEV_ROOT:=$HOME/tjev-work}" "${MIX:=mix-v3}" "${MODELS:=0.8B 2B 4B}" "${SIZES:=4B,2B,0.8B}"
: "${OUT:=$TJEV_ROOT/tpu-results/$(date +%Y%m%d-%H%M)}" "${CAMPAIGN:=tpu-$(date +%Y%m%d)}"
REPO=$(cd "$(dirname "$0")/.." && pwd)
QR="$TPU_NAME-qr"
HW=${TYPE%%-*}
PROJECT_FLAG=${PROJECT:+--project=$PROJECT}
# shellcheck disable=SC2086  # PROJECT_FLAG is empty or one word
g() { gcloud $PROJECT_FLAG "$@"; }
remote() { g compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --command "$*"; }
log() { echo "[tpu] $(date +%H:%M:%S) $*"; }

state() {
  g compute tpus queued-resources describe "$QR" --zone "$ZONE" --format='value(state.state)' 2>/dev/null || echo MISSING
}

up() {
  if [[ "$(state)" == ACTIVE ]]; then log "$QR already ACTIVE"; return; fi
  local common=(--node-id "$TPU_NAME" --zone "$ZONE" --accelerator-type "$TYPE" --runtime-version "$RUNTIME")
  case "$MODE" in
    flex) g alpha compute tpus queued-resources create "$QR" "${common[@]}" \
            --provisioning-model=flex-start --max-run-duration="${MAX_HOURS}h" \
            --valid-until-duration="${WAIT_HOURS}h" ;;
    spot) g compute tpus queued-resources create "$QR" "${common[@]}" --spot \
            --valid-until-duration="${WAIT_HOURS}h" ;;
    on-demand) g compute tpus queued-resources create "$QR" "${common[@]}" \
            --valid-until-duration="${WAIT_HOURS}h" ;;
    *) echo "MODE must be flex, spot or on-demand" >&2; exit 2 ;;
  esac
  log "waiting for $QR ($MODE $TYPE in $ZONE)"
  while :; do
    s=$(state)
    case "$s" in
      ACTIVE) log "ACTIVE"; break ;;
      FAILED|SUSPENDED|MISSING) log "queued resource is $s"; exit 1 ;;
      *) sleep 30 ;;
    esac
  done
}

push() {
  local tmp; tmp=$(mktemp -d)
  if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
    log "working tree is dirty: uploading HEAD $(git -C "$REPO" rev-parse --short HEAD) only"
  fi
  git -C "$REPO" archive --format=tar.gz -o "$tmp/src.tgz" HEAD
  tar czf "$tmp/data.tgz" -C "$TJEV_ROOT" "data/$MIX" data/jevbench
  log "upload $(du -sh "$tmp" | cut -f1)"
  g compute tpus tpu-vm scp "$tmp/src.tgz" "$tmp/data.tgz" "$TPU_NAME:~/" --zone "$ZONE"
  # W&B key (never on a command line): $WANDB_API_KEY, else ~/.netrc's api.wandb.ai entry
  local key=${WANDB_API_KEY:-$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ~/.netrc 2>/dev/null)}
  if [[ -n "$key" ]]; then
    (umask 077; printf '%s' "$key" > "$tmp/wandb_key")
    g compute tpus tpu-vm scp "$tmp/wandb_key" "$TPU_NAME:~/.wandb_key.tmp" --zone "$ZONE"
    remote "mkdir -p ~/tjev-work && mv ~/.wandb_key.tmp ~/tjev-work/.wandb_key && chmod 600 ~/tjev-work/.wandb_key"
  else
    log "no W&B key: runs log offline (wandb sync them after pull)"
  fi
  remote "set -e; mkdir -p ~/tjev-work/src ~/tjev-work/logs; tar xzf ~/src.tgz -C ~/tjev-work/src; \
    echo $(git -C "$REPO" rev-parse --short HEAD) > ~/tjev-work/src/CODE_VERSION; \
    tar xzf ~/data.tgz -C ~/tjev-work; rm ~/src.tgz ~/data.tgz; \
    sed -i \"s#$TJEV_ROOT#\$HOME/tjev-work#g\" ~/tjev-work/data/$MIX/mix.yaml; \
    head -c 300 ~/tjev-work/data/$MIX/mix.yaml"
  rm -rf "$tmp"
}

setup() { remote "MODELS='$MODELS' MIX=$MIX bash ~/tjev-work/src/cloud/tpu_bootstrap.sh"; }

run() {
  [[ $# -gt 0 ]] || { echo "usage: tpu.sh run PHASE..." >&2; exit 2; }
  remote "cd ~/tjev-work/src && rm -f ~/tjev-work/logs/campaign.{done,failed,paused} && \
    MIX=$MIX HARDWARE=$HW CHIPS=${TYPE##*-} SIZES=$SIZES TJEV_WANDB_CAMPAIGN=$CAMPAIGN \
    BUCKET=${BUCKET:-} nohup setsid bash cloud/tpu_campaign.sh $* \
      > ~/tjev-work/logs/campaign.log 2>&1 < /dev/null &
    sleep 2; tail -n 5 ~/tjev-work/logs/campaign.log"
}

status() {
  remote "tail -n 25 ~/tjev-work/logs/campaign.log; ls ~/tjev-work/logs/campaign.* 2>/dev/null; \
    ls ~/tjev-work/runs 2>/dev/null | tr '\n' ' '"
}

wait_done() {
  log "waiting for the campaign (polling every 5 min)"
  while :; do
    if [[ "$(state)" != ACTIVE ]]; then log "TPU is no longer ACTIVE ($(state))"; return 1; fi
    out=$(remote "ls ~/tjev-work/logs/campaign.done ~/tjev-work/logs/campaign.failed 2>/dev/null; \
      tail -n 1 ~/tjev-work/logs/campaign.log" 2>/dev/null || true)
    echo "$out" | tail -n 1
    if grep -q campaign.done <<<"$out"; then log "campaign done"; return 0; fi
    if grep -q campaign.failed <<<"$out"; then log "campaign FAILED"; return 1; fi
    sleep 300
  done
}

pull() {
  mkdir -p "$OUT"
  remote "cd ~/tjev-work/src && bash cloud/tpu_campaign.sh pack"
  g compute tpus tpu-vm scp "$TPU_NAME:~/tjev-work/pack.tgz" "$OUT/" --zone "$ZONE"
  tar xzf "$OUT/pack.tgz" -C "$OUT" && rm "$OUT/pack.tgz"
  log "results in $OUT"
}

down() {
  log "deleting $QR / $TPU_NAME"
  g compute tpus queued-resources delete "$QR" --zone "$ZONE" --force --quiet 2>/dev/null ||
    g compute tpus tpu-vm delete "$TPU_NAME" --zone "$ZONE" --quiet 2>/dev/null || true
  log "remaining: $(state)"
}

cmd=${1:-}; shift || true
case "$cmd" in
  up) up ;;
  push) push ;;
  setup) setup ;;
  run) run "$@" ;;
  status) status ;;
  wait) wait_done ;;
  ssh) g compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" ;;
  pull) pull ;;
  down) down ;;
  all)
    trap down EXIT  # whatever happens, stop paying
    up; push; setup; run "$@"
    ok=0; wait_done || ok=$?
    pull || log "pull failed"
    exit "$ok" ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
