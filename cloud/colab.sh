#!/usr/bin/env bash
# The campaign on a Colab TPU runtime (v6e-1 ≈ 4.6 compute units/h ≈ $0.46/h on Pro or
# pay-as-you-go, and the monthly units Google AI plans now include; v5e-1 on the free tier),
# driven headless by google-colab-cli (https://github.com/googlecolab/google-colab-cli; Linux
# or macOS, so WSL here; Python ≥ 3.13; `uv tool install google-colab-cli`, then log in).
# Sessions end at ~12 h (24 h on Pro+): each one restores the previous session's output,
# runs until 0.4 h before SESSION_HOURS, and saves it back here.
#
#   bash cloud/colab.sh all s0 sweep transfer final
#   bash cloud/colab.sh session N PHASES…     # one session (N > 1 restores session N-1)
#
# Settings: TPU (v6e1 | v5e1), SESSION_HOURS (11), MAX_SESSIONS (4), CAMPAIGN, SIZES
# ("4B,2B,0.8B"), BUDGET (planner knobs), MODELS ("0.8B 2B 4B"), TJEV_ROOT, MIX, OUT,
# WANDB_ENTITY (the W&B team the runs log to; default: the key's default entity).
# HF_DATASET (USER/REPO, optional): the host pulls data/ from that private HF dataset repo
# instead of an uploaded tarball, with HF_TOKEN (or the `hf auth login` token).
# The colab CLI flags follow its README (June 2026); check `colab --help` if one moved.
set -euo pipefail
export COPYFILE_DISABLE=1  # macOS tar: no ._* AppleDouble files or xattrs in the tarballs
: "${TJEV_ROOT:=$HOME/tjev-work}" "${MIX:=mix-v7.5}" "${MODELS:=0.8B 2B 4B}" "${TPU:=v6e1}"
: "${SESSION_HOURS:=11}" "${MAX_SESSIONS:=4}" "${CAMPAIGN:=tjev-$(date +%Y%m%d)}"
: "${SIZES:=4B,2B,0.8B}" "${OUT:=$TJEV_ROOT/colab-results/$CAMPAIGN}"
REPO=$(cd "$(dirname "$0")/.." && pwd)
# one runtime name per campaign and session: a launcher's exit trap (`colab stop -s`) must never
# reach another launcher's runtime (the CLI drops the name even when the stop fails, and the
# other launcher then cannot download or stop its own runtime)
S=tjev-$CAMPAIGN
log() { echo "[colab] $(date +%H:%M:%S) $*"; }
online() { local u; u=$(colab usage 2> /dev/null || true); grep -q "balance:" <<<"$u"; }
listed() { local l; l=$(colab sessions 2> /dev/null || true); grep -q "^\[$S\]" <<<"$l"; }
remote() {  # one command on the runtime over ssh (only for a listed runtime: `colab ssh -s`
  # creates a missing one)
  timeout 90 ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    -o ConnectTimeout=30 -o "ProxyCommand=colab ssh --proxy-mode -s $S" root@colab "$1"
}
wait_done() {  # until the runtime has /content/out.tgz and no session_run.py, or is gone
  while true; do
    if ! online; then sleep 60; continue; fi
    listed || { log "runtime $S is gone"; return 0; }
    remote "test -s /content/out.tgz && ! pgrep -f '[s]ession_run.py' > /dev/null" 2> /dev/null && return 0
    log "session still running on $S: waiting"
    sleep 300
  done
}

session() {  # session N PHASES…
  local n=$1; shift
  local tmp; tmp=$(mktemp -d)
  S=tjev-$CAMPAIGN-s$n
  git -C "$REPO" archive --format=tar.gz -o "$tmp/src.tgz" HEAD
  local data=/content/data.tgz files=(src.tgz job.json run.py session_run.py) hf_token=""
  if [[ -n ${HF_DATASET:-} ]]; then
    data=hf:$HF_DATASET
    hf_token=${HF_TOKEN:-$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2> /dev/null || true)}
  else
    tar czf "$tmp/data.tgz" -C "$TJEV_ROOT" "data/$MIX" data/jevbench
    files+=(data.tgz)
  fi
  # W&B key (never on a command line): $WANDB_API_KEY, else ~/.netrc's api.wandb.ai entry
  local wandb_key=${WANDB_API_KEY:-$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ~/.netrc 2> /dev/null)}
  # every PROBE_* setting of this shell, for the probe phase on the host
  local probe_env
  probe_env=$(python3 -c 'import json, os; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith("PROBE_")}))')
  local restore="[]"
  if [[ $n -gt 1 ]]; then
    tar czf "$tmp/prev.tgz" -C "$OUT/s$((n - 1))" tjev-work
    restore='["/content/prev"]'
    files+=(prev.tgz)
  fi
  cat > "$tmp/job.json" <<EOF
{"home": "/content/home", "code": "/content/src.tgz", "data": "$data", "hf_token": "$hf_token",
 "data_root": "$TJEV_ROOT", "mix": "$MIX", "models": "$MODELS", "restore": $restore,
 "save": "/content/save", "session_hours": $SESSION_HOURS, "chips": 1,
 "phases": "$*", "hardware": "${HARDWARE:-${TPU%?}}", "sizes": "$SIZES", "budget": "${BUDGET:-}",
 "campaign": "$CAMPAIGN",
 "code_version": "$(git -C "$REPO" rev-parse --short HEAD)", "wandb_key": "$wandb_key",
 "wandb_entity": "${WANDB_ENTITY:-}", "env": $probe_env}
EOF
  cat > "$tmp/run.py" <<'EOF'
import os, subprocess, sys, tarfile
if os.path.exists("/content/prev.tgz"):
    os.makedirs("/content/prev", exist_ok=True)
    tarfile.open("/content/prev.tgz").extractall("/content/prev", filter="data")
# a Jupyter kernel does not show a child's stdout: stream it into the cell, and keep a copy
cmd = [sys.executable, "-u", "/content/session_run.py", "/content/job.json"]
with open("/content/session.log", "w") as log:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
    code = proc.wait()
os.makedirs("/content/save", exist_ok=True)
os.replace("/content/session.log", "/content/save/session.log")
parts = [p for p in ("tjev-work", "session.log") if os.path.exists(f"/content/save/{p}")]
subprocess.run(["tar", "czf", "/content/out.tgz", "-C", "/content/save", *parts], check=True)
print(f"[run] session_run exited {code}", flush=True)
EOF
  cp "$REPO/cloud/session_run.py" "$tmp/"
  # stop paying whatever happens: EXIT, since `set -e` leaves without running a RETURN trap
  # shellcheck disable=SC2064  # expand $tmp now: it is local to this function
  trap "colab stop -s '$S' || true; rm -rf '$tmp'" EXIT
  # a TPU assignment can outlast the client's 120 s read timeout (no capacity yet): retry,
  # and look for the runtime first, since a timed-out request may still have created it
  local try
  for try in 1 2 3 4 5 6; do
    grep -q "^\[$S\]" <<<"$(colab sessions 2> /dev/null || true)" && break
    colab new -s "$S" --tpu "$TPU" && break
    [[ $try == 6 ]] && { log "no $TPU runtime after $try tries"; return 1; }
    log "colab new failed (try $try): retrying in $((try * 5)) min"
    sleep $((try * 300))
  done
  for f in "${files[@]}"; do
    colab upload -s "$S" "$tmp/$f" "/content/$f"
  done
  log "session $n: $*"
  colab exec -s "$S" -f "$tmp/run.py" --timeout "$(( (${SESSION_HOURS%.*} + 1) * 3600 ))" || true
  # the exec client also returns when its connection drops (laptop asleep, network change)
  # while the session runs on: wait for the session to finish on the runtime before
  # downloading, and wait out offline spells (no balance) instead of reading them as an end
  wait_done
  mkdir -p "$OUT/s$n"
  local try
  for try in 1 2 3; do
    colab download -s "$S" /content/out.tgz "$OUT/s$n/out.tgz" && break
    log "download attempt $try failed"; sleep 60
  done
  tar xzf "$OUT/s$n/out.tgz" -C "$OUT/s$n" && rm "$OUT/s$n/out.tgz"
  colab stop -s "$S" || true
  rm -rf "$tmp"
  trap - EXIT
}

cmd=${1:-}; shift || true
case "$cmd" in
  session) session "$@" ;;
  all)
    for n in $(seq 1 "$MAX_SESSIONS"); do
      session "$n" "$@"
      logs=$OUT/s$n/tjev-work/logs
      if [[ -f $logs/campaign.done ]]; then log "campaign done after $n session(s): $OUT/s$n"; exit 0; fi
      if [[ -f $logs/campaign.failed || ! -d $logs ]]; then log "session $n failed"; exit 1; fi
      log "paused: next session"
    done
    exit 1 ;;
  *) sed -n '2,18p' "$0"; exit 2 ;;
esac
