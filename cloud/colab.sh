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
# ("4B,2B,0.8B"), BUDGET (planner knobs), MODELS ("0.8B 2B 4B"), TJEV_ROOT, MIX, OUT.
# The colab CLI flags follow its README (June 2026); check `colab --help` if one moved.
set -euo pipefail
: "${TJEV_ROOT:=$HOME/tjev-work}" "${MIX:=mix-v3}" "${MODELS:=0.8B 2B 4B}" "${TPU:=v6e1}"
: "${SESSION_HOURS:=11}" "${MAX_SESSIONS:=4}" "${CAMPAIGN:=tjev-$(date +%Y%m%d)}"
: "${SIZES:=4B,2B,0.8B}" "${OUT:=$TJEV_ROOT/colab-results/$CAMPAIGN}"
REPO=$(cd "$(dirname "$0")/.." && pwd)
S=tjev-tpu
log() { echo "[colab] $(date +%H:%M:%S) $*"; }

session() {  # session N PHASES…
  local n=$1; shift
  local tmp; tmp=$(mktemp -d)
  git -C "$REPO" archive --format=tar.gz -o "$tmp/src.tgz" HEAD
  tar czf "$tmp/data.tgz" -C "$TJEV_ROOT" "data/$MIX" data/jevbench
  local restore="[]"
  if [[ $n -gt 1 ]]; then
    tar czf "$tmp/prev.tgz" -C "$OUT/s$((n - 1))" tjev-work
    restore='["/content/prev"]'
  fi
  cat > "$tmp/job.json" <<EOF
{"home": "/content/home", "code": "/content/src.tgz", "data": "/content/data.tgz",
 "data_root": "$TJEV_ROOT", "mix": "$MIX", "models": "$MODELS", "restore": $restore,
 "save": "/content/save", "session_hours": $SESSION_HOURS, "chips": 1,
 "phases": "$*", "hardware": "${TPU%?}", "sizes": "$SIZES", "budget": "${BUDGET:-}",
 "campaign": "$CAMPAIGN",
 "code_version": "$(git -C "$REPO" rev-parse --short HEAD)", "wandb_key": "${WANDB_API_KEY:-}"}
EOF
  cat > "$tmp/run.py" <<'EOF'
import os, subprocess, sys, tarfile
if os.path.exists("/content/prev.tgz"):
    os.makedirs("/content/prev", exist_ok=True)
    tarfile.open("/content/prev.tgz").extractall("/content/prev", filter="data")
sys.argv = ["session_run.py", "/content/job.json"]
code = subprocess.run([sys.executable, "/content/session_run.py", "/content/job.json"]).returncode
subprocess.run(["tar", "czf", "/content/out.tgz", "-C", "/content/save", "tjev-work"], check=True)
sys.exit(code)
EOF
  cp "$REPO/cloud/session_run.py" "$tmp/"
  colab new -s "$S" --tpu "$TPU"
  trap 'colab stop -s "$S" || true' RETURN  # stop paying whatever happens
  for f in src.tgz data.tgz job.json run.py session_run.py $( [[ $n -gt 1 ]] && echo prev.tgz ); do
    colab upload -s "$S" "$tmp/$f" "/content/$f"
  done
  log "session $n: $*"
  colab exec -s "$S" -f "$tmp/run.py" --timeout "$(( (${SESSION_HOURS%.*} + 1) * 3600 ))" || true
  mkdir -p "$OUT/s$n"
  colab download -s "$S" /content/out.tgz "$OUT/s$n/out.tgz"
  tar xzf "$OUT/s$n/out.tgz" -C "$OUT/s$n" && rm "$OUT/s$n/out.tgz"
  rm -rf "$tmp"
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
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
