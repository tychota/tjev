#!/usr/bin/env bash
# On the TPU host: run campaign phases in order (docs/tpu.md). Started detached by
# `cloud/tpu.sh run PHASE...` (GCP), or by cloud/session_run.py (Kaggle, Colab sessions).
# Writes $TJEV_ROOT/logs/campaign.{done,failed,paused}.
#
#   bash cloud/tpu_campaign.sh s0                       # bring-up: kernel tests, bench, zero-shot
#   bash cloud/tpu_campaign.sh sweep transfer final     # each phase planned from the previous
#   bash cloud/tpu_campaign.sh all                      # s0 sweep transfer final
#   bash cloud/tpu_campaign.sh pack                     # $TJEV_ROOT/pack.tgz for cloud/tpu.sh pull
#
# Environment:
#   TJEV_ROOT (~/tjev-work: src/, data/<MIX>/, data/jevbench/public.jsonl, models/, runs/)
#   TJEV_VENV (~/tjev-venv; absent: none)  MIX (mix-v3)  HARDWARE (v6e | v5e)  CHIPS (8)
#   SIZES (final sizes, longest first: "4B,2B,0.8B")
#   BUDGET  planner knobs for plan, fit and cost, e.g. "--seeds 2 --arm-seeds 2 --arms zclip,b32k"
#   DEADLINE_HOURS  session limit from the campaign's start: at it, runs checkpoint and the
#                   campaign exits 3 (paused); running it again resumes. Finished phases are
#                   skipped; a phase's queue is planned once (queues/PHASE.queue) and reused.
#   BUCKET (gs://…, optional mirror of the runs)  TJEV_WANDB_CAMPAIGN (W&B group prefix)
set -uo pipefail
cd "$(dirname "$0")/.."
: "${TJEV_ROOT:=$HOME/tjev-work}" "${TJEV_VENV:=$HOME/tjev-venv}" "${MIX:=mix-v3}"
: "${HARDWARE:=v6e}" "${CHIPS:=8}" "${SIZES:=4B,2B,0.8B}" "${DEADLINE_HOURS:=0}" "${BUDGET:=}"
ROOT=$TJEV_ROOT
RUNS=$ROOT/runs
MIXDIR=$ROOT/data/$MIX
JEVBENCH=$ROOT/data/jevbench/public.jsonl
MODELS=$ROOT/models
REPORTS=$ROOT/reports
QUEUES=$ROOT/queues
START=$(date +%s)
mkdir -p "$RUNS" "$REPORTS" "$ROOT/logs" "$QUEUES"
# shellcheck disable=SC1091
[[ -f $TJEV_VENV/bin/activate ]] && source "$TJEV_VENV/bin/activate"
export TJEV_ROOT
# shellcheck source=cloud/tpu_env.sh
source cloud/tpu_env.sh  # JAX / libtpu flags, compile cache, W&B, CODE_VERSION
: "${TJEV_WANDB_CAMPAIGN:=tpu-$(date +%Y%m%d)}"
export TJEV_WANDB_CAMPAIGN
say() { echo "[campaign] $(date -Is) $*"; }
KERNELS=--kernels
[[ -f $REPORTS/tpu-kernels.off ]] && KERNELS=--no-kernels
PRESET=tpu-$HARDWARE
read -ra BUDGET_ARGS <<<"$BUDGET"  # e.g. "--seeds 2 --arm-seeds 2 --no-muon": plan, fit and cost alike
plan() { tjev campaign "$@" --runs "$RUNS" --hardware "$HARDWARE" --sizes "$SIZES" "${BUDGET_ARGS[@]}"; }

remaining_hours() {  # what is left of DEADLINE_HOURS (0 = no deadline)
  python -c "import sys; d, s, n = map(float, sys.argv[1:]); print(0 if d <= 0 else max(0.01, d - (n - s) / 3600))" \
    "$DEADLINE_HOURS" "$START" "$(date +%s)"
}

s0() {
  [[ -f $REPORTS/s0.done ]] && { say "s0 already done"; return 0; }
  probe_libtpu_flags
  tjev doctor
  say "TPU kernel tests"
  if JAX_PLATFORMS=tpu python -m pytest -m tpu tests/kernels/test_pallas_tpu.py -q -s \
      > "$REPORTS/tpu-kernel-tests.log" 2>&1; then
    rm -f "$REPORTS/tpu-kernels.off"
  else
    say "kernel tests FAILED: every run falls back to the XLA paths (see tpu-kernel-tests.log)"
    touch "$REPORTS/tpu-kernels.off"
    KERNELS=--no-kernels
  fi
  tail -n 3 "$REPORTS/tpu-kernel-tests.log"
  say "bench"
  tjev campaign bench --models "$MODELS" --sizes "$SIZES" --out "$REPORTS/tpu-bench.json" \
    > "$REPORTS/tpu-bench.log" 2>&1 || say "bench failed (see tpu-bench.log)"
  say "zero-shot controls (JevBench public; temperatures from the mix calibration split)"
  for size in ${SIZES//,/ }; do
    extra=()
    [[ $KERNELS == --no-kernels ]] && extra=(compute.attention=xla compute.gdn_impl=chunked)
    tjev eval-base "$MODELS/Qwen3.5-$size" "$JEVBENCH" "$PRESET" "${extra[@]}" \
      --calibrate-on "$MIXDIR/calibration.jsonl" --out "$REPORTS/zeroshot-$size.json" \
      > /dev/null 2>> "$REPORTS/zeroshot.log" || say "zero-shot $size failed"
  done
  plan cost --chips "$CHIPS" \
    --bench "$REPORTS/tpu-bench.json" | tee "$REPORTS/tpu-cost.txt"
  touch "$REPORTS/s0.done"
}

phase() {  # sweep | transfer | final; returns 3 when paused by the deadline
  local q=$QUEUES/$1.queue
  if [[ -f $QUEUES/$1.done ]]; then say "$1 already done"; return 0; fi
  if [[ ! -s $q ]]; then  # planned once: a resumed phase keeps its jobs whatever it learns
    plan plan "$1" --mix "$MIXDIR" --models "$MODELS" "$KERNELS" --out "$q" || return 1
  fi
  say "$1: $(wc -l < "$q") jobs"
  tjev campaign queue "$q" --runs "$RUNS" --mix "$MIXDIR" --jevbench "$JEVBENCH" \
    --zeroshot "$REPORTS" --chips "$CHIPS" --deadline-hours "$(remaining_hours)" \
    ${BUCKET:+--bucket "$BUCKET"}
  local rc=$?
  plan fit --out "$REPORTS/fit-after-$1.json" > /dev/null
  python - "$REPORTS/fit-after-$1.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("decisions", d["decisions"])
print("recipe", {k: (v["lr"], v["rank"], v["horizon"]) for k, v in d["recipe"].items()})
PY
  [[ $rc -eq 0 ]] && touch "$QUEUES/$1.done"
  return $rc
}

final() {
  phase final
  local rc=$?
  [[ $rc -eq 3 ]] && return 3
  [[ $rc -ne 0 ]] && say "some final runs failed"
  tjev campaign select --runs "$RUNS" | tee "$REPORTS/final-selection.json"
  for run in $(python -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).values()))" "$REPORTS/final-selection.json"); do
    [[ -s $RUNS/$run/export/tjev_decision.json ]] && continue
    size=$(python -c "import json,re,sys; print(re.search(r'Qwen3\.5-([0-9.]+B)', json.load(open(sys.argv[1]))['config']['model']['path']).group(1))" "$RUNS/$run/config.json")
    say "full post-training and export: $run"
    {
      tjev post "$RUNS/$run" --mix "$MIXDIR" --jevbench "$JEVBENCH" --zeroshot "$REPORTS/zeroshot-$size.json" &&
      tjev export "$RUNS/$run" "$RUNS/$run/export" --jevbench "$JEVBENCH"
    } > "$RUNS/$run/post-full.log" 2>&1 || say "post-training $run failed"
  done
  return $rc
}

pack() {  # every run's metrics / evals / post reports, and only its selected checkpoint
  cd "$ROOT" || return 1
  python - <<'PY' > "$ROOT/pack.list"
import json, pathlib
for run in sorted(pathlib.Path("runs").iterdir()):
    for f in run.rglob("*"):
        rel = f.relative_to(run)
        # checkpoints: only the selected one (below); merged weights: rebuild with tjev export
        if f.is_file() and rel.parts[0] != "checkpoints" and not (
            rel.parts[0] == "export" and f.suffix == ".safetensors" and "adapter" not in f.name
        ):
            print(f)
    sel = run / "selection.json"
    if sel.exists():
        print(run / "checkpoints" / str(json.loads(sel.read_text())["step"]))
PY
  tar czf "$ROOT/pack.tgz" -T "$ROOT/pack.list" reports logs queues 2>/dev/null
  ls -la "$ROOT/pack.tgz"
}

phases=("$@")
[[ ${#phases[@]} -eq 1 && ${phases[0]} == all ]] && phases=(s0 sweep transfer final)
rm -f "$ROOT/logs/campaign."{done,failed,paused}
status=0
for p in "${phases[@]}"; do
  say "=== $p"
  case "$p" in
    s0) s0 || status=1 ;;
    sweep|transfer) phase "$p"; rc=$?; [[ $rc -eq 3 ]] && { status=3; break; }; [[ $rc -ne 0 ]] && status=1 ;;
    final) final; rc=$?; [[ $rc -eq 3 ]] && { status=3; break; }; [[ $rc -ne 0 ]] && status=1 ;;
    pack) pack; exit $? ;;
    *) say "unknown phase $p"; status=2 ;;
  esac
done
say "finished (status $status)"
case $status in
  0) touch "$ROOT/logs/campaign.done" ;;
  3) touch "$ROOT/logs/campaign.paused"; say "paused at the session deadline: run again to resume" ;;
  *) touch "$ROOT/logs/campaign.failed" ;;
esac
exit $status
