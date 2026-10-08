#!/usr/bin/env bash
# On the TPU host: run campaign phases in order (docs/tpu.md). Started by cloud/session_run.py
# (a Colab session). Writes $TJEV_ROOT/logs/campaign.{done,failed,paused}.
#
#   bash cloud/tpu_campaign.sh s0                       # bring-up: kernel tests, bench, zero-shot
#   bash cloud/tpu_campaign.sh sweep transfer final     # each phase planned from the previous
#   bash cloud/tpu_campaign.sh all                      # s0 sweep transfer final
#   bash cloud/tpu_campaign.sh probe                    # PROBE_ONLY=<mode> (below); default validate,bench
#
# Environment:
#   TJEV_ROOT (~/tjev-work: src/, data/<MIX>/, data/jevbench/public.jsonl, models/, runs/)
#   TJEV_VENV (~/tjev-venv; absent: none)  MIX (mix-v7.5)  HARDWARE (v6e | v5e)  CHIPS (1)
#   SIZES (final sizes, longest first: "4B,2B,0.8B")
#   BUDGET  planner knobs for plan, fit and cost, e.g. "--seeds 2 --arm-seeds 2 --arms zclip,b32k"
#   DEADLINE_HOURS  session limit from the campaign's start: at it, runs checkpoint and the
#                   campaign exits 3 (paused); running it again resumes. Finished phases are
#                   skipped; a phase's queue is planned once (queues/PHASE.queue) and reused.
#   TJEV_WANDB_CAMPAIGN (W&B group prefix)
#   PROBE_HF_REPO (tychot/tjev-4b): the private HF model repo the sweep's probes read
#                 checkpoints from and write backups, dumps and large captures to (needs HF_TOKEN)
#   PROBE_HF_FINAL_REPO (tychot/tjev-final): the final models' private HF repo: the backup
#                 of every final-* run and of the smoke probe's run, read by calib and finish
#                 (backup_repo picks the repo from the run's name)
#
# Probe modes (PROBE_ONLY), each under reports/probe/:
#   validate  TPU kernel tests (PROBE_TESTS: "test_pallas_tpu test_gdn_mega"), then the ground
#             truth: tjev on the TPU against the official Qwen3.5 (HF, fp32, host CPU), fp32
#             and bf16, per-layer traces (scripts/ground_truth.py)
#   bench     speed + the two xprof captures per variant (docs/profiling.md):
#             PROBE_SIZES ("0.8B,9B")  PROBE_VARIANTS ("kernels,split,xla")
#             PROBE_DEBUG (variants whose compile log and XLA / Mosaic dumps are kept)
#   analyse   every profile of the final train step per size: PROBE_ANALYSE_SIZES ("0.8B,2B,4B,9B")
#   smoke     a final run's exact job, shortened: PROBE_SMOKE_SIZE (4B)  PROBE_SMOKE_STEPS (30)
#             PROBE_SMOKE_EXTRA (more overrides); W&B tjev-final, backed up to PROBE_HF_FINAL_REPO
#   resume    a run and its SIGTERMed + resumed twin, step for step: PROBE_RESUME_SIZE (0.8B)
#   journey   every GDN-core iteration on real training (docs/kernel-journey.md):
#             PROBE_JOURNEY_SIZES ("0.8B,4B")
#   calib     per-item calibration dumps of kept checkpoints (docs/calibration.md):
#             PROBE_CALIB_RUNS ("final-4b-h3000:1200 …"; RUN:STEP:ema scores the EMA weights;
#             RUN may be <campaign>/<run>, the backup's path)  PROBE_CALIB_EXTRA_VALIDATION (held-out
#             files also evaluated, e.g. "validation-v7.jsonl calibration-v7.jsonl")
#   official  the official JevBench harness against tjev's server: PROBE_OFFICIAL_TARGETS
#             (run dirs and/or base:SIZE; "base:4B")  PROBE_OFFICIAL_DI_SUITE (a Decision Index suite)
#   finish    a final run's full post-training and export from its HF backup:
#             PROBE_FINISH_RUN (final-4b-h3000; <campaign>/<run> for the campaign-prefixed
#             backups)  PROBE_FINISH_SIZE (4B)
set -uo pipefail
cd "$(dirname "$0")/.."
: "${TJEV_ROOT:=$HOME/tjev-work}" "${TJEV_VENV:=$HOME/tjev-venv}" "${MIX:=mix-v7.5}"
: "${HARDWARE:=v6e}" "${CHIPS:=1}" "${SIZES:=4B,2B,0.8B}" "${DEADLINE_HOURS:=0}" "${BUDGET:=}"
: "${PROBE_HF_REPO:=tychot/tjev-4b}" "${PROBE_HF_FINAL_REPO:=tychot/tjev-final}"
ROOT=$TJEV_ROOT
RUNS=$ROOT/runs
MIXDIR=$ROOT/data/$MIX
JEVBENCH=$ROOT/data/jevbench/public.jsonl
MODELS=$ROOT/models
REPORTS=$ROOT/reports
QUEUES=$ROOT/queues
START=$(date +%s)
backup_repo() {  # the HF backup a run (bare name or <campaign>/<run>) lives in, by its kind
  local name=${1##*/}
  if [[ $name == final-* || $name == smoke-* ]]; then echo "$PROBE_HF_FINAL_REPO"; else echo "$PROBE_HF_REPO"; fi
}
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
PRESET=tpu-${HARDWARE%1}  # v6e1 (one chip, the planner's Colab entry) uses the v6e preset
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
    --profile "$REPORTS/xprof" > "$REPORTS/tpu-bench.log" 2>&1 || say "bench failed (see tpu-bench.log)"
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

detail_libtpu_flags() {  # the detail capture's libtpu flags, each only if libtpu takes it
  # (an unknown flag aborts libtpu): the LLO debug info (static per-bundle MXU / VPU / spill
  # analysis inside Pallas custom calls, xprof-nightly) and the kernels' named regions. Never
  # runtime custom-call tracing: one traced 0.8B step made a 16.5 GiB XSpace (truncated at
  # 2 GiB, no usable profile, 12 min per step). Prints the accepted flags.
  local flags="" flag
  for flag in --xla_xprof_register_llo_debug_info=true --xla_enable_custom_call_region_trace=true; do
    if LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-}$flags $flag" timeout 300 python -c "import jax; jax.devices()" > /dev/null 2>&1; then
      flags+=" $flag"; say "detail flag on: $flag" >&2
    else
      say "libtpu rejected $flag" >&2
    fi
  done
  echo "$flags"
}

probe_bench() {  # speed (numerics validated): bench + xprof; $1 the report dir, $2 sizes
  local out=$1 sizes=$2 rc=0
  local variants=${PROBE_VARIANTS:-kernels,split,xla} debug=()
  [[ -n ${PROBE_DEBUG:-} ]] && debug=(--debug-dir "$out/debug" --debug-variants "$PROBE_DEBUG")
  # compile logs are timing-safe (compile time only); the detail flags change what the
  # detail capture records, not the timed steps (docs/profiling.md)
  export JAX_LOG_COMPILES=1 JAX_EXPLAIN_CACHE_MISSES=1
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-}$(detail_libtpu_flags)"
  say "bench + xprof: $sizes / $variants"
  tjev campaign bench --models "$MODELS" --sizes "$sizes" --variants "$variants" \
    --profile "$out/xprof" "${debug[@]}" --out "$out/bench.json" > "$out/bench.log" 2>&1 \
    || { say "bench FAILED"; rc=1; }
  grep -E "^[0-9.]+B " "$out/bench.log" | cut -c1-200
  tail -n 12 "$out/bench.log"
  return $rc
}

checkout_pinned() {  # $1 repo URL, $2 directory, $3 commit: a clean checkout of the pin
  [[ -d $2/.git ]] || git clone -q "$1" "$2" || return 1
  git -C "$2" cat-file -e "$3^{commit}" 2> /dev/null || git -C "$2" fetch -q origin || return 1
  git -C "$2" checkout -q --detach "$3"
}

probe_official() {  # tjev eval-official over PROBE_OFFICIAL_TARGETS; $1 the report dir
  local out=$1 rc=0 harness=$ROOT/jevbench-harness kit=$ROOT/decision-index-kit pins
  local targets=${PROBE_OFFICIAL_TARGETS:-base:4B}
  mkdir -p "$out/runs"
  python -c "import fastapi, uvicorn" 2> /dev/null \
    || uv pip install -q --python "$(command -v python)" "fastapi>=0.115" "uvicorn>=0.30" \
    || { say "the serve extra (fastapi, uvicorn) failed to install"; return 1; }
  pins=$(python -c "from tjev.eval.official import HARNESS_COMMIT as h, DI_COMMIT as d; print(h, d)")
  checkout_pinned https://github.com/fstandhartinger/jevbench "$harness" "${pins% *}" \
    || { say "JevBench harness checkout failed"; return 1; }
  local suites=(jevbench-public)
  if [[ -n ${PROBE_OFFICIAL_DI_SUITE:-} ]]; then
    if checkout_pinned https://github.com/sinanuozdemir/decision-index "$kit" "${pins#* }"; then
      python -c "import httpx, huggingface_hub" 2> /dev/null \
        || uv pip install -q --python "$(command -v python)" "httpx>=0.27,<1" "huggingface_hub>=0.34,<2"
      suites+=(decision-index)
    else
      say "Decision Index kit checkout failed: JevBench only"
    fi
  fi
  # a run: tjev's prompt; the base model: tjev's own prompt and semif (the field's control)
  local target prompts prompt suite name path zs args
  for target in ${targets//,/ }; do
    prompts=tjev
    [[ $target == base:* ]] && prompts=tjev,semif
    for prompt in ${prompts//,/ }; do
      for orders in 1; do
        for suite in "${suites[@]}"; do
          args=(--suite "$suite" --prompt "$prompt" --orders "$orders" --harness "$harness"
                --models "$MODELS")
          if [[ $target == base:* ]]; then  # the base model zero-shot
            path=$target
            name=base-${target#base:}
            zs=$REPORTS/zeroshot-${target#base:}.json  # s0: temperatures on the mix split
            [[ $prompt == tjev && -s $zs ]] && args+=(--calibration "$zs")
          else  # a run: its selected checkpoint and post/calibration-jax.json
            path=$target
            [[ -d $path ]] || path=$RUNS/$target
            name=$(basename "$path")
          fi
          [[ $prompt != tjev ]] && name+=-$prompt
          [[ $orders != 1 ]] && name+=-o$orders
          if [[ $suite == decision-index ]]; then
            name+=-di
            args+=(--di-kit "$kit" --di-suite "$PROBE_OFFICIAL_DI_SUITE")
          fi
          say "official: $name ($suite; $target, prompt $prompt, $orders order(s))"
          rm -rf "$out/runs/$name"
          if tjev eval-official "$path" --out "$out/runs/$name" "${args[@]}" > "$out/$name.log" 2>&1; then
            cp "$out/runs/$name/report.json" "$out/$name.json"
            python -c "import json, sys; from tjev.cli.evaluation import official_brief as b; print(json.dumps(b(json.load(open(sys.argv[1])))))" \
              "$out/$name.json" | cut -c1-900
            TJEV_WANDB_EXPERIMENTS_PROJECT=tjev-benchmarks \
              python -m tjev.campaign.experiments log-json official-eval "official-$name" "$out/$name.json"
          else
            say "official $name FAILED (see official/$name.log)"
            tail -n 5 "$out/$name.log"
            rc=1
          fi
        done
      done
    done
  done
  return $rc
}

JOURNEY_VARIANTS=xla-nobarrier,xla,kernels,split,fused-bwd,fused-bwd+states,mega1,mega2,mega2-c128-blocked,mega2-c128-merged
JOURNEY_KEY_VARIANTS=xla,fused-bwd+states,mega2,mega2-c128-blocked,mega2-c128-merged

probe_journey() {  # every GDN iteration on real training; $1 the report dir (docs/kernel-journey.md)
  # 20 steps per size; 4B runs the five key iterations; parity and the IR dumps on 0.8B
  local out=$1 rc=0 q=$1/journey.queue size n
  local sizes=${PROBE_JOURNEY_SIZES:-0.8B,4B} steps=20
  local variants=$JOURNEY_VARIANTS size_variants variants_4b=$JOURNEY_KEY_VARIANTS
  local parity_sizes=0.8B ir_sizes=0.8B ir_variants=xla,fused-bwd,mega1,mega2,mega2-c128-blocked
  local journey=(python -m tjev.campaign.journey)
  mkdir -p "$out/runs" && : > "$q"
  # 1. the queue: per size, the planner's final run with each variant's compute
  for size in ${sizes//,/ }; do
    n=$steps
    # not the plan() wrapper: its trailing --sizes "$SIZES" overrides this size (tjev-journey2
    # queued the 0.8B line as 4B); the line is then picked by its run name, not its position.
    # The generic recipe (fit's rank): the journey's runs stay comparable with the earlier ones
    tjev campaign plan final --runs "$RUNS" --hardware "$HARDWARE" "${BUDGET_ARGS[@]}" \
      --mix "$MIXDIR" --models "$MODELS" "$KERNELS" --horizon 3000 --final-recipe generic \
      --sizes "$size" > "$out/final-$size.plan" \
      && "${journey[@]}" final-line --plan "$out/final-$size.plan" --size "$size" \
        --horizon 3000 > "$out/final-$size.line"
    [[ -s $out/final-$size.line ]] || { say "journey: no final plan for $size"; rc=1; continue; }
    size_variants=$variants; [[ $size != 4B ]] || size_variants=$variants_4b
    "${journey[@]}" queue --final-line "$out/final-$size.line" --size "$size" --steps "$n" \
      --variants "$size_variants" --profile-steps 3 >> "$q" || rc=1
  done
  say "journey: $(wc -l < "$q") runs; first: $(head -n 1 "$q" | cut -c1-300)"
  # 2. correctness first (cheap, and the article's "fast but wrong" row): one fixed batch of
  #    the 4096-token bucket per variant against the XLA reference with the conv barriers
  {
    say "journey: parity $parity_sizes"
    local checked  # the production XLA path first: it sets the gradient bound
    checked=$(python -c "import sys; print(','.join(dict.fromkeys(['xla', *sys.argv[1].split(',')])))" "$variants")
    tjev campaign parity --models "$MODELS" --mix "$MIXDIR" --sizes "$parity_sizes" \
      --variants "$checked" --lengths 4096 --batches 1 --out "$out/parity.json" \
      > "$out/parity.log" 2>&1  # exit 1 when a variant fails: xla-nobarrier does, on purpose
    grep -E "^parity .*: \{" "$out/parity.log" | cut -c1-260
  }
  # 3. the IR statistics: one GDN-core compile per variant with the HLO / Mosaic / LLO dumps
  {
    say "journey: IR dumps $ir_sizes / $ir_variants"
    "${journey[@]}" ir --sizes "$ir_sizes" --variants "$ir_variants" --out "$out/ir" \
      > "$out/ir.log" 2>&1 || say "journey: some IR compiles failed (see ir.log)"
    grep -E '^\{"ir"' "$out/ir.log" | cut -c1-400
  }
  # 4. the training runs, one at a time on the chip (a failure, e.g. an OOM, is a result)
  tjev campaign queue "$q" --runs "$out/runs" --mix "$MIXDIR" --chips 1 --retries 0 \
    --deadline-hours "$(remaining_hours)" || rc=1
  # 5. reports: per run and journey.json, the article's tables and charts, then W&B
  "${journey[@]}" collect --runs "$out/runs" --out "$out" --sizes "$sizes" --variants "$variants" \
    --variants-4b "$variants_4b" --parity "$out/parity.json" --ir "$out/ir/ir.json" --no-wandb | cut -c1-400 || rc=1
  python -c "import matplotlib" 2> /dev/null || uv pip install -q --python "$(command -v python)" "matplotlib>=3.8" \
    > /dev/null 2>&1 || say "matplotlib install failed: tables only"
  python scripts/journey_report.py "$out/journey.json" --out "$out/report" > /dev/null || rc=1
  "${journey[@]}" log --journey "$out/journey.json" --charts "$out/report" || say "journey: W&B logging failed"
  return $rc
}

probe_calib() {  # per-item calibration dumps (docs/calibration.md): $1 the report dir. For each
  # PROBE_CALIB_RUNS entry RUN:STEP (default the 4B finals' kept checkpoints), or RUN:STEP:ema
  # for the EMA of the adapters at that step (its optimizer state must be backed up: the
  # selected and the last step are), that checkpoint from the private HF backup
  # (PROBE_HF_REPO; RUN is its path there: <campaign>/<run>, or <run> for the older backups),
  # `tjev post --orders 2` with the step (and weights) as the selection, and post/ (with
  # post/predictions/) uploaded back as <RUN>/post-calib-<step>[-ema]/ (tjev.campaign.calib).
  # PROBE_CALIB_EXTRA_VALIDATION: space-separated held-out files also evaluated (`tjev post
  # --extra-validation`; names resolve in the mix dir, e.g. "validation-v7.jsonl calibration-v7.jsonl")
  local out=$1 rc=0 repo entry dir upload name zs size fetched
  mkdir -p "$out"
  for entry in ${PROBE_CALIB_RUNS:-final-4b-h3000:1200 final-4b-h3000:3000 final-4b-h1500:1450}; do
    repo=$(backup_repo "${entry%%:*}")
    say "calib $entry from hf:$repo"
    fetched=$(python -m tjev.campaign.calib fetch "$repo" "$entry" "$RUNS") \
      || { say "calib fetch FAILED"; rc=1; continue; }
    read -r dir upload <<<"$(tail -n 1 <<<"$fetched")"
    name=${dir##*/}
    local extra=() f
    for f in ${PROBE_CALIB_EXTRA_VALIDATION:-}; do extra+=(--extra-validation "$f"); done
    # the zero-shot control of the run's size (s0 / finish wrote it) when the session has it
    size=$(python -c "import json,re,sys; print(re.search(r'Qwen3\.5-([0-9.]+B)', json.load(open(sys.argv[1]))['config']['model']['path']).group(1))" "$dir/config.json" 2> /dev/null || true)
    zs=$REPORTS/zeroshot-$size.json
    [[ -n $size && -s $zs ]] && extra+=(--zeroshot "$zs")
    tjev post "$dir" --mix "$MIXDIR" --jevbench "$JEVBENCH" --orders 2 "${extra[@]}" > "$out/post-$name.log" 2>&1 \
      || { say "calib post FAILED"; tail -n 20 "$out/post-$name.log"; rc=1; continue; }
    du -sh "$dir/post/predictions" 2>/dev/null | cut -f1 | sed "s/^/[calib] predictions /"
    python - "$repo" "$dir" "$upload" <<'PY' || { say "calib upload FAILED"; rc=1; }
import sys
from huggingface_hub import HfApi
repo, d, upload = sys.argv[1:]
HfApi().upload_folder(repo_id=repo, folder_path=f"{d}/post", path_in_repo=upload,
                      commit_message=f"{upload}: calibration dump")
PY
    mkdir -p "$out/$name" && cp -r "$dir/post" "$out/$name/"  # small: in the session save
    rm -rf "$dir/checkpoints"
  done
  rm -rf "$RUNS/hf-calib"
  return $rc
}

analyse_summary() {  # $1 a capture's run dir, $2 its report dir, $3 the W&B run name,
  # $4 residuals.json (optional: names the scan residual buffers)
  local run=$1 dest=$2
  [[ -d $run/profile ]] || { say "analyse: no profile in $run"; return 1; }
  mkdir -p "$dest"
  local res=()
  [[ -s ${4:-} ]] && res=(--residuals "$4")
  python -m tjev.campaign.xprof_tools "$run/profile" --out "$dest" --wandb-name "$3" "${res[@]}" \
    > "$dest/summary.log" 2>&1 || { say "analyse: summary FAILED (see $dest/summary.log)"; tail -n 5 "$dest/summary.log"; return 1; }
  [[ -s $dest/scope_phase.txt ]] && head -n 20 "$dest/scope_phase.txt" | sed "s/^/[$3] /"
  python -c "import json, sys; s = json.load(open(sys.argv[1])); print(json.dumps({k: s.get(k) for k in ('source', 'steps', 'step_source', 'step_ms', 'device_ms_per_step', 'busy_fraction', 'by_phase', 'measured')})[:600])" \
    "$dest/summary.json" 2> /dev/null | sed "s/^/[$3] /"
  return 0
}

analyse_llo() {  # $1 a detail capture's profile dir, $2 its report dir: xprof-nightly's kernel
  # checks (its own venv: the CLI, and tensorflow for the HLO protos) into JSON
  local venv=$ROOT/xprof-nightly-venv session k
  mkdir -p "$2"
  if [[ ! -x $venv/bin/xprof ]]; then
    { uv venv -q --python 3.12 "$venv" \
        && uv pip install -q --python "$venv/bin/python" xprof-nightly tensorflow-cpu; } \
      > "$2/xprof-nightly-install.log" 2>&1 \
      || { say "analyse: xprof-nightly install failed: no LLO analysis (the xplane is kept)"; return 1; }
  fi
  session=$(find "$1" -name '*.xplane.pb' | sort | tail -n 1)
  [[ -n $session ]] || { say "analyse: no xplane in $1"; return 1; }
  session=$(dirname "$session")
  timeout 1200 "$venv/bin/xprof" check_kernel_profiling "$session" \
    > "$2/check_kernel_profiling.json" 2> "$2/check_kernel_profiling.log" \
    || say "analyse: check_kernel_profiling failed (see check_kernel_profiling.log)"
  for k in gdn_mega2_fwd gdn_mega2_bwd splash; do  # one analysis per kernel family, then all
    timeout 1800 "$venv/bin/xprof" get_llo_analysis "$session" --kernel "$k" \
      > "$2/llo_analysis_$k.json" 2> "$2/llo_analysis_$k.log" || say "analyse: get_llo_analysis $k failed"
  done
  timeout 2400 "$venv/bin/xprof" get_llo_analysis "$session" > "$2/llo_analysis.json" \
    2> "$2/llo_analysis.log" || say "analyse: get_llo_analysis failed (see llo_analysis.log)"
  python - "$2" <<'PY' || true
import json, sys
from pathlib import Path
for f in sorted(Path(sys.argv[1]).glob("*.json")):
    if not (f.name.startswith("llo_analysis") or f.name == "check_kernel_profiling.json"):
        continue
    try:
        d = json.loads(f.read_text() or "{}")
    except ValueError:
        print(f"[analyse] {f.name}: not JSON ({f.stat().st_size} bytes)")
        continue
    lines = {ln["name"]: ln.get("event_count") for p in d.get("planes", [])
             for ln in p.get("lines", []) if p.get("name", "").startswith("/device:TPU:0")}
    brief = {k: d[k] for k in ("status", "reason") if k in d}
    if lines:
        brief["Pallas Primitives"] = lines.get("Pallas Primitives")
    gates = {k: v for k, v in d.items() if "valid" in k or k.endswith("_ok")}
    print(f"[analyse] {f.name}: {json.dumps(brief | gates)[:300]} ({f.stat().st_size} bytes)")
PY
}

analyse_offload() {  # $1 a profile dir, $2 its HF path: files over 150 MB go to the private HF
  # repo (a session save restores only files under ~150 MB), a .hf pointer stays
  python - "$PROBE_HF_REPO" "$2" 150 "$1" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi
repo, prefix, max_mb, root = sys.argv[1], sys.argv[2], float(sys.argv[3]), Path(sys.argv[4])
big = [p for p in sorted(root.rglob("*")) if p.is_file() and p.stat().st_size > max_mb * 1e6]
for p in [p for p in big if p.name.endswith("op_stats_v2.pb")]:  # xprof's cache: rebuilt from the xplane
    p.unlink()
    big.remove(p)
if not big:
    print(f"[analyse] {prefix}: every file under {max_mb:.0f} MB, kept in the session save")
    sys.exit(0)
api = HfApi()
assert api.model_info(repo).private, "the backup repo must be private"
for p in big:
    dest, mb = f"{prefix}/{p.relative_to(root)}", p.stat().st_size / 1e6
    api.upload_file(path_or_fileobj=str(p), path_in_repo=dest, repo_id=repo,
                    commit_message=f"analyse: {dest}")
    p.with_name(p.name + ".hf").write_text(f"hf://{repo}/{dest}\n")
    p.unlink()
    print(f"[analyse] {p.name} ({mb:.0f} MB) -> hf:{repo}/{dest}")
PY
}

probe_analyse() {  # every profile of the step per size (docs/profiling.md); $1 the report dir.
  # Per size, the planner's v6e1 final run at rank 16: (a) the aggregate capture (25 timed
  # steps, 3 of them traced after the warm-up, timing trace mode), (b) the detail capture
  # (1 step, TRACE_COMPUTE_AND_SYNC, the LLO debug info; never for timing), (c) the
  # compile-only HLO dump of the same step; then the xprof summaries in
  # <size>/{aggregate,detail}/ and the HLO in <size>/hlo.tgz
  local out=$1 rc=0 size dir mode run drun items qrc detail_flags
  local sizes=${PROBE_ANALYSE_SIZES:-0.8B,2B,4B,9B} hw=v6e1
  local campaign=${TJEV_WANDB_CAMPAIGN:-analyse}
  local analyse=(python -m tjev.campaign.analyse)
  mkdir -p "$out/runs"
  python -c "import xprof.convert.raw_to_tool_data" 2> /dev/null \
    || uv pip install -q --python "$(command -v python)" "xprof>=2.23,<3" > "$out/xprof-install.log" 2>&1 \
    || say "analyse: xprof install failed: trace JSON summaries only (see xprof-install.log)"
  detail_flags=$(detail_libtpu_flags)
  for size in ${sizes//,/ }; do
    dir=$out/$size
    mkdir -p "$dir"
    # not the plan() wrapper: its trailing --sizes "$SIZES" would override this size
    tjev campaign plan final --runs "$RUNS" --hardware "$hw" "${BUDGET_ARGS[@]}" \
      --mix "$MIXDIR" --models "$MODELS" "$KERNELS" --horizon 3000 --sizes "$size" > "$dir/final.plan" \
      || { say "analyse: no final plan for $size"; rc=1; continue; }
    for mode in aggregate detail; do
      "${analyse[@]}" line --plan "$dir/final.plan" --size "$size" --mode "$mode" \
        --steps 25 --traced 3 --rank 16 > "$dir/$mode.queue" \
        || { say "analyse: no $mode line for $size"; rc=1; continue 2; }
    done
    say "analyse $size: $(cut -c1-400 "$dir/aggregate.queue")"
    # what each layer saves for its backward (the scan residual stacks), per bucket and remat
    # policy: names the AllocateBuffer stacks of the HBM peak in the summaries
    read -ra items <<<"$("${analyse[@]}" items --line-file "$dir/aggregate.queue")"
    python -m tjev.campaign.residuals "${items[@]}" --policies all --out "$dir/residuals.json" \
      > "$dir/residuals.log" 2>&1 || { say "analyse $size: residuals listing failed (see residuals.log)"; rc=1; }
    grep -E "^T=|^policy" "$dir/residuals.log" | sed "s/^/[analyse $size] /"
    # (a) the aggregate capture: the timing trace mode, the session's own libtpu flags
    tjev campaign queue "$dir/aggregate.queue" --runs "$out/runs" --mix "$MIXDIR" --chips 1 \
      --retries 0 --deadline-hours "$(remaining_hours)"
    qrc=$?
    [[ $qrc -eq 3 ]] && { say "analyse: paused at the deadline"; return 3; }
    [[ $qrc -ne 0 ]] && { say "analyse $size aggregate FAILED"; rc=1; }
    run=$out/runs/$(head -n 1 "$dir/aggregate.queue" | cut -d' ' -f1)
    grep -h '"perf/' "$run/stdout.log" 2> /dev/null | tail -n 2 | cut -c1-300
    analyse_summary "$run" "$dir/aggregate" "analyse-$size-aggregate" "$dir/residuals.json" || rc=1
    # (b) the detail capture: one step with the sync waits and the LLO debug info
    LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-}$detail_flags" tjev campaign queue "$dir/detail.queue" \
      --runs "$out/runs" --mix "$MIXDIR" --chips 1 --retries 0 --deadline-hours "$(remaining_hours)"
    qrc=$?
    [[ $qrc -eq 3 ]] && { say "analyse: paused at the deadline"; return 3; }
    [[ $qrc -ne 0 ]] && { say "analyse $size detail FAILED"; rc=1; }
    drun=$out/runs/$(head -n 1 "$dir/detail.queue" | cut -d' ' -f1)
    analyse_summary "$drun" "$dir/detail" "analyse-$size-detail" "$dir/residuals.json" || rc=1
    # the kernels' LLO capture (before the offload may move the xplane to HF)
    analyse_llo "$drun/profile" "$dir/detail" || rc=1
    # (c) the compile-only HLO dump of the same step (no compile cache: a hit dumps nothing);
    #     kept: the optimized HLO, its buffer assignment and memory-usage report
    rm -rf "$dir/hlo-dump"
    say "analyse $size: HLO dump (compile-check)"
    JAX_ENABLE_COMPILATION_CACHE=false \
      XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$dir/hlo-dump --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*" \
      tjev compile-check "${items[@]}" > "$dir/compile-check.json" 2> "$dir/compile-check.log" \
      || { say "analyse $size: compile-check FAILED (see compile-check.log)"; rc=1; }
    if [[ -d $dir/hlo-dump ]]; then
      (cd "$dir/hlo-dump" && find . -type f \( -name '*after_optimizations*' -o -name '*buffer-assignment*' \
        -o -name '*memory-usage-report*' \) | tar czf "$dir/hlo.tgz" -T -)
      say "analyse $size: hlo.tgz $(du -h "$dir/hlo.tgz" | cut -f1) ($(tar tzf "$dir/hlo.tgz" | wc -l) files)"
      rm -rf "$dir/hlo-dump"
    fi
    # the raw captures: in the session save when small, else on the private HF repo
    analyse_offload "$run/profile" "analyse/$campaign/$size/aggregate" || { say "analyse: offload FAILED"; rc=1; }
    analyse_offload "$drun/profile" "analyse/$campaign/$size/detail" || { say "analyse: offload FAILED"; rc=1; }
    du -sh "$dir" "$run" "$drun" 2> /dev/null | sed "s/^/[analyse] /"
  done
  return $rc
}

probe() {  # kernel iteration, rerun every time: reports/probe/ (and tjev-experiments on W&B)
  local out=$REPORTS/probe sizes=${PROBE_SIZES:-0.8B,9B} rc=0
  rm -rf "$out" && mkdir -p "$out"
  probe_libtpu_flags
  tjev doctor > "$out/doctor.json" 2>&1
  local mode=${PROBE_ONLY:-validate,bench}
  if [[ $mode == finish ]]; then  # a final run's full post-training and export, from
    # its HF backup (PROBE_HF_REPO: the run's metadata and selected checkpoint), then the
    # export and post reports uploaded back next to it
    # PROBE_FINISH_RUN: the run's path on the backup (<campaign>/<run>, or <run> for the older
    # backups); locally RUNS/<run>
    local hfrun=${PROBE_FINISH_RUN:-final-4b-h3000} size=${PROBE_FINISH_SIZE:-4B}
    local repo run=${hfrun##*/}
    repo=$(backup_repo "$hfrun")
    say "finish $hfrun ($size) from hf:$repo"
    python - "$repo" "$hfrun" "$RUNS" <<'PY' || return 1
import json, shutil, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
repo, hfrun, runs = sys.argv[1], sys.argv[2], Path(sys.argv[3])
run = hfrun.rsplit("/", 1)[-1]
api = HfApi()
assert api.model_info(repo).private, "the backup repo must be private"
sel = json.loads(Path(hf_hub_download(repo, f"{hfrun}/selection.json")).read_text())
tmp = Path(snapshot_download(repo, allow_patterns=[f"{hfrun}/*.json", f"{hfrun}/*.jsonl", f"{hfrun}/*.txt",
           f"{hfrun}/*.log", f"{hfrun}/eval/*", f"{hfrun}/checkpoints/{sel['step']}/**"],
           local_dir=str(runs.parent / "hf-finish")))
dest = runs / run
dest.mkdir(parents=True, exist_ok=True)
for f in (tmp / hfrun).iterdir():
    target = dest / f.name
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    shutil.move(str(f), str(target))
(dest / "queue.done").touch()
print(json.dumps({"run": run, "step": sel["step"], "score": sel["score"]}))
PY
    if [[ ! -s $REPORTS/zeroshot-$size.json ]]; then  # the post-training control
      say "zero-shot control $size"
      tjev eval-base "$MODELS/Qwen3.5-$size" "$JEVBENCH" "$PRESET" \
        --calibrate-on "$MIXDIR/calibration.jsonl" --out "$REPORTS/zeroshot-$size.json" \
        > "$out/zeroshot-$size.log" 2>&1 || { say "zero-shot FAILED"; tail -n 20 "$out/zeroshot-$size.log"; return 1; }
    fi
    say "full post-training and export: $run"
    tjev post "$RUNS/$run" --mix "$MIXDIR" --jevbench "$JEVBENCH" --zeroshot "$REPORTS/zeroshot-$size.json" \
      > "$RUNS/$run/post-full.log" 2>&1 || { say "post FAILED"; tail -n 20 "$RUNS/$run/post-full.log"; return 1; }
    tjev export "$RUNS/$run" "$RUNS/$run/export" --jevbench "$JEVBENCH" \
      > "$RUNS/$run/export.log" 2>&1 || { say "export FAILED"; tail -n 20 "$RUNS/$run/export.log"; return 1; }
    say "upload export + post to hf:$repo"
    python - "$repo" "$hfrun" "$RUNS" <<'PY' || return 1
import sys
from pathlib import Path
from huggingface_hub import HfApi
repo, hfrun, runs = sys.argv[1], sys.argv[2], Path(sys.argv[3])
run = hfrun.rsplit("/", 1)[-1]
api = HfApi()
assert api.model_info(repo).private
for sub in ("export", "post"):
    if (runs / run / sub).is_dir():
        api.upload_folder(repo_id=repo, folder_path=str(runs / run / sub), path_in_repo=f"{hfrun}/{sub}",
                          commit_message=f"{hfrun}: {sub}")
for f in ("post-full.log", "export.log"):
    if (runs / run / f).exists():
        api.upload_file(path_or_fileobj=str(runs / run / f), path_in_repo=f"{hfrun}/{f}", repo_id=repo)
listing = {s.rfilename: s.size for s in api.model_info(repo, files_metadata=True).siblings
           if s.rfilename.startswith(f"{hfrun}/export/")}
print(len(listing), "export files on hf;", round(sum(v or 0 for v in listing.values()) / 1e9, 2), "GB")
PY
    # the merged snapshot is on HF now: keep the session's save (the download) small
    find "$RUNS/$run/export" -name 'model*.safetensors' -delete
    rm -rf "$RUNS/$run/checkpoints" "$ROOT/hf-finish"
    return $rc
  fi
  if [[ $mode == smoke ]]; then  # a final run's exact job, PROBE_SMOKE_STEPS long
    # (default 30): real throughput over the buckets, peak HBM, compiles, one checkpoint and
    # one full eval; reports/probe/smoke/<run> (W&B project tjev-final as the final runs,
    # group <campaign>/smoke; backed up to PROBE_HF_FINAL_REPO by session_run.py);
    # PROBE_SMOKE_EXTRA appends overrides
    local size=${PROBE_SMOKE_SIZE:-4B} steps=${PROBE_SMOKE_STEPS:-30} q=$out/smoke.queue
    plan plan final --mix "$MIXDIR" --models "$MODELS" "$KERNELS" --sizes "$size" \
      --horizon 3000 | head -n 1 \
      | sed -E "s/^final-([^ ]+)/smoke-\1/; s#(log\.wandb_group=[^ ]*)/final#\1/smoke#; s/ post=[a-z]+/ post=none/; s/train\.steps=[0-9]+/train.steps=$steps/; s/train\.checkpoint_every=[0-9]+/train.checkpoint_every=$steps/" \
      > "$q" || return 1
    # PROBE_SMOKE_EXTRA: more overrides on the line (e.g. an xprof trace, the chosen rank)
    [[ -z ${PROBE_SMOKE_EXTRA:-} ]] || sed -i "s|\$| $PROBE_SMOKE_EXTRA|" "$q"
    say "smoke: $(cut -c1-300 "$q") … $(grep -o '[^ ]*$' "$q")"
    tjev campaign queue "$q" --runs "$out/smoke" --mix "$MIXDIR" --jevbench "$JEVBENCH" \
      --zeroshot "$REPORTS" --chips 1 --deadline-hours "$(remaining_hours)" || rc=1
    for r in "$out"/smoke/*/; do
      grep -h '"perf/' "$r/stdout.log" 2>/dev/null | tail -n 3 | cut -c1-400
      [[ -s $r/health.json ]] && cut -c1-300 "$r/health.json"
    done
    return $rc
  fi
  if [[ $mode == journey ]]; then  # the kernel journey: reports/probe/journey/
    probe_journey "$out/journey"
    return $?
  fi
  if [[ $mode == resume ]]; then  # resume on the TPU: reports/probe/resume/ (a run
    # straight through and its twin SIGTERMed at step 60 and resumed, from the
    # planner's final line of PROBE_RESUME_SIZE shortened to 120 steps)
    local size=${PROBE_RESUME_SIZE:-0.8B} dir=$out/resume
    mkdir -p "$dir"
    tjev campaign plan final --runs "$RUNS" --hardware "$HARDWARE" "${BUDGET_ARGS[@]}" \
      --mix "$MIXDIR" --models "$MODELS" "$KERNELS" --horizon 3000 --sizes "$size" > "$dir/final.plan" \
      && python -m tjev.campaign.journey final-line --plan "$dir/final.plan" --size "$size" \
        --horizon 3000 > "$dir/final.line" || { say "resume: no final line for $size"; return 1; }
    say "resume test on $size: $(cut -c1-200 "$dir/final.line")"
    python scripts/resume_test.py --line "$dir/final.line" --runs "$dir/runs" \
      --steps 120 --stop-at 60 | tee "$dir/verdict.json"
    return "${PIPESTATUS[0]}"
  fi
  if [[ $mode == analyse ]]; then  # profile analysis: reports/probe/analyse/
    probe_analyse "$out/analyse"
    return $?
  fi
  if [[ $mode == calib ]]; then  # per-item calibration dumps: reports/probe/calib/
    probe_calib "$out/calib"
    return $?
  fi
  if [[ $mode == official ]]; then  # the official harness: reports/probe/official/
    probe_official "$out/official"
    return $?
  fi
  if [[ $mode == bench ]]; then  # speed only (numerics validated)
    probe_bench "$out" "$sizes"
    return $?
  fi
  [[ $mode == validate || $mode == validate,bench ]] || { say "unknown PROBE_ONLY $mode"; return 2; }
  # validate: the TPU kernel tests, then the ground truth: tjev on the TPU against the
  # official Qwen3.5 (HF transformers, fp32, on the host CPU: computed once, cached), the
  # forward and 115 per-layer gradients on long real texts packed in one row, fp32 (the
  # 1e-5 check) and the bf16 production paths (with the official model in bf16 as the noise
  # floor and the per-layer hidden-state errors)
  if ! python -c "import torch, transformers" 2> /dev/null; then
    { uv pip install -q --python "$(command -v python)" --index-url https://download.pytorch.org/whl/cpu torch \
      && uv pip install -q --python "$(command -v python)" "transformers>=5.17,<6"; } \
      > "$out/hf-install.log" 2>&1 || say "torch / transformers install failed (see hf-install.log)"
  fi
  mkdir -p "$ROOT/cache"
  local gt=(python scripts/ground_truth.py "$MODELS/Qwen3.5-0.8B"
            --texts "$MIXDIR/train/gen_long_refund_policy.jsonl" --count 3 --max-tokens 4096
            --stride 8 --gold-cache "$ROOT/cache/gold-0.8B.npz")  # not saved: 3.5 GB
  local tests=${PROBE_TESTS:-test_pallas_tpu test_gdn_mega} xmls=() f
  for f in $tests; do
    xmls+=("$out/tests-$f.xml")
    say "TPU kernel tests: $f"
    JAX_PLATFORMS=tpu python -m pytest -m tpu "tests/kernels/$f.py" -q -s --junitxml="$out/tests-$f.xml" \
      > "$out/tests-$f.log" 2>&1 || { say "$f FAILED"; rc=1; }
    tail -n 2 "$out/tests-$f.log"
  done
  python -m tjev.campaign.experiments junit kernel-tests "${xmls[@]}"
  say "ground truth fp32 (the 1e-5 check) and bf16 vs the official model"
  "${gt[@]}" --dtype float32 --configs xla:chunked:highest:highest --trace-layers \
    --out "$out/gt-validate-fp32.json" > "$out/gt-validate-fp32.log" 2>&1
  "${gt[@]}" --dtype bfloat16 --hf-bf16 --trace-layers \
    --configs xla:chunked:high,splash:pallas_tpu_v2:high,splash:pallas_tpu_mega2:high \
    --out "$out/gt-validate-bf16.json" > "$out/gt-validate-bf16.log" 2>&1
  for f in fp32 bf16; do
    grep -E '^\{"config"' "$out/gt-validate-$f.log" | sed "s/^/[gt-$f] /" | cut -c1-330
    [[ -s $out/gt-validate-$f.json ]] && python -m tjev.campaign.experiments log-json ground-truth "gt-validate-$f-0.8B-tpu" "$out/gt-validate-$f.json"
  done
  if [[ $mode == validate,bench ]]; then probe_bench "$out" "$sizes" || rc=1; fi
  return $rc
}

phase() {  # sweep | transfer | final; returns 3 when paused by the deadline
  # a staged sweep (BUDGET "--stage lr|arms"): each stage its own queue and done flag, the
  # runs shared (stage 2 pairs with stage 1's centre run, the fit reads them all)
  local key=$1 stage
  stage=$(sed -nE 's/.*--stage[ =]([a-z]+).*/\1/p' <<<"$BUDGET")
  [[ $1 == sweep && -n $stage && $stage != all ]] && key=sweep-$stage
  # an explicit arm list (a later stage: --arms a,b,c) gets its own queue too
  [[ $1 == sweep && $BUDGET == *--arms* ]] && key=$key-$(printf %s "$BUDGET" | cksum | cut -c1-6)
  local q=$QUEUES/$key.queue
  if [[ -f $QUEUES/$key.done ]]; then say "$key already done"; return 0; fi
  if [[ ! -s $q ]]; then  # planned once: a resumed phase keeps its jobs whatever it learns
    plan plan "$1" --mix "$MIXDIR" --models "$MODELS" "$KERNELS" --out "$q" || return 1
  fi
  say "$key: $(wc -l < "$q") jobs"
  tjev campaign queue "$q" --runs "$RUNS" --mix "$MIXDIR" --jevbench "$JEVBENCH" \
    --zeroshot "$REPORTS" --chips "$CHIPS" --deadline-hours "$(remaining_hours)"
  local rc=$?
  plan fit --out "$REPORTS/fit-after-$key.json" > /dev/null
  python - "$REPORTS/fit-after-$key.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("decisions", d["decisions"])
print("recipe", {k: (v["lr"], v["rank"], v["horizon"]) for k, v in d["recipe"].items()})
PY
  [[ $rc -eq 0 ]] && touch "$QUEUES/$key.done"
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
    # post=full in the queue already ran the full post-training (post/validation-cal.json);
    # only a run whose queue post was quick or failed gets it here
    say "full post-training and export: $run"
    {
      [[ -s $RUNS/$run/post/validation-cal.json ]] ||
        tjev post "$RUNS/$run" --mix "$MIXDIR" --jevbench "$JEVBENCH" --zeroshot "$REPORTS/zeroshot-$size.json" &&
      tjev export "$RUNS/$run" "$RUNS/$run/export" --jevbench "$JEVBENCH"
    } > "$RUNS/$run/post-full.log" 2>&1 || say "post-training $run failed"
  done
  return $rc
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
    probe) probe || status=1 ;;
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
