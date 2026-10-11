#!/usr/bin/env bash
# Build mix-v7.5-np (mix-v7.5 without the yes/no rebalance; docs/data.md, mix-v7.5-np) on a
# Colab CPU runtime, then upload it to the private HF dataset repo. Run from the worktree root
# on the runtime, detached:
#
#   nohup bash scripts/build_mix_v75_np.sh > /content/np.log 2>&1 &
#
# The rebalance is baked into mix-v6 (5 sources) and mix-v7 (4 sources): their dropped rows
# come back from builds of the same code without it, checked against the built mixes
# (scripts/mix_np.py). STAGES (default all): download, unreb (mix-v6 rebuilt and mix-v7
# re-extended without the rebalance, in parallel; never uploaded), recover (the 9 sources'
# unrebalanced train files, each checked: its seeded rebalance is the built mix's file),
# assemble (mix-v7-np: mix-v7 with those files), seeds, extend (`extend-v75
# --no-rebalance-noul` → mix-v7.5-pre-np), derive (mix-v7.5-np: mix-v7.5's tiers and caps),
# finalize (sources drifted upstream since v7.5 frozen to its files; its held-out files),
# check (held-out files against mix-v7.5's, sha256; the report), upload (only if the repo
# is private; mix.yaml last).
set -euo pipefail
: "${HF_REPO:=tychot/tjev-data}" "${W:=/root/tjev-work}" "${STAGES:=all}" "${STEPS:=5000}"
: "${JEV_FR_MT_SHARE:=2.4}"
want() { [[ $STAGES == all || " $STAGES " == *" $1 "* ]]; }
log() { echo "[build-np] $(date +%H:%M:%S) $*"; }
D=$W/data
mkdir -p "$W/models/Qwen3.5-0.8B" "$D" "$W/np"
TOK=$W/models/Qwen3.5-0.8B
PUB=$D/jevbench/public.jsonl
CAP=(--max-epochs 2 --steps "$STEPS" --packing best_fit --also-steps 2500,3000 --tokenizer "$TOK")
V6_SOURCES=mage,mlma_en,mlma_fr,openjev,raid
V7_SOURCES=forecasting,oasst2,openjev_x,scruples
if want download; then
  [[ -f $TOK/tokenizer.json ]] || hf download Qwen/Qwen3.5-0.8B tokenizer.json --local-dir "$TOK" > /dev/null
  log "download mix-v6, mix-v7, mix-v7.5, jevbench and jev-fr-mt from $HF_REPO"
  python - <<PY
from huggingface_hub import snapshot_download
pats = ["data/jevbench/*", "data/jev-fr-mt/*", "data/jev-fr-mt/train/*", "data/jev-fr-mt/test/*",
        "data/jev-fr-mt/meta/*"]
for m in ("mix-v6", "mix-v7", "mix-v7.5"):
    pats += [f"data/{m}/*", f"data/{m}/**"]
snapshot_download("$HF_REPO", repo_type="dataset", local_dir="$W", allow_patterns=pats)
PY
fi
if want unreb; then
  log "mix-v6-nr (build, no rebalance) and mix-v7-nr (extend mix-v6, no rebalance), in parallel"
  rm -rf "$D/mix-v6-nr" "$D/mix-v7-nr"
  # mix-v6's own sources (the registry has the v7 ones too: their held-out rows would
  # change the post-pass of the v6 train rows)
  v6=$(python -c "import json; print(','.join(json.load(open('$D/mix-v6/manifest.json'))['sources']))")
  tjev data build "$D/mix-v6-nr" --sources "$v6" --jevbench-public "$PUB" --gen-scale 2 \
    --tokenizer "$TOK" > "$W/np/v6nr.log" 2>&1 &
  a=$!
  tjev data extend "$D/mix-v6" "$D/mix-v7-nr" --jevbench-public "$PUB" --tokenizer "$TOK" \
    --steps "$STEPS" --packing best_fit --also-steps 3000 --no-rebalance-noul \
    > "$W/np/v7nr.log" 2>&1 &
  b=$!
  wait $a || { log "mix-v6-nr FAILED (tail $W/np/v6nr.log)"; tail -20 "$W/np/v6nr.log"; exit 1; }
  log "mix-v6-nr done"
  wait $b || { log "mix-v7-nr FAILED (tail $W/np/v7nr.log)"; tail -20 "$W/np/v7nr.log"; exit 1; }
  log "mix-v7-nr done"
fi
if want recover; then
  log "recover the unrebalanced train files (checked against mix-v6 / mix-v7)"
  python scripts/mix_np.py recover "$D/mix-v6-nr" "$D/mix-v6" "$W/np/v6" --sources "$V6_SOURCES"
  python scripts/mix_np.py recover "$D/mix-v7-nr" "$D/mix-v7" "$W/np/v7" --sources "$V7_SOURCES"
fi
if want assemble; then
  log "assemble mix-v7-np"
  python scripts/mix_np.py assemble "$D/mix-v7" "$W/np/v6" "$W/np/v7" "$D/mix-v7-np"
fi
if want seeds; then
  log "gen_v7 real-text seeds (KALI, COLD, ShARC)"
  python -m tjev.data.sources.gen_v7.realtext build-seeds
fi
if want extend; then
  log "extend-v75 mix-v7-np → mix-v7.5-pre-np (no rebalance)"
  rm -rf "$D/mix-v7.5-pre-np"
  tjev data extend-v75 "$D/mix-v7-np" "$D/mix-v7.5-pre-np" --fr-mt "$D/jev-fr-mt" \
    --jevbench-public "$PUB" --jev-fr-mt-share "$JEV_FR_MT_SHARE" --no-rebalance-noul "${CAP[@]}"
fi
if want derive; then
  log "derive mix-v7.5-np (tier A ≤ 2 epochs, tier B ≤ 1, every source ≥ 1)"
  rm -rf "$D/mix-v7.5-np"
  tjev data derive "$D/mix-v7.5-pre-np" "$D/mix-v7.5-np" --tiers --min-epochs 1 "${CAP[@]}"
fi
if want finalize; then
  log "finalize: drifted sources frozen to mix-v7.5's files, mix-v7.5's held-out files"
  python scripts/mix_np.py finalize "$D/mix-v7.5" "$D/mix-v7" "$D/mix-v7-np" \
    "$D/mix-v7.5-pre-np" "$D/mix-v7.5-np" | tee "$W/np/finalize.txt"
fi
if want check; then
  log "pre train files against mix-v7.5's pre (manifest hashes)"
  python - <<PY
import hashlib, json
from pathlib import Path
want = json.loads(Path("$D/mix-v7.5/manifest.json").read_text())["derived_from"]["files"]
pre = Path("$D/mix-v7.5-pre-np/train")
same, diff = [], []
for path, h in sorted(want.items()):
    p = pre / Path(path).name
    (same if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == h else diff).append(p.stem)
print(f"pre train files identical to mix-v7.5-pre's: {len(same)}/{len(want)}; differ: {diff}")
PY
  python scripts/mix_np.py heldout "$D/mix-v7.5" "$D/mix-v7.5-np" || log "HELD-OUT FILES DIFFER"
  python scripts/mix_np.py report "$D/mix-v7.5" "$D/mix-v7.5-np" | tee "$W/np/report.md"
fi
if want upload; then
  log "upload mix-v7.5-np to $HF_REPO"
  python scripts/mix_np.py heldout "$D/mix-v7.5" "$D/mix-v7.5-np"  # never upload otherwise
  python - <<PY
from huggingface_hub import HfApi
api = HfApi()
if not api.dataset_info("$HF_REPO").private:
    raise SystemExit("$HF_REPO is not private: not uploading")
m = "mix-v7.5-np"
api.upload_folder(repo_id="$HF_REPO", repo_type="dataset", folder_path="$D/" + m,
                  path_in_repo="data/" + m, ignore_patterns=["mix.yaml"],
                  commit_message="data: " + m + " (files)")
print("uploaded", m, "files", flush=True)
api.upload_file(repo_id="$HF_REPO", repo_type="dataset", path_or_fileobj="$D/" + m + "/mix.yaml",
                path_in_repo="data/" + m + "/mix.yaml", commit_message="data: " + m)
print("uploaded", m, "mix.yaml", flush=True)
PY
fi
log "done"
