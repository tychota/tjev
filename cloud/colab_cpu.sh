#!/usr/bin/env bash
# Run a command (tests, data jobs) on a Colab high-memory CPU runtime instead of this machine.
#
#   bash cloud/colab_cpu.sh [--keep] "COMMAND"        # from any git worktree of tjev
#
# Uploads the worktree as it is (tracked files, uncommitted changes included), installs it in
# a venv on the runtime (cached across calls on the same runtime), runs COMMAND in the
# worktree root with JAX_PLATFORMS=cpu, streams its output and exits with its status.
# The runtime (name tjev-cpu, ~0.26 CU/h) is created on first use and stopped at the end
# unless --keep (then reuse it and `colab stop -s NAME` when done). TJEV_CPU_TIMEOUT: seconds
# (default 7200). One command at a time per
# runtime: TJEV_CPU_SESSION=NAME gives each concurrent user its own.
set -euo pipefail
export COPYFILE_DISABLE=1
keep=0
[[ ${1:-} == --keep ]] && { keep=1; shift; }
cmd=${1:?usage: colab_cpu.sh [--keep] "COMMAND"}
S=${TJEV_CPU_SESSION:-tjev-cpu}  # one runtime per concurrent user (e.g. per agent)
repo=$(git rev-parse --show-toplevel)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"; [[ $keep == 1 ]] || colab stop -s "$S" > /dev/null 2>&1 || true' EXIT
# the list first: `colab sessions | grep -q` under pipefail fails on SIGPIPE when grep exits
# early, which read as "missing" and created a second runtime (the first one left unnamed)
sessions=$(colab sessions 2>/dev/null || true)
if ! grep -q "^\[$S\]" <<<"$sessions"; then
  colab new -s "$S" --high-mem
fi
# the worktree: tracked files plus modified / untracked ones (not ignored)
(cd "$repo" && { git ls-files; git ls-files --others --exclude-standard; } | sort -u \
  | tar czf "$tmp/src.tgz" -T -)
printf '%s' "$cmd" > "$tmp/cmd.txt"
colab upload -s "$S" "$tmp/src.tgz" /content/cpu-src.tgz > /dev/null
colab upload -s "$S" "$tmp/cmd.txt" /content/cpu-cmd.txt > /dev/null
# the HF token (gated / private datasets), as a file: never on a command line or in the log
hf_token=${HF_TOKEN:-$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2> /dev/null || true)}
if [[ -n $hf_token ]]; then
  (umask 077 && printf '%s' "$hf_token" > "$tmp/hf-token")
  colab upload -s "$S" "$tmp/hf-token" /content/cpu-hf-token > /dev/null
fi
cat > "$tmp/run.py" <<'EOF'
import os, subprocess, sys
os.makedirs("/content/cpu", exist_ok=True)
subprocess.run("rm -rf /content/cpu/src && mkdir -p /content/cpu/src && tar xzf /content/cpu-src.tgz -C /content/cpu/src", shell=True, check=True)
env = dict(os.environ, JAX_PLATFORMS="cpu", UV_SYSTEM_PYTHON="0")
if os.path.exists("/content/cpu-hf-token"):
    env["HF_TOKEN"] = open("/content/cpu-hf-token").read().strip()
setup = ("command -v uv > /dev/null || pip install -q uv; "
         "[ -d /content/cpu/venv ] || uv venv -q --python 3.12 /content/cpu/venv; "
         "uv pip install -q --python /content/cpu/venv/bin/python -e '/content/cpu/src[data,log,report,serve]' pytest httpx; "
         # the parity tests compare against HF transformers (CPU torch: no CUDA wheels)
         "uv pip install -q --python /content/cpu/venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch; "
         "uv pip install -q --python /content/cpu/venv/bin/python 'transformers>=5.17,<6'; "
         # the JevBench harness at the version tjev's official eval pins (tests/eval/test_official.py)
         "[ -d /content/cpu/jevbench ] || git clone -q https://github.com/fstandhartinger/jevbench /content/cpu/jevbench; "
         "git -C /content/cpu/jevbench checkout -q bb05a335bc809e61b20c0f745d25499a82b326fc")
subprocess.run(setup, shell=True, check=True, env=env)
env["JEVBENCH_HARNESS"] = "/content/cpu/jevbench"
env["PATH"] = "/content/cpu/venv/bin:" + env["PATH"]
p = subprocess.Popen(open("/content/cpu-cmd.txt").read(), shell=True, cwd="/content/cpu/src", env=env,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="", flush=True)
print(f"[colab_cpu] exit {p.wait()}", flush=True)
EOF
colab exec -s "$S" -f "$tmp/run.py" --timeout "${TJEV_CPU_TIMEOUT:-7200}" | tee "$tmp/out.log"
code=$(sed -nE 's/^\[colab_cpu\] exit ([0-9]+)$/\1/p' "$tmp/out.log" | tail -n 1)
exit "${code:-1}"
