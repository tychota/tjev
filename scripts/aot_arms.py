"""v6e memory of the train step per size, arm and bucket, compiled on a CPU host (no TPU).

    pip install 'jax[tpu]'   # libtpu's compile-only client (scripts/aot_tpu.py)
    python scripts/aot_arms.py --sizes 4B,9B --buckets 1024,2048,4096 \\
        --arms baseline,flat,remat,candidate [--jobs 3] [--out reports/aot]

Each (size, arm, bucket) is one ``scripts/aot_tpu.py`` process on the run configuration
the campaign planner builds today for Colab v6e-1 (``tjev.campaign.plan``: the tpu-v6e
preset, the size's preset, its microbatch, the v6e1 extras, the final LoRA targets), with
the arm's overrides on top; the model directory holds only the HF config.json of
``plan.model_config`` (the 9B head untied). One microbatch's rows of the bucket, the
accumulation of the planner's batch. Prints a table of XLA's memory analysis (GiB) against
the 29 GiB a v6e chip leaves to a program, and writes every report to ``--out``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

LIMIT_GIB = 29.0


def arms(size: str) -> dict[str, list[str]]:
    """The validation arms (docs/training.md, docs/kernels.md): the planner's configuration,
    each perf feature alone, and the all-in candidate."""
    large = size == "9B"
    # 9B: minimal_states only where it fits (its 4096 bucket keeps core_states)
    remat = (["compute.remat_by_bucket={1024: minimal_states, 2048: minimal_states}"]
             if large else ["compute.remat=minimal_states"])  # fmt: skip
    flat = ["compute.gdn_norm=flat", *(["compute.head_rows=letters"] if large else [])]
    return {
        "baseline": [],
        "flat": flat,
        "packing": ["train.packing=best_fit"],  # the same shapes: memory as the baseline
        "remat": remat,
        "candidate": [*flat, "compute.gdn_precision=mixed",
                      "train.packing=best_fit", *remat],
    }  # fmt: skip


def run_config(size: str, model_dir: Path, extra: list[str]) -> dict:
    from tjev.campaign.plan import FINAL_OVERRIDES, HARDWARE
    from tjev.config import load_config

    hw = HARDWARE["v6e1"]
    cfg = load_config(
        hw["preset"], f"qwen35-{size.lower()}",
        overrides=[f"model.path={model_dir}", f"train.microbatch_tokens={hw['microbatch'][size]}",
                   *hw["extra"][size], *FINAL_OVERRIDES, *extra],
    )  # fmt: skip
    return cfg.to_dict()


def model_dir(size: str, root: Path) -> Path:
    from tjev.campaign.plan import model_config

    config = model_config(size)
    if size == "9B":
        config = dataclasses.replace(config, tie_word_embeddings=False)
    folder = root / f"Qwen3.5-{size}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config.to_hf(), indent=1))
    return folder


def one(size: str, arm: str, bucket: int, overrides: list[str], root: Path, out: Path) -> dict:
    from tjev.data.pack import rows_for

    folder = model_dir(size, root)
    raw = run_config(size, folder, overrides)
    tag = f"{size}-{arm}-T{bucket}"
    path = root / f"{tag}.json"
    path.write_text(json.dumps(raw))
    tokens = raw["train"]["microbatch_tokens"]
    rows = rows_for(bucket, tokens, 1)
    micro = max(1, raw["train"]["tokens_per_step"] // tokens)
    cmd = [sys.executable, str(Path(__file__).with_name("aot_tpu.py")), str(path), str(folder),
           "--bucket", str(bucket), "--rows", str(rows), "--micro", str(micro), "--tag", tag]  # fmt: skip
    # concurrent compile-only clients: libtpu's lockfile otherwise admits one process
    env = dict(os.environ, JAX_PLATFORMS="", ALLOW_MULTIPLE_LIBTPU_LOAD="1")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode:
        return {"tag": tag, "error": (p.stderr or p.stdout)[-2000:]}
    report = json.loads(p.stdout[p.stdout.index("{") :])
    report |= {"size": size, "arm": arm, "rows": rows, "micro": micro, "overrides": overrides}
    (out / f"{tag}.json").write_text(json.dumps(report, indent=1))
    return report


def row(r: dict) -> str:
    if "error" in r:
        last = r["error"].strip().splitlines()[-1] if r["error"].strip() else "?"
        return f"| {r['tag']} | error: {last[:160]} |"
    m = r["memory_gib"]
    args, temp, outp, alias = (m[k] for k in ("argument_size_in_bytes", "temp_size_in_bytes",
                                              "output_size_in_bytes", "alias_size_in_bytes"))  # fmt: skip
    total = args + temp + outp - alias  # donated arguments alias the outputs
    lora = {k: v for k, v in r.get("kernel_calls", {}).items() if "lora" in k or "fwd" in k}
    return (f"| {r['size']} | {r['arm']} | {r['bucket']} | {r['rows']}x{r['micro']} | "
            f"{r['remat']} | {temp:.2f} | {args:.2f} | {outp - alias:.2f} | {total:.2f} | "
            f"{LIMIT_GIB - total:+.2f} | {lora or ''} |")  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="4B,9B")
    p.add_argument("--buckets", default="1024,2048,4096")
    p.add_argument("--arms", default="baseline,flat,remat,candidate")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--out", default="reports/aot")
    p.add_argument("--extra", nargs="*", default=[], help="more overrides, on every arm")
    p.add_argument("--suffix", default="", help="appended to the arm names (with --extra)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="aot-arms-"))
    todo = [(s, a + args.suffix, int(b), [*arms(s)[a], *args.extra]) for s in args.sizes.split(",")
            for a in args.arms.split(",") for b in args.buckets.split(",")]  # fmt: skip
    print("| size | arm | bucket | rows x micro | remat | temp | args | out | total | "
          f"vs {LIMIT_GIB:g} GiB | lora kernels |\n|---|---|---|---|---|---|---|---|---|---|---|",
          flush=True)  # fmt: skip
    results = []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(one, *t, root, out) for t in todo]
        for f in futures:  # in order
            r = f.result()
            results.append(r)
            print(row(r), flush=True)
    (out / "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
