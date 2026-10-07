"""The numbers of an xprof trace JSON, for W&B: where the device time of a train step goes
(the fallback of :mod:`tjev.campaign.xprof_tools`, which reads the xplane with xprof).

    summarize(trace_dir) -> {"steps", "step_ms", "busy_fraction", "by_category",
                             "by_scope", "top_ops"}

Reads the ``*.trace.json.gz`` that ``jax.profiler.trace`` writes (Chrome trace format).
Only the leaf device ops of the TPU's "XLA Ops" line count: a ``while`` op contains its
body's ops, so counting it too would double the time. Each op goes to the innermost
model scope of its ``tf_op`` path (the ``jax.named_scope`` names of the model: gdn_core,
attention_core, lora, base, mlp, …), else ``other``. The interactive views (op profile,
roofline, memory, trace) stay in xprof: ``xprof --logdir DIR`` on the raw trace.
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import re
from pathlib import Path
from typing import Any

SCOPES = (
    "embed", "norm", "final_norm", "readout", "loss", "base", "lora", "mlp",
    "gdn_proj", "gdn_conv", "gdn_gates", "gdn_qk_norm", "gdn_core", "gdn_rule", "gdn_prepare",
    "gdn_local", "gdn_inverse", "gdn_scan", "gdn_out", "attn_proj", "attention_core",
    "attn_out", "grad_accum", "optimizer", "lora_merge", "health", "kl_base", "residual",
)  # fmt: skip


def profile_options(detail: bool = False):
    """The capture every xprof tool can read: HLO protos in the xplane (graph viewer, memory
    viewer, op profile, roofline), device and host activity, no Python tracer (its overhead
    distorts the step; none of these tools needs it). Two fixed modes (docs/profiling.md):
    the timing capture, one event per XLA op or custom call plus the perf counters; the
    detail capture (``detail``, or ``TJEV_PROFILE_DETAIL=1`` in a training run's
    environment; never timed), sync-flag / semaphore waits too, where a Pallas pipeline or
    a DMA stalls."""
    import jax

    detail = detail or os.environ.get("TJEV_PROFILE_DETAIL") == "1"
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 2
    options.enable_hlo_proto = True
    mode = "TRACE_COMPUTE_AND_SYNC" if detail else "TRACE_ONLY_XLA"
    options.advanced_configuration = {"tpu_trace_mode": mode, "tpu_perf_counters": True}
    return options


def _trace_file(trace_dir: Path) -> Path | None:
    """The newest timing trace (the untimed detail capture, ``…/detail/…``, only if alone)."""
    found = sorted(trace_dir.rglob("*.trace.json.gz"))
    timed = [p for p in found if "detail" not in p.relative_to(trace_dir).parts]
    return (timed or found or [None])[-1]


def scope_of(tf_op: str) -> str:
    """The innermost model scope in an op's name path (``jit(step)/…/gdn_core/…``)."""
    if "/lora/merge" in tf_op:  # merged LoRA's W + s·AB (<proj>/lora/merge; was lora_merge)
        return "lora_merge"
    parts = [p for p in re.split(r"[/()]", tf_op) if p]
    for part in reversed(parts):
        if part in SCOPES:
            return part
    return "other"


_UNITS = {"": 1, "b": 1, "k": 1e3, "kb": 1e3, "kib": 1024, "m": 1e6, "mb": 1e6,
          "mib": 1024**2, "g": 1e9, "gb": 1e9, "gib": 1024**3}  # fmt: skip


def bytes_of(value: Any) -> float | None:
    """A trace arg's byte count: a number, or a string like ``"1048576"``, ``"1.5MiB"``,
    ``"3.2 GB"`` (None when it is neither)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    m = re.fullmatch(
        r"\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)\s*", value.replace(",", "")
    )
    if not m or m[2].lower() not in _UNITS:
        return None
    return float(m[1]) * _UNITS[m[2].lower()]


_KEEP_THREADS = ("XLA Ops", "Steps", "XLA Modules")
_KEEP_ARGS = ("tf_op", "hlo_category", "bytes_accessed", "raw_bytes_accessed", "step_num")
_CHUNK = 1 << 24  # 16 MB of trace text per read


def stream_events(path: Path) -> list[dict[str, Any]]:
    """The events a summary needs, streamed out of a (one-line, gzipped) Chrome trace: the
    metadata, TPU:0's "XLA Ops" / "Steps" / "XLA Modules" rows and the host step markers
    (``StepTraceAnnotation``: name ``train``, a ``step_num`` arg). A 9B step's trace is
    3.7 GB of JSON: ``json.loads`` of it does not fit an 18 GB machine, this keeps ~1/10."""
    dec = json.JSONDecoder()
    procs: dict[Any, str] = {}
    threads: dict[tuple[Any, Any], str] = {}
    kept: list[dict[str, Any]] = []

    def keep(e: dict[str, Any]) -> bool:
        if e.get("ph") == "M":
            if e.get("name") == "process_name":
                procs[e.get("pid")] = e.get("args", {}).get("name", "")
            elif e.get("name") == "thread_name":
                threads[(e.get("pid"), e.get("tid"))] = e.get("args", {}).get("name", "")
            return True
        if e.get("ph") != "X":
            return False
        if e.get("name") == "train" and "step_num" in e.get("args", {}):
            return True
        pid = e.get("pid")
        if pid not in procs:  # metadata not seen yet: keep what looks like a device row
            return "hlo_category" in e.get("args", {}) or "tf_op" in e.get("args", {})
        return bool(re.search(r"TPU:0", procs[pid])) and (
            threads.get((pid, e.get("tid"))) in _KEEP_THREADS
        )

    with gzip.open(path, "rt", encoding="utf-8") as f:
        buf = f.read(_CHUNK)
        start = buf.find('"traceEvents"')
        i = buf.index("[", max(start, 0)) + 1  # {"traceEvents": [...]} or a bare list
        while True:
            j = i
            while j < len(buf) and buf[j] in " \n\r\t,":
                j += 1
            if j < len(buf) and buf[j] == "]":
                break
            try:
                if j >= len(buf):
                    raise json.JSONDecodeError("end of chunk", buf, j)
                e, i = dec.raw_decode(buf, j)
            except json.JSONDecodeError:
                more = f.read(_CHUNK)
                if not more:
                    break
                buf, i = buf[j:] + more, 0
                continue
            if i > _CHUNK:
                buf, i = buf[i:], 0
            if isinstance(e, dict) and keep(e):
                if e.get("ph") == "X":  # only what a summary reads (the args are most of it)
                    args = e.get("args", {})
                    e = {k: e[k] for k in ("ph", "pid", "tid", "ts", "dur", "name") if k in e}
                    e["args"] = {k: args[k] for k in _KEEP_ARGS if k in args}
                kept.append(e)
    return kept


def summarize(trace_dir: Path, steps: int = 3, top: int = 15) -> dict[str, Any] | None:
    """Where the device time goes (shares of the busy time; ``ms_by_scope``: milliseconds
    per step) and, when the trace args carry them (``bytes_accessed``, ``raw_bytes_accessed``:
    the HLO cost analysis of each op: estimates, not measurements), the bytes per step by
    scope and per top op. The fallback of ``xprof_tools.summarize_xplane`` (no xprof needed).

    Only the train programs count when the window has others (an eval): a program (an
    "XLA Modules" row) is a train step when one of its ops has a backward path. The steps are
    the train programs' executions; without module rows the host step markers, else the
    device "Steps" rows, else ``steps`` (``step_source`` says which)."""
    path = _trace_file(Path(trace_dir))
    if path is None:
        return None
    events = stream_events(path)
    procs = {e["pid"]: e["args"]["name"] for e in events
             if e.get("ph") == "M" and e.get("name") == "process_name"}  # fmt: skip
    threads = {(e["pid"], e["tid"]): e["args"]["name"] for e in events
               if e.get("ph") == "M" and e.get("name") == "thread_name"}  # fmt: skip
    device = {p for p, n in procs.items() if re.search(r"/device:TPU:0|TPU:0", n)}
    line = lambda e: threads.get((e.get("pid"), e.get("tid")))  # noqa: E731
    xs = [e for e in events if e.get("ph") == "X"]
    ops = [e for e in xs if e["pid"] in device and line(e) == "XLA Ops"
           and e.get("args", {}).get("hlo_category") != "while"]  # fmt: skip
    if not ops:
        return None
    ops.sort(key=lambda e: e["ts"])
    modules = sorted((e for e in xs if e["pid"] in device and line(e) == "XLA Modules"),
                     key=lambda e: e["ts"])  # fmt: skip
    # each op to the module execution around it; train modules: any op with a backward
    owner: list[int] = []
    m = 0
    for e in ops:
        while m < len(modules) and modules[m]["ts"] + modules[m]["dur"] < e["ts"]:
            m += 1
        inside = m < len(modules) and modules[m]["ts"] <= e["ts"]
        owner.append(m if inside else -1)
    module_name = lambda k: modules[k]["name"] if k >= 0 else "?"  # noqa: E731
    train_names = {
        module_name(k) for e, k in zip(ops, owner, strict=True)
        if "grad_accum" in e.get("args", {}).get("tf_op", "")
        or "transpose(" in e.get("args", {}).get("tf_op", "")
    }  # fmt: skip
    if train_names:
        ops = [e for e, k in zip(ops, owner, strict=True) if module_name(k) in train_names]
    host_steps = {e["args"]["step_num"] for e in xs if e.get("name") == "train"
                  and "step_num" in e.get("args", {})}  # fmt: skip
    step_rows = [e for e in xs if e["pid"] in device and line(e) == "Steps"]
    train_modules = [e for e in modules if e["name"] in train_names]
    busy = sum(e["dur"] for e in ops)
    if train_modules:
        steps, source = len(train_modules), "module executions"
        span = sum(e["dur"] for e in train_modules)
    elif host_steps:  # the ops' span: it also counts the host gaps between the steps
        steps, source = len(host_steps), "host step markers"
        span = max(e["ts"] + e["dur"] for e in ops) - min(e["ts"] for e in ops)
    elif step_rows:  # one row per program execution (an eval's too)
        steps, source = len(step_rows), "device steps"
        span = sum(e["dur"] for e in step_rows)
    else:
        source = "given"
        span = max(e["ts"] + e["dur"] for e in ops) - min(e["ts"] for e in ops)
    by_cat: collections.Counter[str] = collections.Counter()
    by_scope: collections.Counter[str] = collections.Counter()
    by_op: collections.Counter[str] = collections.Counter()
    bytes_scope: collections.Counter[str] = collections.Counter()
    bytes_op: collections.Counter[str] = collections.Counter()
    raw_total, have_bytes = 0.0, False
    for e in ops:
        args = e.get("args", {})
        scope, op = scope_of(args.get("tf_op", "")), re.sub(r"\.\d+$", "", e["name"])
        by_cat[args.get("hlo_category", "?")] += e["dur"]
        by_scope[scope] += e["dur"]
        by_op[op] += e["dur"]
        accessed = bytes_of(args.get("bytes_accessed"))
        if accessed is not None:
            have_bytes = True
            bytes_scope[scope] += accessed
            bytes_op[op] += accessed
        raw = bytes_of(args.get("raw_bytes_accessed"))
        raw_total += raw or 0.0
    share = lambda c: {k: round(100 * v / busy, 2) for k, v in c.most_common()}  # noqa: E731
    per_step_gb = lambda v: round(v / steps / 1e9, 4)  # noqa: E731
    hbm: dict[str, Any] = {}
    if have_bytes:  # XLA cost-model bytes (VMEM traffic included): an estimate
        hbm = {
            "hbm_gb_per_step": per_step_gb(sum(bytes_scope.values())),
            "hbm_gb_by_scope": {k: per_step_gb(v) for k, v in bytes_scope.most_common()},
            "top_ops_by_bytes": [
                {"op": k, "gb_per_step": per_step_gb(v)} for k, v in bytes_op.most_common(top)
            ],
            "bytes_source": "XLA cost model (estimated)",
        }
        if raw_total:
            hbm["raw_hbm_gb_per_step"] = per_step_gb(raw_total)
    busy_fraction = round(min(busy / span, 1.0), 4) if span else None
    return {
        "steps": steps,
        "step_ms": round(span / 1e3 / steps, 2),
        "busy_fraction": busy_fraction,
        "busy": busy_fraction,
        "step_source": source,
        "ops_per_step": len(ops) // steps,
        "by_category": share(by_cat),
        "by_scope": share(by_scope),
        "ms_by_scope": {k: round(v / 1e3 / steps, 3) for k, v in by_scope.most_common()},
        "top_ops": [
            {
                "op": k,
                "percent": round(100 * v / busy, 2),
                **({"gb_per_step": per_step_gb(bytes_op[k])} if have_bytes else {}),
            }
            for k, v in by_op.most_common(top)
        ],
        "source": "trace json",
        **hbm,
    }
