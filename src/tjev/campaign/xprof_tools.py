"""Where the device time of a train step goes, from the xprof ``*.xplane.pb`` (no trace JSON).

    summarize_xplane(session_dir) -> {"steps", "step_ms", "busy_fraction", "by_category",
                                      "by_scope", "ms_by_scope", "top_ops", ...}

Reads the xplane with stable xprof's converters (``pip install tjev[profile]``; imported
lazily): ``xspace_to_tool_data(xplane, tool, params)`` gives the UI's tables as Google
DataTable JSON in seconds and a few GB of RAM, where the trace JSON of a 9B step is 3.7 GB of
text. The tools read here:

* ``hlo_stats``: one row per HLO op and program (self time, occurrences, the model FLOP rate,
  the HBM bandwidth, bound-by, the remat flag, the jax op path ``tf_op_name``): the scope,
  phase and roofline tables.
* ``roofline_model``: the device's peaks and ridge points, the program totals and each op's
  roofline efficiency.
* ``overview_page``: the step time (the ``StepTraceAnnotation`` steps) and the duty cycle.
* ``memory_profile``: every module execution of the window (counts the traced steps).
* ``utilization_viewer``: the session's hardware counters (MXU busy cycles, HBM bytes).
* ``memory_viewer``: the train program's static HBM peak and the buffers live at it.

Estimated vs measured (``fields`` in the result says which is which):

* measured: op self times (hence every ms, share and step time), occurrences, the session's
  perf counters (``measured``: MXU busy, HBM bytes / bandwidth utilisation).
* estimated: FLOPs and bytes come from XLA's cost model (``model_flop_rate``, ``hbm_bw``,
  ``operational_intensity``, ``bound_by``) divided by the measured time. On v6e the per-op
  HBM bandwidth is model-derived: per-op counter sampling exists from TPU v7 only. The cost
  model counts some in-place updates and Pallas kernels by their ``CostEstimate``.
* static: the memory viewer's buffers (the compiler's buffer assignment, not a measurement).

Only the train programs count in the step tables (a program is a train step when one of its
ops sits under ``grad_accum`` or a ``transpose(...)`` path): an eval in the window would
otherwise add forward-only time. ``all_programs`` keeps the shares over every program.

Phases, from the jax op path of the train programs: ``transpose(jvp(...))`` is the backward,
``…/checkpoint/rematted_computation/…`` (the HLO remat flag) the recomputed forward of the
remat'd layers, ``jvp(...)`` the forward; everything else (the optimizer, the health
reductions, loop plumbing, unnamed copies) is ``other``.

    python -m tjev.campaign.xprof_tools SESSION_DIR [--out DIR]   # writes summary.json, …
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from tjev.campaign.hlo_labels import fine_path
from tjev.campaign.xprof_summary import scope_of

GIB = 1024**3  # xprof's bandwidths are GiB/s, its FLOP rates GFLOP/s, its times µs
PHASES = ("fwd", "bwd", "remat", "other")
FIELDS = {
    "measured": [
        "steps", "step_ms", "device_ms_per_step", "busy_fraction", "by_scope", "ms_by_scope",
        "by_category", "by_phase", "scope_phase_ms", "top_ops[].percent", "top_ops[].ms",
        "measured.*",
    ],
    "estimated (XLA cost model / measured time)": [
        "scope_stats[].tflops", "scope_stats[].hbm_gbps", "scope_stats[].bound_by",
        "scope_stats[].intensity", "top_ops[].tflops", "top_ops[].hbm_gbps",
        "top_ops[].intensity", "top_ops[].bound_by", "top_ops[].roofline_efficiency",
        "program.*",
    ],
    "static (compiler buffer assignment)": ["memory.*"],
    "device spec (xprof)": ["device.*"],
}  # fmt: skip


def find_xplane(path: Path) -> Path | None:
    """The newest ``*.xplane.pb`` under ``path`` (a run's profile dir or a session dir)."""
    path = Path(path)
    if path.is_file():
        return path
    found = sorted(path.rglob("*.xplane.pb"), key=lambda p: (p.parent.name, p.name))
    # the detail capture (profile/detail/…) is untimed: never the step summary
    timed = [p for p in found if "detail" not in p.relative_to(path).parts]
    return (timed or found or [None])[-1]


def table(data: Any, index: int = 0) -> list[dict[str, Any]]:
    """A DataTable (``{cols, rows: [{c: [{v}]}]}``, or a list of them) as dicts by column id."""
    if isinstance(data, bytes | str):
        data = json.loads(data) if data else {}
    if isinstance(data, list):
        data = data[index] if len(data) > index else {}
    ids = [c.get("id") or c.get("label") for c in data.get("cols", [])]
    return [
        dict(zip(ids, [(c or {}).get("v") for c in row.get("c", [])], strict=False))
        for row in data.get("rows", [])
    ]


def phase_of(tf_op: str, remat: Any = None) -> str:
    """fwd, bwd, remat or other, from an op's jax path (``remat``: hlo_stats' flag)."""
    if "rematted_computation" in tf_op or str(remat).lower() in ("yes", "true", "1"):
        return "remat"
    if "transpose(" in tf_op:
        return "bwd"
    if "jvp(" in tf_op:
        return "fwd"
    return "other"


def readable_kind(category: str, name: str) -> str:
    """xprof's HLO category in our words (``convolution fusion`` → matmul, …)."""
    if category == "custom-call":
        return "allocate (scan buffer)" if name.startswith("custom-call") else "Pallas kernel"
    if "dynamic-update-slice" in name:
        return "scan stack write (DUS)"
    if "dynamic-slice" in name:
        return "scan stack read (DS)"
    return {
        "convolution fusion": "matmul (fused)",
        "convolution": "matmul",
        "data formatting": "copy / relayout",
        "loop fusion": "elementwise (fused)",
        "non-fusion elementwise": "elementwise",
        "input fusion": "reduction (fused)",
        "copy-start": "async copy",
        "copy-done": "async copy",
    }.get(category, category)


def op_kind(name: str) -> str:
    """An HLO op's kind: its name without the numeric and ``.refs_N_`` suffixes."""
    return re.sub(r"(\.\d+|\.refs_\d+_)+$", "", name)


def program_kinds(rows: list[dict[str, Any]]) -> dict[str, str]:
    """``train`` for the programs with a backward (grad_accum / transpose path), else ``eval``."""
    kinds: dict[str, str] = {}
    for r in rows:
        pid, name = str(r.get("program_id")), r.get("tf_op_name") or ""
        if "grad_accum" in name or "transpose(" in name:
            kinds[pid] = "train"
        else:
            kinds.setdefault(pid, "eval")
    return kinds


def executions_from_ops(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Each program's execution count, from its top-level ops (outside every loop): the most
    common occurrence count; a program without any: its ops' smallest count. A fallback
    when the memory profile lists no modules."""
    occ: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    least: dict[str, int] = {}
    for r in rows:
        name, n = r.get("tf_op_name") or "", int(_num(r.get("occurrences")))
        pid = str(r.get("program_id"))
        if n:
            least[pid] = min(least.get(pid, n), n)
            if name and "while" not in name:
                occ[pid][n] += 1
    return {pid: (occ[pid].most_common(1)[0][0] if occ[pid] else n) for pid, n in least.items()}


def executions_from_modules(modules: list[dict[str, Any]]) -> dict[str, float]:
    """Each program's executions in the window from the memory profile's ``hloModules``
    timeline (``jit_…(<program id>)`` per execution, start and end in ps). An execution cut
    by the start or the stop of the trace counts for its traced share (its duration over
    the program's median), so a truncated step does not count as a whole one."""
    spans: dict[str, list[float]] = collections.defaultdict(list)
    for m in modules:
        if found := re.search(r"\((\d+)\)\s*$", m.get("name", "")):
            dur = _num(m.get("endTimePs")) - _num(m.get("startTimePs"))
            spans[found[1]].append(dur)
    out: dict[str, float] = {}
    for pid, durs in spans.items():
        known = sorted(d for d in durs if d > 0)
        if not known:
            out[pid] = float(len(durs))
            continue
        median = known[len(known) // 2]
        n = sum(min(d / median, 1.0) if d > 0 else 1.0 for d in durs)
        out[pid] = round(n, 2) if abs(n - round(n)) > 0.02 else float(round(n))
    return out


def _num(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _shares(counter: dict[str, float], total: float) -> dict[str, float]:
    ordered = sorted(counter.items(), key=lambda kv: -kv[1])
    return {k: round(100 * v / total, 2) for k, v in ordered} if total else {}


def _weighted(acc: dict[str, float], time_us: float) -> dict[str, Any]:
    """Time-weighted rates of an accumulator (``flop``: GFLOP, ``hbm``: GiB, ``mem``: GiB)."""
    t = time_us / 1e6
    out: dict[str, Any] = {
        "tflops": round(acc["flop"] / t / 1e3, 1) if t else None,
        "hbm_gbps": round(acc["hbm"] * GIB / t / 1e9, 1) if t else None,
        "intensity": round(acc["flop"] * 1e9 / (acc["mem"] * GIB), 1) if acc["mem"] else None,
    }
    bound = {k[6:]: v for k, v in acc.items() if k.startswith("bound:")}
    out["bound_by"] = _shares(bound, time_us)
    return out


def attribute(
    rows: list[dict[str, Any]],
    executions: dict[str, float] | None = None,
    top: int = 15,
    programs: str = "train",
    roofline: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The step tables from hlo_stats rows (pure: the unit-tested part).

    ``programs``: ``train`` (the programs with a backward; all of them when none has one) or
    ``all``. ``executions``: per program id (default: from the ops' occurrences); the steps
    are the train programs' executions. ``roofline``: roofline_model rows, joined by
    (program, op) for each top op's roofline efficiency."""
    kinds = program_kinds(rows)
    if programs == "train" and "train" in kinds.values():
        keep = {p for p, k in kinds.items() if k == "train"}
    else:
        keep = set(kinds)
    rows = [r for r in rows if str(r.get("program_id")) in keep and _num(r.get("total_self_time"))]
    if not rows:
        return None
    execs = executions_from_ops(rows) | (executions or {})
    steps = sum(execs.get(p, 0) for p in keep) or 1
    busy = sum(_num(r["total_self_time"]) for r in rows)  # µs over the window

    def acc() -> collections.defaultdict[str, float]:
        return collections.defaultdict(float)

    by_scope: collections.Counter[str] = collections.Counter()
    by_cat: collections.Counter[str] = collections.Counter()
    by_readable: collections.Counter[str] = collections.Counter()
    by_phase: collections.Counter[str] = collections.Counter()
    scope_phase: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    scope_acc: dict[str, collections.defaultdict[str, float]] = collections.defaultdict(acc)
    phase_acc: dict[str, collections.defaultdict[str, float]] = collections.defaultdict(acc)
    total_acc = acc()
    roof = {(str(r.get("hlo_module_id")), r.get("operation")): r for r in roofline or []}

    def phase(r: dict[str, Any]) -> str:
        if kinds[str(r["program_id"])] != "train":
            return "eval"
        return phase_of(r.get("tf_op_name") or "", r.get("hlo_rematerialization"))

    by_kind: collections.Counter[tuple[str, str, str, str]] = collections.Counter()
    by_path: collections.Counter[tuple[str, str]] = collections.Counter()
    path_acc: dict[tuple[str, str], collections.defaultdict[str, float]] = collections.defaultdict(
        acc
    )
    for r in rows:
        t = _num(r["total_self_time"])
        name = r.get("tf_op_name") or ""
        scope, ph = scope_of(name), phase(r)
        kind = op_kind(r.get("hlo_op_name") or "?")
        by_kind[(kind, r.get("category") or "?", scope, ph)] += t
        where = (fine_path(name) or kind, ph)
        by_path[where] += t
        by_scope[scope] += t
        by_cat[r.get("category") or "?"] += t
        by_readable[readable_kind(r.get("category") or "", r.get("hlo_op_name") or "")] += t
        by_phase[ph] += t
        scope_phase[scope][ph] += t
        flop = _num(r.get("model_flop_rate")) * t / 1e6  # GFLOP
        hbm = _num(r.get("hbm_bw")) * t / 1e6  # GiB
        mem = _num(r.get("measured_memory_bw")) * t / 1e6  # GiB, any memory (cost model)
        for a in (scope_acc[scope], phase_acc[ph], total_acc, path_acc[where]):
            a["flop"] += flop
            a["hbm"] += hbm
            a["mem"] += mem
            a["time"] += t
            a["bound:" + (r.get("bound_by") or "Unknown")] += t
    ms = lambda us: round(us / 1e3 / steps, 3)  # noqa: E731  per step
    ordered = sorted(rows, key=lambda r: -_num(r["total_self_time"]))
    top_ops = []
    for r in ordered[:top]:
        t = _num(r["total_self_time"])
        rf = roof.get((str(r["program_id"]), r.get("hlo_op_name")), {})
        top_ops.append(
            {
                "op": r.get("hlo_op_name"),
                "where": fine_path(r.get("tf_op_name") or "")
                or op_kind(r.get("hlo_op_name") or ""),
                "program": str(r["program_id"]),
                "category": r.get("category"),
                "kind": readable_kind(r.get("category") or "", r.get("hlo_op_name") or ""),
                "scope": scope_of(r.get("tf_op_name") or ""),
                "phase": phase(r),
                "percent": round(100 * t / busy, 2),
                "ms": ms(t),
                "tflops": round(_num(r.get("model_flop_rate")) / 1e3, 1),
                "hbm_gbps": round(_num(r.get("hbm_bw")) * GIB / 1e9, 1),
                "intensity": round(_num(r.get("operational_intensity")), 1),
                "bound_by": r.get("bound_by"),
                "dma_stall_percent": round(_num(r.get("dma_stall_percent")), 2),
                **(
                    {"roofline_efficiency": round(_num(rf.get("roofline_efficiency")), 3)}
                    if rf
                    else {}
                ),
                "tf_op": (r.get("tf_op_name") or "")[:200],
            }
        )
    return {
        "steps": steps,
        "device_ms_per_step": round(busy / 1e3 / steps, 2),
        "hlo_ops": len(rows),  # distinct (program, op) rows
        "programs": {p: {"kind": kinds[p], "executions": execs.get(p, 0)} for p in sorted(keep)},
        # ops named after their loop that took a neighbour's model path (reattribute_rows)
        "reattributed_ms": ms(
            sum(_num(r["total_self_time"]) for r in rows if "tf_op_original" in r)
        ),
        "by_category": _shares(by_cat, busy),
        "by_kind": _shares(by_readable, busy),
        "by_scope": _shares(by_scope, busy),
        "ms_by_scope": {k: ms(v) for k, v in by_scope.most_common()},
        "by_phase": _shares(by_phase, busy),
        "ms_by_phase": {k: ms(v) for k, v in by_phase.most_common()},
        "scope_phase_ms": {
            s: {p: ms(c[p]) for p in (*PHASES, "eval") if c.get(p)}
            for s, c in sorted(scope_phase.items(), key=lambda kv: -sum(kv[1].values()))
        },
        "scope_stats": {
            s: {"percent": round(100 * by_scope[s] / busy, 2), "ms": ms(by_scope[s])}
            | _weighted(scope_acc[s], by_scope[s])
            for s, _ in by_scope.most_common()
        },
        "phase_stats": {p: _weighted(phase_acc[p], by_phase[p]) for p, _ in by_phase.most_common()},
        "total_stats": _weighted(total_acc, busy),
        "top_ops": top_ops,
        # every op of one model path (fine_path: layer_gdn/mlp/up_proj/lora/B/dot_general)
        # and phase, summed: the readable "where the time goes"
        "top_paths": [
            {"where": w, "phase": ph, "percent": round(100 * v / busy, 2), "ms": ms(v)}
            | {k: x for k, x in _weighted(path_acc[(w, ph)], v).items() if k != "intensity"}
            for (w, ph), v in by_path.most_common(top)
        ],
        # the same op kind (name without its numeric suffix) in one scope and phase, summed
        "top_kinds": [
            {
                "op": k,
                "category": c,
                "scope": s,
                "phase": ph,
                "percent": round(100 * v / busy, 2),
                "ms": ms(v),
            }
            for (k, c, s, ph), v in by_kind.most_common(top)
        ],
    }


def scope_phase_table(summary: dict[str, Any]) -> str:
    """The by-scope × phase table as text (ms per step; % of the step; TFLOP/s, HBM GB/s)."""
    phases = [p for p in (*PHASES, "eval") if p in summary.get("ms_by_phase", {})]
    head = f"{'scope':<16}" + "".join(f"{p:>9}" for p in phases)
    head += f"{'ms':>10}{'%':>7}{'TFLOP/s':>9}{'HBM GB/s':>10}  bound_by"
    lines = [head]
    for scope, by in summary["scope_phase_ms"].items():
        st = summary["scope_stats"][scope]
        bound = ", ".join(f"{k} {v:.0f}%" for k, v in list(st["bound_by"].items())[:2])
        lines.append(
            f"{scope:<16}" + "".join(f"{by.get(p, 0):>9.1f}" for p in phases)
            + f"{st['ms']:>10.1f}{st['percent']:>7.2f}{st['tflops'] or 0:>9.1f}"
            + f"{st['hbm_gbps'] or 0:>10.1f}  {bound}"
        )  # fmt: skip
    total = summary["total_stats"]
    lines.append(
        f"{'total':<16}" + "".join(f"{summary['ms_by_phase'].get(p, 0):>9.1f}" for p in phases)
        + f"{summary['device_ms_per_step']:>10.1f}{100:>7.2f}{total['tflops'] or 0:>9.1f}"
        + f"{total['hbm_gbps'] or 0:>10.1f}"
    )  # fmt: skip
    return "\n".join(lines)


# ---- the xprof side (lazy imports: tjev[profile]) -------------------------------------------


def _tool_raw(xplane: Path, tool: str, params: dict[str, Any] | None = None) -> Any:
    from xprof.convert import raw_to_tool_data as r

    data, _ = r.xspace_to_tool_data([str(xplane)], tool, params or {})
    return data


def _tool(xplane: Path, tool: str, params: dict[str, Any] | None = None) -> Any:
    data = _tool_raw(xplane, tool, params)
    if isinstance(data, bytes):
        data = data.decode()
    return json.loads(data) if data else None


def run_tools(xplane: Path, tools: tuple[str, ...] | None = None) -> dict[str, Any]:
    """The raw tool outputs (parsed JSON; ``{"error": …}`` per failed tool)."""
    from xprof.convert import raw_to_tool_data as r

    # lists the tools and, as a side effect, writes the per-module HLO protos next to the
    # xplane: memory_viewer and graph_viewer read those
    available = r.xspace_to_tool_names([str(xplane)])
    tools = tools or ("hlo_stats", "roofline_model", "overview_page", "memory_profile",
                      "utilization_viewer")  # fmt: skip
    out: dict[str, Any] = {"_available": available}
    for tool in tools:
        if tool not in available and not any(a.rstrip("@^") == tool for a in available):
            out[tool] = {"error": "not available for this profile"}
            continue
        try:
            out[tool] = _tool(xplane, tool)
        except Exception as e:  # one tool's failure leaves the others
            out[tool] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return out


def device_peaks(roofline_json: Any) -> dict[str, Any]:
    """The device's peaks and ridge points (roofline_model's properties), in TFLOP/s, GB/s."""
    p = (roofline_json[0] if isinstance(roofline_json, list) else roofline_json or {}).get("p", {})
    out: dict[str, Any] = {"device_type": p.get("device_type")}
    if "peak_flop_rate" in p:
        out["peak_tflops"] = round(_num(p["peak_flop_rate"]) / 1e3, 1)
    for k in ("peak_hbm_bw", "peak_vmem_read_bw", "peak_vmem_write_bw"):
        if k in p:
            out[k.replace("_bw", "_gbps")] = round(_num(p[k]) * GIB / 1e9, 1)
    for k in ("hbm_ridge_point", "vmem_read_ridge_point", "vmem_write_ridge_point"):
        if k in p:
            out[k] = round(_num(p[k]), 1)
    return out


def program_totals(roof_rows: list[dict[str, Any]], keep: set[str]) -> dict[str, Any]:
    """roofline_model's per-program totals (estimated FLOP and HBM rates, efficiency)."""
    out = {}
    for r in roof_rows:
        if r.get("category") == "Program" and r.get("step") == "Total":
            pid = str(r.get("hlo_module_id"))
            if pid in keep or pid == "0":
                out["all" if pid == "0" else pid] = {
                    "ms": round(_num(r.get("total_self_time") or r.get("total_time")) / 1e3, 1),
                    "tflops": round(_num(r.get("model_flop_rate")) / 1e3, 1),
                    "hbm_gbps": round(_num(r.get("hbm_bw")) * GIB / 1e9, 1),
                    "roofline_efficiency": round(_num(r.get("roofline_efficiency")), 3),
                    "bound_by": r.get("bound_by"),
                }
    return out


def measured_counters(util_json: Any) -> dict[str, Any]:
    """The session's hardware counters (utilization_viewer; all programs, idle included)."""
    rows = table(util_json)
    by = {}
    for r in rows:
        if int(_num(r.get("node"))) == 0:
            by.setdefault(r.get("name"), r)

    def frac(name: str) -> float | None:
        r = by.get(name)
        return round(_num(r["achieved"]) / _num(r["peak"]), 4) if r and _num(r["peak"]) else None

    out = {
        "scope": "session (every program of the window, idle included)",
        "mxu_busy_fraction": frac("Avg MXU Busy"),
        "mxu_two_busy_fraction": frac("2 MXU Busy"),
        "vector_alu_fraction": frac("Vector ALUs"),
        "vmem_load_fraction": frac("Vmem Loads"),
        "vmem_store_fraction": frac("Vmem Stores"),
        "xlu_busy_fraction": frac("Avg XLU Busy"),
        "hbm_bw_utilization": frac("HBM Rd+Wr - core 0"),
    }
    hbm = by.get("HBM Rd+Wr - core 0")
    if hbm:
        out["hbm_tb"] = round(_num(hbm["achieved"]) / 1e12, 3)
        read = by.get("HBM Read Ratio")
        if read and _num(read["peak"]):
            out["hbm_read_fraction"] = round(_num(read["achieved"]) / _num(read["peak"]), 3)
    return {k: v for k, v in out.items() if v is not None}


def _param_label(tf_op: str) -> str:
    """``frozen['blocks']['mlps'][3]['up_proj']['kernel'].value`` → ``frozen mlps[3].up_proj.kernel``."""
    keys = re.findall(r"\['([^']+)'\]|\[(\d+)\]", tf_op)
    parts = [k or f"[{i}]" for k, i in keys if (k or i) not in ("blocks", "value")]
    head = tf_op.split("[", 1)[0] or "param"
    return f"{head} " + ".".join(parts).replace(".[", "[") if parts else tf_op[:80]


def buffer_label(b: dict[str, Any], allocs: dict[str, dict] | None = None) -> tuple[str, str]:
    """(label, group) of a memory-viewer buffer: a scan residual by its checkpoint name, a
    parameter by its pytree path, a temporary by its op's model path."""
    name = re.sub(r"\{.*\}$", "", b.get("instructionName") or "")
    if allocs and name in allocs:
        a = allocs[name]
        if a.get("residual"):
            return a["label"], f"scan residual: {' / '.join(a['residual'])}"
        writer = "/".join((a.get("producer") or "XLA").split(" | ")[0].split("/")[:2])
        return a["label"], f"loop buffers: {writer}"
    tf_op = b.get("tfOpName") or ""
    if (b.get("groupName") or "") == "Parameter" or b.get("opCode") == "parameter":
        label = _param_label(tf_op) if tf_op else name
        root = re.split(r"[\[.]", tf_op, maxsplit=1)[0] or "parameter"
        kind = "frozen weights" if root == "frozen" else root
        return label, f"parameters: {kind}"
    if tf_op:
        path = fine_path(tf_op)
        top = "/".join(p for p in path.split("/")[:2])
        return f"{path} ({phase_of(tf_op)})", f"temporaries: {top}"
    return f"{name} ({b.get('opCode') or '?'})", f"{(b.get('groupName') or 'other').lower()}"


def peak_buffers(
    viewer: Any, top: int = 20, allocs: dict[str, dict] | None = None
) -> dict[str, Any]:
    """The static HBM peak of a program, the buffers live at it (memory_viewer) labelled
    (``allocs``: hlo_labels.label_allocations of the same module) and grouped."""
    if not isinstance(viewer, dict) or "peakHeapMib" not in viewer:
        return {}
    groups: collections.Counter[str] = collections.Counter()
    composition: dict[str, list[float]] = {}
    for b in viewer.get("maxHeap", []):
        mib = _num(b.get("logicalBufferSizeMib"))
        groups[b.get("groupName") or "?"] += mib
        _, group = buffer_label(b, allocs)
        entry = composition.setdefault(group, [0.0, 0])
        entry[0] += mib
        entry[1] += 1
    gib = lambda mib: round(_num(mib) / 1024, 3)  # noqa: E731
    return {
        "module": viewer.get("moduleName"),
        "peak_heap_gib": gib(viewer["peakHeapMib"]),
        "parameters_gib": gib(viewer.get("entryComputationParametersMib")),
        "live_out_gib": gib(viewer.get("maybeLiveOutMib")),
        "total_allocation_gib": gib(viewer.get("totalBufferAllocationMib")),
        "at_peak_by_group_gib": {k: gib(v) for k, v in groups.most_common()},
        "composition": [
            {"what": k, "gib": gib(v[0]), "buffers": v[1]}
            for k, v in sorted(composition.items(), key=lambda kv: -kv[1][0])
        ],
        "top_buffers": [
            {
                "what": buffer_label(b, allocs)[0][:200],
                "label": (b.get("label") or b.get("instructionName") or "")[:160],
                "instruction": b.get("instructionName"),
                "gib": gib(b.get("logicalBufferSizeMib")),
                "shape": b.get("shapeString"),
                "group": b.get("groupName"),
                "op": b.get("opCode"),
                "tf_op": (b.get("tfOpName") or "")[:160],
            }
            for b in viewer.get("maxHeapBySize", [])[:top]
        ],
    }


def _module_hlo(xplane: Path, module: str) -> str | None:
    """The optimized HLO text of a program (xprof's graph viewer, ``long_txt``)."""
    try:
        data = _tool_raw(xplane, "graph_viewer",
                         {"graph_viewer_options": {"node_name": "", "module_name": module,
                                                   "type": "long_txt"}})  # fmt: skip
    except Exception:
        return None
    return data.decode(errors="replace") if isinstance(data, bytes) else data


def _module_name(modules: list[dict[str, Any]], pid: str) -> str:
    found = next((m["name"] for m in modules if m.get("name", "").endswith(f"({pid})")), None)
    return found or f"jit_step({pid})"


def _train_hlo(
    xplane: Path, rows: list[dict[str, Any]], modules: list[dict[str, Any]], raw: dict[str, Any]
) -> dict[str, Any]:
    """{program id: parsed HLO} of the train programs; the main one's text in raw["hlo_text"]."""
    from tjev.campaign.hlo_labels import parse_hlo

    kinds = program_kinds(rows)
    train = [p for p, k in kinds.items() if k == "train"]
    execs = executions_from_modules(modules)
    out = {}
    for pid in sorted(train, key=lambda p: -execs.get(p, 0)):
        text = _module_hlo(xplane, _module_name(modules, pid))
        if not text:
            continue
        if "hlo_text" not in raw:
            raw["hlo_text"] = text
        op_names = {r["hlo_op_name"]: r["tf_op_name"] for r in rows
                    if str(r.get("program_id")) == pid and r.get("tf_op_name")}  # fmt: skip
        out[pid] = parse_hlo(text, op_names)
    return out


def reattribute_rows(rows: list[dict[str, Any]], parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """hlo_stats rows with the loop-named ops (``…/while``) given their neighbour's model path
    (``hlo_labels.reattribute``); the original name kept in ``tf_op_original``."""
    from tjev.campaign.hlo_labels import reattribute

    out = []
    for r in rows:
        module = parsed.get(str(r.get("program_id")))
        name = r.get("tf_op_name") or ""
        if module is not None and (not name or name.rstrip(":").endswith("while")):
            new = reattribute(module, r.get("hlo_op_name") or "", r.get("category") or "")
            if new:
                r = r | {"tf_op_name": new, "tf_op_original": name}
        out.append(r)
    return out


def pick_bucket(residuals: dict[str, Any] | None, allocs_shapes: list[str]) -> dict | None:
    """The bucket of residuals.json whose stacked shapes match the most allocations."""
    from tjev.campaign.hlo_labels import shape_key
    from tjev.campaign.residuals import stack_labels

    if not residuals:
        return None
    buckets = residuals.get("buckets", {"": residuals})
    keys = [shape_key(sh) for sh in allocs_shapes]
    best = max(buckets.values(), key=lambda r: sum(k in stack_labels(r) for k in keys))
    return best if any(k in stack_labels(best) for k in keys) else None


def dynamic_memory(profile: Any) -> dict[str, Any]:
    """The allocator's peak over the window (memory_profile: measured, all programs)."""
    try:
        summ = next(iter(profile["memoryProfilePerAllocator"].values()))["profileSummary"]
    except (KeyError, TypeError, StopIteration):
        return {}
    peak = summ.get("peakStats", {})
    return {
        "peak_in_use_gib": round(_num(peak.get("peakBytesInUse")) / GIB, 3),
        "peak_lifetime_gib": round(_num(summ.get("peakBytesUsageLifetime")) / GIB, 3),
        "capacity_gib": round(_num(summ.get("memoryCapacity")) / GIB, 3),
        "fragmentation": peak.get("fragmentation"),
    }


def summarize_xplane(
    session_dir: Path,
    top: int = 15,
    programs: str = "train",
    raw: dict[str, Any] | None = None,
    memory: bool = True,
    residuals: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """The step summary of a profile (``session_dir``: a run's profile dir, an xprof session
    dir or the xplane itself). The keys of ``xprof_summary.summarize`` plus the phase and
    roofline tables, the device peaks, the measured counters and the HBM peak.
    ``raw``: tool outputs already read (``run_tools``). ``residuals``: residuals.json of the
    run (tjev.campaign.residuals): names the scan residual buffers of the HBM peak."""
    xplane = find_xplane(Path(session_dir))
    if xplane is None:
        return None
    raw = raw or run_tools(xplane)
    rows = table(raw.get("hlo_stats") or {})
    if not rows:
        return None
    roof_rows = table(raw.get("roofline_model") or {})
    modules = (raw.get("memory_profile") or {}).get("hloModules") or []
    execs = executions_from_modules(modules)
    # the train programs' HLO (graph viewer): ops named after their loop get the model path
    # of their nearest named neighbour (XLA's layout copies inside the layer scans)
    parsed = _train_hlo(xplane, rows, modules, raw) if memory else {}
    if parsed:
        rows = reattribute_rows(rows, parsed)
    out = attribute(rows, execs, top=top, programs=programs, roofline=roof_rows)
    if out is None:
        return None
    out["step_source"] = "module executions" if execs else "op occurrences"
    for m in modules:  # jit_train_step_T1024_mb32(<id>): the program's own name
        if (found := re.search(r"^(.*)\((\d+)\)\s*$", m.get("name", ""))) and found[2] in out[
            "programs"
        ]:
            out["programs"][found[2]]["name"] = found[1]
    overview = raw.get("overview_page")
    step_ms, busy = None, None
    if isinstance(overview, list):
        props = [t.get("p", {}) for t in overview if isinstance(t, dict)]
        for p in props:
            if p.get("steptime_ms_average") and _num(p["steptime_ms_average"]):
                step_ms = _num(p["steptime_ms_average"])
                out["annotated_steps"] = len(table(overview, props.index(p)))
            if p.get("device_duty_cycle_percent"):
                busy = _num(p["device_duty_cycle_percent"].rstrip("%")) / 100
    # the annotated step time when xprof found the step markers (wall per step, every program
    # in it); else the train programs' device time per execution
    out["step_ms"] = round(step_ms, 2) if step_ms else out["device_ms_per_step"]
    out["step_ms_source"] = "step annotations" if step_ms else "device time per execution"
    out["busy_fraction"] = busy
    out["busy"] = busy
    out["all_programs"] = {
        k: v
        for k, v in (attribute(rows, execs, top=0, programs="all") or {}).items()
        if k in ("device_ms_per_step", "by_scope", "by_phase", "by_category", "programs")
    }
    out["device"] = device_peaks(raw.get("roofline_model"))
    out["program"] = program_totals(roof_rows, set(out["programs"]))
    if isinstance(raw.get("utilization_viewer"), dict) and "cols" in raw["utilization_viewer"]:
        out["measured"] = measured_counters(raw["utilization_viewer"])
        peak = out["device"].get("peak_hbm_gbps")
        if peak and "hbm_bw_utilization" in out["measured"]:
            out["measured"]["hbm_gbps"] = round(out["measured"]["hbm_bw_utilization"] * peak, 1)
    if dyn := dynamic_memory(raw.get("memory_profile")):
        out["memory_dynamic"] = dyn
    if memory:
        train = [p for p, v in out["programs"].items() if v["kind"] == "train"] or list(
            out["programs"]
        )
        main = max(train, key=lambda p: out["programs"][p]["executions"])
        name = _module_name(modules, main)
        try:
            viewer = _tool(xplane, "memory_viewer", {"module_name": name, "memory_space": "0"})
            allocs = None
            module = parsed.get(main)
            if module is not None:  # the loop buffers XLA allocated (no op name)
                from tjev.campaign.hlo_labels import label_allocations

                shapes = [i.shape for i in module["instrs"].values()
                          if 'custom_call_target="AllocateBuffer"' in i.attrs]  # fmt: skip
                bucket = pick_bucket(residuals, shapes)
                allocs = label_allocations(module, bucket)
                out["scan_buffers"] = allocs
                if bucket:
                    out["residuals"] = bucket
            if viewer:
                raw["memory_viewer"] = viewer
                out["memory"] = peak_buffers(viewer, allocs=allocs)
        except Exception as e:
            out["memory"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    out["xplane"] = str(xplane)
    out["source"] = "xplane"
    out["fields"] = FIELDS
    return out


def summarize_any(profile_dir: Path, steps: int = 3, **kw: Any) -> dict[str, Any] | None:
    """The xplane summary when xprof is installed and an xplane is there, else the trace JSON
    one (``xprof_summary.summarize``, streamed). ``steps``: the trace fallback's step count
    when the trace has no step markers."""
    profile_dir = Path(profile_dir)
    errors = []
    if find_xplane(profile_dir) is not None:
        try:
            import xprof.convert.raw_to_tool_data  # noqa: F401

            out = summarize_xplane(profile_dir, **kw)
            if out is not None:
                return out
        except ImportError:
            errors.append("xprof not installed (pip install 'tjev[profile]')")
        except Exception as e:
            errors.append(f"xplane: {type(e).__name__}: {e}"[:300])
    from tjev.campaign.xprof_summary import summarize

    out = summarize(profile_dir, steps=steps)
    if out is not None and errors:
        out["xplane_error"] = errors[0]
    return out


def profile_tables(summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The tables of a summary as rows (for ``Experiment.table``): scope × phase (ms per
    step, estimated rates), the top ops' roofline, the op kinds, the buffers at the HBM peak."""
    out: dict[str, list[dict[str, Any]]] = {}
    if summary.get("scope_phase_ms"):
        out["scope_phase"] = [
            {"scope": scope, **{f"{p}_ms": v for p, v in by.items()},
             "ms": summary["scope_stats"][scope]["ms"],
             "percent": summary["scope_stats"][scope]["percent"],
             "tflops_est": summary["scope_stats"][scope]["tflops"],
             "hbm_gbps_est": summary["scope_stats"][scope]["hbm_gbps"],
             "bound_by_est": json.dumps(summary["scope_stats"][scope]["bound_by"])}
            for scope, by in summary["scope_phase_ms"].items()
        ]  # fmt: skip
    elif summary.get("ms_by_scope"):
        out["scope_phase"] = [
            {"scope": k, "ms": v, "percent": summary.get("by_scope", {}).get(k)}
            for k, v in summary["ms_by_scope"].items()
        ]
    if summary.get("top_ops"):
        out["top_ops"] = [dict(op) for op in summary["top_ops"]]
    if summary.get("top_kinds"):
        out["top_kinds"] = [dict(k) for k in summary["top_kinds"]]
    if buffers := (summary.get("memory") or {}).get("top_buffers"):
        out["peak_buffers"] = [dict(b) for b in buffers]
    return out


def log_profile(exp: Any, prefix: str, summary: dict[str, Any] | None) -> None:
    """A summary's tables and scalars to an ``Experiment`` under ``prefix`` (``SIZE/VARIANT/``)."""
    if not isinstance(summary, dict) or ("error" in summary and len(summary) == 1):
        return
    for name, rows in profile_tables(summary).items():
        exp.table(f"{prefix}{name}", rows)
    exp.summary(wandb_scalars(summary, prefix))


def wandb_scalars(summary: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """The flat numbers of a summary for wandb.summary (step, shares, measured counters)."""
    out: dict[str, float] = {}
    for k in ("step_ms", "device_ms_per_step", "busy_fraction", "steps"):
        if isinstance(summary.get(k), int | float):
            out[prefix + k] = summary[k]
    for scope, share in (summary.get("by_scope") or {}).items():
        out[f"{prefix}scope/{scope}"] = share
    for phase, share in (summary.get("by_phase") or {}).items():
        out[f"{prefix}phase/{phase}"] = share
    for k, v in (summary.get("measured") or {}).items():
        if isinstance(v, int | float):
            out[f"{prefix}measured/{k}"] = v
    if (mem := summary.get("memory")) and "peak_heap_gib" in mem:
        out[prefix + "memory/peak_heap_gib"] = mem["peak_heap_gib"]
    return out


def write_reports(
    session_dir: Path, out_dir: Path, top: int = 30, residuals: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """summary.json, scope_phase.txt, report.md / report.html (the readable page) and the raw
    tool tables (gzipped when large) in ``out_dir``: the small, restorable record of a
    capture. ``residuals``: residuals.json of the run (labels the scan residual stacks)."""
    from tjev.campaign import profile_report

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xplane = find_xplane(Path(session_dir))
    if xplane is None:
        return None
    raw = run_tools(
        xplane,
        ("hlo_stats", "roofline_model", "overview_page", "memory_profile", "utilization_viewer",
         "framework_op_stats", "perf_counters"),
    )  # fmt: skip
    summary = summarize_xplane(xplane, top=top, raw=raw, residuals=residuals)
    for tool, data in raw.items():
        if tool.startswith("_") or data is None:
            continue
        if tool == "hlo_text":  # the main train program's optimized HLO (graph viewer)
            with gzip.open(out_dir / "train_program.hlo.txt.gz", "wt") as f:
                f.write(data)
            continue
        text = json.dumps(data)
        if len(text) > 1 << 20:
            with gzip.open(out_dir / f"{tool}.json.gz", "wt") as f:
                f.write(text)
        else:
            (out_dir / f"{tool}.json").write_text(text)
    if summary is not None:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        if summary.get("scope_phase_ms"):
            (out_dir / "scope_phase.txt").write_text(scope_phase_table(summary) + "\n")
        title = f"Profile: {Path(session_dir).resolve().name}"
        (out_dir / "report.md").write_text(profile_report.markdown(summary, title))
        (out_dir / "report.html").write_text(profile_report.page(summary, title))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tjev.campaign.xprof_tools")
    parser.add_argument("session", type=Path, help="a profile dir, xprof session dir or xplane")
    parser.add_argument("--out", type=Path, help="write summary.json and the tool tables here")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--programs", choices=("train", "all"), default="train")
    parser.add_argument("--wandb-name", help="also log the tables to W&B (TJEV_WANDB=1)")
    parser.add_argument("--residuals", type=Path, help="residuals.json (tjev.campaign.residuals)")
    args = parser.parse_args(argv)
    residuals = json.loads(args.residuals.read_text()) if args.residuals else None
    try:
        import xprof.convert.raw_to_tool_data  # noqa: F401

        if args.out:
            summary = write_reports(args.session, args.out, top=args.top, residuals=residuals)
        else:
            summary = summarize_xplane(args.session, top=args.top, programs=args.programs,
                                       residuals=residuals)  # fmt: skip
    except ImportError:  # no xprof: the trace JSON summary (streamed; no phases, no roofline)
        print("xprof not installed (pip install 'tjev[profile]'): trace JSON", file=sys.stderr)
        summary = summarize_any(args.session, top=args.top)
        if summary is not None and args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    if summary is None:
        print(f"no profile under {args.session}", file=sys.stderr)
        return 1
    if args.wandb_name:
        from tjev.campaign.experiments import Experiment

        project = os.environ.get("TJEV_WANDB_BENCHMARKS_PROJECT", "tjev-benchmarks")
        exp = Experiment("profile", args.wandb_name, {"profile": str(args.session)}, project)
        log_profile(exp, "", summary)
        if args.out and (args.out / "summary.json").exists():
            exp.artifact(args.wandb_name, "profile-summary", [args.out / "summary.json"])
        exp.finish()
    if summary.get("scope_phase_ms"):
        print(scope_phase_table(summary))
    keys = ("source", "steps", "step_source", "step_ms", "step_ms_source", "device_ms_per_step",
            "busy_fraction", "by_phase", "by_scope", "device", "measured", "memory_dynamic")  # fmt: skip
    print(json.dumps({k: summary.get(k) for k in keys if k in summary}, indent=1))
    if mem := summary.get("memory"):
        print(json.dumps({k: v for k, v in mem.items() if k != "top_buffers"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
