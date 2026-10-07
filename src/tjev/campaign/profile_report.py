"""The readable page of a profile summary (``xprof_tools.summarize_xplane``): report.md and
report.html with the programs, the scope × phase table, the top ops and model paths by time
(labelled, bound-by), the HBM peak by labelled buffer and the scan residual stacks.

    python -m tjev.campaign.xprof_tools PROFILE --out DIR [--residuals residuals.json]
"""

from __future__ import annotations

import html
from typing import Any

Table = tuple[str, str, list[str], list[list[Any]]]  # (title, note, columns, rows)


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"
    if isinstance(v, dict):
        return ", ".join(f"{k} {x:.0f}%" for k, x in list(v.items())[:2])
    return str(v)


def sections(s: dict[str, Any]) -> list[Table]:
    out: list[Table] = []
    progs = s.get("programs", {})
    out.append((
        "Programs",
        (f"{s.get('steps')} train-step executions ({s.get('step_source')}); step "
         f"{_fmt(s.get('step_ms'))} ms ({s.get('step_ms_source')}), device time "
         f"{_fmt(s.get('device_ms_per_step'))} ms per step, duty cycle "
         f"{_fmt(s.get('busy_fraction'))}."),
        ["program", "name", "kind", "executions"],
        [[p, v.get("name", "?"), v["kind"], v["executions"]] for p, v in progs.items()],
    ))  # fmt: skip
    phases = [p for p in ("fwd", "bwd", "remat", "other", "eval") if p in s.get("ms_by_phase", {})]
    if s.get("scope_phase_ms"):
        rows = []
        for scope, by in s["scope_phase_ms"].items():
            st = s["scope_stats"][scope]
            rows.append([scope, *[by.get(p, 0.0) for p in phases], st["ms"], st["percent"],
                         st["tflops"], st["hbm_gbps"], st["bound_by"]])  # fmt: skip
        out.append((
            "Where the step goes: scope × phase (ms per step)",
            ("TFLOP/s and HBM GB/s: XLA cost model over the measured time (estimated); "
             "bound-by: xprof's per-op roofline label, as shares of the scope's time."),
            ["scope", *phases, "ms", "%", "TFLOP/s est", "HBM GB/s est", "bound by"], rows,
        ))  # fmt: skip
    if s.get("by_kind"):
        out.append((
            "By kind of op (% of the step)",
            (f"{_fmt(s.get('reattributed_ms'))} ms per step of ops XLA named after their loop "
             "(…/while: layout copies, loop plumbing) carry their neighbour's model path."),
            ["kind", "%"],
            [[k, v] for k, v in s["by_kind"].items()],
        ))  # fmt: skip
    if s.get("top_paths"):
        out.append((
            "Top model paths (every op of one path and phase)",
            "",
            ["where", "phase", "ms/step", "%", "TFLOP/s est", "HBM GB/s est", "bound by"],
            [[p["where"], p["phase"], p["ms"], p["percent"], p.get("tflops"), p.get("hbm_gbps"),
              p.get("bound_by")] for p in s["top_paths"]],
        ))  # fmt: skip
    if s.get("top_ops"):
        out.append((
            "Top HLO ops",
            "op: XLA's instruction name (what xprof shows); where: its model path.",
            ["op", "kind", "program", "where", "phase", "ms/step", "%", "TFLOP/s est",
             "HBM GB/s est", "bound by", "roofline eff."],
            [[o["op"], o.get("kind", o.get("category")),
              progs.get(o.get("program"), {}).get("name", o.get("program")), o["where"],
              o["phase"], o["ms"], o["percent"], o["tflops"], o["hbm_gbps"], o["bound_by"],
              o.get("roofline_efficiency")] for o in s["top_ops"]],
        ))  # fmt: skip
    mem = s.get("memory") or {}
    if mem.get("composition"):
        out.append((
            f"HBM peak of {mem.get('module')}: {mem.get('peak_heap_gib')} GiB (static)",
            f"parameters {mem.get('parameters_gib')} GiB; the buffers live at the peak, grouped.",
            ["what", "GiB", "buffers"],
            [[c["what"], c["gib"], c["buffers"]] for c in mem["composition"]],
        ))  # fmt: skip
        out.append((
            "Largest buffers at the peak",
            "",
            ["what", "GiB", "instruction", "shape"],
            [[b["what"], b["gib"], b.get("instruction"), b.get("shape")]
             for b in mem.get("top_buffers", [])],
        ))  # fmt: skip
    if s.get("scan_buffers"):
        big = sorted(s["scan_buffers"].items(), key=lambda kv: -kv[1]["bytes"])
        out.append((
            "Loop buffers XLA allocates (AllocateBuffer, no op name)",
            "named by what the loop body writes into them and, by shape, residuals.json.",
            ["instruction", "MiB", "label", "written by"],
            [[k, round(v["bytes"] / 2**20, 1), v["label"], v.get("producer", "")]
             for k, v in big[:30]],
        ))  # fmt: skip
    if res := s.get("residuals"):
        out.append((
            (f"Saved residuals per layer (remat={res.get('remat')}, T={res.get('length')}, "
             f"{res.get('rows')} rows)"),
            f"stacked over {res.get('super_blocks')} super-blocks: {res.get('stacked_gib')} GiB.",
            ["layer", "residual", "dtype", "shape", "MiB stacked"],
            [[f"{ly['kind']}[{ly['position']}]", r["label"], r["dtype"], r["shape"],
              round(r["stacked_bytes"] / 2**20, 1)] for ly in res["layers"] for r in ly["saved"]],
        ))  # fmt: skip
    if meas := s.get("measured"):
        out.append((
            "Measured (hardware counters, the whole session)",
            meas.get("scope", ""),
            ["counter", "value"],
            [[k, v] for k, v in meas.items() if k != "scope"],
        ))  # fmt: skip
    return out


def markdown(s: dict[str, Any], title: str = "Profile") -> str:
    lines = [f"# {title}", ""]
    for name, note, cols, rows in sections(s):
        lines += [f"## {name}", ""]
        if note:
            lines += [note, ""]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        lines += ["| " + " | ".join(_fmt(v).replace("|", "\\|") for v in row) + " |"
                  for row in rows]  # fmt: skip
        lines.append("")
    return "\n".join(lines)


def page(s: dict[str, Any], title: str = "Profile") -> str:
    """A self-contained HTML page (no scripts, light and dark)."""
    head = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    )
    style = (
        f"<title>{html.escape(title)}</title><style>"
        ":root{--bg:#fff;--fg:#1b1b1b;--mut:#666;--line:#ddd;--head:#f4f4f4}"
        "@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--mut:#9a9a9a;"
        "--line:#333;--head:#222}}"
        "body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;"
        "margin:0 auto;max-width:1200px;padding:16px}"
        "h2{margin-top:2em}p{color:var(--mut)}.wrap{overflow-x:auto}"
        "table{border-collapse:collapse;font-variant-numeric:tabular-nums}"
        "td,th{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;"
        "white-space:nowrap}th{background:var(--head)}"
        "</style></head><body>"
    )
    parts = [head, style, f"<h1>{html.escape(title)}</h1>"]
    for name, note, cols, rows in sections(s):
        parts.append(f"<h2>{html.escape(name)}</h2>")
        if note:
            parts.append(f"<p>{html.escape(note)}</p>")
        parts.append("<div class='wrap'><table><tr>"
                     + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>")  # fmt: skip
        parts.extend("<tr>" + "".join(f"<td>{html.escape(_fmt(v))}</td>" for v in row)
                     + "</tr>" for row in rows)  # fmt: skip
        parts.append("</table></div>")
    parts.append("</body></html>")
    return "\n".join(parts)
