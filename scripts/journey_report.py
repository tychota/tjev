"""The kernel journey's article tables (markdown) and charts (PNG) from journey.json.

    python scripts/journey_report.py reports/probe/journey/journey.json [MORE.json] --out DIR

Writes ``DIR/journey.md`` (throughput, step time per bucket, correctness, time by scope,
IR statistics) and, when matplotlib is installed (``pip install matplotlib``), the charts
``DIR/tokens_per_second.png``, ``DIR/scopes.png`` (step time by model scope, stacked),
``DIR/hbm_peak.png`` and ``DIR/mfu.png``. No JAX, no TPU: it reads the JSON only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# the reference categorical order (dataviz skill palette), light surface
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
OTHER = "#a3a29c"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SURFACE = "#fcfcfb"
SCOPES_SHOWN = 7  # the largest scopes, the rest folds into "other"


def _f(value: Any, fmt: str, dash: str = "–") -> str:
    return dash if value is None else format(value, fmt)


def _label(journey: dict, variant: str) -> str:
    return journey.get("labels", {}).get(variant, variant)


def _runs(journey: dict, size: str) -> list[dict]:
    return [r for r in journey["runs"] if r["size"] == size]


def _correct(r: dict) -> str:
    check = r.get("correctness")
    if not check:
        return "–"
    if "error" in check:
        return "error"
    mark = "yes" if check.get("pass") else "**no**"
    return f"{mark} (loss {_f(check.get('loss_rel'), '.1e')}, logit {_f(check.get('logit_max_abs'), '.2f')})"


def throughput_table(journey: dict, size: str) -> str:
    lines = [
        f"### {size}: throughput (real training, steady state)",
        "",
        (
            "| Iteration | tok/s (real) | tok/s (padded) | vs XLA | Step p50 (p10–p90) | MFU "
            "| Peak HBM | Compile | Correct |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in _runs(journey, size):
        name = _label(journey, r["variant"])
        if r.get("status") != "ok" or not r.get("tokens_per_second"):
            lines.append(
                f"| {name} | {r.get('error') or r.get('status')} | | | | | | | {_correct(r)} |"
            )
            continue
        step = r.get("step_s") or {}
        lines.append(
            f"| {name} | {r['tokens_per_second']:,.0f} | {_f(r.get('padded_tokens_per_second'), ',.0f')} "
            f"| {_f(r.get('speedup_vs_xla'), '.2f')}× | {_f(step.get('p50'), '.3f')} s "
            f"({_f(step.get('p10'), '.3f')}–{_f(step.get('p90'), '.3f')}) "
            f"| {_f(None if r.get('mfu') is None else 100 * r['mfu'], '.1f')}% "
            f"| {_f(r.get('hbm_peak_gb'), '.1f')} GB | {_f(r.get('compile_s_total'), '.0f')} s "
            f"| {_correct(r)} |"
        )
    return "\n".join(lines)


def bucket_table(journey: dict, size: str) -> str:
    runs = _runs(journey, size)
    lengths = sorted({int(k) for r in runs for k in r.get("by_bucket", {})})
    if not lengths:
        return ""
    head = " | ".join(f"{n} p50 / compile" for n in lengths)
    lines = [
        f"### {size}: per sequence bucket",
        "",
        f"| Iteration | {head} |",
        "|---|" + "---|" * len(lengths),
    ]
    for r in runs:
        cells = []
        for n in lengths:
            b = r.get("by_bucket", {}).get(str(n), {})
            p50 = (b.get("step_s") or {}).get("p50")
            cells.append(f"{_f(p50, '.3f')} s / {_f(b.get('compile_s'), '.0f')} s")
        lines.append(f"| {_label(journey, r['variant'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def scope_rows(journey: dict, size: str) -> tuple[list[str], list[str], list[list[float]]]:
    """(variants, scopes, ms per step [variant][scope]): the largest scopes overall, the
    rest as "other"."""
    runs = [
        r
        for r in _runs(journey, size)
        if isinstance(r.get("profile"), dict) and "ms_by_scope" in r["profile"]
    ]
    totals: dict[str, float] = {}
    for r in runs:
        for k, v in r["profile"]["ms_by_scope"].items():
            totals[k] = totals.get(k, 0.0) + v
    top = [k for k, _ in sorted(totals.items(), key=lambda kv: -kv[1]) if k != "other"][
        :SCOPES_SHOWN
    ]
    scopes = [*top, "other"]
    values = []
    for r in runs:
        ms = r["profile"]["ms_by_scope"]
        row = [ms.get(k, 0.0) for k in top]
        row.append(sum(v for k, v in ms.items() if k not in top))
        values.append(row)
    return [r["variant"] for r in runs], scopes, values


def scope_table(journey: dict, size: str) -> str:
    variants, scopes, values = scope_rows(journey, size)
    if not variants:
        return ""
    lines = [
        f"### {size}: device time per step by scope (ms, xprof)",
        "",
        "| Iteration | " + " | ".join(scopes) + " | total |",
        "|---|" + "---|" * (len(scopes) + 1),
    ]
    for v, row in zip(variants, values, strict=True):
        lines.append(
            f"| {_label(journey, v)} | "
            + " | ".join(f"{x:.1f}" for x in row)
            + f" | {sum(row):.1f} |"
        )
    hbm = [
        (r["variant"], r["profile"].get("hbm_gb_per_step"))
        for r in _runs(journey, size)
        if isinstance(r.get("profile"), dict) and r["profile"].get("hbm_gb_per_step")
    ]
    if hbm:
        lines += [
            "",
            "HBM bytes accessed per step (xprof, XLA cost analysis): "
            + ", ".join(f"{_label(journey, v)} {gb:.1f} GB" for v, gb in hbm),
        ]
    return "\n".join(lines)


def ir_table(journey: dict) -> str:
    runs = (journey.get("ir") or {}).get("runs", {})
    if not runs:
        return ""
    from tjev.campaign.journey import brief_ir

    lines = [
        "### GDN core (fwd + bwd), one compile: the compiler's view",
        "",
        (
            "| Variant | Kernel bundles (fwd / bwd) | XLA bundles | MXU ops | Kernel VMEM "
            "| LLO passes | HLO fusions | XLA bytes accessed | Compile |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, entry in runs.items():
        b = brief_ir(entry)
        kernels = ", ".join(f"{k} {v:,}" for k, v in (b["kernel_bundles"] or {}).items()) or "–"
        vmem = b.get("kernel_vmem_bytes") or {}
        vmem_text = ", ".join(f"{k} {v / 2**20:.1f} MiB" for k, v in vmem.items() if v) or "–"
        lines.append(
            f"| {key} | {kernels} | {_f(b['xla_bundles'], ',')} | {_f(b['mxu_ops'], ',')} "
            f"| {vmem_text} | {_f(b['llo_passes'], 'd')} | {_f(b['hlo_fusions'], 'd')} "
            f"| {_f(None if b['xla_bytes_accessed'] is None else b['xla_bytes_accessed'] / 1e9, '.2f')} GB "
            f"| {_f(b['compile_s'], '.0f')} s |"
        )
    flags = journey["ir"].get("flags", [])
    if flags or journey["ir"].get("rejected"):
        lines += [
            "",
            (
                f"Dump flags accepted: `{' '.join(flags)}`; rejected: "
                f"`{' '.join(journey['ir'].get('rejected', [])) or 'none'}`"
            ),
        ]
    return "\n".join(lines)


def markdown(journey: dict) -> str:
    parts = ["# Kernel journey", ""]
    for size in journey["sizes"]:
        for table in (
            throughput_table(journey, size),
            bucket_table(journey, size),
            scope_table(journey, size),
        ):
            if table:
                parts += [table, ""]
    if table := ir_table(journey):
        parts += [table, ""]
    return "\n".join(parts)


# ------------------------------------------------------------------------------------------
# Charts (matplotlib, optional)


def _style(ax, title: str, xlabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    ax.set_facecolor(SURFACE)


def bar_chart(
    journey: dict,
    metric: str,
    title: str,
    xlabel: str,
    path: Path,
    scale: float = 1.0,
    fmt: str = ",.0f",
) -> Path | None:
    """One horizontal bar per iteration, one panel per size; a run that failed the
    correctness check is hatched and labelled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [s for s in journey["sizes"] if any(r.get(metric) for r in _runs(journey, s))]
    if not sizes:
        return None
    rows = max(len(_runs(journey, s)) for s in sizes)
    fig, axes = plt.subplots(
        1,
        len(sizes),
        figsize=(5.2 * len(sizes), 0.42 * rows + 1.2),
        squeeze=False,
        facecolor=SURFACE,
    )
    for ax, size in zip(axes[0], sizes, strict=True):
        runs = list(reversed(_runs(journey, size)))
        names = [_label(journey, r["variant"]) for r in runs]
        values = [(r.get(metric) or 0.0) * scale for r in runs]
        wrong = [r.get("correctness", {}).get("pass") is False for r in runs]
        bars = ax.barh(names, values, color=SERIES[0], height=0.62, edgecolor=SURFACE, linewidth=2)
        for bar, bad, v in zip(bars, wrong, values, strict=True):
            if bad:
                bar.set_facecolor(SURFACE)
                bar.set_edgecolor(SERIES[7])
                bar.set_hatch("////")
            if v:
                label = format(v, fmt) + ("  wrong" if bad else "")
                ax.text(
                    v,
                    bar.get_y() + bar.get_height() / 2,
                    f" {label}",
                    va="center",
                    fontsize=8,
                    color=INK,
                )
        ax.set_xlim(0, max(values or [1]) * 1.25)
        _style(ax, f"{title}, {size}", xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def scope_chart(journey: dict, path: Path) -> Path | None:
    """Device time per step by scope, stacked, one panel per size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(s, *scope_rows(journey, s)) for s in journey["sizes"]]
    panels = [p for p in panels if p[1]]
    if not panels:
        return None
    rows = max(len(p[1]) for p in panels)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6.2 * len(panels), 0.45 * rows + 1.8),
        squeeze=False,
        facecolor=SURFACE,
    )
    for ax, (size, variants, scopes, values) in zip(axes[0], panels, strict=True):
        names = [_label(journey, v) for v in variants][::-1]
        values = values[::-1]
        left = [0.0] * len(values)
        for j, scope in enumerate(scopes):
            color = OTHER if scope == "other" else SERIES[j % len(SERIES)]
            widths = [row[j] for row in values]
            ax.barh(
                names,
                widths,
                left=left,
                color=color,
                height=0.62,
                edgecolor=SURFACE,
                linewidth=2,
                label=scope,
            )
            left = [a + b for a, b in zip(left, widths, strict=True)]
        for y, total in enumerate(left):
            ax.text(total, y, f" {total:,.0f} ms", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, max(left) * 1.18)
        _style(ax, f"Device time per step by scope, {size}", "ms per step (xprof)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 8),
        frameon=False,
        fontsize=8,
        labelcolor=MUTED,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def charts(journey: dict, out: Path) -> list[Path]:
    # files, never a notebook display (Colab exports MPLBACKEND=module://matplotlib_inline…)
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not installed: tables only")
        return []
    made = [
        bar_chart(
            journey,
            "tokens_per_second",
            "Training throughput",
            "tokens/s (real)",
            out / "tokens_per_second.png",
        ),
        scope_chart(journey, out / "scopes.png"),
        bar_chart(journey, "hbm_peak_gb", "Peak HBM", "GB", out / "hbm_peak.png", fmt=".1f"),
        bar_chart(
            journey, "mfu", "Model FLOP utilisation", "%", out / "mfu.png", scale=100, fmt=".1f"
        ),
    ]
    return [p for p in made if p is not None]


def merge(parts: list[dict]) -> dict:
    """Several sessions' journey.json as one (e.g. 0.8B in one session, 4B in the next): the
    runs of every part (a later part's run replaces an earlier one of the same size and
    variant), the sizes in order of appearance, the last IR found."""
    out = dict(parts[0])
    runs: dict[tuple[str, str], dict] = {}
    for part in parts:
        for r in part["runs"]:
            runs[(r["size"], r["variant"])] = r
        if part.get("ir"):
            out["ir"] = part["ir"]
        out["labels"] = {**out.get("labels", {}), **part.get("labels", {})}
    out["sizes"] = list(dict.fromkeys(s for p in parts for s in p["sizes"]))
    out["variants"] = list(dict.fromkeys(v for p in parts for v in p["variants"]))
    order = {v: i for i, v in enumerate(out["variants"])}
    out["runs"] = sorted(
        runs.values(), key=lambda r: (out["sizes"].index(r["size"]), order.get(r["variant"], 0))
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "journey", type=Path, nargs="+", help="journey.json (several: one per session, merged)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args(argv)
    journey = merge([json.loads(p.read_text()) for p in args.journey])
    args.out.mkdir(parents=True, exist_ok=True)
    text = markdown(journey)
    (args.out / "journey.md").write_text(text + "\n")
    print(text)
    if not args.no_charts:
        for path in charts(journey, args.out):
            print(f"chart: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
