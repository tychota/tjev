"""Markdown summary of a trained run's eval reports against the zero-shot control.

Reports are ``tjev eval`` / ``tjev eval-base`` JSON (``all`` plus ``type``, ``source``,
``lang``, ``family`` groups of n / accuracy / nll / brier / ece / confidence). The control
is the *calibrated* zero-shot base model (temperatures fitted on the same calibration split),
the JevBench raw-logit control protocol.
"""

from __future__ import annotations

METRICS = ("accuracy", "nll", "brier", "ece")
TIERS = ("jevbench_easy", "jevbench_original", "jevbench_hard")


def _row(name: str, entry: dict, base: dict | None = None) -> str:
    cells = [name, str(entry["n"])]
    for m in METRICS:
        v = entry[m]
        cells.append(f"{v:.3f}" if base is None else f"{v:.3f} ({v - base[m]:+.3f})")
    return "| " + " | ".join(cells) + " |"


def _table(title: str, rows: list[str]) -> str:
    head = "| | n | accuracy | NLL | Brier | ECE |\n|---|---|---|---|---|---|"
    return f"### {title}\n\n{head}\n" + "\n".join(rows) + "\n"


def summary(
    name: str,
    raw: dict,
    calibrated: dict,
    zeroshot: dict | None = None,
    extra: dict[str, dict] | None = None,
) -> str:
    """JevBench overall / by tier / family / type / language (Δ vs the control when given),
    then one table per ``extra`` report (e.g. held-out generators, mix validation)."""
    control = (zeroshot.get("calibrated", zeroshot["raw"]) if zeroshot else None) or {}
    against = " (Δ vs calibrated zero-shot control)" if control else ""
    out = [f"## {name}: JevBench public{against}\n"]
    out.append(
        f"checkpoint step {calibrated.get('step')}; temperatures {calibrated.get('temperatures')}\n"
    )
    rows = [_row("zero-shot, calibrated", control["all"])] if control else []
    rows += [
        _row(f"{name}, raw", raw["all"], control.get("all")),
        _row(f"{name}, calibrated", calibrated["all"], control.get("all")),
    ]
    out.append(_table("Overall", rows))
    tiers = [
        _row(t.removeprefix("jevbench_"), calibrated["source"][t], control.get("source", {}).get(t))
        for t in TIERS
        if t in calibrated.get("source", {})
    ]
    if tiers:
        out.append(_table("By tier (calibrated)", tiers))
    for group in ("family", "type", "lang"):
        entries = sorted(calibrated.get(group, {}).items(), key=lambda kv: kv[1]["accuracy"])
        rows = [_row(k, v, control.get(group, {}).get(k)) for k, v in entries]
        if rows:
            out.append(_table(f"By {group} (calibrated, weakest first)", rows))
    for title, rep in (extra or {}).items():
        rows = [_row("all", rep["all"])]
        rows += [_row(f"lang={k}", v) for k, v in sorted(rep.get("lang", {}).items())]
        rows += [_row(f"type={k}", v) for k, v in sorted(rep.get("type", {}).items())]
        out.append(_table(f"{title} (calibrated)", rows))
    return "\n".join(out)
