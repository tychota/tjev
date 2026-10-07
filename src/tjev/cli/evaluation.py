"""``tjev eval``, ``tjev calibrate``, ``tjev eval-base``, ``tjev eval-official``, ``tjev eval-v15``,
``tjev post``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tjev.cli.common import (
    Abstention,
    Band,
    CommitEps,
    CommitMargin,
    NoulOrderOpt,
    NoulTemperature,
    Orders,
    ScoreTemperature,
    SnapOpt,
    echo_json,
    noul_order_value,
    serving_policy,
)
from tjev.utils import write_json

Step = Annotated[int | None, typer.Option(help="checkpoint step (default: the selected one)")]
Out = Annotated[Path | None, typer.Option(help="write the full report here")]
Grouping = Annotated[
    str,
    typer.Option(help="temperatures per 'type', or per 'type_lang' shrunk to the type's (CV)"),
]
Predictions = Annotated[
    Path | None, typer.Option(help="also dump per-item predictions here (.jsonl.gz)")
]


def evaluate(
    run: Path,
    data: Path,
    step: Step = None,
    calibration: Annotated[Path | None, typer.Option(help="a calibration artifact")] = None,
    out: Out = None,
    orders: Orders = 1,
    noul_order: NoulOrderOpt = "shuffled",
    predictions: Predictions = None,
) -> None:
    """Grouped report of a trained run on DATA (JSONL items); prints the overall metrics."""
    from tjev.eval.runs import evaluate as run_eval

    report = run_eval(run, data, step=step, calibration=calibration, orders=orders,
                      noul_order=noul_order_value(noul_order), predictions=predictions)  # fmt: skip
    if out:
        write_json(out, report)
    echo_json(report["all"])


def calibrate(
    run: Path,
    data: Annotated[
        list[Path],
        typer.Argument(
            help="the mix calibration split, or a calibration pool: several "
            "files or directories of held-out sources"
        ),
    ],
    step: Step = None,
    out: Out = None,
    orders: Orders = 1,
    noul_order: NoulOrderOpt = "shuffled",
    grouping: Grouping = "type",
    predictions: Predictions = None,
) -> None:
    """Fit temperatures of a run on DATA (the mix calibration split, or a pool of held-out
    sources), for the readout it will be served with (--orders, --noul-order);
    docs/calibration.md."""
    from tjev.eval.runs import calibrate as run_calibrate

    pool = data[0] if len(data) == 1 else data
    echo_json(run_calibrate(run, pool, step=step, out=out, orders=orders,
                            noul_order=noul_order_value(noul_order), grouping=grouping,
                            predictions=predictions))  # fmt: skip


def eval_base(
    model: Path,
    data: Path,
    items: Annotated[list[str] | None, typer.Argument(help="presets and key=value")] = None,
    calibrate_on: Annotated[
        Path | None, typer.Option(help="fit per-type temperatures here (never on DATA)")
    ] = None,
    per_source: Annotated[int, typer.Option(help="first N items per source (0: all)")] = 0,
    out: Out = None,
    orders: Orders = 1,
    noul_order: NoulOrderOpt = "shuffled",
) -> None:
    """Zero-shot control: the untrained base MODEL with the same prompt and readout."""
    from tjev.config import load_config, split_args
    from tjev.eval.runs import evaluate_base

    files, overrides = split_args(items or [])
    report = evaluate_base(
        model,
        data,
        calibrate_on=calibrate_on,
        cfg=load_config(*files, overrides=overrides),
        per_source=per_source,
        orders=orders,
        noul_order=noul_order_value(noul_order),
    )
    if out:
        write_json(out, report)
    echo_json({k: report[k]["all"] for k in ("raw", "calibrated") if k in report})


def post(
    run: Path,
    mix: Annotated[Path, typer.Option(help="the built mix directory")],
    jevbench: Annotated[Path, typer.Option(help="JevBench public.jsonl")],
    zeroshot: Annotated[Path | None, typer.Option(help="`tjev eval-base` report")] = None,
    quick: Annotated[bool, typer.Option(help="skip the full validation pass")] = False,
    orders: Orders = 1,
    noul_order: NoulOrderOpt = "shuffled",
    grouping: Grouping = "type",
    calibration_pool: Annotated[
        list[Path] | None,
        typer.Option(
            help="fit the temperatures on these held-out files / directories "
            "instead of the mix calibration split (files get a -pool suffix)"
        ),
    ] = None,
    abstention: Abstention = "off",
    band: Band = (0.2, 0.8),
    snap: SnapOpt = "nearest",
    commit_margin: CommitMargin = 0.0,
    commit_eps: CommitEps = 0.0,
    noul_temperature: NoulTemperature = 1.0,
    score_temperature: ScoreTemperature = 1.0,
    extra_validation: Annotated[
        list[Path] | None,
        typer.Option(
            help="also evaluate these held-out files (e.g. validation-v7.jsonl, "
            "calibration-v7.jsonl; relative paths fall back to the mix directory): a "
            "report, a per-item dump and a per-source table each"
        ),
    ] = None,
) -> None:
    """After training: calibrate, evaluate (JevBench, held-out generators, validation),
    v1.5 scoring of the per-item predictions (as calibrated, under the serving policy if
    one is given, and the policy sweep), summary."""
    from tjev.eval.post import post_train

    policy = serving_policy(
        abstention, band, snap, commit_margin, commit_eps, noul_temperature, score_temperature
    )
    summary = post_train(run, mix=mix, jevbench=jevbench, zeroshot=zeroshot, quick=quick,
                         orders=orders, noul_order=noul_order_value(noul_order),
                         grouping=grouping, calibration_pool=calibration_pool,
                         policy=policy, extra_validation=extra_validation or ())  # fmt: skip
    typer.echo(summary.read_text())


def eval_official(
    target: Annotated[str, typer.Argument(help="a run dir, an HF snapshot, or base:SIZE")],
    out: Annotated[Path, typer.Option(help="report directory (report.json + harness files)")],
    suite: Annotated[
        str, typer.Option(help="jevbench-public | decision-index")
    ] = "jevbench-public",
    harness: Annotated[Path | None, typer.Option(help="JevBench harness checkout (pinned)")] = None,
    models: Annotated[
        Path | None, typer.Option(help="base:SIZE resolves to MODELS/Qwen3.5-SIZE")
    ] = None,
    calibration: Annotated[
        Path | None,
        typer.Option(help="artifact or eval-base report (default: RUN/post/calibration-jax.json)"),
    ] = None,
    orders: Orders = 1,
    noul_order: NoulOrderOpt = "shuffled",
    abstention: Abstention = "off",
    band: Band = (0.2, 0.8),
    snap: SnapOpt = "nearest",
    commit_margin: CommitMargin = 0.0,
    commit_eps: CommitEps = 0.0,
    noul_temperature: NoulTemperature = 1.0,
    score_temperature: ScoreTemperature = 1.0,
    prompt: Annotated[str, typer.Option(help="tjev | semif (the frozen-model control)")] = "tjev",
    step: Step = None,
    limit: Annotated[int | None, typer.Option(help="first N items only (a smoke run)")] = None,
    per_tier: Annotated[int, typer.Option(help="first N items of each tier (a smoke run)")] = 0,
    endpoint_kind: Annotated[str, typer.Option(help="gpu|cpu (own server) or api")] = "gpu",
    price_in: Annotated[float | None, typer.Option(help="USD per M input tokens")] = None,
    price_out: Annotated[float | None, typer.Option(help="USD per M output tokens")] = None,
    di_kit: Annotated[Path | None, typer.Option(help="Decision Index kit checkout")] = None,
    di_suite: Annotated[Path | None, typer.Option(help="built Decision Index suite dir")] = None,
    allow_unpinned: Annotated[bool, typer.Option(help="accept another harness commit")] = False,
) -> None:
    """The official evaluation: serve TARGET on a local port, run the official harness on it
    (unmodified, pinned), write the official metrics (docs/official-eval.md)."""
    import os

    from tjev.eval.official import evaluate_official

    root = Path(os.environ.get("TJEV_ROOT", "~/tjev-work")).expanduser()
    if target.startswith("base:"):
        path = (models or root / "models") / f"Qwen3.5-{target.removeprefix('base:')}"
    else:
        path = Path(target)
    if price_in is not None and price_out is None:
        price_out = 0.0
    report = evaluate_official(
        path,
        harness=harness or root / "jevbench",
        out=out,
        suite=suite,
        calibration=calibration,
        orders=orders,
        noul_order=noul_order_value(noul_order),
        policy=serving_policy(
            abstention, band, snap, commit_margin, commit_eps, noul_temperature, score_temperature
        ),
        step=step,
        prompt=prompt,
        limit=limit,
        per_tier=per_tier,
        endpoint_kind=endpoint_kind,
        price=(price_in, price_out or 0.0) if price_in is not None else None,
        di_kit=di_kit,
        di_suite=di_suite,
        allow_unpinned=allow_unpinned,
    )
    echo_json(official_brief(report))


def eval_v15(
    predictions: Annotated[
        list[Path], typer.Argument(help="per-item prediction dumps (post/predictions/*.jsonl.gz)")
    ],
    calibration: Annotated[
        Path | None,
        typer.Option(help="re-apply this artifact's temperatures (default: the dumps' probs)"),
    ] = None,
    fit_on: Annotated[
        list[Path] | None,
        typer.Option(help="fit per-type temperatures on these dumps (a calibration pool)"),
    ] = None,
    harness: Annotated[
        Path | None, typer.Option(help="JevBench checkout: the hard items' gold distributions")
    ] = None,
    abstention: Abstention = "off",
    band: Band = (0.2, 0.8),
    snap: SnapOpt = "nearest",
    commit_margin: CommitMargin = 0.0,
    commit_eps: CommitEps = 0.0,
    noul_temperature: NoulTemperature = 1.0,
    score_temperature: ScoreTemperature = 1.0,
    sweep: Annotated[bool, typer.Option(help="also sweep the policy knobs one at a time")] = False,
    out: Out = None,
) -> None:
    """JevBench v1.5 Intelligence and Calibration (per type and tier) of per-item
    predictions, as calibrated and under a serving policy (docs/official-eval.md)."""
    import json

    from tjev.eval import calibrate as cal
    from tjev.eval import predictions as dump
    from tjev.eval import v15

    if calibration and fit_on:
        raise typer.BadParameter("give --calibration or --fit-on, not both")
    temperatures = None
    fit = None
    if calibration:
        temperatures = json.loads(calibration.read_text(encoding="utf-8"))["temperatures"]
    elif fit_on:
        fit = cal.fit_from_predictions([r for p in fit_on for r in dump.read(p)])
        temperatures = fit["temperatures"]
    policy = serving_policy(
        abstention, band, snap, commit_margin, commit_eps, noul_temperature, score_temperature
    )
    reports = v15.score_dumps(
        predictions,
        temperatures=temperatures,
        policy=policy,
        gold_probs=v15.harness_gold_probs(harness) if harness else None,
        with_sweep=sweep,
    )
    full = {"temperatures": temperatures, "pool_fit": fit, "reports": reports}
    if out:
        write_json(out, full)
    echo_json({name: {k: v15.headline(r[k]) for k in ("calibrated", "policy") if k in r}
               for name, r in reports.items()})  # fmt: skip


def official_brief(report: dict) -> dict:
    """The headline numbers of an official report."""
    if report["suite"] == "decision-index":
        brief = report.get("decision_index") or {}
        return {"model": report["served"]["model"], "decision_index": brief.get("decision_index")}
    v15 = report["v15"]
    return {
        "model": report["served"]["model"],
        "harness": report["harness"]["commit"],
        "n": report["n"],
        "failed": report["failed"],
        "accuracy": report["accuracy"],
        "tiers": {t: v["accuracy"] for t, v in report["tiers"].items()},
        "types": {t: v["accuracy"] for t, v in report["types"].items()},
        "v13_intelligence_public": report["v13"]["intelligence_public"],
        "v13_calibration": report["v13"]["calibration"]["axis"],
        "v15_intelligence_open_public": v15["intelligence_open_public"]["intelligence_equal"],
        "v15_calibration_typed": v15["calibration_typed"]["calibration_equal"],
        "noul_abstentions": v15["intelligence_open_public"]["types"]
        .get("noul", {})
        .get("abstentions"),
        "speed": report["latency"]["standard"]["speed"] if report["latency"]["standard"] else None,
        "p50_s": report["latency"]["all"]["p50_s"],
        "cost_usd_per_1000": report["cost"]["usd_per_1000"],
        "v14_composite_public_proxy": report["v14"]["composite_public_proxy"],
        "v15_composite_public_proxy_A": v15["composites_public_proxy"]["A"]["score"],
    }


def register(app: typer.Typer) -> None:
    app.command("eval")(evaluate)
    app.command()(calibrate)
    app.command("eval-base")(eval_base)
    app.command("eval-official")(eval_official)
    app.command("eval-v15")(eval_v15)
    app.command()(post)
