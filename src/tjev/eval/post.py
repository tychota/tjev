"""After training (``tjev post RUN``): calibrate, evaluate, summarise; each step skipped if done.

Writes ``RUN/post/``: ``calibration-jax.json`` (temperatures fitted on the mix's
calibration split, never on JevBench; per type by default, or per (type, language) shrunk
to per type with ``grouping="type_lang"``, :mod:`tjev.eval.calibrate`), ``jevbench-{raw,cal}.json``, ``generators-cal.json`` (the
held-out generator *reporting* set, a different seed from the selection set),
``validation-cal.json`` (unless ``quick``), ``summary.md``, ``predictions/`` (per-item
raw logits of the calibration split, the JevBench public items, the held-out generators
and the validation split, ``{calibration,jevbench,generators,validation}.jsonl.gz``,
:mod:`tjev.eval.predictions`: for offline calibration studies,
``scripts/calibration_study.py``) and ``v15.json`` (JevBench v1.5 scoring of those
predictions, :mod:`tjev.eval.v15`: as calibrated, under the serving ``policy`` if one is
given, and the one-knob policy sweep; cheap, recomputed on every call). A post dir from before the
predictions existed gets them on the next ``tjev post`` (one forward pass per split; the
reports and the calibration are kept). The checkpoint is the run's
selected step (``selection.json``), else its latest.

``orders`` / ``noul_order`` (the readout, :mod:`tjev.eval.evalset`): the temperatures are
fitted on the same order-averaged distributions the evals report and the server will
serve (the artifact records the readout; ``tjev serve`` refuses another). A non-default
readout writes its own files (``calibration-jax-k2.json``, ``jevbench-cal-k2.json``, …,
``summary-k2.md``), so it never reuses a K = 1 step's outputs.

``calibration_pool``: fit the temperatures on held-out *sources* (files or directories:
e.g. the mix's ``heldout-select.jsonl`` and sources the run never trained on) instead of
the in-distribution calibration split; every file then gets a ``-pool`` suffix
(``calibration-jax-k2-pool.json``, ``jevbench-cal-k2-pool.json``, …). The pool must not
contain the reporting sets (the JevBench public items, ``generators-heldout.jsonl``).

``extra_validation`` (mix-v7.5: the mix's ``validation-v7.jsonl`` / ``calibration-v7.jsonl``,
the per-source held-out rows of the sources added after mix-v6): each file is evaluated
with the run's temperatures like the validation split, into ``<stem>-cal{tag}.json``, with
its per-item dump ``predictions/<stem>{tag}.jsonl.gz``, a per-source table in the summary
and its v1.5 scoring. A relative path that does not exist is read from the mix directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from tjev.data.heldout import REPORT_SEED, write_heldout
from tjev.data.render import NoulOrder
from tjev.eval import v15
from tjev.eval.calibrate import DEFAULT_GROUPING
from tjev.eval.report import summary
from tjev.eval.runs import Data, calibrate, evaluate, pool_files, predict, readout_tag
from tjev.serve.policy import ServingPolicy
from tjev.utils import file_hash, write_json


def _log(message: str) -> None:
    print(f"[post] {message}", flush=True)


def post_train(
    run: str | Path,
    *,
    mix: str | Path,
    jevbench: str | Path,
    zeroshot: str | Path | None = None,
    quick: bool = False,
    orders: int = 1,
    noul_order: NoulOrder = "shuffled",
    grouping: str = DEFAULT_GROUPING,
    calibration_pool: Data | None = None,
    policy: ServingPolicy | None = None,
    extra_validation: list[str | Path] | tuple = (),
) -> Path:
    """``mix``: the built mix directory; ``jevbench``: public.jsonl (``tjev data jevbench``);
    ``zeroshot``: an ``tjev eval-base`` report of the same base model (the control)."""
    run, mix = Path(run), Path(mix)
    post = run / "post"
    post.mkdir(parents=True, exist_ok=True)
    tag = readout_tag(orders, noul_order) + ("-pool" if calibration_pool else "")
    cal = post / f"calibration-jax{tag}.json"
    preds = post / "predictions"
    heldout = post / "generators-heldout.jsonl"
    if not heldout.exists():
        write_heldout(heldout, per=20, seed=REPORT_SEED)
    if calibration_pool:
        reporting = {file_hash(jevbench), file_hash(heldout)}
        if any(file_hash(f) in reporting for f in pool_files(calibration_pool)):
            raise ValueError("the calibration pool contains a reporting set (JevBench public "
                             "or the held-out generator reporting set)")  # fmt: skip

    def dumped(split: str) -> Path:
        return preds / f"{split}{tag}.jsonl.gz"

    if not cal.exists():
        _log(f"calibrate (orders={orders}, noul_order={noul_order}, grouping={grouping})")
        fit_on = calibration_pool or mix / "calibration.jsonl"
        calibrate(run, fit_on, out=cal, orders=orders, noul_order=noul_order,
                  grouping=grouping,
                  predictions=dumped("pool" if calibration_pool else "calibration"))  # fmt: skip

    def report(name: str, data: Path | str, calibration: Path | None = cal,
               split: str | None = None) -> dict:  # fmt: skip
        path = post / f"{name}{tag}.json"
        out = dumped(split) if split else None
        if not path.exists():
            _log(f"eval {name}{tag}")
            rep = evaluate(run, data, calibration=calibration, orders=orders,
                           noul_order=noul_order, predictions=out)  # fmt: skip
            write_json(path, rep)
        elif out is not None and not out.exists():
            _log(f"predictions {split}{tag}")
            predict(run, data, out, orders=orders, noul_order=noul_order)
        return json.loads(path.read_text(encoding="utf-8"))

    if not calibration_pool and not dumped("calibration").exists():
        _log(f"predictions calibration{tag}")
        predict(run, mix / "calibration.jsonl", dumped("calibration"), orders=orders,
                noul_order=noul_order)  # fmt: skip

    raw = report("jevbench-raw", jevbench, None)
    calibrated = report("jevbench-cal", jevbench, split="jevbench")
    extra = {"held-out generators": report("generators-cal", heldout, split="generators")}
    if not quick:
        extra["mix validation"] = report("validation-cal", mix / "validation.jsonl",
                                         split="validation")  # fmt: skip
    extra_names = []
    for path in extra_validation:
        path = Path(path)
        if not path.exists() and not path.is_absolute():
            path = mix / path
        if not path.is_file():
            raise FileNotFoundError(f"extra validation file {path} not found")
        name = path.name.removesuffix(".jsonl")
        if name in ("calibration", "jevbench", "generators", "validation", "pool"):
            raise ValueError(f"extra validation {path}: {name} is a split name of post")
        title = f"{name} (per source)"
        extra[title] = report(f"{name}-cal", path, split=name)
        extra_names.append(name)
    # the zero-shot control is optional: a missing report (no s0 bring-up in this session)
    # leaves the Δ columns out instead of failing the whole post-training
    control = None
    if zeroshot and Path(zeroshot).is_file():
        control = json.loads(Path(zeroshot).read_text(encoding="utf-8"))
    elif zeroshot:
        _log(f"no zero-shot control at {zeroshot}: summary without the Δ columns")
    splits = [dumped(s) for s in ("jevbench", "generators", "validation", *extra_names)
              if dumped(s).exists()]  # fmt: skip
    _log(f"v1.5 scoring of {', '.join(p.name for p in splits)}")
    write_json(post / f"v15{tag}.json", v15.score_dumps(splits, policy=policy, with_sweep=True))
    text = summary(run.name + tag, raw, calibrated, control, extra,
                   by_source=[t for t in extra if t.endswith("(per source)")])  # fmt: skip
    out = post / f"summary{tag}.md"
    out.write_text(text, encoding="utf-8")
    _log(f"done: {out}")
    return out
