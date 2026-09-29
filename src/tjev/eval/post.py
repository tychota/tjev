"""After training (``tjev post RUN``): calibrate, evaluate, summarise; each step skipped if done.

Writes ``RUN/post/``: ``calibration-jax.json`` (per-type temperatures fitted on the mix's
calibration split, never on JevBench), ``jevbench-{raw,cal}.json``, ``generators-cal.json``
(the held-out generator *reporting* set, a different seed from the selection set),
``validation-cal.json`` (unless ``quick``) and ``summary.md``. The checkpoint is the run's
selected step (``selection.json``), else its latest.
"""

from __future__ import annotations

import json
from pathlib import Path

from tjev.data.heldout import REPORT_SEED, write_heldout
from tjev.eval.report import summary
from tjev.eval.runs import calibrate, evaluate
from tjev.utils import write_json


def _log(message: str) -> None:
    print(f"[post] {message}", flush=True)


def post_train(
    run: str | Path,
    *,
    mix: str | Path,
    jevbench: str | Path,
    zeroshot: str | Path | None = None,
    quick: bool = False,
) -> Path:
    """``mix``: the built mix directory; ``jevbench``: public.jsonl (``tjev data jevbench``);
    ``zeroshot``: an ``tjev eval-base`` report of the same base model (the control)."""
    run, mix = Path(run), Path(mix)
    post = run / "post"
    post.mkdir(parents=True, exist_ok=True)
    cal = post / "calibration-jax.json"
    if not cal.exists():
        _log("calibrate")
        calibrate(run, mix / "calibration.jsonl", out=cal)
    heldout = post / "generators-heldout.jsonl"
    if not heldout.exists():
        write_heldout(heldout, per=20, seed=REPORT_SEED)

    def report(name: str, data: Path | str, calibration: Path | None = cal) -> dict:
        path = post / f"{name}.json"
        if not path.exists():
            _log(f"eval {name}")
            write_json(path, evaluate(run, data, calibration=calibration))
        return json.loads(path.read_text(encoding="utf-8"))

    raw = report("jevbench-raw", jevbench, None)
    calibrated = report("jevbench-cal", jevbench)
    extra = {"held-out generators": report("generators-cal", heldout)}
    if not quick:
        extra["mix validation"] = report("validation-cal", mix / "validation.jsonl")
    control = json.loads(Path(zeroshot).read_text(encoding="utf-8")) if zeroshot else None
    text = summary(run.name, raw, calibrated, control, extra)
    (post / "summary.md").write_text(text, encoding="utf-8")
    _log(f"done: {post / 'summary.md'}")
    return post / "summary.md"
