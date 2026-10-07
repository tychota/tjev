"""W&B runs for the campaign's measurements (kernel tests, bench, parity), next to training.

They log to ``<entity>/tjev-experiments`` (``TJEV_WANDB_EXPERIMENTS_PROJECT``), speed
benchmarks to ``<entity>/tjev-benchmarks`` (``TJEV_WANDB_BENCHMARKS_PROJECT``), one run per
measurement, grouped by campaign (``TJEV_WANDB_CAMPAIGN``). On only when ``TJEV_WANDB=1``
(cloud/tpu_env.sh sets it), with the entity and mode of training (``TJEV_WANDB_ENTITY``,
``TJEV_WANDB_MODE``). W&B is for watching: a failure to start or log is printed and the
measurement goes on (its JSON report is the record).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class Experiment:
    """A W&B run, or nothing: every method is a no-op when W&B is off or broken."""

    def __init__(
        self, job_type: str, name: str, config: dict[str, Any], project: str | None = None
    ):
        self.run = None
        if os.environ.get("TJEV_WANDB") != "1":
            return
        try:
            import wandb

            campaign = os.environ.get("TJEV_WANDB_CAMPAIGN", "")
            entity = os.environ.get("TJEV_WANDB_ENTITY")
            self.run = wandb.init(
                project=project
                or os.environ.get("TJEV_WANDB_EXPERIMENTS_PROJECT", "tjev-experiments"),
                entity=entity or None,
                mode=os.environ.get("TJEV_WANDB_MODE", "online"),  # ty: ignore[invalid-argument-type]
                job_type=job_type,
                group=campaign or None,
                name=f"{campaign}-{name}" if campaign else name,
                config=config | {"code_version": os.environ.get("CODE_VERSION", "")},
            )
        except Exception as e:
            print(json.dumps({"event": "wandb_experiment_failed", "error": str(e)[:200]}))

    def _call(self, fn: str, *args, **kwargs) -> None:
        if self.run is None:
            return
        try:
            getattr(self, fn)(*args, **kwargs)
        except Exception as e:
            print(json.dumps({"event": "wandb_log_failed", "what": fn, "error": str(e)[:200]}))

    def summary(self, values: dict[str, Any]) -> None:
        self._call("_summary", values)

    def table(self, key: str, rows: list[dict[str, Any]]) -> None:
        self._call("_table", key, rows)

    def artifact(self, name: str, kind: str, paths: list[Path]) -> None:
        self._call("_artifact", name, kind, paths)

    def bar(self, key: str, values: dict[str, float], label: str = "value") -> None:
        """A bar chart (one bar per name)."""
        self._call("_bar", key, values, label)

    def series(self, key: str, xs: list[float], ys: dict[str, list[float]], xname: str) -> None:
        """A line chart of several series over one x axis (e.g. error per layer)."""
        self._call("_series", key, xs, ys, xname)

    def image(self, key: str, path: Path) -> None:
        """A small image (a PNG chart) under ``key``."""
        self._call("_image", key, path)

    def finish(self) -> None:
        self._call("_finish")

    def _summary(self, values):
        assert self.run is not None
        self.run.summary.update(values)

    def _table(self, key, rows):
        import wandb

        assert self.run is not None
        if not rows:
            return
        columns = sorted({k for row in rows for k in row})
        data: list[Iterable[Any]] = [[row.get(c) for c in columns] for row in rows]
        self.run.log({key: wandb.Table(columns=columns, data=data)})

    def _bar(self, key, values, label):
        import wandb

        assert self.run is not None
        table = wandb.Table(columns=["name", label], data=[[k, v] for k, v in values.items()])
        self.run.log({key: wandb.plot.bar(table, "name", label, title=key)})

    def _series(self, key, xs, ys, xname):
        import wandb

        assert self.run is not None
        names = list(ys)
        chart = wandb.plot.line_series(
            xs=xs, ys=[ys[n] for n in names], keys=names, title=key, xname=xname
        )
        self.run.log({key: chart})

    def _image(self, key, path):
        import wandb

        assert self.run is not None
        self.run.log({key: wandb.Image(str(path))})

    def _artifact(self, name, kind, paths):
        import wandb

        assert self.run is not None
        art = wandb.Artifact(name, type=kind)
        for path in paths:
            if path.is_dir():
                art.add_dir(str(path), name=path.name)
            elif path.exists():
                art.add_file(str(path))
        self.run.log_artifact(art)

    def _finish(self):
        assert self.run is not None
        self.run.finish()


def log_junit(name: str, paths: list[Path]) -> int:
    """One W&B run with the cases of pytest JUnit XML files (``pytest --junitxml``): name,
    outcome, seconds and the failure message. Returns the number of failed cases."""
    import xml.etree.ElementTree as ET

    rows = []
    for path in paths:
        if not path.exists():
            continue
        for case in ET.parse(path).getroot().iter("testcase"):
            bad = case.find("failure")
            if bad is None:
                bad = case.find("error")
            skipped = case.find("skipped") is not None
            rows.append({
                "suite": path.stem,
                "test": case.get("name", ""),
                "outcome": "skipped" if skipped else "failed" if bad is not None else "passed",
                "seconds": float(case.get("time", 0)),
                "message": (bad.get("message", "") if bad is not None else "")[:500],
            })  # fmt: skip
    failed = sum(r["outcome"] == "failed" for r in rows)
    exp = Experiment("kernel-tests", name, {"suites": [p.stem for p in paths]})
    exp.table("cases", rows)
    exp.summary({"passed": sum(r["outcome"] == "passed" for r in rows), "failed": failed})
    exp.artifact(name, "junit", paths)
    exp.finish()
    return failed


def _flat(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flat(f"{prefix}/{k}" if prefix else str(k), v, out)
    elif isinstance(value, int | float | str | bool) or value is None:
        out[prefix] = value


def log_json(job_type: str, name: str, path: Path) -> None:
    """One W&B run with a JSON report (ground_truth.py, parity, bench): every scalar in the
    summary (nested keys joined with /), lists of records as tables, the file as artifact.
    For reports made where W&B is off (a CPU runtime without the key): upload them after."""
    report = json.loads(path.read_text(encoding="utf-8"))
    exp = Experiment(job_type, name, {"report": path.name})
    summary: dict[str, Any] = {}
    _flat("", report, summary)
    exp.summary(summary)

    def tables(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                tables(f"{prefix}/{k}" if prefix else str(k), v)
        elif isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
            rows = [{k: v for k, v in x.items() if not isinstance(v, dict | list)} for x in value]
            exp.table(prefix, rows)

    tables("", report)
    exp.artifact(name, job_type, [path])
    exp.finish()


if __name__ == "__main__":
    # python -m tjev.campaign.experiments junit NAME FILE...
    # python -m tjev.campaign.experiments log-json JOB_TYPE NAME FILE
    import sys

    if len(sys.argv) >= 4 and sys.argv[1] == "junit":
        failed = log_junit(sys.argv[2], [Path(p) for p in sys.argv[3:]])
        print(json.dumps({"junit": sys.argv[2], "failed": failed}))
    elif len(sys.argv) == 5 and sys.argv[1] == "log-json":
        log_json(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
        print(json.dumps({"logged": sys.argv[4]}))
