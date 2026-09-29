"""Metric sinks: JSONL (always), TensorBoard (tensorboardX), W&B (optional)."""

from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any


def _flatten(prefix: str, value, out: dict):
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}/{k}" if prefix else str(k), v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


class MetricLogger:
    """``resume_step``: the checkpoint step a resumed run restarts from. Metrics logged past
    it by the interrupted process are discarded in every sink (JSONL rows dropped,
    TensorBoard ``purge_step``, W&B rewind), so resumed curves continue instead of overlapping.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        tensorboard: bool,
        wandb: bool,
        project: str,
        config: dict,
        name: str,
        resume_step: int = 0,
        entity: str = "",
        group: str = "",
        tags: tuple[str, ...] = (),
        mode: str = "online",
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "metrics.jsonl"
        if path.exists():  # also a restart before the first checkpoint (resume_step 0)
            kept = [
                x for x in path.read_text().splitlines() if json.loads(x)["step"] <= resume_step
            ]
            path.write_text("".join(x + "\n" for x in kept))
        self.jsonl = path.open("a", buffering=1)
        self.tb = None
        if tensorboard:
            from tensorboardX import SummaryWriter

            tb_dir = self.directory / "tb"
            purge = {"purge_step": resume_step + 1} if tb_dir.exists() else {}
            self.tb = SummaryWriter(str(tb_dir), **purge)
        self.wandb = None
        if wandb:
            import wandb as wb

            id_file = self.directory / "wandb_id.txt"
            run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
            id_file.write_text(run_id)
            common: dict[str, Any] = {
                "project": project,
                "name": name,
                "config": config,
                "dir": str(self.directory),
                "group": group or None,
                "tags": list(tags) or None,
                "job_type": name.split("-", 1)[0],  # the campaign phase (s1, s2, final, …)
                "mode": mode,
                **({"entity": entity} if entity else {}),
            }
            self.wandb = None
            if resume_step:  # rewind: drop the interrupted process's steps past the checkpoint
                try:
                    self.wandb = wb.init(resume_from=f"{run_id}?_step={resume_step}", **common)
                except Exception as e:
                    print(json.dumps({"event": "wandb_rewind_failed", "error": str(e)[:200]}))
            if self.wandb is None:
                self.wandb = wb.init(id=run_id, resume="allow", **common)

    def log(self, step: int, metrics: dict) -> None:
        flat: dict[str, float] = {}
        _flatten("", metrics, flat)
        self.jsonl.write(json.dumps({"step": step, "time": time.time(), **flat}) + "\n")
        if self.tb is not None:
            for key, value in flat.items():
                if math.isfinite(value):
                    self.tb.add_scalar(key, value, step)
        if self.wandb is not None:
            self.wandb.log(flat, step=step)

    def summary(self, values: dict) -> None:
        """Run-level values (the selected checkpoint): W&B summary and summary.json."""
        flat: dict[str, float] = {}
        _flatten("", values, flat)
        (self.directory / "summary.json").write_text(json.dumps(flat, indent=2) + "\n")
        if self.wandb is not None:
            self.wandb.summary.update(flat)

    def close(self) -> None:
        self.jsonl.close()
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()
