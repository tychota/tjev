"""The source record every adapter module registers, and the download cache."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Downloads that are not Hugging Face datasets (GitHub files, sampled CSVs).
CACHE = Path(os.environ.get("TJEV_CACHE", Path.home() / ".cache" / "tjev"))

Row = dict
Adapter = Callable[[Row, np.random.Generator, dict], list[dict]]


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    config: str | None
    train_split: str
    eval_split: str | None  # None: hash-split a held-out slice of train
    adapter: Adapter
    family: str
    lang: str
    license: str
    commercial_ok: bool
    cap: int = 20000
    # extra load_dataset kwargs (revision/data_dir/data_files for script-only repos), or a
    # custom loader ``load(split) -> Dataset`` (e.g. a streamed sample of a huge split)
    load_kwargs: dict = field(default_factory=dict)
    load: Callable | None = None
    # eval_split None: held-out rows are a hash-split of train by this group key (rows of
    # one group, e.g. one source document, never straddle train and held-out)
    group: Callable[[Row], str] | None = None
    near_dedup: bool = False  # drop train rows near-duplicate (5-gram Jaccard) of held-out
