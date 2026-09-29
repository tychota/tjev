"""Content identities and atomic JSON writes, shared by runs, mixes and calibration artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    """Key-sorted, whitespace-free JSON: equal values give equal strings."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: Any) -> str:
    """SHA-256 of the canonical JSON of ``value``."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    """SHA-256 of a file's bytes, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Atomic JSON write: a temporary file in the same directory, then a rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)
