"""Layered configuration: dataclass defaults → preset files → ``key=value`` overrides.

A preset is a YAML file, named by path or by its stem (``tpu-v6e``, ``qwen35-2b``) under
``configs/``; it may ``extends`` other presets. Later layers win; mappings merge key by key,
except ``data.mixture``, which a layer replaces whole. Every value is coerced to its
annotated type, and unknown keys, invalid literals and malformed values are errors.
"""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from .schema import RunConfig

CONFIG_DIR = Path(__file__).with_name("configs")
REPLACED_WHOLE = frozenset({"mixture"})


def preset_names() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.rglob("*.yaml"))


def preset_path(name: str) -> Path:
    path = Path(name)
    if path.suffix in (".yaml", ".yml"):
        if path.exists():
            return path
        name = path.stem
    matches = sorted(CONFIG_DIR.rglob(f"{name}.yaml"))  # "qwen35-0.8b": not a suffix
    if len(matches) != 1:
        raise FileNotFoundError(f"no preset or file {name!r} (presets: {preset_names()})")
    return matches[0]


def merge(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key not in REPLACED_WHOLE:
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(name: str) -> dict:
    raw = yaml.safe_load(preset_path(name).read_text(encoding="utf-8")) or {}
    parents = raw.pop("extends", [])
    merged: dict = {}
    for parent in [parents] if isinstance(parents, str) else parents:
        merged = merge(merged, load_yaml(parent))
    return merge(merged, raw)


def apply_override(raw: dict, item: str) -> None:
    """``a.b.c=value``; the value is parsed as YAML (``[1, 2]``, ``true``, ``3e-5``)."""
    if "=" not in item:
        raise ValueError(f"override must be key=value: {item!r}")
    key, value = item.split("=", 1)
    node = raw
    *parents, leaf = key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(value) if value else value


def coerce(tp: Any, value: Any, where: str) -> Any:
    origin = get_origin(tp)
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected a mapping, got {value!r}")
        return build(tp, value, where)
    if origin is Literal:
        if value not in get_args(tp):
            raise ValueError(f"{where}: {value!r} is not one of {get_args(tp)}")
        return value
    if origin in (Union, types.UnionType):
        raise TypeError(f"{where}: union types are not supported in the schema")
    if origin is tuple:
        if isinstance(value, str):
            value = [v for v in value.split(",") if v]
        item, *_ = get_args(tp)
        return tuple(coerce(item, v, where) for v in value)
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected a mapping, got {value!r}")
        _, value_type = get_args(tp)
        return {str(k): coerce(value_type, v, where) for k, v in value.items()}
    if tp is bool:
        if isinstance(value, str):
            if value.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{where}: not a boolean: {value!r}")
            return value.lower() in ("true", "1")
        return bool(value)
    if tp is int and isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{where}: not an integer: {value!r}")
    if tp in (int, float, str):
        return tp(value)
    raise TypeError(f"{where}: unsupported type {tp}")


def build[T](cls: type[T], raw: dict, where: str = "") -> T:
    """``cls(**raw)`` with every value coerced; unknown keys are errors."""
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # ty: ignore[invalid-argument-type]
    unknown = set(raw) - names
    if unknown:
        raise KeyError(f"unknown config keys at {where or 'root'}: {sorted(unknown)}")
    return cls(**{k: coerce(hints[k], v, f"{where}.{k}".strip(".")) for k, v in raw.items()})


def load_config(*files: str, overrides: tuple[str, ...] | list[str] = ()) -> RunConfig:
    """Defaults → each preset or file (with ``extends``) → ``a.b=value`` overrides."""
    raw: dict = {}
    for name in files:
        raw = merge(raw, load_yaml(name))
    for item in overrides:
        apply_override(raw, item)
    return build(RunConfig, raw)


def from_dict(raw: dict) -> RunConfig:
    """A config saved by a run (``RunConfig.to_dict``)."""
    return build(RunConfig, raw)


def split_args(items: list[str]) -> tuple[list[str], list[str]]:
    """Command-line items → (preset names or files, ``key=value`` overrides)."""
    return [a for a in items if "=" not in a], [a for a in items if "=" in a]
