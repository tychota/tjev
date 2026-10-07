"""Statistics of the compiler dumps of one compile: Mosaic MLIR passes, LLO / VLIW bundles, HLO.

    summarize_dumps(DIR) -> {"mosaic": {...}, "llo": {...}, "hlo": {...}}

``DIR`` holds the dumps of one process (tjev.campaign.journey ``ir-compile``):

* ``mosaic/`` (``--xla_mosaic_dump_to``): one numbered file per Mosaic pass and kernel,
  ``0001-original.txt`` … ``00NN-post-finalize-llo.txt`` (the pass index may also sit
  after a kernel prefix, or the files of a kernel in its own directory).
* ``llo/`` (``--xla_jf_dump_to`` with ``--xla_jf_dump_llo_text=true``): the LLO passes of
  every compiled module and Pallas kernel, numbered the same way; the last pass holds the
  scheduled VLIW bundles. A bundle is one line starting with its address,
  ``0x1f  : > { %p222 = pnand %p51265, %p221  ;;  %s202 = scalar_lea.vmem … }``: the ops
  between ``;;`` issue together; the MXU ops name their unit (``vmatmul.….mxu0``,
  ``vmatpush1.bf16.xpose.msra.mxu1``, ``vpop.f32.mrf.mxu0``). Allocations are listed as
  ``#allocation1 [shape = 'u32[144,128]', space=vmem, size = 0x12000]``.
* ``hlo/`` (``--xla_dump_to`` + ``--xla_dump_hlo_as_text``): ``*after_optimizations.txt``,
  the fusions and custom calls XLA really runs.

The dump layouts are libtpu's and undocumented: files are grouped by what remains of their
name (and directory) without the pass index, and a group is named after the Pallas kernel
it mentions (``gdn_v2_fwd``, ``gdn_mega2_bwd``, … ``name=`` of each ``pallas_call``) when
there is one. Every number here is static (what the compiler emitted), not a timing.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path
from typing import Any

BUNDLE = re.compile(r"^\s*0x[0-9a-fA-F]+\s*:", re.MULTILINE)
MXU_OP = re.compile(r"\b([a-z][a-z0-9]*)((?:\.[A-Za-z0-9_]+)*?)\.mxu(\d)\b")
ALLOCATION = re.compile(r"#allocation(\d+)\s*\[([^\n]*)")  # its shape may hold brackets
KERNEL = re.compile(r"\b(gdn_[a-z0-9]+(?:_[a-z0-9]+)*?_(?:fwd|bwd)(?:_dw)?)\b")
SUFFIXES = (".txt", ".mlir", ".llo", ".ll")
# "0001-original.txt" | "<prefix>.0012.post-finalize-llo.txt" | "<prefix>-05-sched.txt"
LEADING = re.compile(r"^(?P<idx>\d{1,5})[-_.](?P<name>.+)$")
INNER = re.compile(
    r"^(?P<prefix>.+)[-_.](?P<idx>\d{1,5})[-_.](?P<name>[A-Za-z][^/]*)$"
)  # the last index


def count_bundles(text: str) -> int:
    """VLIW bundles: the lines that start with a bundle address (``0x1f  :``)."""
    return len(BUNDLE.findall(text))


def bundle_ops(text: str) -> int:
    """Ops issued in the bundles: one per bundle plus one per ``;;`` separator inside them."""
    total = 0
    for block in _bundles(text):
        total += 1 + block.count(";;")
    return total


def _bundles(text: str) -> list[str]:
    """Each bundle's text: from its address line up to the next address (or a blank line)."""
    out, current = [], None
    for line in text.splitlines():
        if BUNDLE.match(line):
            if current is not None:
                out.append(current)
            current = line
        elif current is not None:
            if not line.strip():
                out.append(current)
                current = None
            else:
                current += "\n" + line
    if current is not None:
        out.append(current)
    return out


def mxu_ops(text: str) -> dict[str, Any]:
    """MXU ops by family (``vmatmul``, ``vmatpush``, ``vmatprep``, ``vpop``, …; the digits
    of ``vmatpush1`` dropped) and by unit (``mxu0``, ``mxu1``), and their total."""
    families: collections.Counter[str] = collections.Counter()
    units: collections.Counter[str] = collections.Counter()
    for m in MXU_OP.finditer(text):
        families[re.sub(r"\d+$", "", m[1])] += 1
        units[f"mxu{m[3]}"] += 1
    return {
        "total": sum(families.values()),
        "by_op": dict(families.most_common()),
        "by_unit": dict(sorted(units.items())),
    }


def _int(text: str) -> int:
    return int(text, 16) if text.lower().startswith("0x") else int(text)


def allocations(text: str) -> dict[str, Any]:
    """The ``#allocationN [...]`` declarations: count and bytes per memory space (each id
    once; a size in hex or decimal)."""
    seen: dict[str, tuple[str, int]] = {}
    for m in ALLOCATION.finditer(text):
        body = m[2]
        space = re.search(r"space\s*=\s*([A-Za-z_]+)", body)
        size = re.search(r"size\s*=\s*(0x[0-9a-fA-F]+|\d+)", body)
        if space and size:
            seen.setdefault(m[1], (space[1].lower(), _int(size[1])))
    by_space: collections.Counter[str] = collections.Counter()
    for space, size in seen.values():
        by_space[space] += size
    return {"count": len(seen), "bytes": dict(sorted(by_space.items()))}


def pass_of(path: Path, root: Path) -> tuple[str, int | None, str]:
    """(group, pass index or None, pass name) of a dump file: the group is the file's
    directory under ``root`` plus whatever prefixes the pass index in its name."""
    name = path.name
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    parent = path.parent.relative_to(root).as_posix()
    parent = "" if parent == "." else parent
    if m := LEADING.match(name):
        return parent or ".", int(m["idx"]), m["name"]
    if m := INNER.match(name):
        group = f"{parent}/{m['prefix']}" if parent else m["prefix"]
        return group, int(m["idx"]), m["name"]
    return (f"{parent}/{name}" if parent else name), None, name


def kernel_name(group: str, texts: list[str]) -> str | None:
    """The Pallas kernel a dump group belongs to: named in the group, else in its files."""
    if m := KERNEL.search(group):
        return m[1]
    for text in texts:
        if m := KERNEL.search(text):
            return m[1]
    return None


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in SUFFIXES)


def _read(path: Path, limit: int = 256 << 20) -> str:
    with path.open("rb") as f:
        return f.read(limit).decode("utf-8", "replace")


def summarize_passes(root: Path) -> dict[str, dict[str, Any]]:
    """Per dump group (a kernel or an XLA module): its passes and, from its last pass, the
    bundles, ops per bundle, MXU ops and allocations. Keyed by the kernel name when one is
    found (``gdn_v2_fwd``), else by the group."""
    groups: dict[str, list[tuple[int | None, str, Path]]] = collections.defaultdict(list)
    for path in _files(root):
        group, idx, name = pass_of(path, root)
        groups[group].append((idx, name, path))
    out: dict[str, dict[str, Any]] = {}
    for group, files in sorted(groups.items()):
        indexed = sorted((f for f in files if f[0] is not None), key=lambda f: f[0])
        last = indexed[-1] if indexed else files[-1]
        text = _read(last[2])
        first = _read(indexed[0][2], 1 << 20) if indexed else ""
        kernel = kernel_name(group, [first, text])
        stats = {
            "group": group,
            "kernel": kernel,
            "passes": len(indexed),
            "pass_names": [f[1] for f in indexed],
            "last_pass": last[2].relative_to(root).as_posix(),
            "bundles": count_bundles(text),
            "mxu": mxu_ops(text),
            "allocations": allocations(text),
        }
        if stats["bundles"]:
            ops = bundle_ops(text)
            stats["ops"] = ops
            stats["ops_per_bundle"] = round(ops / stats["bundles"], 3)
        key = kernel or group
        while key in out:  # two groups of one kernel (e.g. fwd compiled twice): keep both
            key += "'"
        out[key] = stats
    return out


HLO_FUSION = re.compile(r"=\s*[^=\n]*?\bfusion\(", re.MULTILINE)
HLO_CUSTOM = re.compile(r'custom_call_target="([^"]+)"')


def hlo_stats(text: str) -> dict[str, Any]:
    """Fusions and custom calls of an optimised HLO module (``*after_optimizations.txt``)."""
    targets = collections.Counter(HLO_CUSTOM.findall(text))
    return {
        "fusions": len(HLO_FUSION.findall(text)),
        "custom_calls": dict(targets.most_common()),
        "instructions": sum(
            1
            for line in text.splitlines()
            if " = " in line and not line.lstrip().startswith(("HloModule", "//"))
        ),
    }


def summarize_hlo(root: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(root.rglob("*after_optimizations*.txt")):
        if "buffer-assignment" in path.name or "memory-usage" in path.name:
            continue
        out[path.name] = hlo_stats(_read(path))
    return out


def totals(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The Pallas kernels' bundles one by one (fwd / bwd), and the rest (XLA) summed."""
    kernels = {k: v["bundles"] for k, v in groups.items() if v.get("kernel")}
    xla = sum(v["bundles"] for v in groups.values() if not v.get("kernel"))
    mxu = sum(v["mxu"]["total"] for v in groups.values())
    return {
        "kernel_bundles": kernels,
        "fwd_bundles": sum(b for k, b in kernels.items() if "_fwd" in k),
        "bwd_bundles": sum(b for k, b in kernels.items() if "_bwd" in k),
        "xla_bundles": xla,
        "all_bundles": xla + sum(kernels.values()),
        "mxu_ops": mxu,
        "kernel_vmem_bytes": {
            k: v["allocations"]["bytes"].get("vmem", 0)
            for k, v in groups.items()
            if v.get("kernel")
        },
        "llo_passes": max((v["passes"] for v in groups.values()), default=0),
    }


def summarize_dumps(root: Path) -> dict[str, Any]:
    root = Path(root)
    out: dict[str, Any] = {}
    for name in ("mosaic", "llo"):
        if (root / name).is_dir():
            groups = summarize_passes(root / name)
            out[name] = {
                "groups": groups,
                "totals": totals(groups),
                "files": len(_files(root / name)),
            }
    if (root / "hlo").is_dir():
        out["hlo"] = summarize_hlo(root / "hlo")
    return out
