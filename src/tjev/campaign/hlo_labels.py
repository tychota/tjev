"""Readable names for an optimized HLO module: what a buffer or an op of the profile is.

XLA names instructions after their opcode (``multiply_add_fusion.222``, ``custom-call.133``):
our scopes survive only in the ``op_name`` metadata, and the buffers XLA creates itself carry
none. The biggest of those are the layer scan's residual stacks: the forward ``while`` writes
each layer's saved residuals into buffers allocated up front (``AllocateBuffer`` custom calls)
and the backward reads them. This module parses the HLO text (xprof's graph viewer, or the
``*after_optimizations.txt`` dump) and follows each such buffer into its ``while`` loop to
the value written into it, whose ``op_name`` says what it is; ``residuals.json``
(:mod:`tjev.campaign.residuals`) then gives it its checkpoint name by shape.

    fine_path("jit(train_step_T1024_mb32)/grad_accum/while/body/eval_jaxpr/transpose(jvp())/"
              "while/body/eval_jaxpr/checkpoint/layer1_gdn/mlp/up_proj/lora/B/dot_general")
    -> "layer1_gdn/mlp/up_proj/lora/B/dot_general"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# name-stack parts that only say how the op was reached (autodiff, loops, remat, the
# microbatch scan), not what it is
_NOISE = {"while", "body", "cond", "eval_jaxpr", "checkpoint", "rematted_computation",
          "grad_accum", "jit(step)", "", "transpose", "jvp"}  # fmt: skip
_WRAPPER = re.compile(r"^(?:transpose|jvp|vmap|remat|checkpoint|pjit|jit|custom_jvp_call|"
                      r"custom_vjp_call)\((.*)\)$")  # fmt: skip
_DTYPES = {"f32": "float32", "bf16": "bfloat16", "f16": "float16", "s32": "int32",
           "s8": "int8", "u8": "uint8", "pred": "bool", "f64": "float64", "s64": "int64",
           "u32": "uint32"}  # fmt: skip
_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "int32": 4, "int8": 1, "uint8": 1,
          "bool": 1, "float64": 8, "int64": 8, "uint32": 4}  # fmt: skip


def fine_path(op_name: str) -> str:
    """An op's jax path without the program (``jit(train_step…)``) and the autodiff / loop /
    remat wrappers: the model scopes and the primitive
    (``layer1_gdn/mlp/up_proj/lora/B/dot_general``)."""
    parts = []
    for n, raw in enumerate(op_name.rstrip(":").split("/")):
        if n == 0 and raw.startswith("jit("):
            continue  # the program's own name
        part = raw
        while m := _WRAPPER.match(part):
            part = m[1]
        if part in _NOISE:
            continue
        parts.append(part)
    if not parts and "grad_accum" in op_name:
        return "grad_accum"
    path = "/".join(parts)
    # the scans' own stacking: the residual stacks (forward) and their reads (backward)
    path = path.replace("dynamic_update_index_in_dim/dynamic_update_slice", "scan stack write")
    return path.replace("dynamic_index_in_dim/dynamic_slice", "scan stack read")


def phase_hint(op_name: str) -> str:
    if "rematted_computation" in op_name:
        return "remat"
    if "transpose(" in op_name:
        return "bwd"
    if "jvp(" in op_name:
        return "fwd"
    return ""


@dataclass
class Instr:
    name: str
    shape: str
    opcode: str
    operands: list[str]
    op_name: str
    attrs: str
    computation: str
    root: bool = False
    users: list[str] = field(default_factory=list)


def _balanced(text: str, start: int) -> int:
    """The index after the bracket group opening at ``text[start]``."""
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


_LINE = re.compile(r"^\s*(ROOT\s+)?%([^\s=]+)\s*=\s*(.*)$")
_COMP = re.compile(r"^\s*(ENTRY\s+)?%([^\s(]+)\s*\(.*\{\s*$")


def parse_hlo(text: str, op_names: dict[str, str] | None = None) -> dict[str, Any]:
    """{"computations": {name: [Instr]}, "instrs": {name: Instr}, "entry": name}.
    ``op_names``: instruction → op_name (xprof's hlo_stats ``tf_op_name`` of the same
    program) for the instructions whose printed metadata is empty (XLA's loop clones)."""
    comps: dict[str, list[Instr]] = {}
    instrs: dict[str, Instr] = {}
    current, entry = None, None
    for line in text.splitlines():
        if m := _COMP.match(line):
            current = m[2]
            comps[current] = []
            if m[1]:
                entry = current
            continue
        m = _LINE.match(line)
        if not m or current is None:
            continue
        rest = m[3]
        end = _balanced(rest, 0) if rest.startswith("(") else rest.find(" ")
        shape, after = rest[:end], rest[end:].lstrip()
        op = re.match(r"([\w\-.]+)\(", after)
        if not op:
            continue
        close = _balanced(after, op.end() - 1)
        operands = re.findall(r"%([^\s,()]+)", after[op.end() : close - 1])
        attrs = after[close:]
        meta = re.search(r'op_name="([^"]*)"', attrs)
        op_name = meta[1] if meta else (op_names or {}).get(m[2], "")
        ins = Instr(m[2], shape, op[1], operands, op_name, attrs, current, bool(m[1]))
        comps[current].append(ins)
        instrs[ins.name] = ins
    for ins in instrs.values():
        for o in ins.operands:
            if o in instrs:
                instrs[o].users.append(ins.name)
    return {"computations": comps, "instrs": instrs, "entry": entry}


def shape_key(shape: str) -> tuple[tuple[int, ...], str] | None:
    """(dims, dtype) of an array shape string (``f32[8,2,32]{2,1,0:T(8,128)}``)."""
    m = re.match(r"(\w+)\[([\d,]*)\]", shape)
    if not m or m[1] not in _DTYPES:
        return None
    dims = tuple(int(d) for d in m[2].split(",") if d)
    return dims, _DTYPES[m[1]]


def shape_bytes(shape: str) -> int:
    key = shape_key(shape)
    if key is None:
        return 0
    n = 1
    for d in key[0]:
        n *= d
    return n * _BYTES[key[1]]


def _attr(ins: Instr, key: str) -> str | None:
    m = re.search(rf"{key}=%?([^\s,}}]+)", ins.attrs)
    return m[1] if m else None


def _named_producer(module: dict[str, Any], name: str, depth: int = 12) -> Instr | None:
    """The first instruction with an op_name, walking back through layout and tuple ops."""
    instrs = module["instrs"]
    seen = 0
    while name in instrs and seen < depth:
        ins = instrs[name]
        if ins.op_name and ins.opcode not in ("get-tuple-element", "bitcast", "copy"):
            return ins
        if ins.opcode == "custom-call" and not ins.name.startswith("custom-call"):
            # a kernel (gdn_mega2_fwd.25, splash_mqa_fwd…): its name says what it is
            return Instr(ins.name, ins.shape, ins.opcode, ins.operands,
                         re.sub(r"(\.\d+)+$", "", ins.name), ins.attrs, ins.computation)  # fmt: skip
        if not ins.operands:
            return ins if ins.op_name else None
        # the largest operand: the data, not an index
        name = max(ins.operands, key=lambda o: shape_bytes(instrs[o].shape) if o in instrs else 0)
        seen += 1
    return None


def _writers(module: dict[str, Any], name: str) -> list[Instr]:
    """The named producers of the values a loop body writes into its buffer ``name`` (a
    dynamic-update-slice, or a multi-output fusion of them): its operands that are not the
    loop's own buffers (get-tuple-elements of the body parameter) nor scalar indices."""
    instrs = module["instrs"]
    write = instrs.get(name)
    while write is not None and write.opcode in ("get-tuple-element", "bitcast", "copy"):
        write = instrs.get(write.operands[0]) if write.operands else None
    if write is None:
        return []

    def own(o: str) -> bool:
        ins = instrs[o]
        return ins.opcode == "get-tuple-element" and instrs[ins.operands[0]].opcode == "parameter"

    data = [o for o in write.operands if o in instrs and not own(o)
            and shape_bytes(instrs[o].shape) > 8]  # fmt: skip
    found = [p for o in data if (p := _named_producer(module, o)) is not None]
    return found or ([write] if write.op_name else [])


def _while_slot(module: dict[str, Any], buffer: str) -> tuple[Instr, int] | None:
    """The ``while`` loop that carries ``buffer`` and its tuple index."""
    instrs = module["instrs"]
    name, index = buffer, None
    for _ in range(8):
        users = [instrs[u] for u in instrs[name].users]
        if not users:
            return None
        user = users[0]
        if user.opcode == "tuple":
            index = user.operands.index(name)
        if user.opcode == "while" and index is not None:
            return user, index
        name = user.name
    return None


def label_allocations(module: dict[str, Any], residuals: dict | None = None) -> dict[str, dict]:
    """{AllocateBuffer instruction: {"label", "producer", "phase", "bytes", "shape", "loop"}}:
    each buffer XLA allocates for a loop, named by the value its loop body writes into it
    and, with ``residuals`` (residuals.json of the same bucket), by its checkpoint name."""
    from tjev.campaign.residuals import stack_labels

    instrs, comps = module["instrs"], module["computations"]
    names = stack_labels(residuals) if residuals else {}
    out: dict[str, dict] = {}
    for ins in instrs.values():
        if ins.opcode != "custom-call" or 'custom_call_target="AllocateBuffer"' not in ins.attrs:
            continue
        entry: dict[str, Any] = {"shape": ins.shape, "bytes": shape_bytes(ins.shape)}
        slot = _while_slot(module, ins.name)
        producers: list[Instr] = []
        if slot is not None:
            loop, index = slot
            body = comps.get(_attr(loop, "body") or "", [])
            root = next((i for i in body if i.root), None)
            if root is not None and root.opcode == "tuple" and index < len(root.operands):
                producers = _writers(module, root.operands[index])
            entry["loop"] = loop.name
            entry["loop_op"] = fine_path(loop.op_name)
        key = shape_key(ins.shape)
        if producers:
            paths = list(dict.fromkeys(fine_path(p.op_name) for p in producers))
            entry["producer"] = " | ".join(paths)
            entry["phase"] = phase_hint(producers[0].op_name)
        candidates = names.get(key, []) if key else []
        if candidates:  # same-shape residuals of several layers: one per buffer, in order
            entry["residual"] = sorted({c.split("] ", 1)[1] for c in candidates})
            writer = f", written by {producers[0].name}" if producers else ""
            entry["label"] = (
                f"scan residual {' / '.join(entry['residual'])} "
                f"({len(candidates)} of this shape{writer}; {key[0][0]} super-blocks)"
            )
        elif producers:
            entry["label"] = f"loop buffer written by {entry['producer']}"
        else:
            entry["label"] = "loop buffer (XLA AllocateBuffer)"
        out[ins.name] = entry
    return out


def _vague(op_name: str) -> bool:
    """No model scope: unnamed, or named after the enclosing loop (XLA's layout copies and
    loop plumbing inherit ``…/while``)."""
    return not op_name or op_name.rstrip(":").endswith("while")


def _nearest_named(module: dict[str, Any], name: str, up: bool, depth: int = 10) -> Instr | None:
    instrs = module["instrs"]
    seen, frontier = {name}, [name]
    for _ in range(depth):
        nxt = []
        for n in frontier:
            for o in instrs[n].operands if up else instrs[n].users:
                if o in seen or o not in instrs:
                    continue
                seen.add(o)
                if not _vague(instrs[o].op_name) and instrs[o].opcode != "parameter":
                    return instrs[o]
                nxt.append(o)
        frontier = nxt
    return None


def reattribute(module: dict[str, Any], name: str, category: str = "") -> str | None:
    """A model path for an op named after its loop (``…/transpose(jvp())/while``): the op_name
    of the nearest named producer (else consumer) in the HLO, plus what the op does there
    (``relayout`` for copies and transposes, ``scan stack write`` / ``read``)."""
    ins = module["instrs"].get(name)
    if ins is None or not _vague(ins.op_name):
        return None
    near = _nearest_named(module, name, up=True) or _nearest_named(module, name, up=False)
    if near is None:
        return None
    if "dynamic-update-slice" in name:
        what = "scan stack write"
    elif "dynamic-slice" in name:
        what = "scan stack read"
    elif category in ("data formatting", "copy", "copy-start", "copy-done") or ins.opcode in (
        "copy", "transpose", "bitcast", "reshape", "copy-start", "copy-done"
    ):  # fmt: skip
        what = "relayout"
    else:
        what = "loop"
    return f"{near.op_name.rstrip(':')}/{what}"


def describe(shape: str) -> str:
    """``bf16[8,2,1024,4096]`` → ``bfloat16 8×2×1024×4096, 128 MiB``."""
    key = shape_key(shape)
    if key is None:
        return shape
    return f"{key[1]} {'×'.join(map(str, key[0]))}, {shape_bytes(shape) / 2**20:.0f} MiB"
