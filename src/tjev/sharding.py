"""Device mesh and placement.

Mesh axes are ``("data", "fsdp")`` and the batch is split over both. With LoRA only the
adapter gradients are all-reduced, so pure data parallelism (base replicated) is the
default; ``fsdp > 1`` shards the frozen base along its largest divisible axis (chips whose
HBM cannot hold a replica, e.g. 4B on 16 GB TPU v5e). Adapters and optimizer state are
replicated (they are small).
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tjev.config import ComputeSpec, MeshSpec

BATCH_AXES = ("data", "fsdp")


def make_mesh(spec: MeshSpec, devices: Sequence[jax.Device] | None = None) -> Mesh:
    devices = list(devices if devices is not None else jax.devices())
    fsdp = spec.fsdp
    data = spec.data if spec.data > 0 else len(devices) // fsdp
    if data * fsdp != len(devices):
        raise ValueError(f"mesh data={data} × fsdp={fsdp} != {len(devices)} devices")
    return Mesh(np.asarray(devices).reshape(data, fsdp), BATCH_AXES)


def install_mesh(compute: ComputeSpec, mesh: Mesh) -> None:
    """Make ``mesh`` the process-wide mesh when the Pallas TPU kernels run: they execute
    under shard_map over the batch (:mod:`tjev.kernels.sharding`), which reads it."""
    jax.set_mesh(mesh if compute.uses_tpu_kernels else None)


def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


def batch_sharding(mesh: Mesh, *, stacked: bool) -> NamedSharding:
    """Rows are split over all devices; a leading accumulation axis is not."""
    return NamedSharding(mesh, P(None, BATCH_AXES) if stacked else P(BATCH_AXES))


def weight_spec(shape: tuple[int, ...], fsdp: int) -> P:
    if fsdp <= 1 or len(shape) < 2:
        return P()
    # Skip the leading stacked-layer axis for 3-D weights; pick the largest divisible axis.
    start = 1 if len(shape) >= 3 else 0
    candidates = [(shape[i], i) for i in range(start, len(shape)) if shape[i] % fsdp == 0]
    if not candidates:
        return P()
    _, axis = max(candidates)
    spec: list[str | None] = [None] * len(shape)
    spec[axis] = "fsdp"
    return P(*spec)


def place_frozen(frozen: Any, mesh: Mesh) -> Any:
    """Shard or replicate the frozen base. Callers must drop other references to the
    original arrays, or the base exists twice on device during the transfer."""
    if mesh.size == 1:
        return frozen  # already on the only device: no second copy of the base
    fsdp = mesh.shape["fsdp"]
    return jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(mesh, weight_spec(x.shape, fsdp))), frozen
    )


def place_replicated(tree: Any, mesh: Mesh) -> Any:
    return jax.device_put(tree, replicated(mesh))


def configure_runtime(compute: ComputeSpec) -> str:
    """Process settings that must precede JAX's backend initialisation (call first). Returns
    where the process's LIBTPU_INIT_ARGS come from: "env" (set before the process: e.g.
    cloud/tpu_env.sh's probed subset of the preset's flags), "preset" (compute.libtpu_flags,
    set here) or "none"."""
    if "LIBTPU_INIT_ARGS" in os.environ:
        return "env"
    if compute.libtpu_flags:
        os.environ["LIBTPU_INIT_ARGS"] = " ".join(compute.libtpu_flags)
        return "preset"
    return "none"


SCOPED_VMEM_FLAG = re.compile(r"--xla_tpu_scoped_vmem_limit_kib=(\d+)")


def libtpu_settings(source: str = "") -> dict[str, Any]:
    """The libtpu flags this process runs with (LIBTPU_INIT_ARGS, after configure_runtime)
    and the scoped VMEM limit they imply: XLA's ``--xla_tpu_scoped_vmem_limit_kib`` (None:
    libtpu's default) and the Pallas GDN kernels' vmem_limit_bytes, 90% of it
    (tjev.kernels.gated_delta_rule.pallas_tpu_v2._vmem_limit; None: Mosaic's default)."""
    args = os.environ.get("LIBTPU_INIT_ARGS", "")
    found = SCOPED_VMEM_FLAG.search(args)
    kib = int(found.group(1)) if found else None
    return {
        "libtpu_init_args": args,
        "libtpu_source": source,
        "scoped_vmem_limit_kib": kib,
        "kernel_vmem_limit_bytes": int(0.9 * kib * 1024) if kib else None,
    }
