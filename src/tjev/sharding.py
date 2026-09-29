"""Device mesh and placement.

Mesh axes are ``("data", "fsdp")`` and the batch is split over both. With LoRA only the
adapter gradients are all-reduced, so pure data parallelism (base replicated) is the
default; ``fsdp > 1`` shards the frozen base along its largest divisible axis (chips whose
HBM cannot hold a replica, e.g. 4B on 16 GB TPU v5e). Adapters and optimizer state are
replicated (they are small).
"""

from __future__ import annotations

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
