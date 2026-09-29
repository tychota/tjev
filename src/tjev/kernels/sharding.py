"""Run a per-device Pallas kernel under data parallelism.

XLA cannot partition a Pallas custom call: on a multi-device mesh it would replicate it
(every device doing every row). Inside a jitted step the kernels therefore run under
``shard_map`` over the batch axis, which is split over all mesh axes (as the batch is,
``tjev.sharding.BATCH_AXES``). The mesh comes from ``jax.set_mesh``, which
``tjev.sharding.install_mesh`` sets for runs that use these kernels.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
from jax.sharding import PartitionSpec as P


def batch_parallel[T](fn: Callable[..., T], *args: jax.Array) -> T:
    """``fn(*args)`` on each device's rows; every argument and output is batch-leading."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty or mesh.size == 1:
        return fn(*args)
    spec = P(tuple(mesh.axis_names))
    return jax.shard_map(fn, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False)(*args)
