import os

# Before JAX is imported anywhere: CPU with 4 virtual devices, so the sharded (data, fsdp)
# paths and the shard_map around the Pallas kernels run for real in every test.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import pytest

jax.config.update("jax_default_matmul_precision", "highest")


def pytest_runtest_makereport(item, call):
    # Tests never skip silently: a skip is a failure. Hardware- or weight-dependent tests are
    # deselected by marker (pyproject addopts), never skipped at run time.
    if call.excinfo is not None and call.excinfo.errisinstance(pytest.skip.Exception):
        call.excinfo = None
        raise AssertionError(f"{item.nodeid} was skipped; skips are not allowed")
