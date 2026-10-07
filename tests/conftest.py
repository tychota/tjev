import os

# Before JAX is imported anywhere: CPU with 4 virtual devices, so the sharded (data, fsdp)
# paths and the shard_map around the Pallas kernels run for real in every test.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import pytest

jax.config.update("jax_default_matmul_precision", "highest")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    # Tests never skip silently: a skip is a failure. Hardware- or weight-dependent tests are
    # deselected by marker (pyproject addopts), never skipped at run time. The report is
    # turned into a failure (raising here would be a pytest INTERNALERROR: exit 3, the
    # session aborted); an xfail (also reported as skipped) stays one.
    report = yield
    if report.skipped and not hasattr(report, "wasxfail"):
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr
        report.outcome = "failed"
        report.longrepr = f"{item.nodeid} was skipped ({reason}); skips are not allowed"
    return report
