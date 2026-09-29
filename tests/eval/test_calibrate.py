"""Temperature scaling: recovers a known temperature, never hurts, and binds its artifact."""

import numpy as np
import pytest

from tjev.eval.calibrate import artifact, fit_per_type, fit_temperature, validate


def _sampled(n=3000, k=4, temperature=2.5, seed=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logits whose labels are drawn from softmax(logits / temperature)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 3, (n, k))
    p = np.exp(logits / temperature)
    p /= p.sum(-1, keepdims=True)
    gold = np.array([rng.choice(k, p=row) for row in p])
    return logits, np.ones((n, k), bool), np.eye(k)[gold]


def test_fit_recovers_the_temperature_and_never_hurts():
    logits, mask, target = _sampled()
    t, before, after = fit_temperature(logits, mask, target)
    assert abs(t - 2.5) < 0.25
    assert after < before
    t, before, after = fit_temperature(*_sampled(temperature=1.0))
    assert after <= before


def test_per_type_fit_falls_back_to_global_on_few_slots():
    logits, mask, target = _sampled(n=600)
    types = ["choice"] * 580 + ["score"] * 20
    fit = fit_per_type(logits, mask, target, types)
    assert fit["detail"]["score"]["fallback"] == "global"
    assert fit["temperatures"]["score"] == fit["global"]


def test_artifact_is_bound_to_what_it_was_fitted_on():
    fit = fit_per_type(*_sampled(n=200), ["noul"] * 200)
    art = artifact(fit, checkpoint="run@10", data="d", template=1, backend="jax")
    assert validate(art, checkpoint="run@10", template=1, backend="jax") == fit["temperatures"]
    with pytest.raises(ValueError, match="checkpoint"):
        validate(art, checkpoint="run@20", template=1, backend="jax")
    with pytest.raises(ValueError, match="backend"):
        validate(art, checkpoint="run@10", template=1, backend="mlx")
    with pytest.raises(ValueError, match="corrupt"):
        validate({**art, "temperatures": {"noul": 9.0}}, checkpoint="run@10", template=1,
                 backend="jax")  # fmt: skip
