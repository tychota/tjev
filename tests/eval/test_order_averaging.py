"""Order-averaged distributions on host arrays: one view is exactly the plain softmax, more
views are the mean of their softmaxes, temperatures are fitted on the averaged
distribution, and the calibration artifact is bound to the readout."""

import numpy as np
import pytest

from tjev.eval.calibrate import artifact, fit_per_type, fit_temperature, nll, validate
from tjev.eval.metrics import grouped_report, label_mask, softmax, view_probs
from tjev.utils import fingerprint


def _logits(n=400, k=5, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2, (n, k))
    mask = np.ones((n, k), bool)
    mask[: n // 2, 3:] = False  # half the items have 3 options
    gold = np.where(np.arange(n) < n // 2, rng.integers(0, 3, n), rng.integers(0, k, n))
    return logits, mask, np.eye(k)[gold]


def test_one_view_is_exactly_the_plain_softmax():
    logits, mask, target = _logits()
    temps = np.linspace(0.5, 3.0, len(logits))
    np.testing.assert_array_equal(view_probs(logits[:, None], mask[:, None], temps),
                                  softmax(logits, mask, temps))  # fmt: skip
    for t in (1.0, 2.3):
        assert nll(logits[:, None], mask[:, None], target, t) == nll(logits, mask, target, t)
    assert fit_temperature(logits[:, None], mask[:, None], target) == fit_temperature(
        logits, mask, target
    )
    types = ["choice"] * 200 + ["score"] * 200
    groups = {"type": types, "source": ["a", "b"] * 200}
    assert grouped_report(logits[:, None], mask[:, None], target, groups, {"score": 2.0}) == (
        grouped_report(logits, mask, target, groups, {"score": 2.0})
    )
    assert fit_per_type(logits[:, None], mask[:, None], target, types) == fit_per_type(
        logits, mask, target, types
    )


def test_views_average_their_softmaxes_and_absent_views_are_ignored():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2, (3, 3, 4))
    mask = np.ones((3, 3, 4), bool)
    mask[0, 2] = False  # item 0: two views only
    mask[1, :, 3] = False  # item 1: three labels
    logits[0, 2] = 1e6  # an absent view's values never matter
    temps = np.array([1.0, 2.0, 0.5])
    got = view_probs(logits, mask, temps)
    for i in range(3):
        views = [v for v in range(3) if mask[i, v].any()]
        want = np.mean([softmax(logits[i, v][None], mask[i, v][None], temps[i])[0]
                        for v in views], 0)  # fmt: skip
        np.testing.assert_allclose(got[i], want, rtol=1e-12)
    np.testing.assert_allclose(got.sum(-1), 1.0)
    assert got[1, 3] == 0.0
    np.testing.assert_array_equal(label_mask(mask)[1], [True, True, True, False])
    with pytest.raises(ValueError, match="at least one view"):
        view_probs(logits, np.zeros_like(mask), 1.0)


def test_averaging_two_mirrored_views_removes_a_position_bias():
    """A model that adds +b to whatever is shown first: averaging an ordering and its
    reverse leaves both options with the same mass when their content is the same."""
    bias, n = 1.5, 200
    content = np.zeros((n, 2))
    logits = np.stack(
        [content + np.array([bias, 0.0]), content + np.array([0.0, bias])], 1
    )  # canonical order
    probs = view_probs(logits, np.ones_like(logits, bool))
    np.testing.assert_allclose(probs, 0.5)
    single = softmax(logits[:, 0], np.ones((n, 2), bool))
    assert abs(single[0, 0] - 0.5) > 0.2


def test_temperature_is_fitted_on_the_averaged_distribution():
    rng = np.random.default_rng(2)
    n, k = 2000, 4
    z = rng.normal(0, 3, (n, k))
    p = np.exp(z / 2.0)
    p /= p.sum(-1, keepdims=True)
    gold = np.array([rng.choice(k, p=row) for row in p])
    target = np.eye(k)[gold]
    views = np.stack([z + rng.normal(0, 1.0, z.shape) for _ in range(2)], 1)
    mask = np.ones_like(views, bool)
    t2, before, after = fit_temperature(views, mask, target)
    assert after <= before
    assert after == pytest.approx(nll(views, mask, target, t2))
    t1, *_ = fit_temperature(views[:, :1], mask[:, :1], target)
    assert t1 != t2  # a K = 1 temperature is not the K = 2 one


def test_artifact_records_the_readout_and_refuses_another():
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 1, (100, 2, 2))
    fit = fit_per_type(logits, np.ones_like(logits, bool), np.eye(2)[rng.integers(0, 2, 100)],
                       ["noul"] * 100)  # fmt: skip
    common = {"checkpoint": "run@1", "template": 1, "backend": "jax"}
    art = artifact(fit, data="d", orders=2, noul_order="fixed", **common)
    assert art["orders"] == 2 and art["noul_order"] == "fixed"
    assert validate(art, orders=2, noul_order="fixed", **common) == fit["temperatures"]
    with pytest.raises(ValueError, match="orders=2, not 1"):
        validate(art, **common)
    with pytest.raises(ValueError, match="noul_order"):
        validate(art, orders=2, **common)
    # an artifact from before the readout was recorded: orders 1, shuffled noul
    legacy = {k: v for k, v in artifact(fit, data="d", **common).items()
              if k not in ("orders", "noul_order", "identity")}  # fmt: skip
    legacy["identity"] = fingerprint(legacy)
    assert validate(legacy, **common) == fit["temperatures"]
    with pytest.raises(ValueError, match="orders=1, not 2"):
        validate(legacy, orders=2, **common)
