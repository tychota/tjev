"""Decision metrics with soft targets: expected correctness, KL, ECE."""

import numpy as np

from tjev.eval.metrics import summarize


def _soft(n=400, k=3, seed=0):
    rng = np.random.default_rng(seed)
    target = rng.dirichlet(np.ones(k), size=n)
    return target, np.ones((n, k), bool)


def test_a_model_that_outputs_a_soft_target_is_calibrated():
    target, mask = _soft()
    report = summarize(target, target, mask)
    assert report["ece"] < 1e-9
    assert abs(report["kl"]) < 1e-9
    assert abs(report["accuracy"] - target.max(-1).mean()) < 1e-12  # expected correctness


def test_one_hot_targets_keep_the_argmax_accuracy():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), size=300)
    gold = rng.integers(0, 4, 300)
    target = np.eye(4)[gold]
    report = summarize(probs, target, np.ones((300, 4), bool))
    assert report["accuracy"] == (probs.argmax(-1) == gold).mean()
    assert abs(report["kl"] - report["nll"]) < 1e-9  # H(one-hot) = 0
