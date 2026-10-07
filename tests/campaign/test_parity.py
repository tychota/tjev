"""tjev campaign parity: the comparison and the pass rule (no model: synthetic outputs)."""

import numpy as np

from tjev.campaign import parity


def _out(grads: dict[str, np.ndarray], loss: float = 10.0) -> dict:
    logits = np.zeros((2, 3, 4), np.float32)
    return {"batches": [{
        "length": 1024, "loss_sum": loss, "logits": logits,
        "label_mask": np.ones_like(logits, bool), "grads": grads, "fwd_bwd_s": 0.1,
    }]}  # fmt: skip


def test_compare_measures_gradient_error_and_cosine():
    rng = np.random.default_rng(0)
    ref = {"a": rng.normal(size=100).astype(np.float32), "b": rng.normal(size=50)}
    noisy = {k: v + 1e-3 * rng.normal(size=v.shape) for k, v in ref.items()}
    same = parity.compare(_out(ref), _out(ref))
    close = parity.compare(_out(noisy, loss=10.001), _out(ref))
    assert same["grad_rel_l2"] == 0 and abs(same["grad_cosine"] - 1) < 1e-6
    assert 0 < close["grad_rel_l2"] < 1e-2 and close["grad_cosine"] > 0.9999
    assert abs(close["loss_rel"] - 1e-4) < 1e-6
    assert {x["leaf"] for x in close["worst_leaves"]} == {"a", "b"}


def test_verdict_bounds_kernels_by_the_xla_error():
    ok = {
        "grad_finite": True,
        "grad_cosine": 0.99999,
        "worst_leaf_cosine": 0.9999,
        "loss_rel": 1e-4,
    }
    results = {
        "xla": ok | {"grad_rel_l2": 2e-2},
        "kernels": ok | {"grad_rel_l2": 3.9e-2},  # within 2 × xla
        "split": ok | {"grad_rel_l2": 4.1e-2},  # beyond 2 × xla = 4e-2
        "bad-cos": ok | {"grad_rel_l2": 1e-3, "grad_cosine": 0.999},
        "bad-leaf": ok | {"grad_rel_l2": 1e-3, "worst_leaf_cosine": 0.99},
        "broken": {"error": "compile"},
    }
    parity.verdict(results)
    assert [n for n, r in results.items() if r["pass"]] == ["xla", "kernels"]
    assert results["kernels"]["grad_bound"] == 4e-2


def test_verdict_floor_is_the_bf16_scale():
    ok = {"grad_finite": True, "grad_cosine": 0.9999, "worst_leaf_cosine": 0.999, "loss_rel": 1e-3}
    results = {"xla": ok | {"grad_rel_l2": 1e-3}, "gdn": ok | {"grad_rel_l2": 2.1e-2}}
    parity.verdict(results)
    assert results["gdn"]["pass"] and results["gdn"]["grad_bound"] == parity.GRAD_FLOOR


def test_shared_lora_sets_only_lora_b():
    tree = {"layer": {"lora_a": np.ones((4, 2), np.float32), "lora_b": np.zeros((2, 4))}}
    out = parity.shared_lora(tree)
    assert np.array_equal(out["layer"]["lora_a"], tree["layer"]["lora_a"])
    assert 0 < float(np.std(out["layer"]["lora_b"])) < 0.05
