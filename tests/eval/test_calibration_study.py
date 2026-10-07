"""scripts/calibration_study.py on synthetic dumps whose best grouping is known: French
choice is over-confident (true T = 2 vs 1), so per (type × lang) must beat per type on the
validation split, and the ridge must not lose to the unshrunk cells."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from tjev.eval import calibrate as cal
from tjev.eval import predictions as dump

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "calibration_study", ROOT / "scripts" / "calibration_study.py"
)
cs = importlib.util.module_from_spec(_spec)
sys.modules["calibration_study"] = cs
_spec.loader.exec_module(cs)

TRUE_T = {("choice", "en"): 1.0, ("choice", "fr"): 2.0, ("noul", "en"): 1.3,
          ("noul", "fr"): 1.3, ("score", "en"): 1.1, ("score", "fr"): 1.1}  # fmt: skip


def _rows(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        kind = rng.choice(["choice", "noul", "score"], p=[0.45, 0.45, 0.1])
        lang = "fr" if rng.random() < 0.3 else "en"
        k = 2 if kind == "noul" else int(rng.integers(3, 7))
        z = rng.normal(0, 3, k)
        p = np.exp(z / TRUE_T[(kind, lang)])
        gold = rng.choice(k, p=p / p.sum())
        order = rng.permutation(k).tolist()
        rows.append({"id": f"{seed}-{i}", "source": f"src_{kind}_{lang}", "lang": lang,
                     "lang_guess": lang if rng.random() < 0.9 else None, "family": "f",
                     "type": str(kind),
                     "labels": ["no", "yes"] if kind == "noul" else [str(j) for j in range(k)],
                     "target": np.eye(k)[gold].tolist(), "logits": [z.round(4).tolist()],
                     "orders": [order], "probs": (p / p.sum()).tolist()})  # fmt: skip
    return rows


def test_study_ranks_the_known_grouping_first(tmp_path):
    for split, n, seed in (("calibration", 3000, 0), ("validation", 2000, 1),
                           ("generators", 300, 2)):  # fmt: skip
        dump.write(tmp_path / "pred" / f"{split}.jsonl.gz", _rows(n, seed))
    sys.argv = ["calibration_study.py", str(tmp_path / "pred"), "--out", str(tmp_path / "out"),
                "--folds", "2"]  # fmt: skip
    cs.main()
    res = json.loads((tmp_path / "out" / "study.json").read_text())["methods"]

    def val(name):
        return res[name]["validation"]["all"]["nll"]

    impl = "per type × lang, ridge (CV tau) [impl]"
    assert val(impl) < val("per type (old artifact)") < val("none (T = 1)")
    assert val(impl) <= val("per type × lang, unshrunk") + 0.003
    assert abs(val("per type × lang, ridge, served (guessed lang)") - val(impl)) < 0.02
    assert res[impl]["cv_nll"] < res["per type (old artifact)"]["cv_nll"]
    temps = res[impl]["params"]["temperatures"]
    assert abs(temps["choice/fr"] - 2.0) < 0.35 and abs(temps["choice/en"] - 1.0) < 0.2
    assert (tmp_path / "out" / "study.md").read_text().count("|") > 50


# ------------------------------------------------- the follow-up methods, each on a known effect


def _scenario(n, seed, *, t_of_n=False, pos_bias=0.0, yes_bias=0.0, views=1, noise=0.0,
              checkpoint_noise=None):  # fmt: skip
    """Rows whose gold is drawn from softmax(z) while the model reports
    z / T + effects: T = (n / 5)^0.8 (``t_of_n``) else 1; ``pos_bias`` added to the
    option displayed first; ``yes_bias`` to noul's "yes"; ``noise`` per view;
    ``checkpoint_noise``: a second checkpoint's rows (same items, its own noise)."""
    rng = np.random.default_rng(seed)
    rows, rows2 = [], []
    for i in range(n):
        kind = "noul" if rng.random() < 0.4 else "choice"
        k = 2 if kind == "noul" else int(rng.integers(3, 11))
        z = rng.normal(0, 2.5, k)
        gold = rng.choice(k, p=np.exp(z) / np.exp(z).sum())
        t = np.exp(0.8 * (np.log(k) - np.log(5))) if t_of_n else 1.0

        row = {"id": f"{seed}-{i}", "source": "s", "lang": "en", "lang_guess": "en",
               "family": "f", "type": kind,
               "labels": ["no", "yes"] if kind == "noul" else [str(j) for j in range(k)],
               "target": np.eye(k)[gold].tolist(), "tokens": int(rng.integers(50, 500)),
               "probs": [1 / k] * k}  # fmt: skip
        args = (rng, z, t, kind, views, pos_bias, yes_bias)
        logits, orders = _views(*args, noise)
        rows.append(row | {"logits": logits, "orders": orders})
        if checkpoint_noise is not None:
            logits, orders = _views(*args, checkpoint_noise)
            rows2.append(row | {"logits": logits, "orders": orders})
    return rows, rows2


def _views(rng, z, t, kind, views, pos_bias, yes_bias, noise_sd):
    """``views`` shuffled orderings of the model's logits (label order) and the orders."""
    out, orders = [], []
    for _ in range(views):
        order = rng.permutation(len(z))
        m = z / t + rng.normal(0, noise_sd, len(z))
        m[order[0]] += pos_bias
        if kind == "noul":
            m[1] += yes_bias
        out.append(m.round(4).tolist())
        orders.append(order.tolist())
    return out, orders


def _load(tmp_path, name, **kw):
    out = {}
    for split, n, seed in (("calibration", 2500, 10), ("validation", 1500, 11)):
        rows, rows2 = _scenario(n, seed, **kw)
        dump.write(tmp_path / name / f"{split}.jsonl.gz", rows)
        out[split] = cs.load(tmp_path / name, split)
        if rows2:
            dump.write(tmp_path / f"{name}-2" / f"{split}.jsonl.gz", rows2)
            out[split] = cs.align(out[split], cs.load(tmp_path / f"{name}-2", split))
    return out["calibration"], out["validation"]


def _nll(method, calib, val):
    return cs.sum_nll(val, cs.METHODS[method](calib)(val)) / len(val["type"])


PER_TYPE = "per type (old artifact)"


def test_feature_temperature_learns_t_of_n_and_reduces_to_per_type(tmp_path):
    calib, val = _load(tmp_path, "tn", t_of_n=True)
    assert _nll("feature T (thermometer-lite)", calib, val) < _nll(PER_TYPE, calib, val) - 0.01
    fit = cal.fit_features(calib["logits"], calib["mask"], calib["target"], calib["type"],
                           cs.feats(calib))  # fmt: skip
    assert fit["lambda"] != "inf"
    log_t = cal.feature_log_t(fit, calib["type"], cs.feats(calib))
    # the model reports z / t, t = (n / 5)^0.8: the right calibration T is 1 / t
    choice = calib["type"] == "choice"
    true = -0.8 * (np.log(calib["n"][choice]) - np.log(5))
    assert np.corrcoef(log_t[choice], true)[0, 1] > 0.8
    # no feature effect: shrunk to (about) per type
    calib, val = _load(tmp_path, "flat")
    fit = cal.fit_features(calib["logits"], calib["mask"], calib["target"], calib["type"],
                           cs.feats(calib))  # fmt: skip
    assert np.abs(fit["w"]).max() < 0.1
    assert _nll("feature T (thermometer-lite)", calib, val) < _nll(PER_TYPE, calib, val) + 0.003
    w0 = cal.fit_features(calib["logits"], calib["mask"], calib["target"], calib["type"],
                          cs.feats(calib), lambdas=(float("inf"),))  # fmt: skip
    assert w0["w"] == [0.0] * len(cal.FEATURES)


def test_feature_temperature_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2, (6, 2, 4))
    mask = np.ones((6, 2, 4), bool)
    mask[0, 1] = False
    mask[1, :, 3] = False
    target = np.eye(4)[[0, 1, 2, 0, 3, 1]]
    target[1] = [0.5, 0.5, 0, 0]
    u = rng.normal(0, 0.3, 6)
    _, g = cal.nll_and_grad(logits, mask, target, u)
    h = 1e-5
    num = (cal.nll_and_grad(logits, mask, target, u + h)[0]
           - cal.nll_and_grad(logits, mask, target, u - h)[0]) / (2 * h)  # fmt: skip
    assert np.allclose(g, num, atol=1e-5)


def test_position_bias_is_found_by_vector_scaling_and_batch_calibration(tmp_path):
    calib, val = _load(tmp_path, "pos", pos_bias=1.5)
    per_type = _nll(PER_TYPE, calib, val)
    vs = cs.METHODS["vector scaling: T + position bias, choice"](calib)
    assert cs.sum_nll(val, vs(val)) / len(val["type"]) < per_type - 0.02
    # the model adds +1.5 to the first displayed option: positions 1.. get +1.5 against it
    assert np.allclose(vs.params["choice"]["bias"][:5], 1.5, atol=0.3), vs.params["choice"]
    for m in ("batch calibration (eval batch), choice+noul",
              "batch calibration (calibration split), choice+noul"):  # fmt: skip
        assert _nll(m, calib, val) < per_type - 0.01


def test_platt_finds_the_yes_bias_and_reports_the_band(tmp_path):
    calib, val = _load(tmp_path, "yes", yes_bias=1.0)
    platt = cs.METHODS["per type T + Platt with yes-bias (noul)"](calib)
    # the same items without the bias: the yes-bias moves c by -a * 1.0, a unchanged
    unbiased = cs.METHODS["per type T + Platt with yes-bias (noul)"](
        _load(tmp_path, "yes0")[0]).params  # fmt: skip
    shift = platt.params["yes_bias_c"] - unbiased["yes_bias_c"]
    assert abs(shift + platt.params["a"]) < 0.05 and abs(platt.params["a"] - unbiased["a"]) < 0.01
    assert cs.sum_nll(val, platt(val)) < cs.sum_nll(val, cs.METHODS[PER_TYPE](calib)(val))
    band = cs.metrics(val, platt(val))["noul_band"]
    assert 0 < band["abstain"] < 1 and band["v15_accuracy"] <= band["committed_accuracy"]


def test_order_and_checkpoint_ensembles_beat_single_members(tmp_path):
    calib, val = _load(tmp_path, "ens", views=2, noise=1.0, checkpoint_noise=1.0)
    first = _nll("orderings: first only, per type T", calib, val)
    assert _nll("orderings: mean of probabilities, per type T", calib, val) < first - 0.01
    assert _nll("orderings: log-space mean, per type T", calib, val) < first
    both = _nll("checkpoints 1 + 2, mean of calibrated probabilities", calib, val)
    assert both < _nll(PER_TYPE, calib, val) and both < _nll("checkpoint 2 alone, per type T",
                                                             calib, val)  # fmt: skip
    assert (
        cs.orderings("mean")(
            cs.take(calib, np.ones(len(calib["type"]), bool))
            | {"logits": calib["logits"][:, :1], "mask": calib["mask"][:, :1]}
        )
        is None
    )
    assert cs.checkpoint_ensemble({k: v for k, v in calib.items() if k != "second"}) is None


def test_flags_catch_zero_probabilities_and_moved_argmax(tmp_path):
    _, val = _load(tmp_path, "flags")
    base = cs.view_probs(val["logits"], val["mask"])
    probs = base.copy()
    gold = val["target"][0].argmax()
    probs[0] = 0.0
    probs[0, (gold + 1) % val["n"][0]] = 1.0
    f = cs.flags(val, probs, base)
    assert f["p0_on_gold"] == 1 and f["p1"] >= 1 and f["argmax_changed"] > 0
    assert cs.flags(val, base, base) == {"p0_on_gold": 0, "p1": int(np.sum(base.max(-1) > 1 - 1e-9)),
                                         "argmax_changed": 0.0, "accuracy_delta": 0.0}  # fmt: skip
