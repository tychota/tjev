"""Temperature scaling: recovers a known temperature, never hurts, and binds its artifact."""

import numpy as np
import pytest

from tjev.eval.calibrate import (
    artifact,
    fit_grouped,
    fit_per_type,
    fit_temperature,
    lookup,
    validate,
)


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


# ------------------------------------------------- per (type, language), shrunk to the type


def _cells(spec, seed=0, k=4):
    """Concatenated _sampled cells: spec = [(type, lang, n, temperature), ...]."""
    parts = [_sampled(n=n, k=k, temperature=t, seed=seed + i) for i, (_, _, n, t) in
             enumerate(spec)]  # fmt: skip
    types = [kind for kind, _, n, _ in spec for _ in range(n)]
    langs = [lang for _, lang, n, _ in spec for _ in range(n)]
    logits, mask, target = (np.concatenate([p[j] for p in parts]) for j in range(3))
    return logits, mask, target, types, langs


def test_type_lang_recovers_a_language_specific_temperature():
    # choice: fr is over-confident (true T = 2), en is not (T = 1); plenty of slots each
    data = _cells([("choice", "en", 3000, 1.0), ("choice", "fr", 2000, 2.0),
                   ("noul", "en", 1500, 1.3), ("noul", "fr", 1500, 1.3)], k=2)  # fmt: skip
    fit = fit_grouped(*data, grouping="type_lang")
    temps = fit["temperatures"]
    assert fit["grouping"] == "type_lang" and fit["tau"] != "inf"
    assert abs(temps["choice/fr"] - 2.0) < 0.3 and abs(temps["choice/en"] - 1.0) < 0.15
    assert 1.0 < temps["choice"] < 2.0  # the type's (the fallback) lies in between
    # the CV choice beats per type (tau = inf) on the held-out folds
    assert fit["cv_nll"][str(fit["tau"])] < fit["cv_nll"]["inf"]
    assert lookup(temps, "choice", "fr") == temps["choice/fr"]
    assert lookup(temps, "choice", None) == lookup(temps, "choice", "de") == temps["choice"]
    assert lookup(temps, "unknown") == 1.0


def test_a_tiny_cell_is_shrunk_to_its_type():
    # fr score: 60 slots drawn at the type's own temperature; unshrunk, its own fit is noisy
    data = _cells([("score", "en", 3000, 1.5), ("score", "fr", 60, 1.5),
                   ("choice", "en", 1000, 1.0)], seed=3)  # fmt: skip
    fit = fit_grouped(*data, grouping="type_lang")
    shrunk, own = (
        fit["temperatures"]["score/fr"],
        fit_grouped(*data, grouping="type_lang", taus=(0.0,)),
    )
    assert (
        abs(np.log(shrunk / fit["temperatures"]["score"]))
        <= abs(np.log(own["temperatures"]["score/fr"] / own["temperatures"]["score"])) + 1e-9
    )
    assert abs(np.log(shrunk / fit["temperatures"]["score"])) < 0.1


def test_no_language_effect_gives_the_per_type_temperatures():
    data = _cells([("noul", "en", 2000, 1.4), ("noul", "fr", 800, 1.4)], k=2, seed=7)
    fit = fit_grouped(*data, grouping="type_lang")
    per_type = fit_per_type(*data[:4])["temperatures"]
    own = fit_grouped(*data, grouping="type_lang", taus=(0.0,))["temperatures"][
        "noul/fr"
    ]  # the cell alone
    shrunk, prior = fit["temperatures"]["noul/fr"], per_type["noul"]
    assert fit["temperatures"]["noul"] == prior
    assert min(own, prior) - 1e-6 <= shrunk <= max(own, prior) + 1e-6  # between the two
    assert fit["cv_nll"][str(fit["tau"])] <= fit["cv_nll"]["inf"]  # never worse than per type


def test_grouping_type_is_the_old_fit_and_old_artifacts_read_as_per_type():
    data = _cells([("noul", "en", 300, 1.2), ("noul", "fr", 300, 2.0)], k=2)
    fit = fit_grouped(*data, grouping="type")
    assert set(fit["temperatures"]) == {"noul"} and fit["grouping"] == "type"
    with pytest.raises(ValueError, match="languages"):
        fit_grouped(*data[:4], grouping="type_lang")
    with pytest.raises(ValueError, match="grouping"):
        fit_grouped(*data, grouping="source")
    art = artifact(fit, checkpoint="c", data="d", template=1, backend="jax")
    assert art["grouping"] == "type"
    # an artifact written before groupings existed (no "grouping" key) is per type
    from tjev.utils import fingerprint

    old = {k: v for k, v in art.items() if k not in ("identity", "grouping")}
    old["identity"] = fingerprint(old)
    assert validate(old, checkpoint="c", template=1, backend="jax") == fit["temperatures"]
    bad = {**old, "grouping": "source"}
    bad["identity"] = fingerprint({k: v for k, v in bad.items() if k != "identity"})
    with pytest.raises(ValueError, match="grouping"):
        validate(bad, checkpoint="c", template=1, backend="jax")


def test_grouped_report_applies_cell_temperatures_by_language():
    from tjev.eval.metrics import grouped_report

    logits, mask, target, types, langs = _cells(
        [("choice", "en", 50, 1.0), ("choice", "fr", 50, 1.0)]
    )
    temps = {"choice": 1.0, "choice/fr": 3.0}
    rep = grouped_report(logits, mask, target, {"type": types, "lang": langs}, temps)
    plain = grouped_report(logits, mask, target, {"type": types, "lang": langs}, {"choice": 1.0})
    assert rep["lang"]["en"] == plain["lang"]["en"]
    assert rep["lang"]["fr"]["confidence"] < plain["lang"]["fr"]["confidence"]


def test_a_small_all_correct_cell_does_not_run_to_the_temperature_bound():
    # 30 fr score slots that the model always gets right with a clear margin: alone, NLL
    # keeps falling as T -> 0, and every CV fold agrees, so only a minimum size protects it
    rng = np.random.default_rng(5)
    big = _cells([("score", "en", 2000, 1.3)], seed=5)
    logits = rng.normal(0, 1, (30, 4))
    logits[:, 0] += 4.0
    target = np.eye(4)[np.zeros(30, int)]
    data = (np.concatenate([big[0], logits]), np.concatenate([big[1], np.ones((30, 4), bool)]),
            np.concatenate([big[2], target]), big[3] + ["score"] * 30, big[4] + ["fr"] * 30)  # fmt: skip
    fit = fit_grouped(*data, grouping="type_lang")
    assert "score/fr" not in fit["temperatures"]
    assert fit["detail"]["score/fr"]["fallback"] == "type"
    assert lookup(fit["temperatures"], "score", "fr") == fit["temperatures"]["score"]
