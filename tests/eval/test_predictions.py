"""Per-item prediction dumps round-trip, and served items get their guessed language's
temperature."""

import numpy as np

from tjev.data.item import parse_item
from tjev.data.lang import guess_lang, item_text
from tjev.eval import predictions as dump
from tjev.eval.metrics import view_probs
from tjev.serve.backends import temperature


def test_dump_round_trips_to_the_same_arrays(tmp_path):
    logits = np.array([[[2.0, 0.0, -1.0], [1.5, 0.5, 0.0]], [[0.3, -0.3, 0.0], [0, 0, 0]]])
    mask = np.array([[[1, 1, 1], [1, 1, 1]], [[1, 1, 0], [0, 0, 0]]], bool)
    target = np.array([[1.0, 0, 0], [0.25, 0.75, 0]])
    arrays = {"logits": logits, "mask": mask, "target": target, "type": np.array(["choice", "noul"]),
              "lang": ["en", "fr"], "lang_guess": [None, "fr"], "source": ["a", "b"],
              "family": ["f", "g"], "id": ["x", "y"], "labels": [("a", "b", "c"), ("no", "yes")]}  # fmt: skip
    probs = view_probs(logits, mask)
    rows = dump.records(arrays, probs, [[[0, 1, 2], [1, 2, 0]], [[1, 0]]])
    back = dump.arrays(dump.read(dump.write(tmp_path / "p.jsonl.gz", rows)))
    assert np.allclose(back["logits"][back["mask"]], logits[mask], atol=1e-4)
    assert (back["mask"] == mask).all() and np.allclose(back["target"], target)
    assert back["lang_guess"] == [None, "fr"] and back["orders"][0][1] == [1, 2, 0]
    assert list(back["n"]) == [3, 2]
    assert np.allclose(view_probs(back["logits"], back["mask"]), probs, atol=1e-4)


FR = ("Le client a demandé un remboursement pour sa commande, mais il ne nous a pas "
      "envoyé le reçu dans les délais et nous ne pouvons pas le traiter.")  # fmt: skip
EN = ("The customer asked for a refund of the order, but they did not send us the receipt "
      "in time and we cannot process it for them.")  # fmt: skip


def _noul(state: str):
    return parse_item({"state": state, "expected": "yes", "question": {
        "type": "noul", "instructions": "?", "criteria": {"false": "no", "true": "yes"}}})  # fmt: skip


def test_served_items_get_their_guessed_language_temperature():
    assert guess_lang(item_text(_noul(FR))) == "fr" and guess_lang(item_text(_noul(EN))) == "en"
    temps = {"noul": 1.3, "noul/fr": 1.8, "noul/en": 1.2}
    assert temperature(temps, _noul(FR)) == 1.8
    assert temperature(temps, _noul(EN)) == 1.2
    assert temperature(temps, _noul("x = 3")) == 1.3  # unsure: the type's
    assert temperature({"noul": 1.3}, _noul(FR)) == 1.3  # a per-type artifact
    assert temperature({}, _noul(FR)) == 1.0
