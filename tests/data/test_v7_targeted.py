"""The mix-v7 targeted sources (tjev.data.sources.targeted_v7) on hand-made rows: schema,
recomputed labels, exact distractors, licence fields, no answer leakage. No download."""

import hashlib
import json

import numpy as np
import pytest

from tjev.data.item import LICENSE_USES, parse_item
from tjev.data.licences import licence_of
from tjev.data.sources import targeted_v7 as t
from tjev.data.sources.targeted import _hash01


def _rng(i=0):
    return np.random.default_rng(i)


def _parsed(rows, lang="en"):
    return [parse_item({"id": "t", "source": "t", "family": "t", "lang": lang, **r}) for r in rows]


def _gold_text(item):
    return item.question.descriptions[item.gold]


# --- registry and licences ---------------------------------------------------------------


def test_registry_blocks_licences_and_uses():
    assert "fiscal_fr" not in t.TARGETED_V7_SOURCES  # no version histories: not built
    for name, s in t.TARGETED_V7_SOURCES.items():
        assert s.block == "targeted" and s.license_use in LICENSE_USES
        assert s.commercial_ok == (s.license_use == "commercial")
        card, redistributable, note = licence_of(name)
        assert card == s.license == t.LICENCES[name][0] and note
        assert redistributable in ("yes", "share-alike")
        assert s.eval_split or s.group or s.train_only, name
    assert t.TARGETED_V7_SOURCES["cold_law_fr_x"].train_only
    assert t.TARGETED_V7_SOURCES["argimi_fr_x"].train_only


def test_slices_are_disjoint_from_mix_v6_samples():
    assert t.COLD_X_SLICE[0] == 0.035 and t.ARGIMI_X_SLICE[0] == 0.12
    held = [k for k in (f"titre {i}" for i in range(2000)) if t.v6_held_group(k)]
    assert 150 < len(held) < 250  # ~10%
    k = held[0]
    assert int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % 1000 < 100


# --- TempReason L2 / L3 ------------------------------------------------------------------

FACTS = (
    "C. N. Brunsdale holds the position of Governor of North Dakota from Jan, 1951 to Jan, 1957."
    " \n C. N. Brunsdale holds the position of member of the State Senate from Jan, 1940 to "
    "Jan, 1951. \n C. N. Brunsdale holds the position of United States senator from Nov, 1959 "
    "to Aug, 1960."
)
CANDS = ["member of the State Senate", "Governor of North Dakota", "United States senator"]


@pytest.mark.parametrize(
    ("question", "gold"),
    [
        ("Which position did C. N. Brunsdale hold in May, 1946?", "member of the State Senate"),
        ("Which position did C. N. Brunsdale hold in Jul, 1960?", "United States senator"),
        ("Which position did C. N. Brunsdale hold in Jan, 1951?", None),  # shared boundary
        ("Which position did C. N. Brunsdale hold in Jan, 1958?", None),  # no fact
        ("Which position did C. N. Brunsdale hold before Governor of North Dakota?",
         "member of the State Senate"),
        ("Which position did C. N. Brunsdale hold after Governor of North Dakota?",
         "United States senator"),
        ("Which position did C. N. Brunsdale hold before member of the State Senate?", None),
    ],
)  # fmt: skip
def test_tempreason_gold_is_recomputed_from_the_facts(question, gold):
    assert t.tempreason_l23_gold(question, FACTS, CANDS) == gold


def test_tempreason_rows_keep_only_agreeing_labels_and_use_fact_distractors():
    base = {"id": "L2_1", "level": "L2", "facts": FACTS,
            "question": "Which position did C. N. Brunsdale hold in May, 1946?"}  # fmt: skip
    good = {**base, "answer": CANDS[0], "negatives": CANDS[1:]}
    wrong = {**base, "answer": CANDS[1], "negatives": [CANDS[0], CANDS[2]]}
    assert t.tempreason_l23(wrong, _rng(), {}) == []
    kinds = set()
    for i in range(40):
        (row,) = t.tempreason_l23(good, _rng(i), {})
        (item,) = _parsed([row])
        kinds.add(item.question.type)
        assert item.state.startswith("Facts:") and "1946" not in item.state
        if item.question.type == "choice":
            assert _gold_text(item) == CANDS[0]
            assert sorted(item.question.descriptions) == sorted(CANDS)
        else:
            yes = item.question.labels[item.gold] == "yes"
            assert yes == (f'"{CANDS[0]}"' in item.question.instructions)
    assert kinds == {"choice", "noul"}


# --- ArGiMi expansions and the Conseil d'État ---------------------------------------------

CETAT = (
    "Vu la requête, enregistrée le 14 mars 2003, présentée par le PREFET DE POLICE ; "
    + "le préfet demande l'annulation du jugement ; " * 20
    + "<br>Considérant que l'arrêté a porté une atteinte disproportionnée ; que, par suite, le "
    "préfet n'est pas fondé à soutenir que c'est à tort que le jugement attaqué a annulé cet "
    "arrêté ;<br><br>D E C I D E  :<br>--------------<br>Article 1er : La requête du PREFET DE "
    "POLICE est rejetée.<br>Article 2 : L'Etat versera la somme de 500 euros en application de "
    "l'article L. 761-1 du code de justice administrative.<br>Article 3 : La présente décision "
    "sera notifiée au PREFET DE POLICE.<br>"
)


@pytest.mark.parametrize(
    ("operative", "outcome"),
    [
        ("Article 1er : La requête est rejetée. Article 2 : notifiée à M. X.", "rejet"),
        ("Article 1er : Le jugement du tribunal administratif est annulé.", "annulation"),
        (("Article 1er : Le jugement est annulé. Article 2 : Le surplus des conclusions est "
         "rejeté."), "annulation_partielle"),
        ("Article 1er : Il n'y a pas lieu de statuer sur la requête.", "non_lieu"),
        (("Article 1er : La requête est rejetée. Article 2 : Les conclusions au titre de "
         "l'article L. 761-1 sont rejetées."), "rejet"),
        ("Article 1er : L'imposition est réduite. Article 2 : rejeté.", None),
        ("Article 1er : Une expertise est ordonnée.", None),
    ],
)  # fmt: skip
def test_cetat_outcome_from_the_operative_part(operative, outcome):
    assert t.cetat_outcome(operative) == outcome


def test_cetat_state_has_no_operative_part():
    reasoning, operative = t.cetat_split(CETAT)
    assert t.cetat_outcome(operative) == "rejet"
    state = t.cetat_state(reasoning)
    assert state is not None
    for leak in ("D E C I D E", "Article 1er", "est rejetée", "versera"):
        assert leak not in state
    (row,) = t.cetat({"title": "Conseil d'Etat, du 9 mai 2005, inédit au recueil Lebon",
                      "text": state, "outcome": "rejet"}, _rng(), {})  # fmt: skip
    (item,) = _parsed([row], "fr")
    assert item.question.labels[item.gold] == "rejet" and "inédit" not in item.state
    assert t.cetat_split("pas de dispositif") is None


def test_argimi_x_reuses_the_v6_adapter():
    src = t.TARGETED_V7_SOURCES["argimi_fr_x"]
    assert src.adapter is t.argimi and t.ARGIMI_X_SLICE[1] > t.ARGIMI_X_SLICE[0]
    assert t.TARGETED_V7_SOURCES["cold_law_fr_x"].adapter is t.cold_law
    assert not any(t.ARGIMI_X_SLICE[0] <= _hash01(f"argimi:{i}") < t.ARGIMI_X_SLICE[1]
                   for i in ("x",) if _hash01(f"argimi:{i}") < 0.12)  # fmt: skip


# --- TRAM ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("What is 23:48 - 01:31?", "22:17"),
        ("What is 22:50 + 03:15?", "02:05"),
        ("Which year comes 1 year after 1286?", "1287"),
        ("Which month comes 3 months after April?", "July"),
        ("Which month comes 2 months before January?", "November"),
        ("What will be the time 8 years and 8 months after February 1798?", "October 1806"),
        ("What will be the time 5 months before March 1800?", "October 1799"),
        ("A plane flies at 300 km/h. How long ...?", None),
    ],
)
def test_tram_recompute(question, answer):
    assert t.tram_recompute(question) == answer


def test_tram_rows_drop_wrong_answers_and_keep_the_question_in_the_state():
    row = {"task": "arithmetic", "category": "Year Shift",
           "question": "Which year comes 1 year after 1286?",
           "options": ["1304", "1239", "1287", "1298"], "answer": 2}  # fmt: skip
    (r,) = t.tram(row, _rng(), {})
    (item,) = _parsed([r])
    assert _gold_text(item) == "1287" and item.state == row["question"]
    assert t.tram({**row, "answer": 0}, _rng(), {}) == []
    assert "Commonsense" not in t.TRAM_KEEP["duration"]
    assert all("Reading Comprehension" not in (k or ()) for k in t.TRAM_KEEP.values())
    assert "relation" not in t.TRAM_KEEP and "nli_mcq" not in t.TRAM_KEEP


# --- numeric distractors, DROP, TAT-QA, FinQA -----------------------------------------------


@pytest.mark.parametrize(
    ("text", "value", "decimals"),
    [("$1,910", 1910.0, 0), ("(12.5)", -12.5, 1), ("87.04%", 87.04, 2), ("abc", None, 0)],
)
def test_parse_number(text, value, decimals):
    assert t.parse_number(text) == value
    assert t.decimals_of(text) == decimals


@pytest.mark.parametrize("x", [3.0, 17.0, 250.0, 87.04, -12.5, 0.0])
def test_numeric_distractors_are_distinct_and_exactly_wrong(x):
    dec = 2 if x == 87.04 else 1 if x == -12.5 else 0
    for i in range(20):
        out = t.numeric_distractors(x, dec, _rng(i), k=4, nonnegative=x >= 0)
        gold = t.format_number(x, dec)
        assert out and len(set(out)) == len(out) and gold not in out
        for d in out:
            v = float(d.replace(",", ""))
            assert v != x
            if x >= 0:
                assert v >= 0


def test_drop_number_and_date_items():
    row = {"kind": "number", "number": "3", "date": "{}", "passage": "The Lions kicked.",
           "question": "How many points?"}  # fmt: skip
    seen = set()
    for i in range(30):
        (r,) = t.drop_num(row, _rng(i), {})
        (item,) = _parsed([r])
        seen.add(item.question.type)
        if item.question.type == "choice":
            assert _gold_text(item) == "3"
        else:
            assert (
                item.question.labels[item.gold] == "yes"
            ) == item.question.instructions.endswith("answer 3?")
    assert seen == {"choice", "noul"}
    date = {"kind": "date", "number": "", "passage": "p", "question": "When?",
            "date": json.dumps({"day": "4", "month": "June", "year": "2000"})}  # fmt: skip
    gold = "June 4, 2000"
    others = t.date_distractors(4, 6, 2000, _rng(), k=6)
    assert gold not in others and len(set(others)) == len(others)
    assert t.date_text(None, None, 1995) == "1995"
    (r,) = t.drop_num(date, _rng(1), {})
    assert gold in json.dumps(r, ensure_ascii=False)
    assert t._drop_answer({"number": "", "date": {"day": "", "month": "", "year": ""}}) is None


def test_tatqa_scale_and_count():
    row = {"answer": "8325", "kind": "arithmetic", "scale": "thousand",
           "context": "Table:\nA | 1", "question": "What is the average?"}  # fmt: skip
    for i in range(10):
        (r,) = t.tatqa(row, _rng(i), {})
        (item,) = _parsed([r])
        texts = item.question.descriptions if item.question.type == "choice" else ()
        if texts:
            assert _gold_text(item) == "8325 thousand"
            assert all(d.endswith(" thousand") for d in texts)
    pct = {**row, "answer": "87.04", "scale": "percent"}
    choices = [r for i in range(20) for r in t.tatqa(pct, _rng(i), {})
               if r["question"]["type"] == "choice"]  # fmt: skip
    assert choices and all("87.04%" in r["question"]["criteria"].values() for r in choices)
    assert t.tatqa({**row, "scale": "weird"}, _rng(), {}) == []


@pytest.mark.parametrize(
    ("answer", "exe", "ok"),
    [("14.1%", 0.14136, True), ("14.1%", 14.136, True), ("3.2", 3.2, True), ("94", 0.5, False),
     ("2.5", "n/a", False)],
)  # fmt: skip
def test_finqa_answer_must_agree_with_the_program(answer, exe, ok):
    assert t.finqa_consistent(answer, exe) is ok


def test_finqa_yes_no_and_numeric_items():
    (r,) = t.finqa({"kind": "yesno", "answer": "yes", "context": "c", "question": "Did it rise?"},
                   _rng(), {})  # fmt: skip
    (item,) = _parsed([r])
    assert item.question.type == "noul" and item.question.labels[item.gold] == "yes"
    rows = [
        r
        for i in range(20)
        for r in t.finqa(
            {"kind": "number", "answer": "14.1%", "context": "c", "question": "q"}, _rng(i), {}
        )
    ]
    for r in rows:
        (item,) = _parsed([r])
        if item.question.type == "choice":
            assert _gold_text(item) == "14.1%"
            assert all(d.endswith("%") for d in item.question.descriptions)


# --- temporal NLI, forecasting ----------------------------------------------------------------


def test_temporal_nli_item():
    (r,) = t.temporal_nli({"premise": "They got married on Saturday.",
                           "hypothesis": "They got married before Friday.",
                           "label": "contradiction"}, _rng(), {})  # fmt: skip
    (item,) = _parsed([r])
    assert item.question.labels[item.gold] == "contradiction"
    assert "contradiction" not in item.state


def test_forecasting_resolution_target_crowd_teacher_and_leak_filter():
    row = {"question": "Will X happen by 2016?", "background": "Context.", "criteria": "Rules.",
           "begin": "2015-10-02", "close": "2015-12-15", "resolve": "2016-02-01",
           "source": "metaculus", "resolution": 0, "crowd": 0.58}  # fmt: skip
    (r,) = t.forecasting(row, _rng(), {})
    (item,) = _parsed([r])
    assert item.question.labels[item.gold] == "no"
    assert item.teacher == pytest.approx((0.42, 0.58))
    assert "2016-02-01" not in item.state  # the resolution date is not shown
    (r,) = t.forecasting({**row, "crowd": -1.0}, _rng(), {})
    assert "teacher" not in r
    assert t._FORECAST_LEAK.search("UPDATE: this question resolved yes")
    assert t._FORECAST_LEAK.search("This question has resolved")
    assert not t._FORECAST_LEAK.search("The market will be updated weekly by the agency")
