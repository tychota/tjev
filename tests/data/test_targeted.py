"""The mix-v6 targeted sources (adapters on hand-made rows: schema, label validity, soft
targets, licences, no answer leakage) and the mix-v6 hygiene checks (held-out passages,
JevBench 8-grams). No download."""

import json
import re
from datetime import date

import numpy as np
import pytest

from tjev.data.hygiene import HeldIndex, clean
from tjev.data.item import parse_item
from tjev.data.jevbench import JevBenchGrams
from tjev.data.licences import OPEN, licence_of
from tjev.data.mix import BLOCKS, FR_TARGET, SEPARATE_HELD, block_of_source
from tjev.data.sources import ALL_SOURCES, TARGETED_SOURCES, V7_SOURCES
from tjev.data.sources import targeted as t


def _rng(i=0):
    return np.random.default_rng(i)


def _parsed(rows, lang="en"):
    return [parse_item({"id": "t", "source": "t", "family": "t", "lang": lang, **r}) for r in rows]


def _gold(item):
    return item.question.labels[item.gold]


# --- registry and licences ---------------------------------------------------------------


def test_targeted_sources_are_registered_in_their_own_block_with_a_card_licence():
    assert set(TARGETED_SOURCES) <= set(ALL_SOURCES)
    v6 = [n for n in ALL_SOURCES if n not in V7_SOURCES]  # mix-v7 sources come after
    assert v6[-len(TARGETED_SOURCES) :] == list(TARGETED_SOURCES)  # built last
    for name in TARGETED_SOURCES:
        assert block_of_source(name) == "targeted"
        card, redistributable, _ = licence_of(name)
        assert card != "not recorded" and redistributable in OPEN, name  # no NC / unknown
    assert "targeted" in BLOCKS and "targeted" in FR_TARGET and "targeted" in SEPARATE_HELD
    assert abs(sum(BLOCKS.values()) - 1.0) < 1e-9
    assert BLOCKS["analysis"] + BLOCKS["writing"] <= 0.20


# --- eikos ---------------------------------------------------------------------------------


def _eikos_row(**kw):
    row = {
        "id": "e1",
        "family": "temporal_numeric",
        "label_source": "program",
        "state": "Ticket opened Tue 09:15, resolved Thu 09:55; SLA 16 business hours.",
        "question_type": "noul",
        "instructions": "Did this ticket breach the SLA?",
        "options": json.dumps(
            [
                {"label": "yes", "description": "After the deadline."},
                {"label": "no", "description": "At or before the deadline."},
            ]
        ),
        "expected": "yes",
        "target_probs": json.dumps([0.95, 0.05]),
    }
    return {**row, **kw}


def test_eikos_program_rows_are_exact_and_noul_options_become_no_yes():
    (item,) = _parsed(t.eikos(_eikos_row(), _rng(), {}))
    assert item.question.labels == ("no", "yes") and item.target == (0.0, 1.0)
    assert item.question.descriptions == ("At or before the deadline.", "After the deadline.")


def test_eikos_teacher_and_exact_probability_rows_keep_their_distribution():
    for source in ("teacher", "exact_probability"):
        (item,) = _parsed(t.eikos(_eikos_row(label_source=source, expected="no",
                                             target_probs="[0.3, 0.7]"), _rng(), {}))  # fmt: skip
        assert np.allclose(item.target, (0.7, 0.3)) and _gold(item) == "no"


def test_eikos_choice_and_score_rows():
    opts = [{"label": f"t_{i}", "description": f"time {i}"} for i in range(4)]
    choice = _eikos_row(question_type="choice", options=json.dumps(opts), expected="t_2",
                        target_probs="[0.0167, 0.0167, 0.95, 0.0166]")  # fmt: skip
    (c,) = _parsed(t.eikos(choice, _rng(), {}))
    assert _gold(c) == "t_2" and c.target[2] == 1.0
    levels = [{"label": n, "description": d} for n, d in [("low", "Low"), ("high", "High")]]
    score = _eikos_row(question_type="score", options=json.dumps(levels), expected="high",
                       target_probs="[0.05, 0.95]")  # fmt: skip
    (s,) = _parsed(t.eikos(score, _rng(), {}))
    assert s.question.type == "score" and s.question.labels == ("0", "1") and s.gold == 1
    assert t.eikos(_eikos_row(expected="maybe"), _rng(), {}) == []  # an unknown gold: dropped


def _dossier(n=6, size=1500):
    return "\n\n".join(
        f"--- Document {i} (log) ---\n" + f"doc{i} " * (size // 5) for i in range(1, n + 1)
    )


def test_eikos_dossier_keeps_the_asked_document_renumbered_within_budget():
    state = _dossier()
    for seed in range(20):
        cut = t.eikos_dossier(state, "Regarding Document 3: is it late?", 3, _rng(seed), 5000)
        assert cut is not None
        text, question = cut
        numbers = [int(n) for n in re.findall(r"^--- Document (\d+) ", text, re.MULTILINE)]
        assert numbers == list(range(1, len(numbers) + 1))
        k = int(re.match(r"Regarding Document (\d+):", question).group(1))
        body = text.split(f"--- Document {k} (log) ---\n")[1].split("\n\n--- Document")[0]
        assert body.startswith("doc3 ")  # the renumbered question points at the same document
        assert len(text) < 5000 + 40 * len(numbers)
    assert t.eikos_dossier(state, "Regarding Document 9: x", 9, _rng(), 5000) is None
    assert t.eikos_dossier(state, "Regarding Document 3: x", 3, _rng(), 100) is None


# --- procedural ------------------------------------------------------------------------------


def test_procedural_one_item_per_question_exact_and_posterior_targets():
    questions = {
        "allowed": {"type": "noul", "instructions": "Allowed?"},
        "posterior": {"type": "noul", "instructions": "How likely is the incident real?"},
        "policy": {"type": "choice", "instructions": "Which policy?",
                   "criteria": {"P1": "P1", "P2": "P2"}},
        "risk": {"type": "score", "instructions": "Risk?", "criteria": ["0", "1", "Level 2"]},
    }  # fmt: skip
    answers = {
        "allowed": {"type": "noul", "noul": 0.0},
        "posterior": {"type": "noul", "noul": 0.287129},
        "policy": {"type": "choice", "choice": "P2", "probabilities": {"P1": 0.3, "P2": 0.7}},
        "risk": {"type": "score", "score": 2.0, "probabilities": {"0": 0, "1": 0, "2": 1}},
    }
    row = {"state": json.dumps({"request": {"role": "x"}}), "questions": json.dumps(questions),
           "answers": json.dumps(answers), "task": "policy_applicability", "level": "0"}  # fmt: skip
    items = _parsed(t.procedural(row, _rng(), {}))
    by = {i.question.instructions: i for i in items}
    assert by["Allowed?"].target == (1.0, 0.0)
    assert np.allclose(by["How likely is the incident real?"].target, (0.712871, 0.287129))
    assert np.allclose(by["Which policy?"].target, (0.3, 0.7))
    assert by["Which policy?"].question.descriptions == ("", "")  # label-only options
    assert by["Risk?"].gold == 2 and by["Risk?"].question.descriptions == ("", "", "Level 2")
    assert items[0].state.startswith("{\n")  # JSON states are rendered like the hard tier
    assert all(i.family == "procedural_policy_applicability" for i in items)


# --- TempReason ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "parsed"),
    [
        ("What is the time 1 year and 7 month after Mar, 1873", (1873 * 12 + 2, 19, 1)),
        ("What is the time 8 year after Oct, 1092", (1092 * 12 + 9, 96, 1)),
        ("What is the time 6 month before Jan, 1509", (1509 * 12, -6, -1)),
        ("What is the time sometime after Mar, 1873", None),
    ],
)
def test_tempreason_parse(question, parsed):
    assert t.tempreason_parse(question) == parsed


def test_tempreason_labels_are_recomputed_and_options_are_near_misses():
    row = {"question": "What is the time 1 year and 7 month after Mar, 1873", "answer": "Oct, 1874"}
    langs, kinds = set(), set()
    for seed in range(200):
        (item,) = _parsed(t.tempreason(row, _rng(seed), {}), lang="en")
        q = item.question
        kinds.add(q.type)
        langs.add("fr" if "mars 1873" in item.state else "en")
        if q.type == "choice":
            assert _gold(item) == "1874-10" and len(q.labels) >= 4
            assert "October 1874" not in q.instructions and "octobre 1874" not in q.instructions
        else:
            shown = "October 1874" in q.instructions or "octobre 1874" in q.instructions
            assert _gold(item) == ("yes" if shown else "no")
    assert kinds == {"choice", "noul"} and langs == {"en", "fr"}
    assert t.tempreason({**row, "answer": "Nov, 1874"}, _rng(), {}) == []  # upstream disagrees
    near = t.tempreason_candidates(1873 * 12 + 2, 19)
    assert 1873 * 12 + 2 + 19 not in near and 1873 * 12 + 2 - 19 in near


# --- HelpSteer3 ----------------------------------------------------------------------------


def _hs3(scores):
    return {"context": json.dumps([{"role": "user", "content": "Write a haiku."}]),
            "response1": "First answer.", "response2": "Second answer.",
            "scores": scores, "domain": "general"}  # fmt: skip


def test_helpsteer3_soft_votes_and_score_distribution():
    seen = set()
    for seed in range(60):
        (item,) = _parsed(t.helpsteer3(_hs3([-2, -1, 1]), _rng(seed), {}))
        first_is_1 = "Response 1:\nFirst answer." in item.state
        assert 0.999 < sum(item.target) < 1.001 and all(0 <= p <= 1 for p in item.target)
        if item.question.type == "choice":
            want = (2 / 3, 1 / 3, 0) if first_is_1 else (1 / 3, 2 / 3, 0)
            assert np.allclose(item.target, want)
        else:
            dist = dict(zip(item.question.labels, item.target, strict=True))
            if first_is_1:  # scores -2, -1, 1 → levels 1, 2, 4
                assert np.allclose([dist["1"], dist["2"], dist["4"]], [1 / 3] * 3)
            else:
                assert np.allclose([dist["5"], dist["4"], dist["2"]], [1 / 3] * 3)
        seen.add((item.question.type, first_is_1))
    assert len(seen) == 4
    assert t.helpsteer3(_hs3([5]), _rng(), {}) == []


# --- CUAD ------------------------------------------------------------------------------------


def test_cuad_window_contains_the_span_and_fits():
    context = " ".join(f"w{i}" for i in range(5000))
    span = (12000, 12100)
    for seed in range(50):
        s, e = t.cuad_window(context, span, _rng(seed), width=3000)
        assert s <= span[0] and span[1] <= e and e - s <= 3000
        s, e = t.cuad_window(context, None, _rng(seed), width=3000)
        assert 0 <= s < e <= len(context) and e - s <= 3000
    assert t.cuad_window("short", None, _rng(), width=3000) == (0, 5)


def test_cuad_rows_ask_about_the_clause_and_label_presence():
    row = {"title": "ACME_2001-EX-10_DISTRIBUTOR AGREEMENT", "category": "Non-Compete",
           "details": "Is there a restriction on competing?", "excerpt": "The Distributor shall",
           "cut_left": True, "cut_right": False, "present": True}  # fmt: skip
    (item,) = _parsed(t.cuad(row, _rng(), {}))
    assert _gold(item) == "yes" and '"Non-Compete"' in item.question.instructions
    assert item.state.startswith("Contract: ACME 2001-EX-10 DISTRIBUTOR AGREEMENT\nExcerpt:\n[…]")
    (neg,) = _parsed(t.cuad({**row, "present": False}, _rng(), {}))
    assert _gold(neg) == "no"


# --- COLD French law --------------------------------------------------------------------------


def _parse_fr(text: str) -> date:
    if m := re.search(r"(\d{4})-(\d{2})-(\d{2})", text):
        return date(int(m[1]), int(m[2]), int(m[3]))
    if m := re.search(r"(\d{2})/(\d{2})/(\d{4})", text):
        return date(int(m[3]), int(m[2]), int(m[1]))
    m = re.search(r"(1er|\d{1,2}) (\w+) (\d{4})", text)
    return date(int(m[3]), t.MONTHS_FR.index(m[2]) + 1, 1 if m[1] == "1er" else int(m[1]))


def _cold_row(**kw):
    row = {"article_identifier": "LEGIARTI1", "article_num": "L343-11", "article_etat": "VIGUEUR",
           "article_date_debut": "2017-01-01", "article_date_fin": "2999-01-01",
           "texte_date_publi": "2015-12-29", "texte_date_signature": "2015-12-28",
           "texte_nature": "CODE", "texte_titre": "Code des pensions",
           "texte_contexte": "Partie législative\nLivre III", "article_contenu_text": "Texte."}  # fmt: skip
    return {**row, **kw}


@pytest.mark.parametrize("fin", ["2999-01-01", "2021-07-01"])
def test_cold_labels_follow_the_stated_dates(fin):
    debut, end = date(2017, 1, 1), (None if fin.startswith("2999") else date.fromisoformat(fin))
    labels = set()
    for seed in range(300):
        rows = t.cold_law(_cold_row(article_date_fin=fin), _rng(seed), {})
        if not rows:
            continue
        (item,) = _parsed(rows, lang="fr")
        asked = _parse_fr(item.question.instructions.split("à la date du ")[1])
        applicable = debut <= asked and (end is None or asked < end)
        assert _gold(item) == ("yes" if applicable else "no"), (seed, asked)
        assert "en vigueur à compter du" in item.state
        labels.add(_gold(item))
    assert labels == {"no", "yes"}


def test_cold_negatives_include_near_misses_and_bad_rows_are_dropped():
    near = 0
    for seed in range(400):
        query = t.cold_query_date(date(2017, 1, 1), None, _rng(seed))
        if query and not query[1]:
            near += (date(2017, 1, 1) - query[0]).days <= 31
    assert near > 40
    assert t.cold_law(_cold_row(article_date_debut="bad"), _rng(), {}) == []
    assert t.cold_law(_cold_row(article_date_fin="2010-01-01"), _rng(), {}) == []


# --- ArGiMi ----------------------------------------------------------------------------------

ARRET = (
    "<p>CIV. 1<br>Rejet<br>M. X, président<br>LA COUR DE CASSATION, PREMIÈRE CHAMBRE CIVILE, "
    "a rendu l'arrêt suivant : </p><p>Sur le moyen unique :<br>Attendu, selon l'arrêt "
    "attaqué, que Mme X... fait grief à l'arrêt de rejeter sa demande "
    + "de fixation "
    * 40
    + ";<br>Mais attendu que la cour d'appel en a exactement déduit que la demande devait "
    "être rejetée ; que le moyen n'est pas fondé ;<br>PAR CES MOTIFS :<br>REJETTE le pourvoi ;"
    "<br>Condamne Mme X... aux dépens ;</p><p>MOYEN ANNEXE au présent arrêt. Cassation</p>"
)


def test_argimi_strips_the_header_and_the_operative_part():
    text = t.argimi_text(ARRET)
    assert text is not None and text.startswith("Sur le moyen unique")
    assert "REJETTE" not in text and "PAR CES MOTIFS" not in text and "Rejet\n" not in text
    assert not t.ARGIMI_LEAK.search(text)
    assert t.argimi_text(ARRET.replace("PAR CES MOTIFS", "")) is None  # no operative marker
    leaky = ARRET.replace("Mais attendu", "CASSE ET ANNULE l'arrêt ; Mais attendu")
    assert t.argimi_text(leaky) is None


def test_argimi_old_criminal_format_loses_its_outcome_head():
    old = ("<p>CASSATION sur le pourvoi formé par :<br>- X... Charles,<br>contre l'arrêt de la "
           "cour d'assises<br>LA COUR,<br>Vu le mémoire produit ;<br>" + "Attendu que " * 60
           + "<br>PAR CES MOTIFS :<br>CASSE ET ANNULE</p>")  # fmt: skip
    text = t.argimi_text(old)
    assert text is not None and text.startswith("LA COUR,") and "CASSATION" not in text
    assert t.argimi_text(old.replace("LA COUR,", "La juridiction,")) is None  # still leaky


@pytest.mark.parametrize(
    ("solution", "outcome"),
    [("REJET.", "rejet"), ("Cassation", "cassation"), ("CASSATION PARTIELLE.",
     "cassation_partielle"), ("Irrecevabilité", "irrecevabilite"), ("Cassation sans renvoi",
     "cassation"), ("Cassation partielle REJET Cassation", None), ("Désistement", None)],
)  # fmt: skip
def test_argimi_outcomes(solution, outcome):
    assert t.argimi_outcome(solution) == outcome


def test_argimi_rows_are_outcome_choices_without_the_answer():
    row = {"id": "J1", "title": "Cour de cassation, civile, Chambre civile 1, 11 janvier 2017, "
           "15-27.784, Publié au bulletin", "text": t.argimi_text(ARRET), "outcome": "rejet"}  # fmt: skip
    (item,) = _parsed(t.argimi(row, _rng(), {}), lang="fr")
    assert _gold(item) == "rejet" and set(item.question.labels) == set(t.ARGIMI_CRITERIA)
    assert "Publié" not in item.state and "REJETTE" not in item.state


# --- negation twins --------------------------------------------------------------------------


def test_twins_negate_the_question_swap_the_criteria_and_flip_the_label():
    base = t.squad_abstain("en")
    twin = t.negated(base, t.TWIN_NEGATIONS["twin_squad2"])
    row = {"context": "Paris is in France.", "question": "Where is Paris?",
           "answers": {"text": ["France"], "answer_start": [12]}}  # fmt: skip
    (orig,) = _parsed(base(row, _rng(), {}))
    (neg,) = _parsed(twin(row, _rng(), {}))
    assert neg.state == orig.state and neg.target == tuple(reversed(orig.target))
    assert neg.question.descriptions == tuple(reversed(orig.question.descriptions))
    assert (
        neg.question.instructions == "Is the answer missing from the passage, for: Where is Paris?"
    )
    piaf = t.negated(t.squad_abstain("fr"), t.TWIN_NEGATIONS["twin_piaf_fr"])
    (fr,) = _parsed(piaf({**row, "answers": {"text": []}}, _rng(), {}), lang="fr")
    assert _gold(fr) == "yes" and fr.question.instructions.startswith("La réponse est-elle absente")
    boolq = t.negated(t.boolq, t.TWIN_NEGATIONS["twin_boolq"])
    (b,) = _parsed(boolq({"passage": "P.", "question": "is it", "answer": True}, _rng(), {}))
    assert _gold(b) == "no" and b.question.instructions.startswith("Based only on the text, is")
    soft = t.negated(lambda r, g, m: [{**orig_row(), "target": {"no": 0.4, "yes": 0.6}}],
                     t.TWIN_NEGATIONS["twin_squad2"])  # fmt: skip
    assert soft({}, _rng(), {}) == []  # only exact rows are negated


def orig_row():
    return {"state": "s", "question": {"type": "noul", "instructions": "Does the passage "
            "contain the answer to: q", "criteria": {"false": "a", "true": "b"}}}  # fmt: skip


# --- mix-v6 hygiene ----------------------------------------------------------------------------


def _item(state, instructions="Is it?", expected="yes", id_="x"):
    return parse_item({"id": id_, "state": state, "expected": expected, "source": "s",
                       "family": "f", "lang": "en",
                       "question": {"type": "noul", "instructions": instructions,
                                    "criteria": {"false": "", "true": ""}}})  # fmt: skip


PASSAGE = " ".join(f"word{i}" for i in range(80))


def test_passage_dedup_drops_train_rows_on_a_held_out_passage_whatever_the_question():
    held = HeldIndex.of([_item(PASSAGE, "Held question?", id_="h")])
    train = [
        _item(PASSAGE, "Another question?", id_="same_passage"),
        _item(PASSAGE.replace("word70", "other"), "A third question?", id_="near_passage"),
        _item("A fresh passage with nothing in common.", id_="fresh"),
    ]
    kept, drops = clean(train, held, set())
    assert [it.id for it in kept] == ["same_passage", "near_passage", "fresh"] and not drops
    kept, drops = clean(train, held, set(), passages=True)
    assert [it.id for it in kept] == ["fresh"]
    assert drops == {"train_state_in_heldout": 1, "train_state_near_heldout": 1}


def test_jevbench_8gram_scan_covers_state_instructions_and_options():
    jev = _item("Purpose and scope: this policy governs the reimbursement of travel costs.",
                "Is the claim covered under the policy for remote staff?")  # fmt: skip
    grams = JevBenchGrams([jev])
    rows = [
        _item("Our handbook: this policy governs the reimbursement of travel costs, in full."),
        _item("Fresh text.", "Is the claim covered under the policy for remote staff?"),
        _item("this policy governs the reimbursement of travel", "Unrelated?"),  # 7 words
    ]
    assert [grams.overlaps(r) for r in rows] == [True, True, False]
    kept, drops = clean(rows, HeldIndex.of([]), set(), grams=grams)
    assert len(kept) == 1 and drops == {"jevbench_8gram": 2}
