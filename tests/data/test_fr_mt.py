"""jev_fr_mt gate: quantity / entity preservation, segmentation, label keys (no model)."""

import json

import pytest

from tjev.data.sources.gen_v7 import fr_mt

EN = (
    "Date: Tuesday, March 17, 2026. Meeting at 2:30 pm, budget $1,234.50 for 3 people, "
    "15% off; asset KB-2217, contact legal@example.com. Due March 21st."
)
FR = (
    "Date : mardi 17 mars 2026. Réunion à 14 h 30, budget 1 234,50 $ pour 3 personnes, "
    "15 % de remise ; actif KB-2217, contact legal@example.com. Échéance le 21 mars."
)


def test_dates_times_numbers_survive_across_formats():
    assert fr_mt.compare_quantities(EN, FR) == []
    a = fr_mt.quantities(EN, "en")
    assert a.dates == {(2026, 3, 17): 1, (0, 3, 21): 1}
    assert a.times == {(14, 30): 1} and a.weekdays == {1: 1}
    assert a.numbers["1234.5"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        FR.replace("17 mars", "18 mars"),  # a date moved
        FR.replace("mardi", "mercredi"),  # the weekday of a date changed
        FR.replace("14 h 30", "14 h 00"),  # a time changed
        FR.replace("1 234,50", "1 234,05"),  # an amount changed
        FR.replace("3 personnes", "trois personnes"),  # a number spelled out
    ],
)
def test_changed_quantities_are_rejected(bad):
    assert fr_mt.compare_quantities(EN, bad)


def test_upper_case_ordinal_day():
    assert fr_mt.quantities("LE 1ER MARS 2026", "fr").dates == {(2026, 3, 1): 1}


def test_iso_dates_ordinals_and_kept_decimal_points():
    assert (
        fr_mt.compare_quantities(
            "From 2025-03-01 to the 1st of April", "Du 2025-03-01 au 1er avril"
        )
        == []
    )
    assert fr_mt.compare_quantities("A 3.5% fee", "Des frais de 3,5 %") == []
    assert fr_mt.compare_quantities("A 3.5% fee", "Des frais de 3.5 %") == []


def test_identifiers_must_survive_verbatim_and_names_by_recall():
    problems, _ = fr_mt.compare_entities(EN, FR)
    assert problems == []
    problems, _ = fr_mt.compare_entities(EN, FR.replace("KB-2217", "KB 2217"))
    assert problems and "identifiers" in problems[0]
    src = "Escalate to Priya Raman and tell Jonah Vale. The tool email_send failed."
    ok = "Escaladez à Priya Raman et prévenez Jonah Vale. L'outil email_send a échoué."
    assert fr_mt.compare_entities(src, ok)[0] == []
    lost = "Escaladez à Priya et prévenez Jonah. L'outil email_send a échoué."
    assert fr_mt.compare_entities(src, lost)[0]


def test_plan_rebuilds_layout_and_skips_identifiers():
    text = "Hello there. This is a test.\n- Item one: buy milk\n\nKB-2217"
    plan = fr_mt.plan_text(text)
    assert plan.assemble(plan.segments) == text
    assert "KB-2217" not in plan.segments
    assert plan.assemble(s.upper() for s in plan.segments).endswith("\n\nKB-2217")


FAKE = {
    "Email the contract today.": "Envoyez le contrat par e-mail aujourd'hui.",
    "Which tool fits the request?": "Quel outil convient à la demande ?",
    "Send an e-mail": "Envoyer un e-mail",
    "Do nothing": "Ne rien faire",
}
BACK = {v: k for k, v in FAKE.items()}


def _fake(texts):
    return [FAKE.get(t, t) for t in texts]


def _back(texts):
    return [BACK.get(t, t) for t in texts]


def test_rows_keep_label_keys_targets_and_json_keys():
    rows = [
        {
            "id": "mghafiri_scenarios-train-1-0",
            "state": json.dumps({"user_request": "Email the contract today.", "tool": "email_send"}, indent=1),
            "question": {"type": "choice", "instructions": "Which tool fits the request?",
                         "criteria": {"email_send": "Send an e-mail", "noop": "Do nothing"}},
            "labels": ["email_send", "noop"], "target": {"email_send": 0.9, "noop": 0.1},
            "source": "mghafiri_scenarios", "family": "agent_tool_routing", "lang": "en",
        },
        {
            "id": "plumb-train-1-0", "state": "Spell the word correctly.",
            "question": {"type": "noul", "instructions": "Is the spelling right?",
                         "criteria": {"false": "No", "true": "Yes"}},
            "expected": "yes", "source": "plumb", "family": "x", "lang": "en",
        },
    ]  # fmt: skip
    plans = [fr_mt.plan_row(r) for r in rows]
    assert plans[0].json_strings == ["Email the contract today."]  # "email_send" kept
    assert plans[1].skip == "english-specific question"
    accepted, rejected, _ = fr_mt.translate_rows(rows, _fake, _back)
    assert [r["id"] for r in rejected] == ["plumb-train-1-0"]
    (row,) = accepted
    assert list(row["question"]["criteria"]) == ["email_send", "noop"]
    assert row["target"] == rows[0]["target"]
    assert json.loads(row["state"])["tool"] == "email_send"
    assert row["source"] == "jev_fr_mt" and row["lang"] == "fr"
    assert row["id"] == "jev_fr_mt-mghafiri_scenarios-train-1-0"
    assert row["question"]["criteria"]["noop"] == "Ne rien faire"
    assert row["meta"]["chrf"] == 100.0
    assert "MIT" in row["license_use"] and "CC-BY-4.0" in row["license_use"]
    # the gate of a large batch runs in a fork pool: same rows, same order
    many = [{**rows[i % 2], "id": f"{rows[i % 2]['id']}-{i}"} for i in range(12)]
    serial = fr_mt.translate_rows(many, _fake, _back)[:2]
    old = fr_mt.GATE_PARALLEL_MIN
    try:
        fr_mt.GATE_PARALLEL_MIN = 1
        assert fr_mt.translate_rows(many, _fake, _back)[:2] == serial
    finally:
        fr_mt.GATE_PARALLEL_MIN = old


def test_chrf_and_degeneration():
    assert fr_mt.chrf("abc def", "abc def") == 100.0
    assert fr_mt.chrf("totally different", "the cat sat") < 30
    assert fr_mt.degenerate("A short sentence here.", "le le le le le le le le le le le le le le")
    assert fr_mt.degenerate("A short sentence here.", "Une courte phrase ici.") is None


def test_json_states_keep_keys_layout_and_numbers():
    raw = '[{"id":"A101","note":"Brake pads replaced today.","cost":150.00}]'
    out = fr_mt.json_substitute(raw, {"Brake pads replaced today.": "Plaquettes remplacées."})
    assert out == '[{"id":"A101","note":"Plaquettes remplacées.","cost":150.00}]'


def test_post_edit_state_is_not_the_government():
    assert fr_mt.post_edit("L'État établit-il si la porte est ouverte ?").startswith("La situation")


def _kiwi_rows():
    return [
        {
            "id": "mghafiri_scenarios-train-1-0",
            "state": json.dumps({"user_request": "Email the contract today.", "tool": "email_send"}, indent=1),
            "question": {"type": "choice", "instructions": "Which tool fits the request?",
                         "criteria": {"email_send": "Send an e-mail", "noop": "Do nothing"}},
            "labels": ["email_send", "noop"], "target": {"email_send": 0.9, "noop": 0.1},
            "source": "mghafiri_scenarios", "family": "agent_tool_routing", "lang": "en",
        },
    ]  # fmt: skip


def test_cometkiwi_scores_are_logged_and_gate_only_with_a_threshold():
    """A mocked CometKiwi: each (source, translation) pair is scored once, the row score is
    the length-weighted mean, the model and its licence are recorded, and a threshold drops
    the row while the mandatory gates are unchanged."""
    calls = []

    def qe(pairs):
        calls.append(list(pairs))
        return [0.5 if src.startswith("Email") else 0.9 for src, _ in pairs]

    accepted, _, timing = fr_mt.translate_rows(
        _kiwi_rows(), _fake, _back, qe=qe, min_kiwi=None, min_kiwi_segment=None
    )
    (row,) = accepted
    (pairs,) = calls
    assert len(pairs) == len({src for src, _ in pairs})  # each segment scored once
    assert all(mt == FAKE.get(src, src) for src, mt in pairs)
    meta = row["meta"]
    segs = [s for s, _ in pairs]
    want = sum((0.5 if s.startswith("Email") else 0.9) * len(s) for s in segs) / sum(map(len, segs))
    assert meta["kiwi"] == round(want, 4)
    assert meta["kiwi_worst_segment"] <= meta["kiwi"]
    assert meta["qe_model"] == fr_mt.KIWI_MODEL and "CC-BY-NC-SA-4.0" in meta["qe"]
    assert "kiwi" not in meta["checks"]  # no threshold: logged only
    assert timing["tables"]["kiwi"] == dict(zip(segs, qe(pairs), strict=True))
    assert fr_mt.regate(meta) and not fr_mt.regate(meta, min_kiwi=0.95)
    accepted, rejected, _ = fr_mt.translate_rows(
        _kiwi_rows(), _fake, _back, qe=qe, min_kiwi=0.95, min_kiwi_segment=None
    )
    assert not accepted and rejected[0]["checks"]["kiwi"] is False
    assert all(v for k, v in rejected[0]["checks"].items() if k != "kiwi")  # the rest pass


def test_saved_kiwi_tables_replay_without_the_model():
    _, _, timing = fr_mt.translate_rows(_kiwi_rows(), _fake, _back, qe=lambda p: [0.8] * len(p))
    tables = timing["tables"]
    mt, rt = fr_mt.table_translators(tables)
    accepted, _, _ = fr_mt.translate_rows(
        _kiwi_rows(), mt, rt, qe=fr_mt.table_scorer(tables["kiwi"]), min_kiwi=0.75
    )
    assert accepted[0]["meta"]["kiwi"] == 0.8 and accepted[0]["meta"]["checks"]["kiwi"]


def test_regate_keeps_the_mandatory_checks():
    verdict = {"checks": {"labels": True, "quantities": False, "round_trip": True},
               "chrf": 90.0, "worst_segment_chrf": 80.0, "kiwi": 0.9}  # fmt: skip
    assert not fr_mt.regate(verdict)  # a lost number is never forgiven by a high score
    verdict["checks"]["quantities"] = True
    assert fr_mt.regate(verdict, min_kiwi=0.85)
    assert not fr_mt.regate(verdict, min_chrf=95)
    assert not fr_mt.regate({"checks": {"selection": False}})


def test_default_kiwi_gate_drops_one_bad_segment():
    """A good mean does not save a row with one badly scored segment (the defaults)."""
    accepted, rejected, _ = fr_mt.translate_rows(
        _kiwi_rows(),
        _fake,
        _back,
        qe=lambda p: [0.3 if s.startswith("Email") else 0.95 for s, _ in p],
    )
    assert not accepted and rejected[0]["kiwi_worst_segment"] == 0.3
    assert fr_mt.regate(rejected[0], min_kiwi=0.5) and not fr_mt.regate(
        rejected[0], min_kiwi_segment=fr_mt.MIN_KIWI_SEGMENT
    )


def test_tables_pipes_fences_and_code_spans_are_protected():
    text = (
        "| Item | Price |\n|---|---:|\n| Blue chair | $40 |\n"
        "Run `kubectl get pods | grep api` then check the logs.\n"
        "ps aux | grep nginx\n```\necho hello | wc -c\n```\nDone today."
    )
    plan = fr_mt.plan_text(text)
    segs = plan.segments
    assert "|---|---:|" not in " ".join(segs) and all("|" not in s for s in segs)
    assert "Run TJVC0 then check the logs." in segs
    assert not any("grep" in s or "echo" in s for s in segs)  # code stays as is
    assert {"Item", "Price", "Blue chair", "Done today."} <= set(segs)
    assert plan.assemble(segs) == text
    upper = plan.assemble(s.upper() for s in segs)
    assert "`kubectl get pods | grep api`" in upper and upper.count("|") == text.count("|")
    assert fr_mt.markup_problems(text, upper) == []


def test_lost_placeholder_or_pipe_is_a_markup_problem():
    src = "Run `make test` now. | a | b |"
    assert fr_mt.markup_problems(src, "Exécutez maintenant. | a | b |")
    assert fr_mt.markup_problems(src, "Exécutez `make test` maintenant. a | b |")
    assert fr_mt.markup_problems(src, "Exécutez TJVC0 maintenant. | a | b |")
    assert fr_mt.markup_problems(src, "Exécutez `make test` maintenant. | a | b |") == []
