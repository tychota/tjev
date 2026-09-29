"""Contender adapters (the mix's "jev" block).

Sample rows mirror each dataset's real fields (checked on 2026-09-29)."""

import json

import numpy as np
import pytest

from tjev.data.item import parse_item
from tjev.data.sources.contenders import (
    JEV_SOURCES,
    certo,
    decider,
    decider_situation,
    mghafiri,
    openjev,
    openjev_options,
    plumb,
)

RNG = np.random.default_rng(0)


def _parse(rows):
    items = [parse_item(r) for r in rows]
    for item in items:
        assert abs(sum(item.target) - 1.0) < 1e-6
    return items


def test_every_contender_source_is_in_the_jev_registry():
    assert set(JEV_SOURCES) == {
        "plumb",
        "openjev",
        "decider_teacher",
        "mghafiri_scenarios",
        "certo",
    }
    held_out = [s for s in JEV_SOURCES.values() if s.eval_split is None]
    assert all(s.group is not None for s in held_out)


PLUMB_NOUL = {
    "id": "train-21-000131-q0",
    "family": "probability",
    "domain": "insurance claims",
    "state": "Hartwell Mutual Q3 Claims Feed Audit Memo ...",
    "question": json.dumps(
        {
            "type": "noul",
            "instructions": "If one valid Q3 claim is selected at random, is it denied BI?",
            "criteria": {"true": "Denied bodily injury.", "false": "Not denied bodily injury."},
        }
    ),
    "expected": "false",
    "teacher_probs": json.dumps({"true": 0.36, "false": 0.64}),
}


def test_plumb_probability_rows_are_soft_and_noul_labels_are_no_yes():
    (row,) = plumb(PLUMB_NOUL, RNG, {})
    (item,) = _parse([row])
    assert item.question.labels == ("no", "yes")
    assert item.target == pytest.approx((0.64, 0.36))
    assert item.question.descriptions == ("Not denied bodily injury.", "Denied bodily injury.")
    assert row["family"] == "probability"


def test_plumb_non_probability_rows_use_the_answer_key():
    raw = {
        **PLUMB_NOUL,
        "family": "rubric",
        "question": {
            "type": "score",
            "instructions": "Rate the risk that the claim is misleading.",
            "criteria": ["Substantiated", "Limited support", "Misleading"],
        },
        "expected": "1",
        "teacher_probs": {"0": 0.0, "1": 1.0, "2": 0.0},
    }
    (item,) = _parse(plumb(raw, RNG, {}))
    assert item.question.labels == ("0", "1", "2") and item.gold == 1 and max(item.target) == 1


def _openjev(kind, options, target, state='{"a": 1}', source="customer-control-v1"):
    return {
        "id": "x",
        "group_id": "g",
        "source": source,
        "kind": kind,
        "question": "Q?",
        "options": options,
        "target": target,
        "state_json": state,
    }


def test_openjev_noul_is_reordered_to_no_yes():
    (row,) = openjev(_openjev("noul", ["yes", "no"], [0.8, 0.2]), RNG, {})
    (item,) = _parse([row])
    assert item.question.labels == ("no", "yes")
    assert item.target == pytest.approx((0.2, 0.8))
    (row,) = openjev(_openjev("noul", ["no", "yes"], [1.0, 0.0]), RNG, {})
    assert row["expected"] == "no"


def test_openjev_choice_menus_and_states():
    options = ["USD: US dollars", "EUR: Euros", "review: Currency unsupported"]
    (row,) = openjev(_openjev("choice", options, [0.0, 0.0, 1.0]), RNG, {})
    item = parse_item(row)
    assert item.question.labels == ("USD", "EUR", "review")
    assert item.question.descriptions[1] == "Euros" and row["expected"] == "review"
    assert json.loads(item.state) == {"a": 1}  # state_json dict rendered as JSON text
    # a JSON-string state stays text; bare identifiers become labels; sentences option_N
    (row,) = openjev(_openjev("choice", ["K11", "abstain"], [0.5, 0.5], '"Customer: hi"'), RNG, {})
    assert row["state"] == "Customer: hi" and row["target"] == {"K11": 0.5, "abstain": 0.5}
    labels, descs = openjev_options(["The premise supports the claim.", "It contradicts it."])
    assert labels == ["option_1", "option_2"] and descs[0].startswith("The premise")


def test_openjev_score_and_dropped_kinds():
    (row,) = openjev(_openjev("score", ["0", "1", "2", "3"], [0, 0, 1, 0]), RNG, {})
    item = parse_item(row)
    assert item.question.labels == ("0", "1", "2", "3") and item.gold == 2
    assert item.question.descriptions == ("", "", "", "")
    assert openjev(_openjev("multilabel", ["a", "b"], [1, 0]), RNG, {}) == []
    assert openjev(_openjev("ordinal", ["a", "b"], [1, 0]), RNG, {}) == []


def test_decider_one_item_per_question_and_teacher_disagreement_dropped():
    row = {
        "state": "rm -rf /var/lib/postgresql/data",
        "domain": "shell",
        "recipe": "commands",
        "questions": json.dumps(
            [
                {
                    "type": "choice",
                    "instructions": "How risky is running this shell command?",
                    "criteria": {"safe": "read-only", "destructive": "irreversible"},
                    "answer": "destructive",
                    "teacher_ok": True,
                },
                {"type": "noul", "instructions": "Outside the project?", "answer": True, "teacher_ok": True},
                {"type": "noul", "instructions": "Dropped", "answer": True, "teacher_ok": False},
                {"type": "choice", "instructions": "Route it", "criteria": {"a": None, "b": None},
                 "answer": "b"},
                {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"],
                 "answer": 2, "teacher_p": 0.97},
                {"type": "noul", "instructions": "Low teacher_p", "criteria": None, "answer": False,
                 "teacher_p": 0.3},
            ]
        ),
    }  # fmt: skip
    rows = decider(row, RNG, {})
    items = _parse(rows)
    assert [i.question.instructions for i in items] == [
        "How risky is running this shell command?",
        "Outside the project?",
        "Route it",
        "Urgency?",
    ]
    assert items[1].question.labels == ("no", "yes") and items[1].gold == 1
    assert items[1].question.descriptions == ("", "")
    assert items[2].question.descriptions == ("", "")  # null choice descriptions
    assert items[3].gold == 2 and rows[0]["family"] == "commands"


def test_decider_situations_become_choice_rows():
    raw = {
        "situation": "Smoke pours from the doorway.",
        "question": "What now?",
        "options": ["Charge", "Evacuate"],
        "answer": 1,
        "danger": True,
        "domain": "a firefighter",
        "reason": "flashover",
    }
    (item,) = _parse(decider({**decider_situation(raw), "questions": json.dumps(
        decider_situation(raw)["questions"])}, RNG, {}))  # fmt: skip
    assert item.question.labels == ("option_1", "option_2") and item.gold == 1


def test_mghafiri_soft_targets_per_question():
    row = {
        "id": "b01-0001",
        "domain": "customer_support",
        "state": json.dumps({"ticket": {"body": "Please fix invoice INV-88213"}}),
        "questions": json.dumps(
            {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle `ticket`?",
                    "criteria": {"technical": "Bugs", "billing": "Invoices"},
                },
                "asks_refund": {"type": "noul", "instructions": "Money back?"},
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated?",
                    "criteria": ["Neutral", "Mild", "Clear"],
                },
            }
        ),
        "targets": json.dumps(
            {
                "department": {"probabilities": {"technical": 0.06, "billing": 0.94}},
                "asks_refund": {"noul": 0.104},
                "frustration": {"probabilities": {"0": 0.12, "1": 0.76, "2": 0.12}},
            }
        ),
    }
    items = _parse(mghafiri(row, RNG, {}))
    assert [i.question.type for i in items] == ["choice", "noul", "score"]
    assert items[1].target == pytest.approx((0.896, 0.104))
    assert items[2].gold == 1 and all(i.family == "customer_support" for i in items)
    assert '"ticket"' in items[0].state


def test_mghafiri_structured_instructions_become_text():
    q = {"question": "Does `emails` hold every approval?", "counts_as_approval": "A written OK"}
    row = {
        "domain": "hr_recruiting",
        "state": json.dumps("Plain text state"),
        "questions": json.dumps({"q": {"type": "noul", "instructions": q}}),
        "targets": json.dumps({"q": {"noul": 1.0}}),
    }
    (item,) = _parse(mghafiri(row, RNG, {}))
    assert item.question.instructions == (
        "Does `emails` hold every approval?\nCounts as approval: A written OK"
    )
    assert item.state == "Plain text state"


def _certo(question, target, family="coding_safe"):
    return {
        "record_id": "r",
        "task_family": family,
        "source_id": "domain_generator",
        "question_type": question["type"],
        "state": "A script is about to run `echo hello`",
        "question": json.dumps(question),
        "target": json.dumps(target),
    }


def test_certo_binary_score_and_posterior():
    binary = {
        "type": "binary",
        "instructions": "Should this command be blocked?",
        "options": [
            {"id": "yes", "description": "destructive"},
            {"id": "no", "description": "safe"},
        ],
    }
    (item,) = _parse(
        certo(_certo(binary, {"kind": "categorical_label", "label_id": "no"}), RNG, {})
    )
    assert item.question.type == "noul" and item.gold == 0
    assert item.question.descriptions == ("safe", "destructive")
    score = {"type": "score", "levels": 4, "instructions": "Assign priority.", "rubric": "cap 3"}
    t = {"kind": "score_distribution", "probabilities": [0, 0, 0, 1.0]}
    (item,) = _parse(certo(_certo(score, t), RNG, {}))
    assert item.question.labels == ("0", "1", "2", "3") and item.gold == 3
    assert item.question.instructions.endswith("Rule: cap 3")
    choice = {
        "type": "choice",
        "instructions": "Which option best matches?",
        "options": [{"id": "opt0", "description": "a"}, {"id": "opt1", "description": "b"}],
    }
    t = {"kind": "categorical_distribution", "probabilities": {"opt0": 0.25, "opt1": 0.75}}
    (row,) = certo(_certo(choice, t, "known_posterior"), RNG, {})
    assert row["target"] == {"opt0": 0.25, "opt1": 0.75}
    multi = {**choice, "type": "independent_binary"}
    assert certo(_certo(multi, {"kind": "bernoulli_labels"}), RNG, {}) == []
