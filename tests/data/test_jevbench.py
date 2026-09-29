"""The JevBench contamination filter."""

import pytest

from tjev.data.item import parse_item
from tjev.data.jevbench import JevBenchFilter

PUBLIC_STATE = (
    "Refund policy: customers may return unused items within thirty days of delivery "
    "for a full refund, provided the original receipt is shown at the service desk. "
    "Opened software and gift cards cannot be returned under any circumstances."
)
PUBLIC_INSTRUCTION = "Under this policy, may the customer return the opened software for a refund?"


def _row(state, instructions="Is this allowed?"):
    return {"state": state, "question": {"type": "noul", "instructions": instructions}}


@pytest.fixture
def jb():
    return JevBenchFilter([(PUBLIC_STATE, PUBLIC_INSTRUCTION), ("A short public state.", "Hi?")])


def test_exact_state_match_after_normalisation(jb):
    assert PUBLIC_STATE[:24] == "Refund policy: customers"
    assert jb.reason("  REFUND policy: customers" + PUBLIC_STATE[24:], "x") == "state"
    assert jb.reason("An unrelated ticket about login failures.", "x") is None


def test_eight_gram_overlap(jb):
    words = PUBLIC_STATE.split()
    two = " ".join(words[:9])  # 9 words = 2 shared 8-grams: allowed
    three = " ".join(words[:10])  # 3 shared 8-grams: dropped
    assert jb.reason(f"Ticket 42. {two} Customer asks.", "x") is None
    assert jb.reason(f"Ticket 42. {three} Customer asks.", "x") == "8gram"


def test_instruction_match_and_generic_exemption(jb):
    other = "A different document about shipping delays in the north warehouse."
    assert jb.reason(other, PUBLIC_INSTRUCTION) == "instructions"
    assert jb.reason(other, PUBLIC_INSTRUCTION, generic=True) is None
    # short instructions (<60 chars) are generic
    kept, drops = jb.filter([_row(other, "Hi?")])
    assert len(kept) == 1 and not drops
    # a long public instruction is generic when ≥20 training rows share it
    rows = [_row(f"{other} Case {i}.", PUBLIC_INSTRUCTION) for i in range(20)]
    kept, drops = jb.filter(rows)
    assert len(kept) == 20
    kept, drops = jb.filter([*rows[:19], _row(PUBLIC_STATE)])
    assert len(kept) == 0
    assert drops == {"jevbench_overlap": 20, "jevbench_overlap:instructions": 19,
                     "jevbench_overlap:state": 1}  # fmt: skip


def test_structured_states_are_compared_as_rendered_text():
    state = {"ticket": {"body": "Charged twice for invoice INV-1"}}
    rendered = parse_item(
        {"state": state, "question": {"type": "noul", "instructions": "x"}, "expected": "no"}
    ).state
    jb = JevBenchFilter([(rendered, "x")])
    kept, drops = jb.filter([_row(state)])
    assert not kept and drops["jevbench_overlap:state"] == 1
