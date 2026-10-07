"""mix-v7 tool / rule / legal sources (tjev.data.sources.tools_v7) on hand-made rows: schema,
label mapping, menus (gold present, ≤ 26 options), licence fields, no answer leakage. No
download."""

import json

import numpy as np
import pytest

from tjev.data.item import LICENSE_USES, parse_item
from tjev.data.sources import tools_v7 as t


def _rng(i=0):
    return np.random.default_rng(i)


def _parsed(rows, lang="en"):
    return [parse_item({"id": "t", "source": "t", "family": "t", "lang": lang, **r}) for r in rows]


def _gold(item):
    return item.question.labels[item.gold]


def _many(adapter, row, n=60, meta=None):
    items = []
    for i in range(n):
        items += _parsed(adapter(row, _rng(i), meta or {}))
    return items


def test_registry_block_licences_and_uses():
    assert set(t.TOOLS_V7_SOURCES) == set(t.LICENCES)
    for name, s in t.TOOLS_V7_SOURCES.items():
        assert s.block == "targeted" and s.license_use in LICENSE_USES, name
        assert s.commercial_ok == (s.license_use == "commercial"), name
        card, redistributable, note = t.LICENCES[name]
        assert card == s.license and redistributable in {"yes", "share-alike", "NC"} and note
        assert s.eval_split or s.group is not None, name  # a held-out split
    assert t.TOOLS_V7_SOURCES["cxm_fr"].license_use == "non-commercial"
    assert t.TOOLS_V7_SOURCES["cxm_fr"].lang == "fr"
    w2c = t.TOOLS_V7_SOURCES["when2call"]
    assert w2c.eval_split is None and "test" not in w2c.config.split("+")  # BFCL-derived test


# --- ShARC -------------------------------------------------------------------------------

SHARC_NAMES = ["Yes", "No", "Irrelevant", "Clarification needed"]


@pytest.mark.parametrize(
    ("label", "gold"), [(0, "yes"), (1, "no"), (2, "irrelevant"), (3, "ask_follow_up")]
)
def test_sharc_labels_and_no_follow_up_leak(label, gold):
    row = {
        "snippet": "## Payment\n\n* have income under the limits\n* meet residence rules",
        "scenario": "I live in Sydney.",
        "question": "Can I get the payment?",
        "history": "Q: Is your income under the limits?\nA: Yes",
        "label": label,
        "follow_up_question": "Do you meet the residence rules?" if label == 3 else None,
    }
    (item,) = _parsed(t.sharc(row, _rng(), {"label_names": SHARC_NAMES}))
    assert _gold(item) == gold and item.question.type == "choice"
    assert set(item.question.labels) == set(t.SHARC_LABELS)
    assert "residence rules?" not in item.state  # the follow-up question is the answer
    assert "Is your income under the limits?" in item.state


# --- tools -------------------------------------------------------------------------------

TOOLS = [
    {"name": "get_weather", "description": "Current weather for a city.",
     "parameters": {"city": {"description": "City name", "type": "str", "default": ""}}},
    {"name": "convert_currency", "description": "Convert an amount between currencies.",
     "parameters": {"type": "dict", "properties": {"amount": {"description": "Amount", "type": "float"}},
                    "required": ["amount"]}},
    {"name": "search_news", "description": "Search news articles.", "parameters": {}},
]  # fmt: skip


def test_tool_text_reads_both_parameter_formats():
    assert "city (str): City name" in t.tool_text(TOOLS[0])
    assert "amount (float, required): Amount" in t.tool_text(TOOLS[1])


def test_xlam_items_are_valid_and_the_gold_tool_is_on_the_menu():
    row = {
        "query": "What's the weather in Paris?",
        "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}]),
        "tools": json.dumps(TOOLS),
    }
    items = _many(t.xlam, row)
    kinds = {it.question.type for it in items}
    assert kinds == {"choice", "noul"}
    for it in items:
        q = it.question
        assert len(q.labels) <= 26
        if q.type == "choice":
            assert _gold(it) == "get_weather" and t.NONE_LABEL in q.labels
        elif "`get_weather`" in q.instructions or "Can one" in q.instructions:
            assert _gold(it) == "yes"
        else:
            assert _gold(it) == "no"
        assert 'Paris"' not in it.state  # the call's arguments are never shown


def test_xlam_drops_calls_to_unlisted_tools_and_irrelevance_gold_is_none():
    row = {"query": "q", "answers": json.dumps([{"name": "other", "arguments": {}}]),
           "tools": json.dumps(TOOLS)}  # fmt: skip
    assert t.xlam(row, _rng(), {}) == []
    irr = {"query": "Who won the 2019 Final Four?", "answers": "[]", "tools": json.dumps(TOOLS)}
    items = _many(t.xlam_irrelevance, irr)
    assert items and all(_gold(it) in ("no", t.NONE_LABEL) for it in items)
    assert t.xlam_irrelevance({**irr, "tools": "[]"}, _rng(), {}) == []  # trivial: no tool


def test_when2call_categories_and_items():
    assert (
        t.when2call_category('<TOOLCALL>[{"name": "f", "arguments": {}}]</TOOLCALL>') == "call_tool"
    )
    assert t.when2call_category("Could you please provide the numbers?") == "ask_follow_up"
    assert t.when2call_category("To proceed, I'll need the URL of the image.") == "ask_follow_up"
    assert (
        t.when2call_category("Apologies, but I'm unable to perform web searches.") == "cannot_help"
    )
    assert t.when2call_category("Sure.") is None
    assert t.when2call_called(
        '<TOOLCALL>[{"name": "get_weather", "arguments": {}}]</TOOLCALL>'
    ) == {"get_weather"}
    row = {
        "tools": json.dumps(TOOLS),
        "messages": json.dumps([{"role": "user", "content": "Weather in Rome?"}]),
        "reply": '<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Rome"}}]</TOOLCALL>',
        "query": "Weather in Rome?",
    }
    items = _many(t.when2call, row)
    behaviour = [it for it in items if "what should the assistant do" in it.question.instructions]
    assert behaviour and all(_gold(it) == "call_tool" for it in behaviour)
    assert any("answer_directly" in it.question.labels for it in behaviour)
    assert all("TOOLCALL" not in it.state and 'Rome"' not in it.state for it in items)
    refusal = {**row, "reply": "Apologies, I'm unable to do that."}
    assert {_gold(it) for it in _many(t.when2call, refusal, 10)} == {"cannot_help"}


# --- MetaTool ----------------------------------------------------------------------------

DESCS = {
    "ResearchHelper": "Find research papers and academic articles on a topic.",
    "ResearchFinder": "Find the latest research papers on arXiv.",
    "FinanceTool": "Stock prices, market data and financial news.",
    "NewsTool": "Up-to-date news from around the world.",
    "MapTool": "Generate map images for a location.",
    "LawTool": "Search relevant laws quickly.",
    "NotesTool": "Manage reminders and to-do lists.",
    "ChartTool": "Draw charts and diagrams.",
    "calculator": "Execute a formula and return the result.",
}


def test_metatool_distractors_are_dissimilar_and_menus_valid():
    pool = t.dissimilar_tools(["ResearchHelper"], DESCS)
    assert "ResearchFinder" not in pool and "FinanceTool" in pool  # shares "research"
    row = {"kind": "single", "query": "Find me papers on quantum computing.",
           "tools": ["ResearchHelper"], "label": ""}  # fmt: skip
    items = _many(lambda r, g, m: t.metatool(r, g, m, descs=DESCS), row)
    assert {it.question.type for it in items} == {"choice", "noul"}
    for it in items:
        assert "ResearchFinder" not in it.question.labels and len(it.question.labels) <= 26
        if it.question.type == "choice":
            gold = _gold(it)
            assert gold in ("ResearchHelper", t.NONE_LABEL)
            assert (gold == t.NONE_LABEL) == ("ResearchHelper" not in it.question.labels)


def test_metatool_awareness_and_multi_tool_rows():
    aware = {"kind": "aware", "query": "How's the market today?", "tools": [], "label": "positive"}
    (it,) = _parsed(t.metatool(aware, _rng(), {}, descs=DESCS))
    assert it.question.type == "noul" and _gold(it) == "yes"
    multi = {"kind": "multi", "query": "Tesla news and stock price", "tools": ["FinanceTool", "NewsTool"],
             "label": ""}  # fmt: skip
    for it in _many(lambda r, g, m: t.metatool(r, g, m, descs=DESCS), multi, 30):
        named = it.state.split("Tool `")[1].split("`")[0]
        assert _gold(it) == ("yes" if named in multi["tools"] else "no")


# --- LexGLUE -----------------------------------------------------------------------------


def test_casehold_five_holdings():
    row = {"context": "See United States v. Newman (<HOLDING>); and so on.",
           "endings": [f"holding that {k}" for k in "abcde"], "label": 2}  # fmt: skip
    (it,) = _parsed(t.casehold(row, _rng(), {}))
    assert _gold(it) == "holding_3" and it.question.descriptions[2] == "holding that c"
    assert t.casehold({**row, "context": "no marker"}, _rng(), {}) == []


def test_ledgar_menu_of_eight_without_near_synonyms():
    names = ["Assignments", "Assigns", "Waivers", "Notices", "Governing Laws", "Severability",
             "Counterparts", "Entire Agreements", "Amendments", "Terminations", "Taxes"]  # fmt: skip
    for i in range(20):
        (it,) = _parsed(t.ledgar({"text": "This Agreement may be assigned…", "label": 0}, _rng(i),
                                 {"label_names": names}))  # fmt: skip
        assert len(it.question.labels) == 8 and _gold(it) == "assignments"
        assert "assigns" not in it.question.labels


# --- CXM Arena French ----------------------------------------------------------------------

CONV = "Customer: Bonjour, je veux annuler.\nAgent: Bien sûr."


def test_cxm_quality_intent_and_tool_rows():
    q = {"kind": "quality", "conversation": CONV, "question": "Did the agent show empathy?",
         "gold": "na", "extra": "", "group": "g", "key": "k"}  # fmt: skip
    (it,) = _parsed(t.cxm(q, _rng(), {}), "fr")
    assert _gold(it) == "sans_objet" and it.family.startswith("cxm_fr_synthetic")
    tax = {f"Produit:Motif {i}": f"Description {i}" for i in range(12)}
    intent = {**q, "kind": "intent", "gold": "Produit:Motif 3", "extra": json.dumps(tax)}
    for i in range(10):
        (it,) = _parsed(t.cxm(intent, _rng(i), {}), "fr")
        assert len(it.question.labels) == 8 and _gold(it) == "motif_produit_motif_3"
    defs = [{"type": "function", "function": {"name": f"tool_{i}", "description": f"d{i}",
                                               "parameters": {}}} for i in range(16)]  # fmt: skip
    tool = {**q, "kind": "tool", "gold": json.dumps(["tool_5"]), "extra": json.dumps(defs)}
    (it,) = _parsed(t.cxm(tool, _rng(), {}), "fr")
    assert _gold(it) == "tool_5" and len(it.question.labels) == 16
    two = {**tool, "gold": json.dumps(["tool_5", "tool_7"])}
    for i in range(10):
        (it,) = _parsed(t.cxm(two, _rng(i), {}), "fr")
        named = it.state.split("Outil `")[1].split("`")[0]
        assert _gold(it) == ("yes" if named in ("tool_5", "tool_7") else "no")
    assert t._cxm_group("CUSTOMER[1]:Bonjour!\nAGENT[2]: x") == t._cxm_group("Customer: Bonjour!")


def test_tool_level_required_and_defaults_are_shown():
    tool = {
        "name": "get_meta_data",
        "description": "Market meta data.",
        "parameters": {
            "type": "dict",
            "properties": {
                "locale_info": {"description": "Language code", "type": "str"},
                "lang_id": {"description": "Language id", "type": "int", "default": 1},
            },
        },
        "required": ["locale_info"],
    }
    text = t.tool_text(tool)
    assert "locale_info (str, required)" in text and "lang_id (int, default 1)" in text
