"""mix-v4 hygiene: the cross-source train post-pass (held-out leaks, near-duplicates, long
prompts, exact duplicates), the adapter fixes found by the mix-v3 data report, and the
halved gen_long_refund_policy weight. Tiny synthetic rows only (no download)."""

import json

import numpy as np

from tjev.data.hygiene import HeldIndex, clean, exact_key, too_long
from tjev.data.item import parse_item
from tjev.data.mix import BLOCKS, mixture_weights
from tjev.data.near import MinHasher, near_hits, shingle_hashes
from tjev.data.sources.contenders import criterion_text, decider, mghafiri
from tjev.data.sources.public import SOURCES, helpsteer, jailbreak
from tjev.data.sources.text_analysis import hc3_fr, mlma, tweet_sentiment

RNG = np.random.default_rng(0)
WORDS = " ".join(f"word{i}" for i in range(120))


def _item(state, instructions="Is it red?", expected="yes", id_="x", **criteria):
    return parse_item(
        {
            "id": id_,
            "state": state,
            "question": {
                "type": "noul",
                "instructions": instructions,
                "criteria": criteria or {"false": "no", "true": "yes"},
            },
            "expected": expected,
            "source": "s",
        }
    )


# --- 1. hc3_fr ---------------------------------------------------------------------------


def test_hc3_fr_reads_the_classlabel_ints_and_yields_both_labels():
    meta = {"label_names": ["HUMAN", "CHATGPT"]}  # the dataset's ClassLabel names
    rows = [
        {"question": "Pourquoi ?", "answer": f"Parce que {i}.", "label": i % 2} for i in range(6)
    ]
    gold = [
        parse_item(r).question.labels[parse_item(r).gold]
        for x in rows
        for r in hc3_fr(x, RNG, meta)
    ]
    assert set(gold) == {"no", "yes"}
    assert gold[:2] == ["no", "yes"]  # 0 = HUMAN, 1 = CHATGPT
    assert (
        hc3_fr({"question": "Q", "answer": "A", "label": "CHATGPT"}, RNG, {})[0]["expected"]
        == "yes"
    )


# --- 2. and 3. structured criteria -------------------------------------------------------


def test_criterion_text():
    assert (
        criterion_text({"level": "Minor", "definition": "Cosmetic defect"})
        == "Minor: Cosmetic defect"
    )
    assert criterion_text({"value": None}) == ""
    assert criterion_text({"_null": None}) == ""
    assert criterion_text(None) == ""
    assert criterion_text({"label": "Refund"}) == "Refund"
    assert criterion_text({"level": 2, "desc": "Sensitive data"}) == "2: Sensitive data"
    text = criterion_text({"definition": "Broken", "examples": ["install fails", "crash"]})
    assert text == "Broken; Examples: install fails; crash"
    assert "{" not in criterion_text({"does": "Run tests", "args": {"branch": "string"}})


def test_structured_instruction_fields_are_text():
    from tjev.data.sources.contenders import _instructions

    text = _instructions({"question": "Which?", "exemption_list": {"privacy": "Private data"}})
    assert text == "Which?\nExemption list: Privacy: Private data"


def _no_repr(item):
    rendered = "\n".join(item.question.descriptions)
    assert "{'" not in rendered and ": None" not in rendered and rendered.strip() != "None"


def test_mghafiri_renders_score_levels_and_dict_criteria_as_text():
    questions = {
        "sev": {
            "type": "score",
            "instructions": "How severe?",
            "criteria": [
                {"level": "None", "definition": "No defect"},
                {"level": "Minor", "definition": "Cosmetic"},
            ],
        },
        "kind": {
            "type": "choice",
            "instructions": "Which kind?",
            "criteria": {"a": {"definition": "Kind A"}, "b": {"definition": "Kind B"}},
        },
        "ok": {
            "type": "noul",
            "instructions": "Fine?",
            "criteria": {"true": {"definition": "It is fine"}, "false": {"definition": "Not fine"}},
        },
    }
    targets = {
        "sev": {"probabilities": {"0": 0.2, "1": 0.8}},
        "kind": {"probabilities": {"a": 1.0, "b": 0.0}},
        "ok": {"noul": 0.9},
    }
    row = {"state": json.dumps({"x": 1}), "questions": json.dumps(questions),
           "targets": json.dumps(targets), "domain": "d"}  # fmt: skip
    items = [parse_item(r) for r in mghafiri(row, RNG, {})]
    by = {it.question.instructions: it for it in items}
    assert by["How severe?"].question.descriptions == ("None: No defect", "Minor: Cosmetic")
    assert by["Which kind?"].question.descriptions == ("Kind A", "Kind B")
    assert by["Fine?"].question.descriptions == ("Not fine", "It is fine")
    for it in items:
        _no_repr(it)


def test_decider_drops_empty_option_descriptors():
    q = {"type": "choice", "instructions": "Route?", "answer": "card",
         "criteria": {"card": {"value": None}, "misc": None, "fees": {"description": "Fees"}}}  # fmt: skip
    item = parse_item(
        decider({"state": "S", "recipe": "r", "questions": json.dumps([q])}, RNG, {})[0]
    )
    assert item.question.descriptions == ("", "", "Fees")
    _no_repr(item)


# --- 6. HTML entities --------------------------------------------------------------------


def test_html_entities_are_unescaped():
    assert (
        tweet_sentiment("en")({"text": "A &amp; B &gt; C", "label": 0}, RNG, {})[0]["state"]
        == "A & B > C"
    )
    row = {"tweet": "@user x &amp; y", "sentiment": "normal"}
    assert mlma("fr")(row, RNG, {})[0]["state"] == "@utilisateur x & y"
    hs = helpsteer("helpfulness")(
        {"prompt": "D&amp;D?", "response": "&lt;b&gt;", "helpfulness": 3}, RNG, {}
    )
    assert hs[0]["state"] == "Prompt:\nD&D?\n\nResponse:\n<b>"
    assert jailbreak({"prompt": "&lt;DAN&gt;", "type": "jailbreak"}, RNG, {})[0]["state"] == "<DAN>"
    nli = SOURCES["mnli"].adapter
    state = nli({"premise": "S&amp;L", "hypothesis": "R &amp; R", "label": 0}, RNG, {})[0]["state"]
    assert "&amp;" not in state and "S&L" in state


# --- 4. and 5. the post-pass -------------------------------------------------------------


def test_near_hits_needs_the_same_key():
    hasher = MinHasher()
    sig = lambda t: hasher.signature(shingle_hashes(t))  # noqa: E731
    index = np.stack([sig(WORDS)])
    query = np.stack(
        [sig(WORDS.replace("word60", "other")), sig(WORDS), sig("unrelated text " * 20)]
    )
    k = np.array([7], np.uint64)
    assert near_hits(query, index).tolist() == [True, True, False]
    hits = near_hits(query, index, query_key=np.array([7, 8, 7], np.uint64), index_key=k)
    assert hits.tolist() == [True, False, False]


def test_clean_drops_leaks_near_leaks_long_prompts_and_duplicates():
    held = HeldIndex.of(
        [_item(WORDS, id_="h1"), _item("The held state, exactly.", "Is it blue?", id_="h2")]
    )
    train = [
        _item(WORDS.replace("word60", "other"), id_="near"),  # near a held state, same instructions
        _item(WORDS.replace("word60", "other"), "Another question?", id_="near_other_q"),  # kept
        _item("the held STATE exactly", "is it blue", id_="equal"),  # equal after normalisation
        _item("Fresh state one.", id_="a"),
        _item("Fresh state one.", id_="a_dup"),  # exact duplicate of a (other id)
        _item("Fresh state one.", expected="no", id_="a_other_target"),  # another target: kept
        _item("x " * 3000, id_="long"),
    ]
    seen = set()
    count = lambda texts: [len(t.split()) for t in texts]  # noqa: E731 (a whitespace "tokenizer")
    kept, drops = clean(train, held, seen, count_tokens=count, limit=2000)
    assert [it.id for it in kept] == ["near_other_q", "a", "a_other_target"]
    assert drops == {"train_near_heldout": 1, "train_equals_heldout": 1,
                     "train_exact_duplicate": 1, "prompt_too_long": 1}  # fmt: skip
    # a later source: its exact duplicates of earlier sources' rows are dropped too
    kept2, drops2 = clean([_item("Fresh state one.", id_="b")], held, seen, count_tokens=count)
    assert kept2 == [] and drops2 == {"train_exact_duplicate": 1}
    assert len(seen) == 3


def test_too_long_needs_a_tokenizer_and_skips_short_prompts():
    items = [_item("short"), _item("y " * 5000)]
    assert too_long(items, None).tolist() == [False, False]
    calls = []

    def count(texts):
        calls.append(len(texts))
        return [len(t) for t in texts]

    assert too_long(items, count).tolist() == [False, True]
    assert calls == [2]  # only the long row was tokenized (both layouts)


def test_exact_key_ignores_case_and_punctuation_but_not_the_target():
    a, b = _item("Hello, World!"), _item("hello world")
    assert exact_key(a) == exact_key(b)
    assert exact_key(a) != exact_key(_item("hello world", expected="no"))


# --- 7. weights --------------------------------------------------------------------------


def test_long_refund_policy_weight_is_halved_within_its_block():
    sources = {n: {"block": "generators", "train": 4000, "lang": {"en": 2800, "fr": 1200}}
               for n in ("gen_refund_policy", "gen_long_refund_policy")}  # fmt: skip
    w = mixture_weights(sources, dict.fromkeys(sources, "generators"))
    assert abs(w["gen_long_refund_policy"] / w["gen_refund_policy"] - 0.5) < 1e-4
    assert abs(sum(w.values()) - 1.0) < 1e-5  # the block keeps its share
    assert BLOCKS["generators"] > 0
