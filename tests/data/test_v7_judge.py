"""The mix-v7 judge sources (adapters on hand-made rows: schema, label and soft-target
mapping, side-order randomisation, licence fields). No download."""

import json

import numpy as np
import pandas as pd
import pytest

from tjev.data.item import LICENSE_USES, parse_item
from tjev.data.licences import licence_of, license_use_of
from tjev.data.mix import block_of_source
from tjev.data.sources import ALL_SOURCES, V7_SOURCES
from tjev.data.sources import judge as j


def _rng(i=0):
    return np.random.default_rng(i)


def _items(rows, lang="en"):
    return [parse_item({"id": "t", "source": "t", "family": "t", "lang": lang, **r}) for r in rows]


def _gold(item):
    return item.question.labels[item.gold]


def _dist(item):
    return dict(zip(item.question.labels, item.target, strict=True))


# --- registry and licences -------------------------------------------------------------------


def test_judge_sources_are_registered_with_block_licence_and_use():
    assert set(j.JUDGE_SOURCES) <= set(V7_SOURCES) <= set(ALL_SOURCES)
    assert set(j.LICENCES) == set(j.JUDGE_SOURCES)
    for name, source in j.JUDGE_SOURCES.items():
        assert block_of_source(name) == ("analysis" if name == "x_stance_fr" else "judge")
        card, redistributable, _ = licence_of(name)
        assert card == source.license and card != "not recorded"
        assert source.license_use in LICENSE_USES
        assert license_use_of(name) == source.license_use
        assert source.commercial_ok == (source.license_use == "commercial")
        assert source.eval_split or source.group or source.train_only, name
        if source.license_use == "non-commercial":
            assert redistributable == "NC"
    assert license_use_of("x_stance_fr") == "non-commercial"
    assert license_use_of("webgpt") == "unspecified"


# --- helpers ---------------------------------------------------------------------------------


def test_interpolated_mean_and_histogram():
    assert j.interpolated(2.4, 5) == pytest.approx([0, 0, 0.6, 0.4, 0])
    assert j.interpolated(4.0, 5) == [0, 0, 0, 0, 1.0]
    assert sum(k * p for k, p in enumerate(j.interpolated(1.75, 5))) == pytest.approx(1.75)
    assert j.histogram([0, 0, 2], 3) == pytest.approx([2 / 3, 0, 1 / 3])


# --- helpsteer3_x ------------------------------------------------------------------------------


def test_helpsteer3_x_keeps_the_rows_language():
    row = {
        "context": json.dumps([{"role": "user", "content": "Bonjour, aide-moi."}]),
        "response1": "Réponse un.",
        "response2": "Réponse deux.",
        "scores": [-2, -1, 0],
        "domain": "multilingual",
        "lang": "fr",
    }
    (item,) = _items(j.helpsteer3_x(row, _rng(), {}), "fr")
    assert item.lang == "fr" and item.question.type in ("choice", "score")


# --- oasst2 --------------------------------------------------------------------------------------


def _msg(mid, parent, role, text, rank=None, labels=None, lang="en"):
    names = list(labels or {})
    return {
        "message_id": mid,
        "parent_id": parent,
        "role": role,
        "text": text,
        "lang": lang,
        "rank": rank,
        "deleted": False,
        "review_result": True,
        "message_tree_id": "t1",
        "labels": {
            "name": names,
            "value": [labels[n][0] for n in names],
            "count": [labels[n][1] for n in names],
        },
    }


def test_oasst2_rows_rank_and_rating_items():
    lab = {"helpfulness": (0.6875, 4), "quality": (0.625, 4), "fails_task": (0.25, 4)}
    messages = [
        _msg("p", None, "prompter", "How do I boil an egg?"),
        _msg("a1", "p", "assistant", "Boil water, add the egg for 9 minutes.", 0, lab),
        _msg("a2", "p", "assistant", "Eggs are laid by hens.", 1, lab),
        _msg("x", "p", "assistant", "Spanish reply", 2, lab, lang="es"),
    ]
    rows = j.oasst2_rows(messages)
    kinds = [r["kind"] for r in rows]
    assert kinds.count("rank") == 1 and len(rows) <= 3  # the Spanish reply is not kept
    rank = next(r for r in rows if r["kind"] == "rank")
    seen = set()
    for seed in range(20):
        (item,) = _items(j.oasst2(rank, _rng(seed), {}))
        better = (
            "reply_1" if item.state.index("9 minutes") < item.state.index("hens") else "reply_2"
        )
        assert _gold(item) == better
        seen.add(better)
    assert seen == {"reply_1", "reply_2"}  # the better reply's side is random
    score = {"kind": "helpfulness", "turns": json.dumps([{"role": "user", "content": "Q"},
             {"role": "assistant", "content": "A"}]), "reply_b": "", "value": 0.6875,
             "count": 4, "lang": "fr", "tree": "t"}  # fmt: skip
    (item,) = _items(j.oasst2(score, _rng(), {}), "fr")
    assert item.question.type == "score" and len(item.question.labels) == 5
    assert _dist(item) == pytest.approx({"0": 0, "1": 0, "2": 0.25, "3": 0.75, "4": 0})
    assert "évaluateurs" in item.question.instructions
    fails = {**score, "kind": "fails_task", "value": 0.25, "lang": "en"}
    (item,) = _items(j.oasst2(fails, _rng(), {}))
    assert item.question.type == "noul" and _dist(item) == pytest.approx({"no": 0.75, "yes": 0.25})


# --- lmarena / compar:IA pairwise ------------------------------------------------------------


def _conv(prompt, reply):
    return json.dumps(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
    )


@pytest.mark.parametrize(("winner", "expect"), [("model_a", "AAA"), ("model_b", "BBB")])
def test_lmarena_winner_follows_the_side_it_is_shown_on(winner, expect):
    row = {"id": "1", "session": "s", "winner": winner, "lang": "en",
           "conv_a": _conv("Hi", "AAA"), "conv_b": _conv("Hi", "BBB")}  # fmt: skip
    sides = set()
    for seed in range(20):
        (item,) = _items(j.lmarena(row, _rng(seed), {}))
        assert set(item.question.labels) == {"assistant_1", "assistant_2", "tie", "both_bad"}
        first = item.state.index("AAA") < item.state.index("BBB")
        shown_first = (expect == "AAA") == first
        assert _gold(item) == ("assistant_1" if shown_first else "assistant_2")
        sides.add(_gold(item))
    assert sides == {"assistant_1", "assistant_2"}
    tie = {**row, "winner": "both_bad"}
    (item,) = _items(j.lmarena(tie, _rng(), {}))
    assert _gold(item) == "both_bad"


def test_arena_turns_drop_images_and_join_text():
    conv = [{"role": "user", "content": [{"type": "text", "text": "a", "image": None}]}]
    assert j._arena_turns(conv) == [{"role": "user", "content": "a"}]
    img = [{"role": "user", "content": [{"type": "image", "text": None, "image": "x"}]}]
    assert j._arena_turns(img) is None


def test_comparia_arena_votes_reactions():
    row = {"id": "r", "opening": "Bonjour", "earlier": json.dumps(["Premier message"]),
           "conv_a": _conv("Deuxième", "AAA"), "conv_b": _conv("Deuxième", "BBB"),
           "choice": "both_good"}  # fmt: skip
    (item,) = _items(j.comparia_arena(row, _rng(), {}), "fr")
    assert item.lang == "fr" and _gold(item) == "both_good"
    assert item.state.startswith("Messages précédents") and "both_bad" in item.question.labels
    votes = {"id": "v", "opening": "x", "conv_a": _conv("q", "AAA"), "conv_b": _conv("q", "BBB"),
             "winner": "equal"}  # fmt: skip
    (item,) = _items(j.comparia_votes(votes, _rng(), {}), "fr")
    assert _gold(item) == "equal" and set(item.question.labels) == {"assistant_1", "assistant_2",
                                                                   "equal"}  # fmt: skip
    react = {"id": "x", "opening": "o", "question": "Q ?", "answer": "R.", "liked": False}
    (item,) = _items(j.comparia_reactions(react, _rng(), {}), "fr")
    assert item.question.type == "noul" and _gold(item) == "no"
    assert j.comparia_group({"opening": "  Bonjour !"}) == j.comparia_group({"opening": "bonjour"})


# --- webgpt / summarize ----------------------------------------------------------------------


def test_webgpt_margin_becomes_a_graded_target():
    row = {"question": "Who?", "answer_0": "Kent [1]", "answer_1": "Apu",
           "quotes_0": json.dumps({"title": ["T"], "extract": ["E"]}),
           "quotes_1": json.dumps({}), "score": 0.5}  # fmt: skip
    for seed in range(10):
        (item,) = _items(j.webgpt(row, _rng(seed), {}))
        d = _dist(item)
        kent_first = item.state.index("Kent") < item.state.index("Apu")
        win = "answer_1" if kent_first else "answer_2"
        assert d[win] == pytest.approx(0.5) and d["tie"] == pytest.approx(0.5)
    (item,) = _items(j.webgpt({**row, "score": -1.0}, _rng(3), {}))
    assert max(item.target) == 1.0 and _gold(item) != "tie"


def test_summarize_rows_merge_votes_and_axis_ratings():
    info = {"id": "p1", "post": "A long post about cats.", "title": "Cats", "subreddit": "pets"}
    cmp = [
        {"info": info, "summaries": [{"text": "s1"}, {"text": "s2"}], "choice": 0},
        {"info": info, "summaries": [{"text": "s2"}, {"text": "s1"}], "choice": 1},
        {"info": info, "summaries": [{"text": "s1"}, {"text": "s2"}], "choice": 1},
    ]
    axis = [
        {"info": info, "summary": {"text": "s1", "axes": {"overall": 7, "accuracy": None}}},
        {"info": info, "summary": {"text": "s1", "axes": {"overall": 5}}},
    ]
    rows = j.summarize_rows(cmp, axis)
    compare = next(r for r in rows if r["kind"] == "compare")
    assert json.loads(compare["votes"]) == [2, 1]
    (item,) = _items(j.summarize_fb(compare, _rng(), {}))
    d = _dist(item)
    s1_first = item.state.index("Summary 1:\ns1") >= 0 if "Summary 1:\ns1" in item.state else False
    # mix-v7.5: smoothed vote shares, (votes + alpha / k) / (n + alpha)
    assert d["summary_1" if s1_first else "summary_2"] == pytest.approx(2.25 / 3.5)
    rated = next(r for r in rows if r["kind"] == "overall")
    (item,) = _items(j.summarize_fb(rated, _rng(), {}))
    a = j.VOTE_ALPHA / 7
    assert item.question.type == "score" and _dist(item) == pytest.approx(
        {str(i): (c + a) / 2.5 for i, c in enumerate([0, 0, 0, 0, 1, 0, 1])}
    )


# --- scruples -------------------------------------------------------------------------------------


def _scruple(**scores):
    base = {"AUTHOR": 0, "OTHER": 0, "EVERYBODY": 0, "NOBODY": 0, "INFO": 0}
    return {"title": "AITA for hiding my controller?", "text": "Long story " * 10,
            "label_scores": {**base, **scores}, "label": max(scores, key=scores.get)}  # fmt: skip


def test_scruples_votes_are_soft_from_five_and_one_hot_below():
    seen = set()
    for seed in range(30):
        (item,) = _items(j.scruples(_scruple(AUTHOR=3, OTHER=6, INFO=1), _rng(seed), {}))
        seen.add(item.question.type)
        if item.question.type == "choice":
            assert _dist(item)["other"] == pytest.approx(0.6)
        else:  # author or everybody among the 9 verdicts
            assert _dist(item)["yes"] == pytest.approx(3 / 9)
    assert seen == {"choice", "noul"}
    (item,) = _items(j.scruples(_scruple(NOBODY=2, AUTHOR=1), _rng(), {}))
    assert max(item.target) == 1.0 and _gold(item) == "nobody"


# --- community alignment ------------------------------------------------------------------------


def _ca(pref, prompt="Quel est le meilleur fromage ?", second=None):
    row = {"assigned_lang": "fr", "first_turn_prompt": prompt,
           **{f"first_turn_response_{x}": f"Réponse {x.upper()}" for x in "abcd"},
           "first_turn_preferred_response": f"response_{pref}"}  # fmt: skip
    if second:
        row |= {"second_turn_prompt": "Et le vin ?",
                **{f"second_turn_response_{x}": f"Vin {x}" for x in "abcd"},
                "second_turn_preferred_response": f"response_{second}"}  # fmt: skip
    return row


def test_community_alignment_votes_are_aggregated_and_smoothed():
    raw = [_ca("a")] * 6 + [_ca("b")] * 4 + [_ca("c", "Autre question ?", second="d")]
    rows = j.community_alignment_rows([*raw, {**_ca("a"), "assigned_lang": "en"}])
    many = [r for r in rows if r["votes"] == 10]
    assert len(many) == 1
    a = j.VOTE_ALPHA / 4
    assert json.loads(many[0]["target"]) == pytest.approx([(6 + a) / 10.5, (4 + a) / 10.5,
                                                            a / 10.5, a / 10.5])  # fmt: skip
    later = [r for r in rows if json.loads(r["context"])]
    assert len(later) == 1 and "Réponse C" in later[0]["context"]  # continues the preferred one
    single = json.loads(later[0]["target"])
    assert max(single) == pytest.approx((1 + a) / 1.5)  # one annotator: 0.75
    (item,) = _items(j.community_alignment_fr(many[0], _rng(1), {}), "fr")
    d = _dist(item)
    # the label of "Réponse A" follows its shuffled position
    k = item.state.split("Réponse A")[0].count("Réponse ")
    assert d[f"response_{k}"] == pytest.approx((6 + a) / 10.5)
    assert "annotateurs" in item.question.instructions
    few = j.community_alignment_rows([_ca("a"), _ca("b")])
    assert len(few) == 1 and json.loads(few[0]["target"])[:2] == pytest.approx([0.45, 0.45])


def test_votes_on_one_pair_are_aggregated_whatever_the_side():
    a, b = _conv("Hi", "AAA"), _conv("Hi", "BBB")
    battles = [
        {"id": "1", "session": "s2", "winner": "model_a", "lang": "en", "conv_a": a, "conv_b": b},
        {"id": "2", "session": "s1", "winner": "model_b", "lang": "en", "conv_a": b, "conv_b": a},
        {"id": "3", "session": "s3", "winner": "tie", "lang": "en", "conv_a": b, "conv_b": a},
    ]
    (row,) = j.aggregate_battles(battles)
    assert row["session"] == "s1" and json.loads(row["votes"]) == [2, 0, 1, 0]
    (item,) = _items(j.lmarena(row, _rng(), {}))
    d = _dist(item)
    side = "assistant_1" if item.state.index("AAA") < item.state.index("BBB") else "assistant_2"
    assert d[side] == pytest.approx((2 + 0.125) / 3.5) and d["tie"] == pytest.approx(1.125 / 3.5)
    # one weak vote (tie / both bad) is a low-confidence target
    (item,) = _items(j.lmarena({**battles[2], "votes": None}, _rng(), {}))
    assert max(item.target) == pytest.approx((1 + 1.5 / 4) / 2.5) and _gold(item) == "tie"
    react = [{"id": str(i), "opening": "o", "question": "Q ?", "answer": "R.", "liked": like}
             for i, like in enumerate([True, True, False])]  # fmt: skip
    (row,) = j.aggregate_reactions(react)
    (item,) = _items(j.comparia_reactions(row, _rng(), {}), "fr")
    assert _dist(item)["yes"] == pytest.approx((2 + 0.2) / 3.4)


def test_english_conversations_in_comparia_get_the_english_item():
    en = _conv(
        "What is the capital of France and why is it the one that the people chose?",
        "The capital of France is Paris, and it has been the capital for a long time.",
    )
    row = {"id": "v", "opening": "x", "conv_a": en, "conv_b": en.replace("Paris", "Paris!"),
           "winner": "a"}  # fmt: skip
    (item,) = _items(j.comparia_votes(row, _rng(), {}), "en")
    assert item.lang == "en" and item.question.instructions.startswith("Over the whole")


# --- dices / social chem / mic / x-stance ---------------------------------------------------------


def test_dices_rater_shares():
    frame = pd.DataFrame(
        {
            "item_id": ["1"] * 6,
            "context": ["USER: hi\nLAMDA: hello\nUSER: smack him"] * 6,
            "response": ["I'm not picking up on your vibe."] * 6,
            **{q: ["Yes", "No", "No", "Unsure", "No", "Yes"] for q in j.DICES_QUESTIONS},
        }
    )
    rows = j.dices_rows(frame, "350")
    assert len(rows) == len(j.DICES_QUESTIONS)
    (item,) = _items(j.dices(rows[0], _rng(), {}))
    assert _dist(item)["yes"] == pytest.approx(2.5 / 6) and "CHATBOT:" in item.state


def test_social_chem_merges_breakdowns_of_one_rot():
    frame = pd.DataFrame(
        {
            "rot-id": ["r1", "r1"],
            "situation": ["losing trust in my friend"] * 2,
            "action": ["causing people to lose trust in you"] * 2,
            "rot": ["It's bad to lose trust."] * 2,
            "action-moral-judgment": ["-1.0", "-2.0"],
            "rot-agree": ["4.0", "3.0"],
        }
    )
    (row,) = j.social_chem_rows(frame)
    types = set()
    for seed in range(10):
        (item,) = _items(j.social_chem(row, _rng(seed), {}))
        types.add(item.question.instructions)
        assert sorted(p for p in item.target if p) == pytest.approx([0.5, 0.5])
    assert len(types) == 2


def test_mic_agreement_and_severity():
    row = {"Q": "Am I a bad BF?", "A": "Talk to him.", "rot": "Communicate honestly.",
           "A_agrees": 2, "severity": 3}  # fmt: skip
    kinds = {}
    for seed in range(20):
        (item,) = _items(j.mic(row, _rng(seed), {}))
        kinds[item.question.type] = _gold(item)
    assert kinds == {"choice": "agrees", "score": "2"}


def test_x_stance_noul_and_its_negation():
    row = {"question": "Êtes-vous favorable ?", "comment": "Oui, c'est indispensable.",
           "label": "FAVOR"}  # fmt: skip
    golds = set()
    for seed in range(20):
        (item,) = _items(j.x_stance_fr(row, _rng(seed), {}), "fr")
        favour_q = "favorable à la proposition" in item.question.instructions
        assert _gold(item) == ("yes" if favour_q else "no")
        golds.add(_gold(item))
    assert golds == {"yes", "no"}
