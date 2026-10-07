"""The mix-v7 typed-decision sources (tjev.data.sources.jev_v7) on hand-made rows: schema
(parse_item), label mapping, soft targets, licence fields, the tasksource family filter and
quotas, and the disjoint selections of the expansions. No download."""

import json

import numpy as np
import pytest

from tjev.data.item import LICENSE_USES, parse_item
from tjev.data.licences import OPEN, licence_of, license_use_of
from tjev.data.mix import block_of_source
from tjev.data.sources import ALL_SOURCES
from tjev.data.sources import jev_v7 as j


def _items(rows, lang="en", source="t"):
    return [
        parse_item({"id": "t", "source": source, "family": "t", "lang": lang, **r}) for r in rows
    ]


def _gold(item):
    return item.question.labels[item.gold]


def _rng():
    return np.random.default_rng(0)


# --- registry ------------------------------------------------------------------------------


def test_sources_are_registered_with_block_licence_and_use():
    assert set(j.JEV_V7_SOURCES) <= set(ALL_SOURCES)
    blocks = {
        "openjev_x": "jev",
        "openjev_ctrl": "jev",
        "certo_x": "jev",
        "procedural_x": "targeted",
        "eikos_x": "targeted",
        "kaivoss_s1": "jev",
        "n4ze3m_td": "jev",
        "helmo_td": "jev",
        "mlagnajit_td": "jev",
        "tasksource_jev": "breadth",
    }
    assert set(blocks) == set(j.JEV_V7_SOURCES)
    for name, source in j.JEV_V7_SOURCES.items():
        assert block_of_source(name) == blocks[name]
        assert source.license_use in LICENSE_USES
        assert source.commercial_ok == (source.license_use == "commercial")
        card, redistributable, note = licence_of(name)
        assert card != "not recorded" and note, name
        assert license_use_of(name) == source.license_use
        if source.license_use == "commercial":
            assert redistributable in OPEN, name
    # expansions keep their held-out rows in the v6 source
    for name in ("openjev_x", "certo_x", "procedural_x", "eikos_x"):
        assert j.JEV_V7_SOURCES[name].train_only
    for name in ("openjev_ctrl", "n4ze3m_td", "mlagnajit_td", "tasksource_jev"):
        assert j.JEV_V7_SOURCES[name].eval_split == "validation"
    for name in ("kaivoss_s1", "helmo_td"):
        assert j.JEV_V7_SOURCES[name].group is not None


# --- Open-Jev expansions ---------------------------------------------------------------------


def _oj(i, source="gen-v1", group=None):
    return {"id": f"r{i}", "group_id": group or f"g{i}", "source": source}


def test_openjev_x_leaves_out_v6_draws_and_v11_rows_and_caps_wanli_and_generators():
    v11 = [_oj(i) for i in range(10)] + [_oj(100 + i, "wanli-decisions-v1") for i in range(400)]
    v6_ids = {"r0", "r1", "r2"}
    expansion = [_oj(i) for i in range(5)] + [_oj(1000 + i, "snake-v1") for i in range(20)]
    old = j.OPENJEV_X_WANLI, j.OPENJEV_X_PER_SOURCE
    j.OPENJEV_X_WANLI, j.OPENJEV_X_PER_SOURCE = 7, 6
    try:
        out = j.openjev_x_select(v11, v6_ids, expansion, {r["id"] for r in v11})
    finally:
        j.OPENJEV_X_WANLI, j.OPENJEV_X_PER_SOURCE = old
    ids = [r["id"] for r in out]
    assert len(ids) == len(set(ids))
    assert not set(ids) & v6_ids
    assert {f"r{i}" for i in range(3, 10)} <= set(ids)  # the other v1.1 rows
    wanli = [r for r in out if r["source"].startswith("wanli")]
    assert len(wanli) == 7
    keep = min(1.0, 0.10 / 0.90 * 10 / 400)  # v6's WANLI share: these never were in its pool
    assert all(j._hash01(r["group_id"]) >= keep for r in wanli)
    snake = [r for r in out if r["source"] == "snake-v1"]
    assert len(snake) == 6  # per-generator cap; expansion rows already in v1.1 left out


def test_openjev_rows_go_through_the_v6_adapter():
    row = {
        "kind": "noul",
        "options": ["no: not owed", "yes: owed"],
        "target": [0.25, 0.75],
        "state_json": json.dumps({"email": {"body": "Amount owed: EUR 5."}}),
        "question": "Does the body state an amount owed?",
        "source": "mailroom-control-v1",
    }
    (item,) = _items(j.JEV_V7_SOURCES["openjev_ctrl"].adapter(row, _rng(), {}))
    assert item.question.labels == ("no", "yes")
    assert item.target == (0.25, 0.75)
    assert item.question.descriptions == ("not owed", "owed")
    assert '"body"' in item.state


# --- certo -----------------------------------------------------------------------------------


def _certo(i, family, state=None):
    q = {"type": "choice", "instructions": "Pick", "options": [{"id": "a"}, {"id": "b"}]}
    return {
        "record_id": f"c{i}",
        "task_family": family,
        "state": state or f"state {i}",
        "question": json.dumps(q),
    }


def test_certo_x_is_disjoint_from_v6_distinct_and_family_capped():
    rows = [_certo(i, "math") for i in range(100)] + [_certo(200 + i, "law") for i in range(10)]
    rows.append(_certo(999, "law", state="state 200"))  # same decision as c200
    v6 = {"c0", "c1", "c200"}
    out = j.certo_x_select(rows, v6, n=50)  # 8% per family: 4 rows each
    ids = [r["record_id"] for r in out]
    assert not set(ids) & v6
    assert "c999" not in ids  # a duplicate of a v6 decision is not a new one
    fam = [r["task_family"] for r in out]
    assert fam.count("math") == 4 and fam.count("law") == 4


# --- eikos -----------------------------------------------------------------------------------


def test_eikos_x_keeps_only_the_teacher_choice_and_score_rows_v6_left_out():
    base = {"lang": "English", "label_source": "teacher", "question_type": "choice"}
    assert j.eikos_x_keep(base)
    assert j.eikos_x_keep({**base, "question_type": "score"})
    assert not j.eikos_x_keep({**base, "question_type": "noul"})  # in v6's eikos
    assert not j.eikos_x_keep({**base, "label_source": "program"})  # in v6's eikos
    assert not j.eikos_x_keep({**base, "lang": "Spanish"})


# --- kaivoss -----------------------------------------------------------------------------------


def _kv_prompt(options, question="Should this alert be escalated?"):
    lines = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
    return (
        "<state>\n2026-09-21 ERROR AuthService: MFA failure\nline two\n</state>\n\n"
        f"Question: {question}\nOptions:\n{lines}\n\nAnswer with one letter.\nAnswer:"
    )


def test_kaivoss_noul_choice_score_and_soft_only_when_ambiguous():
    noul = {
        "prompt": _kv_prompt(["no - Not warranted.", "yes - Warranted."]),
        "target": [0.3, 0.7],
        "qtype": "noul",
        "ambiguity": "borderline",
        "domain": "security alert",
    }
    (item,) = _items(j.kaivoss(noul, _rng(), {}))
    assert item.question.labels == ("no", "yes") and item.target == pytest.approx((0.3, 0.7))
    assert item.question.descriptions == ("Not warranted.", "Warranted.")
    assert item.state.startswith("2026-09-21") and "line two" in item.state
    assert item.question.instructions == "Should this alert be escalated?"
    clear = {**noul, "ambiguity": "clear", "target": [0.0001, 0.9999]}
    (item,) = _items(j.kaivoss(clear, _rng(), {}))
    assert item.target == (0.0, 1.0)  # clear rows are one-hot
    choice = {
        **noul,
        "qtype": "choice",
        "prompt": _kv_prompt(["auth - Login.", "mfa_error - MFA.", "system - Other."]),
        "target": [0.1, 0.8, 0.1],
    }
    (item,) = _items(j.kaivoss(choice, _rng(), {}))
    assert item.question.labels == ("auth", "mfa_error", "system") and _gold(item) == "mfa_error"
    score = {**choice, "qtype": "score", "prompt": _kv_prompt(["low - L", "mid - M", "high - H"])}
    (item,) = _items(j.kaivoss(score, _rng(), {}))
    assert item.question.labels == ("0", "1", "2")
    assert item.question.descriptions == ("low: L", "mid: M", "high: H")
    assert j.kaivoss({**noul, "prompt": "no state"}, _rng(), {}) == []
    assert j.kaivoss({**noul, "target": [1.0]}, _rng(), {}) == []


# --- n4ze3m -----------------------------------------------------------------------------------


def test_n4ze3m_one_item_per_question_soft_teacher_when_it_agrees():
    row = {
        "state": json.dumps({"text": "Oh great, another meeting.", "hashtags": ["#Work"]}),
        "state_is_json": True,
        "generator": "v1-general",
        "questions": json.dumps(
            {
                "tone": {
                    "type": "choice",
                    "instructions": "Tone?",
                    "criteria": {"sincere": "Praise", "sarcastic": "Mock"},
                },
                "level": {"type": "score", "instructions": "How negative?", "criteria": ["a", "b"]},
                "tag": {
                    "type": "noul",
                    "instructions": "Hashtag?",
                    "criteria": {"true": "Has one", "false": "None"},
                },
                "bad": {"type": "noul", "instructions": "Disagrees?", "criteria": {}},
            }
        ),
        "gold": json.dumps({"tone": "sarcastic", "level": 1, "tag": True, "bad": False}),
        "teacher": json.dumps(
            {
                "tone": {"probabilities": {"sincere": 0.05, "sarcastic": 0.95}},
                "level": {"probabilities": {"0": 0.2, "1": 0.8}},
                "tag": {"noul": 0.9},
                "bad": {"noul": 0.8},  # disagrees with gold: gold one-hot
            }
        ),
    }
    items = _items(j.n4ze3m(row, _rng(), {}))
    assert [i.question.type for i in items] == ["choice", "score", "noul", "noul"]
    tone, level, tag, bad = items
    assert _gold(tone) == "sarcastic" and tone.target == pytest.approx((0.05, 0.95))
    assert level.target == pytest.approx((0.2, 0.8))
    assert tag.target == pytest.approx((0.1, 0.9)) and tag.question.descriptions == (
        "None",
        "Has one",
    )
    assert bad.target == (1.0, 0.0)
    assert '"hashtags"' in tone.state


# --- helmo --------------------------------------------------------------------------------------


def _helmo(q, g):
    return {
        "state": "A situation.",
        "questions": json.dumps({"q1": q}),
        "gold": json.dumps({"q1": g}),
    }


def test_helmo_types_and_fractional_score_means():
    (item,) = _items(
        j.helmo(
            _helmo({"type": "noul", "instructions": "True?", "criteria": {}}, {"noul": 0.15}),
            _rng(),
            {},
        )
    )
    assert item.question.labels == ("no", "yes") and item.target == pytest.approx((0.85, 0.15))
    (item,) = _items(
        j.helmo(
            _helmo({"type": "noul", "instructions": "T?", "criteria": {}}, {"noul": 1.0}),
            _rng(),
            {},
        )
    )
    assert item.target == (0.0, 1.0)
    choice = {"type": "choice", "instructions": "Main issue?", "criteria": {"A": "x", "B": "y"}}
    (item,) = _items(j.helmo(_helmo(choice, {"probabilities": {"B": 1.0}}), _rng(), {}))
    assert _gold(item) == "B" and item.target == (0.0, 1.0)
    score = {"type": "score", "instructions": "Rate", "criteria": ["l", "m", "h", "vh"]}
    (item,) = _items(j.helmo(_helmo(score, {"mean": 2.25, "variance": 0.4}), _rng(), {}))
    assert item.target == pytest.approx((0.0, 0.0, 0.75, 0.25))
    (item,) = _items(j.helmo(_helmo(score, {"mean": 3.0, "variance": 0.0}), _rng(), {}))
    assert item.target == (0.0, 0.0, 0.0, 1.0)
    assert j.helmo(_helmo(choice, {"probabilities": {"Z": 1.0}}), _rng(), {}) == []


# --- m-lagnajit --------------------------------------------------------------------------------


def test_mlagnajit_human_gold_with_the_calibrated_teacher():
    choice = {
        "type": "choice",
        "instructions": "Which intent?",
        "state": "how much have i spent on food",
        "options": ["travel notification", "spending history", "no"],
        "gold": "spending history",
        "teacher_probs": [0.1, 0.85, 0.05],
        "source": "clinc/clinc_oos",
    }
    (item,) = _items(j.mlagnajit(choice, _rng(), {}))
    assert _gold(item) == "spending history" and max(item.target) == 1.0
    assert item.teacher == pytest.approx((0.1, 0.85, 0.05))
    assert j.mlagnajit(choice, _rng(), {})[0]["family"] == "mlagnajit_clinc_oos"
    noul = {
        **choice,
        "type": "noul",
        "options": ["false", "true"],
        "gold": "false",
        "teacher_probs": [0.7, 0.3],
    }
    (item,) = _items(j.mlagnajit(noul, _rng(), {}))
    assert item.question.labels == ("no", "yes") and _gold(item) == "no"
    assert item.teacher == pytest.approx((0.7, 0.3))
    assert j.mlagnajit({**choice, "gold": "absent"}, _rng(), {}) == []


# --- tasksource-jev ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "split", "rule"),
    [
        ("procedural-typed-decisions/arithmetic", "train", "procedural"),
        ("HelpSteer3/preference", "train", "helpsteer"),
        ("HelpSteer2/coherence", "train", "helpsteer"),
        ("WANLI", "train", "wanli"),
        ("glue/mnli", "train", "mnli"),
        ("chaos-mnli-ambiguity/votes", "train", "mnli"),
        ("multilingual/xnli", "train", "mnli"),
        ("dataset_train_nli", "train", "mnli"),
        ("snli", "train", "snli"),
        ("counterfactually-augmented-snli", "train", "snli"),
        ("super_glue/boolq", "train", "boolq"),
        ("boolq-natural-perturbations", "train", "boolq"),
        ("paws/labeled_final", "train", "paws"),
        ("multilingual/paws-x/fr", "train", "paws"),
        ("contract-nli/contractnli_a/seg", "train", "contract_nli"),
        ("glue/stsb", "train", "glue_sst_stsb"),
        ("banking77", "train", "intents"),
        ("clinc_oos/plus", "train", "intents"),
        ("go_emotions/simplified", "train", "go_emotions"),
        ("dbpedia_14/dbpedia_14", "train", "dbpedia"),
        ("vitaminc", "train", "vitaminc"),
        ("scruples/verdict_votes", "train", "judge_v7"),
        ("SChem_Disagreement", "train", "judge_v7"),
        ("oasst2/quality", "train", "judge_v7"),
        ("oasst2_pairwise_rlhf_reward", "train", "judge_v7"),
        ("webgpt_comparisons", "train", "judge_v7"),
        ("summarize_from_feedback/comparisons", "train", "judge_v7"),
        ("multilingual/x-stance", "train", "x_stance"),
        ("sharc", "train", "reading_v7"),
        ("temporal-nli", "train", "reading_v7"),
        ("mc_taco", "train", "reading_v7"),
        ("lex_glue/case_hold", "train", "reading_v7"),
        ("lex_glue/ledgar", "train", "reading_v7"),
        ("ambient", "train", "eval_only"),
        ("chatbot_arena_conversations", "train", "eval_only"),
        ("anli/a1", "dev", "anli_eval"),
        ("anli/a2", "test", "anli_eval"),
        ("multilingual/clue/afqmc", "train", "multilingual"),
        ("multilingual/language-identification", "train", "language_id"),
        ("multilingual/MLMA_hate_speech", "train", "tweets_mlma"),
        # kept
        ("anli/a1", "train", None),
        ("lex_glue/scotus", "train", None),
        ("hellaswag", "train", None),
        ("civil_comments/toxicity_share", "train", None),
        ("multilingual/masakhanews/fra", "train", None),
        ("multilingual/xcsr/X-CSQA-en", "train", None),
        ("multilingual/xlwic/xlwic_fr_fr", "train", None),
    ],
)
def test_tasksource_drop_rules(source, split, rule):
    assert j.ts_drop_rule(source, split) == rule


def test_tasksource_language_and_family():
    assert j.ts_lang("multilingual/masakhanews/fra") == "fr"
    assert j.ts_lang("multilingual/xcsr/X-CODAH-fr") == "fr"
    assert j.ts_lang("hellaswag") == "en"
    assert j.ts_family("civil_comments/toxicity_share") == "ts_civil_comments"
    assert j.ts_family("multilingual/masakhanews/fra") == "ts_masakhanews"


def test_tasksource_quotas_follow_sqrt_and_cap_each_source():
    counts = {"big": 100000, "mid": 10000, "small": 50}
    q = j.ts_quotas(counts, total=6000, share=0.5)
    assert abs(sum(q.values()) - 6000) <= 3
    assert q["small"] == 50  # fewer rows than its quota: all of them
    assert q["big"] <= 3000  # the share cap
    assert (
        q["mid"] / q["big"] == pytest.approx((10000 / 100000) ** 0.5, rel=0.05) or q["big"] == 3000
    )
    q = j.ts_quotas({"a": 10, "b": 20}, total=1000, share=1.0)
    assert q == {"a": 10, "b": 20}


def _ts(kind, options, target, **kw):
    return {
        "state": "My body cast a shadow over the grass.",
        "question": "What was the cause?",
        "kind": kind,
        "options": options,
        "target": target,
        "source": "super_glue/copa",
        "license": "bsd-2-clause",
        "license_use": "commercial",
        **kw,
    }


def test_tasksource_rows_map_kinds_targets_and_per_row_licences():
    (item,) = _items(j.tasksource_jev(_ts("noul", [], [0.7]), _rng(), {}))
    assert item.question.labels == ("no", "yes") and item.target == pytest.approx((0.3, 0.7))
    assert item.license == "bsd-2-clause" and item.license_use == "commercial"
    (item,) = _items(j.tasksource_jev(_ts("noul", [], [1.0]), _rng(), {}))
    assert item.target == (0.0, 1.0)
    rows = j.tasksource_jev(
        _ts(
            "choice",
            ["The sun was rising.", "The grass was cut."],
            [1.0, 0.0],
            license_use="non-commercial",
        ),
        _rng(),
        {},
    )
    (item,) = _items(rows)
    assert item.question.labels == ("option_1", "option_2")
    assert item.question.descriptions == ("The sun was rising.", "The grass was cut.")
    assert _gold(item) == "option_1" and item.license_use == "non-commercial"
    (item,) = _items(
        j.tasksource_jev(_ts("choice", ["entailment", "neutral"], [0.6, 0.4]), _rng(), {})
    )
    assert item.question.labels == ("entailment", "neutral")
    assert item.target == pytest.approx((0.6, 0.4))
    (item,) = _items(
        j.tasksource_jev(
            _ts("score", ["1", "2", "3"], [0.0, 0.6, 0.4], license_use="?"), _rng(), {}
        )
    )
    assert item.question.type == "score" and item.target == pytest.approx((0.0, 0.6, 0.4))
    assert item.question.descriptions == ("1", "2", "3")
    assert item.license_use == "unspecified"  # unknown values
    (item,) = _items(
        j.tasksource_jev(
            _ts("noul", [], [0.0], state='{"ticket": {"channel": "email"}}'), _rng(), {}
        )
    )
    assert '\n "ticket"' in item.state  # JSON states rendered like the other structured states
    assert j.tasksource_jev(_ts("choice", ["a"], [1.0]), _rng(), {}) == []
    assert j.tasksource_jev(_ts("noul", [], [1.5]), _rng(), {}) == []
    assert j.tasksource_jev(_ts("multilabel", ["a", "b"], [1, 1]), _rng(), {}) == []


def test_tasksource_french_rows_are_french():
    rows = j.tasksource_jev(
        _ts("noul", [], [0.0], source="multilingual/masakhanews/fra"), _rng(), {}
    )
    assert rows[0]["lang"] == "fr"
