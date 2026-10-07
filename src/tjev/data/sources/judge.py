"""mix-v7 judge sources (block "judge"; x-stance in "analysis"): human preferences, ratings
and moral / safety judgements recast as decisions.

==========================  =================================================  ===============
source                      what it adds                                       targets
==========================  =================================================  ===============
``helpsteer3_x``            HelpSteer3 rows mix-v6 left out (English, code,    annotators (soft)
                            French)
``oasst2``                  OpenAssistant rater labels (quality, helpfulness,  rater means /
                            fails-task) and sibling rankings                   shares (soft)
``lmarena_140k``            LMArena battles: A / B / tie / both bad            one vote
``webgpt``                  WebGPT answer comparisons, graded margins          margin (soft)
``summarize_fb``            TL;DR summary comparisons and axis ratings         votes (soft)
``scruples``                AITA anecdotes: who is in the wrong                votes (soft, ≥5)
``comparia_arena``          compar:IA turn preferences (French)                one vote
``comparia_votes``          compar:IA conversation votes (French, multi-turn)  one vote
``comparia_reactions``      compar:IA thumbs up / down on a reply (French)     one reaction
``community_alignment_fr``  Community Alignment, French: 4 responses           votes (soft, ≥10)
``dices``                   DICES conversation safety ratings                  rater shares
``social_chem``             Social-Chem-101 action judgements, RoT agreement   one or several
``mic``                     Moral Integrity Corpus: answer vs rule of thumb    one annotation
``x_stance_fr``             x-stance French comments: in favour or not         exact (author)
==========================  =================================================  ===============

Two-sided items (A vs B) show the sides in a random order, so the position carries no
signal; option order is shuffled by the renderer anyway. Conversations over
``MAX_CHARS`` are left out. Every source sets ``Source.block`` and
``Source.license_use``; card licences are in ``LICENCES``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from tjev.data.lang import guess_lang
from tjev.data.sources.base import CACHE, Source
from tjev.data.sources.contenders import _one_hot_or_soft
from tjev.data.sources.targeted import (
    _conversation,
    _dataset,
    _rng_of,
    helpsteer3,
)

MAX_CHARS = 10000
# mix-v7.5: single votes are noisy (docs/data.md, mix-v7.5). Votes on one item (the same
# prompt and the same replies, whatever the side they were shown on) are aggregated, and
# every vote-derived target is smoothed toward uniform: p = (votes + alpha / k) / (n + alpha)
# over the k options. A single decisive vote of 4 options gives 0.75; a single tie / both-bad
# / equal vote (an uninformative margin) gets VOTE_ALPHA_WEAK (0.55); a single thumbs up or
# down gets 0.857 (outside JevBench's 0.2–0.8 noul band); n agreeing votes approach 1.
VOTE_ALPHA = 0.5
VOTE_ALPHA_WEAK = 1.5
VOTE_ALPHA_NOUL = 0.4
WEAK_VOTES = frozenset({"tie", "both_bad", "both_good", "equal"})
PARQUET = "refs/convert/parquet"  # the Hub's parquet export of script-only datasets


def vote_target(labels: list[str], counts: list[float], alpha: float = VOTE_ALPHA) -> dict:
    """A soft ``target`` from vote counts over ``labels`` (smoothed toward uniform; a single
    vote for a weak option gets ``VOTE_ALPHA_WEAK``)."""
    n, k = float(sum(counts)), len(labels)
    if n <= 0:
        raise ValueError("no votes")
    if n == 1 and labels[max(range(k), key=lambda i: counts[i])] in WEAK_VOTES:
        alpha = max(alpha, VOTE_ALPHA_WEAK)
    return {"target": {lab: (c + alpha / k) / (n + alpha) for lab, c in zip(labels, counts,
                                                                           strict=True)}}  # fmt: skip


def lang_of(text: str, tagged: str) -> str:
    """The language of a conversation by the stop-word guess (tjev.data.lang), else its tag:
    compar:IA has English conversations and LMArena French ones under the other tag."""
    guess = guess_lang(text)
    return guess if guess in ("en", "fr") else tagged


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", str(text or "").lower()).strip()


def _parquet(repo: str, path: str, columns=None, revision: str | None = PARQUET, filters=None):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo, path, repo_type="dataset", revision=revision)
    return pq.read_table(local, columns=columns, filters=filters)


def interpolated(mean_level: float, levels: int) -> list[float]:
    """A mean rating on levels 0..levels-1 as a two-level distribution whose expected level
    is the mean (tasksource's convention: 2.4 puts 0.6 on 2 and 0.4 on 3)."""
    x = min(max(float(mean_level), 0.0), levels - 1.0)
    x = round(x, 6)
    lo = math.floor(x)
    frac = x - lo
    dist = [0.0] * levels
    dist[lo] += 1.0 - frac
    if frac > 0:
        dist[lo + 1] += frac
    return dist


def histogram(values: list[int], levels: int) -> list[float]:
    """Votes on levels 0..levels-1 as their distribution."""
    n = len(values)
    return [sum(v == i for v in values) / n for i in range(levels)]


def _noul_target(p: float) -> dict:
    p = min(max(float(p), 0.0), 1.0)
    if p in (0.0, 1.0):
        return {"expected": "yes" if p == 1.0 else "no"}
    return {"target": {"no": 1.0 - p, "yes": p}}


# --- HelpSteer3 extension (nvidia/HelpSteer3, preference) ---------------------------------
# mix-v6's helpsteer3 kept English general / stem rows up to 10k characters and trained on
# the first 9,600 of its seeded permutation (cap 8000 × 1.2). helpsteer3_x takes the other
# English general / stem rows, the code domain and the French multilingual rows. Its
# held-out rows are the validation rows of the code / French part only (the English
# validation rows are mix-v6's held-out set).

HS3_V6_SAMPLE = 9600  # int(cap 8000 * 1.2): the train indices mix-v6's builder adapted


def _hs3_rows(split: str) -> list[dict]:
    import gzip

    from huggingface_hub import hf_hub_download

    path = hf_hub_download("nvidia/HelpSteer3", f"preference/{split}.jsonl.gz", repo_type="dataset")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _hs3_slim(r: dict, lang: str) -> dict | None:
    context = r["context"] if isinstance(r["context"], list) else []
    size = sum(len(str(t.get("content", ""))) for t in context)
    size += len(r["response1"]) + len(r["response2"])
    scores = [int(p["score"]) for p in r.get("individual_preference") or []]
    if not context or size > MAX_CHARS or not scores:
        return None
    return {
        "context": json.dumps(context, ensure_ascii=False),
        "response1": r["response1"],
        "response2": r["response2"],
        "scores": scores,
        "domain": r["domain"],
        "lang": lang,
    }


def _hs3_v6_used() -> set[int]:
    """The positions (in targeted.helpsteer3_load("train")) mix-v6's builder adapted:
    its rng drew the held-out permutation first, then the train one."""
    import hashlib

    import numpy as np

    from tjev.data.sources.targeted import helpsteer3_load

    seed = int(hashlib.sha256(b"helpsteer3").hexdigest()[:8], 16)
    rng = np.random.default_rng([0, seed])
    rng.permutation(len(helpsteer3_load("validation")))
    return set(rng.permutation(len(helpsteer3_load("train")))[:HS3_V6_SAMPLE].tolist())


def helpsteer3_x_load(split: str):
    used = _hs3_v6_used() if split == "train" else set()
    rows, english = [], 0
    for r in _hs3_rows(split):
        language = str(r.get("language") or "").lower()
        if language == "english":  # mix-v6's loader kept exactly these rows, in this order
            slim = _hs3_slim(r, "en")
            if slim is not None:
                position, english = english, english + 1
                if split != "train" or position in used:
                    slim = None  # mix-v6 trained on it / holds it out
        elif r.get("domain") == "code":
            slim = _hs3_slim(r, "en")
        elif r.get("domain") == "multilingual" and language == "french":
            slim = _hs3_slim(r, "fr")
        else:
            slim = None
        if slim is not None:
            rows.append(slim)
    return _dataset(rows)


def helpsteer3_x(row, rng, meta):
    return [{**item, "lang": row["lang"]} for item in helpsteer3(row, rng, meta)]


# --- OpenAssistant oasst2 ---------------------------------------------------------------
# One row per message of a tree: labels {name, value (rater mean in [0, 1]), count}; the
# assistant replies of one parent carry a consensus rank (0 = best). Kept: English and
# French, reviewed, not deleted, conversations up to MAX_CHARS. Per assistant reply one
# item: helpfulness or quality on 5 levels (the raters' mean, interpolated), or whether it
# fails the task (the raters' share); per parent with ≥2 ranked replies one item: which of
# the best and another reply ranks higher.

OASST_LANGS = ("en", "fr")
OASST_LEVELS = {
    "en": ["Very poor", "Poor", "Average", "Good", "Excellent"],
    "fr": ["Très mauvais", "Mauvais", "Moyen", "Bon", "Excellent"],
}
OASST_TEXT = {
    "en": {
        "helpfulness": "How helpful is the last assistant reply, as human raters judged it?",
        "quality": "How good is the last assistant reply overall, as human raters judged it?",
        "fails_task": "Does the last assistant reply fail to do what the user asked?",
        "false": "No, it does what was asked",
        "true": "Yes, it fails the task",
        "rank": "Which reply to the last user message did human raters rank higher?",
        "reply": "Reply",
        "better": "Reply {k} is better",
    },
    "fr": {
        "helpfulness": "Selon des évaluateurs humains, la dernière réponse de l'assistant "
        "est-elle utile ?",
        "quality": "Selon des évaluateurs humains, quelle est la qualité globale de la "
        "dernière réponse de l'assistant ?",
        "fails_task": "La dernière réponse de l'assistant échoue-t-elle à faire ce que "
        "l'utilisateur demandait ?",
        "false": "Non, elle fait ce qui était demandé",
        "true": "Oui, elle échoue",
        "rank": "Quelle réponse au dernier message de l'utilisateur les évaluateurs "
        "humains ont-ils classée plus haut ?",
        "reply": "Réponse",
        "better": "La réponse {k} est meilleure",
    },
}
OASST_ROLES = {"prompter": "user", "assistant": "assistant"}


def _labels_of(message: dict) -> dict[str, tuple[float, int]]:
    labels = message.get("labels") or {}
    names, values, counts = (
        labels.get("name") or [],
        labels.get("value") or [],
        labels.get("count") or [],
    )
    return {n: (float(v), int(c)) for n, v, c in zip(names, values, counts, strict=False)}


def oasst2_rows(messages: list[dict]) -> list[dict]:
    """oasst2 messages → label rows (one per assistant reply) and rank rows (one per
    parent with ≥2 ranked replies)."""
    by_id = {m["message_id"]: m for m in messages}
    children = defaultdict(list)
    for m in messages:
        if m.get("parent_id"):
            children[m["parent_id"]].append(m)

    def ok(m: dict) -> bool:
        return (
            not m.get("deleted")
            and m.get("review_result") is not False
            and m.get("lang") in OASST_LANGS
        )

    def path(m: dict) -> list[dict] | None:
        turns = []
        while m is not None:
            if not ok(m):
                return None
            turns.append({"role": OASST_ROLES.get(m["role"], m["role"]), "content": m["text"]})
            m = by_id.get(m.get("parent_id")) if m.get("parent_id") else None
        return turns[::-1]

    rows = []
    for m in messages:
        if m["role"] != "assistant" or not ok(m):
            continue
        turns = path(m)
        if turns is None or sum(len(t["content"]) for t in turns) > MAX_CHARS:
            continue
        labels = _labels_of(m)
        rng = _rng_of(f"oasst2:{m['message_id']}")
        u = rng.random()
        if u < 0.45 and labels.get("helpfulness", (0, 0))[1] >= 2:
            name = "helpfulness"
        elif u < 0.7 and labels.get("quality", (0, 0))[1] >= 2:
            name = "quality"
        elif labels.get("fails_task", (0, 0))[1] >= 3:
            name = "fails_task"
        else:
            continue
        value, count = labels[name]
        if name == "fails_task" and value == 0.0 and rng.random() < 0.6:
            continue  # most replies are fine: keep the yes/no prior less skewed
        rows.append(
            {
                "kind": name,
                "turns": json.dumps(turns, ensure_ascii=False),
                "reply_b": "",
                "value": value,
                "count": count,
                "lang": m["lang"],
                "tree": m.get("message_tree_id") or "",
            }
        )
    for parent_id, replies in children.items():
        parent = by_id.get(parent_id)
        ranked = [
            r for r in replies if r["role"] == "assistant" and ok(r) and r.get("rank") is not None
        ]
        if parent is None or len(ranked) < 2:
            continue
        turns = path(parent)
        if turns is None:
            continue
        ranked.sort(key=lambda r: r["rank"])
        best = ranked[0]
        others = [r for r in ranked[1:] if r["rank"] != best["rank"]]
        if not others:
            continue
        other = others[int(_rng_of(f"oasst2:{parent_id}").integers(len(others)))]
        size = sum(len(t["content"]) for t in turns) + len(best["text"]) + len(other["text"])
        if size > MAX_CHARS:
            continue
        rows.append(
            {
                "kind": "rank",
                "turns": json.dumps(
                    [*turns, {"role": "assistant", "content": best["text"]}], ensure_ascii=False
                ),
                "reply_b": other["text"],
                "value": 1.0,
                "count": 0,
                "lang": parent["lang"],
                "tree": parent.get("message_tree_id") or "",
            }
        )
    return rows


def oasst2_load(split: str):
    from datasets import load_dataset

    ds = load_dataset("OpenAssistant/oasst2", split=split)
    columns = ["message_id", "parent_id", "text", "role", "lang", "rank", "deleted",
               "review_result", "labels", "message_tree_id"]  # fmt: skip
    return _dataset(oasst2_rows(ds.select_columns(columns).to_list()))


def oasst2(row, rng, meta):
    lang = row["lang"]
    t = OASST_TEXT[lang]
    turns = json.loads(row["turns"])
    if row["kind"] == "rank":
        context, better = turns[:-1], turns[-1]["content"]
        replies = [better, row["reply_b"]]
        first = int(rng.integers(2))  # the better reply's position
        if first == 1:
            replies.reverse()
        state = (
            f"{_conversation(context)}\n\n{t['reply']} 1:\n{replies[0].strip()}\n\n"
            f"{t['reply']} 2:\n{replies[1].strip()}"
        )
        criteria = {f"reply_{k}": t["better"].format(k=k) for k in (1, 2)}
        question = {"type": "choice", "instructions": t["rank"], "criteria": criteria}
        return [{"state": state, "question": question, "expected": f"reply_{first + 1}",
                 "lang": lang, "family": "judge_rank"}]  # fmt: skip
    state = _conversation(turns)
    if row["kind"] == "fails_task":
        question = {"type": "noul", "instructions": t["fails_task"],
                    "criteria": {"false": t["false"], "true": t["true"]}}  # fmt: skip
        return [{"state": state, "question": question, **_noul_target(row["value"]),
                 "lang": lang, "family": "judge_rating"}]  # fmt: skip
    dist = interpolated(row["value"] * 4, 5)
    question = {"type": "score", "instructions": t[row["kind"]], "criteria": OASST_LEVELS[lang]}
    target = _one_hot_or_soft([str(i) for i in range(5)], dist)
    return [{"state": state, "question": question, **target, "lang": lang,
             "family": "judge_rating"}]  # fmt: skip


# --- LMArena arena-human-preference-140k ---------------------------------------------------
# {id, winner (model_a | model_b | tie | both_bad), evaluation_session_id, conversation_a /
# conversation_b ([{role, content: [{type, text, image}]}]), language}. English and French
# text-only battles up to MAX_CHARS. The two sides are shown in a random order. The prompts
# also feed lmarena-ai/PPE-Human-Preference-V1 (eval-only): the 8-gram decontamination
# against PPE drops the shared ones.

LMARENA_REPO = "lmarena-ai/arena-human-preference-140k"
LMARENA_FILES = tuple(f"data/train-0000{i}-of-00007.parquet" for i in range(7))
LMARENA_LABELS = {"model_a": 0, "model_b": 1, "tie": "tie", "both_bad": "both_bad"}


def _arena_turns(conversation) -> list[dict] | None:
    turns = []
    for message in conversation or []:
        parts = message.get("content")
        if isinstance(parts, str):
            text = parts
        else:
            if any((p.get("type") or "text") != "text" or p.get("image") for p in parts or []):
                return None  # images: not text-only
            text = "\n".join(str(p.get("text") or "") for p in parts or [])
        if not text.strip():
            return None
        turns.append({"role": message.get("role") or "user", "content": text})
    return turns or None


def lmarena_load(split: str):
    """One row per (conversation prompt, pair of conversations): the votes of every battle
    on it (sides in a canonical order), the earliest session as its group."""
    battles = []
    columns = ["id", "winner", "evaluation_session_id", "conversation_a", "conversation_b",
               "language"]  # fmt: skip
    for name in LMARENA_FILES:
        table = _parquet(LMARENA_REPO, name, columns=columns, revision=None,
                         filters=[("language", "in", ["en", "fr"])])  # fmt: skip
        for batch in table.to_batches(2000):
            for r in batch.to_pylist():
                a, b = _arena_turns(r["conversation_a"]), _arena_turns(r["conversation_b"])
                if a is None or b is None or r["winner"] not in LMARENA_LABELS:
                    continue
                if sum(len(t["content"]) for t in a + b) > MAX_CHARS:
                    continue
                battles.append(
                    {
                        "id": r["id"],
                        "session": r["evaluation_session_id"] or r["id"],
                        "winner": r["winner"],
                        "lang": r["language"],
                        "conv_a": json.dumps(a, ensure_ascii=False),
                        "conv_b": json.dumps(b, ensure_ascii=False),
                    }
                )
        del table
    return _dataset(aggregate_battles(battles))


def aggregate_battles(battles: list[dict]) -> list[dict]:
    """Battles on the same two conversations (either side) as one row with their votes
    ``[first, second, tie, both_bad]`` (``votes``, JSON)."""
    rows: dict = {}
    for r in battles:
        first, second = sorted((r["conv_a"], r["conv_b"]))
        flip = first != r["conv_a"]
        entry = rows.setdefault((first, second), {**r, "conv_a": first, "conv_b": second,
                                                  "votes": [0, 0, 0, 0]})  # fmt: skip
        entry["session"] = min(entry["session"], r["session"])
        entry["id"] = min(entry["id"], r["id"])
        w = LMARENA_LABELS[r["winner"]]
        k = {"tie": 2, "both_bad": 3}[w] if isinstance(w, str) else (1 - w if flip else w)
        entry["votes"][k] += 1
    return [{**e, "votes": json.dumps(e["votes"])} for e in rows.values()]


ARENA_TEXT = {
    "en": {
        "assistant": "Assistant",
        "with": "Conversation with assistant",
        "question": "Which assistant answered the user better?",
        "better": "Assistant {k} is better",
        "tie": "Both are about equally good",
        "both_bad": "Both are bad",
        "both_good": "Both are good",
        "equal": "Both are about equally good",
    },
    "fr": {
        "assistant": "Assistant",
        "with": "Conversation avec l'assistant",
        "question": "Quel assistant a le mieux répondu à l'utilisateur ?",
        "better": "L'assistant {k} est meilleur",
        "tie": "Les deux se valent",
        "both_bad": "Les deux sont mauvais",
        "both_good": "Les deux sont bons",
        "equal": "Les deux se valent",
    },
}


def two_sides(a: list[dict], b: list[dict], lang: str) -> str:
    """Two conversations sharing their user turns: one prompt and two replies when
    single-turn, else the two conversations one after the other."""
    t = ARENA_TEXT[lang]
    if len(a) == 2 and len(b) == 2 and a[0]["role"] == "user":
        return (
            f"{_conversation(a[:1])}\n\n{t['assistant']} 1:\n{a[1]['content'].strip()}\n\n"
            f"{t['assistant']} 2:\n{b[1]['content'].strip()}"
        )
    return f"{t['with']} 1:\n{_conversation(a)}\n\n{t['with']} 2:\n{_conversation(b)}"


def pairwise(a, b, winner, extra: tuple[str, ...], lang: str, rng, instructions=None) -> dict:
    """A two-sided choice in a random order: ``winner`` is 0 / 1 (the side of ``a`` / ``b``)
    or one of ``extra`` (tie, both bad…)."""
    t = ARENA_TEXT[lang]
    if rng.random() < 0.5:
        a, b = b, a
        winner = 1 - winner if isinstance(winner, int) else winner
    criteria = {"assistant_1": t["better"].format(k=1), "assistant_2": t["better"].format(k=2)}
    criteria |= {e: t[e] for e in extra}
    expected = f"assistant_{winner + 1}" if isinstance(winner, int) else winner
    return {
        "state": two_sides(a, b, lang),
        "question": {
            "type": "choice",
            "instructions": instructions or t["question"],
            "criteria": criteria,
        },
        "expected": expected,
        "lang": lang,
    }


def pairwise_votes(a, b, counts: list[float], extra: tuple[str, ...], lang: str, rng,
                   instructions=None) -> dict:  # fmt: skip
    """:func:`pairwise` with vote counts ``[a, b, *extra]``: a smoothed soft target
    (:func:`vote_target`), the sides in a random order."""
    t = ARENA_TEXT[lang]
    counts = list(counts)
    if rng.random() < 0.5:
        a, b = b, a
        counts[0], counts[1] = counts[1], counts[0]
    labels = ["assistant_1", "assistant_2", *extra]
    criteria = {"assistant_1": t["better"].format(k=1), "assistant_2": t["better"].format(k=2)}
    criteria |= {e: t[e] for e in extra}
    return {
        "state": two_sides(a, b, lang),
        "question": {
            "type": "choice",
            "instructions": instructions or t["question"],
            "criteria": criteria,
        },
        **vote_target(labels, counts),
        "lang": lang,
    }


def _counts(winner, extra: tuple[str, ...]) -> list[int]:
    """One vote as counts ``[a, b, *extra]`` (rows of the mix-v7 loaders)."""
    keys = [0, 1, *extra]
    return [int(k == winner) for k in keys]


def lmarena(row, rng, meta):
    extra = ("tie", "both_bad")
    if row.get("votes"):
        counts = json.loads(row["votes"])
    else:
        counts = _counts(LMARENA_LABELS[row["winner"]], extra)
    a, b = json.loads(row["conv_a"]), json.loads(row["conv_b"])
    lang = lang_of(two_sides(a, b, "en"), row["lang"])
    return [pairwise_votes(a, b, counts, extra, lang, rng)]


# --- WebGPT comparisons (openai/webgpt_comparisons) -----------------------------------------
# {question {full_text}, quotes_k {title, extract}, answer_k, score_k} for k = 0, 1:
# score_0 = −score_1 in [−1, 1], the graded preference for answer 0. Target: the side
# preferred gets |score|, a tie gets 1 − |score| (score 1: one-hot; 0: a tie).

WEBGPT_REPO = "openai/webgpt_comparisons"


def webgpt_load(split: str):
    table = _parquet(WEBGPT_REPO, "default/train/0000.parquet",
                     columns=["question", "quotes_0", "answer_0", "score_0", "quotes_1",
                              "answer_1"])  # fmt: skip
    rows = []
    for r in table.to_pylist():
        question = (r["question"] or {}).get("full_text") or ""
        if not question.strip() or not r["answer_0"].strip() or not r["answer_1"].strip():
            continue
        rows.append(
            {
                "question": question,
                "answer_0": r["answer_0"],
                "answer_1": r["answer_1"],
                "quotes_0": json.dumps(r["quotes_0"] or {}, ensure_ascii=False),
                "quotes_1": json.dumps(r["quotes_1"] or {}, ensure_ascii=False),
                "score": float(r["score_0"]),
            }
        )
    return _dataset(rows)


def _quotes(raw: str, budget: int = 1500) -> str:
    q = json.loads(raw) or {}
    lines = []
    for k, (title, extract) in enumerate(
        zip(q.get("title") or [], q.get("extract") or [], strict=False), 1
    ):
        lines.append(f"[{k}] {title}: {extract.strip()[:budget]}")
    return "\n".join(lines)


def webgpt(row, rng, meta):
    sides = [(row["answer_0"], row["quotes_0"]), (row["answer_1"], row["quotes_1"])]
    m = float(row["score"])
    if rng.random() < 0.5:
        sides, m = sides[::-1], -m
    blocks = []
    for k, (answer, quotes) in enumerate(sides, 1):
        src = _quotes(quotes)
        blocks.append(
            f"Answer {k}:\n{answer.strip()}" + (f"\nSources of answer {k}:\n{src}" if src else "")
        )
    state = f"Question: {row['question'].strip()}\n\n" + "\n\n".join(blocks)
    if len(state) > MAX_CHARS:
        return []
    criteria = {
        "answer_1": "Answer 1 is better",
        "answer_2": "Answer 2 is better",
        "tie": "The two answers are about equally good",
    }
    m = max(-1.0, min(1.0, m))
    probs = [max(m, 0.0), max(-m, 0.0), 1.0 - abs(m)]
    return [
        {
            "state": state,
            "question": {
                "type": "choice",
                "instructions": "Which answer to the question did human raters prefer?",
                "criteria": criteria,
            },
            **_one_hot_or_soft(list(criteria), probs),
        }
    ]


# --- Summarize from feedback (openai/summarize_from_feedback) -------------------------------
# comparisons: {info {id, post | article, title, subreddit, site}, summaries [2] {text},
# choice}; axis: {info, summary {text, axes {overall, accuracy, coverage, coherence}}}
# (1–7). Comparisons of the same pair of summaries are merged (their votes are the
# target); axis ratings of the same summary are merged (their histogram). All four parts
# are pooled and held out by post (group split), so a post is never on both sides.

SUMMARIZE_REPO = "openai/summarize_from_feedback"
SUMMARIZE_FILES = {
    "comparisons": ("comparisons/train/0000.parquet", "comparisons/validation/0000.parquet"),
    "axis": ("axis/validation/0000.parquet", "axis/test/0000.parquet"),
}
SUMMARY_AXES = {
    "overall": "How good is this summary overall?",
    "accuracy": "How accurate is this summary: does it only state what the post says?",
    "coverage": "How well does this summary cover the important information of the post?",
    "coherence": "How coherent and easy to read is this summary?",
}
SUMMARY_LEVELS = [
    "1 (very poor)", "2", "3", "4 (acceptable)", "5", "6", "7 (excellent)",
]  # fmt: skip
SUMMARY_TEXT_CHARS = 8000


def _post(info: dict) -> tuple[str, str] | None:
    body = info.get("post") or info.get("article") or ""
    if not body.strip() or len(body) > SUMMARY_TEXT_CHARS:
        return None
    where = f"r/{info['subreddit']}" if info.get("subreddit") else (info.get("site") or "")
    head = f"Title: {info.get('title') or ''}".strip()
    # some articles (CNN / DM) have no id: the text identifies the post
    post_id = info.get("id") or "text:" + hashlib.sha256(body.encode()).hexdigest()[:16]
    return post_id, "\n".join(p for p in (f"Source: {where}" if where else "", head,
                                             f"Text:\n{body.strip()}") if p)  # fmt: skip


def summarize_rows(comparisons: list[dict], axis: list[dict]) -> list[dict]:
    pairs: dict = {}
    for r in comparisons:
        post = _post(r["info"] or {})
        texts = [s["text"].strip() for s in r["summaries"] or []]
        if post is None or len(texts) != 2 or texts[0] == texts[1] or r["choice"] not in (0, 1):
            continue
        key = (post[0], *sorted(texts))
        entry = pairs.setdefault(key, {"post": post, "votes": defaultdict(int)})
        entry["votes"][texts[int(r["choice"])]] += 1
    rated: dict = {}
    for r in axis:
        post = _post(r["info"] or {})
        summary = r["summary"] or {}
        if post is None or not (summary.get("text") or "").strip():
            continue
        entry = rated.setdefault(
            (post[0], summary["text"].strip()), {"post": post, "axes": defaultdict(list)}
        )
        for name, value in (summary.get("axes") or {}).items():
            if name in SUMMARY_AXES and value is not None and 1 <= int(value) <= 7:
                entry["axes"][name].append(int(value) - 1)
    rows = []
    for (post_id, s1, s2), e in pairs.items():
        rows.append({"kind": "compare", "post_id": post_id, "post": e["post"][1],
                     "summary_1": s1, "summary_2": s2,
                     "votes": json.dumps([e["votes"][s1], e["votes"][s2]])})  # fmt: skip
    for (post_id, text), e in rated.items():
        axes = {k: v for k, v in e["axes"].items() if v}
        if not axes:
            continue
        name = sorted(axes)[int(_rng_of(f"axis:{post_id}:{text}").integers(len(axes)))]
        rows.append({"kind": name, "post_id": post_id, "post": e["post"][1], "summary_1": text,
                     "summary_2": "", "votes": json.dumps(axes[name])})  # fmt: skip
    return rows


def summarize_load(split: str):
    columns = {"comparisons": ["info", "summaries", "choice"], "axis": ["info", "summary"]}
    parts = {
        kind: [
            r for f in files for r in _parquet(SUMMARIZE_REPO, f, columns=columns[kind]).to_pylist()
        ]
        for kind, files in SUMMARIZE_FILES.items()
    }
    return _dataset(summarize_rows(parts["comparisons"], parts["axis"]))


def summarize_fb(row, rng, meta):
    votes = json.loads(row["votes"])
    if row["kind"] == "compare":
        texts = [row["summary_1"], row["summary_2"]]
        if rng.random() < 0.5:
            texts, votes = texts[::-1], votes[::-1]
        n = sum(votes)
        if n == 0:
            return []
        state = f"{row['post']}\n\nSummary 1:\n{texts[0]}\n\nSummary 2:\n{texts[1]}"
        criteria = {"summary_1": "Summary 1 is better", "summary_2": "Summary 2 is better"}
        return [
            {
                "state": state,
                "question": {
                    "type": "choice",
                    "instructions": "Which summary of the post did human raters prefer?",
                    "criteria": criteria,
                },
                **vote_target(list(criteria), votes),
            }
        ]
    state = f"{row['post']}\n\nSummary:\n{row['summary_1']}"
    return [
        {
            "state": state,
            "question": {
                "type": "score",
                "instructions": SUMMARY_AXES[row["kind"]],
                "criteria": SUMMARY_LEVELS,
            },
            **vote_target([str(i) for i in range(7)], [votes.count(i) for i in range(7)]),
        }
    ]


# --- Scruples anecdotes (tasksource/scruples) ----------------------------------------------
# {title, text, label_scores {AUTHOR, OTHER, EVERYBODY, NOBODY, INFO}, binarized_label_scores
# {RIGHT, WRONG}}: Reddit AITA votes. With ≥5 votes the target is their distribution, else
# the majority label. 30% of the rows with ≥5 votes ask the yes/no form ("Is the author in
# the wrong?": the share of AUTHOR + EVERYBODY votes among the four verdicts).

SCRUPLES_LABELS = {
    "author": ("AUTHOR", "The author of the post is in the wrong"),
    "other": ("OTHER", "The other party is in the wrong, not the author"),
    "everybody": ("EVERYBODY", "Everybody involved is in the wrong"),
    "nobody": ("NOBODY", "Nobody is in the wrong"),
    "info": ("INFO", "There is not enough information to judge"),
}
SCRUPLES_MIN_VOTES = 5


def scruples(row, rng, meta):
    scores = row["label_scores"] or {}
    counts = [int(scores.get(key) or 0) for key, _ in SCRUPLES_LABELS.values()]
    n = sum(counts)
    state = f"{(row['title'] or '').strip()}\n\n{(row['text'] or '').strip()}".strip()
    if n == 0 or len(state) > MAX_CHARS or len(state) < 50:
        return []
    labels = list(SCRUPLES_LABELS)
    if n >= SCRUPLES_MIN_VOTES and rng.random() < 0.3:
        verdict = n - counts[4]
        if verdict == 0:
            return []
        p = (counts[0] + counts[2]) / verdict
        return [
            {
                "state": state,
                "question": {
                    "type": "noul",
                    "instructions": "Do readers judge the author of this post to be in the wrong?",
                    "criteria": {
                        "false": "No, the author is not in the wrong",
                        "true": "Yes, the author is in the wrong",
                    },
                },
                **_noul_target(p),
            }
        ]
    if n >= SCRUPLES_MIN_VOTES:
        target = _one_hot_or_soft(labels, [c / n for c in counts])
    else:
        label = str(row.get("label") or "").lower()
        if label not in labels:
            return []
        target = {"expected": label}
    return [
        {
            "state": state,
            "question": {
                "type": "choice",
                "instructions": "Who do readers judge to be in the wrong in this story?",
                "criteria": {k: d for k, (_, d) in SCRUPLES_LABELS.items()},
            },
            **target,
        }
    ]


# --- compar:IA (ministere-culture/comparia-fr-arena, -votes, -reactions) ---------------------
# fr-arena: one row per turn {comparison_id, turn, response_a / response_b [user,
# assistant], full_conversation_a, choice (a_better | b_better | both_good | both_bad |
# idk | null), metadata {languages}}: the rated French turns. votes: one row per
# conversation {opening_msg, conv_turns, chosen_model_name, both_equal, model_a_name,
# model_b_name, conversation_a / conversation_b}: the multi-turn conversations only (the
# single-turn ones are fr-arena turns). reactions: {question_content, response_content,
# liked, disliked}: a thumbs up or down on one reply. Held out by the normalised opening
# message, the same key in the three sources, so a held-out conversation is never trained
# on in another compar:IA source.

COMPARIA_ARENA = "ministere-culture/comparia-fr-arena"
COMPARIA_VOTES = "ministere-culture/comparia-votes"
COMPARIA_REACTIONS = "ministere-culture/comparia-reactions"
ARENA_CHOICES = {"a_better": 0, "b_better": 1, "both_good": "both_good", "both_bad": "both_bad"}


COMPARIA_TEXT = {  # mix-v7.5: conversations in English get the English item
    "arena": {
        "fr": "Quel assistant a le mieux répondu au dernier message de l'utilisateur, "
        "selon l'utilisateur ?",
        "en": "Which assistant answered the user's last message better, according to the user?",
    },
    "earlier": {"fr": "Messages précédents de l'utilisateur :",
                "en": "Earlier messages of the user:"},
    "votes": {
        "fr": "Sur l'ensemble de la conversation, quel assistant l'utilisateur a-t-il préféré ?",
        "en": "Over the whole conversation, which assistant did the user prefer?",
    },
    "reaction": {
        "fr": ("Message de l'utilisateur :", "Réponse de l'assistant :",
               ("L'utilisateur a-t-il jugé cette réponse positivement (pouce levé plutôt que "
               "pouce baissé) ?"), "Non : il l'a jugée négativement",
               "Oui : il l'a jugée positivement"),
        "en": ("User message:", "Assistant reply:",
               "Did the user rate this reply positively (thumbs up rather than thumbs down)?",
               "No: they rated it negatively", "Yes: they rated it positively"),
    },
}  # fmt: skip


def _aggregate(rows: list[dict], key, labels: list, label_of) -> list[dict]:
    """Rows with the same ``key(row)`` as one row whose ``votes`` (JSON) count ``label_of``
    over ``labels``; the first row's other fields (sides already canonical in ``key``)."""
    out: dict = {}
    for r in rows:
        entry = out.setdefault(key(r), {**r, "votes": [0] * len(labels)})
        entry["votes"][labels.index(label_of(r, entry))] += 1
    return [{**e, "votes": json.dumps(e["votes"])} for e in out.values()]


def _messages(value) -> list[dict]:
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else []
    out = []
    for m in value or []:
        content = m.get("content")
        if isinstance(content, str) and content.strip() and m.get("role") in ("user", "assistant"):
            out.append({"role": m["role"], "content": content})
    return out


def _is_french(languages) -> bool:
    langs = [str(x).lower() for x in languages or []]
    return not langs or any(x in ("fr", "french", "français", "francais") for x in langs)


def _hub_files(repo: str, suffix: str = ".parquet") -> list[str]:
    from huggingface_hub import HfApi

    return sorted(
        f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(suffix)
    )


def comparia_arena_load(split: str):
    rows = []
    columns = ["response_id", "choice", "response_a", "response_b", "turn", "comparison_id",
               "full_conversation_a", "metadata"]  # fmt: skip
    for name in _hub_files(COMPARIA_ARENA):
        table = _parquet(COMPARIA_ARENA, name, columns=columns, revision=None,
                         filters=[("choice", "in", list(ARENA_CHOICES))])  # fmt: skip
        for r in table.to_pylist():
            if not _is_french((r.get("metadata") or {}).get("languages")):
                continue
            a, b = _messages(r["response_a"]), _messages(r["response_b"])
            full = _messages(r["full_conversation_a"])
            users = [m["content"] for m in full if m["role"] == "user"]
            if len(a) != 2 or len(b) != 2 or not users:
                continue
            earlier = users[: int(r["turn"] or 0)]
            size = len(a[1]["content"]) + len(b[1]["content"]) + len(a[0]["content"])
            if size + sum(map(len, earlier)) > MAX_CHARS:
                continue
            rows.append(
                {
                    "id": r["response_id"],
                    "opening": users[0],
                    "earlier": json.dumps(earlier, ensure_ascii=False),
                    "conv_a": json.dumps(a, ensure_ascii=False),
                    "conv_b": json.dumps(b, ensure_ascii=False),
                    "choice": r["choice"],
                }
            )
    return _dataset(aggregate_arena(rows))


def aggregate_arena(rows: list[dict]) -> list[dict]:
    """Votes on the same turn (earlier messages, the two replies, either side) as one row:
    ``votes`` = [first, second, both_good, both_bad], sides in a canonical order."""
    canon = []
    for r in rows:
        first, second = sorted((r["conv_a"], r["conv_b"]))
        choice = r["choice"]
        if first != r["conv_a"]:
            choice = {"a_better": "b_better", "b_better": "a_better"}.get(choice, choice)
        canon.append({**r, "conv_a": first, "conv_b": second, "choice": choice})
    labels = ["a_better", "b_better", "both_good", "both_bad"]
    return _aggregate(canon, lambda r: (r["earlier"], r["conv_a"], r["conv_b"]), labels,
                      lambda r, e: r["choice"])  # fmt: skip


def comparia_arena(row, rng, meta):
    earlier = json.loads(row["earlier"])
    extra = ("both_good", "both_bad")
    a, b = json.loads(row["conv_a"]), json.loads(row["conv_b"])
    counts = (json.loads(row["votes"]) if row.get("votes")
              else _counts(ARENA_CHOICES[row["choice"]], extra))  # fmt: skip
    lang = lang_of(two_sides(a, b, "fr") + " ".join(earlier), "fr")
    item = pairwise_votes(a, b, counts, extra, lang, rng, COMPARIA_TEXT["arena"][lang])
    if earlier:
        previous = "\n".join(f"- {m.strip()}" for m in earlier)
        item["state"] = f"{COMPARIA_TEXT['earlier'][lang]}\n{previous}\n\n{item['state']}"
    return [item]


def comparia_votes_load(split: str):
    from datasets import load_dataset

    ds = load_dataset(COMPARIA_VOTES, split="train")
    rows = []
    for r in ds:
        if int(r.get("conv_turns") or 0) < 2:
            continue
        if r.get("both_equal"):
            winner = "equal"
        elif r.get("chosen_model_name") and r["chosen_model_name"] == r.get("model_a_name"):
            winner = "a"
        elif r.get("chosen_model_name") and r["chosen_model_name"] == r.get("model_b_name"):
            winner = "b"
        else:
            continue
        a, b = _messages(r["conversation_a"]), _messages(r["conversation_b"])
        if len(a) < 4 or len(b) < 4 or sum(len(m["content"]) for m in a + b) > MAX_CHARS:
            continue
        rows.append(
            {
                "id": str(r["id"]),
                "opening": r.get("opening_msg") or a[0]["content"],
                "conv_a": json.dumps(a, ensure_ascii=False),
                "conv_b": json.dumps(b, ensure_ascii=False),
                "winner": winner,
            }
        )
    return _dataset(aggregate_votes(rows))


def aggregate_votes(rows: list[dict]) -> list[dict]:
    """Votes on the same two conversations (either side) as one row: ``votes`` =
    [first, second, equal]."""
    canon = []
    for r in rows:
        first, second = sorted((r["conv_a"], r["conv_b"]))
        w = r["winner"]
        if first != r["conv_a"]:
            w = {"a": "b", "b": "a"}.get(w, w)
        canon.append({**r, "conv_a": first, "conv_b": second, "winner": w})
    return _aggregate(canon, lambda r: (r["conv_a"], r["conv_b"]), ["a", "b", "equal"],
                      lambda r, e: r["winner"])  # fmt: skip


def comparia_votes(row, rng, meta):
    a, b = json.loads(row["conv_a"]), json.loads(row["conv_b"])
    if row.get("votes"):
        counts = json.loads(row["votes"])
    else:
        counts = _counts({"a": 0, "b": 1, "equal": "equal"}[row["winner"]], ("equal",))
    lang = lang_of(two_sides(a, b, "fr"), "fr")
    return [pairwise_votes(a, b, counts, ("equal",), lang, rng, COMPARIA_TEXT["votes"][lang])]


def comparia_reactions_load(split: str):
    from datasets import load_dataset

    ds = load_dataset(COMPARIA_REACTIONS, split="train")
    columns = ["id", "opening_msg", "question_content", "response_content", "liked", "disliked"]
    rows = []
    for r in ds.select_columns(columns):
        if bool(r["liked"]) == bool(r["disliked"]):
            continue
        q, a = (r["question_content"] or "").strip(), (r["response_content"] or "").strip()
        if a.startswith('"') and a.endswith('"'):
            with contextlib.suppress(ValueError):
                a = json.loads(a)
        if not q or not a or len(q) + len(a) > MAX_CHARS:
            continue
        rows.append({"id": str(r["id"]), "opening": r["opening_msg"] or q, "question": q,
                     "answer": a, "liked": bool(r["liked"])})  # fmt: skip
    return _dataset(aggregate_reactions(rows))


def aggregate_reactions(rows: list[dict]) -> list[dict]:
    """Reactions to the same reply of the same message as one row: ``votes`` =
    [dislikes, likes]."""
    return _aggregate(rows, lambda r: (r["question"], r["answer"]), [False, True],
                      lambda r, e: r["liked"])  # fmt: skip


def comparia_reactions(row, rng, meta):
    counts = json.loads(row["votes"]) if row.get("votes") else [int(not row["liked"]),
                                                                int(row["liked"])]  # fmt: skip
    lang = lang_of(f"{row['question']}\n{row['answer']}", "fr")
    user, reply, question, no, yes = COMPARIA_TEXT["reaction"][lang]
    return [
        {
            "state": f"{user}\n{row['question']}\n\n{reply}\n{row['answer']}",
            "question": {
                "type": "noul",
                "instructions": question,
                "criteria": {"false": no, "true": yes},
            },
            **vote_target(["no", "yes"], counts, VOTE_ALPHA_NOUL),
            "lang": lang,
        }
    ]


def comparia_group(row) -> str:
    return _norm(row["opening"])[:500]


# --- Community Alignment, French (facebook/community-alignment-dataset) -----------------------
# One row per annotator conversation: up to four turns, each with a prompt, four candidate
# responses (a–d) and the annotator's preferred one; later turns continue the preferred
# response. First turns shown identically to ≥10 annotators become one item with their
# votes as the target; other turns one item per annotator (one-hot).

CA_REPO = "facebook/community-alignment-dataset"
CA_TURNS = ("first", "second", "third", "fourth")
CA_MIN_SOFT = 10


def community_alignment_rows(raw: list[dict]) -> list[dict]:
    """mix-v7.5: every turn shown identically (same context, prompt and responses) to
    several annotators is one row with their votes; the target is the smoothed vote share
    (:func:`vote_target`): soft from 2 annotators on, a single annotator's choice at 0.75."""
    turns: dict = defaultdict(list)
    for r in raw:
        if r.get("assigned_lang") != "fr":
            continue
        context: list[dict] = []
        for turn in CA_TURNS:
            prompt = r.get(f"{turn}_turn_prompt")
            responses = [r.get(f"{turn}_turn_response_{x}") for x in "abcd"]
            preferred = str(r.get(f"{turn}_turn_preferred_response") or "")
            if not prompt or not preferred.startswith("response_"):
                break
            letters = [x for x, t in zip("abcd", responses, strict=True) if t and t.strip()]
            choice = preferred.removeprefix("response_")
            if len(letters) < 2 or choice not in letters:
                break
            entry = {
                "group": _norm(r.get("first_turn_prompt"))[:500],
                "context": json.dumps(context, ensure_ascii=False),
                "prompt": prompt,
                "responses": json.dumps(
                    [responses["abcd".index(x)] for x in letters], ensure_ascii=False
                ),
            }
            turns[(entry["context"], prompt, entry["responses"])].append(
                (entry, letters.index(choice))
            )
            context = [
                *context,
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": responses["abcd".index(choice)]},
            ]
    rows = []
    for votes in turns.values():
        n = len(json.loads(votes[0][0]["responses"]))
        counts = [sum(v == i for _, v in votes) for i in range(n)]
        dist = list(vote_target([str(i) for i in range(n)], counts)["target"].values())
        rows.append({**votes[0][0], "target": json.dumps(dist), "votes": len(votes)})
    return rows


def community_alignment_load(split: str):
    from datasets import load_dataset

    ds = load_dataset(CA_REPO, split="train")
    columns = ["assigned_lang"] + [
        f"{t}_turn_{f}" for t in CA_TURNS
        for f in ("prompt", "preferred_response", *(f"response_{x}" for x in "abcd"))
    ]  # fmt: skip
    rows = [r for r in community_alignment_rows(ds.select_columns(columns).to_list())
            if len(r["context"]) + len(r["prompt"]) + sum(map(len, json.loads(r["responses"])))
            <= MAX_CHARS]  # fmt: skip
    return _dataset(rows)


def community_alignment_fr(row, rng, meta):
    responses = json.loads(row["responses"])
    target = json.loads(row["target"])
    order = [int(i) for i in rng.permutation(len(responses))]
    context = json.loads(row["context"])
    head = f"{_conversation(context)}\n\n" if context else ""
    blocks = [f"Réponse {k}:\n{responses[i].strip()}" for k, i in enumerate(order, 1)]
    state = (
        f"{head}{_conversation([{'role': 'user', 'content': row['prompt']}])}\n\n"
        + "\n\n".join(blocks)
    )
    labels = [f"response_{k}" for k in range(1, len(order) + 1)]
    criteria = {lab: f"La réponse {k} est la meilleure" for k, lab in enumerate(labels, 1)}
    soft = int(row.get("votes") or 1) > 1 if "votes" in row else max(target) < 1.0
    return [
        {
            "state": state,
            "question": {
                "type": "choice",
                "instructions": "Quelle réponse les annotateurs préfèrent-ils ?"
                if soft
                else "Quelle réponse l'annotateur préfère-t-il ?",
                "criteria": criteria,
            },
            **_one_hot_or_soft(labels, [target[i] for i in order]),
            "lang": "fr",
        }
    ]


# --- DICES (github.com/google-research-datasets/dices-dataset) -------------------------------
# One row per (rater, conversation): context, response, and Yes / No / Unsure answers to
# the overall safety question and its parts. Per conversation and question, the target is
# P(yes) = (yes + unsure / 2) / raters (DICES-350: 123 raters, DICES-990: 60–70).

DICES_URL = "https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main"
DICES_FILES = ("350/diverse_safety_adversarial_dialog_350.csv",
               "990/diverse_safety_adversarial_dialog_990.csv")  # fmt: skip
DICES_QUESTIONS = {
    "Q_overall": "Is the chatbot's last reply unsafe in any way?",
    "Q2_harmful_content_overall": "Does the chatbot's last reply contain harmful content?",
    "Q3_bias_overall": "Does the chatbot's last reply contain unfair bias?",
    "Q3_unfair_bias_overall": "Does the chatbot's last reply contain unfair bias?",
    "Q4_misinformation": "Does the chatbot's last reply contain misinformation?",
    "Q4_misinformation_overall": "Does the chatbot's last reply contain misinformation?",
    "Q5_political_affiliation": "Does the chatbot's last reply take a political side?",
    "Q6_policy_guidelines_overall": "Does the chatbot's last reply break a policy guideline "
    "(polarising topics, endorsements)?",
}


def _dices_file(name: str) -> Path:
    import urllib.request

    path = CACHE / "dices" / Path(name).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with urllib.request.urlopen(f"{DICES_URL}/{name}", timeout=300) as r, tmp.open("wb") as f:
            f.write(r.read())
        tmp.replace(path)
    return path


def dices_rows(frame, tag: str) -> list[dict]:
    rows = []
    for item_id, group in frame.groupby("item_id"):
        first = group.iloc[0]
        for column in DICES_QUESTIONS:  # the 350 and 990 files name some columns apart
            if column not in group:
                continue
            answers = [str(a).strip().lower() for a in group[column] if isinstance(a, str)]
            answers = [a for a in answers if a in ("yes", "no", "unsure")]
            if len(answers) < 5:
                continue
            p = (answers.count("yes") + 0.5 * answers.count("unsure")) / len(answers)
            rows.append({"id": f"{tag}-{item_id}", "context": str(first["context"]),
                         "response": str(first["response"]), "column": column, "p": p,
                         "raters": len(answers)})  # fmt: skip
    return rows


def dices_load(split: str):
    import pandas as pd

    rows = []
    for name in DICES_FILES:
        frame = pd.read_csv(_dices_file(name), dtype=str)
        rows += dices_rows(frame, name.split("/")[0])
    return _dataset(rows)


def dices(row, rng, meta):
    context = re.sub(r"^LAMDA:", "CHATBOT:", row["context"].strip(), flags=re.MULTILINE)
    state = f"{context}\nCHATBOT: {row['response'].strip()}"
    return [
        {
            "state": state,
            "question": {
                "type": "noul",
                "instructions": DICES_QUESTIONS[row["column"]]
                + " (as a panel of raters judged it)",
                "criteria": {"false": "No", "true": "Yes"},
            },
            **_noul_target(row["p"]),
        }
    ]


# --- Social-Chem-101 (tasksource/social-chemestry-101) ---------------------------------------
# One row per (rule of thumb, worker breakdown): situation, action, rot, rot-agree (0–4:
# share of people who agree), action-moral-judgment (−2..2), rot-bad, split. Breakdowns of
# the same rot are merged (several workers give a distribution). Half the items ask how
# people judge the action, half how widely the rule of thumb is shared.

SOCIAL_CHEM_FILE = "social-chem-101.v1.0.tsv"
JUDGMENT_LEVELS = ["Very bad", "Bad", "Expected / OK", "Good", "Very good"]
AGREE_LEVELS = [
    "Almost nobody agrees (<1%)",
    "Few people agree (5–25%)",
    "Controversial (about 50%)",
    "Most people agree (75–90%)",
    "Almost everybody agrees (>99%)",
]


def _numbers(values) -> list[int]:
    out = []
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(round(x))
    return out


def social_chem_rows(frame) -> list[dict]:
    rows = []
    for rot_id, group in frame.groupby("rot-id", sort=True):
        first = group.iloc[0]
        judged = [v + 2 for v in _numbers(group["action-moral-judgment"]) if -2 <= v <= 2]
        agree = [v for v in _numbers(group["rot-agree"]) if 0 <= v <= 4]
        if not str(first.get("situation") or "").strip():
            continue
        rows.append({"id": rot_id, "situation": str(first["situation"]),
                     "action": str(first.get("action") or ""), "rot": str(first.get("rot") or ""),
                     "judged": json.dumps(judged), "agree": json.dumps(agree)})  # fmt: skip
    return rows


def social_chem_load(split: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("tasksource/social-chemestry-101", SOCIAL_CHEM_FILE, repo_type="dataset")
    columns = ["split", "rot-agree", "rot-bad", "action-moral-judgment", "action", "situation",
               "rot", "rot-id"]  # fmt: skip
    frame = pd.read_csv(path, sep="\t", usecols=columns, dtype=str, quoting=3)
    frame = frame[(frame["split"] == split) & (frame["rot-bad"].fillna("0") == "0")]
    return _dataset(social_chem_rows(frame))


def social_chem(row, rng, meta):
    judged, agree = json.loads(row["judged"]), json.loads(row["agree"])
    ask_action = bool(judged and row["action"].strip()) and (not agree or rng.random() < 0.5)
    if ask_action:
        state = f"Situation: {row['situation'].strip()}\nAction: {row['action'].strip()}"
        question = {"type": "score", "instructions": "How do people generally judge this action?",
                    "criteria": JUDGMENT_LEVELS}  # fmt: skip
        dist = histogram(judged, 5)
    elif agree and row["rot"].strip():
        state = f"Situation: {row['situation'].strip()}\nRule of thumb: {row['rot'].strip()}"
        question = {"type": "score", "instructions": "How many people would agree with this "
                    "rule of thumb?", "criteria": AGREE_LEVELS}  # fmt: skip
        dist = histogram(agree, 5)
    else:
        return []
    return [{"state": state, "question": question,
             **_one_hot_or_soft([str(i) for i in range(5)], dist)}]  # fmt: skip


# --- MIC (SALT-NLP/MIC) --------------------------------------------------------------------
# {Q, A (a chatbot's answer), rot (a rule of thumb about the answer), A_agrees (0 the answer
# disagrees with the rot, 1 neutral, 2 agrees), rot-agree (1–5), violation-severity (1–5)}.
# 60% of the items ask whether the answer agrees with the rule of thumb, 40% how severe
# breaking the rule would be.

MIC_SPLITS = {"train": "MIC/train/0000.parquet", "validation": "MIC/validation/0000.parquet",
              "test": "MIC/test/0000.parquet"}  # fmt: skip
SEVERITY_LEVELS = ["Fine (1)", "Unfavourable (2)", "Bad (3)", "Horrible (4)", "Worst (5)"]


def mic_load(split: str):
    table = _parquet("SALT-NLP/MIC", MIC_SPLITS[split],
                     columns=["Q", "A", "rot", "A_agrees", "violation-severity"])  # fmt: skip
    return _dataset([{**r, "severity": r.pop("violation-severity")} for r in table.to_pylist()])


def mic(row, rng, meta):
    q, a, rot = (str(row.get(k) or "").strip() for k in ("Q", "A", "rot"))
    if not q or not a or not rot:
        return []
    if rng.random() < 0.6 and row.get("A_agrees") in (0, 1, 2):
        labels = ["disagrees", "neutral", "agrees"]
        return [
            {
                "state": f"User: {q}\nChatbot: {a}\nRule of thumb: {rot}",
                "question": {
                    "type": "choice",
                    "instructions": "Does the chatbot's answer agree with the rule of thumb?",
                    "criteria": {
                        "disagrees": "The answer goes against the rule of thumb",
                        "neutral": "The answer neither follows nor goes against it",
                        "agrees": "The answer follows the rule of thumb",
                    },
                },
                "expected": labels[int(row["A_agrees"])],
            }
        ]
    severity = row.get("severity")
    if severity not in (1, 2, 3, 4, 5):
        return []
    return [
        {
            "state": f"Rule of thumb: {rot}\n(It was written about this exchange.)\n"
            f"User: {q}\nChatbot: {a}",
            "question": {
                "type": "score",
                "instructions": "How bad would it be to break this rule of thumb?",
                "criteria": SEVERITY_LEVELS,
            },
            "expected": str(int(severity) - 1),
        }
    ]


# --- x-stance, French (strombergnlp/x-stance CSVs of ZurichNLP/x_stance) ---------------------
# {question, comment, label (FAVOR | AGAINST)}: a Swiss election candidate's comment on a
# policy question, labelled by the candidate's own answer. 30% of the items ask the
# negated question (against), so the yes share stays near the data's.

XSTANCE_REPO = "strombergnlp/x-stance"
XSTANCE_FILES = {s: f"x-stance-{s}-fr.csv" for s in ("train", "valid", "test")}


def x_stance_load(split: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(XSTANCE_REPO, XSTANCE_FILES[split], repo_type="dataset")
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame = frame[frame["label"].isin(["FAVOR", "AGAINST"])]
    return _dataset(frame[["question", "comment", "label"]].to_dict("records"))


def x_stance_fr(row, rng, meta):
    q, c = row["question"].strip(), row["comment"].strip()
    if not q or len(c) < 10:
        return []
    favour = row["label"] == "FAVOR"
    state = f"Question posée aux candidats : {q}\nCommentaire d'un candidat : {c}"
    if rng.random() < 0.3:
        instructions = "D'après son commentaire, le candidat est-il opposé à la proposition ?"
        criteria = {"false": "Non, il y est favorable", "true": "Oui, il y est opposé"}
        expected = "no" if favour else "yes"
    else:
        instructions = "D'après son commentaire, le candidat est-il favorable à la proposition ?"
        criteria = {"false": "Non, il y est opposé", "true": "Oui, il y est favorable"}
        expected = "yes" if favour else "no"
    return [{"state": state, "question": {"type": "noul", "instructions": instructions,
                                          "criteria": criteria},
             "expected": expected, "lang": "fr"}]  # fmt: skip


# --- registry ------------------------------------------------------------------------------

# name -> (licence as stated on the card, redistributable, note)
LICENCES: dict[str, tuple[str, str, str]] = {
    "helpsteer3_x": ("CC BY 4.0", "yes", "nvidia/HelpSteer3 card tag (preference)"),
    "oasst2": ("Apache-2.0", "yes", "OpenAssistant/oasst2 card tag"),
    "lmarena_140k": (
        "CC BY 4.0",
        "yes",
        ("lmarena-ai/arena-human-preference-140k card tag; responses are model outputs "
        "(providers' terms may apply)"),
    ),
    "webgpt": (
        "not stated on the card",
        "unknown",
        ("openai/webgpt_comparisons card has no licence (its loading script's Apache-2.0 "
        "header covers the script)"),
    ),
    "summarize_fb": (
        "Modified MIT (OpenAI, 'Software and Datasets')",
        "yes",
        ("openai/summarize_from_feedback card states none; github.com/openai/"
        "summarize-from-feedback LICENSE; TL;DR posts CC BY 4.0"),
    ),
    "scruples": ("Apache-2.0", "yes", "tasksource/scruples card tag (Reddit AITA posts)"),
    "comparia_arena": (
        "Etalab 2.0 + CC BY 4.0",
        "yes",
        ("ministere-culture/comparia-fr-arena card (gated; subject to third-party claims on "
        "model outputs)"),
    ),
    "comparia_votes": (
        "Etalab 2.0",
        "yes",
        "ministere-culture/comparia-votes card (gated; third-party claims on model outputs)",
    ),
    "comparia_reactions": (
        "Etalab 2.0",
        "yes",
        "ministere-culture/comparia-reactions card (gated; third-party claims on model outputs)",
    ),
    "community_alignment_fr": (
        "CC BY 4.0",
        "yes",
        "facebook/community-alignment-dataset card (gated); French rows",
    ),
    "dices": ("CC BY 4.0", "yes", "github.com/google-research-datasets/dices-dataset README"),
    "social_chem": (
        "CC BY-SA 4.0",
        "share-alike",
        "tasksource copy has no card licence; github.com/mbforbes/social-chemistry-101 README",
    ),
    "mic": ("CC BY-SA 4.0", "share-alike", "SALT-NLP/MIC card has none; github.com/SALT-NLP/mic LICENSE"),
    "x_stance_fr": (
        "CC BY-NC 4.0",
        "NC",
        ("ZurichNLP/x_stance card (rows read from the strombergnlp/x-stance CSV copy, "
        "tagged MIT there)"),
    ),
}  # fmt: skip


def _src(name, repo, train, held, adapter, family, lang, use, cap, load, **kw) -> Source:
    return Source(
        name, repo, None, train, held, adapter, family, lang, LICENCES[name][0],
        use == "commercial", cap=cap, load=load, block=kw.pop("block", "judge"),
        license_use=use, **kw,
    )  # fmt: skip


JUDGE_SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        _src("helpsteer3_x", "nvidia/HelpSteer3", "train", "validation", helpsteer3_x,
             "judge_hard", "en", "commercial", 23000, helpsteer3_x_load),
        _src("oasst2", "OpenAssistant/oasst2", "train", "validation", oasst2, "judge_rating",
             "en", "commercial", 40000, oasst2_load),
        _src("lmarena_140k", LMARENA_REPO, "train", None, lmarena, "judge_pref", "en",
             "commercial", 60000, lmarena_load, group=lambda r: r["session"]),
        _src("webgpt", WEBGPT_REPO, "train", None, webgpt, "judge_pref", "en", "unspecified",
             20000, webgpt_load, group=lambda r: _norm(r["question"])),
        _src("summarize_fb", SUMMARIZE_REPO, "train", None, summarize_fb, "judge_summary", "en",
             "commercial", 20000, summarize_load, group=lambda r: r["post_id"]),
        _src("scruples", "tasksource/scruples", "train", "validation", scruples,
             "moral_judgment", "en", "commercial", 28000, None),
        _src("comparia_arena", COMPARIA_ARENA, "train", None, comparia_arena, "judge_pref_fr",
             "fr", "commercial", 40000, comparia_arena_load, group=comparia_group),
        _src("comparia_votes", COMPARIA_VOTES, "train", None, comparia_votes, "judge_pref_fr",
             "fr", "commercial", 12000, comparia_votes_load, group=comparia_group),
        _src("comparia_reactions", COMPARIA_REACTIONS, "train", None, comparia_reactions,
             "judge_rating_fr", "fr", "commercial", 12000, comparia_reactions_load,
             group=comparia_group),
        _src("community_alignment_fr", CA_REPO, "train", None, community_alignment_fr,
             "judge_pref_fr", "fr", "commercial", 20000, community_alignment_load,
             group=lambda r: r["group"]),
        _src("dices", "github:google-research-datasets/dices-dataset", "train", None, dices,
             "safety_rating", "en", "commercial", 8000, dices_load,
             group=lambda r: _norm(r["context"])[:300]),
        _src("social_chem", "tasksource/social-chemestry-101", "train", "dev", social_chem,
             "moral_judgment", "en", "commercial", 15000, social_chem_load),
        _src("mic", "SALT-NLP/MIC", "train", "validation", mic, "moral_judgment", "en",
             "commercial", 15000, mic_load),
        _src("x_stance_fr", "ZurichNLP/x_stance", "train", "valid", x_stance_fr, "stance", "fr",
             "non-commercial", 15000, x_stance_load, block="analysis"),
    ]
}  # fmt: skip
