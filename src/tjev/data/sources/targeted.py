"""Sources aimed at the JevBench weaknesses of the mix-v5 models (the mix's "targeted" block,
since mix-v6; docs/data.md, section mix-v6).

The weak JevBench families of the mix-v5b 4B were long_policy, judge_hard, temporal_numeric,
ambiguous and multi_hop, plus over-confident yes/no (noul) abstention and French decisions.

==========================  =================================================  ===============
source                      what it adds                                       labels
==========================  =================================================  ===============
``eikos``                   rule / temporal / numeric programs, noul rows      exact or teacher
``eikos_long``              long dossiers cut around the asked document        as ``eikos``
``procedural``              policy applicability, arithmetic, state tracking,  exact (posteriors
                            Bayesian posteriors (13 configs, sampled per one)  are soft)
``tempreason``              month arithmetic with near-miss options (EN + FR)  exact
``helpsteer3``              A / B / tie preference and its 7-level scale       annotators (soft)
``cuad``                    clause presence in real contract excerpts          exact (annotations)
``cold_law_fr``             "is this article version in force on date D?"      exact (dates)
``argimi_fr``               outcome of a Cour de cassation appeal              exact (decision)
``twin_*``                  negated questions of exact-label noul sources      1 − p of the base
==========================  =================================================  ===============

Same contract as the other adapter modules: each adapter maps one row to item dicts. Loaders
that sample a large upstream file (COLD, ArGiMi) cache their sample under ``CACHE``.
Licences are in :mod:`tjev.data.licences`.
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from tjev.data.item import rendered_state
from tjev.data.sources.base import CACHE, Adapter, Source
from tjev.data.sources.contenders import NOUL_MAP, _one_hot_or_soft
from tjev.data.sources.public import boolq, piaf_load, squad_abstain


def _hash01(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16**12


def _rng_of(key: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))


def _dataset(rows: list[dict]):
    from datasets import Dataset

    return Dataset.from_list(rows)


# --- Eikos (caiovicentino1/eikos-decisions) ---------------------------------------------
# core: {id, family, lang ("English" / "Brazilian Portuguese" / "Spanish"), item_writer,
# label_source (program | exact_probability | writer_exact_probs | teacher | human), state,
# question_type, instructions, options [{label, description}], expected, target_probs}.
# Kept: English rows whose label is computed (program, exact or writer-exact
# probabilities), and the teacher-labelled noul rows. Program targets are the exact
# answer (the card's 0.95 smoothing is not kept); the others keep their distribution.
# long_context: the same fields plus long_n_docs / long_answer_doc; the dossier is
# "--- Document k (kind) ---" sections. The eval/ files are never read.

EIKOS_REPO = "caiovicentino1/eikos-decisions"
EIKOS_EXACT = ("program", "exact_probability", "writer_exact_probs")
EIKOS_LONG_CHARS = 10000  # dossier budget after the cut (~2.5-3k tokens)
_DOC = re.compile(r"^--- Document (\d+) \(([^)\n]*)\) ---[ \t]*$", re.MULTILINE)


def _eikos_rows(path: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(EIKOS_REPO, path, repo_type="dataset")
    with Path(local).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _eikos_keep(r: dict) -> bool:
    return r.get("lang") == "English" and (
        r["label_source"] in EIKOS_EXACT
        or (r["label_source"] == "teacher" and r["question_type"] == "noul")
    )


def _eikos_row(r: dict, **extra) -> dict:
    return {
        "id": r["id"],
        "family": r["family"],
        "label_source": r["label_source"],
        "state": r["state"],
        "question_type": r["question_type"],
        "instructions": r["instructions"],
        "options": json.dumps(r["options"], ensure_ascii=False),
        "expected": str(r["expected"]),
        "target_probs": json.dumps(r.get("target_probs")),
        **extra,
    }


def eikos_load(split: str):
    rows = [_eikos_row(r) for r in _eikos_rows(f"data/core/{split}.jsonl") if _eikos_keep(r)]
    return _dataset(rows)


def eikos_long_load(split: str):
    rows = [
        _eikos_row(r, answer_doc=int(r["long_answer_doc"]))
        for r in _eikos_rows(f"data/long_context/{split}.jsonl")
        if _eikos_keep(r)
    ]
    return _dataset(rows)


def eikos(row, rng, meta):
    options = json.loads(row["options"])
    labels = [str(o["label"]) for o in options]
    descs = [o.get("description") or "" for o in options]
    probs = json.loads(row["target_probs"]) or []
    kind, expected = row["question_type"], str(row["expected"])
    if len(probs) != len(labels) or len(set(labels)) != len(labels) or expected not in labels:
        return []
    if kind == "noul":
        named = {
            NOUL_MAP.get(lab.lower()): (d, p)
            for lab, d, p in zip(labels, descs, probs, strict=True)
        }
        if set(named) != {"no", "yes"}:
            return []
        criteria = {"false": named["no"][0], "true": named["yes"][0]}
        labels, probs = ["no", "yes"], [named["no"][1], named["yes"][1]]
        expected = NOUL_MAP[expected.lower()]
    elif kind == "choice":
        criteria = dict(zip(labels, descs, strict=True))
    elif kind == "score":  # ordinal levels in their given order
        criteria = [
            f"{lab}: {d}" if d and d != lab else lab for lab, d in zip(labels, descs, strict=True)
        ]
        expected = str(labels.index(expected))
        labels = [str(i) for i in range(len(labels))]
    else:
        return []
    exact = row["label_source"] == "program"
    target = {"expected": expected} if exact else _one_hot_or_soft(labels, probs)
    return [
        {
            "state": row["state"],
            "question": {"type": kind, "instructions": row["instructions"], "criteria": criteria},
            **target,
            "family": row["family"],
        }
    ]


def eikos_dossier(
    state: str, instructions: str, answer_doc: int, rng, budget: int = EIKOS_LONG_CHARS
) -> tuple[str, str] | None:
    """A long dossier cut to ``budget`` characters: the asked document plus other documents
    (random ones, in a random order) while they fit, renumbered 1..m, and the question's
    "Document k" rewritten to the asked document's new number. None when the asked
    document alone is over the budget or the question does not name it."""
    heads = list(_DOC.finditer(state))
    docs = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(state)
        docs.append((int(m.group(1)), m.group(2), state[m.end() : end].strip()))
    asked = [d for d in docs if d[0] == answer_doc]
    named = re.compile(rf"\bDocument {answer_doc}\b")
    if len(asked) != 1 or len(asked[0][2]) > budget or not named.search(instructions):
        return None
    kept, total = [asked[0]], len(asked[0][2])
    for k in rng.permutation(len(docs)):
        doc = docs[int(k)]
        if doc[0] != answer_doc and total + len(doc[2]) <= budget:
            kept.append(doc)
            total += len(doc[2])
    order = [kept[int(k)] for k in rng.permutation(len(kept))]
    new = 1 + next(i for i, d in enumerate(order) if d[0] == answer_doc)
    text = "\n\n".join(
        f"--- Document {i} ({kind}) ---\n{body}" for i, (_, kind, body) in enumerate(order, 1)
    )
    return text, named.sub(f"Document {new}", instructions)


def eikos_long(row, rng, meta):
    cut = eikos_dossier(row["state"], row["instructions"], int(row["answer_doc"]), rng)
    if cut is None:
        return []
    state, instructions = cut
    return eikos({**row, "state": state, "instructions": instructions}, rng, meta)


# --- procedural-typed-decisions (tasksource) ---------------------------------------------
# One row per state with several questions: questions {name: {type, instructions,
# criteria?}}, answers {name: {type, noul | choice + probabilities | score + probabilities}}.
# A noul answer is a probability (exact: 0 / 1, or an exact posterior). Sampled per config
# (hash of the row id), states over PROC_MAX_STATE_CHARS left out (they would not fit the
# 4096-token bucket); menus over 26 options are dropped by the builder.

PROC_REPO = "tasksource/procedural-typed-decisions"
PROC_CONFIGS = (
    "arithmetic",
    "entity_belief_tracking",
    "event_state_reconstruction",
    "evidence_sufficiency",
    "multi_view_adjudication",
    "needle_retrieval",
    "partial_observation_calibration",
    "policy_applicability",
    "policy_under_uncertainty",
    "record_aggregation",
    "state_perturbation",
    "table_lookup",
    "taxonomy_routing",
)
PROC_PER_CONFIG = {"train": 800, "validation": 150}
PROC_MAX_STATE_CHARS = 10000


def procedural_load(split: str):
    from datasets import load_dataset

    rows = []
    for config in PROC_CONFIGS:
        ds = load_dataset(PROC_REPO, config, split=split)
        ds = ds.select_columns(["id", "level", "state", "questions", "answers"])
        picked = sorted(
            (r for r in ds if len(r["state"]) <= PROC_MAX_STATE_CHARS),
            key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest(),
        )[: PROC_PER_CONFIG.get(split, PROC_PER_CONFIG["validation"])]
        rows += [{**r, "level": str(r["level"]), "task": config} for r in picked]
    return _dataset(rows)


def _proc_state(state: str):
    try:
        parsed = json.loads(state)
    except ValueError:
        return state
    return rendered_state(parsed) if isinstance(parsed, (dict, list)) else state


def procedural(row, rng, meta):
    state = _proc_state(row["state"])
    questions, answers = json.loads(row["questions"]), json.loads(row["answers"])
    out = []
    for name, q in questions.items():
        a = answers.get(name)
        kind = q.get("type")
        if a is None or kind not in ("noul", "choice", "score"):
            continue
        criteria = q.get("criteria")
        if kind == "noul":
            p = float(a["noul"])
            if not 0.0 <= p <= 1.0:
                continue
            given = criteria if isinstance(criteria, dict) else {}
            criteria = {
                "false": str(given.get("false", given.get("no", ""))),
                "true": str(given.get("true", given.get("yes", ""))),
            }
            target = (
                {"expected": "yes" if p == 1.0 else "no"}
                if p in (0.0, 1.0)
                else {"target": {"no": 1.0 - p, "yes": p}}
            )
        elif kind == "choice":
            criteria = {str(k): "" if str(v) == str(k) else str(v) for k, v in criteria.items()}
            probs = a.get("probabilities") or {a["choice"]: 1.0}
            labels = list(criteria)
            target = _one_hot_or_soft(labels, [float(probs.get(k, 0.0)) for k in labels])
        else:
            criteria = ["" if str(c) == str(i) else str(c) for i, c in enumerate(criteria)]
            labels = [str(i) for i in range(len(criteria))]
            probs = a.get("probabilities") or {str(int(a["score"])): 1.0}
            target = _one_hot_or_soft(labels, [float(probs.get(k, 0.0)) for k in labels])
        out.append(
            {
                "state": state,
                "question": {"type": kind, "instructions": q["instructions"], "criteria": criteria},
                **target,
                "family": f"procedural_{row['task']}",
            }
        )
    return out


# --- TempReason L1 (tonytan48/TempReason) -------------------------------------------------
# {"question": "What is the time 1 year and 7 month after Mar, 1873", "text_answers":
# {"text": ["Oct, 1874"]}, ...}: month arithmetic. The answer is recomputed from the
# question and rows that disagree are dropped. Rendered as a choice among the answer and
# near misses (±1 / ±2 months, ±1 year, the other direction, the offset's years or months
# only), or as a yes/no on one candidate; 30% in French (the arithmetic is language-free).

TEMPREASON_REPO = "tonytan48/TempReason"
TEMPREASON_FILES = {"train": "train_l1.json", "val": "val_l1.json"}
TEMPREASON_SAMPLE = {"train": 14000, "val": 1800}
TEMPREASON_FR = 0.3
_L1 = re.compile(
    r"^What is the time (?:(\d+) years?)?(?: and )?(?:(\d+) months?)? (after|before) "
    r"([A-Z][a-z]{2}), (\d{1,4})\s*\??$"
)
MONTHS_EN = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTHS_ABBR = [m[:3] for m in MONTHS_EN]
MONTHS_FR = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]


def tempreason_load(split: str):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TEMPREASON_REPO, TEMPREASON_FILES[split], repo_type="dataset")
    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: hashlib.sha256(f"{split}:{r['id']}".encode()).hexdigest())
    rows = rows[: TEMPREASON_SAMPLE[split]]
    return _dataset(
        [
            {
                "id": str(r["id"]),
                "question": r["question"],
                "answer": (r["text_answers"]["text"] or [""])[0],
            }
            for r in rows
        ]
    )


def tempreason_parse(question: str) -> tuple[int, int, int] | None:
    """(start month index, signed offset in months, direction sign) of an L1 question."""
    m = _L1.match(question.strip())
    if m is None or m.group(4) not in MONTHS_ABBR or not (m.group(1) or m.group(2)):
        return None
    years, months = int(m.group(1) or 0), int(m.group(2) or 0)
    sign = 1 if m.group(3) == "after" else -1
    start = int(m.group(5)) * 12 + MONTHS_ABBR.index(m.group(4))
    return start, sign * (12 * years + months), sign


def _month_label(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _month_text(index: int, lang: str) -> str:
    names = MONTHS_FR if lang == "fr" else MONTHS_EN
    return f"{names[index % 12]} {index // 12}"


def _offset_text(years: int, months: int, lang: str) -> str:
    if lang == "fr":
        parts = [
            f"{years} an{'s' * (years > 1)}" if years else "",
            f"{months} mois" if months else "",
        ]
        return " et ".join(p for p in parts if p)
    parts = [
        f"{years} year{'s' * (years != 1)}" if years else "",
        f"{months} month{'s' * (months != 1)}" if months else "",
    ]
    return " and ".join(p for p in parts if p)


TEMP_TEXT = {
    "en": {
        "states": [
            "Reference month: {start}.\nOffset: {offset} {direction} it.",
            "A record is dated {start}. Another event happened {offset} {direction} that.",
            "Start from {start} and move {offset} {direction_word}.",
        ],
        "after": "after",
        "before": "before",
        "forward": "forward",
        "back": "back",
        "choice": "Which month and year is {offset} {direction} {start}?",
        "noul": "Is {offset} {direction} {start} in {candidate}?",
        "false": "No, it is another month",
        "true": "Yes, that is the month",
    },
    "fr": {
        "states": [
            "Mois de référence : {start}.\nDécalage : {offset} {direction}.",
            "Un document est daté de {start}. Un autre événement a eu lieu {offset} {direction}.",
            "Partir de {start} et avancer de {offset} {direction_word}.",
        ],
        "after": "après",
        "before": "avant",
        "forward": "vers l'avant",
        "back": "vers l'arrière",
        "choice": "Quel mois tombe {offset} {direction} {start} ?",
        "noul": "Le mois situé {offset} {direction} {start} est-il {candidate} ?",
        "false": "Non, c'est un autre mois",
        "true": "Oui, c'est ce mois-là",
    },
}


def tempreason_candidates(start: int, offset: int) -> list[int]:
    """Near misses of start + offset: ±1 / ±2 months, ±1 year, the other direction, and
    the offset's years only / months only (when it has both)."""
    answer = start + offset
    years, months = divmod(abs(offset), 12)
    sign = 1 if offset >= 0 else -1
    near = [answer + d for d in (-1, 1, -2, 2, -12, 12)] + [start - offset]
    if years and months:
        near += [start + sign * 12 * years, start + sign * months]
    return [c for c in dict.fromkeys(near) if c != answer and c >= 0]


def tempreason(row, rng, meta):
    parsed = tempreason_parse(row["question"])
    if parsed is None:
        return []
    start, offset, sign = parsed
    answer = start + offset
    if row["answer"].strip() != f"{MONTHS_ABBR[answer % 12]}, {answer // 12}":
        return []  # the upstream answer disagrees with the arithmetic
    lang = "fr" if rng.random() < TEMPREASON_FR else "en"
    t = TEMP_TEXT[lang]
    years, months = divmod(abs(offset), 12)
    direction = t["after"] if sign > 0 else t["before"]
    fields = {
        "start": _month_text(start, lang),
        "offset": _offset_text(years, months, lang),
        "direction": direction,
        "direction_word": t["forward"] if sign > 0 else t["back"],
    }
    state = t["states"][int(rng.integers(len(t["states"])))].format(**fields)
    near = tempreason_candidates(start, offset)
    if rng.random() < 0.55:
        k = int(rng.integers(3, 6))
        picked = [
            near[int(i)] for i in rng.choice(len(near), size=min(k, len(near)), replace=False)
        ]
        options = sorted([answer, *picked])
        criteria = {_month_label(c): _month_text(c, lang) for c in options}
        question = {
            "type": "choice",
            "instructions": t["choice"].format(**fields),
            "criteria": criteria,
        }
        expected = _month_label(answer)
    else:
        right = rng.random() < 0.5
        candidate = answer if right else near[int(rng.integers(len(near)))]
        question = {
            "type": "noul",
            "instructions": t["noul"].format(**fields, candidate=_month_text(candidate, lang)),
            "criteria": {"false": t["false"], "true": t["true"]},
        }
        expected = "yes" if right else "no"
    return [{"state": state, "question": question, "expected": expected, "lang": lang}]


# --- HelpSteer3 preference (nvidia/HelpSteer3) --------------------------------------------
# {domain, language, context: [{role, content}], response1, response2, overall_preference,
# individual_preference: [{score, reasoning, ...}]}: scores -3 (response 1 much better) ..
# 3 (response 2 much better). English rows only (general and stem); the responses are
# shown in a random order. Half the rows ask which response is better (A / B / tie, the
# annotators' votes as the target), half ask the 7-level comparison (the annotators' score
# distribution as the target).

HELPSTEER3_MAX_CHARS = 10000
HS3_LEVELS = [
    "Response 1 is much better than Response 2",
    "Response 1 is better than Response 2",
    "Response 1 is slightly better than Response 2",
    "The two responses are about the same",
    "Response 2 is slightly better than Response 1",
    "Response 2 is better than Response 1",
    "Response 2 is much better than Response 1",
]


def helpsteer3_load(split: str):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("nvidia/HelpSteer3", f"preference/{split}.jsonl.gz", repo_type="dataset")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("language") != "english":
                continue
            context = r["context"] if isinstance(r["context"], list) else []
            size = sum(len(str(t.get("content", ""))) for t in context)
            size += len(r["response1"]) + len(r["response2"])
            scores = [int(p["score"]) for p in r.get("individual_preference") or []]
            if not context or size > HELPSTEER3_MAX_CHARS or not scores:
                continue
            rows.append(
                {
                    "context": json.dumps(context, ensure_ascii=False),
                    "response1": r["response1"],
                    "response2": r["response2"],
                    "scores": scores,
                    "domain": r["domain"],
                }
            )
    return _dataset(rows)


def _conversation(turns: list[dict]) -> str:
    names = {"user": "User", "assistant": "Assistant", "system": "System"}
    return "\n\n".join(
        f"{names.get(t.get('role'), str(t.get('role')).capitalize())}: {str(t['content']).strip()}"
        for t in turns
    )


def helpsteer3(row, rng, meta):
    scores = [int(s) for s in row["scores"]]
    if not scores or any(abs(s) > 3 for s in scores):
        return []
    first, second = row["response1"].strip(), row["response2"].strip()
    if rng.random() < 0.5:  # the order carries no signal
        first, second, scores = second, first, [-s for s in scores]
    state = (
        f"Conversation:\n{_conversation(json.loads(row['context']))}\n\n"
        f"Response 1:\n{first}\n\nResponse 2:\n{second}"
    )
    n = len(scores)
    if rng.random() < 0.5:
        votes = {
            "response_1": sum(s < 0 for s in scores) / n,
            "response_2": sum(s > 0 for s in scores) / n,
            "tie": sum(s == 0 for s in scores) / n,
        }
        question = {
            "type": "choice",
            "instructions": "Which response better answers the last user turn?",
            "criteria": {
                "response_1": "Response 1 is better",
                "response_2": "Response 2 is better",
                "tie": "The two responses are about equally good",
            },
        }
        labels = list(question["criteria"])
        target = _one_hot_or_soft(labels, [votes[k] for k in labels])
    else:
        dist = [sum(s + 3 == i for s in scores) / n for i in range(7)]
        question = {
            "type": "score",
            "instructions": "How do the two responses compare?",
            "criteria": HS3_LEVELS,
        }
        target = _one_hot_or_soft([str(i) for i in range(7)], dist)
    return [{"state": state, "question": question, **target}]


# --- CUAD (theatticusproject/cuad-qa: the GitHub data.zip its loading script reads) -------
# SQuAD format: one paragraph per contract (the whole contract), 41 questions "Highlight
# the parts (if any) of this contract related to "<Category>" ... Details: <...>", answers
# = every annotated span. Contracts run to ~50k characters, so a row shows one excerpt of
# CUAD_WINDOW characters: for a contract with the clause, a window containing its first
# span at a random offset (yes); for one without it, a random window (no: the contract has
# no annotated span of that category anywhere). Negatives are subsampled to the positives'
# count. Train = the 408 contracts of CUADv1.json not in test.json; held out = test.json.

CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
CUAD_WINDOW = 6000
CUAD_MAX_SPAN = 4000
CUAD_SKIP = frozenset({"Document Name", "Parties", "Agreement Date"})  # (nearly) always present
_CUAD_Q = re.compile(
    r'related to "(.+?)" that should be reviewed by a lawyer\. Details: (.+)$', re.DOTALL
)


def _cuad_zip():
    import urllib.request
    import zipfile

    path = CACHE / "cuad" / "data.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with urllib.request.urlopen(CUAD_URL, timeout=300) as r, tmp.open("wb") as f:
            f.write(r.read())
        tmp.replace(path)
    return zipfile.ZipFile(path)


def cuad_window(context: str, span: tuple[int, int] | None, rng, width: int = CUAD_WINDOW):
    """(start, end) of an excerpt of ``width`` characters: containing ``span`` (at a random
    offset) or anywhere when ``span`` is None; ends moved to whitespace."""
    n = len(context)
    if n <= width:
        return 0, n
    if span is None:
        start = int(rng.integers(0, n - width + 1))
    else:
        lo, hi = max(0, span[1] - width), min(span[0], n - width)
        start = int(rng.integers(lo, hi + 1)) if hi >= lo else max(0, min(span[0], n - width))
    end = min(n, start + width)
    if start > 0:
        space = context.find(" ", start)
        start = space + 1 if 0 <= space < (span[0] if span else end) else start
    if end < n:
        space = context.rfind(" ", start, end)
        end = space if space > (span[1] if span else start) else end
    return start, end


def cuad_load(split: str):
    with _cuad_zip() as z:
        test = json.loads(z.read("test.json"))["data"]
        if split == "test":
            docs = test
        else:
            held = {d["title"] for d in test}
            docs = [d for d in json.loads(z.read("CUADv1.json"))["data"] if d["title"] not in held]
    pos, neg = [], []
    for doc in docs:
        para = doc["paragraphs"][0]
        context = para["context"]
        for qa in para["qas"]:
            m = _CUAD_Q.search(qa["question"])
            if m is None or m.group(1) in CUAD_SKIP:
                continue
            rng = _rng_of(qa["id"])
            spans = sorted(
                (a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]
            )
            span = spans[0] if spans else None
            if span is not None and span[1] - span[0] > CUAD_MAX_SPAN:
                continue
            start, end = cuad_window(context, span, rng)
            row = {
                "id": qa["id"],
                "title": doc["title"],
                "category": m.group(1),
                "details": m.group(2).strip(),
                "excerpt": context[start:end].strip(),
                "cut_left": start > 0,
                "cut_right": end < len(context),
                "present": span is not None,
            }
            (pos if span is not None else neg).append(row)
    keep = len(pos) / max(1, len(neg))
    neg = [r for r in neg if _hash01(r["id"]) < keep]
    return _dataset(sorted(pos + neg, key=lambda r: r["id"]))


def cuad(row, rng, meta):
    title = re.sub(r"[_]+", " ", row["title"]).strip()
    excerpt = (
        ("[…] " if row["cut_left"] else "") + row["excerpt"] + (" […]" if row["cut_right"] else "")
    )
    return [
        {
            "state": f"Contract: {title}\nExcerpt:\n{excerpt}",
            "question": {
                "type": "noul",
                "instructions": f'Does this excerpt contain a clause on "{row["category"]}" '
                f"({row['details'].rstrip('.')})?",
                "criteria": {
                    "false": "The excerpt has no such clause",
                    "true": "The excerpt contains such a clause",
                },
            },
            "expected": "yes" if row["present"] else "no",
        }
    ]


# --- COLD French Law (harvard-lil/cold-french-law) ----------------------------------------
# One row per article in force (LEGI): article_date_debut / article_date_fin (2999-01-01:
# no end), texte_titre, texte_contexte, article_num, article_contenu_text, texte_date_publi
# / texte_date_signature (2999-01-01 when unknown). The question asks whether this version
# applies on a date D; the label follows from the stated dates (debut ≤ D, and D < fin
# when the version has an end). Negatives are dates before entry into force (or on / after
# the end), many of them near misses (the day before, a few days or months before). The
# publication and signature dates, which precede entry into force, are shown as decoys.
# The loader keeps a hash sample of the 2.3 GB CSV (COLD_SAMPLE_SHARE), cached.

COLD_REPO = "harvard-lil/cold-french-law"
COLD_SAMPLE_SHARE = 0.035
COLD_TEXT_CHARS = 2500
COLD_COLUMNS = [
    "article_identifier",
    "article_num",
    "article_etat",
    "article_date_debut",
    "article_date_fin",
    "texte_date_publi",
    "texte_date_signature",
    "texte_nature",
    "texte_titre",
    "texte_contexte",
    "article_contenu_text",
]
NO_END = date(2999, 1, 1)
LATEST = date(2035, 12, 31)  # positives stay before this date


def cold_load(split: str):
    import pandas as pd

    cache = CACHE / "cold-french-law" / f"sample-{COLD_SAMPLE_SHARE}.parquet"
    if not cache.exists():
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(COLD_REPO, "cold-french-law.csv", repo_type="dataset")
        parts = []
        for chunk in pd.read_csv(path, usecols=COLD_COLUMNS, dtype=str, chunksize=50000):
            keep = chunk["article_identifier"].map(
                lambda i: _hash01(f"cold:{i}") < COLD_SAMPLE_SHARE
            )
            parts.append(chunk[keep])
        sample = pd.concat(parts).fillna("")
        cache.parent.mkdir(parents=True, exist_ok=True)
        sample.to_parquet(cache.with_suffix(".tmp"), index=False)
        cache.with_suffix(".tmp").replace(cache)
    df = pd.read_parquet(cache)
    df = df[(df["article_contenu_text"].str.strip() != "") & (df["texte_titre"] != "")]
    return _dataset(df.to_dict("records"))


def _iso(text: str) -> date | None:
    try:
        return date.fromisoformat(str(text)[:10])
    except ValueError:
        return None


def date_fr(d: date, style: int) -> str:
    """A date in French: "1er mars 2024" (0), "01/03/2024" (1), "2024-03-01" (2)."""
    if style == 1:
        return d.strftime("%d/%m/%Y")
    if style == 2:
        return d.isoformat()
    day = "1er" if d.day == 1 else str(d.day)
    return f"{day} {MONTHS_FR[d.month - 1]} {d.year}"


def _shift(d: date, rng, sign: int) -> date | None:
    """d moved by a near-miss offset in direction ``sign`` (never 0 days)."""
    u = rng.random()
    if u < 0.2:
        days = 1
    elif u < 0.5:
        days = int(rng.integers(2, 32))
    elif u < 0.75:
        days = int(rng.integers(32, 366))
    else:
        days = int(rng.integers(366, 365 * 15))
    try:
        return d + timedelta(days=sign * days)
    except OverflowError:
        return None


def cold_query_date(debut: date, fin: date | None, rng) -> tuple[date, bool] | None:
    """(D, applicable): applicable iff debut ≤ D and (no end or D < fin)."""
    if rng.random() < 0.5:  # applicable: the start day itself, or after it
        last = min(fin - timedelta(days=1), LATEST) if fin else LATEST
        d = debut if rng.random() < 0.15 else _shift(debut, rng, 1)
        if d is None or d > last:
            d = debut + (last - debut) * float(rng.random()) if last >= debut else None
        return (d, True) if d is not None else None
    if fin is not None and rng.random() < 0.3:  # on or after the end
        d = fin if rng.random() < 0.3 else _shift(fin, rng, 1)
    else:
        d = _shift(debut, rng, -1)
    return (d, False) if d is not None and d.year >= 1000 else None


def cold_law(row, rng, meta):
    debut, fin = _iso(row["article_date_debut"]), _iso(row["article_date_fin"])
    if debut is None or debut.year < 1000:
        return []
    fin = None if fin is None or fin >= NO_END else fin
    if fin is not None and fin <= debut:
        return []
    query = cold_query_date(debut, fin, rng)
    if query is None:
        return []
    when, applicable = query
    style = lambda: int(rng.integers(3))  # noqa: E731
    context = [line.strip() for line in str(row["texte_contexte"]).splitlines() if line.strip()]
    text = re.sub(r"\n{3,}", "\n\n", str(row["article_contenu_text"]).strip())
    if len(text) > COLD_TEXT_CHARS:
        text = text[:COLD_TEXT_CHARS].rsplit(" ", 1)[0] + " […]"
    num = str(row["article_num"]).strip()
    lines = [f"{row['texte_titre'].strip()} — Article {num}" if num else row["texte_titre"].strip()]
    if context:
        lines.append(" > ".join(context[-2:]))
    validity = f"Version en vigueur à compter du {date_fr(debut, style())}"
    validity += (
        f" ; elle cesse de s'appliquer le {date_fr(fin, style())}."
        if fin
        else " (sans date de fin)."
    )
    lines.append(validity)
    for key, what in (("texte_date_signature", "signé"), ("texte_date_publi", "publié")):
        d = _iso(row[key])
        if d is not None and d < NO_END and d != debut and rng.random() < 0.7:
            lines.append(f"Texte {what} le {date_fr(d, style())}.")
    lines.append(f"Texte de l'article :\n{text}")
    return [
        {
            "state": "\n".join(lines),
            "question": {
                "type": "noul",
                "instructions": "D'après ses dates de vigueur, cette version de l'article "
                f"s'applique-t-elle à la date du {date_fr(when, style())} ?",
                "criteria": {
                    "false": "Non : à cette date, cette version n'est pas (ou plus) en vigueur",
                    "true": "Oui : à cette date, cette version est en vigueur (le jour de "
                    "l'entrée en vigueur compris)",
                },
            },
            "expected": "yes" if applicable else "no",
        }
    ]


# --- ArGiMi French jurisprudence (artefactory/Argimi-Legal-French-Jurisprudence) ----------
# The "juri" config (Cour de cassation): content (HTML), title, solution ("REJET.",
# "Cassation", "CASSATION PARTIELLE.", "Irrecevabilité", or several for joined appeals).
# The state is the decision from "a rendu l'arrêt suivant" (the header above it may name
# the outcome) up to "PAR CES MOTIFS" (the operative part and everything after it, such
# as the annexed grounds, are cut). Rows where an outcome phrase remains are dropped.
# Long decisions keep their start and their end (the court's answer) around "[…]".

ARGIMI_REPO = "artefactory/Argimi-Legal-French-Jurisprudence"
ARGIMI_FILES = tuple(f"cass_chunks/cass_{i}_over_6.parquet" for i in range(1, 7))
ARGIMI_SAMPLE_SHARE = 0.12
ARGIMI_HEAD, ARGIMI_TAIL = 2000, 6500
ARGIMI_OUTCOMES = {
    "REJET": "rejet",
    "CASSATION": "cassation",
    "CASSATION SANS RENVOI": "cassation",
    "CASSATION PARTIELLE": "cassation_partielle",
    "CASSATION PARTIELLE SANS RENVOI": "cassation_partielle",
    "IRRECEVABILITE": "irrecevabilite",
}
ARGIMI_CRITERIA = {
    "rejet": "Rejet : le pourvoi est rejeté, la décision attaquée est maintenue",
    "cassation": "Cassation : la décision attaquée est cassée et annulée",
    "cassation_partielle": "Cassation partielle : la décision attaquée n'est cassée qu'en partie",
    "irrecevabilite": "Irrecevabilité : le pourvoi est déclaré irrecevable",
}
ARGIMI_LEAK = re.compile(
    r"PAR CES MOTIFS|CASSE ET ANNULE|REJETTE (?:le|les|la) (?:pourvois?|demandes?)|"
    r"DÉCLARE (?:le pourvoi )?IRRECEVABLE|"
    r"^\s*(?:REJET|CASSATION|IRRECEVABILIT[EÉ])\b|"
    r"^\s*(?i:rejet|cassation(?: partielle)?(?: sans renvoi)?|irrecevabilit[ée])\s*\.?\s*$",
    re.MULTILINE,
)


# an outcome word opening the decision (the old criminal-chamber format)
ARGIMI_OUTCOME_HEAD = re.compile(
    r"\s*(?:REJET|CASSATION|IRRECEVABILIT[EÉ]|IRRECEVABILITE|NON-LIEU|ANNULATION)\b"
)


def argimi_outcome(solution: str) -> str | None:
    import unicodedata

    key = unicodedata.normalize("NFKD", str(solution)).encode("ascii", "ignore").decode()
    key = re.sub(r"[\s.;,]+", " ", key).strip().upper()
    return ARGIMI_OUTCOMES.get(key)


def html_text(content: str) -> str:
    text = re.sub(r"<br[^>]*>|</p>|<p[^>]*>", "\n", content)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def argimi_text(content: str) -> str | None:
    """The decision without its header and its operative part, or None (no "PAR CES
    MOTIFS", or an outcome phrase left in the text)."""
    text = html_text(content)
    head = re.search(r"a rendu l['’]arr[êe]t suivant\s*:?", text, re.IGNORECASE)
    if head is not None:
        text = text[head.end() :]
    elif ARGIMI_OUTCOME_HEAD.match(text):  # old format: "CASSATION sur le pourvoi … LA COUR,"
        court = re.search(r"^\s*LA COUR\s*,", text, re.MULTILINE)
        text = text[court.start() :] if court is not None else text
    cut = re.search(r"PAR CES MOTIFS", text, re.IGNORECASE)
    if cut is None:
        return None
    text = text[: cut.start()].strip()
    if len(text) < 300 or ARGIMI_LEAK.search(text):
        return None
    if len(text) > ARGIMI_HEAD + ARGIMI_TAIL:
        head_part = text[:ARGIMI_HEAD].rsplit(" ", 1)[0]
        tail_part = text[-ARGIMI_TAIL:].split(" ", 1)[-1]
        text = f"{head_part}\n[…]\n{tail_part}"
    return text


def argimi_load(split: str):
    import pandas as pd

    cache = CACHE / "argimi" / f"juri-v2-{ARGIMI_SAMPLE_SHARE}.parquet"
    if not cache.exists():
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        rows = []
        for name in ARGIMI_FILES:
            path = hf_hub_download(ARGIMI_REPO, name, repo_type="dataset")
            table = pq.read_table(path, columns=["id", "title", "content", "solution"])
            for r in table.to_pylist():
                if _hash01(f"argimi:{r['id']}") >= ARGIMI_SAMPLE_SHARE:
                    continue
                outcome = argimi_outcome(r["solution"] or "")
                text = argimi_text(r["content"] or "") if outcome else None
                if text is not None:
                    rows.append(
                        {"id": r["id"], "title": r["title"] or "", "text": text, "outcome": outcome}
                    )
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(cache.with_suffix(".tmp"), index=False)
        cache.with_suffix(".tmp").replace(cache)
    return _dataset(pd.read_parquet(cache).to_dict("records"))


def argimi(row, rng, meta):
    if row["outcome"] not in ARGIMI_CRITERIA:
        return []
    title = re.sub(r",?\s*(Publié|Inédit)[^,]*$", "", row["title"]).strip()
    return [
        {
            "state": f"{title}\n\n{row['text']}" if title else row["text"],
            "question": {
                "type": "choice",
                "instructions": "Au vu des motifs, quelle est l'issue de ce pourvoi devant la "
                "Cour de cassation ?",
                "criteria": ARGIMI_CRITERIA,
            },
            "expected": row["outcome"],
        }
    ]


# --- Negation twins ----------------------------------------------------------------------
# A twin asks the negated question of an exact-label noul row: the same state, the
# instructions rewritten to their negation, the two criteria swapped, the label flipped.
# The model is taught that P(yes | negated) = 1 − P(yes | original).


def negated(base: Adapter, negate) -> Adapter:
    """The base adapter's exact noul rows, negated (``negate``: instructions → negated
    instructions); other rows are dropped."""

    def adapter(row, rng, meta):
        out = []
        for item in base(row, rng, meta):
            q = item["question"]
            if q["type"] != "noul" or item.get("expected") not in ("no", "yes"):
                continue
            crit = q["criteria"]
            out.append(
                {
                    **item,
                    "question": {
                        **q,
                        "instructions": negate(q["instructions"]),
                        "criteria": {"false": crit["true"], "true": crit["false"]},
                    },
                    "expected": "no" if item["expected"] == "yes" else "yes",
                }
            )
        return out

    return adapter


def _replace_prefix(prefix: str, new: str):
    def negate(instructions: str) -> str:
        if not instructions.startswith(prefix):
            raise ValueError(f"unexpected instructions {instructions[:40]!r}")
        return new + instructions[len(prefix) :]

    return negate


TWIN_NEGATIONS = {
    "twin_squad2": _replace_prefix(
        "Does the passage contain the answer to: ",
        "Is the answer missing from the passage, for: ",
    ),
    "twin_piaf_fr": _replace_prefix(
        "Le passage contient-il la réponse à : ",
        "La réponse est-elle absente du passage, pour : ",
    ),
    "twin_boolq": _replace_prefix(
        "Based only on the text: ",
        "Based only on the text, is the answer no to: ",
    ),
}


TARGETED_SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source(
            "eikos",
            EIKOS_REPO,
            "core",
            "train",
            "validation",
            eikos,
            "eikos",
            "en",
            "CC BY 4.0",
            True,
            cap=12000,
            load=eikos_load,
        ),
        Source(
            "eikos_long",
            EIKOS_REPO,
            "long_context",
            "train",
            "validation",
            eikos_long,
            "long_policy",
            "en",
            "CC BY 4.0",
            True,
            cap=1500,
            load=eikos_long_load,
        ),
        Source(
            "procedural",
            PROC_REPO,
            None,
            "train",
            "validation",
            procedural,
            "procedural",
            "en",
            "Apache-2.0",
            True,
            cap=15000,
            load=procedural_load,
        ),
        Source(
            "tempreason",
            TEMPREASON_REPO,
            "l1",
            "train",
            "val",
            tempreason,
            "temporal_numeric",
            "en",
            "CC BY-SA 3.0",
            True,
            cap=10000,
            load=tempreason_load,
        ),
        Source(
            "helpsteer3",
            "nvidia/HelpSteer3",
            "preference",
            "train",
            "validation",
            helpsteer3,
            "judge_hard",
            "en",
            "CC BY 4.0",
            True,
            cap=8000,
            load=helpsteer3_load,
        ),
        Source(
            "cuad",
            "theatticusproject/cuad-qa",
            None,
            "train",
            "test",
            cuad,
            "contract_clause",
            "en",
            "CC BY 4.0",
            True,
            cap=8000,
            load=cuad_load,
        ),
        Source(
            "cold_law_fr",
            COLD_REPO,
            None,
            "train",
            None,
            cold_law,
            "temporal_law",
            "fr",
            "CC BY 4.0",
            True,
            cap=8000,
            load=cold_load,
            group=lambda r: r["texte_titre"],
        ),
        Source(
            "argimi_fr",
            ARGIMI_REPO,
            "juri",
            "train",
            None,
            argimi,
            "legal_outcome",
            "fr",
            "CC BY-SA 4.0",
            True,
            cap=6000,
            load=argimi_load,
            group=lambda r: r["id"],
        ),
        Source(
            "twin_squad2",
            "rajpurkar/squad_v2",
            None,
            "train",
            "validation",
            negated(squad_abstain("en"), TWIN_NEGATIONS["twin_squad2"]),
            "abstain_twin",
            "en",
            "CC BY-SA 4.0",
            True,
            cap=4000,
        ),
        Source(
            "twin_boolq",
            "google/boolq",
            None,
            "train",
            "validation",
            negated(boolq, TWIN_NEGATIONS["twin_boolq"]),
            "reading_twin",
            "en",
            "CC BY-SA 3.0",
            True,
            cap=3000,
        ),
        Source(
            "twin_piaf_fr",
            "etalab-ia/piaf",
            None,
            "train",
            None,
            negated(squad_abstain("fr"), TWIN_NEGATIONS["twin_piaf_fr"]),
            "abstain_twin",
            "fr",
            "MIT (card tag)",
            True,
            cap=2000,
            load=piaf_load,
            group=lambda r: r["title"],
        ),
    ]
}
