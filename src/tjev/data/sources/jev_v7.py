"""mix-v7 typed-decision corpora (blocks "jev", "targeted" and "breadth"): expansions of
the mix-v6 contender / targeted sources and new JevBench-style decision sets.

==================  ==========  ==========================================================
source              block       what it is
==================  ==========  ==========================================================
``openjev_x``       jev         Open-Jev v1.1 rows v6's ``openjev`` did not draw (a 10k
                                WANLI slice from the complement of v6's hash sample) and
                                the English rows of Open-Jev v1 ``browser-drone-expansion``
                                (5k per generator); train only (held out: ``openjev``)
``openjev_ctrl``    jev         the ten Open-Jev v1 control corpora (citation, entity,
                                amount / email / phone extraction, context retention,
                                sponsor segments, silent failures, IR, mailroom), English,
                                ≤ 4k train rows each, rows already in v1.1 left out
``certo_x``         jev         certo-decisions-v2 rule / posterior rows after v6's
                                ``certo`` sample, round robin over task families (each ≤ 8%)
``procedural_x``    targeted    procedural-typed-decisions states after v6's first 800 per
                                config (hash order); train only (held out: ``procedural``)
``eikos_x``         targeted    eikos core English teacher-labelled choice / score rows (v6
                                kept the program / exact rows and the teacher noul rows)
``kaivoss_s1``      jev         kaivoss/system-one-270m-data (gpt-oss-20b ensemble): soft
                                targets on the rows whose measured ambiguity is not "clear"
``n4ze3m_td``       jev         n4ze3m/typed-decisions-synth (DeepSeek, 3-read soft labels)
``helmo_td``        jev         helmo/synthetic-typed-decisions (DeepSeek, one question)
``mlagnajit_td``    jev         m-lagnajit/calibrated-typed-decisions: human gold of intent /
                                emotion / polarity sets, a calibrated Qwen2.5 teacher
``tasksource_jev``  breadth     tasksource-jev-typed-decisions ``full``: families we do not
                                already train on, a √n quota per source (≤ 1.5% each),
                                per-row licence and licence use kept
==================  ==========  ==========================================================

None of these was labelled by Jev (Jev's licence forbids distillation): see LICENCES.
Every source here sets ``Source.block`` and ``Source.license_use``; licences are in
``LICENCES`` (merged into :mod:`tjev.data.licences`).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
from collections import Counter, defaultdict

import numpy as np

from tjev.data.item import rendered_state
from tjev.data.sources.base import Source
from tjev.data.sources.contenders import (
    CERTO_REPO,
    CERTO_SLICES,
    NOUL_MAP,
    OPENJEV_CONFIG,
    OPENJEV_REPO,
    _one_hot_or_soft,
    certo,
    certo_load,
    openjev,
    openjev_load,
    openjev_options,
)
from tjev.data.sources.targeted import (
    PROC_CONFIGS,
    PROC_MAX_STATE_CHARS,
    PROC_PER_CONFIG,
    PROC_REPO,
    _eikos_row,
    _eikos_rows,
    eikos,
    procedural,
)


def _hash01(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16**12


def _hkey(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _seed_of(name: str) -> int:
    """tjev.data.mix._seed_of (not imported: mix imports the source registries)."""
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def _dataset(rows: list[dict]):
    from datasets import Dataset

    return Dataset.from_list(rows)


def _repo_files(repo: str, pattern: str) -> list[str]:
    """Local paths of the files of a dataset repo matching ``pattern`` (fnmatch), sorted."""
    from huggingface_hub import HfApi, hf_hub_download

    names = sorted(
        f for f in HfApi().list_repo_files(repo, repo_type="dataset") if fnmatch.fnmatch(f, pattern)
    )
    if not names:
        raise FileNotFoundError(f"{repo}: no file matches {pattern}")
    return [hf_hub_download(repo, f, repo_type="dataset") for f in names]


# --- Open-Jev expansions (ZefanCai/Open-Jev-v1.1, ZefanCai/Open-Jev) ----------------------
# Same columns as v6's openjev (contenders.openjev): kind, question, options, target,
# state_json, source, group_id, metadata_json (only its ``language`` is read: English rows).

OPENJEV_V1_REPO = "ZefanCai/Open-Jev"
OPENJEV_V1_EXPANSION = "browser-drone-expansion-v1-redistributable"
OPENJEV_CONTROLS = (
    "citation-control-v1",
    "entity-alignment-control-v1",
    "amount-extraction-control-v1",
    "email-selection-control-v1",
    "phone-extraction-control-v1",
    "context-retention-control-v1",
    "sponsor-segment-control-v1",
    "silent-failure-control-v1",
    "ir-control-v1",
    "mailroom-control-v1",
)
OPENJEV_X_WANLI = 10000  # extra WANLI rows (an NLI family we already have)
OPENJEV_X_PER_SOURCE = 5000  # rows per generator of the v1 expansion
OPENJEV_CTRL_PER_CONFIG = 4000
OPENJEV_COLUMNS = ["id", "group_id", "source", "kind", "question", "options", "target"]
# v6's openjev draw: Builder.public with seed 0, scale 1, cap 40000 (mix.HELD_OUT_PER_PART 300)
OPENJEV_V6 = {"seed": 0, "cap": 40000, "held_out_per_part": 300}


def _openjev_parquet(repo: str, config: str, split: str, *, ids_only: bool = False) -> list[dict]:
    """English rows (metadata language en or absent) of one Open-Jev parquet split."""
    import pyarrow.parquet as pq

    rows = []
    for path in _repo_files(repo, f"data/{config}/{split}-*.parquet"):
        columns = ["id"] if ids_only else [*OPENJEV_COLUMNS, "state_json", "metadata_json"]
        if ids_only:
            rows += pq.read_table(path, columns=columns).to_pylist()
            continue
        for r in pq.read_table(path, columns=columns).to_pylist():
            language = json.loads(r.pop("metadata_json") or "{}").get("language")
            if language in (None, "en"):
                rows.append(r)
    return rows


def openjev_v6_ids(part: str = "train") -> set[str]:
    """Ids of the rows v6's openjev drew: from its train pool (the candidates its Builder
    adapted, whatever the hygiene later kept) or, ``part="eval"``, from v1.1 validation
    (its held-out candidates)."""
    pool, ev = openjev_load("train"), openjev_load("validation")
    rng = np.random.default_rng([OPENJEV_V6["seed"], _seed_of("openjev")])
    held = rng.permutation(len(ev))[: 3 * OPENJEV_V6["held_out_per_part"] * 2]
    if part == "eval":
        ids = ev["id"]
        return {ids[int(i)] for i in held}
    drawn = rng.permutation(len(pool))[: int(OPENJEV_V6["cap"] * 1.2)]
    ids = pool["id"]
    return {ids[int(i)] for i in drawn}


def _per_source(rows: list[dict], per: int, key: str = "group_id") -> list[dict]:
    """At most ``per`` rows per ``source``, whole groups first in hash order."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (_hkey(f"ojx:{r[key]}"), r["id"])):
        if len(by[r["source"]]) < per:
            by[r["source"]].append(r)
    return [r for s in sorted(by) for r in by[s]]


def openjev_x_select(
    v11: list[dict], v6_ids: set[str], expansion: list[dict], v11_ids: set[str]
) -> list[dict]:
    """openjev_x rows: v1.1 rows v6 did not draw (WANLI: ≤ OPENJEV_X_WANLI from the hash
    complement of v6's group sample, keep < share), then the v1 expansion rows not in v1.1,
    ≤ OPENJEV_X_PER_SOURCE per generator."""
    rest = [r for r in v11 if r["id"] not in v6_ids]
    wanli = [r for r in rest if r["source"].startswith("wanli")]
    other = [r for r in rest if not r["source"].startswith("wanli")]
    n_wanli = sum(r["source"].startswith("wanli") for r in v11)
    n_other = len(v11) - n_wanli
    keep = min(1.0, 0.10 / 0.90 * n_other / max(1, n_wanli))  # v6's openjev_load share
    wanli = [r for r in wanli if _hash01(r["group_id"]) >= keep]  # never in v6's pool
    wanli.sort(key=lambda r: _hkey(f"ojx:{r['group_id']}"))
    extra = [r for r in expansion if r["id"] not in v11_ids]
    return other + wanli[:OPENJEV_X_WANLI] + _per_source(extra, OPENJEV_X_PER_SOURCE)


def openjev_x_load(split: str):
    v11 = _openjev_parquet(OPENJEV_REPO, OPENJEV_CONFIG, "train")
    expansion = _openjev_parquet(OPENJEV_V1_REPO, OPENJEV_V1_EXPANSION, "train")
    rows = openjev_x_select(v11, openjev_v6_ids(), expansion, {r["id"] for r in v11})
    return _dataset(rows)


def _v11_all_ids() -> set[str]:
    ids = set()
    for split in ("train", "calibration", "validation", "test", "ood"):
        ids |= {
            r["id"] for r in _openjev_parquet(OPENJEV_REPO, OPENJEV_CONFIG, split, ids_only=True)
        }
    return ids


def openjev_ctrl_load(split: str):
    """The control configs: train ≤ OPENJEV_CTRL_PER_CONFIG rows each (whole groups, hash
    order) without the rows v1.1 replays (any split: v6's openjev may hold them); held out =
    their validation splits without v6's openjev held-out candidates (v1.1 keeps the
    controls' split assignments, so its validation shares rows with theirs)."""
    in_v11 = _v11_all_ids() if split == "train" else openjev_v6_ids("eval")
    rows = []
    for config in OPENJEV_CONTROLS:
        part = [
            r for r in _openjev_parquet(OPENJEV_V1_REPO, config, split) if r["id"] not in in_v11
        ]
        if split == "train":
            part = _per_source(part, OPENJEV_CTRL_PER_CONFIG)
        rows += part
    return _dataset(rows)


# --- certo-decisions-v2 beyond v6's sample ------------------------------------------------

CERTO_X_ROWS = 60000  # asked for; the rule families v6 left yield ~19.6k (3 families)
CERTO_X_FAMILY_SHARE = 0.08
# the exact-posterior family (soft targets) may take more: v6 kept 5k of its 60k rows
CERTO_X_FAMILY_ROWS = {"known_posterior": 10000}


def certo_x_select(rows: list[dict], v6_ids: set[str], n: int = CERTO_X_ROWS) -> list[dict]:
    """Distinct (state, instructions, rubric) rows not drawn by v6, round robin over task
    families in hash order, each family at most CERTO_X_FAMILY_SHARE of ``n``."""
    rows = sorted(rows, key=lambda r: _hkey(r["record_id"]))
    seen, families = set(), defaultdict(list)
    for r in rows:
        q = json.loads(r["question"])
        key = (r["state"], str(q.get("instructions")), str(q.get("rubric")))
        if key in seen:
            continue
        seen.add(key)
        if r["record_id"] not in v6_ids:
            families[r["task_family"]].append(r)
    per = max(1, int(n * CERTO_X_FAMILY_SHARE))
    queues = [families[f][: CERTO_X_FAMILY_ROWS.get(f, per)] for f in sorted(families)]
    out, depth = [], 0
    while len(out) < n and any(depth < len(q) for q in queues):
        out += [q[depth] for q in queues if depth < len(q)][: n - len(out)]
        depth += 1
    return out


def certo_x_load(split: str):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    v6 = certo_load(split)
    v6_ids = set(v6["record_id"])
    path = hf_hub_download(CERTO_REPO, f"{split}.parquet", repo_type="dataset")
    columns = ["record_id", "task_family", "source_id", "question_type", "state", "question",
               "target"]  # fmt: skip
    table = pq.read_table(
        path,
        columns=columns,
        filters=[
            ("source_id", "in", list(CERTO_SLICES)),
            ("question_type", "!=", "independent_binary"),
        ],
    )
    return _dataset(certo_x_select(table.to_pylist(), v6_ids))


# --- procedural-typed-decisions after v6's states ----------------------------------------

PROC_X_STATES = 1700  # states per config after v6's first PROC_PER_CONFIG["train"]


def procedural_x_load(split: str):
    from datasets import load_dataset

    first = PROC_PER_CONFIG["train"]
    rows = []
    for config in PROC_CONFIGS:
        ds = load_dataset(PROC_REPO, config, split="train")
        ds = ds.select_columns(["id", "level", "state", "questions", "answers"])
        picked = sorted(
            (r for r in ds if len(r["state"]) <= PROC_MAX_STATE_CHARS),
            key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest(),
        )[first : first + PROC_X_STATES]
        rows += [{**r, "level": str(r["level"]), "task": config} for r in picked]
    return _dataset(rows)


# --- eikos core: the teacher-labelled choice / score rows v6 left out ---------------------


def eikos_x_keep(r: dict) -> bool:
    return (
        r.get("lang") == "English"
        and r["label_source"] == "teacher"
        and r["question_type"] in ("choice", "score")
    )


def eikos_x_load(split: str):
    return _dataset(
        [_eikos_row(r) for r in _eikos_rows(f"data/core/{split}.jsonl") if eikos_x_keep(r)]
    )


# --- kaivoss/system-one-270m-data ----------------------------------------------------------
# One row per question: ``prompt`` = "<state>\n…\n</state>\n\nQuestion: …\nOptions:\nA. label -
# description\n…\n\nAnswer with one letter.\nAnswer:", ``target`` aligned with the letters,
# ``qtype``, ``ambiguity`` (measured: clear / borderline / ambiguous), ``state_id``. Rows
# measured "clear" are one-hot (their 1e-4 floors are an artefact of the logprob reads);
# the others keep the ensemble's distribution.

_KV_PROMPT = re.compile(
    r"^<state>\n(?P<state>.*)\n</state>\s*\nQuestion: (?P<question>.*?)\nOptions:\n"
    r"(?P<options>.*?)\n\s*\nAnswer with one letter\.",
    re.DOTALL,
)
_KV_OPTION = re.compile(r"^([A-Z])\. (.*)$")


def kaivoss_parse(prompt: str) -> tuple[str, str, list[tuple[str, str]]] | None:
    """(state, question, [(label, description)]) of a rendered prompt, or None."""
    m = _KV_PROMPT.match(prompt.strip())
    if m is None:
        return None
    options, current = [], None
    for line in m.group("options").split("\n"):
        o = _KV_OPTION.match(line)
        if o is not None and o.group(1) == chr(ord("A") + len(options)):
            label, _, desc = o.group(2).partition(" - ")
            current = [label.strip(), desc.strip()]
            options.append(current)
        elif current is not None:  # a description running over several lines
            current[1] = f"{current[1]}\n{line}".strip()
        else:
            return None
    return m.group("state").strip(), m.group("question").strip(), [tuple(o) for o in options]


def kaivoss(row, rng, meta):
    parsed = kaivoss_parse(row["prompt"])
    if parsed is None:
        return []
    state, question, options = parsed
    probs = [float(p) for p in row["target"]]
    if len(probs) != len(options) or len(options) < 2:
        return []
    kind = row["qtype"]
    labels = [lab for lab, _ in options]
    if kind == "noul":
        named = {
            NOUL_MAP.get(lab.lower()): (d, p) for (lab, d), p in zip(options, probs, strict=True)
        }
        if set(named) != {"no", "yes"}:
            return []
        criteria = {"false": named["no"][0], "true": named["yes"][0]}
        labels, probs = ["no", "yes"], [named["no"][1], named["yes"][1]]
    elif kind == "choice":
        if len(set(labels)) != len(labels):
            return []
        criteria = dict(options)
    elif kind == "score":  # ordinal levels in their given order
        criteria = [f"{lab}: {d}" if d and d != lab else lab for lab, d in options]
        labels = [str(i) for i in range(len(options))]
    else:
        return []
    if row.get("ambiguity") == "clear":
        target = {"expected": labels[int(np.argmax(probs))]}
    else:
        target = _one_hot_or_soft(labels, probs)
    return [
        {
            "state": state,
            "question": {"type": kind, "instructions": question, "criteria": criteria},
            **target,
            "family": "kaivoss_"
            + re.sub(r"\W+", "_", str(row.get("domain") or "other")).strip("_"),
        }
    ]


# --- n4ze3m/typed-decisions-synth ----------------------------------------------------------
# One row per case: questions {qid: {type, instructions, criteria}} (noul criteria
# {true, false}, choice a dict, score a list), gold {qid: key | level | bool}, teacher
# {qid: {probabilities} | {noul}} (the mean of three blind reads by the writer model, kept
# only when it agrees with gold). Target: the teacher's distribution.


def _noul_criteria(criteria) -> dict:
    given = criteria if isinstance(criteria, dict) else {}
    return {
        "false": str(given.get("false", given.get("no", "")) or ""),
        "true": str(given.get("true", given.get("yes", "")) or ""),
    }


def n4ze3m(row, rng, meta):
    state = row["state"]
    if row.get("state_is_json"):
        try:
            state = rendered_state(json.loads(state))
        except ValueError:
            return []
    questions, gold = json.loads(row["questions"]), json.loads(row["gold"])
    teacher = json.loads(row.get("teacher") or "{}")
    out = []
    for qid, q in questions.items():
        kind, g, t = q.get("type"), gold.get(qid), teacher.get(qid) or {}
        if g is None:
            continue
        if kind == "noul":
            criteria = _noul_criteria(q.get("criteria"))
            labels = ["no", "yes"]
            gold_label = "yes" if g is True or str(g).lower() == "true" else "no"
            p = t.get("noul")
            probs = [1 - float(p), float(p)] if p is not None else None
        elif kind == "choice" and isinstance(q.get("criteria"), dict):
            criteria = {str(k): str(v) for k, v in q["criteria"].items()}
            labels, gold_label = list(criteria), str(g)
            dist = t.get("probabilities")
            probs = [float(dist.get(k, 0.0)) for k in labels] if dist else None
        elif kind == "score" and isinstance(q.get("criteria"), list):
            criteria = [str(c) for c in q["criteria"]]
            labels, gold_label = [str(i) for i in range(len(criteria))], str(int(g))
            dist = t.get("probabilities")
            probs = [float(dist.get(k, 0.0)) for k in labels] if dist else None
        else:
            continue
        if gold_label not in labels:
            continue
        if probs and 0.98 <= sum(probs) <= 1.02 and labels[int(np.argmax(probs))] == gold_label:
            target = _one_hot_or_soft(labels, [p / sum(probs) for p in probs])
        else:
            target = {"expected": gold_label}
        out.append(
            {
                "state": state,
                "question": {"type": kind, "instructions": q["instructions"], "criteria": criteria},
                **target,
                "family": f"n4ze3m_{row.get('generator') or 'synth'}",
            }
        )
    return out


# --- helmo/synthetic-typed-decisions ---------------------------------------------------------
# One question per row (q1): choice {probabilities: {B: 1.0}}, score {mean, variance} (a
# fractional mean is split between its two nearest levels), noul {noul: p}.


def _score_mean_target(mean: float, levels: int) -> list[float]:
    mean = min(max(float(mean), 0.0), levels - 1.0)
    lo = math.floor(mean)
    hi = min(lo + 1, levels - 1)
    probs = [0.0] * levels
    frac = mean - lo
    probs[lo] += 1.0 - frac
    probs[hi] += frac
    return probs


def helmo(row, rng, meta):
    q = json.loads(row["questions"]).get("q1")
    g = json.loads(row["gold"]).get("q1")
    if not q or not g:
        return []
    kind = q.get("type")
    if kind == "noul":
        p = float(g["noul"])
        if not 0.0 <= p <= 1.0:
            return []
        criteria, labels, probs = _noul_criteria(q.get("criteria")), ["no", "yes"], [1 - p, p]
    elif kind == "choice" and isinstance(q.get("criteria"), dict):
        criteria = {str(k): str(v) for k, v in q["criteria"].items()}
        labels = list(criteria)
        dist = g.get("probabilities") or {}
        if set(dist) - set(labels):
            return []
        probs = [float(dist.get(k, 0.0)) for k in labels]
    elif kind == "score" and isinstance(q.get("criteria"), list) and len(q["criteria"]) >= 2:
        criteria = [str(c) for c in q["criteria"]]
        labels = [str(i) for i in range(len(criteria))]
        probs = _score_mean_target(g["mean"], len(criteria))
    else:
        return []
    if not 0.98 <= sum(probs) <= 1.02:
        return []
    return [
        {
            "state": row["state"],
            "question": {"type": kind, "instructions": q["instructions"], "criteria": criteria},
            **_one_hot_or_soft(labels, probs),
            "family": "helmo_synthetic",
        }
    ]


# --- m-lagnajit/calibrated-typed-decisions ---------------------------------------------------
# type choice | noul (options ["false", "true"]), options in canonical order, gold (human,
# from the source dataset), teacher_probs (Qwen2.5-7B, temperature / shift calibrated per
# (source, type)) aligned with options. Target: the human gold; the teacher rides along as
# the row's ``teacher`` (loss.teacher_*).


def mlagnajit(row, rng, meta):
    options = [str(o) for o in row["options"]]
    probs = [float(p) for p in row.get("teacher_probs") or []]
    kind, gold = row["type"], str(row["gold"])
    if gold not in options:
        return []
    if kind == "noul":
        if set(options) != {"false", "true"}:
            return []
        criteria = {"false": "", "true": ""}
        labels = ["no", "yes"]
        expected = NOUL_MAP[gold]
        teacher = dict(zip([NOUL_MAP[o] for o in options], probs, strict=True)) if probs else None
    elif kind == "choice":
        if len(set(options)) != len(options):
            return []
        criteria, labels, expected = dict.fromkeys(options, ""), options, gold
        teacher = dict(zip(options, probs, strict=True)) if len(probs) == len(options) else None
    else:
        return []
    out = {
        "state": row["state"],
        "question": {"type": kind, "instructions": row["instructions"], "criteria": criteria},
        "expected": expected,
        "family": f"mlagnajit_{str(row['source']).split('/')[-1]}",
    }
    if teacher and 0.98 <= sum(teacher.values()) <= 1.02 and set(teacher) <= set(labels):
        out["teacher"] = teacher
    return [out]


# --- tasksource/tasksource-jev-typed-decisions (full) ----------------------------------------
# Flat rows: state, question, kind (choice | score | noul), options ([] for noul), target
# (distribution over options; [p] = P(yes) for noul), source (the tasksource task), split
# (the source's own split), group_id, license, license_use. Families that duplicate a mix
# source (or a v7 source), eval-only sets and non-English / non-French tasks are dropped
# (TS_DROP, first rule wins); each remaining source gets a quota ∝ √n (at most
# TS_SOURCE_SHARE of the rows), whole groups drawn in hash order.

TS_REPO = "tasksource/tasksource-jev-typed-decisions"
TS_ROWS = 330000  # drawn (the builder caps at 300k after its own drops)
TS_SOURCE_SHARE = 0.015
# English and French tasks among tasksource's multilingual/* (the others are dropped)
TS_MULTI_KEEP = re.compile(
    r"^multilingual/(masakhanews/(eng|fra)|NusaX-senti/eng|xcsr/X-(CODAH|CSQA)-(en|fr)|"
    r"xlwic/xlwic_fr_fr|disrpt/fra\..*|xstory_cloze/en)$"
)
TS_FRENCH = re.compile(r"^multilingual/(masakhanews/fra|xcsr/X-(CODAH|CSQA)-fr|xlwic/xlwic_fr_fr|"
                       r"disrpt/fra\..*)$")  # fmt: skip
# (rule, regex over ``source``): families we train on elsewhere, then eval-only sets
TS_DROP: tuple[tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        ("procedural", r"^procedural-typed-decisions/"),  # procedural / procedural_x
        ("helpsteer", r"(?i)^helpsteer"),  # helpsteer2_*, helpsteer3 (HelpSteer 1 too)
        ("wanli", r"(?i)^wanli$"),  # openjev / openjev_x
        ("mnli", r"(?i)mnli|^dataset_train_nli$|^multilingual/(xnli|multilingual-NLI-26lang)"),
        ("snli", r"(?i)snli"),  # snli (and its counterfactual / debiased / defeasible recasts)
        ("boolq", r"(?i)boolq"),  # boolq, twin_boolq
        ("paws", r"(?i)paws"),  # paws, pawsx_fr
        ("contract_nli", r"^contract-nli/"),  # cuad
        ("glue_sst_stsb", r"^glue/(sst2|stsb)$"),  # sst5, stsb
        ("intents", r"^(banking77|clinc_oos/.*|multilingual/(massive|mtop))$"),
        ("go_emotions", r"^go_emotions/"),
        ("dbpedia", r"^dbpedia_14/"),
        ("vitaminc", r"^vitaminc$"),
        ("prompt_injection", r"^prompt-injections$"),  # deepset/prompt-injections
        ("tweets_mlma", r"^multilingual/(MLMA_hate_speech|tweet_sentiment_multilingual)$"),
        (
            "judge_v7",
            (
                r"^(scruples/|SChem_Disagreement$|oasst2|multilingual/oasst1|webgpt_comparisons$"
                r"|summarize_from_feedback/)"
            ),
        ),
        ("x_stance", r"^multilingual/x-stance$"),
        ("reading_v7", r"^(sharc|temporal-nli|mc_taco|lex_glue/(case_hold|ledgar))$"),
        ("eval_only", r"^(ambient|chatbot_arena_conversations)$"),  # AmbiEnt; arena ≈ PPE
        ("language_id", r"^(multilingual/)?language-identification$|^multilingual/wili_2018$"),
        ("multilingual", r"^multilingual/"),  # not English / French (TS_MULTI_KEEP kept first)
    )
)


def ts_drop_rule(source: str, split: str = "train") -> str | None:
    """The TS_DROP rule that drops a tasksource row, or None (kept). ANLI rows from its dev /
    test rounds are eval-only (ANLI train is allowed)."""
    if re.match(r"(?i)^anli/", source) and split not in ("train", ""):
        return "anli_eval"
    if TS_MULTI_KEEP.match(source):
        return None
    for name, pattern in TS_DROP:
        if pattern.search(source):
            return name
    return None


def ts_quotas(counts: dict[str, int], total: int = TS_ROWS, share: float = TS_SOURCE_SHARE) -> dict:
    """Rows per source: min(n, c·√n, share·total), with c set so that they sum to
    ``total`` (or every row, when there are fewer)."""
    cap = share * total
    if sum(min(n, cap) for n in counts.values()) <= total:
        return {s: int(min(n, cap)) for s, n in counts.items()}
    lo, hi = 0.0, float(max(counts.values()))
    for _ in range(60):
        c = (lo + hi) / 2
        got = sum(min(n, c * math.sqrt(n), cap) for n in counts.values())
        lo, hi = (c, hi) if got < total else (lo, c)
    return {s: int(min(n, lo * math.sqrt(n), cap)) for s, n in counts.items()}


def ts_family(source: str) -> str:
    name = source.removeprefix("multilingual/").split("/")[0]
    return "ts_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def ts_lang(source: str) -> str:
    return "fr" if TS_FRENCH.match(source) else "en"


TS_COLUMNS = ["state", "kind", "id", "options", "target", "question", "source", "split",
              "group_id", "license", "license_use"]  # fmt: skip


def tasksource_load(split: str):
    """Two passes over the ``full`` shards of ``split``: (source, split, group_id) to count
    the kept rows and set each source's quota, then the full columns of the drawn groups."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    paths = _repo_files(TS_REPO, f"full/{split}-*.parquet")
    keys = pa.concat_tables(
        [pq.read_table(p, columns=["source", "split", "group_id"]) for p in paths]
    ).to_pydict()
    rule_of: dict[tuple[str, str], str | None] = {}
    groups: dict[str, Counter] = defaultdict(Counter)
    dropped: Counter = Counter()
    for source, orig, group in zip(keys["source"], keys["split"], keys["group_id"], strict=True):
        k = (source, orig or "")
        if k not in rule_of:
            rule_of[k] = ts_drop_rule(*k)
        if rule_of[k] is None:
            groups[source][group] += 1
        else:
            dropped[rule_of[k]] += 1
    print(f"[tasksource_jev] {split}: dropped by rule {dict(dropped)}", flush=True)
    del keys
    if split == "train":
        counts = {s: sum(g.values()) for s, g in groups.items()}
        quota = ts_quotas(counts)
        chosen = set()
        for s, g in groups.items():
            got = 0
            for group in sorted(g, key=lambda x: _hkey(f"ts:{x}")):
                if got >= quota[s]:
                    break
                chosen.add(group)
                got += g[group]
    else:
        chosen = {group for g in groups.values() for group in g}
    del groups
    rows = []
    wanted = pa.array(sorted(chosen))
    for path in paths:
        table = pq.read_table(path, columns=TS_COLUMNS)
        table = table.filter(pc.is_in(table["group_id"], value_set=wanted))
        rows.extend(
            r for r in table.to_pylist() if rule_of.get((r["source"], r["split"] or "")) is None
        )
    rows.sort(key=lambda r: (r["source"], r["group_id"], r["id"]))
    return _dataset(rows)


TS_LICENSE_USES = {"commercial", "non-commercial", "unspecified"}


def tasksource_jev(row, rng, meta):
    kind, options = row["kind"], [str(o) for o in row["options"] or []]
    probs = [float(p) for p in row["target"] or []]
    if any(p < 0 or not math.isfinite(p) for p in probs):
        return []
    state = row["state"]
    if isinstance(state, str) and state[:1] in "{[":
        try:
            parsed = json.loads(state)
            state = rendered_state(parsed) if isinstance(parsed, (dict, list)) else state
        except ValueError:
            pass
    if kind == "noul":
        if len(probs) != 1 or not 0.0 <= probs[0] <= 1.0:
            return []
        p = probs[0]
        criteria = {"false": "", "true": ""}
        target = {"expected": "yes" if p == 1.0 else "no"} if p in (0.0, 1.0) else {
            "target": {"no": 1.0 - p, "yes": p}}  # fmt: skip
    elif kind == "choice":
        if len(options) != len(probs) or len(options) < 2:
            return []
        labels, descs = openjev_options(options)
        criteria = dict(zip(labels, descs, strict=True))
        target = _one_hot_or_soft(labels, probs)
    elif kind == "score":
        if len(options) != len(probs) or len(options) < 2:
            return []
        criteria = ["" if o.strip() == str(i) else o for i, o in enumerate(options)]
        target = _one_hot_or_soft([str(i) for i in range(len(options))], probs)
    else:
        return []
    use = str(row.get("license_use") or "")
    return [
        {
            "state": state,
            "question": {"type": kind, "instructions": row["question"], "criteria": criteria},
            **target,
            "family": ts_family(row["source"]),
            "lang": ts_lang(row["source"]),
            "license": str(row.get("license") or "unspecified"),
            "license_use": use if use in TS_LICENSE_USES else "unspecified",
        }
    ]


# --- licences and registry -------------------------------------------------------------------

LICENCES: dict[str, tuple[str, str, str]] = {
    "openjev_x": (
        "CC0-1.0 (generated) + CC BY 4.0 (WANLI)",
        "yes",
        "ZefanCai/Open-Jev-v1.1 + ZefanCai/Open-Jev (CC0) cards; generated by Open-Jev code",
    ),
    "openjev_ctrl": (
        "CC0-1.0",
        "yes",
        "ZefanCai/Open-Jev control configs (card: original generated records CC0-1.0)",
    ),
    "certo_x": ("MIT", "yes", "altslate/certo-decisions-v2 card tag (rule / posterior slices)"),
    "procedural_x": ("Apache-2.0", "yes", "tasksource/procedural-typed-decisions card tag"),
    "eikos_x": (
        "CC BY 4.0",
        "yes",
        "caiovicentino1/eikos-decisions card tag; GLM-5.3-Flash teacher labels (MIT model)",
    ),
    "kaivoss_s1": (
        "Apache-2.0",
        "yes",
        "kaivoss/system-one-270m-data card tag; written and labelled by gpt-oss-20b",
    ),
    "n4ze3m_td": (
        "MIT",
        "yes",
        "n4ze3m/typed-decisions-synth card; written and labelled by DeepSeek V4.1 Flash",
    ),
    "helmo_td": (
        "MIT",
        "yes",
        "helmo/synthetic-typed-decisions card; written and labelled by DeepSeek-V4-Flash",
    ),
    "mlagnajit_td": (
        "CC BY 4.0",
        "yes",
        (
            "m-lagnajit/calibrated-typed-decisions card; texts CC BY 3.0 / 4.0 / Apache-2.0, "
            "teacher Qwen2.5-7B-Instruct"
        ),
    ),
    "tasksource_jev": (
        "other (per-row source licences, sources.yaml)",
        "unknown",
        (
            "tasksource/tasksource-jev-typed-decisions card: license other; each row carries "
            "license / license_use"
        ),
    ),
}


def _src(name, repo, config, train, ev, adapter, family, block, licence, use, cap, **kw):
    return Source(name, repo, config, train, ev, adapter, family, kw.pop("lang", "en"), licence,
                  use == "commercial", cap=cap, block=block, license_use=use, **kw)  # fmt: skip


JEV_V7_SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        _src("openjev_x", OPENJEV_REPO, OPENJEV_CONFIG, "train", None, openjev, "openjev",
             "jev", LICENCES["openjev_x"][0], "commercial", 75000, load=openjev_x_load,
             train_only=True),
        _src("openjev_ctrl", OPENJEV_V1_REPO, "controls", "train", "validation", openjev,
             "openjev_ctrl", "jev", "CC0-1.0", "commercial", 35000, load=openjev_ctrl_load),
        _src("certo_x", CERTO_REPO, None, "train", None, certo, "certo", "jev", "MIT",
             "commercial", 20000, load=certo_x_load, train_only=True),
        _src("procedural_x", PROC_REPO, None, "train", None, procedural, "procedural",
             "targeted", "Apache-2.0", "commercial", 64000, load=procedural_x_load,
             train_only=True),
        _src("eikos_x", "caiovicentino1/eikos-decisions", "core", "train", None, eikos, "eikos",
             "targeted", "CC BY 4.0", "commercial", 10000, load=eikos_x_load, train_only=True),
        _src("kaivoss_s1", "kaivoss/system-one-270m-data", None, "train", None, kaivoss,
             "kaivoss", "jev", "Apache-2.0", "commercial", 25000,
             group=lambda r: r["state_id"]),
        _src("n4ze3m_td", "n4ze3m/typed-decisions-synth", None, "train", "validation", n4ze3m,
             "n4ze3m", "jev", "MIT", "commercial", 24000),
        _src("helmo_td", "helmo/synthetic-typed-decisions", None, "train", None, helmo,
             "helmo_synthetic", "jev", "MIT", "commercial", 10000,
             group=lambda r: re.sub(r"\W+", " ", r["state"].lower()).strip()),
        _src("mlagnajit_td", "m-lagnajit/calibrated-typed-decisions", None, "train",
             "validation", mlagnajit, "mlagnajit", "jev", "CC BY 4.0", "commercial", 5000),
        _src("tasksource_jev", TS_REPO, "full", "train", "validation", tasksource_jev,
             "tasksource", "breadth", LICENCES["tasksource_jev"][0], "unspecified", 300000,
             load=tasksource_load),
    ]
}  # fmt: skip
