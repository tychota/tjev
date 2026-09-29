"""Contender Jev data: public training sets of JevBench contenders (the mix's "jev" block).

Same contract as ``sources.py``: each adapter maps one row to item dicts; a row may be a
whole scenario and yield one item per question. Adapters set ``family`` (and ``lang``)
from the dataset's own fields; ``tjev.data.mix.adapt`` keeps them. Structured states are
rendered exactly as ``parse_item`` renders them, so dedup and contamination checks see
the text the model sees. Field mappings were checked against the Hub / GitHub on
2026-09-29 (Open-Jev v1.1, Plumb, mghafiri v2, certo v2.1, decider ``23579f7``).

Every row is later checked against the public JevBench items (:mod:`tjev.data.jevbench`).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from tjev.data.item import rendered_state
from tjev.data.sources.base import CACHE, Source


def _instructions(value) -> str:
    """Structured instructions ({question, rule, …}, mghafiri v2) as text: the question,
    then one "Rule: …" line per extra field."""
    if not isinstance(value, dict):
        return str(value)
    lines = [str(value.get("question", ""))]
    for key, text in value.items():
        if key != "question":
            lines.append(f"{key.replace('_', ' ').capitalize()}: {text}")
    return "\n".join(line for line in lines if line)


def _hash01(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16**12


def _one_hot_or_soft(labels: list[str], probs: list[float]) -> dict:
    """``expected`` for a one-hot distribution, a soft ``target`` otherwise."""
    if max(probs) == 1.0 and sum(probs) == 1.0:
        return {"expected": labels[int(np.argmax(probs))]}
    return {"target": dict(zip(labels, map(float, probs), strict=True))}


NOUL_MAP = {"no": "no", "false": "no", "yes": "yes", "true": "yes"}


# --- Plumb (crh225/plumb-decisions) ----------------------------------------------------
# JevBench schema already. Noul answers and teacher_probs use "true"/"false". teacher_probs
# is the two-solver share (one-hot on every kept question) except in the ``probability``
# family, where it is the exact distribution: that becomes the soft target. The test
# split (131 questions) uses three domains train never uses.


def plumb_load(split: str):
    """The raw jsonl, with the heterogeneous ``question`` / ``teacher_probs`` kept as JSON
    strings (score criteria are lists, choice criteria dicts: no common Arrow type)."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("crh225/plumb-decisions", f"data/{split}.jsonl", repo_type="dataset")
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append(
                    {
                        "id": r["id"],
                        "family": r["family"],
                        "domain": r["domain"],
                        "state": r["state"],
                        "question": json.dumps(r["question"], ensure_ascii=False),
                        "expected": str(r["expected"]),
                        "teacher_probs": json.dumps(r.get("teacher_probs") or {}),
                    }
                )
    return Dataset.from_list(rows)


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def plumb(row, rng, meta):
    q = _json(row["question"])
    probs = _json(row.get("teacher_probs")) or {}
    if q["type"] == "noul":
        probs = {NOUL_MAP[k]: v for k, v in probs.items()}
        expected = NOUL_MAP[str(row["expected"]).lower()]
    else:
        expected = str(row["expected"])
    out = {
        "state": row["state"],
        "question": {
            "type": q["type"],
            "instructions": _instructions(q["instructions"]),
            "criteria": q.get("criteria")
            or ({"false": "", "true": ""} if q["type"] == "noul" else {}),
        },
        "family": row["family"],
    }
    if row["family"] == "probability" and probs and max(probs.values()) < 1.0:
        out["target"] = probs
    else:
        out["expected"] = expected
    return [out]


# --- Open-Jev v1.1 (ZefanCai/Open-Jev-v1.1) --------------------------------------------
# Columns: kind (choice|noul|score), question (str), options (list[str], "LABEL: desc" or
# bare), target (list, aligned with options), state_json (JSON str or dict), source,
# group_id. metadata_json is supervision/provenance (not model input): only its
# ``language`` is read, to keep English rows. WANLI (source "wanli-decisions-v1", 56% of
# train) is hash-subsampled by group to ~10% of the source.

OPENJEV_REPO = "ZefanCai/Open-Jev-v1.1"
OPENJEV_CONFIG = "community-hard-mix-v2-redistributable"
WANLI_SHARE = 0.10
_PREFIXED = re.compile(r"^([A-Za-z0-9_.\-]{1,40}): (.+)$", re.DOTALL)
_IDENT = re.compile(r"^[A-Za-z0-9_.\-]{1,40}$")


def openjev_options(options: list[str]) -> tuple[list[str], list[str]]:
    """(labels, descriptions) of a choice menu: "LABEL: desc" options, bare identifiers,
    or free sentences (labelled option_1…)."""
    parsed = [m for o in options if (m := _PREFIXED.match(o))]
    if len(parsed) == len(options):
        labels, descs = [m.group(1) for m in parsed], [m.group(2) for m in parsed]
    elif all(_IDENT.match(o) for o in options):
        labels, descs = list(options), [""] * len(options)
    else:
        labels, descs = [f"option_{i + 1}" for i in range(len(options))], list(options)
    if len(set(labels)) != len(labels):
        labels, descs = [f"option_{i + 1}" for i in range(len(options))], list(options)
    return labels, descs


def openjev(row, rng, meta):
    kind, options, probs = row["kind"], list(row["options"]), [float(p) for p in row["target"]]
    if len(options) != len(probs) or len(options) < 2:
        return []
    state = row["state_json"]
    state = rendered_state(json.loads(state) if isinstance(state, str) else state)
    if kind == "noul":
        by_label, criteria = {}, {"false": "", "true": ""}
        for option, p in zip(options, probs, strict=True):
            m = _PREFIXED.match(option)
            name, desc = (m.group(1), m.group(2)) if m else (option, "")
            label = NOUL_MAP.get(name.strip().lower())
            if label is None:
                return []
            by_label[label] = p
            criteria["false" if label == "no" else "true"] = desc
        if set(by_label) != {"no", "yes"}:
            return []
        labels, probs = ["no", "yes"], [by_label["no"], by_label["yes"]]
    elif kind == "choice":
        labels, descs = openjev_options(options)
        criteria = dict(zip(labels, descs, strict=True))
    elif kind == "score":
        labels = [str(i) for i in range(len(options))]
        criteria = ["" if o.strip() == str(i) else o for i, o in enumerate(options)]
    else:  # multilabel / ordinal kinds have no single-answer form
        return []
    return [
        {
            "state": state,
            "question": {"type": kind, "instructions": row["question"], "criteria": criteria},
            **_one_hot_or_soft(labels, probs),
            "family": row["source"],
        }
    ]


def openjev_load(split: str):
    """One split, English rows only, WANLI hash-subsampled by group to ~10% of the rows."""
    import pyarrow.parquet as pq
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        OPENJEV_REPO,
        f"data/{OPENJEV_CONFIG}/{split}-00000-of-00001.parquet",
        repo_type="dataset",
    )
    columns = ["id", "group_id", "source", "kind", "question", "options", "target"]
    table = pq.read_table(path, columns=[*columns, "state_json", "metadata_json"])
    rows = []
    for r in table.to_pylist():
        language = json.loads(r.pop("metadata_json") or "{}").get("language")
        if language in (None, "en"):
            rows.append(r)
    wanli = [r for r in rows if r["source"].startswith("wanli")]
    other = len(rows) - len(wanli)
    keep = min(1.0, WANLI_SHARE / (1 - WANLI_SHARE) * other / max(1, len(wanli)))
    rows = [r for r in rows if not r["source"].startswith("wanli") or _hash01(r["group_id"]) < keep]
    return Dataset.from_list(rows)


# --- decider teacher_data (github.com/Mapika/decider) ---------------------------------
# commands / custom_questions / routing_messages / routing_terse: {state, domain, recipe,
# questions: [{type, instructions, criteria, answer, teacher_p, teacher_ok?}]}; noul
# criteria may be null, choice descriptions may be null. situations: {situation, question,
# options: [str], answer: int}. Questions with teacher_ok false are dropped; so are
# custom questions whose teacher probability of the answer (teacher_p) is below 0.5.

DECIDER_SHA = "23579f7a7e8f10e1045be492af3c1c05a005d67c"
DECIDER_FILES = (
    "commands.jsonl",
    "custom_questions.jsonl",
    "routing_messages.jsonl",
    "routing_terse.jsonl",
    "situations.jsonl",
)


def _decider_file(name: str) -> Path:
    import urllib.request

    path = CACHE / f"decider-{DECIDER_SHA[:7]}" / name
    if not path.exists():
        url = f"https://raw.githubusercontent.com/Mapika/decider/{DECIDER_SHA}/teacher_data/{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
            f.write(r.read())
        tmp.replace(path)
    return path


def decider_situation(raw: dict) -> dict:
    """A situations.jsonl row in the shape of the other teacher files."""
    labels = [f"option_{i + 1}" for i in range(len(raw["options"]))]
    return {
        "state": raw["situation"],
        "domain": raw["domain"],
        "recipe": "situations",
        "questions": [
            {
                "type": "choice",
                "instructions": raw["question"],
                "criteria": dict(zip(labels, raw["options"], strict=True)),
                "answer": labels[int(raw["answer"])],
            }
        ],
    }


def decider_load(split: str):
    from datasets import Dataset

    rows = []
    for name in DECIDER_FILES:
        with _decider_file(name).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if name == "situations.jsonl":
                    raw = decider_situation(raw)
                rows.append(
                    {
                        "state": rendered_state(raw["state"]),
                        "domain": raw.get("domain") or "",
                        "recipe": raw.get("recipe") or name.removesuffix(".jsonl"),
                        "questions": json.dumps(raw["questions"], ensure_ascii=False),
                    }
                )
    return Dataset.from_list(rows)


def decider(row, rng, meta):
    out = []
    for q in json.loads(row["questions"]):
        if q.get("teacher_ok") is False:
            continue
        if q.get("teacher_ok") is None and q.get("teacher_p") is not None and q["teacher_p"] < 0.5:
            continue
        kind, answer, criteria = q["type"], q["answer"], q.get("criteria")
        if kind == "noul":
            criteria = {k: v or "" for k, v in (criteria or {"false": "", "true": ""}).items()}
            expected = "yes" if answer is True or str(answer).lower() == "true" else "no"
        elif kind == "choice":
            criteria = {str(k): v or "" for k, v in criteria.items()}
            expected = str(answer)
        elif kind == "score":
            criteria = [c or "" for c in criteria]
            expected = str(int(answer))
        else:
            continue
        out.append(
            {
                "state": row["state"],
                "question": {
                    "type": kind,
                    "instructions": _instructions(q["instructions"]),
                    "criteria": criteria,
                },
                "expected": expected,
                "family": row["recipe"],
            }
        )
    return out


# --- mghafiri/decision-model-scenarios-v2 ----------------------------------------------
# One row per scenario: state / questions / targets are JSON strings; questions is
# {key: {type, instructions, criteria}} (noul criteria absent or {true,false}); targets[k]
# has "probabilities" (choice: by option, score: "0".."K-1") or "noul" = P(yes).


def mghafiri(row, rng, meta):
    state = rendered_state(json.loads(row["state"]))
    questions, targets = json.loads(row["questions"]), json.loads(row["targets"])
    out = []
    for key, q in questions.items():
        t = targets.get(key)
        if t is None:
            continue
        criteria = q.get("criteria")
        if q["type"] == "noul":
            p = float(t["noul"])
            target = {"no": 1.0 - p, "yes": p}
            criteria = criteria or {"false": "", "true": ""}
        else:
            target = {str(k): float(v) for k, v in t["probabilities"].items()}
        out.append(
            {
                "state": state,
                "question": {
                    "type": q["type"],
                    "instructions": _instructions(q["instructions"]),
                    "criteria": criteria,
                },
                "target": target,
                "family": row["domain"],
            }
        )
    return out


# --- altslate/certo-decisions-v2 (v2.1) ------------------------------------------------
# Only the rule-computed slices: domain_generator + own_policy_generator
# (program_verified) and genworld (exact_posterior, soft), without independent_binary
# (multi-label). question JSON: {type: choice|binary|score, instructions, rubric?,
# levels?, options: [{id, description}]}; target JSON: categorical_label (label_id),
# categorical_distribution / score_distribution (probabilities). The slices are highly
# templated (460k domain rows share 16 instructions), so the loader keeps a
# deterministic hash-ordered sample of distinct (state, instructions) pairs, round-robin
# over task families, with genworld posteriors at 10%.

CERTO_REPO = "altslate/certo-decisions-v2"
CERTO_SAMPLE = {"train": 50000, "dev": 6000}
CERTO_GENWORLD_SHARE = 0.1
CERTO_SLICES = ("domain_generator", "own_policy_generator", "genworld")


def certo_load(split: str):
    import pyarrow.parquet as pq
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CERTO_REPO, f"{split}.parquet", repo_type="dataset")
    columns = [
        "record_id",
        "task_family",
        "source_id",
        "question_type",
        "state",
        "question",
        "target",
    ]
    table = pq.read_table(
        path,
        columns=columns,
        filters=[
            ("source_id", "in", list(CERTO_SLICES)),
            ("question_type", "!=", "independent_binary"),
        ],
    )
    rows = table.to_pylist()
    rows.sort(key=lambda r: hashlib.sha256(r["record_id"].encode()).hexdigest())
    # distinct (state, instructions) per task family, in hash order (a state asked with a
    # different distractor set is the same decision)
    seen, families = set(), defaultdict(list)
    for r in rows:
        q = json.loads(r["question"])
        key = (r["state"], str(q.get("instructions")), str(q.get("rubric")))
        if key not in seen:
            seen.add(key)
            families[r["task_family"]].append(r)
    n = CERTO_SAMPLE.get(split, CERTO_SAMPLE["dev"])
    genworld = families.pop("known_posterior", [])[: int(n * CERTO_GENWORLD_SHARE)]
    out = list(genworld)
    # round-robin over the rule families: 460k domain rows are 70% two math templates
    queues = [families[f] for f in sorted(families)]
    depth = 0
    while len(out) < n and any(depth < len(q) for q in queues):
        out += [q[depth] for q in queues if depth < len(q)][: n - len(out)]
        depth += 1
    return Dataset.from_list(out)


def certo(row, rng, meta):
    q, t = json.loads(row["question"]), json.loads(row["target"])
    kind = q["type"]
    instructions = _instructions(q["instructions"])
    if q.get("rubric"):
        instructions = f"{instructions} Rule: {q['rubric']}"
    options = q.get("options") or []
    ids = [str(o["id"]) for o in options]
    if kind == "binary" and set(ids) == {"yes", "no"}:
        kind = "noul"
        desc = {str(o["id"]): o.get("description") or "" for o in options}
        criteria = {"false": desc["no"], "true": desc["yes"]}
        labels = ["no", "yes"]
    elif kind in ("choice", "binary"):
        kind = "choice"
        criteria = {str(o["id"]): o.get("description") or "" for o in options}
        labels = ids
    elif kind == "score":
        labels = [str(i) for i in range(int(q["levels"]))]
        criteria = [""] * len(labels)
    else:
        return []
    if t["kind"] == "categorical_label":
        target = {"expected": str(t["label_id"])}
    elif t["kind"] == "categorical_distribution":
        target = _one_hot_or_soft(labels, [float(t["probabilities"].get(k, 0.0)) for k in labels])
    elif t["kind"] == "score_distribution":
        target = _one_hot_or_soft(labels, [float(p) for p in t["probabilities"]])
    else:
        return []
    return [
        {
            "state": row["state"],
            "question": {"type": kind, "instructions": instructions, "criteria": criteria},
            **target,
            "family": row["task_family"],
        }
    ]


JEV_SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source(
            "plumb", "crh225/plumb-decisions", None, "train", "test", plumb, "plumb", "en",
            "Apache-2.0 (Qwen3.8-27B-written)", True, cap=6000, load=plumb_load,
        ),
        Source(
            "openjev", OPENJEV_REPO, OPENJEV_CONFIG, "train", "validation", openjev,
            "openjev", "en", "CC0-1.0 (generated) + CC BY 4.0 (WANLI)", True, cap=40000,
            load=openjev_load,
        ),
        Source(
            "decider_teacher", "github:Mapika/decider/teacher_data", None, "train", None,
            decider, "decider", "en", "Apache-2.0", True, cap=30000, load=decider_load,
            group=lambda r: re.sub(r"\W+", " ", r["state"].lower()).strip(),
        ),
        Source(
            "mghafiri_scenarios", "mghafiri/decision-model-scenarios-v2", None, "train",
            "validation", mghafiri, "mghafiri", "en", "MIT (Claude-written)", True,
            cap=20000,
        ),
        Source(
            "certo", CERTO_REPO, None, "train", "dev", certo, "certo", "en", "MIT", True,
            cap=15000, load=certo_load,  # few distinct rules: keep its share small
        ),
    ]
}  # fmt: skip
