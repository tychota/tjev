"""mix-v7 targeted sources: temporal, numeric and legal decisions (block "targeted").

==================  ==========================================  ==============================
source              what it adds                                labels
==================  ==========================================  ==============================
``tempreason_l23``  TempReason L2 / L3 over the given facts:    recomputed from the fact
                    "which X in month M", "which X before /     intervals (rows that disagree
                    after Y" (gold + same-subject distractors)  are dropped)
``cold_law_fr_x``   more COLD articles in force on a date D     exact (dates), as cold_law_fr
                    (a hash slice disjoint from mix-v6's)
``argimi_fr_x``     more Cour de cassation outcomes (a slice    exact (decision), as argimi_fr
                    disjoint from mix-v6's)
``cetat_fr``        administrative-court outcomes (Conseil      read from the operative part,
                    d'État, CAA, TA: JADE) without the "DÉCIDE" which is cut from the state
                    part
``tram``            TRAM program-generated categories           exact (generated; recomputed
                    (arithmetic, shifts, computations,          for the date / time
                    comparisons, chronological order)          categories)
``drop_num``        DROP number / date answers, exact ±         exact (gold; validation: all
                    distractors                                 annotators agree)
``tatqa``           TAT-QA arithmetic / count / numeric spans   exact (gold), ± distractors
                    over a table and its paragraphs
``finqa``           FinQA numeric answers (program-executed)    exact (exe_ans agrees with
                    and yes/no comparisons                      the answer), ± distractors
``temporal_nli``    temporal-expression NLI (order, duration,   gold (templated; capped)
                    unit magnitude)
``forecasting``     resolved binary forecasting questions,      the resolution (hard), the
                    temporal split (held out: after June 2024)  crowd's last forecast as the
                                                                teacher distribution
==================  ==========================================  ==============================

Not built: ``fiscal_fr`` ("which version of this article applies on date D"). Every
public LEGI-derived HF dataset we found holds one version per article (COLD: articles in
force only, ``article_etat`` VIGUEUR or empty; louisbrulenaudet/code-impots and
livre-procedures-fiscales: the current version, ``multipleVersions`` always false, only
the id of the previous version; Nicolas-BZRD/LEGI_opendata: id + text, no dates), so the
label cannot be computed from version histories. FiscalQA's own questions are never used.

Same contract as the other adapter modules. Every source sets ``block`` and
``license_use``; licences are in ``LICENCES``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import re
from pathlib import Path

from tjev.data.sources.base import CACHE, Source
from tjev.data.sources.targeted import (
    ARGIMI_FILES,
    ARGIMI_HEAD,
    ARGIMI_REPO,
    ARGIMI_TAIL,
    COLD_COLUMNS,
    COLD_REPO,
    MONTHS_ABBR,
    MONTHS_EN,
    TEMPREASON_REPO,
    _dataset,
    _hash01,
    argimi,
    argimi_outcome,
    argimi_text,
    cold_law,
    html_text,
)

V6_HELD_GROUP_SHARE = 0.1  # tjev.data.mix.HELD_OUT_GROUP_SHARE (mix-v6 group splits)


def v6_held_group(key: str) -> bool:
    """Is ``key`` (a group) in a mix-v6 held-out group split (tjev.data.mix.group_split)?"""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 1000 < V6_HELD_GROUP_SHARE * 1000


def _labels(n: int) -> list[str]:
    return [f"option_{i + 1}" for i in range(n)]


def _menu(gold: str, others: list[str], rng) -> tuple[dict, str]:
    """A choice menu: the gold text and the distractor texts in a random order, labelled
    option_1..n; returns (criteria, gold label)."""
    texts = [gold, *others]
    order = [int(i) for i in rng.permutation(len(texts))]
    labels = _labels(len(texts))
    criteria = {labels[k]: texts[j] for k, j in enumerate(order)}
    return criteria, labels[order.index(0)]


# --- TempReason L2 / L3 (tonytan48/TempReason) ---------------------------------------------
# {question, date, text_answers: {text: [gold]}, neg_answers: [...], fact_context: lines
# "<subject> <relation> <object> from Mon, YYYY[ to Mon, YYYY]."}. L2: "Which position did
# X hold in May, 1946?" — the object whose interval contains the month. L3: "... before /
# after <object Y>?" — the object whose interval comes right before / after Y's. The label
# is recomputed from the facts (state = the shuffled facts) and rows that disagree, or
# whose month falls on a boundary shared by two objects, are dropped.

TR_FILES = {"train": ("train_l2.json", "train_l3.json"), "val": ("val_l2.json", "val_l3.json")}
TR_SAMPLE = {
    "train": 36000,
    "val": 1500,
}  # per level: every train row (L2 + L3 ≈ 26.6k), ~63% keep a label
_TR_MONTH = re.compile(r"([A-Z][a-z]{2}), (\d{3,4})")
_TR_L2 = re.compile(r" in ([A-Z][a-z]{2}), (\d{3,4})\s*\?\s*$")
_TR_L3 = re.compile(r" (before|after) (.+?)\s*\?\s*$")


def tempreason_l23_load(split: str):
    from huggingface_hub import hf_hub_download

    rows = []
    for name in TR_FILES[split]:
        path = hf_hub_download(TEMPREASON_REPO, name, repo_type="dataset")
        with Path(path).open(encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
        raw.sort(key=lambda r: hashlib.sha256(f"{split}:{r['id']}".encode()).hexdigest())
        for r in raw[: TR_SAMPLE[split]]:
            gold = (r.get("text_answers") or {}).get("text") or []
            if len(gold) != 1 or not r.get("fact_context"):
                continue
            rows.append(
                {
                    "id": str(r["id"]),
                    "level": name.split("_")[1][:2].upper(),
                    "question": r["question"],
                    "answer": gold[0],
                    "negatives": [str(n) for n in r.get("neg_answers") or []],
                    "facts": r["fact_context"],
                }
            )
    return _dataset(rows)


def _month(text: str) -> int | None:
    m = _TR_MONTH.fullmatch(text.strip())
    if m is None or m.group(1) not in MONTHS_ABBR:
        return None
    return int(m.group(2)) * 12 + MONTHS_ABBR.index(m.group(1))


def tr_intervals(facts: str, objects: list[str]) -> dict[str, list[tuple[int, int]]] | None:
    """Month intervals [start, end] of each object in the fact lines (an open end: far in
    the future); None when an object has no fact line or a date does not parse."""
    lines = [ln.strip().rstrip(".").strip() for ln in facts.split("\n") if ln.strip()]
    out: dict[str, list[tuple[int, int]]] = {}
    for obj in objects:
        spans = []
        for line in lines:
            head, sep, dates = line.rpartition(" from ")
            if not sep or not head.endswith(" " + obj):
                continue
            first, _, last = dates.partition(" to ")
            start = _month(first)
            end = _month(last) if last else 10**6
            if start is None or end is None or end < start:
                return None
            spans.append((start, end))
        if not spans:
            return None
        out[obj] = spans
    return out


def tempreason_l23_gold(question: str, facts: str, candidates: list[str]) -> str | None:
    """The candidate the facts make the answer, or None (no unique answer)."""
    spans = tr_intervals(facts, candidates)
    if spans is None:
        return None
    m = _TR_L2.search(question)
    if m is not None:
        when = _month(f"{m.group(1)}, {m.group(2)}")
        if when is None:
            return None
        hits = [c for c in candidates if any(a <= when <= b for a, b in spans[c])]
        return hits[0] if len(hits) == 1 else None
    m = _TR_L3.search(question)
    if m is None or m.group(2) not in spans:
        return None
    anchor = spans[m.group(2)]
    if len(anchor) != 1:
        return None
    a_start, _a_end = anchor[0]
    others = {c: s for c, s in spans.items() if c != m.group(2)}
    if any(len(s) != 1 for s in others.values()):
        return None
    if m.group(1) == "before":
        prior = {c: s[0][0] for c, s in others.items() if s[0][0] < a_start}
        if not prior:
            return None
        best = max(prior.values())
        hits = [c for c, v in prior.items() if v == best]
    else:
        later = {c: s[0][0] for c, s in others.items() if s[0][0] > a_start}
        if not later:
            return None
        best = min(later.values())
        hits = [c for c, v in later.items() if v == best]
    return hits[0] if len(hits) == 1 else None


def tempreason_l23(row, rng, meta):
    gold, negatives = row["answer"], [n for n in row["negatives"] if n != row["answer"]]
    negatives = list(dict.fromkeys(negatives))
    if not negatives:
        return []
    if tempreason_l23_gold(row["question"], row["facts"], [gold, *negatives]) != gold:
        return []  # the facts do not make the stated answer the unique one
    lines = [ln.strip() for ln in row["facts"].split("\n") if ln.strip()]
    lines = [lines[int(i)] for i in rng.permutation(len(lines))]
    state = "Facts:\n" + "\n".join(lines)
    family = f"temporal_facts_{row['level'].lower()}"
    if rng.random() < 0.6:
        others = negatives[:25]
        criteria, label = _menu(gold, others, rng)
        return [
            {
                "state": state,
                "question": {
                    "type": "choice",
                    "instructions": f"Based only on the facts: {row['question']}",
                    "criteria": criteria,
                },
                "expected": label,
                "family": family,
            }
        ]
    right = rng.random() < 0.5
    candidate = gold if right else negatives[int(rng.integers(len(negatives)))]
    question = row["question"].rstrip(" ?")
    return [
        {
            "state": state,
            "question": {
                "type": "noul",
                "instructions": f'Based only on the facts, is "{candidate}" the answer to: '
                f"{question}?",
                "criteria": {
                    "false": "No, the facts point to another answer",
                    "true": "Yes, the facts make it the answer",
                },
            },
            "expected": "yes" if right else "no",
            "family": family,
        }
    ]


# --- COLD and ArGiMi expansions (disjoint from the mix-v6 samples) -------------------------

COLD_X_SLICE = (0.035, 0.15)  # mix-v6 sampled _hash01(f"cold:{id}") < 0.035
ARGIMI_X_SLICE = (0.12, 0.55)  # mix-v6 sampled _hash01(f"argimi:{id}") < 0.12


def cold_x_load(split: str):
    """COLD articles of the hash slice COLD_X_SLICE, without the titles of mix-v6's
    held-out title groups (train only: the held-out rows are mix-v6's)."""
    import pandas as pd

    lo, hi = COLD_X_SLICE
    cache = CACHE / "cold-french-law" / f"slice-{lo}-{hi}.parquet"
    if not cache.exists():
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(COLD_REPO, "cold-french-law.csv", repo_type="dataset")
        parts = []
        for chunk in pd.read_csv(path, usecols=COLD_COLUMNS, dtype=str, chunksize=50000):
            u = chunk["article_identifier"].map(lambda i: _hash01(f"cold:{i}"))
            parts.append(chunk[(u >= lo) & (u < hi)])
        sample = pd.concat(parts).fillna("")
        cache.parent.mkdir(parents=True, exist_ok=True)
        sample.to_parquet(cache.with_suffix(".tmp"), index=False)
        cache.with_suffix(".tmp").replace(cache)
    df = pd.read_parquet(cache)
    df = df[(df["article_contenu_text"].str.strip() != "") & (df["texte_titre"] != "")]
    df = df[~df["texte_titre"].map(v6_held_group)]
    return _dataset(df.to_dict("records"))


def argimi_x_load(split: str):
    """ArGiMi Cour de cassation decisions of the hash slice ARGIMI_X_SLICE (mix-v6's
    held-out decisions are inside its own slice; they are excluded all the same)."""
    import pandas as pd

    lo, hi = ARGIMI_X_SLICE
    cache = CACHE / "argimi" / f"juri-v2-slice-{lo}-{hi}.parquet"
    if not cache.exists():
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        rows = []
        for name in ARGIMI_FILES:
            path = hf_hub_download(ARGIMI_REPO, name, repo_type="dataset")
            table = pq.read_table(path, columns=["id", "title", "content", "solution"])
            for r in table.to_pylist():
                u = _hash01(f"argimi:{r['id']}")
                if not lo <= u < hi or v6_held_group(r["id"]):
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


# --- Conseil d'État and administrative courts (ArGiMi "cetat" config: JADE) ----------------
# {id, content (HTML), title, jurisdiction, solution (90% empty)}. The outcome is read from
# the operative part ("D E C I D E :" then "Article 1er : …"), which is cut from the state
# with everything after it. Articles on costs (L. 761-1, frais) and notification are not
# read. rejet: only rejections; annulation: only annulments (of a judgment or a decision);
# annulation_partielle: an annulment and a rejection of the rest; non_lieu: "il n'y a pas
# lieu de statuer" and nothing else. Anything else (réformation, réduction, renvoi,
# expertise…) is dropped. Rows whose reasoning keeps an outcome phrase of the operative
# part are dropped. Rejections are subsampled (CETAT_REJET_KEEP).

CETAT_FILES = tuple(f"jade_chunks/jade_{i}_over_35.parquet" for i in range(1, 25))
CETAT_SAMPLE_SHARE = 0.12
CETAT_REJET_KEEP = 0.45  # rejections are ~75% of the outcomes: subsampled to ~55%
_CETAT_DECIDE = re.compile(r"D\s*[EÉ]\s*C\s*I\s*D\s*E\s*:", re.IGNORECASE)
_CETAT_ARTICLE = re.compile(r"Article\s+(?:1er|premier|\d+)\s*:?", re.IGNORECASE)
_CETAT_IGNORED = re.compile(r"761-1|frais|notifi|dépens|copie|transmi", re.IGNORECASE)
_CETAT_REJET = re.compile(r"\brejet(?:é|ée|és|ées)\b|\brejette\b", re.IGNORECASE)
_CETAT_ANNUL = re.compile(r"\bannul(?:é|ée|és|ées|e|ent)\b", re.IGNORECASE)
_CETAT_NON_LIEU = re.compile(r"n'y a pas lieu de statuer|non-lieu", re.IGNORECASE)
_CETAT_OTHER = re.compile(
    r"réform|rédu|décharg|renvo|expert|sursis|condamn|versera|est tenu|enjoint|ordonn|"
    r"transmis|attribu|rectifi|rétabli|remis à la charge",
    re.IGNORECASE,
)
_CETAT_LEAK = re.compile(r"D\s*[EÉ]\s*C\s*I\s*D\s*E\s*:|^\s*Article\s+1er\s*:", re.MULTILINE)
CETAT_CRITERIA = {
    "rejet": "Rejet : la requête (ou le recours) est rejetée",
    "annulation": "Annulation : la décision ou le jugement attaqué est annulé",
    "annulation_partielle": "Annulation partielle : une partie est annulée, le surplus des "
    "conclusions est rejeté",
    "non_lieu": "Non-lieu : il n'y a pas lieu de statuer",
}


def cetat_outcome(operative: str) -> str | None:
    """The outcome of a JADE operative part ("Article 1er : …" lines), or None."""
    articles = [a.strip() for a in _CETAT_ARTICLE.split(operative)[1:] if a.strip()]
    read = [a for a in articles if not _CETAT_IGNORED.search(a)]
    if not read:
        return None
    text = " ".join(read)
    rejet, annul = bool(_CETAT_REJET.search(text)), bool(_CETAT_ANNUL.search(text))
    non_lieu = bool(_CETAT_NON_LIEU.search(text))
    if _CETAT_OTHER.search(text):
        return None
    if non_lieu:
        return "non_lieu" if not (rejet or annul) else None
    if annul and rejet:
        return "annulation_partielle"
    if annul:
        return "annulation"
    if rejet:
        return "rejet"
    return None


def cetat_split(content: str) -> tuple[str, str] | None:
    """(reasoning, operative part) of a JADE decision, or None (no "DÉCIDE")."""
    text = html_text(content)
    cut = list(_CETAT_DECIDE.finditer(text))
    if len(cut) != 1:
        return None
    return text[: cut[0].start()].strip(), text[cut[0].end() :].strip()


def cetat_state(reasoning: str) -> str | None:
    if len(reasoning) < 300 or _CETAT_LEAK.search(reasoning):
        return None
    if len(reasoning) > ARGIMI_HEAD + ARGIMI_TAIL:
        head = reasoning[:ARGIMI_HEAD].rsplit(" ", 1)[0]
        tail = reasoning[-ARGIMI_TAIL:].split(" ", 1)[-1]
        reasoning = f"{head}\n[…]\n{tail}"
    return reasoning


def cetat_load(split: str):
    import pandas as pd

    cache = (
        CACHE
        / "argimi"
        / f"cetat-{CETAT_SAMPLE_SHARE}-{len(CETAT_FILES)}-{CETAT_REJET_KEEP}.parquet"
    )
    if not cache.exists():
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        rows = []
        for name in CETAT_FILES:
            path = hf_hub_download(ARGIMI_REPO, name, repo_type="dataset")
            table = pq.read_table(path, columns=["id", "title", "content", "jurisdiction"])
            for r in table.to_pylist():
                if _hash01(f"cetat:{r['id']}") >= CETAT_SAMPLE_SHARE:
                    continue
                parts = cetat_split(r["content"] or "")
                if parts is None:
                    continue
                outcome = cetat_outcome(parts[1])
                if outcome == "rejet" and _hash01(f"cetat-rejet:{r['id']}") >= CETAT_REJET_KEEP:
                    continue
                state = cetat_state(parts[0]) if outcome else None
                if state is not None:
                    rows.append(
                        {
                            "id": r["id"],
                            "title": r["title"] or "",
                            "jurisdiction": r["jurisdiction"] or "",
                            "text": state,
                            "outcome": outcome,
                        }
                    )
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(cache.with_suffix(".tmp"), index=False)
        cache.with_suffix(".tmp").replace(cache)
    return _dataset(pd.read_parquet(cache).to_dict("records"))


def cetat(row, rng, meta):
    if row["outcome"] not in CETAT_CRITERIA:
        return []
    title = re.sub(r",?\s*(publié|inédit)[^,]*$", "", row["title"], flags=re.IGNORECASE).strip()
    return [
        {
            "state": f"{title}\n\n{row['text']}" if title else row["text"],
            "question": {
                "type": "choice",
                "instructions": "Au vu des visas et des motifs, quelle est l'issue de cette "
                "affaire devant la juridiction administrative ?",
                "criteria": CETAT_CRITERIA,
            },
            "expected": row["outcome"],
        }
    ]


# --- TRAM (github.com/EternityYW/TRAM-Benchmark) ---------------------------------------------
# MCQ CSVs (Question, Option A..D, Answer letter, Category). TRAM ships a test set and
# 5-shot files only; held-out rows are a hash group split by question text. Kept: the
# program-generated categories. Dropped: "Commonsense" (MC-TACO), "Reading Comprehension"
# (SQuAD: in the mix), the web / human "Facts" of duration, frequency and typical time
# (memorised knowledge), ambiguity "Interpretation" (human-written), and the tasks built on
# other benchmarks (relation: TempEval-3, nli: MNLI / SNLI, causality: COPA, storytelling:
# ROC / SCT). Ordering "Facts" (programmatic shuffles of timeline events) is subsampled.
# Date / time categories are recomputed and rows that disagree dropped.

TRAM_URL = "https://raw.githubusercontent.com/EternityYW/TRAM-Benchmark/main/datasets/{}.zip"
TRAM_KEEP = {
    "arithmetic": None,  # every category
    "ambiguity_resolution": {"Shift - ST", "Shift - MT", "Shift - LT", "Shift - Calendar"},
    "duration": {"Computation", "Direct Comparison", "Multi-Step Comparison", "Analogy Inference"},
    "frequency": {"Comparison", "Computation", "Application"},
    "typical_time": {"Comparison"},
    "ordering": {"Facts"},
}
TRAM_ORDERING_SHARE = 0.3


def _tram_csv(task: str) -> list[dict]:
    import urllib.request
    import zipfile

    path = CACHE / "tram" / f"{task}.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with urllib.request.urlopen(TRAM_URL.format(task), timeout=300) as r, tmp.open("wb") as f:
            f.write(r.read())
        tmp.replace(path)
    csv.field_size_limit(10**8)
    with zipfile.ZipFile(path) as z:
        text = z.read(f"{task}_mcq.csv").decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def tram_load(split: str):
    rows = []
    for task, keep in TRAM_KEEP.items():
        for r in _tram_csv(task):
            category = r.get("Category", "")
            if keep is not None and category not in keep:
                continue
            if task == "ordering" and _hash01(f"tram:{r['Question']}") >= TRAM_ORDERING_SHARE:
                continue
            options = [r[k] for k in ("Option A", "Option B", "Option C", "Option D") if r.get(k)]
            letter = r["Answer"].strip().upper()
            idx = "ABCD".find(letter)
            if not 0 <= idx < len(options):
                continue
            rows.append(
                {
                    "task": task,
                    "category": category,
                    "question": r["Question"].strip(),
                    "options": options,
                    "answer": idx,
                }
            )
    return _dataset(rows)


_TR_YEAR = re.compile(r"^Which year comes (\d+) years? (after|before) (\d+)\?$")
_TR_MSHIFT = re.compile(r"^Which month comes (\d+) months? (after|before) ([A-Z][a-z]+)\?$")
_TR_DATE = re.compile(
    r"^What will be the time (?:(\d+) years?)?(?: and )?(?:(\d+) months?)? (after|before) "
    r"([A-Z][a-z]+) (\d+)\?$"
)
_TR_HOUR = re.compile(r"^What is (\d{1,2}):(\d{2}) ([+-]) (\d{1,2}):(\d{2})\?$")


def tram_recompute(question: str) -> str | None:
    """The answer of a recomputable TRAM arithmetic question (year / month shift, date
    computation, 24h hour adjustment), or None (not one of those)."""
    if m := _TR_YEAR.match(question):
        sign = 1 if m.group(2) == "after" else -1
        return str(int(m.group(3)) + sign * int(m.group(1)))
    if m := _TR_MSHIFT.match(question):
        if m.group(3) not in MONTHS_EN:
            return None
        sign = 1 if m.group(2) == "after" else -1
        return MONTHS_EN[(MONTHS_EN.index(m.group(3)) + sign * int(m.group(1))) % 12]
    if m := _TR_DATE.match(question):
        if m.group(4) not in MONTHS_EN or not (m.group(1) or m.group(2)):
            return None
        sign = 1 if m.group(3) == "after" else -1
        start = int(m.group(5)) * 12 + MONTHS_EN.index(m.group(4))
        end = start + sign * (12 * int(m.group(1) or 0) + int(m.group(2) or 0))
        return f"{MONTHS_EN[end % 12]} {end // 12}"
    if m := _TR_HOUR.match(question):
        a = int(m.group(1)) * 60 + int(m.group(2))
        b = int(m.group(4)) * 60 + int(m.group(5))
        t = (a + b if m.group(3) == "+" else a - b) % (24 * 60)
        return f"{t // 60:02d}:{t % 60:02d}"
    return None


def tram(row, rng, meta):
    options, gold = list(row["options"]), int(row["answer"])
    if len(set(options)) != len(options):
        return []
    computed = tram_recompute(row["question"])
    if computed is not None and computed != options[gold].strip():
        return []
    others = [o for i, o in enumerate(options) if i != gold]
    criteria, label = _menu(options[gold], others, rng)
    family = f"tram_{row['task']}"
    return [
        {
            "state": row["question"],
            "question": {
                "type": "choice",
                "instructions": "Which option correctly answers this temporal question?",
                "criteria": criteria,
            },
            "expected": label,
            "family": family,
        }
    ]


# --- Exact numeric distractors ---------------------------------------------------------------


def decimals_of(text: str) -> int:
    m = re.search(r"\.(\d+)", text.replace(",", ""))
    return len(m.group(1)) if m else 0


def parse_number(text: str) -> float | None:
    """A number written in ``text`` ("$1,910", "(12.5)", "87.04%", "-3"), or None."""
    t = text.strip().replace("$", "").replace("%", "").replace(",", "").strip()
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()").strip()
    try:
        x = float(t)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return -x if neg else x


def format_number(x: float, decimals: int, commas: bool = False) -> str:
    text = f"{x:,.{decimals}f}" if commas else f"{x:.{decimals}f}"
    return "0" if text in ("-0", "-0.0") else text


def _digit_swap(text: str, rng) -> str | None:
    digits = [i for i, c in enumerate(text) if c.isdigit()]
    pairs = [(a, b) for a, b in itertools.pairwise(digits) if b == a + 1 and text[a] != text[b]]
    if not pairs:
        return None
    a, b = pairs[int(rng.integers(len(pairs)))]
    return text[:a] + text[b] + text[a] + text[b + 1 :]


def numeric_distractors(
    x: float, decimals: int, rng, k: int = 4, nonnegative: bool = False, commas: bool = False
) -> list[str]:
    """Up to ``k`` wrong answers near ``x``, written with ``decimals`` decimals: ±1 / ±2 for
    small integers, ±10 / ±25 %, ×2, ÷2, ×10, ÷10, the opposite sign, a swap of two
    adjacent digits. Each one differs from ``x`` once rounded (by ≥ 1 % or ≥ 1 for small
    integers), and from the others."""
    gold = format_number(x, decimals, commas)
    cands = []
    if decimals == 0 and abs(x) <= 20:
        cands += [x - 1, x + 1, x - 2, x + 2]
    cands += [x * 1.1, x * 0.9, x * 1.25, x * 0.75, x * 2, x / 2, x * 10, x / 10]
    if x != 0:
        cands.append(-x)
    out: list[str] = []
    for c in [cands[int(i)] for i in rng.permutation(len(cands))]:
        if nonnegative and c < 0:
            continue
        text = format_number(c, decimals, commas)
        value = float(text.replace(",", ""))
        small = decimals == 0 and abs(x) <= 20
        far = abs(value - x) >= (1 if small else max(0.01 * abs(x), 10**-decimals))
        if text != gold and far and text not in out:
            out.append(text)
    swap = _digit_swap(gold, rng)
    if swap is not None and swap != gold and swap not in out:
        value = parse_number(swap)
        if value is not None and abs(value - x) >= max(0.01 * abs(x), 10**-decimals):
            out.insert(int(rng.integers(len(out) + 1)), swap)
    return out[:k]


def numeric_items(
    state: str, question: str, gold: str, distractors: list[str], rng, family: str
) -> list[dict]:
    """A choice among the gold and its distractors (60%), or a yes/no on one candidate."""
    if not distractors:
        return []
    if rng.random() < 0.6:
        criteria, label = _menu(gold, distractors, rng)
        return [
            {
                "state": state,
                "question": {"type": "choice", "instructions": question, "criteria": criteria},
                "expected": label,
                "family": family,
            }
        ]
    right = rng.random() < 0.5
    candidate = gold if right else distractors[int(rng.integers(len(distractors)))]
    return [
        {
            "state": state,
            "question": {
                "type": "noul",
                "instructions": f"{question.rstrip()}\nIs the answer {candidate}?",
                "criteria": {
                    "false": "No, the answer is a different value",
                    "true": "Yes, that is exactly the answer",
                },
            },
            "expected": "yes" if right else "no",
            "family": family,
        }
    ]


# --- DROP (EleutherAI/drop; licence of ucinlp/drop) ---------------------------------------
# {passage, question, answer: {number, date: {day, month, year}, spans}, validated_answers}.
# Number and date answers only. Validation rows are kept when every validated answer
# agrees with the answer. Distractors: numeric ones (above) for numbers; ±1 / ±2 years,
# ±1 month, ±1 / ±2 days (the parts the answer gives) for dates.

DROP_MAX_CHARS = 10000


def _drop_date(d: dict) -> tuple[int | None, int | None, int | None] | None:
    day, month, year = (str(d.get(k) or "").strip() for k in ("day", "month", "year"))
    if not (day or month or year):
        return None
    try:
        y = int(year) if year else None
        d_ = int(day) if day else None
    except ValueError:
        return None
    m = None
    if month:
        names = [n.lower() for n in MONTHS_EN]
        key = month.strip().lower()
        if key in names:
            m = names.index(key) + 1
        elif key[:3] in [n[:3] for n in names]:
            m = [n[:3] for n in names].index(key[:3]) + 1
        elif key.isdigit() and 1 <= int(key) <= 12:
            m = int(key)
        else:
            return None
    if d_ is not None and not 1 <= d_ <= 31:
        return None
    return d_, m, y


def date_text(day: int | None, month: int | None, year: int | None) -> str:
    parts = []
    if month is not None:
        parts.append(MONTHS_EN[month - 1])
    if day is not None:
        parts.append(str(day) + ("," if year is not None else ""))
    if year is not None:
        parts.append(str(year))
    return " ".join(parts)


def date_distractors(day, month, year, rng, k: int = 4) -> list[str]:
    gold = date_text(day, month, year)
    cands = []
    if year is not None:
        cands += [(day, month, year + d) for d in (-1, 1, -2, 2)]
    if month is not None:
        cands += [(day, (month - 1 + d) % 12 + 1, year) for d in (-1, 1)]
    if day is not None:
        cands += [(day + d, month, year) for d in (-1, 1, -2, 2) if 1 <= day + d <= 28]
    out = []
    for c in [cands[int(i)] for i in rng.permutation(len(cands))]:
        text = date_text(*c)
        if text != gold and text not in out:
            out.append(text)
    return out[:k]


def _drop_answer(answer: dict) -> tuple[str, str] | None:
    """("number" | "date", canonical text) of a DROP answer, or None (spans / empty)."""
    number = str(answer.get("number") or "").strip()
    if number:
        x = parse_number(number)
        return ("number", format_number(x, decimals_of(number))) if x is not None else None
    parsed = _drop_date(answer.get("date") or {})
    return ("date", date_text(*parsed)) if parsed else None


def drop_load(split: str):
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/drop", split=split)
    rows = []
    for r in ds:
        got = _drop_answer(r["answer"])
        if got is None or len(r["passage"]) > DROP_MAX_CHARS:
            continue
        if split != "train":
            v = r.get("validated_answers") or {}
            others = [
                _drop_answer({"number": n, "date": d})
                for n, d in zip(v.get("number") or [], v.get("date") or [], strict=False)
            ]
            if any(o is not None and o != got for o in others) or None in others:
                continue
        rows.append(
            {
                "id": r["query_id"],
                "section": r["section_id"],
                "passage": r["passage"],
                "question": r["question"],
                "kind": got[0],
                "number": str(r["answer"].get("number") or ""),
                "date": json.dumps(r["answer"].get("date") or {}),
            }
        )
    return _dataset(rows)


def drop_num(row, rng, meta):
    if row["kind"] == "number":
        x = parse_number(row["number"])
        if x is None:
            return []
        dec = decimals_of(row["number"])
        gold = format_number(x, dec)
        others = numeric_distractors(x, dec, rng, k=int(rng.integers(3, 5)), nonnegative=x >= 0)
        family = "numeric_reading"
    else:
        parsed = _drop_date(json.loads(row["date"]))
        if parsed is None:
            return []
        gold = date_text(*parsed)
        others = date_distractors(*parsed, rng, k=int(rng.integers(3, 5)))
        family = "temporal_reading"
    return numeric_items(row["passage"], row["question"], gold, others, rng, family)


# --- TAT-QA (next-tat/TAT-QA) ---------------------------------------------------------------
# [{table: {table: [[cells]]}, paragraphs: [{order, text}], questions: [{question, answer,
# answer_type (span | multi-span | arithmetic | count), scale}]}]. Kept: arithmetic and
# count answers, and single spans that are one number. The answer is shown with its scale
# ("8,325 thousand", "87.04%"), the distractors likewise. Held out: the dev file.

TATQA_FILES = {"train": "tatqa_dataset_train.json", "dev": "tatqa_dataset_dev.json"}
TATQA_MAX_CHARS = 9000
SCALE_TEXT = {"": "", "thousand": " thousand", "million": " million", "billion": " billion"}


def table_text(table: list[list]) -> str:
    return "\n".join(" | ".join(str(c).strip() for c in row) for row in table)


def tatqa_load(split: str):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("next-tat/TAT-QA", TATQA_FILES[split], repo_type="dataset")
    docs = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for doc in docs:
        paragraphs = sorted(doc.get("paragraphs") or [], key=lambda p: p.get("order", 0))
        context = (
            "Table:\n"
            + table_text(doc["table"]["table"])
            + "\n\nText:\n"
            + "\n".join(p["text"] for p in paragraphs)
        )
        if len(context) > TATQA_MAX_CHARS:
            continue
        for q in doc.get("questions") or []:
            answer, kind = q.get("answer"), q.get("answer_type")
            if kind in ("arithmetic", "count"):
                if isinstance(answer, list):
                    answer = answer[0] if len(answer) == 1 else None
                text = str(answer) if answer is not None else ""
            elif kind == "span" and isinstance(answer, list) and len(answer) == 1:
                text = str(answer[0])
            else:
                continue
            if parse_number(text) is None or (
                kind == "span"
                and not re.fullmatch(r"\$?\s*\(?-?[\d,]+(?:\.\d+)?\)?\s*%?", text.strip())
            ):
                continue
            rows.append(
                {
                    "id": q["uid"],
                    "doc": doc["table"]["uid"],
                    "context": context,
                    "question": q["question"],
                    "answer": text,
                    "kind": kind,
                    "scale": str(q.get("scale") or ""),
                }
            )
    return _dataset(rows)


def tatqa(row, rng, meta):
    x = parse_number(row["answer"])
    scale = row["scale"]
    if x is None or (scale not in SCALE_TEXT and scale != "percent"):
        return []
    dec = min(decimals_of(row["answer"]), 4)
    if row["kind"] == "arithmetic" and dec == 0 and not float(x).is_integer():
        dec = 2
    commas = "," in row["answer"] or abs(x) >= 10000
    suffix = "%" if scale == "percent" or row["answer"].strip().endswith("%") else SCALE_TEXT[scale]
    gold = format_number(x, dec, commas) + suffix
    others = [
        d + suffix
        for d in numeric_distractors(
            x,
            dec,
            rng,
            k=int(rng.integers(3, 5)),
            nonnegative=row["kind"] == "count",
            commas=commas,
        )
    ]
    return numeric_items(row["context"], row["question"], gold, others, rng, "numeric_table")


# --- FinQA (github.com/czyssrs/FinQA dataset/*.json) ------------------------------------------
# [{pre_text: [..], post_text: [..], table_ori: [[..]], qa: {question, answer, exe_ans,
# program}}]: the answer is the program's result. Numeric answers are kept when the shown
# answer agrees with exe_ans (as a ratio or a percentage, to its rounding); yes/no answers
# (exe_ans "yes" / "no") become yes/no questions. Held out: dev.json.

FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{}.json"
FINQA_MAX_CHARS = 9000


def _finqa_file(split: str) -> Path:
    import urllib.request

    path = CACHE / "finqa" / f"{split}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with urllib.request.urlopen(FINQA_URL.format(split), timeout=300) as r, tmp.open("wb") as f:
            f.write(r.read())
        tmp.replace(path)
    return path


def finqa_consistent(answer: str, exe) -> bool:
    """Does the shown answer equal the executed program's result (to the answer's rounding,
    as a ratio or as a percentage)?"""
    x = parse_number(answer)
    try:
        e = float(exe)
    except (TypeError, ValueError):
        return False
    if x is None or not math.isfinite(e):
        return False
    tol = 0.5 * 10 ** -decimals_of(answer) + 1e-9
    candidates = [e, e * 100] if "%" in answer else [e, e * 100, e / 100]
    return any(abs(x - c) <= max(tol, 0.005 * abs(c)) for c in candidates)


def finqa_load(split: str):
    docs = json.loads(_finqa_file(split).read_text(encoding="utf-8"))
    rows = []
    for doc in docs:
        qa = doc.get("qa") or {}
        answer, exe = str(qa.get("answer", "")).strip(), qa.get("exe_ans")
        context = (
            " ".join(doc.get("pre_text") or [])
            + "\n\nTable:\n"
            + table_text(doc.get("table_ori") or doc.get("table") or [])
            + "\n\n"
            + " ".join(doc.get("post_text") or [])
        ).strip()
        if len(context) > FINQA_MAX_CHARS or not qa.get("question"):
            continue
        if str(exe).lower() in ("yes", "no"):
            kind = "yesno"
            answer = str(exe).lower()
        elif finqa_consistent(answer, exe):
            kind = "number"
        else:
            continue
        rows.append(
            {
                "id": doc["id"],
                "context": context,
                "question": qa["question"],
                "answer": answer,
                "kind": kind,
            }
        )
    return _dataset(rows)


def finqa(row, rng, meta):
    if row["kind"] == "yesno":
        return [
            {
                "state": row["context"],
                "question": {
                    "type": "noul",
                    "instructions": row["question"],
                    "criteria": {"false": "No", "true": "Yes"},
                },
                "expected": row["answer"],
                "family": "numeric_table",
            }
        ]
    x = parse_number(row["answer"])
    if x is None:
        return []
    dec = min(decimals_of(row["answer"]), 4)
    suffix = "%" if row["answer"].strip().endswith("%") else ""
    prefix = "$ " if row["answer"].strip().startswith("$") else ""
    gold = prefix + format_number(x, dec) + suffix
    others = [
        prefix + d + suffix for d in numeric_distractors(x, dec, rng, k=int(rng.integers(3, 5)))
    ]
    return numeric_items(row["context"], row["question"], gold, others, rng, "numeric_table")


# --- temporal-nli (tasksource/temporal-nli: Thukral et al. 2021) ----------------------------
# {Premise, Hypothesis, Label}: templated probes of order, duration and unit magnitude, with
# many repeated rows (deduplicated by the loader). Held out: its test split.

TNLI_LABELS = {
    "entailment": "The premise implies the hypothesis",
    "neutral": "The premise neither implies nor contradicts the hypothesis",
    "contradiction": "The premise contradicts the hypothesis",
}


def temporal_nli_load(split: str):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("tasksource/temporal-nli", f"{split}.csv", repo_type="dataset")
    seen, rows = set(), []
    with Path(path).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["Premise"].strip(), r["Hypothesis"].strip())
            label = r["Label"].strip().lower()
            if key in seen or label not in TNLI_LABELS:
                continue
            seen.add(key)
            rows.append({"premise": key[0], "hypothesis": key[1], "label": label})
    return _dataset(rows)


def temporal_nli(row, rng, meta):
    return [
        {
            "state": f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
            "question": {
                "type": "choice",
                "instructions": "What is the relation between the premise and the hypothesis, "
                "given what the time expressions mean?",
                "criteria": TNLI_LABELS,
            },
            "expected": row["label"],
        }
    ]


# --- Forecasting (YuehHanChen/forecasting: Halawi et al. 2024) -----------------------------
# Resolved binary questions from Metaculus, GJOpen, INFER, Polymarket, Manifold: question,
# background, resolution_criteria, date_begin / date_close / date_resolve_at, resolution
# (0 / 1), community_predictions [[date, p], ...]. The card's own split is temporal (test:
# published after June 1, 2024): train = its train + validation, held out = its test. The
# target is the resolution; the crowd's last forecast before the close date is the
# teacher. Rows whose text reads like a resolution update are dropped.

FORECAST_FILES = {"train": ("train.json", "validation.json"), "test": ("test.json",)}
_FORECAST_LEAK = re.compile(
    r"\bresolved (?:as )?(?:yes|no)\b|\bresolves (?:as )?(?:yes|no)\b|\bhas resolved\b|"
    r"\bupdate[d]?\s*(?:\(|:|\d)|\bedit\s*(?:\(|:)|\bthis question (?:has )?resolved\b",
    re.IGNORECASE,
)
FORECAST_MAX_CHARS = 6000


def forecasting_load(split: str):
    from huggingface_hub import hf_hub_download

    rows = []
    for name in FORECAST_FILES[split]:
        path = hf_hub_download("YuehHanChen/forecasting", name, repo_type="dataset")
        for r in json.loads(Path(path).read_text(encoding="utf-8")):
            if r.get("question_type") != "binary" or r.get("resolution") not in (0, 1, 0.0, 1.0):
                continue
            background = str(r.get("background") or "")
            if background.strip().lower().startswith("not applicable"):
                background = ""
            criteria = str(r.get("resolution_criteria") or "")
            text = f"{background}\n{criteria}"
            if _FORECAST_LEAK.search(text) or len(text) > FORECAST_MAX_CHARS:
                continue
            try:
                preds = json.loads(r.get("community_predictions") or "[]")
            except ValueError:
                preds = []
            close = str(r.get("date_close") or "")[:10]
            before = [float(p) for d, p in preds if str(d)[:10] <= close and p is not None]
            rows.append(
                {
                    "id": str(r.get("url") or r["question"]),
                    "question": r["question"],
                    "background": background,
                    "criteria": criteria,
                    "begin": str(r.get("date_begin") or "")[:10],
                    "close": close,
                    "resolve": str(r.get("date_resolve_at") or "")[:10],
                    "source": str(r.get("data_source") or ""),
                    "resolution": int(r["resolution"]),
                    "crowd": before[-1] if before else -1.0,
                }
            )
    return _dataset(rows)


def forecasting(row, rng, meta):
    lines = [f"Question opened on {row['begin']}; forecasts close on {row['close']}."]
    if row["background"].strip():
        lines.append(f"Background:\n{row['background'].strip()}")
    if row["criteria"].strip():
        lines.append(f"Resolution criteria:\n{row['criteria'].strip()}")
    item = {
        "state": "\n\n".join(lines),
        "question": {
            "type": "noul",
            "instructions": f"As of {row['begin']}, forecast: {row['question'].strip()}",
            "criteria": {
                "false": "It does not happen (resolves no)",
                "true": "It happens (resolves yes)",
            },
        },
        "expected": "yes" if row["resolution"] == 1 else "no",
    }
    crowd = float(row["crowd"])
    if 0.0 <= crowd <= 1.0:
        item["teacher"] = {"no": 1.0 - crowd, "yes": crowd}
    return [item]


# --- Registry ----------------------------------------------------------------------------------

LICENCES: dict[str, tuple[str, str, str]] = {
    "tempreason_l23": (
        "CC BY-SA 3.0",
        "share-alike",
        "tonytan48/TempReason card tag (L2 / L3; Wikidata facts)",
    ),
    "cold_law_fr_x": (
        "CC BY 4.0",
        "yes",
        "harvard-lil/cold-french-law card tag (LEGI: Licence Ouverte); slice [0.035, 0.15)",
    ),
    "argimi_fr_x": (
        "CC BY-SA 4.0",
        "share-alike",
        "artefactory/Argimi-Legal-French-Jurisprudence card tag (juri, slice [0.12, 0.55))",
    ),
    "cetat_fr": (
        "CC BY-SA 4.0",
        "share-alike",
        "artefactory/Argimi-Legal-French-Jurisprudence card tag (cetat: JADE, pseudonymised)",
    ),
    "tram": (
        "MIT",
        "yes",
        ("github.com/EternityYW/TRAM-Benchmark LICENSE; only the program-generated categories "
        "(no MC-TACO / SQuAD / TempEval / MNLI / SNLI / COPA / ROC rows)"),
    ),
    "drop_num": (
        "CC BY-SA 4.0",
        "share-alike",
        "ucinlp/drop card tag (EleutherAI/drop mirror, tagged CC BY 4.0, is read)",
    ),
    "tatqa": ("CC BY 4.0", "yes", "next-tat/TAT-QA card tag"),
    "finqa": (
        "MIT",
        "yes",
        "github.com/czyssrs/FinQA LICENSE (ibm-research/finqa card: CC BY 4.0)",
    ),
    "temporal_nli": ("Apache-2.0", "yes", "tasksource/temporal-nli card tag"),
    "forecasting": (
        "Apache-2.0",
        "yes",
        ("YuehHanChen/forecasting card tag; question texts come from forecasting platforms "
        "(Metaculus, GJOpen, INFER, Polymarket, Manifold) whose terms also apply"),
    ),
}  # fmt: skip


def _source(name: str, *args, license_use: str = "commercial", **kw) -> Source:
    return Source(
        name, *args, LICENCES[name][0], license_use == "commercial", block="targeted",
        license_use=license_use, **kw,
    )  # fmt: skip


TARGETED_V7_SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        _source("tempreason_l23", TEMPREASON_REPO, "l2+l3", "train", "val", tempreason_l23,
                "temporal_facts", "en", cap=17000, load=tempreason_l23_load),
        _source("cold_law_fr_x", COLD_REPO, None, "train", None, cold_law, "temporal_law", "fr",
                cap=22000, load=cold_x_load, train_only=True),
        _source("argimi_fr_x", ARGIMI_REPO, "juri", "train", None, argimi, "legal_outcome",
                "fr", cap=19000, load=argimi_x_load, train_only=True),
        _source("cetat_fr", ARGIMI_REPO, "cetat", "train", None, cetat, "legal_outcome_admin",
                "fr", cap=12000, load=cetat_load, group=lambda r: r["id"]),
        _source("tram", "github:EternityYW/TRAM-Benchmark", None, "train", None, tram,
                "tram", "en", cap=30000, load=tram_load, group=lambda r: r["question"]),
        _source("drop_num", "EleutherAI/drop", None, "train", "validation", drop_num,
                "numeric_reading", "en", cap=30000, load=drop_load),
        _source("tatqa", "next-tat/TAT-QA", None, "train", "dev", tatqa, "numeric_table",
                "en", cap=11000, load=tatqa_load),
        _source("finqa", "github:czyssrs/FinQA", None, "train", "dev", finqa, "numeric_table",
                "en", cap=11000, load=finqa_load),
        _source("temporal_nli", "tasksource/temporal-nli", None, "train", "test",
                temporal_nli, "temporal_nli", "en", cap=20000, load=temporal_nli_load),
        _source("forecasting", "YuehHanChen/forecasting", None, "train", "test", forecasting,
                "forecasting", "en", cap=6000, load=forecasting_load),
    ]
}  # fmt: skip
