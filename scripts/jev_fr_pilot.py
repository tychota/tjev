"""French translations of contender (jev) rows: selection, checks, assembly.

The pilot (mix-v5 source ``jev_fr_pilot``) translates ~300 English rows of the jev block,
temporal and policy rows first. Translation itself is done outside this script (an LLM
translator writes one JSON object per row); this script only selects, checks and assembles,
so it scales to thousands of rows unchanged.

    # 1. pick rows from a built mix: train rows -> pilot train, held-out rows -> pilot held-out
    python scripts/jev_fr_pilot.py select ~/tjev-work/data/mix-v4 OUT/to_translate.jsonl \
        --train 200 --heldout 100
    # 2. translate OUT/to_translate.jsonl -> OUT/translated*.jsonl (see TRANSLATION_BRIEF)
    # 3. check and assemble the source file the builder reads
    python scripts/jev_fr_pilot.py assemble OUT/to_translate.jsonl OUT/translated*.jsonl \
        --out src/tjev/data/assets/jev_fr_pilot.jsonl

A translated row is ``{"id", "state", "instructions", "criteria"}`` (criteria: the same
label ids, or the same number of score levels, with French descriptions); the type, label
ids, target and family are the original's. A translator may drop a row
(``{"id": …, "drop": "reason"}``) whose label depends on English wording (spelling, word
play, English-only identifiers in the decisive text). ``assemble`` rejects a row when its
labels, type or target changed, when a number of the original state is missing from the
French state (amounts, counts and dates must survive localisation: 1,234.50 -> 1 234,50;
dates may be rewritten, so date parts are compared as a multiset of integers), or when the
French text is not French.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

JEV = ("certo", "decider_teacher", "mghafiri_scenarios", "openjev")
# (source, families or None for any, train rows, held-out rows): temporal and policy first
QUOTAS = [
    ("certo", ("finance_reimburse", "finance_refund", "supplied_policy_conjunction",
               "supplied_policy_threshold", "math_shift", "math_budget"), 0.25),
    ("mghafiri_scenarios", ("insurance_claims", "legal_compliance", "hr_recruiting",
                            "banking_fintech", "ecommerce", "customer_support",
                            "healthcare_admin", "government_services"), 0.35),
    ("openjev", ("workflow-controls-v1/customer_service",
                 "workflow-controls-v1/invoice_processing", "customer-control-v1"), 0.20),
    ("decider_teacher", ("situations", "routing", "specific_with_catchall"), 0.20),
]  # fmt: skip
TIME_POLICY = re.compile(
    r"\b(\d+\s+(business\s+|working\s+|calendar\s+)?(days?|weeks?|months?|hours?)|deadline|due"
    r"|within|before|after|expire[sd]?|window|polic(y|ies)|eligible|allowed|refund|reimburs\w*"
    r"|approv\w+|cap|limit|budget|weekday|monday|tuesday|wednesday|thursday|friday|saturday"
    r"|sunday|january|february|march|april|may|june|july|august|september|october|november"
    r"|december|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
ENGLISH_SPECIFIC = re.compile(
    r"\b(spell\w*|grammar|rhym\w*|letters?|word count|synonyms?|puns?|anagrams?|translat\w*"
    r"|english|capitali[sz]\w*|vowels?|plurals?|syllables?|typos?|acronyms?)\b",
    re.IGNORECASE,
)
MAX_WORDS = 160


def read(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def candidate(row: dict) -> bool:
    text = row["state"] + " " + row["question"]["instructions"]
    return (
        len(row["state"].split()) <= MAX_WORDS
        and bool(TIME_POLICY.search(text))
        and not ENGLISH_SPECIFIC.search(text)
        and row.get("lang", "en") == "en"
    )


def select(mix: Path, n_train: int, n_heldout: int, seed: int = 0) -> list[dict]:
    """Rows to translate: pilot train rows from the mix's train files, pilot held-out rows
    from its validation / calibration / test files (so no French held-out row translates an
    English row the model trains on). One row per state, so the pilot does not repeat a
    state under several questions."""
    held = read(mix / "validation.jsonl") + read(mix / "calibration.jsonl")
    for name in JEV:
        held += read(mix / "test" / f"{name}.jsonl")
    pools = {"train": {n: read(mix / "train" / f"{n}.jsonl") for n in JEV}, "heldout": {}}
    for name in JEV:
        pools["heldout"][name] = [r for r in held if r["source"] == name]
    rng = np.random.default_rng(seed)
    out, seen_states = [], set()
    for split, total in (("train", n_train), ("heldout", n_heldout)):
        for name, families, share in QUOTAS:
            want = round(total * share)
            rows = [
                r
                for r in pools[split][name]
                if r["family"] in families and candidate(r) and r["state"] not in seen_states
            ]
            # round-robin over families (certo's are near-templates: few of each)
            by_family: dict[str, list[dict]] = {}
            for i in rng.permutation(len(rows)):
                by_family.setdefault(rows[int(i)]["family"], []).append(rows[int(i)])
            queues = [by_family[f] for f in sorted(by_family)]
            picked = 0
            while picked < want and any(queues):
                for q in queues:
                    while q and q[0]["state"] in seen_states:
                        q.pop(0)
                    if not q or picked == want:
                        continue
                    r = q.pop(0)
                    seen_states.add(r["state"])
                    out.append({**r, "pilot_split": split})
                    picked += 1
    return out


NUMBER = re.compile(r"\d+(?:[.,   ]\d{3})*(?:[.,]\d+)?")


def numbers(text: str, lang: str) -> Counter:
    """The numeric values of a text, normalised across EN / FR number formats."""
    out: Counter = Counter()
    for m in NUMBER.finditer(text):
        s = m.group(0)
        if lang == "en":
            s = s.replace(",", "")
        else:
            s = re.sub(r"[   ](?=\d{3})", "", s).replace(",", ".")
        s = s.replace(" ", "")
        try:
            v = float(s)
        except ValueError:
            continue
        out[f"{v:g}"] += 1
    return out


def integers(text: str) -> Counter:
    return Counter(int(x) for x in re.findall(r"\d+", text))


CLOCK_12H = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([AaPp])\.?[Mm]\b\.?")


def to_24h(text: str) -> str:
    """English 12-hour times as 24-hour ones ("2:00 PM" -> "14:00"), as the French has them."""

    def sub(m: re.Match) -> str:
        h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
        return f"{h}:{m.group(2) or '00'}"

    return CLOCK_12H.sub(sub, text)


FRENCH = re.compile(r"\b(le|la|les|des|est|une|du|pour|dans|que|qui|pas|sur|avec|et)\b")


def check(original: dict, translated: dict) -> list[str]:
    """Reasons to reject a translated row (empty: accepted)."""
    problems = []
    oq, tq = original["question"], translated.get("question", {})
    if tq.get("type") != oq["type"]:
        problems.append("type changed")
    ocrit, tcrit = oq.get("criteria"), tq.get("criteria")
    if isinstance(ocrit, dict):
        if not isinstance(tcrit, dict) or list(tcrit) != list(ocrit):
            problems.append("label ids changed")
    elif not isinstance(tcrit, list) or len(tcrit) != len(ocrit):
        problems.append("score levels changed")
    problems += [
        f"{key} changed"
        for key in ("expected", "target", "labels")
        if original.get(key) != translated.get(key)
    ]
    src, dst = to_24h(original["state"]), translated.get("state", "")
    missing = numbers(src, "en") - numbers(dst, "fr")
    if missing and integers(src) - integers(dst):  # dates may be reformatted: compare parts
        problems.append(f"numbers lost: {sorted(missing)[:5]}")
    text = dst + " " + tq.get("instructions", "")
    if len(FRENCH.findall(text.lower())) + len(re.findall(r"[éèêàçùâîô]", text)) < 3:
        problems.append("not French")
    return problems


# Rows a reviewer dropped after translation (the label rests on English wording).
REVIEW_DROPS = {
    "mghafiri_scenarios-eval-19-3": "policy matches English gendered descriptors literally",
}


def assemble(to_translate: Path, translated: list[Path], out: Path) -> dict:
    originals = {r["id"]: r for r in read(to_translate)}
    rows, report = [], Counter()
    dropped, rejected = {}, {}
    for path in translated:
        for t in read(path):
            o = originals.get(t.get("id"))
            if o is None:
                report["unknown id"] += 1
                continue
            if t.get("drop") or o["id"] in REVIEW_DROPS:
                dropped[o["id"]] = t.get("drop") or REVIEW_DROPS[o["id"]]
                continue
            t = {  # the translator returns only the French text; the rest is the original's
                "state": t.get("state", ""),
                "question": {
                    "type": o["question"]["type"],
                    "instructions": t.get("instructions", ""),
                    "criteria": t.get("criteria"),
                },
                **{k: o[k] for k in ("expected", "target", "labels") if k in o},
            }
            problems = check(o, t)
            if problems:
                rejected[o["id"]] = problems
                continue
            row = {
                "id": f"jev_fr_pilot-{o['pilot_split']}-{o['id']}",
                "state": t["state"],
                "question": t["question"],
                "labels": o["labels"],
                "source": "jev_fr_pilot",
                "family": o["family"],
                "lang": "fr",
                "pilot_split": o["pilot_split"],
                "origin": o["id"],
            }
            row.update({k: o[k] for k in ("expected", "target") if k in o})
            rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rows, key=lambda r: r["id"])
        )
    return {
        "selected": len(originals),
        "kept": len(rows),
        "kept_by_split": dict(Counter(r["pilot_split"] for r in rows)),
        "kept_by_origin": dict(Counter(r["origin"].split("-")[0] for r in rows)),
        "dropped": dropped,
        "rejected": rejected,
        "missing": sorted(set(originals) - {r["origin"] for r in rows} - set(dropped)
                          - set(rejected)),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("mix", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--train", type=int, default=200)
    s.add_argument("--heldout", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("assemble")
    a.add_argument("to_translate", type=Path)
    a.add_argument("translated", nargs="+")
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "select":
        rows = select(args.mix.expanduser(), args.train, args.heldout, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(json.dumps(Counter(f"{r['pilot_split']}/{r['source']}" for r in rows), indent=1))
    else:
        files = [Path(p) for p in args.translated]
        summary = assemble(args.to_translate, files, args.out)
        print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
