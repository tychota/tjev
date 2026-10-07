"""jev_fr_mt: QE-gated EN→FR machine translation of the exhausted contender sources.

The contender sources with human-quality (or teacher-written) English decisions — plumb,
decider_teacher, mghafiri_scenarios — are seen 1.5–2 times in mix-v6 and cannot grow. Their
rows are translated to French with an open MT model and kept only when a CPU-feasible
quality gate passes:

1. **Labels.** The question type, the label ids / score levels, ``expected`` / ``target``
   and ``teacher`` are copied, never translated (only the criteria *descriptions* are).
   Rows whose question rests on English wording (spelling, grammar, letters…) are skipped.
2. **Numbers, dates, times.** Dates are read in both languages into (year, month, day)
   ("Tuesday, March 17, 2026" = "mardi 17 mars 2026"; ISO dates as such), weekdays attached
   to dates must map onto each other, 12-hour times are compared as 24-hour ones, and every
   remaining number (EN / FR digit grouping and decimal marks) must survive, as a multiset,
   both ways. Since no date or number may change, any date arithmetic the label depends on
   is unchanged; a row whose dates do not survive is rejected, never patched.
3. **Entities.** Codes and identifiers (``SO-26-118374``, ``KB-2217``, ``email_send``),
   e-mails and URLs must appear verbatim (strict). Proper names and acronyms must survive
   at a recall of at least ``NAME_RECALL``.
4. **Round trip.** The French is translated back to English (greedy) and scored with
   chrF against the source (``MIN_CHRF`` for the row, ``MIN_SEGMENT_CHRF`` for its worst
   segment of 40+ characters). Length ratios and repeated n-grams catch hallucinations.
5. **French.** The output must read as French (function words and accents).

6. **CometKiwi** (optional, on top of 1–5, which stay mandatory). Reference-free QE of each
   (English segment, French segment) pair with ``Unbabel/wmt22-cometkiwi-da`` (CC BY-NC-SA
   4.0, gated; the project accepts NC-SA). The row's score is the length-weighted mean of
   its segments' scores; its worst segment of 20+ characters is logged too. A row needs
   ``kiwi >= MIN_KIWI`` (0.72) and ``worst >= MIN_KIWI_SEGMENT`` (0.45). The scores (and the round-trip
   chrF) are logged in ``meta`` and in the segment tables, so ``regate`` can change the
   thresholds without translating again. Structured (JSON) states are translated value by value: keys and
identifier-like values are kept. Translation is segment-wise (lines, then sentences), so
the layout of long states is preserved.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

MT_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-fr"  # card tag: license:cc-by-4.0, not gated
RT_MODEL = "Helsinki-NLP/opus-mt-tc-big-fr-en"  # card tag: license:cc-by-4.0, not gated
MT_LICENCE = "CC-BY-4.0"
UPSTREAM = {  # base source -> licence of its rows (tjev.data.licences)
    "plumb": "Apache-2.0 (crh225/plumb-decisions; rows written by a Qwen model upstream)",
    "decider_teacher": "Apache-2.0 (github.com/Mapika/decider teacher_data)",
    "mghafiri_scenarios": "MIT (mghafiri/decision-model-scenarios-v2; rows written by Claude "
    "upstream)",
}
KIWI_MODEL = "Unbabel/wmt22-cometkiwi-da"  # card tag: license:cc-by-nc-sa-4.0, gated (auto)
KIWI_LICENCE = "CC-BY-NC-SA-4.0"
# from the 200-row sample (docs/data.md, jev_fr_mt): a row needs a mean score of 0.72 and no
# segment of 20+ characters under 0.45 (the latter drops hallucinated titles, mangled shell
# commands and translated file names); None = logged only
MIN_KIWI: float | None = 0.72
MIN_KIWI_SEGMENT: float | None = 0.45
MIN_KIWI_SEGMENT_CHARS = 20  # the worst-segment score ignores shorter segments
SOURCE = "jev_fr_mt"
MIN_CHRF = 50.0
MIN_SEGMENT_CHRF = 25.0
NAME_RECALL = 0.8
MAX_SEGMENT_CHARS = 600  # longer sentences are split at ; / , / spaces before MT

ENGLISH_SPECIFIC = re.compile(  # as scripts/jev_fr_pilot.py: the label rests on English wording
    r"\b(spell\w*|grammar|rhym\w*|letters?|word count|synonyms?|puns?|anagrams?|translat\w*"
    r"|english|capitali[sz]\w*|vowels?|plurals?|syllables?|typos?|acronyms?)\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------------------
# Dates, times and numbers

MONTHS_EN = ["january", "february", "march", "april", "may", "june", "july", "august",
             "september", "october", "november", "december"]  # fmt: skip
MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
             "septembre", "octobre", "novembre", "décembre"]  # fmt: skip
WEEKDAYS_EN = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MON_EN = (
    {m: i + 1 for i, m in enumerate(MONTHS_EN)}
    | {m[:3]: i + 1 for i, m in enumerate(MONTHS_EN)}
    | {"sept": 9}
)
_MON_FR = {m: i + 1 for i, m in enumerate(MONTHS_FR)} | {
    "fevrier": 2, "aout": 8, "decembre": 12, "janv": 1, "févr": 2, "fevr": 2, "avr": 4,
    "juil": 7, "sept": 9, "oct": 10, "nov": 11, "déc": 12, "dec": 12,
}  # fmt: skip
# French text: French month names, and the English ones an MT model may leave in place
_MON_ANY = _MON_EN | _MON_FR
_EN_MONTH = r"(?P<mon>" + "|".join(sorted(_MON_EN, key=len, reverse=True)) + r")\.?"
_FR_MONTH = r"(?P<mon>" + "|".join(sorted(_MON_ANY, key=len, reverse=True)) + r")\.?"
_EN_WD = r"(?:(?P<wd>" + "|".join(WEEKDAYS_EN) + r"),?\s+)?"
_FR_WD = r"(?:(?P<wd>" + "|".join(WEEKDAYS_FR) + r")\s+)?"
_YEAR = r"(?:,?[ \t\u00a0]+(?P<y>\d{4}))?"  # never across a line break
_ORD = r"(?:st|nd|rd|th)?"
ISO = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")
EN_DATES = [  # "Tuesday, March 17, 2026", "March 17th", "17 March 2026", "17th of March"
    re.compile(
        _EN_WD + r"\b" + _EN_MONTH + r"[ \t\u00a0]+(?P<d>\d{1,2})" + _ORD + r"\b" + _YEAR,
        re.IGNORECASE,
    ),
    re.compile(
        _EN_WD
        + r"(?:the\s+)?\b(?P<d>\d{1,2})"
        + _ORD
        + r"[ \t\u00a0]+(?:of\s+)?"
        + _EN_MONTH
        + _YEAR,
        re.IGNORECASE,
    ),
    re.compile(r"\b" + _EN_MONTH + r"[ \t\u00a0]+(?P<y>\d{4})\b", re.IGNORECASE),  # "March 2026"
]
FR_DATES = [  # "mardi 17 mars 2026", "1er mars", "15 janv. 2023", "mars 2026", "avril 7, 2026"
    re.compile(
        _FR_WD + r"(?:le\s+)?(?P<d>1er|\d{1,2})[ \t\u00a0]+" + _FR_MONTH + _YEAR + r"(?!\w)",
        re.IGNORECASE,
    ),
    re.compile(
        _FR_WD + r"\b" + _FR_MONTH + r"[ \t\u00a0]+(?P<d>\d{1,2})" + _ORD + r"\b" + _YEAR,
        re.IGNORECASE,
    ),
    re.compile(r"\b" + _FR_MONTH + r"[ \t\u00a0]+(?P<y>\d{4})\b", re.IGNORECASE),
]
EN_TIME = [
    re.compile(r"\b(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>[ap])\.?\s?m\b\.?", re.IGNORECASE),
    re.compile(r"\b(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)\b"),
]
FR_TIME = [
    # "à 14 heures" is a time, "pendant 2 heures" a duration
    re.compile(
        r"\b(?:à|avant|après|vers|dès|jusqu'à|jusqu’à)\s+(?P<h>[01]?\d|2[0-3])\s?heures?"
        r"(?:\s?(?P<m>[0-5]\d)\b)?",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?P<h>[01]?\d|2[0-3])\s?h\s?(?P<m>[0-5]\d)?\b"),
    re.compile(r"\b(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)\b"),
    # a 12-hour time left as is by the MT ("2 pm")
    re.compile(r"\b(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>[ap])\.?\s?m\b\.?", re.IGNORECASE),
]
NUMBER = re.compile(r"\d+(?:[,.   ]\d+)*")


@dataclass
class Quantities:
    """What must survive translation: dates, weekdays named with a date, times, numbers."""

    dates: Counter = field(default_factory=Counter)
    weekdays: Counter = field(default_factory=Counter)  # weekday index (0 = Monday)
    times: Counter = field(default_factory=Counter)
    numbers: Counter = field(default_factory=Counter)


def _blank(text: str, span: tuple[int, int]) -> str:
    return text[: span[0]] + " " * (span[1] - span[0]) + text[span[1] :]


def _number_value(tok: str, lang: str) -> str | None:
    """A digit token as a normalised value: EN "1,234.50" / FR "1 234,50" -> "1234.5"."""
    tok = tok.strip()
    seps = re.findall(r"[,.   ]", tok)
    if not seps:
        return str(int(tok))
    groups = re.split(r"[,.   ]", tok)
    if any(s in "   " for s in seps):  # spaces only group thousands (FR)
        if not all(len(g) == 3 for g in groups[1:-1]):
            return None
        last_sep = seps[-1]
        if last_sep in ",." and len(seps) > 1:
            whole, dec = "".join(groups[:-1]), groups[-1]
            return f"{float(whole + '.' + dec):g}"
        if last_sep in ",.":  # "1 234,5" handled above; "12,5" alone
            return f"{float(groups[0] + '.' + groups[1]):g}"
        return str(int("".join(groups))) if len(groups[-1]) == 3 else None
    if len(set(seps)) == 2:  # both marks: the last one is the decimal mark
        dec = seps[-1]
        whole = "".join(groups[:-1])
        return f"{float(whole + '.' + groups[-1]):g}" if dec else None
    sep = seps[0]
    thousands = all(len(g) == 3 for g in groups[1:]) and 1 <= len(groups[0]) <= 3
    decimal_mark = "." if lang == "en" else ","
    if len(seps) == 1 and sep == decimal_mark:
        return f"{float(groups[0] + '.' + groups[1]):g}"
    if thousands:
        return str(int("".join(groups)))
    if len(seps) == 1:  # the other language's decimal mark, left by the MT ("3.5" in FR)
        return f"{float(groups[0] + '.' + groups[1]):g}"
    return None


def quantities(text: str, lang: str) -> Quantities:
    """Dates (as (year or 0, month, day or 0)), weekdays of dates, times (24h), numbers."""
    q = Quantities()
    months = _MON_EN if lang == "en" else _MON_ANY
    weekdays = WEEKDAYS_EN if lang == "en" else WEEKDAYS_FR
    for m in ISO.finditer(text):
        q.dates[(int(m["y"]), int(m["m"]), int(m["d"]))] += 1
        text = _blank(text, m.span())
    for pattern in EN_DATES if lang == "en" else FR_DATES:
        for m in pattern.finditer(text):
            mon = months.get(m["mon"].lower().rstrip("."))
            if mon is None:
                continue
            day = m.groupdict().get("d")
            day = 0 if day is None else 1 if day.lower() == "1er" else int(day)
            if day > 31:
                continue
            q.dates[(int(m["y"] or 0), mon, day)] += 1
            if m.groupdict().get("wd"):
                q.weekdays[weekdays.index(m["wd"].lower())] += 1
            text = _blank(text, m.span())
    for pattern in EN_TIME if lang == "en" else FR_TIME:
        for m in pattern.finditer(text):
            h, mm = int(m["h"]), int(m["m"] or 0)
            ap = m.groupdict().get("ap")
            if ap:
                if not 1 <= h <= 12:
                    continue
                h = h % 12 + (12 if ap.lower() == "p" else 0)
            q.times[(h, mm)] += 1
            text = _blank(text, m.span())
    if lang == "fr":  # ordinals: "1er", "2e", "3ème" -> the number
        text = re.sub(r"\b(\d+)(?:er|re|e|ème|eme)\b", r"\1", text)
    for m in NUMBER.finditer(text):
        tok = m.group(0).rstrip("   ,.")
        try:
            value = _number_value(tok, lang)
        except ValueError:
            value = None
        if value is None:  # an odd grouping: count its digit runs
            for part in re.findall(r"\d+", tok):
                q.numbers[str(int(part))] += 1
        else:
            q.numbers[value] += 1
    return q


def compare_quantities(src: str, dst: str) -> list[str]:
    """Problems (empty: every date, weekday, time and number survived, both ways)."""
    a, b = quantities(src, "en"), quantities(dst, "fr")
    problems = []
    for what in ("dates", "weekdays", "times", "numbers"):
        x, y = getattr(a, what), getattr(b, what)
        if x != y:
            lost, added = x - y, y - x
            problems.append(f"{what}: lost {sorted(lost)[:4]} added {sorted(added)[:4]}")
    return problems


# --------------------------------------------------------------------------------------
# Entities

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+")
CODE = re.compile(  # identifiers: letters and digits joined by - _ . / (KB-2217, v5.2, a_b)
    r"\b(?=[\w./-]*\d)(?=[\w./-]*[A-Za-z])[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)+\b"
    r"|\b[A-Z]{1,}\d+[A-Z0-9]*\b"
    r"|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"
)
WORD = re.compile(r"\b[^\W\d_][\w'’-]*\b")
NAME_STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "we",
        "you",
        "he",
        "she",
        "they",
        "our",
        "your",
        "their",
        "my",
        "his",
        "her",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "please",
        "note",
        "dear",
        "hi",
        "hello",
        "thanks",
        "regards",
        "yes",
        "no",
        "not",
        "all",
        "any",
        "each",
        "every",
        "for",
        "from",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "as",
        "so",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    + MONTHS_EN
    + [m[:3] for m in MONTHS_EN]
    + ["sept", "mon", "tue", "wed", "thu", "fri", "sat", "sun"]
)


def strict_entities(text: str) -> Counter:
    """E-mails, URLs and identifiers: must appear verbatim in the translation."""
    out: Counter = Counter()
    for pattern in (EMAIL, URL):
        for m in pattern.finditer(text):
            out[m.group(0).rstrip(".,;:)")] += 1
            text = _blank(text, m.span())
    for m in CODE.finditer(text):
        tok = m.group(0).rstrip(".")
        if is_identifier(tok):
            out[tok] += 1
    return out


QUARTER = re.compile(r"Q[1-4]|H[12]|FY\d{2,4}")  # localised by the MT (T3, S1, exercice 2025)


def is_identifier(tok: str) -> bool:
    """A code that must stay verbatim: snake_case, or parts that are digits, upper-case
    codes or a letter + digits (KB-2217, SO-26-118374, v5.2). Hyphenated words
    ("15-minute", "Tier-1", "pre-2022") and quarters are translatable."""
    if re.fullmatch(r"\d+(?:[.,]\d+)*", tok) or QUARTER.fullmatch(tok):
        return False
    if "_" in tok:
        return True
    parts = re.split(r"[-./]", tok)
    return all(re.fullmatch(r"\d+|[A-Z][A-Z0-9]*|[A-Za-z]\d+[A-Za-z]?", p) for p in parts) and any(
        re.search(r"\d", p) for p in parts
    )


def lowercase_vocabulary(texts: Iterable[str]) -> set[str]:
    """Words written in lower case somewhere in ``texts``: common words, not names."""
    return {w.lower() for t in texts for w in WORD.findall(t) if w[0].islower()}


def soft_entities(text: str, vocabulary: set[str] | None = None) -> set[str]:
    """Proper names and acronyms: capitalised mid-sentence words never written lowercase in
    the text (or in ``vocabulary``, the batch's lower-case words), and all-caps words;
    title-case lines (headings) are ignored. Possessives are stripped."""
    lower = lowercase_vocabulary([text]) | (vocabulary or set())
    out = set()
    for line in text.split("\n"):
        words = WORD.findall(line)
        if len(words) >= 3 and sum(w[0].isupper() for w in words) / len(words) > 0.6:
            continue  # a heading in title case
        for sentence in re.split(r"(?<=[.!?:;])\s+|^\s*[-•*]\s*", line):
            tokens = WORD.findall(sentence)
            for i, w in enumerate(tokens):
                if re.search(r"['’](?!s$)", w):  # contractions: I'm, don't
                    continue
                w = re.sub(r"['’]s$", "", w)
                lw = w.lower()
                if len(w) < 3 or lw in NAME_STOP or lw in lower or re.search(r"\d", w):
                    continue
                if w.isupper() or (i > 0 and w[0].isupper()):
                    out.add(w)
    return out


def compare_entities(src: str, dst: str, vocabulary: set[str] | None = None,
                     back: str = "") -> tuple[list[str], float]:  # fmt: skip
    """Identifiers must be in ``dst`` verbatim. A name counts as kept when it is in ``dst``
    or in ``back`` (the round trip): a common word the MT translates comes back, a name
    the MT mangles does not."""
    problems = []
    lost = strict_entities(src) - strict_entities(dst)
    if lost:
        problems.append(f"identifiers lost: {sorted(lost)[:4]}")
    names = soft_entities(src, vocabulary)
    kept = {n for n in names if n in dst or n in back}
    recall = len(kept) / len(names) if names else 1.0
    if recall < NAME_RECALL:
        problems.append(f"names: recall {recall:.2f} ({sorted(names - kept)[:4]})")
    return problems, recall


# --------------------------------------------------------------------------------------
# Segmentation and quality

FRENCH = re.compile(r"\b(le|la|les|des|est|une|du|pour|dans|que|qui|pas|sur|avec|et)\b")


def is_french(text: str) -> bool:
    return len(FRENCH.findall(text.lower())) + len(re.findall(r"[éèêàçùâîô]", text)) >= 3


def translatable(s: str, min_words: int = 1) -> bool:
    """A string worth translating: ``min_words`` or more words of letters, not an
    identifier, e-mail or URL."""
    s = s.strip()
    if not s or EMAIL.fullmatch(s) or URL.fullmatch(s) or CODE.fullmatch(s):
        return False
    return len(re.findall(r"[^\W\d_]{2,}", s)) >= min_words


def split_sentences(line: str) -> list[str]:
    """A line as sentence pieces whose concatenation (with the kept separators) is the line."""
    pieces = re.split(r"(?<=[.!?])(\s+)(?=[A-Z\"'“(])", line)
    out = []
    for p in pieces:
        while len(p) > MAX_SEGMENT_CHARS:  # cut very long sentences at a ; , or space
            cut = max(p.rfind(c, 0, MAX_SEGMENT_CHARS) for c in ";,")
            cut = cut + 1 if cut > MAX_SEGMENT_CHARS // 3 else p.rfind(" ", 0, MAX_SEGMENT_CHARS)
            cut = cut if cut > 0 else MAX_SEGMENT_CHARS
            out.append(p[:cut])
            p = p[cut:]
        out.append(p)
    return out


# Markup the MT must not touch (the 200-row sample lost `|` from tables and shell commands):
# fenced code blocks and markdown table rules stay as lines; a line with pipes is cut at
# them (table cells / pipeline stages are translated one by one, the pipes stay, a stage
# that looks like code stays); an inline code span is replaced by a placeholder the MT
# copies (an identifier, checked verbatim), restored after translation.
FENCE = re.compile(r"^\s*(?:```|~~~)")
TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$")
CODE_SPAN = re.compile(r"`[^`\n]+`")
PLACEHOLDER = "TJVC{}"
PLACEHOLDER_RE = re.compile(r"TJVC(\d+)")
SHELL = re.compile(
    r"^\s*(?:\$\s*)?(?:grep|egrep|awk|sed|cat|ls|sort|head|tail|wc|xargs|cut|tr|uniq|echo"
    r"|docker|kubectl|git|npm|npx|yarn|pip|curl|wget|find|ps|kill|tee|jq|less|more|python3?"
    r"|bash|sh|sudo|tar|gzip|zcat|journalctl|systemctl|du|df|top|netstat|ss|lsof|awk|perl"
    r"|node|go|cargo|make|helm|terraform|aws|gcloud|az|ssh|scp|rsync|chmod|chown|mkdir|rm"
    r"|cp|mv|touch|env|export|select|SELECT)\b"
)
CODE_CHARS = re.compile(r"[=${}<>;\\]|--|&&|\w\(\)|^\s*[\w.-]*/[\w./-]+\s*$")


def looks_like_code(text: str) -> bool:
    """A pipeline stage or cell that is code (a command, a path, an expression)."""
    return bool(SHELL.match(text) or CODE_CHARS.search(text))


def protect(line: str) -> tuple[str, list[str]]:
    """``line`` with its inline code spans replaced by placeholders, and the spans."""
    spans: list[str] = []

    def sub(m: re.Match) -> str:
        spans.append(m.group(0))
        return PLACEHOLDER.format(len(spans) - 1)

    return CODE_SPAN.sub(sub, line), spans


def restore(text: str, spans: list[str]) -> str:
    """Placeholders back to their code spans (each once; a lost or repeated placeholder
    shows in the markup check of the row)."""
    if not spans:
        return text
    done: set[int] = set()

    def sub(m: re.Match) -> str:
        j = int(m.group(1))
        if j < len(spans) and j not in done:
            done.add(j)
            return spans[j]
        return m.group(0)

    return PLACEHOLDER_RE.sub(sub, text)


@dataclass
class Plan:
    """A text cut into fixed parts and translatable segments; ``assemble`` rebuilds it.
    ``spans``: the code spans of each part's line (placeholders restored on assembly)."""

    parts: list[tuple[bool, str]]  # (translate?, text)
    spans: list[list[str]] = field(default_factory=list)

    @property
    def segments(self) -> list[str]:
        return [t.strip() for k, t in self.parts if k]

    def assemble(self, translated: Iterable[str]) -> str:
        it = iter(translated)
        out = []
        for i, (k, t) in enumerate(self.parts):
            spans = self.spans[i] if i < len(self.spans) else []
            if k:
                lead = t[: len(t) - len(t.lstrip())]
                trail = t[len(t.rstrip()) :]
                out.append(lead + restore(next(it), spans) + trail)
            else:
                out.append(restore(t, spans))
        return "".join(out)


def _plan_line(line: str, parts: list, spans: list, line_spans: list[str]) -> None:
    m = re.match(r"^(\s*(?:[-•*]|\d+[.)]|[A-Za-z][.)])?\s*)", line)
    lead = m.group(1) if m else ""
    if lead:
        parts.append((False, lead))
        spans.append(line_spans)
    for piece in split_sentences(line[len(lead) :]):
        parts.append((translatable(piece), piece))
        spans.append(line_spans)


def plan_text(text: str) -> Plan:
    """Lines, then sentences. A line's list marker or "Label:" prefix of an identifier-like
    label stays as is; segments without words stay as is. Code fences, table rules, pipes
    and code-like pipeline stages stay as is; inline code spans become placeholders."""
    parts: list[tuple[bool, str]] = []
    spans: list[list[str]] = []
    fenced = False
    for i, line in enumerate(text.split("\n")):
        if i:
            parts.append((False, "\n"))
            spans.append([])
        if FENCE.match(line) or fenced or TABLE_RULE.match(line):
            if FENCE.match(line):
                fenced = not fenced
            parts.append((False, line))
            spans.append([])
            continue
        protected, line_spans = protect(line)
        if "|" not in protected:
            _plan_line(protected, parts, spans, line_spans)
            continue
        for k, cell in enumerate(re.split(r"(\s*\|\s*)", protected)):
            if not cell:
                continue
            if k % 2 or looks_like_code(cell):
                parts.append((False, cell))
                spans.append(line_spans)
            else:
                _plan_line(cell, parts, spans, line_spans)
    return Plan(parts, spans)


def markup_problems(src: str, dst: str) -> list[str]:
    """Pipes, backticks and code spans must survive, and no placeholder may be left."""
    problems = [
        f"markup: {mark!r} {src.count(mark)} -> {dst.count(mark)}"
        for mark in ("|", "`")
        if src.count(mark) != dst.count(mark)
    ]
    if Counter(CODE_SPAN.findall(src)) != Counter(CODE_SPAN.findall(dst)):
        problems.append("markup: code spans changed")
    if PLACEHOLDER_RE.search(dst) and not PLACEHOLDER_RE.search(src):
        problems.append("markup: placeholder left")
    return problems


def _json_strings(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        if translatable(obj, min_words=2):  # single-word values are often enums
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _json_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _json_strings(v, out)


JSON_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"(\s*:)?')


def json_substitute(raw: str, table: dict[str, str]) -> str:
    """``raw`` JSON text with its string *values* replaced through ``table``; keys, layout
    and number formatting stay byte for byte."""

    def sub(m: re.Match) -> str:
        if m.group(2):  # a key
            return m.group(0)
        value = json.loads(m.group(0))
        return json.dumps(table[value], ensure_ascii=False) if value in table else m.group(0)

    return JSON_STRING.sub(sub, raw)


# Post-edits of systematic MT errors on our schema's vocabulary: "the state" (the item's
# input) is not the government.
GLOSSARY = [
    (re.compile(r"\b([Ll])'État\b"), r"\1a situation"),
    (re.compile(r"\b([Dd])e l'État\b"), r"\1e la situation"),
    (re.compile(r"\b([Dd])ans l'État\b"), r"\1ans la situation"),
]


def post_edit(text: str) -> str:
    for pattern, repl in GLOSSARY:
        text = pattern.sub(repl, text)
    return text


def parse_json_state(state: str) -> Any | None:
    s = state.strip()
    if not s.startswith(("{", "[")):
        return None
    try:
        return json.loads(s)
    except ValueError:
        return None


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """chrF (character n-grams 1..n, whitespace removed, F-beta), 0–100."""
    h, r = re.sub(r"\s+", "", hyp), re.sub(r"\s+", "", ref)
    if not h or not r:
        return 100.0 if h == r else 0.0
    precisions, recalls = [], []
    for k in range(1, n + 1):
        hc = Counter(h[i : i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i : i + k] for i in range(len(r) - k + 1))
        if not hc or not rc:
            continue
        match = sum((hc & rc).values())
        precisions.append(match / sum(hc.values()))
        recalls.append(match / sum(rc.values()))
    if not precisions:
        return 0.0
    p, rr = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    if p + rr == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * rr / (beta**2 * p + rr)


def degenerate(src: str, dst: str) -> str | None:
    """A hallucination sign in one segment: an odd length ratio or a looping n-gram."""
    if len(src) >= 20 and not 0.6 <= len(dst) / len(src) <= 2.2:
        return f"length ratio {len(dst) / len(src):.2f}"
    words = dst.lower().split()
    grams = Counter(tuple(words[i : i + 3]) for i in range(len(words) - 2))
    src_words = src.lower().split()
    src_grams = Counter(tuple(src_words[i : i + 3]) for i in range(len(src_words) - 2))
    if (
        grams
        and max(grams.values()) >= 3
        and max(grams.values()) > 2 * max(src_grams.values() or [1])
    ):
        return "repeated n-gram"
    return None


# --------------------------------------------------------------------------------------
# Rows

Translate = Callable[[list[str]], list[str]]


@dataclass
class RowPlan:
    row: dict
    state: Plan | None  # text state
    json_state: Any | None  # structured state
    json_strings: list[str]
    instructions: Plan
    criteria: list[tuple[Any, Plan]]  # (label key or level index, description)
    skip: str | None = None

    def segments(self) -> list[str]:
        segs = list(self.state.segments) if self.state else list(self.json_strings)
        segs += self.instructions.segments
        for _, p in self.criteria:
            segs += p.segments
        return segs


def plan_row(row: dict) -> RowPlan:
    q = row["question"]
    crit = q.get("criteria") or {}
    pairs = list(crit.items()) if isinstance(crit, dict) else list(enumerate(crit))
    text = q.get("instructions", "") + " " + " ".join(str(v) for _, v in pairs)
    skip = "english-specific question" if ENGLISH_SPECIFIC.search(text) else None
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"])
    js = parse_json_state(state)
    strings: list[str] = []
    if js is not None:
        _json_strings(js, strings)
        strings = list(dict.fromkeys(strings))
    return RowPlan(
        row=row,
        state=None if js is not None else plan_text(state),
        json_state=js,
        json_strings=strings,
        instructions=plan_text(q.get("instructions", "")),
        criteria=[(k, plan_text(str(v))) for k, v in pairs],
        skip=skip,
    )


def _assemble(rp: RowPlan, table: dict[str, str]) -> dict:
    row = copy.deepcopy(rp.row)
    if rp.json_state is not None:
        raw = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"])
        row["state"] = json_substitute(raw, table)
    else:
        row["state"] = rp.state.assemble(table[s] for s in rp.state.segments)
    q = row["question"]
    q["instructions"] = post_edit(
        rp.instructions.assemble(table[s] for s in rp.instructions.segments)
    )
    descs = [post_edit(p.assemble(table[s] for s in p.segments)) for _, p in rp.criteria]
    if isinstance(q.get("criteria"), dict):
        q["criteria"] = dict(zip(q["criteria"], descs, strict=True))
    elif isinstance(q.get("criteria"), list):
        q["criteria"] = descs
    return row


def _flat(row: dict) -> str:
    q = row["question"]
    crit = q.get("criteria") or {}
    descs = crit.values() if isinstance(crit, dict) else crit
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"])
    return "\n".join([state, q.get("instructions", ""), *map(str, descs)])


def kiwi_of(segments: list[str], kiwi_table: dict[str, float]) -> tuple[float, float]:
    """(row score, worst segment score): the length-weighted mean of the segments' scores,
    and the lowest score among segments of ``MIN_KIWI_SEGMENT_CHARS``+ characters."""
    scored = [s for s in segments if s in kiwi_table]
    if not scored:
        return 1.0, 1.0
    total = sum(len(s) for s in scored)
    mean = sum(kiwi_table[s] * len(s) for s in scored) / total
    long = [kiwi_table[s] for s in scored if len(s) >= MIN_KIWI_SEGMENT_CHARS]
    return mean, min(long) if long else mean


def check_row(original: dict, translated: dict, rt_table: dict[str, str],
              fr_table: dict[str, str], segments: list[str],
              vocabulary: set[str] | None = None, kiwi_table: dict[str, float] | None = None,
              min_kiwi: float | None = MIN_KIWI,
              min_kiwi_segment: float | None = MIN_KIWI_SEGMENT) -> dict:  # fmt: skip
    """The gate's measurements and verdict for one row. ``kiwi_table``: CometKiwi score of
    each source segment's translation (None: not scored); ``min_kiwi``: the row threshold
    (None: logged, not gated)."""
    problems: list[str] = []
    oq, tq = original["question"], translated["question"]
    if oq["type"] != tq["type"]:
        problems.append("type changed")
    oc, tc = oq.get("criteria"), tq.get("criteria")
    if isinstance(oc, dict) and list(oc) != list(tc):
        problems.append("label ids changed")
    if isinstance(oc, list) and len(oc) != len(tc):
        problems.append("score levels changed")
    problems.extend(
        f"{key} changed"
        for key in ("expected", "target", "labels", "teacher")
        if original.get(key) != translated.get(key)
    )
    src, dst = _flat(original), _flat(translated)
    checks = {"labels": not problems}
    qp = compare_quantities(src, dst)
    checks["quantities"] = not qp
    back = " ".join(rt_table[s] for s in segments)
    ep, recall = compare_entities(src, dst, vocabulary, back)
    checks["entities"] = not ep
    mp = markup_problems(src, dst)
    checks["markup"] = not mp
    seg_scores = [chrf(rt_table[s], s) for s in segments if len(s) >= 40]
    row_chrf = (
        chrf(" ".join(rt_table[s] for s in segments), " ".join(segments)) if segments else 100.0
    )
    worst = min(seg_scores) if seg_scores else row_chrf
    checks["round_trip"] = row_chrf >= MIN_CHRF and worst >= MIN_SEGMENT_CHRF
    degen = [d for s in segments if (d := degenerate(s, fr_table[s]))]
    checks["fluency"] = not degen
    checks["french"] = is_french(dst)
    kiwi = worst_kiwi = None
    if kiwi_table is not None:
        kiwi, worst_kiwi = kiwi_of(segments, kiwi_table)
        if min_kiwi is not None or min_kiwi_segment is not None:
            checks["kiwi"] = (min_kiwi is None or kiwi >= min_kiwi) and (
                min_kiwi_segment is None or worst_kiwi >= min_kiwi_segment
            )
            if not checks["kiwi"]:
                problems.append(f"CometKiwi {kiwi:.3f} (worst segment {worst_kiwi:.3f})")
    problems += qp + ep + mp + degen
    if not checks["round_trip"]:
        problems.append(f"round trip chrF {row_chrf:.1f} (worst segment {worst:.1f})")
    if not checks["french"]:
        problems.append("not French")
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "problems": problems,
        "chrf": round(row_chrf, 1),
        "worst_segment_chrf": round(worst, 1),
        "name_recall": round(recall, 3),
        "segments": len(segments),
        "kiwi": None if kiwi is None else round(kiwi, 4),
        "kiwi_worst_segment": None if worst_kiwi is None else round(worst_kiwi, 4),
    }


def regate(verdict: dict, *, min_kiwi: float | None = None,
           min_kiwi_segment: float | None = None, min_chrf: float = MIN_CHRF,
           min_segment_chrf: float = MIN_SEGMENT_CHRF) -> bool:  # fmt: skip
    """A logged verdict (``meta`` of a kept row, or a rejected row) decided again with other
    thresholds: the mandatory checks (labels, quantities, entities, fluency, French) as
    logged, the round trip and CometKiwi from their logged scores."""
    checks = verdict.get("checks", {})
    if "chrf" not in verdict:  # skipped before translation
        return False
    fixed = all(v for k, v in checks.items() if k not in ("round_trip", "kiwi"))
    trip = verdict["chrf"] >= min_chrf and verdict["worst_segment_chrf"] >= min_segment_chrf
    kiwi = min_kiwi is None or (verdict.get("kiwi") is not None and verdict["kiwi"] >= min_kiwi)
    seg = min_kiwi_segment is None or (
        verdict.get("kiwi_worst_segment") is not None
        and verdict["kiwi_worst_segment"] >= min_kiwi_segment
    )
    return fixed and trip and kiwi and seg


def base_source(row: dict) -> str:
    name = row.get("source") or ""
    if name in UPSTREAM:
        return name
    return next((s for s in UPSTREAM if str(row.get("id", "")).startswith(s)), name)


def table_translators(tables: dict) -> tuple[Translate, Translate]:
    """Translators replaying saved segment tables (``timings["tables"]``): re-gate offline."""
    fr, rt = tables["fr"], tables["rt"]
    back = {fr[s]: rt[s] for s in fr}
    return (lambda xs: [fr[x] for x in xs]), (lambda xs: [back[x] for x in xs])


Score = Callable[[list[tuple[str, str]]], list[float]]  # (source, translation) -> QE score
GATE_WORKERS = 16  # the gate of a large batch runs in a fork pool (rows are independent)
GATE_PARALLEL_MIN = 256
_GATE: dict = {}


def _gate(p: RowPlan, fr_table, rt_table, vocabulary, kiwi_table, min_kiwi, min_kiwi_segment,
          mt_model, qe_model) -> tuple[bool, dict]:  # fmt: skip
    """(accepted?, the translated item or the rejected row with its reasons) of one row."""
    base = base_source(p.row)
    if p.skip:
        return False, {"id": p.row.get("id"), "base_source": base, "problems": [p.skip],
                       "checks": {"selection": False}}  # fmt: skip
    out = _assemble(p, fr_table)
    segs = list(dict.fromkeys(p.segments()))
    verdict = check_row(p.row, out, rt_table, fr_table, segs, vocabulary, kiwi_table,
                        min_kiwi, min_kiwi_segment)  # fmt: skip
    qe_note = "round-trip chrF + quantity / entity preservation"
    if kiwi_table is not None:
        qe_note += f" + CometKiwi ({qe_model}, {KIWI_LICENCE})"
    out.update(
        id=f"{SOURCE}-{p.row.get('id')}",
        source=SOURCE,
        lang="fr",
        family=p.row.get("family", "unknown"),
        license_use=f"{UPSTREAM.get(base, 'see base source')}; machine translation by "
        f"{mt_model} ({MT_LICENCE})",
        meta={"base_source": base, "base_id": p.row.get("id"), "mt_model": mt_model,
              "qe": qe_note, "qe_model": qe_model if kiwi_table is not None else None,
              "min_kiwi": min_kiwi, "min_kiwi_segment": min_kiwi_segment, **verdict},
    )  # fmt: skip
    if verdict["accepted"]:
        return True, out
    return False, {"id": p.row.get("id"), "base_source": base, **verdict,
                   "state_fr": out["state"][:1500]}  # fmt: skip


def _gate_safe(p: RowPlan, *args) -> tuple[bool, dict]:
    """:func:`_gate`, a row the checks cannot read rejected (never a crashed batch)."""
    try:
        return _gate(p, *args)
    except Exception as e:
        return False, {"id": p.row.get("id"), "base_source": base_source(p.row),
                       "problems": [f"gate error: {type(e).__name__}: {e}"[:200]],
                       "checks": {"gate": False}}  # fmt: skip


def _gate_index(i: int) -> tuple[bool, dict]:
    return _gate_safe(_GATE["plans"][i], *_GATE["args"])


def translate_rows(rows: list[dict], mt: Translate, rt: Translate, *,
                   mt_model: str = MT_MODEL, qe: Score | None = None,
                   min_kiwi: float | None = MIN_KIWI,
                   min_kiwi_segment: float | None = MIN_KIWI_SEGMENT,
                   qe_model: str = KIWI_MODEL) -> tuple[list[dict], list[dict], dict]:  # fmt: skip
    """Translate, round-trip and gate ``rows``. Returns (accepted items, rejected rows with
    their reasons, timings and the segment tables). Segments are translated once each, all
    rows batched together. ``qe`` (e.g. :func:`kiwi_scorer`) scores each segment pair once;
    ``min_kiwi`` gates on the row score (None: logged only)."""
    plans = [plan_row(r) for r in rows]
    live = [p for p in plans if p.skip is None]
    segments = list(dict.fromkeys(s for p in live for s in p.segments()))
    t0 = time.perf_counter()
    fr = mt(segments)
    t1 = time.perf_counter()
    fr_table = dict(zip(segments, fr, strict=True))
    fr_unique = list(dict.fromkeys(fr))
    back = rt(fr_unique)
    t2 = time.perf_counter()
    back_of_fr = dict(zip(fr_unique, back, strict=True))
    rt_table = {s: back_of_fr[fr_table[s]] for s in segments}
    kiwi_table = None
    t3 = time.perf_counter()
    if qe is not None:
        pairs = [(s, fr_table[s]) for s in segments if translatable(s)]
        kiwi_table = dict(zip((p[0] for p in pairs), qe(pairs), strict=True))
    t4 = time.perf_counter()
    vocabulary = lowercase_vocabulary(_flat(p.row) for p in plans)
    args = (fr_table, rt_table, vocabulary, kiwi_table, min_kiwi, min_kiwi_segment, mt_model,
            qe_model)  # fmt: skip
    workers = min(os.cpu_count() or 1, GATE_WORKERS)
    if workers > 1 and len(plans) >= GATE_PARALLEL_MIN:  # the gate is pure Python: fork
        import multiprocessing as mp

        _GATE.update(plans=plans, args=args)
        try:
            with mp.get_context("fork").Pool(workers) as pool:
                results = pool.map(_gate_index, range(len(plans)), chunksize=8)
        finally:
            _GATE.clear()
    else:
        results = [_gate_safe(p, *args) for p in plans]
    accepted = [r for ok, r in results if ok]
    rejected = [r for ok, r in results if not ok]
    t5 = time.perf_counter()
    timings = {
        "gate_seconds": t5 - t4,
        "segments": len(segments),
        "segment_chars": sum(map(len, segments)),
        "mt_seconds": t1 - t0,
        "rt_seconds": t2 - t1,
        "rt_segments": len(fr_unique),
        "rt_chars": sum(map(len, fr_unique)),
        "qe_seconds": t4 - t3,
        "qe_segments": len(kiwi_table or {}),
        "qe_chars": sum(len(k) + len(fr_table[k]) for k in kiwi_table or {}),
        "tables": {"fr": fr_table, "rt": rt_table, **({"kiwi": kiwi_table} if kiwi_table else {})},
    }
    return accepted, rejected, timings


def hf_translator(model_id: str, *, beams: int = 4, batch_size: int = 16,
                  max_new_tokens: int = 512) -> Translate:  # fmt: skip
    """A Marian / seq2seq HF model as a batch translator (CPU), longest-first batches."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).eval()

    def run(texts: list[str]) -> list[str]:
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out: list[str] = [""] * len(texts)
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=512)  # fmt: skip
            with torch.inference_mode():
                gen = model.generate(**enc, num_beams=beams, max_new_tokens=max_new_tokens)
            for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True), strict=True):
                out[i] = t
        return out

    return run


def ct2_translator(model_id: str, *, beams: int = 4, batch_size: int = 32, threads: int = 0,
                   compute_type: str = "int8", cache: str | None = None,
                   device: str = "cpu") -> Translate:  # fmt: skip
    """The same model through CTranslate2 (converted once, int8 by default): several times
    faster than torch on CPU. ``threads`` 0: all cores. ``device="cuda"`` with
    ``compute_type="float16"``: the GPU run of the full corpus."""
    import os
    from pathlib import Path

    import ctranslate2
    from transformers import AutoTokenizer

    root = Path(cache or os.environ.get("TJEV_CT2_CACHE", Path.home() / ".cache" / "tjev-ct2"))
    out = root / f"{model_id.replace('/', '--')}-{compute_type}"
    if not (out / "model.bin").exists():
        ctranslate2.converters.TransformersConverter(model_id).convert(
            str(out), quantization=compute_type, force=True
        )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = ctranslate2.Translator(str(out), device=device, compute_type=compute_type,
                                   inter_threads=1, intra_threads=threads or os.cpu_count())  # fmt: skip

    def run(texts: list[str]) -> list[str]:
        if not texts:
            return []
        tokens = [tok.convert_ids_to_tokens(tok.encode(t, truncation=True, max_length=512))
                  for t in texts]  # fmt: skip
        res = model.translate_batch(tokens, beam_size=beams, max_batch_size=batch_size,
                                    batch_type="examples", max_decoding_length=512)  # fmt: skip
        return [tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True)
                for r in res]  # fmt: skip

    return run


def table_scorer(kiwi_table: dict[str, float]) -> Score:
    """A scorer replaying saved CometKiwi scores (``tables["kiwi"]``)."""
    return lambda pairs: [kiwi_table[src] for src, _ in pairs]


def kiwi_scorer(model_id: str = KIWI_MODEL, *, batch_size: int = 16, gpus: int = 0) -> Score:
    """CometKiwi through ``unbabel-comet`` (the model is gated: needs a Hugging Face token
    with access). Scores are in [0, 1], higher is better."""
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(model_id))

    def run(pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        data = [{"src": src, "mt": mt} for src, mt in pairs]
        out = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False)
        return [float(x) for x in out.scores]

    return run
