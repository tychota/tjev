"""mix-v7.5 cleaning of mix-v7's rows (docs/data.md, mix-v7.5; the v7 data report's
findings, ``reports/data/mix-v7``).

Text fixes, applied to the train rows and to the mix-v7 held-out rows of the v7 sources
(``validation-v7`` / ``calibration-v7`` / ``test/<v7 source>``), so that train and
evaluation read the same text (the mix-v6 held-out files are never touched):

- **HTML entities** (``&amp;``, ``&#39;``…) are unescaped in the prose sources
  (``UNESCAPE_SOURCES``: scruples, drop_num, forecasting, webgpt…). Sources with code
  (helpsteer3_x, the arenas) keep them: ``&lt;`` may be the point there.
- **Python reprs.** An option written as a dict repr (``{'level': 'High', 'definition':
  …}``, ``{'value': None}``) becomes text (``High: …``; ``None`` values dropped); a dict
  or list repr inside a state or instructions becomes JSON.
- **Control characters** are removed.
- **Language tags** of the judge sources follow the state's language by the stop-word
  guess (:func:`tjev.data.lang.guess_lang`) when it is clear and differs from the tag.

Row drops (train only): U+FFFD (encoding damage) and mojibake, dices' one-hot rows (all
``no``: dices' signal is in its raters' shares), and near-duplicate templated states of
``CLUSTER_CAPS`` sources beyond ``cap`` rows per near-duplicate cluster (tempreason_l23:
53% of its states are near-duplicates).
"""

from __future__ import annotations

import ast
import dataclasses
import html
import json
import re
from collections import Counter
from collections.abc import Iterable

import numpy as np

from tjev.data.item import Item, Question
from tjev.data.lang import guess_lang
from tjev.data.near import NEAR_THRESHOLD, MinHasher, band_keys, shingle_hashes

HTML_ENTITY = re.compile(r"&(?:amp|lt|gt|quot|apos|nbsp|#\d{2,5}|#x[0-9a-f]{2,4});", re.IGNORECASE)
PY_REPR = re.compile(r"[{\[]'[A-Za-z_]+'\s*:|\{'[A-Za-z_]+': (?:None|'|\[|\{|\d)")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REPLACEMENT = "�"
# double-encoded UTF-8 (the report's pattern, with Â only before a Latin-1 symbol: "ÂGE"
# is French, "Â°" / "Â\xa0" are mojibake)
MOJIBAKE = re.compile(r"Ã[\u0080-¿©¨ª«®¯§¢]|â€[™œ\u009d˜”“¦]|Â[\u00a0-\u00bf]")
UNESCAPE_SOURCES = frozenset({"scruples", "drop_num", "forecasting", "webgpt", "finqa",
                              "tatqa", "tram", "temporal_nli", "social_chem", "mic",
                              "summarize_fb", "casehold", "ledgar", "tasksource_jev"})  # fmt: skip
RETAG_BLOCKS = frozenset({"judge"})
CLUSTER_CAPS = {"tempreason_l23": 2}
HARD_DROP = frozenset({"dices"})  # one-hot rows dropped: the signal is in the soft shares


def _balanced(text: str, start: int) -> int:
    """The end (exclusive) of the bracketed literal starting at ``start``, or -1."""
    pairs = {"{": "}", "[": "]"}
    stack, quote, i = [], None, start
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c in pairs:
            stack.append(pairs[c])
        elif c in "}]":
            if not stack or stack.pop() != c:
                return -1
            if not stack:
                return i + 1
        i += 1
    return -1


def _literal(text: str):
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, (dict, list)) else None


def repr_to_json(text: str) -> str:
    """Python dict / list reprs inside ``text`` rewritten as JSON (others unchanged)."""
    if not PY_REPR.search(text):
        return text
    out, i = [], 0
    for m in PY_REPR.finditer(text):
        if m.start() < i:
            continue
        end = _balanced(text, m.start())
        value = _literal(text[m.start() : end]) if end > 0 else None
        try:
            dumped = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):  # Ellipsis, bytes, sets, nan: not JSON, left as is
            continue
        if value is None:
            continue
        out.append(text[i : m.start()])
        out.append(dumped)
        i = end
    out.append(text[i:])
    return "".join(out)


def _plain(value) -> str:
    if isinstance(value, dict):
        parts = [_plain(v) for v in value.values() if v is not None and _plain(v)]
        if len(parts) == 2 and len(parts[0]) <= 40:  # {'level': x, 'definition': y}
            return f"{parts[0]}: {parts[1]}"
        return "; ".join(parts)
    if isinstance(value, list):
        return "; ".join(_plain(v) for v in value if v is not None)
    return "" if value is None else str(value)


def repr_to_text(text: str) -> str:
    """An option description that is a dict / list repr, as text; else unchanged."""
    s = text.strip()
    if s[:1] in "{[" and PY_REPR.search(s):
        value = _literal(s)
        if value is not None and "Ellipsis" not in repr(value):
            return _plain(value)
    return repr_to_json(text)


def fix_text(text: str, unescape: bool, reprs: bool = True) -> str:
    text = CONTROL.sub("", text)
    for _ in range(3):  # double-escaped text (&amp;#39;) needs a second pass
        if not (unescape and HTML_ENTITY.search(text)):
            break
        text = html.unescape(text)
    return repr_to_json(text) if reprs else text


def fix_item(item: Item, *, unescape: bool, retag: bool) -> tuple[Item, list[str]]:
    """The text fixes of one row; returns the new row and the fixes applied."""
    q = item.question
    state = fix_text(item.state, unescape)
    instructions = fix_text(q.instructions, unescape)
    descriptions = tuple(repr_to_text(fix_text(d, unescape, reprs=False)) for d in q.descriptions)
    fixes = []
    if state != item.state:
        fixes.append("state")
    if instructions != q.instructions:
        fixes.append("instructions")
    if descriptions != q.descriptions:
        fixes.append("options")
    lang = item.lang
    if retag:
        guess = guess_lang(state)
        if guess in ("en", "fr") and guess != lang:
            lang = guess
            fixes.append(f"lang:{item.lang}->{guess}")
    if not fixes:
        return item, []
    question = Question(q.type, instructions, q.labels, descriptions)
    return dataclasses.replace(item, state=state, question=question, lang=lang), fixes


def damaged(item: Item) -> str | None:
    text = "\n".join([item.state, item.question.instructions, *item.question.descriptions])
    if REPLACEMENT in text:
        return "replacement_char"
    if MOJIBAKE.search(text):
        return "mojibake"
    return None


def cluster_cap(items: list[Item], cap: int, threshold: float = NEAR_THRESHOLD) -> list[bool]:
    """Keep flags: at most ``cap`` rows per cluster of near-duplicate states (MinHash, J ≥
    ``threshold``; clusters by union-find over LSH candidates), the first rows of a
    cluster in file order kept, preferring rows whose question differs."""
    n = len(items)
    if n == 0:
        return []
    hasher = MinHasher()
    sigs = np.stack([hasher.signature(shingle_hashes(it.state)) for it in items])
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keys = band_keys(sigs)
    for band in range(keys.shape[1]):
        k = keys[:, band]
        order = np.argsort(k, kind="stable")
        ks = k[order]
        for s in np.flatnonzero(ks[1:] == ks[:-1]):
            a, b = int(order[s]), int(order[s + 1])
            if (sigs[a] == sigs[b]).mean() >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    kept: Counter = Counter()
    questions: dict[int, set[str]] = {}
    keep = [False] * n
    for second_pass in (False, True):
        for i, it in enumerate(items):
            if keep[i]:
                continue
            root = find(i)
            if kept[root] >= cap:
                continue
            seen = questions.setdefault(root, set())
            if not second_pass and it.question.instructions in seen:
                continue  # a different question first
            keep[i] = True
            kept[root] += 1
            seen.add(it.question.instructions)
    return keep


def clean_rows(
    items: Iterable[Item], source: str, block: str, *, train: bool
) -> tuple[list[Item], Counter]:
    """The mix-v7.5 cleaning of one source's rows: text fixes (train and held-out), drops
    (train only). Returns the rows and counts (``fix:<what>``, ``drop:<why>``)."""
    unescape = source in UNESCAPE_SOURCES
    retag = block in RETAG_BLOCKS
    out, stats = [], Counter()
    for item in items:
        if train:
            why = damaged(item)
            if why is None and source in HARD_DROP and max(item.target) == 1.0:
                why = "one_hot"
            if why is not None:
                stats[f"drop:{why}"] += 1
                continue
        item, fixes = fix_item(item, unescape=unescape, retag=retag)
        for f in fixes:
            stats[f"fix:{f.split(':')[0] if f.startswith('lang') else f}"] += 1
        if any(f.startswith("lang") for f in fixes):
            stats[f"fix:{next(f for f in fixes if f.startswith('lang'))}"] += 1
        out.append(item)
    if train and source in CLUSTER_CAPS:
        keep = cluster_cap(out, CLUSTER_CAPS[source])
        stats["drop:near_cluster_cap"] += keep.count(False)
        out = [it for it, k in zip(out, keep, strict=True) if k]
    return out, stats
