"""A cheap English / French guess from stop words (no model, no dependency).

Used by the data report (lang-tag mismatches) and by the server, whose wire format carries
no language: a calibration fitted per (type, language) applies the detected language's
temperature, and the per-type one when the guess is ``None`` (too short, mixed, code).
"""

from __future__ import annotations

import re

EN_WORDS = frozenset(
    [
        "the",
        "and",
        "is",
        "are",
        "was",
        "were",
        "of",
        "to",
        "that",
        "this",
        "it",
        "for",
        "you",
        "with",
        "have",
        "has",
        "not",
        "what",
        "which",
        "would",
        "could",
        "should",
        "there",
        "their",
        "they",
        "from",
        "be",
        "been",
        "my",
        "your",
        "i",
        "he",
        "she",
        "we",
        "our",
    ]
)
FR_WORDS = frozenset(
    [
        "le",
        "la",
        "les",
        "et",
        "est",
        "sont",
        "de",
        "des",
        "du",
        "un",
        "une",
        "que",
        "qui",
        "pas",
        "pour",
        "dans",
        "je",
        "vous",
        "il",
        "elle",
        "nous",
        "ce",
        "cette",
        "avec",
        "sur",
        "au",
        "aux",
        "ne",
        "mon",
        "ma",
        "mes",
        "ou",
        "mais",
        "très",
        "été",
        "être",
    ]
)
_WORD = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœæ']+")


def guess_lang(text: str, min_hits: int = 6, ratio: float = 4.0) -> str | None:
    """'en' / 'fr' when stop words clearly favour one language, else None (too short,
    mixed, code, numbers)."""
    words = [w.strip("'") for w in _WORD.findall(text.lower())]
    en = sum(w in EN_WORDS for w in words)
    fr = sum(w in FR_WORDS for w in words)
    if en + fr < min_hits:
        return None
    if fr >= ratio * max(en, 0.5):
        return "fr"
    if en >= ratio * max(fr, 0.5):
        return "en"
    return None


def item_text(item) -> str:
    """The text a guess is made on: state, instructions and option descriptions."""
    q = item.question
    return " ".join([item.state, q.instructions, *q.descriptions])
