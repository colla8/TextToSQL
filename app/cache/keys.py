# -*- coding: utf-8 -*-
"""
Cache key derivation.

normalize_question() exists because the SQL/retrieval caches are keyed by
question text, and this project's questions come from Persian-speaking
Telegram users who don't type consistently: extra whitespace, ZWNJ
(half-space) vs a plain space, Arabic vs Persian letterforms for the same
sound (ي/ی, ك/ک), Arabic-Indic/Persian/Western digits, and stray trailing
punctuation. Without normalizing first, "چند تا سفارش داریم؟" and
"چند تا سفارش داریم ؟ " would hash to two different cache entries for a
question that's identical in every way that matters here.

This is intentionally simple (no hazm dependency) -- it only needs to make
near-duplicate questions collide, not do linguistically correct
normalization.
"""

import hashlib
import re
import unicodedata

# Arabic -> Persian letterform normalization. Visually near-identical but
# distinct code points; Persian keyboards/typing habits mix both freely.
_CHAR_MAP = {
    "\u064A": "\u06CC",  # ARABIC YEH -> PERSIAN YEH (ي -> ی)
    "\u0643": "\u06A9",  # ARABIC KAF -> PERSIAN KEHEH (ك -> ک)
    "\u0629": "\u0647",  # ARABIC TEH MARBUTA -> HEH (ة -> ه)
    "\u06C0": "\u0647",  # HEH WITH YEH ABOVE -> HEH
    "\u0649": "\u06CC",  # ARABIC ALEF MAKSURA -> PERSIAN YEH
}

# Persian + Arabic-Indic digits -> ASCII, so "۱۲" / "١٢" / "12" all hash
# the same.
_DIGIT_MAP: dict[str, str] = {}
for _i, _ch in enumerate("۰۱۲۳۴۵۶۷۸۹"):
    _DIGIT_MAP[_ch] = str(_i)
for _i, _ch in enumerate("٠١٢٣٤٥٦٧٨٩"):
    _DIGIT_MAP[_ch] = str(_i)

_TRANSLATE = str.maketrans({**_CHAR_MAP, **_DIGIT_MAP})

# Zero-width characters (ZWNJ/half-space, ZWJ, BOM) meaningful for
# rendering but not for treating two questions as "the same question".
_ZERO_WIDTH_RE = re.compile("[\u200c\u200d\ufeff]")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[؟?!.,،؛;:\"'`]+")


def normalize_question(question: str) -> str:
    text = unicodedata.normalize("NFKC", question)
    text = text.translate(_TRANSLATE)
    text = _ZERO_WIDTH_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return text


def hash_text(text: str) -> str:
    """Stable, fixed-length cache key. sha256, not Python's built-in
    hash() -- the latter is salted per-process by default (PYTHONHASHSEED),
    so keys would silently miss on every restart."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def question_cache_key(question: str) -> str:
    return hash_text(normalize_question(question))


def sql_cache_key(sql: str) -> str:
    """SQL isn't Persian text -- just collapse whitespace before hashing so
    two queries differing only in formatting hit the same cache entry."""
    normalized = _WHITESPACE_RE.sub(" ", sql).strip()
    return hash_text(normalized)
