"""Text normalisation used only for comparison. Original text is never modified."""
from __future__ import annotations

import hashlib
import re
import unicodedata

_WORD = re.compile(r"\w+", flags=re.UNICODE)

# Graded levels, weakest first, so a report can say how strong a match is.
LEVELS = {
    "byte_identical": lambda s: s,
    "whitespace_trimmed": str.strip,
    "whitespace_normalised": lambda s: " ".join(s.split()),
    "unicode_case_whitespace": lambda s: " ".join(unicodedata.normalize("NFKC", s).casefold().split()),
    "token_level_punct_insensitive": lambda s: " ".join(tokens(s)),
}


def tokens(text: str) -> list[str]:
    """Unicode-, case- and punctuation-insensitive word tokens."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def word_ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def doc_hash(text: str) -> str:
    """SHA-256 of the fully normalised document, safe to store in reports."""
    return hashlib.sha256(" ".join(tokens(text)).encode("utf-8")).hexdigest()


def strongest_match_level(a: str, b: str) -> str | None:
    """Return the first (strictest) level at which two documents are equal."""
    for name, fn in LEVELS.items():
        if fn(a) == fn(b):
            return name
    return None
