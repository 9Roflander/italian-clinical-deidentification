"""BIO token labels, sliding windows and span decoding for the encoder baseline. Pure Python; positions and labels only.

Offsets are the tokenizer's character offsets into the original note (half-open, code points), so decoded spans are exact
positions in the note and the text itself is never changed.
"""
from __future__ import annotations

from src.evaluation.alignment import CATEGORIES

LABELS = ("O",) + tuple(f"{p}-{c}" for c in CATEGORIES for p in ("B", "I"))
LABEL_ID = {label: i for i, label in enumerate(LABELS)}
IGNORE = -100


def token_labels(offsets, spans) -> tuple[list[int], dict]:
    """A label per token: B on the first token overlapping a span, I on the following ones, O elsewhere.
    Also counts spans whose edges fall inside a token, which no token labelling can reproduce exactly."""
    labels = [LABEL_ID["O"]] * len(offsets)
    stats = {"spans": 0, "spans_misaligned": 0, "spans_without_tokens": 0}
    for start, end, cat in sorted(spans):
        stats["spans"] += 1
        inside = [i for i, (s, e) in enumerate(offsets) if e > s and s < end and e > start]
        if not inside:
            stats["spans_without_tokens"] += 1
            continue
        if offsets[inside[0]][0] != start or offsets[inside[-1]][1] != end:
            stats["spans_misaligned"] += 1
        labels[inside[0]] = LABEL_ID[f"B-{cat}"]
        for i in inside[1:]:
            labels[i] = LABEL_ID[f"I-{cat}"]
    return labels, stats


def decode(offsets, label_ids) -> list[tuple[int, int, str]]:
    """Spans from token labels. B starts an entity; I continues one of the same type or, without one, starts it; O ends it.
    A span runs from its first token's start to its last token's end."""
    spans, current = [], None
    for (s, e), i in zip(offsets, label_ids):
        if e <= s:
            continue
        label = LABELS[i]
        if label == "O":
            current = None
            continue
        prefix, cat = label.split("-", 1)
        if prefix == "I" and current is not None and current[2] == cat:
            current[1] = e
        else:
            current = [s, e, cat]
            spans.append(current)
    return [tuple(x) for x in spans]


def windows(n_tokens: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Token windows [a, b) of at most `size`, overlapping by `stride`, covering every token."""
    if size <= stride:
        raise ValueError("window size must exceed the overlap")
    if n_tokens <= size:
        return [(0, n_tokens)]
    out, a = [], 0
    while True:
        b = min(a + size, n_tokens)
        out.append((a, b))
        if b == n_tokens:
            return out
        a = b - stride


def merge_windows(n_tokens: int, spans_of_windows, predictions) -> list[int]:
    """One label per token, taken from the window where that token has the most context on both sides."""
    best, score = [LABEL_ID["O"]] * n_tokens, [-1] * n_tokens
    for (a, b), pred in zip(spans_of_windows, predictions):
        for k, i in enumerate(range(a, b)):
            margin = min(k, b - a - 1 - k)
            if margin > score[i]:
                best[i], score[i] = pred[k], margin
    return best
