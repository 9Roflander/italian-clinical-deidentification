"""Recover predicted spans in source-note character offsets.

Three input formats, all reduced to half-open (start, end, label) spans:
  segments  the preferred format: [{"text": ..., "label": "O" | category}, ...]. Concatenated text must equal the note.
  spans     direct offsets, for encoder models.
  inline    a redacted string, read by a conservative adapter: literal text must match the note exactly and the
            alignment must be unique. There is no fuzzy matching, and failures earn no credit.

Nothing here receives gold annotations, so gold cannot influence how an output is read.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

CATEGORIES = ("NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO")
PLACEHOLDER_OF = {c: f"[{c}]" for c in CATEGORIES}
CATEGORY_OF = {v: k for k, v in PLACEHOLDER_OF.items()}
OUTSIDE = "O"
UNKNOWN = "UNKNOWN"
BRACKET_TOKEN = re.compile(r"\[[^\[\]\n]{1,40}\]")
LABEL_IN_RAW_JSON = re.compile(r'"label"\s*:\s*"([^"]*)"')
LENGTH_REASONS = {"length", "max_tokens", "MAX_TOKENS", "LENGTH"}

VALID = "valid"
EMPTY = "empty_output"
TRUNCATED = "truncated_output"
AMBIGUOUS = "ambiguous_alignment"
NO_ALIGNMENT = "no_exact_alignment"
WHITESPACE_CHANGED = "no_exact_alignment_whitespace_only"
INVALID_JSON = "invalid_segments_json"
SEGMENTS_MISMATCH = "segments_do_not_reproduce_input"
OVERLAPPING = "overlapping_predictions"
SPAN_OUT_OF_RANGE = "span_out_of_range"
MISSING = "missing_prediction"
GENERATION_ERROR = "generation_error"
SOURCE_HAS_PLACEHOLDER = "source_contains_placeholder"
UNKNOWN_FORMAT = "unknown_output_format"
UNALIGNABLE = {AMBIGUOUS, NO_ALIGNMENT, WHITESPACE_CHANGED, SEGMENTS_MISMATCH, TRUNCATED, EMPTY}


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Resolution:
    """How one output was read. `spans` is non-empty only for a valid output.

    `invalid_labels` holds predictions present in an invalid output. They have no position and earn no credit.
    `compared_text` and `gaps` feed the copy-fidelity diagnostic only. They are never used to create spans,
    and they are never written to any report.
    """
    status: str
    spans: tuple[Span, ...] = ()
    invalid_labels: tuple[str, ...] = ()
    literal_text_preserved: bool = False
    compared_text: str | None = None
    gaps: frozenset = frozenset()
    output_normalized: bool = False

    @property
    def is_valid(self) -> bool:
        return self.status == VALID


def normalize_output(source: str, raw: str) -> tuple[str, bool]:
    """The only permitted normalization: NFC, plus outer whitespace when the note itself has none."""
    text = unicodedata.normalize("NFC", raw)
    if source == source.strip():
        text = text.strip()
    return text, text != raw


def _parse_inline(source: str, output: str) -> tuple[list[str], list[str]]:
    """Split into n+1 literals around n placeholders. Bracketed text that occurs in the note stays literal."""
    literals, labels, pos = [], [], 0
    for m in BRACKET_TOKEN.finditer(output):
        token = m.group(0)
        if token in CATEGORY_OF:
            label = CATEGORY_OF[token]
        elif token in source:
            continue
        else:
            label = UNKNOWN
        literals.append(output[pos : m.start()])
        labels.append(label)
        pos = m.end()
    literals.append(output[pos:])
    return literals, labels


def _occurrences(source: str, literal: str) -> list[int]:
    if not literal:
        return list(range(len(source) + 1))
    found, i = [], source.find(literal)
    while i != -1:
        found.append(i)
        i = source.find(literal, i + 1)
    return found


def count_alignments(source: str, literals: list[str], anchor_end: bool = True) -> tuple[int, list[tuple[int, int]] | None]:
    """Count ways to place the literals in order with a non-empty gap for each placeholder.

    Returns (count capped at 2, gaps when the count is exactly 1). With anchor_end=False the last literal may stop
    anywhere, which recognizes an output that is a prefix of a full redaction.
    """
    if not source.startswith(literals[0]):
        return 0, None
    n = len(literals) - 1
    if n == 0 and anchor_end:
        return (1, []) if source == literals[0] else (0, None)
    table = [{0: 1}]
    for i in range(1, n + 1):
        lit = literals[i]
        if i == n and anchor_end:
            starts = [len(source) - len(lit)] if len(lit) <= len(source) and source.endswith(lit) else []
        else:
            starts = _occurrences(source, lit)
        previous = sorted(table[-1].items())
        prev_len = len(literals[i - 1])
        ends = [q + prev_len for q, _ in previous]
        current, accumulated, pointer = {}, 0, 0
        for p in starts:
            while pointer < len(ends) and ends[pointer] < p:   # strict: every gap holds at least one character
                accumulated = min(2, accumulated + previous[pointer][1])
                pointer += 1
            if accumulated:
                current[p] = accumulated
        if not current:
            return 0, None
        table.append(current)
    final = table[n]
    total = min(2, sum(final.values()))
    if total != 1:
        return total, None
    p = next(iter(final))
    gaps = []
    for i in range(n, 0, -1):
        prev_len = len(literals[i - 1])
        q = next(q for q in table[i - 1] if q + prev_len < p)   # exactly one predecessor, because the count is 1
        gaps.append((q + prev_len, p))
        p = q
    return 1, list(reversed(gaps))


def _no_alignment_status(source: str, literals: list[str]) -> str:
    """Name the cause. This is a diagnostic only and never yields spans."""
    if count_alignments(source, literals, anchor_end=False)[0]:
        return TRUNCATED
    squeeze = lambda s: re.sub(r"\s+", "", s)
    if count_alignments(squeeze(source), [squeeze(x) for x in literals])[0]:
        return WHITESPACE_CHANGED
    return NO_ALIGNMENT


def _gap_positions(literals: list[str]) -> frozenset:
    """Where each placeholder sits in the placeholder-free text."""
    positions, total = set(), 0
    for lit in literals[:-1]:
        total += len(lit)
        positions.add(total)
    return frozenset(positions)


def align_inline(source: str, raw_output: str) -> Resolution:
    """Conservative adapter for inline-placeholder strings. Takes no gold annotations."""
    output, normalized = normalize_output(source, raw_output)
    literals, labels = _parse_inline(source, output)
    compared, gaps = "".join(literals), _gap_positions(literals)
    common = {"compared_text": compared, "gaps": gaps, "output_normalized": normalized}
    if any(p in source for p in CATEGORY_OF):
        return Resolution(SOURCE_HAS_PLACEHOLDER, invalid_labels=tuple(labels), **common)
    if not output:
        return Resolution(EMPTY, **common)
    if not labels:
        if output == source:
            return Resolution(VALID, literal_text_preserved=True, **common)
        return Resolution(_no_alignment_status(source, literals), **common)
    count, found = count_alignments(source, literals)
    if count == 0:
        return Resolution(_no_alignment_status(source, literals), invalid_labels=tuple(labels), **common)
    if count > 1:
        return Resolution(AMBIGUOUS, invalid_labels=tuple(labels), literal_text_preserved=True, **common)
    spans = tuple(Span(a, b, label) for (a, b), label in zip(found, labels))
    return Resolution(VALID, spans=spans, literal_text_preserved=True, **common)


def resolve_segments(source: str, raw) -> Resolution:
    """Structured output. Concatenated text must reproduce the note exactly."""
    normalized = False
    if isinstance(raw, str):
        text, normalized = normalize_output(source, raw)
        if not text:
            return Resolution(EMPTY, compared_text="", output_normalized=normalized)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            labels = tuple(x if x in CATEGORIES else UNKNOWN for x in LABEL_IN_RAW_JSON.findall(text) if x != OUTSIDE)
            return Resolution(INVALID_JSON, invalid_labels=labels, output_normalized=normalized)
    else:
        data = raw
    if isinstance(data, dict) and set(data) == {"segments"}:
        data = data["segments"]
    well_formed = isinstance(data, list) and all(
        isinstance(s, dict) and set(s) == {"text", "label"} and isinstance(s["text"], str) and isinstance(s["label"], str) for s in data)
    if not well_formed:
        return Resolution(INVALID_JSON, output_normalized=normalized)
    if not data:
        return Resolution(EMPTY, compared_text="", output_normalized=normalized)
    segments = [{"text": unicodedata.normalize("NFC", s["text"]), "label": s["label"]} for s in data]
    labels = tuple(s["label"] if s["label"] in CATEGORIES else UNKNOWN for s in segments if s["label"] != OUTSIDE)
    joined = "".join(s["text"] for s in segments)
    common = {"compared_text": joined, "output_normalized": normalized}
    if joined != source:
        if joined and source.startswith(joined):
            return Resolution(TRUNCATED, invalid_labels=labels, **common)
        return Resolution(SEGMENTS_MISMATCH, invalid_labels=labels, **common)
    if any(not s["text"] and s["label"] != OUTSIDE for s in segments):
        return Resolution(SEGMENTS_MISMATCH, invalid_labels=labels, **common)
    spans, pos = [], 0
    for s in segments:
        end = pos + len(s["text"])
        if s["label"] != OUTSIDE:
            spans.append(Span(pos, end, s["label"] if s["label"] in CATEGORIES else UNKNOWN))
        pos = end
    return Resolution(VALID, spans=tuple(spans), literal_text_preserved=True, **common)


def resolve_spans(source: str, predicted) -> Resolution:
    """Direct offsets. Exact duplicates stay valid and are scored as false positives. Other overlaps are invalid."""
    if predicted is None:
        return Resolution(MISSING)
    parsed = []
    for item in predicted:
        try:
            start, end, label = (item["start"], item["end"], item["label"]) if isinstance(item, dict) else item
        except (KeyError, TypeError, ValueError):
            return Resolution(SPAN_OUT_OF_RANGE, compared_text=source)
        if type(start) is not int or type(end) is not int or not isinstance(label, str):
            return Resolution(SPAN_OUT_OF_RANGE, compared_text=source)
        parsed.append((start, end, label if label in CATEGORIES else UNKNOWN))
    labels = tuple(x[2] for x in parsed)
    if any(not 0 <= a < b <= len(source) for a, b, _ in parsed):
        return Resolution(SPAN_OUT_OF_RANGE, invalid_labels=labels, literal_text_preserved=True, compared_text=source)
    distinct = sorted(set(parsed))
    if any(x[1] > y[0] for x, y in zip(distinct, distinct[1:])):
        return Resolution(OVERLAPPING, invalid_labels=labels, literal_text_preserved=True, compared_text=source)
    return Resolution(VALID, spans=tuple(Span(*x) for x in sorted(parsed)), literal_text_preserved=True, compared_text=source)


def resolve_prediction(source: str, output_format: str, raw_output=None, predicted_spans=None,
                       completion_status: str = "complete", truncation_flag: bool = False, finish_reason: str | None = None) -> Resolution:
    """Read one model output. A truncated or failed generation is invalid even when its text happens to align."""
    if output_format == "inline":
        resolved = align_inline(source, raw_output if isinstance(raw_output, str) else "")
    elif output_format == "segments":
        resolved = resolve_segments(source, raw_output if raw_output is not None else "")
    elif output_format == "spans":
        resolved = resolve_spans(source, predicted_spans)
    else:
        return Resolution(UNKNOWN_FORMAT)
    labels = resolved.invalid_labels if not resolved.is_valid else tuple(s.label for s in resolved.spans)
    if completion_status in ("error", "failed"):
        return Resolution(GENERATION_ERROR, invalid_labels=labels, compared_text=resolved.compared_text, gaps=resolved.gaps)
    if truncation_flag or completion_status == "truncated" or finish_reason in LENGTH_REASONS:
        return Resolution(TRUNCATED, invalid_labels=labels, compared_text=resolved.compared_text, gaps=resolved.gaps)
    return resolved


def render_inline(source: str, spans) -> str:
    """Deterministic redaction: replace each span with its placeholder."""
    out, pos = [], 0
    for span in sorted(spans):
        start, end, label = span
        if start < pos or end <= start or end > len(source) or label not in CATEGORIES:
            raise ValueError("spans must be in range, non-overlapping and use an official category")
        out += [source[pos:start], PLACEHOLDER_OF[label]]
        pos = end
    return "".join(out) + source[pos:]


def spans_to_segments(source: str, spans) -> list[dict]:
    out, pos = [], 0
    for span in sorted(spans):
        start, end, label = span
        if start > pos:
            out.append({"text": source[pos:start], "label": OUTSIDE})
        out.append({"text": source[start:end], "label": label})
        pos = end
    if pos < len(source):
        out.append({"text": source[pos:], "label": OUTSIDE})
    return out
