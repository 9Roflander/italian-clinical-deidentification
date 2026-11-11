"""Structured segment targets built from entity spans.

Offsets are half-open Python code-point positions in NFC text, the convention the evaluator freezes. Every target is read
back through the evaluator's own segment reader, so a target that the evaluator would not score exactly can never be used.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata

from src.evaluation.alignment import CATEGORIES, OUTSIDE, Span, VALID, resolve_segments


class TargetError(ValueError):
    """A note or its spans cannot form a valid target. Messages hold positions and categories, never text."""


def _normalize_span(span) -> tuple[int, int, str]:
    try:
        start, end, label = (span["start"], span["end"], span["type"]) if isinstance(span, dict) else span
    except (KeyError, TypeError, ValueError) as e:
        raise TargetError("span must be (start, end, category) or a dict with start, end and type") from e
    if type(start) is not int or type(end) is not int:
        raise TargetError("span offsets must be integers")
    if label not in CATEGORIES:
        raise TargetError(f"unknown category at {start}-{end}")
    return start, end, label


def build_segments(text: str, spans) -> list[dict]:
    """Ordinary-text and entity segments that concatenate exactly to `text`. Empty segments are never produced."""
    if not isinstance(text, str) or not text:
        raise TargetError("note text must be a non-empty string")
    if not unicodedata.is_normalized("NFC", text):
        raise TargetError("note text must be NFC, because the evaluator reads model output as NFC")
    ordered = sorted(_normalize_span(s) for s in spans)
    segments, pos = [], 0
    for start, end, label in ordered:
        if not 0 <= start < end <= len(text):
            raise TargetError(f"span {start}-{end} is out of range or empty")
        if start < pos:
            raise TargetError(f"span {start}-{end} overlaps the previous span")
        if text[start:end] != text[start:end].strip():
            raise TargetError(f"span {start}-{end} has outer whitespace")
        if start > pos:
            segments.append({"text": text[pos:start], "label": OUTSIDE})
        segments.append({"text": text[start:end], "label": label})
        pos = end
    if pos < len(text):
        segments.append({"text": text[pos:], "label": OUTSIDE})
    validate_segments(text, segments, ordered)
    return segments


def validate_segments(text: str, segments, expected_spans=None) -> None:
    """Rejects anything malformed. Exact reconstruction and evaluator read-back are both required."""
    if not isinstance(segments, list) or not segments:
        raise TargetError("segments must be a non-empty list")
    for s in segments:
        if not isinstance(s, dict) or set(s) != {"text", "label"} or not isinstance(s["text"], str) or not isinstance(s["label"], str):
            raise TargetError("each segment needs exactly text and label")
        if not s["text"]:
            raise TargetError("empty segment")
        if s["label"] != OUTSIDE and s["label"] not in CATEGORIES:
            raise TargetError("unknown segment label")
    if "".join(s["text"] for s in segments) != text:
        raise TargetError("segments do not reconstruct the note")
    resolved = resolve_segments(text, serialize_target(segments))
    if resolved.status != VALID or not resolved.literal_text_preserved:
        raise TargetError(f"evaluator read-back failed: {resolved.status}")
    if expected_spans is not None and list(resolved.spans) != [Span(*s) for s in sorted(expected_spans)]:
        raise TargetError("evaluator read-back gave different spans")


def serialize_target(segments) -> str:
    """Canonical JSON: the exact string a model is trained to produce."""
    return json.dumps(segments, ensure_ascii=False, separators=(",", ":"))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
