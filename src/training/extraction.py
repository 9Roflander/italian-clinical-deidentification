"""Extraction output format (experiment E3-X): the model lists the sensitive items, and a fixed locator finds them.

The segments format asks the model to copy the whole note, and a single changed character voids the note. Here the model
copies only the sensitive strings, in the order they appear:

    [{"text": "69", "label": "ETÀ"}, {"text": "Ospedale San Fittizio", "label": "LUOGO/INDIRIZZO"}]

Locator, fixed before any E3-X result exists:
- a cursor starts at 0; each item is searched with plain `str.find` from the cursor;
- found: a sensitive item becomes the span [p, p + len); every item, sensitive or not, moves the cursor to p + len;
- an item with label "O" is a skip: it marks an identical, non-sensitive earlier occurrence, so the next copy is the one
  meant. Targets contain a skip only where the cursor would otherwise land on an unannotated copy;
- not found, or empty text: the item is dropped and counted as unlocated; the cursor does not move;
- labels outside the four categories are kept, and the evaluator scores them as unknown false positives.
Output that is not a JSON array of {"text", "label"} objects is unparseable and scored as an invalid output.

Every training target is read back through the locator and must give exactly its spans.
"""
from __future__ import annotations

import json

from src.evaluation.alignment import CATEGORIES

SKIP = "O"
FORMAT_NAME = "extraction_v1"


class ExtractionTargetError(ValueError):
    """A target cannot be built exactly. Messages hold positions and categories, never text."""


def extraction_items(text: str, spans) -> list[dict]:
    """Items in text order, with a skip before any entity whose string first occurs, from the cursor, at an unannotated place."""
    items, cursor = [], 0
    for start, end, label in sorted(spans):
        if label not in CATEGORIES:
            raise ExtractionTargetError(f"unknown category at {start}-{end}")
        if not 0 <= cursor <= start < end <= len(text):
            raise ExtractionTargetError(f"span {start}-{end} overlaps or is out of range")
        literal = text[start:end]
        p = text.find(literal, cursor)
        while p != start:
            if p == -1 or p > start:
                raise ExtractionTargetError(f"span {start}-{end} cannot be reached by the locator")
            items.append({"text": literal, "label": SKIP})
            cursor = p + len(literal)
            p = text.find(literal, cursor)
        items.append({"text": literal, "label": label})
        cursor = end
    return items


def parse_extraction(raw) -> list[dict] | None:
    """The model's items, or None when the output is not a JSON array of {"text": str, "label": str} objects."""
    try:
        data = json.loads(raw.strip()) if isinstance(raw, str) else None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    items = []
    for x in data:
        if not isinstance(x, dict) or set(x) != {"text", "label"} or not isinstance(x["text"], str) or not isinstance(x["label"], str):
            return None
        items.append({"text": x["text"], "label": x["label"]})
    return items


def locate(text: str, items) -> tuple[list[dict], dict]:
    """Spans as {"start", "end", "label"} plus counts: items, located, unlocated, skips. See the module docstring."""
    spans, cursor = [], 0
    stats = {"items": len(items), "located": 0, "unlocated": 0, "skips": 0}
    for item in items:
        literal, label = item["text"], item["label"]
        p = text.find(literal, cursor) if literal else -1
        if p == -1:
            stats["unlocated"] += 1
            continue
        stats["located"] += 1
        if label == SKIP:
            stats["skips"] += 1
        else:
            spans.append({"start": p, "end": p + len(literal), "label": label})
        cursor = p + len(literal)
    return spans, stats


def serialize_extraction_target(text: str, spans) -> str:
    """Canonical JSON the model is trained to produce, checked to read back to exactly these spans."""
    items = extraction_items(text, spans)
    back, _ = locate(text, items)
    if [(s["start"], s["end"], s["label"]) for s in back] != sorted((a, b, c) for a, b, c in spans):
        raise ExtractionTargetError("locator read-back gave different spans")
    target = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    if parse_extraction(target) != items:
        raise ExtractionTargetError("target does not parse back")
    return target
