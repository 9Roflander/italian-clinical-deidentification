"""Gold annotation helpers. The gold file stores (type, string) pairs without offsets."""
from __future__ import annotations

import collections
import re

ENTITY_TYPES = ["NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"]
PLACEHOLDERS = {t: f"[{t}]" for t in ENTITY_TYPES}


def canonical_type(raw: str | None) -> str | None:
    value = (raw or "").upper()
    if "LUOGO" in value or "INDIRIZZO" in value:
        return "LUOGO/INDIRIZZO"
    return value if value in ENTITY_TYPES else None


def word_boundary_regex(value: str, flags: int = 0) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)", flags)


def unique_annotations(entities) -> list[tuple[str, str]]:
    """Unique (type, string) pairs, longest string first."""
    seen, out = set(), []
    for e in entities:
        t, x = canonical_type(e.get("type")), e.get("text", "")
        if t and x and (t, x) not in seen:
            seen.add((t, x))
            out.append((t, x))
    return sorted(out, key=lambda p: -len(p[1]))


def audit_note_annotations(text: str, entities) -> list[dict]:
    """One row per unique annotated string, with its status and no string content."""
    counts = collections.Counter((canonical_type(e.get("type")), e.get("text", "")) for e in entities)
    rows = []
    for (t, x), annotated in counts.items():
        if not t or not x:
            rows.append({"type": t, "annotated": annotated, "boundary_occurrences": 0, "status": "invalid_annotation"})
            continue
        sub = text.count(x)
        bnd = len(word_boundary_regex(x).findall(text))
        if sub == 0:
            status = "string_not_in_note"
        elif bnd == 0:
            status = "only_inside_longer_token"
        elif bnd == annotated:
            status = "occurrences_equal_annotations"
        elif bnd > annotated:
            status = "more_occurrences_than_annotations"
        else:
            status = "fewer_occurrences_than_annotations"
        rows.append({"type": t, "annotated": annotated, "boundary_occurrences": bnd, "status": status})
    return rows


def annotations_locatable(text: str, entities) -> bool:
    """True when every annotated string can be found at word boundaries in the note."""
    return all(r["status"] not in ("string_not_in_note", "only_inside_longer_token", "invalid_annotation")
               for r in audit_note_annotations(text, entities))


def mask_annotated(text: str, entities) -> tuple[str, int]:
    """Replace every word-boundary occurrence of each annotated string with its placeholder."""
    out, n = text, 0
    for t, x in unique_annotations(entities):
        out, k = word_boundary_regex(x).subn(PLACEHOLDERS[t], out)
        n += k
    return out, n
