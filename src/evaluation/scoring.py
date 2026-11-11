"""Score one note: strict entity match, privacy measures, copy fidelity and an error breakdown.

Strict rule: a prediction is correct only with the exact start, end and category. Matching is one-to-one, so a
duplicate earns no second true positive. An invalid output earns no credit: every gold entity is a false negative,
every prediction present in it is a false positive, and no characters count as redacted.
"""
from __future__ import annotations

import collections
import difflib
from dataclasses import dataclass, field

from src.evaluation.alignment import CATEGORIES, UNKNOWN, Resolution, Span
from src.evaluation.metrics import COUNT_COLUMNS

CORRECT = "correct"
WRONG_CATEGORY = "wrong_category"
WRONG_BOUNDARY = "wrong_boundary"
WRONG_BOTH = "wrong_boundary_and_category"
MISSED = "missed"
INVALID_OUTPUT = "invalid_output"
SPURIOUS = "spurious"
DUPLICATE = "duplicate"
UNKNOWN_PLACEHOLDER = "unknown_placeholder"


@dataclass
class NoteScore:
    note_id: str
    fold: int
    status: str
    counts: dict
    errors: collections.Counter = field(default_factory=collections.Counter)
    n_predictions: int = 0
    output_normalized: bool = False
    predicted_spans: tuple = ()

    def row(self) -> dict:
        return {"note_id": self.note_id, "fold": self.fold, "status": self.status, "n_predictions": self.n_predictions, **self.counts}


def _overlaps(a, b) -> bool:
    return a.start < b.end and b.start < a.end


def _classify_gold(g: Span, predictions) -> str:
    near = [p for p in predictions if _overlaps(g, p)]
    if not near:
        return MISSED
    if any(p.start == g.start and p.end == g.end for p in near):
        return WRONG_CATEGORY
    return WRONG_BOUNDARY if any(p.label == g.label for p in near) else WRONG_BOTH


def _classify_prediction(p: Span, gold) -> str:
    if p.label == UNKNOWN:
        return UNKNOWN_PLACEHOLDER
    if p in gold:
        return DUPLICATE
    near = [g for g in gold if _overlaps(p, g)]
    if not near:
        return SPURIOUS
    if any(g.start == p.start and g.end == p.end for g in near):
        return WRONG_CATEGORY
    return WRONG_BOUNDARY if any(g.label == p.label for g in near) else WRONG_BOTH


def _fidelity(source: str, resolved: Resolution, gold_mask: list[bool]):
    """Inserted, deleted and modified characters, plus non-sensitive characters kept.

    A valid output changed nothing outside its redactions. For an invalid output the counts come from a character diff
    and describe the damage only. The diff is never used to create spans or credit. Returns None counts when unmeasurable.
    """
    non_sensitive = len(source) - sum(gold_mask)
    if resolved.is_valid:
        masked = [False] * len(source)
        for s in resolved.spans:
            for i in range(s.start, s.end):
                masked[i] = True
        kept = sum(1 for i in range(len(source)) if not gold_mask[i] and not masked[i])
        return 0, 0, 0, kept, non_sensitive
    if resolved.compared_text is None:
        return None, None, None, 0, non_sensitive
    matcher = difflib.SequenceMatcher(None, source, resolved.compared_text, autojunk=False)
    inserted = deleted = modified = kept = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            kept += sum(1 for i in range(i1, i2) if not gold_mask[i])
        elif tag == "delete":
            if j1 not in resolved.gaps:        # text under a placeholder is a redaction, not a copying error
                deleted += i2 - i1
        elif tag == "insert":
            inserted += j2 - j1
        else:
            a, b = i2 - i1, j2 - j1
            modified += min(a, b)
            inserted += max(b - a, 0)
            deleted += max(a - b, 0)
    if resolved.literal_text_preserved:
        inserted = deleted = modified = 0
    return inserted, deleted, modified, kept, non_sensitive


def score_note(note_id: str, fold: int, source: str, gold_spans, resolved: Resolution) -> NoteScore:
    gold = {Span(*g) for g in gold_spans}
    counts = dict.fromkeys(COUNT_COLUMNS, 0)
    errors = collections.Counter()
    predictions = list(resolved.spans) if resolved.is_valid else []
    invalid_labels = list(resolved.invalid_labels) if not resolved.is_valid else []

    unmatched = set(gold)
    correct = []
    for p in sorted(predictions):
        if p.label == UNKNOWN:
            counts["fp_unknown"] += 1
            errors[(UNKNOWN, "prediction", UNKNOWN_PLACEHOLDER)] += 1
        elif p in unmatched:
            unmatched.remove(p)
            correct.append(p)
            counts[f"tp_{p.label}"] += 1
            errors[(p.label, "gold", CORRECT)] += 1
        else:
            counts[f"fp_{p.label}"] += 1
            errors[(p.label, "prediction", _classify_prediction(p, gold))] += 1
    for g in sorted(unmatched):
        counts[f"fn_{g.label}"] += 1
        kind = _classify_gold(g, predictions) if resolved.is_valid else INVALID_OUTPUT
        errors[(g.label, "gold", kind)] += 1
    for label in invalid_labels:
        counts[f"fp_{label}" if label in CATEGORIES else "fp_unknown"] += 1
        errors[(label if label in CATEGORIES else UNKNOWN, "prediction", INVALID_OUTPUT)] += 1

    gold_mask = [False] * len(source)
    for g in gold:
        for i in range(g.start, g.end):
            gold_mask[i] = True
    predicted_chars = set()
    for p in predictions:
        predicted_chars.update(range(p.start, p.end))
    counts["gold_chars"] = sum(gold_mask)
    counts["covered_chars"] = sum(1 for i in predicted_chars if gold_mask[i])
    counts["predicted_chars"] = len(predicted_chars)
    counts["over_redacted_chars"] = counts["predicted_chars"] - counts["covered_chars"]
    counts["entities"] = len(gold)
    counts["entities_complete"] = sum(1 for g in gold if all(i in predicted_chars for i in range(g.start, g.end)))

    inserted, deleted, modified, kept, non_sensitive = _fidelity(source, resolved, gold_mask)
    counts["fidelity_unmeasurable"] = int(inserted is None)
    counts["inserted_chars"], counts["deleted_chars"], counts["modified_chars"] = inserted or 0, deleted or 0, modified or 0
    counts["non_sensitive_chars"], counts["non_sensitive_preserved"] = non_sensitive, kept
    counts["notes"] = 1
    counts["notes_invalid"] = int(not resolved.is_valid)
    counts["notes_text_preserved"] = int(resolved.literal_text_preserved)
    counts["notes_exact"] = int(resolved.is_valid and not unmatched and len(predictions) == len(gold) and len(correct) == len(gold))
    return NoteScore(note_id, fold, resolved.status, counts, errors, len(predictions) + len(invalid_labels),
                     resolved.output_normalized, tuple(predictions))


def totals(scores) -> dict:
    out = dict.fromkeys(COUNT_COLUMNS, 0)
    for s in scores:
        for k in COUNT_COLUMNS:
            out[k] += s.counts[k]
    return out
