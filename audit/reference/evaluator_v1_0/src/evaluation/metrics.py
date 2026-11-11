"""Metric equations, shared by point estimates and bootstrap replicates so the two cannot disagree.

Undefined values stay undefined. P, R and F1 are NaN (None in reports) when their denominator is zero. They are never
set to 1.0, so an empty output cannot look perfect. When gold or predictions exist and TP = 0, F1 is 0.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.alignment import CATEGORIES

COUNT_COLUMNS = tuple(f"{k}_{c}" for k in ("tp", "fp", "fn") for c in CATEGORIES) + (
    "fp_unknown", "gold_chars", "covered_chars", "predicted_chars", "over_redacted_chars",
    "entities", "entities_complete", "notes", "notes_exact", "notes_invalid", "notes_text_preserved",
    "inserted_chars", "deleted_chars", "modified_chars", "fidelity_unmeasurable",
    "non_sensitive_chars", "non_sensitive_preserved")


def _ratio(numerator, denominator):
    a, b = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b > 0)


def _f1(tp, fp, fn):
    return _ratio(2 * np.asarray(tp, dtype=float), 2 * np.asarray(tp, dtype=float) + fp + fn)


def compute_metrics(t: dict) -> dict:
    """Works on scalars or arrays of totals. An unknown placeholder has no category, so it lowers micro only."""
    out = {}
    per_category = []
    for c in CATEGORIES:
        tp, fp, fn = t[f"tp_{c}"], t[f"fp_{c}"], t[f"fn_{c}"]
        out[f"precision_{c}"] = _ratio(tp, np.asarray(tp) + fp)
        out[f"recall_{c}"] = _ratio(tp, np.asarray(tp) + fn)
        out[f"f1_{c}"] = _f1(tp, fp, fn)
        per_category.append(out[f"f1_{c}"])
    tp = sum(t[f"tp_{c}"] for c in CATEGORIES)
    fp = sum(t[f"fp_{c}"] for c in CATEGORIES) + t["fp_unknown"]
    fn = sum(t[f"fn_{c}"] for c in CATEGORIES)
    out["micro_precision"] = _ratio(tp, np.asarray(tp) + fp)
    out["micro_recall"] = _ratio(tp, np.asarray(tp) + fn)
    out["micro_f1"] = _f1(tp, fp, fn)
    stacked = np.stack([np.atleast_1d(x) for x in per_category])
    defined = np.sum(~np.isnan(stacked), axis=0)
    macro = np.divide(np.nansum(stacked, axis=0), defined, out=np.full(defined.shape, np.nan, dtype=float), where=defined > 0)
    out["macro_f1"] = macro if np.ndim(per_category[0]) else macro[0]
    out["removal_coverage"] = _ratio(t["covered_chars"], t["gold_chars"])
    out["over_redaction_rate"] = _ratio(t["over_redacted_chars"], t["predicted_chars"])
    out["complete_entity_recall"] = _ratio(t["entities_complete"], t["entities"])
    out["non_sensitive_preservation"] = _ratio(t["non_sensitive_preserved"], t["non_sensitive_chars"])
    out["note_exact_rate"] = _ratio(t["notes_exact"], t["notes"])
    out["invalid_output_rate"] = _ratio(t["notes_invalid"], t["notes"])
    out["text_preserved_rate"] = _ratio(t["notes_text_preserved"], t["notes"])
    return out


def as_report(metrics: dict) -> dict:
    """Plain numbers for JSON and CSV, with undefined values as None."""
    report = {}
    for k, v in metrics.items():
        x = float(v)
        report[k] = None if np.isnan(x) else x
    return report
