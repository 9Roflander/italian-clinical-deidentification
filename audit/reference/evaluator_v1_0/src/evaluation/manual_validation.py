"""Support for the human checks in protocol section 10: second annotation and the blinded output audit."""
from __future__ import annotations

import csv
import json
import pathlib
import random

from src.data.loaders import ROOT
from src.evaluation.alignment import CATEGORIES, Span, resolve_spans
from src.evaluation.metrics import as_report, compute_metrics
from src.evaluation.scoring import score_note, totals

AUDIT_COLUMNS = ("residual_sensitive_information", "incorrect_entity_categories", "missing_redactions", "unnecessary_redactions", "changes_to_clinical_content", "auditor_notes")


def annotator_agreement(gold, second_annotations: dict) -> dict:
    """Exact-span agreement by category. gold_v2 is treated as the reference and the second annotator as the prediction."""
    scores = []
    for note_id, spans in second_annotations.items():
        source = gold.notes[note_id].text
        resolved = resolve_spans(source, spans)
        if not resolved.is_valid:
            raise ValueError(f"{note_id}: second annotation is {resolved.status}")
        scores.append(score_note(note_id, gold.fold_of[note_id], source, gold.spans[note_id], resolved))
    t = totals(scores)
    m = as_report(compute_metrics(t))
    return {"notes_compared": len(scores),
            "by_category": {c: {"tp": t[f"tp_{c}"], "fp": t[f"fp_{c}"], "fn": t[f"fn_{c}"], "precision": m[f"precision_{c}"], "recall": m[f"recall_{c}"], "f1": m[f"f1_{c}"]} for c in CATEGORIES},
            "overall": {"precision": m["micro_precision"], "recall": m["micro_recall"], "f1": m["micro_f1"]},
            "note": "Agreement is reported before adjudication."}


def write_blinded_audit(outputs_by_model: dict, sources: dict, out_dir=None, seed: int = 42) -> pathlib.Path:
    """A shuffled sheet with model names removed, plus a separate key. Holds clinical text, so it stays in data_secure/."""
    out_dir = pathlib.Path(out_dir or ROOT / "data_secure" / "review" / "blinded_output_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [(model, note_id) for model, outputs in outputs_by_model.items() for note_id in outputs]
    random.Random(seed).shuffle(items)
    key = {}
    with open(out_dir / "audit_sheet.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audit_id", "source_text", "model_output", *AUDIT_COLUMNS])
        for n, (model, note_id) in enumerate(items, 1):
            audit_id = f"A{n:04d}"
            key[audit_id] = {"model": model, "note_id": note_id}
            writer.writerow([audit_id, sources[note_id], outputs_by_model[model][note_id], *[""] * len(AUDIT_COLUMNS)])
    (out_dir / "audit_key_do_not_open_before_finishing.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    return out_dir
