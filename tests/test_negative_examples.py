"""Guide test 3: an unchanged-output example must contain no target PII."""
import csv

from src.data.loaders import ROOT
from src.generation.synthetic_validation import validate_negative_example

CLEAN = "Il paziente riferisce dolore toracico e viene sottoposto a controllo cardiologico."


def test_clean_note_is_a_valid_negative():
    assert validate_negative_example(CLEAN, CLEAN, 0) == []


def test_annotated_pii_with_unchanged_output_is_rejected():
    assert "annotated_pii_present" in validate_negative_example(CLEAN, CLEAN, 3)


def test_changed_output_is_not_a_negative():
    assert "output_differs_from_input" in validate_negative_example(CLEAN, CLEAN + " altro", 0)


def test_unannotated_date_and_age_patterns_are_rejected():
    assert "residual_date" in validate_negative_example("Ricovero nel marzo 2015.", "Ricovero nel marzo 2015.", 0)
    assert "residual_age" in validate_negative_example("Paziente di 45 anni.", "Paziente di 45 anni.", 0)


def test_no_real_crf_row_is_permitted_as_a_negative():
    with open(ROOT / "data_manifests" / "negative_audit.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 80
    assert all(r["permitted_for_training"] == "False" for r in rows)
    assert sum(r["reason"] == "duplicate_of_gold_note" for r in rows) == 77
    assert sum(r["reason"] == "near_duplicate_of_gold_note" for r in rows) == 1
    assert sum(r["decision"] == "quarantined" for r in rows) == 2
