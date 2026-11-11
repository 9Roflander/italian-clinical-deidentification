"""The 12 acceptance criteria of protocol section 12, on the real gold_v2. Output is numeric only, so failures show no text.

`pytest tests/test_evaluation_acceptance.py -v` gives one line per criterion.
"""
import json

import pytest

from src.evaluation.alignment import AMBIGUOUS, align_inline
from src.evaluation.controls import CONTROLS, run_control
from src.evaluation.gold_verification import verify_gold
from src.evaluation.runner import load_config, compare_runs, evaluate_records, evaluator_fingerprint, verify_run, write_run
from src.evaluation.scoring import totals


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["bootstrap"]["iterations"] = 2000
    return c


@pytest.fixture(scope="module")
def gold(cfg):
    return verify_gold(cfg)


@pytest.fixture(scope="module")
def controls(gold):
    return {name: run_control(name, gold) for name in CONTROLS}


def test_criterion_01_strict_precision_recall_f1(gold, controls):
    for name in ("perfect_spans", "every_other_entity_missed", "every_boundary_off_by_one"):
        assert all(controls[name][k] is not None for k in ("micro_f1", "macro_f1", "micro_recall"))
    assert 0 < controls["every_other_entity_missed"]["micro_f1"] < 1


def test_criterion_02_empty_output_gets_zero_f1(controls):
    c = controls["empty_output"]
    assert c["micro_f1"] == 0 and c["macro_f1"] == 0 and c["micro_recall"] == 0 and c["removal_coverage"] == 0 and c["invalid_outputs"] == 82


@pytest.mark.parametrize("name", ["perfect_segments", "perfect_spans", "perfect_inline"])
def test_criterion_03_perfect_predictions_get_f1_of_one(controls, name):
    c = controls[name]
    assert c["micro_f1"] == 1 and c["macro_f1"] == 1 and c["micro_recall"] == 1 and c["invalid_outputs"] == 0
    assert c["removal_coverage"] == 1 and c["over_redaction_rate"] == 0 and c["note_exact_rate"] == 1


def test_criterion_04_wrong_type_and_wrong_boundary(controls):
    assert controls["every_entity_wrong_category"]["micro_f1"] == 0 and controls["every_entity_wrong_category"]["fp_total"] == 252
    assert controls["every_boundary_off_by_one"]["micro_f1"] == 0 and controls["every_boundary_off_by_one"]["fp_total"] == 252


def test_criterion_05_repeated_entities_use_individual_offsets(gold, cfg):
    """Redacting only the first occurrence of repeated strings on real notes earns one true positive per occurrence kept."""
    records, expected_tp = [], 0
    for i in gold.order:
        note = gold.notes[i]
        seen, keep = set(), []
        for s in gold.spans[i]:
            value = note.text[s[0] : s[1]]
            if value not in seen:
                seen.add(value)
                keep.append(s)
        expected_tp += len(keep)
        records.append(CONTROLS["perfect_spans"](gold, i, "repeat") | {"predicted_spans": [list(s) for s in keep]})
    t = totals(evaluate_records(records, gold).scores)
    assert sum(t[f"tp_{c}"] for c in ("NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO")) == expected_tp
    assert expected_tp < 252, "gold_v2 must contain repeated strings for this test to mean anything"


def test_criterion_06_unicode_and_punctuation_on_real_notes(gold):
    """Every real note contains accents or punctuation near entities, and every offset is recovered exactly."""
    accents = sum(any(ord(ch) > 127 for ch in gold.notes[i].text) for i in gold.order)
    assert accents > 0
    records = [CONTROLS["perfect_inline"](gold, i, "u") for i in gold.order]
    result = evaluate_records(records, gold)
    assert all(s.status == "valid" and [(p.start, p.end, p.label) for p in s.predicted_spans] == gold.spans[s.note_id] for s in result.scores)


def test_criterion_07_ambiguity_is_detected_not_guessed(gold):
    """Two placeholders split by one space fit every space in a real note, so no single reading exists."""
    assert all(align_inline(gold.notes[i].text, "[NOME] [DATA]").status == AMBIGUOUS for i in gold.order)


def test_criterion_08_coverage_and_over_redaction(controls):
    assert controls["every_entity_wrong_category"]["removal_coverage"] == 1
    assert 0 < controls["every_boundary_off_by_one"]["removal_coverage"] < 1
    assert 0 < controls["all_numbers_tagged_as_age"]["over_redaction_rate"] < 1
    assert controls["single_placeholder_for_whole_note"]["over_redaction_rate"] > 0.9
    assert controls["unchanged_input"]["over_redaction_rate"] is None


def test_criterion_09_frozen_folds_are_preserved(gold, cfg):
    assert gold.fold_sizes == {1: 16, 2: 17, 3: 17, 4: 16, 5: 16}
    assert all(v for v in gold.fold_file_sha256.values())
    verify_gold(cfg)


def test_criterion_10_results_reproduce_from_saved_predictions(gold, cfg, tmp_path):
    records = [CONTROLS["every_other_entity_missed"](gold, i, "EX") for i in gold.order]
    public, secure = tmp_path / "public", tmp_path / "secure"
    run = write_run(records, gold, cfg, public, secure)
    report = verify_run(run, gold, cfg, secure)
    assert report["reproduced"] and not report["problems"]


def test_criterion_11_confidence_intervals_are_calculated(gold, cfg, tmp_path):
    public, secure = tmp_path / "public", tmp_path / "secure"
    a = write_run([CONTROLS["every_other_entity_missed"](gold, i, "EA") for i in gold.order], gold, cfg, public, secure)
    b = write_run([CONTROLS["perfect_inline"](gold, i, "EB") for i in gold.order], gold, cfg, public, secure)
    boot = json.loads((a / "bootstrap_results.json").read_text(encoding="utf-8"))
    micro = boot["metrics"]["micro_f1"]
    assert micro["lower"] < micro["point"] < micro["upper"] and boot["configuration"]["strata"] == "fold" and boot["configuration"]["seed"] == 42
    compared = compare_runs(a, b, cfg)["metrics"]["micro_f1"]
    assert compared["lower"] > 0 and compared["difference"] > 0


def test_criterion_12_results_link_to_a_manifest_and_evaluator_version(gold, cfg, tmp_path):
    public, secure = tmp_path / "public", tmp_path / "secure"
    run = write_run([CONTROLS["perfect_inline"](gold, i, "EM") for i in gold.order], gold, cfg, public, secure)
    m = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["evaluator"]["fingerprint_sha256"] == evaluator_fingerprint() and m["gold"]["sha256"] == gold.sha256
    assert len(m["manifest_sha256"]) == 64 and len(m["predictions"]["sha256"]) == 64
    assert set(m["folds"]["file_sha256"]) == {"1", "2", "3", "4", "5"}
    assert all(not (p.stat().st_mode & 0o222) for p in run.iterdir())


def test_old_metric_rewards_broken_outputs_that_strict_scoring_rejects(controls):
    """Why the protocol replaced it. These numbers come from the same constructed outputs."""
    for name in ("empty_output", "every_boundary_off_by_one", "unknown_placeholders", "truncated_at_half_unflagged", "single_placeholder_for_whole_note"):
        assert controls[name]["micro_f1"] == 0 and controls[name]["historical_paper_metric"] > 0.9, name
