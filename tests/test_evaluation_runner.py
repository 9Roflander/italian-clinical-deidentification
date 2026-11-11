"""Run-level rules: required files, out-of-fold checks, immutability, reproducibility, no text in public files."""
import ast
import csv
import json
import pathlib

import pytest

from src.data.loaders import GoldNote
from src.data.splits import LeakageError
from src.evaluation import manual_validation
from src.evaluation.alignment import MISSING, render_inline
from src.evaluation.gold_verification import VerifiedGold
from src.evaluation.runner import (GENERATED_FILES, EvaluationInputError, RunExistsError, compare_runs, evaluate_records, evaluator_fingerprint,
                                   load_config, read_predictions, text_sha256, verify_run, write_run)
from src.evaluation.scoring import totals

SENTINEL = "ZZNOTETEXT"


@pytest.fixture()
def gold():
    """Ten invented notes in five folds. The text carries a sentinel to catch leaks into public files."""
    notes, spans, fold_of = {}, {}, {}
    for k in range(10):
        name, age = f"Zelinda{chr(97 + k)}", str(30 + k)
        text = f"{SENTINEL} {k}: {name} ha {age} anni, trasferita a Borgoverde."
        s = sorted([(text.index(name), text.index(name) + len(name), "NOME"), (text.index(f"ha {age}") + 3, text.index(f"ha {age}") + 5, "ETÀ"),
                    (text.index("Borgoverde"), text.index("Borgoverde") + 10, "LUOGO/INDIRIZZO")])
        i = f"mini_{k:02d}"
        ents = tuple({"start": a, "end": b, "type": c, "text": text[a:b], "annotated_text": text[a:b]} for a, b, c in s)
        notes[i] = GoldNote(i, k, text, ents, redacted_text=render_inline(text, s))
        spans[i] = s
        fold_of[i] = k % 5 + 1
    return VerifiedGold(notes, tuple(notes), fold_of, spans, "mini-sha", "v2", {k: "x" for k in range(1, 6)})


@pytest.fixture()
def cfg():
    c = load_config()
    c["bootstrap"]["iterations"] = 2000
    return c


def record(gold, i, output, **extra):
    r = {"experiment": "T1", "model": "m", "model_revision": "r1", "checkpoint": "c1", "config_id": "cfg", "seed": 1, "note_id": i, "fold_id": gold.fold_of[i],
         "input_sha256": text_sha256(gold.notes[i].text), "output_format": "inline", "raw_model_output": output, "generation_parameters": {"do_sample": False},
         "completion_status": "complete", "finish_reason": "stop", "truncation_flag": False, "prompt_tokens": 10, "generated_tokens": 10}
    r.update(extra)
    return r


def perfect(gold, **extra):
    return [record(gold, i, gold.notes[i].redacted_text, **extra) for i in gold.order]


def public_paths(tmp_path):
    return tmp_path / "public", tmp_path / "secure"


def test_all_required_files_are_written(gold, cfg, tmp_path):
    public, secure = public_paths(tmp_path)
    run = write_run(perfect(gold), gold, cfg, public, secure)
    required = ["run_manifest.json", "alignment_report.json", "per_note_metrics.csv", "per_fold_metrics.csv", "per_category_metrics.csv",
                "pooled_metrics.json", "error_breakdown.csv", "bootstrap_results.json", "evaluation_summary.md"]
    assert sorted(p.name for p in run.iterdir()) == sorted(required)
    assert (secure / run.name / "predictions_oof.jsonl").exists()


def test_summary_contains_every_required_item(gold, cfg, tmp_path):
    public, secure = public_paths(tmp_path)
    text = (write_run(perfect(gold), gold, cfg, public, secure) / "evaluation_summary.md").read_text(encoding="utf-8")
    for needed in ("Gold dataset", "Notes and entities", "Fold assignments", "Model", "Checkpoint", "Random seed", "Strict micro-F1", "Macro-F1", "Precision", "Recall",
                   "F1 [interval]", "Invalid-output rate", "Output-copy fidelity", "Removal coverage", "Over-redaction rate", "Known limitations"):
        assert needed in text, needed


def test_public_files_hold_no_note_text_while_the_secure_file_does(gold, cfg, tmp_path):
    public, secure = public_paths(tmp_path)
    run = write_run(perfect(gold), gold, cfg, public, secure)
    for p in run.iterdir():
        content = p.read_text(encoding="utf-8")
        assert SENTINEL not in content and "Zelinda" not in content and "Borgoverde" not in content, p.name
    assert SENTINEL in (secure / run.name / "predictions_oof.jsonl").read_text(encoding="utf-8")


def test_a_run_cannot_be_overwritten(gold, cfg, tmp_path):
    public, secure = public_paths(tmp_path)
    write_run(perfect(gold), gold, cfg, public, secure)
    with pytest.raises(RunExistsError):
        write_run(perfect(gold), gold, cfg, public, secure)


def test_run_files_are_read_only(gold, cfg, tmp_path):
    public, secure = public_paths(tmp_path)
    run = write_run(perfect(gold), gold, cfg, public, secure)
    assert all(not (p.stat().st_mode & 0o222) for p in run.iterdir())


def test_results_reproduce_from_saved_predictions(gold, cfg, tmp_path):
    public, secure = public_paths(tmp_path)
    run = write_run(perfect(gold), gold, cfg, public, secure)
    report = verify_run(run, gold, cfg, secure)
    assert report["reproduced"] and not report["problems"] and not report["evaluator_changed_since_run"]


@pytest.mark.parametrize("target", ["pooled_metrics.json", "per_note_metrics.csv", "bootstrap_results.json"])
def test_verification_detects_a_changed_result(gold, cfg, tmp_path, target):
    public, secure = public_paths(tmp_path)
    run = write_run(perfect(gold), gold, cfg, public, secure)
    path = run / target
    path.chmod(0o644)
    path.write_text(path.read_text(encoding="utf-8").replace("1", "0", 1), encoding="utf-8")
    report = verify_run(run, gold, cfg, secure)
    assert not report["reproduced"] and any(target in x for x in report["problems"])


def test_verification_detects_changed_predictions(gold, cfg, tmp_path):
    public, secure = public_paths(tmp_path)
    run = write_run(perfect(gold), gold, cfg, public, secure)
    path = secure / run.name / "predictions_oof.jsonl"
    path.chmod(0o644)
    rows = read_predictions(path)
    rows[0]["raw_model_output"] = gold.notes[rows[0]["note_id"]].text
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    report = verify_run(run, gold, cfg, secure)
    assert not report["reproduced"] and "predictions_oof.jsonl was changed" in report["problems"]


def test_manifest_links_results_to_gold_evaluator_and_predictions(gold, cfg, tmp_path):
    public, secure = public_paths(tmp_path)
    run = write_run(perfect(gold), gold, cfg, public, secure)
    m = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["evaluator"]["fingerprint_sha256"] == evaluator_fingerprint() and m["evaluator"]["version"]
    assert "git_commit" in m["evaluator"] and "git_dirty" in m["evaluator"]
    assert m["gold"]["sha256"] == "mini-sha" and m["predictions"]["records"] == 10 and len(m["predictions"]["sha256"]) == 64
    assert set(m["files"]) == set(GENERATED_FILES) and m["seed"] == 1 and m["checkpoint"] == "c1" and m["bootstrap"]["iterations"] == 2000


def test_clean_tree_can_be_required(gold, cfg, tmp_path, monkeypatch):
    public, secure = public_paths(tmp_path)
    monkeypatch.setattr("src.evaluation.runner.git_state", lambda: {"commit": "abc", "dirty": True})
    with pytest.raises(EvaluationInputError, match="uncommitted"):
        write_run(perfect(gold), gold, cfg, public, secure, require_clean_tree=True)


def test_a_missing_note_is_counted_as_invalid_not_dropped(gold):
    rows = perfect(gold)[1:]
    result = evaluate_records(rows, gold)
    assert len(result.scores) == 10 and result.scores[0].status == MISSING
    t = totals(result.scores)
    assert t["notes"] == 10 and t["notes_invalid"] == 1 and t["tp_NOME"] == 9 and t["fn_NOME"] == 1 and t["fidelity_unmeasurable"] == 1


def test_a_prediction_from_the_wrong_fold_model_is_refused(gold):
    rows = perfect(gold)
    rows[0]["fold_id"] = rows[0]["fold_id"] % 5 + 1
    with pytest.raises(LeakageError):
        evaluate_records(rows, gold)


def test_two_predictions_for_one_note_are_refused(gold):
    rows = perfect(gold)
    with pytest.raises(EvaluationInputError, match="more than one"):
        evaluate_records(rows + [rows[0]], gold)


def test_a_prediction_on_different_text_is_refused(gold):
    rows = perfect(gold)
    rows[0]["input_sha256"] = text_sha256("other")
    with pytest.raises(EvaluationInputError, match="different input"):
        evaluate_records(rows, gold)


def test_mixing_models_or_seeds_in_one_file_is_refused(gold):
    rows = perfect(gold)
    rows[1]["seed"] = 2
    with pytest.raises(EvaluationInputError, match="one model"):
        evaluate_records(rows, gold)


def test_unknown_note_and_missing_fields_are_refused(gold):
    rows = perfect(gold)
    rows[0]["note_id"] = "other"
    with pytest.raises(EvaluationInputError, match="not a gold note"):
        evaluate_records(rows, gold)
    with pytest.raises(EvaluationInputError, match="lacks"):
        evaluate_records([{"note_id": "mini_00"}], gold)


def test_pooled_differs_from_fold_mean_and_both_are_reported(gold, cfg, tmp_path):
    """Folds of unequal entity counts: pooled micro and the mean of fold micro must differ, and each stays labelled."""
    rows = [record(gold, i, gold.notes[i].redacted_text if gold.fold_of[i] != 1 else gold.notes[i].text) for i in gold.order]
    public, secure = public_paths(tmp_path)
    run = write_run(rows, gold, cfg, public, secure)
    fold_rows = {r["scope"]: r for r in csv.DictReader(open(run / "per_fold_metrics.csv", encoding="utf-8"))}
    assert {"fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "mean_of_folds", "sd_of_folds"} == set(fold_rows)
    assert float(fold_rows["fold_1"]["micro_f1"]) == 0 and float(fold_rows["fold_2"]["micro_f1"]) == 1
    assert float(fold_rows["mean_of_folds"]["micro_f1"]) == pytest.approx(0.8)
    pooled = json.loads((run / "pooled_metrics.json").read_text(encoding="utf-8"))
    assert pooled["metrics"]["micro_f1"] == pytest.approx(2 * 0.8 / 1.8)
    assert "arithmetic mean" in pooled["distinction"]


def test_per_category_file_has_pooled_fold_and_summary_rows(gold, cfg, tmp_path):
    public, secure = public_paths(tmp_path)
    run = write_run(perfect(gold), gold, cfg, public, secure)
    rows = list(csv.DictReader(open(run / "per_category_metrics.csv", encoding="utf-8")))
    assert {r["scope"] for r in rows} == {"pooled", "fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "mean_of_folds", "sd_of_folds"}
    assert {r["category"] for r in rows} == {"NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"}


def test_error_breakdown_accounts_for_every_entity_and_false_positive(gold, cfg, tmp_path):
    rows = [record(gold, i, gold.notes[i].text if n % 2 else gold.notes[i].redacted_text.replace("[ETÀ]", "[NOME]")) for n, i in enumerate(gold.order)]
    public, secure = public_paths(tmp_path)
    run = write_run(rows, gold, cfg, public, secure)
    breakdown = list(csv.DictReader(open(run / "error_breakdown.csv", encoding="utf-8")))
    pooled = json.loads((run / "pooled_metrics.json").read_text(encoding="utf-8"))["counts"]
    for c, n in pooled.items():
        gold_side = sum(int(r["count"]) for r in breakdown if r["category"] == c and r["side"] == "gold")
        fp_side = sum(int(r["count"]) for r in breakdown if r["category"] == c and r["side"] == "prediction")
        assert gold_side == n["tp"] + n["fn"] and fp_side == n["fp"]


def test_alignment_report_counts_invalid_outputs(gold, cfg, tmp_path):
    rows = perfect(gold)
    rows[0]["raw_model_output"] = ""
    rows[1]["raw_model_output"] = "Testo diverso [NOME]"
    rows[2]["truncation_flag"] = True
    public, secure = public_paths(tmp_path)
    run = write_run(rows, gold, cfg, public, secure)
    report = json.loads((run / "alignment_report.json").read_text(encoding="utf-8"))
    assert report["invalid_outputs"] == 3 and report["valid_outputs"] == 7 and report["outputs_not_alignable_unambiguously"] == 3
    assert report["by_status"]["empty_output"] == 1 and report["by_status"]["truncated_output"] == 1
    assert report["invalid_note_ids"] == ["mini_00", "mini_01", "mini_02"]


def test_comparison_uses_the_same_documents_and_gives_b_minus_a(gold, cfg, tmp_path):
    public, secure = public_paths(tmp_path)
    a = write_run(perfect(gold), gold, cfg, public, secure)
    worse = [record(gold, i, gold.notes[i].text, experiment="T2") for i in gold.order]
    b = write_run(worse, gold, cfg, public, secure)
    report = compare_runs(a, b, cfg)
    micro = report["metrics"]["micro_f1"]
    assert micro["a"] == 1 and micro["b"] == 0 and micro["difference"] == -1 and micro["lower"] == micro["upper"] == -1
    assert report["difference"] == "b minus a" and report["configuration"]["strata"] == "fold"


def test_comparison_refuses_different_seeds(gold, cfg, tmp_path):
    public, secure = public_paths(tmp_path)
    a = write_run(perfect(gold), gold, cfg, public, secure)
    b = write_run(perfect(gold, seed=2), gold, cfg, public, secure)
    with pytest.raises(EvaluationInputError, match="same training seed"):
        compare_runs(a, b, cfg)


def test_evaluator_is_independent_of_generation_and_training():
    banned = ("src.generation", "google", "torch", "transformers", "peft", "trl")
    for path in (pathlib.Path(__file__).parents[1] / "src" / "evaluation").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.startswith(banned) for n in names), (path.name, names)


def test_agreement_uses_exact_spans_by_category(gold):
    second = {"mini_00": [list(s) for s in gold.spans["mini_00"]],
              "mini_01": [list(gold.spans["mini_01"][0]), [gold.spans["mini_01"][1][0], gold.spans["mini_01"][1][1] - 1, gold.spans["mini_01"][1][2]]]}
    out = manual_validation.annotator_agreement(gold, second)
    assert out["notes_compared"] == 2
    total = out["by_category"]
    assert sum(v["tp"] for v in total.values()) == 4 and sum(v["fn"] for v in total.values()) == 2 and sum(v["fp"] for v in total.values()) == 1


def test_blinded_audit_hides_the_model_and_keeps_a_separate_key(tmp_path):
    outputs = {"model_alpha": {"n1": "x [NOME]", "n2": "y"}, "model_beta": {"n1": "x Zelda", "n2": "y"}}
    out = manual_validation.write_blinded_audit(outputs, {"n1": "x Zelda", "n2": "y"}, tmp_path)
    sheet = (out / "audit_sheet.csv").read_text(encoding="utf-8")
    assert "model_alpha" not in sheet and "model_beta" not in sheet and "residual_sensitive_information" in sheet
    key = json.loads((out / "audit_key_do_not_open_before_finishing.json").read_text(encoding="utf-8"))
    assert len(key) == 4 and {v["model"] for v in key.values()} == {"model_alpha", "model_beta"}
