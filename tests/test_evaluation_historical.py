"""The old placeholder-count metrics stay unchanged, and disagree with strict scoring in known ways."""
import ast
import json
import re

import pytest

from src.data.loaders import GOLD_V1, GOLD_V2, ROOT, load_gold
from src.data.splits import load_fold
from src.evaluation import historical as hist
from src.evaluation.alignment import resolve_prediction
from src.evaluation.scoring import score_note

ORIGINAL = ROOT / "original_work" / "src" / "train_evaluate_cv_v4.py"
RESULTS = ROOT / "audit" / "results"


def definitions(path):
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.dump(node)
    return out


@pytest.mark.parametrize("name", ["paper_metric", "sanitize_type", "VALID_TYPES"])
def test_old_metric_code_is_unchanged(name):
    assert definitions(ROOT / "src" / "evaluation" / "historical.py")[name] == definitions(ORIGINAL)[name]


def as_entities(*items):
    return [{"type": t, "text": x} for t, x in items]


def test_worked_example_two_mentions_one_left():
    ents = as_entities(("NOME", "Zelinda"), ("NOME", "Zelinda"))
    output = "Zelinda [NOME]"
    assert hist.paper_metric(output, ents)["NOME"] == (0, 0, 2)
    assert hist.miranda_deterministic(output, ents)["NOME"] == (1, 0, 1)


def test_short_age_with_the_digit_elsewhere():
    ents = as_entities(("ETÀ", "8"))
    output = "[ETÀ] anni, 18 mg, 28 mg, 80 mg"
    assert hist.paper_metric(output, ents)["ETÀ"] == (0, 0, 1)
    assert hist.miranda_deterministic(output, ents)["ETÀ"] == (0, 0, 3)
    source = "8 anni, 18 mg, 28 mg, 80 mg"
    strict = score_note("n", 1, source, [(0, 1, "ETÀ")], resolve_prediction(source, "inline", output))
    assert (strict.counts["tp_ETÀ"], strict.counts["fp_ETÀ"], strict.counts["fn_ETÀ"]) == (1, 0, 0)


def test_count_metric_and_strict_metric_disagree_where_they_should():
    source = "Zelda Quaglia ha 45 anni."
    gold = [(0, 13, "NOME"), (17, 19, "ETÀ")]
    ents = as_entities(("NOME", "Zelda Quaglia"), ("ETÀ", "45"))
    cases = {
        "empty": ("", (2, 0)),
        "wrong_tags": ("[ETÀ] ha [NOME] anni.", (2, 0)),
        "first_name_only": ("[NOME] Quaglia ha [ETÀ] anni.", (2, 1)),
    }
    for output, (old_tp, strict_tp) in cases.values():
        old = hist.paper_metric(output, ents)
        assert old["NOME"][0] + old["ETÀ"][0] == old_tp
        s = score_note("n", 1, source, gold, resolve_prediction(source, "inline", output)).counts
        assert s["tp_NOME"] + s["tp_ETÀ"] == strict_tp


def score_known_outputs(gold_path):
    """Reproduces the outputs from audit/scripts/v9_source_metric.py. Compares numbers only."""
    notes = load_gold(gold_path)
    fold_of = {i: k for k in range(1, 6) for i in load_fold(k).test_ids}
    entities = {n.note_id: n.entities for n in notes}
    nxt = dict(zip(hist.VALID_TYPES, hist.VALID_TYPES[1:] + hist.VALID_TYPES[:1]))
    wrong = lambda n: re.sub(r"\[(NOME|ETÀ|DATA|LUOGO/INDIRIZZO)\]", lambda m: f"[{nxt[m.group(1)]}]", n.redacted_text)
    outputs = {"correct_redaction_gold_v1_target": lambda n: n.redacted_text, "empty_output": lambda n: "", "input_unchanged": lambda n: n.text, "every_entity_wrong_tag": wrong}
    return {k: hist.historical_scores({n.note_id: fn(n) for n in notes}, entities, fold_of) for k, fn in outputs.items()}


@pytest.mark.parametrize("gold_path,result_file", [(GOLD_V1, "v9_source_metric.json"), (GOLD_V2, "v9_source_metric_v2.json")])
def test_restated_metrics_reproduce_the_audit_run_of_the_original_functions(gold_path, result_file):
    recorded = json.loads((RESULTS / result_file).read_text(encoding="utf-8"))
    got = score_known_outputs(gold_path)
    for output, values in got.items():
        assert round(values["paper_metric"]["macro_f1_mean_over_folds"], 4) == recorded[output]["paper_metric_as_used"]["macro_f1_mean"], output
        assert round(values["miranda_deterministic"]["macro_f1_mean_over_folds"], 4) == recorded[output]["miranda_eval_py"]["macro_f1_mean"], output
