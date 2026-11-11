"""Metric equations and privacy measures, including an independent recomputation on random predictions."""
import random

import numpy as np
import pytest

from src.evaluation.alignment import CATEGORIES, resolve_prediction, resolve_spans
from src.evaluation.metrics import COUNT_COLUMNS, as_report, compute_metrics
from src.evaluation.scoring import INVALID_OUTPUT, score_note, totals


def totals_with(**values):
    t = dict.fromkeys(COUNT_COLUMNS, 0)
    t.update(values)
    return t


def counts_for(tp, fp, fn, label="NOME"):
    return {f"tp_{label}": tp, f"fp_{label}": fp, f"fn_{label}": fn}


@pytest.mark.parametrize("tp,fp,fn", [(1, 0, 0), (3, 1, 2), (7, 3, 11), (90, 10, 40)])
def test_f1_matches_the_protocol_formula(tp, fp, fn):
    m = as_report(compute_metrics(totals_with(**counts_for(tp, fp, fn))))
    p, r = tp / (tp + fp), tp / (tp + fn)
    assert m["precision_NOME"] == pytest.approx(p) and m["recall_NOME"] == pytest.approx(r)
    assert m["f1_NOME"] == pytest.approx(2 * p * r / (p + r))


def test_micro_and_macro_by_hand():
    t = totals_with(**counts_for(8, 2, 0, "NOME"), **counts_for(1, 0, 9, "ETÀ"), **counts_for(5, 5, 5, "DATA"), **counts_for(0, 1, 1, "LUOGO/INDIRIZZO"))
    m = as_report(compute_metrics(t))
    # micro: TP 14, FP 8, FN 15, so F1 = 28 / 51
    assert m["micro_precision"] == pytest.approx(14 / 22) and m["micro_recall"] == pytest.approx(14 / 29) and m["micro_f1"] == pytest.approx(28 / 51)
    # category F1: 16/18, 2/11, 1/2, 0
    assert m["macro_f1"] == pytest.approx((16 / 18 + 2 / 11 + 0.5 + 0) / 4)
    assert m["macro_f1"] != pytest.approx(m["micro_f1"])


def test_macro_ignores_only_categories_with_no_gold_and_no_prediction():
    m = as_report(compute_metrics(totals_with(**counts_for(1, 0, 0))))
    assert m["macro_f1"] == 1.0 and m["f1_ETÀ"] is None
    m = as_report(compute_metrics(totals_with(**counts_for(1, 0, 0), **counts_for(0, 3, 0, "DATA"))))
    assert m["macro_f1"] == 0.5 and m["f1_DATA"] == 0.0


def test_undefined_is_never_reported_as_perfect():
    m = as_report(compute_metrics(totals_with()))
    for k in ("micro_precision", "micro_recall", "micro_f1", "macro_f1", "removal_coverage", "over_redaction_rate"):
        assert m[k] is None, k


def test_unknown_placeholder_lowers_micro_but_not_a_category():
    m = as_report(compute_metrics(totals_with(**counts_for(1, 0, 0), fp_unknown=1)))
    assert m["micro_precision"] == 0.5 and m["precision_NOME"] == 1.0


def test_coverage_and_over_redaction_by_hand():
    source = "Zelda Quaglia ha 45 anni."
    gold = [(0, 13, "NOME"), (17, 19, "ETÀ")]
    # P covers "Zelda" (5 gold) and " anni" (5 non-gold). |G| = 15, |G∩P| = 5, |P| = 10, |P\G| = 5.
    score = score_note("n", 1, source, gold, resolve_spans(source, [[0, 5, "NOME"], [19, 24, "ETÀ"]]))
    m = as_report(compute_metrics(totals([score])))
    assert m["removal_coverage"] == pytest.approx(5 / 15) and m["over_redaction_rate"] == pytest.approx(5 / 10)
    assert m["complete_entity_recall"] == 0
    assert score.counts["non_sensitive_preserved"] == len(source) - 15 - 5 and score.counts["non_sensitive_chars"] == len(source) - 15


def test_note_exact_rate_is_equivalent_to_matching_the_target_string():
    source = "Zelda ha 45 anni."
    gold = [(0, 5, "NOME"), (9, 11, "ETÀ")]
    target = "[NOME] ha [ETÀ] anni."
    for output, exact in [(target, 1), ("[NOME] ha 45 anni.", 0), ("[DATA] ha [ETÀ] anni.", 0), (target + " ", 1), ("[NOME] [NOME] [ETÀ] anni.", 0), (source, 0)]:
        score = score_note("n", 1, source, gold, resolve_prediction(source, "inline", output))
        assert score.counts["notes_exact"] == exact, output


def test_correct_redaction_recall_is_strict_recall():
    source = "Zelda ha 45 anni."
    score = score_note("n", 1, source, [(0, 5, "NOME"), (9, 11, "ETÀ")], resolve_prediction(source, "inline", "[NOME] ha 45 anni."))
    assert as_report(compute_metrics(totals([score])))["micro_recall"] == 0.5


def test_fidelity_counts_on_invalid_outputs():
    source = "Zelda ha febbre alta."
    typo = score_note("n", 1, source, [(0, 5, "NOME")], resolve_prediction(source, "inline", "[NOME] ha febbri alta."))
    assert (typo.counts["inserted_chars"], typo.counts["deleted_chars"], typo.counts["modified_chars"]) == (0, 0, 1)
    added = score_note("n", 1, source, [], resolve_prediction(source, "inline", source + " Altro"))
    assert (added.counts["inserted_chars"], added.counts["deleted_chars"], added.counts["modified_chars"]) == (6, 0, 0)
    cut = score_note("n", 1, source, [], resolve_prediction(source, "inline", "Zelda ha febbre"))
    assert cut.counts["deleted_chars"] == 6 and cut.counts["inserted_chars"] == 0
    assert typo.counts["notes_text_preserved"] == added.counts["notes_text_preserved"] == 0


def test_fidelity_is_zero_for_valid_outputs_even_with_over_redaction():
    source = "Zelda ha febbre alta."
    score = score_note("n", 1, source, [(0, 5, "NOME")], resolve_prediction(source, "inline", "[NOME] ha [DATA] alta."))
    assert (score.counts["inserted_chars"], score.counts["deleted_chars"], score.counts["modified_chars"]) == (0, 0, 0)
    assert score.counts["notes_text_preserved"] == 1 and score.counts["over_redacted_chars"] == 6


def test_unmeasurable_fidelity_is_reported():
    source = "Zelda ha febbre."
    score = score_note("n", 1, source, [(0, 5, "NOME")], resolve_prediction(source, "segments", "not json"))
    assert score.counts["fidelity_unmeasurable"] == 1 and score.counts["non_sensitive_preserved"] == 0


def test_invalid_output_gets_no_credit_of_any_kind():
    source = "a b c b d"
    score = score_note("n", 1, source, [(0, 1, "NOME"), (4, 9, "DATA")], resolve_prediction(source, "inline", "[NOME] b [DATA]"))
    assert score.counts["covered_chars"] == score.counts["predicted_chars"] == score.counts["entities_complete"] == 0
    assert score.errors[("NOME", "gold", INVALID_OUTPUT)] == 1 and score.errors[("DATA", "prediction", INVALID_OUTPUT)] == 1


def test_scorer_agrees_with_independent_recomputation():
    rng = random.Random(5)
    for _ in range(3000):
        length = 60
        source = "x" * length
        gold, pos = [], 0
        while pos < length - 5 and len(gold) < 5:
            a = pos + rng.randint(0, 8)
            b = a + rng.randint(1, 5)
            if b > length:
                break
            gold.append((a, b, rng.choice(CATEGORIES)))
            pos = b + 1
        predictions, pos = [], 0
        for g in gold:
            r = rng.random()
            if r < 0.4:
                predictions.append(list(g))
            elif r < 0.6:
                predictions.append([g[0], g[1], rng.choice(CATEGORIES)])
            elif r < 0.75 and g[1] - g[0] > 1:
                predictions.append([g[0], g[1] - 1, g[2]])
        if rng.random() < 0.4 and len(gold) and gold[-1][1] + 3 < length:
            predictions.append([gold[-1][1] + 1, gold[-1][1] + 3, rng.choice(CATEGORIES)])
        if predictions and rng.random() < 0.2:
            predictions.append(list(predictions[0]))
        score = score_note("n", 1, source, gold, resolve_spans(source, predictions))
        pred_set = {tuple(p) for p in predictions}
        gold_set = set(gold)
        for c in CATEGORIES:
            tp = sum(1 for p in pred_set & gold_set if p[2] == c)
            total_pred = sum(1 for p in predictions if p[2] == c)
            total_gold = sum(1 for g in gold if g[2] == c)
            assert score.counts[f"tp_{c}"] == tp
            assert score.counts[f"fp_{c}"] == total_pred - tp
            assert score.counts[f"fn_{c}"] == total_gold - tp
        g_chars = {i for g in gold for i in range(g[0], g[1])}
        p_chars = {i for p in predictions for i in range(p[0], p[1])}
        assert score.counts["gold_chars"] == len(g_chars) and score.counts["covered_chars"] == len(g_chars & p_chars)
        assert score.counts["predicted_chars"] == len(p_chars) and score.counts["over_redacted_chars"] == len(p_chars - g_chars)
        gold_side = sum(n for (c, side, _), n in score.errors.items() if side == "gold")
        prediction_side = sum(n for (c, side, _), n in score.errors.items() if side == "prediction")
        assert gold_side == len(gold), "every gold entity has exactly one outcome"
        assert prediction_side == sum(score.counts[f"fp_{c}"] for c in CATEGORIES), "every false positive has exactly one cause"


def test_scalar_and_array_metrics_agree():
    t = totals_with(**counts_for(3, 1, 2), **counts_for(1, 0, 1, "DATA"), gold_chars=10, covered_chars=7, predicted_chars=9, over_redacted_chars=2, notes=3, notes_exact=1)
    scalar = as_report(compute_metrics(t))
    array = compute_metrics({k: np.array([v, v]) for k, v in t.items()})
    for k, v in scalar.items():
        if v is None:
            assert np.isnan(array[k]).all(), k
        else:
            assert array[k][0] == pytest.approx(v) and array[k][1] == pytest.approx(v), k
