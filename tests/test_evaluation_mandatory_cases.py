"""The 16 mandatory cases of protocol section 6, with expected results worked out by hand. Invented text only."""
import json

import pytest

from src.evaluation.alignment import (AMBIGUOUS, EMPTY, OVERLAPPING, TRUNCATED, UNKNOWN, VALID, render_inline, resolve_prediction, spans_to_segments)
from src.evaluation.metrics import as_report, compute_metrics
from src.evaluation.scoring import DUPLICATE, MISSED, SPURIOUS, WRONG_BOUNDARY, WRONG_CATEGORY, score_note, totals

# 20 characters, then an 11-character name at [20, 31), the protocol's example positions.
SOURCE = "La paziente seguita Zelda Quagl riferisce dolore."
NAME = (20, 31, "NOME")


def run(source, gold, fmt, raw=None, spans=None, **kw):
    score = score_note("n", 1, source, gold, resolve_prediction(source, fmt, raw, spans, **kw))
    return score, as_report(compute_metrics(totals([score])))


def tpfpfn(score, label):
    return score.counts[f"tp_{label}"], score.counts[f"fp_{label}"], score.counts[f"fn_{label}"]


def test_source_positions_match_the_protocol_example():
    assert SOURCE[20:31] == "Zelda Quagl" and len(SOURCE[:20]) == 20


@pytest.mark.parametrize("fmt,raw,spans", [
    ("spans", None, [list(NAME)]),
    ("inline", "La paziente seguita [NOME] riferisce dolore.", None),
    ("segments", json.dumps([{"text": "La paziente seguita ", "label": "O"}, {"text": "Zelda Quagl", "label": "NOME"}, {"text": " riferisce dolore.", "label": "O"}]), None),
])
def test_01_perfect_redaction(fmt, raw, spans):
    score, m = run(SOURCE, [NAME], fmt, raw, spans)
    assert score.status == VALID and tpfpfn(score, "NOME") == (1, 0, 0)
    assert m["micro_precision"] == m["micro_recall"] == m["micro_f1"] == m["macro_f1"] == 1.0
    assert m["removal_coverage"] == 1.0 and m["over_redaction_rate"] == 0.0 and m["note_exact_rate"] == 1.0


@pytest.mark.parametrize("fmt,raw", [("inline", ""), ("inline", "   \n "), ("inline", None), ("segments", ""), ("segments", "[]")])
def test_02_empty_output_scores_zero(fmt, raw):
    score, m = run(SOURCE, [NAME], fmt, raw)
    assert score.status == EMPTY and tpfpfn(score, "NOME") == (0, 0, 1)
    assert m["micro_recall"] == 0 and m["micro_f1"] == 0 and m["macro_f1"] == 0
    assert m["removal_coverage"] == 0 and m["note_exact_rate"] == 0 and m["invalid_output_rate"] == 1


def test_critical_empty_output_is_never_perfect_even_on_a_note_without_entities():
    score, m = run("Nessun dato identificativo.", [], "inline", "")
    assert score.status == EMPTY and m["note_exact_rate"] == 0 and m["invalid_output_rate"] == 1
    assert m["micro_f1"] is None and m["macro_f1"] is None and score.counts["deleted_chars"] == 27


def test_03_unchanged_input_scores_zero():
    score, m = run(SOURCE, [NAME], "inline", SOURCE)
    assert score.status == VALID and tpfpfn(score, "NOME") == (0, 0, 1)
    assert m["micro_recall"] == 0 and m["micro_f1"] == 0 and m["removal_coverage"] == 0
    assert m["micro_precision"] is None and m["over_redaction_rate"] is None


def test_04_wrong_category_is_false_positive_and_false_negative():
    for fmt, raw, spans in [("spans", None, [[20, 31, "ETÀ"]]), ("inline", "La paziente seguita [ETÀ] riferisce dolore.", None)]:
        score, m = run(SOURCE, [NAME], fmt, raw, spans)
        assert tpfpfn(score, "NOME") == (0, 0, 1) and tpfpfn(score, "ETÀ") == (0, 1, 0)
        assert m["micro_f1"] == 0 and m["macro_f1"] == 0
        assert m["removal_coverage"] == 1.0, "the text is gone, though the tag is wrong"
        assert score.errors[("NOME", "gold", WRONG_CATEGORY)] == 1 and score.errors[("ETÀ", "prediction", WRONG_CATEGORY)] == 1


@pytest.mark.parametrize("start,end", [(20, 30), (19, 31), (21, 31), (20, 32)])
def test_05_wrong_boundary_is_false_positive_and_false_negative(start, end):
    score, m = run(SOURCE, [NAME], "spans", spans=[[start, end, "NOME"]])
    assert tpfpfn(score, "NOME") == (0, 1, 1) and m["micro_f1"] == 0
    assert score.errors[("NOME", "gold", WRONG_BOUNDARY)] == 1


def test_06_partial_overlap_gets_no_strict_credit():
    score, m = run(SOURCE, [NAME], "inline", "La paziente seguita [NOME] Quagl riferisce dolore.")
    assert score.status == VALID and tpfpfn(score, "NOME") == (0, 1, 1) and m["micro_f1"] == 0
    assert m["removal_coverage"] == pytest.approx(5 / 11) and m["complete_entity_recall"] == 0
    assert m["over_redaction_rate"] == 0


def test_07_spurious_placeholder_is_a_false_positive():
    score, m = run(SOURCE, [NAME], "inline", "La paziente seguita [NOME] riferisce [NOME].")
    assert tpfpfn(score, "NOME") == (1, 1, 0)
    assert m["micro_precision"] == 0.5 and m["micro_recall"] == 1.0 and m["micro_f1"] == pytest.approx(2 / 3)
    assert m["over_redaction_rate"] == pytest.approx(6 / 17) and m["note_exact_rate"] == 0
    assert score.errors[("NOME", "prediction", SPURIOUS)] == 1


def test_08_duplicate_prediction_earns_one_true_positive():
    score, m = run(SOURCE, [NAME], "spans", spans=[list(NAME), list(NAME), list(NAME)])
    assert score.status == VALID and tpfpfn(score, "NOME") == (1, 2, 0)
    assert m["micro_precision"] == pytest.approx(1 / 3) and m["micro_recall"] == 1
    assert score.errors[("NOME", "prediction", DUPLICATE)] == 2 and m["note_exact_rate"] == 0


def test_09_repeated_name_is_scored_at_each_offset():
    source = "Zelda vede Zelda."
    gold = [(0, 5, "NOME"), (11, 16, "NOME")]
    first, m1 = run(source, gold, "inline", "[NOME] vede Zelda.")
    second, m2 = run(source, gold, "inline", "Zelda vede [NOME].")
    both, m3 = run(source, gold, "inline", "[NOME] vede [NOME].")
    assert first.predicted_spans[0].start == 0 and second.predicted_spans[0].start == 11
    assert tpfpfn(first, "NOME") == (1, 0, 1) == tpfpfn(second, "NOME") and tpfpfn(both, "NOME") == (2, 0, 0)
    assert m1["micro_recall"] == m2["micro_recall"] == 0.5 and m3["micro_recall"] == 1


def test_09b_short_age_is_not_confused_with_digits_in_other_numbers():
    source = "Bimbo di 8 anni, peso 18 kg, altezza 88 cm."
    score, m = run(source, [(9, 10, "ETÀ")], "inline", "Bimbo di [ETÀ] anni, peso 18 kg, altezza 88 cm.")
    assert score.status == VALID and tpfpfn(score, "ETÀ") == (1, 0, 0) and m["micro_f1"] == 1


def test_10_ambiguous_alignment_earns_no_guessed_credit():
    """Both 'a' + 'c b d' and 'a b c' + 'd' fit. Gold equals the first, and it still earns nothing."""
    source = "a b c b d"
    gold = [(0, 1, "NOME"), (4, 9, "DATA")]
    score, m = run(source, gold, "inline", "[NOME] b [DATA]")
    assert score.status == AMBIGUOUS and score.predicted_spans == ()
    assert tpfpfn(score, "NOME") == (0, 1, 1) and tpfpfn(score, "DATA") == (0, 1, 1)
    assert m["micro_f1"] == 0 and m["removal_coverage"] == 0 and m["invalid_output_rate"] == 1
    assert score.counts["notes_text_preserved"] == 1, "the literal text was copied, but its position is unknown"


def test_11_unicode_accents_use_exact_code_point_offsets():
    source = "Età già nota: Niccolò Città, più volte è tornata."
    start = source.index("Niccolò")
    gold = [(start, start + 7, "NOME"), (start + 8, start + 13, "NOME")]
    assert source[gold[0][0] : gold[0][1]] == "Niccolò" and source[gold[1][0] : gold[1][1]] == "Città"
    target = render_inline(source, gold)
    for variant in (target, __import__("unicodedata").normalize("NFD", target)):
        score, m = run(source, gold, "inline", variant)
        assert score.status == VALID and m["micro_f1"] == 1
    assert run(source, gold, "segments", json.dumps(spans_to_segments(source, gold), ensure_ascii=False))[1]["micro_f1"] == 1


def test_11b_accented_placeholder_is_recognized_when_decomposed():
    score, m = run("Eta 45 anni.", [(4, 6, "ETÀ")], "inline", "Eta [ETÀ] anni.")
    assert score.status == VALID and m["micro_f1"] == 1


@pytest.mark.parametrize("source,word,label", [
    ("Ricoverata dell'Ospedale Verdeluna, poi dimessa.", "Verdeluna", "LUOGO/INDIRIZZO"),
    ("Sig.ra (Zelda) riferita.", "Zelda", "NOME"),
    ("Vista all'Aquilonia; rientro.", "Aquilonia", "LUOGO/INDIRIZZO"),
    ("Data: 12/03/2015.", "12/03/2015", "DATA"),
])
def test_12_punctuation_and_apostrophes_use_exact_offsets(source, word, label):
    start = source.index(word)
    gold = [(start, start + len(word), label)]
    score, m = run(source, gold, "inline", render_inline(source, gold))
    assert score.status == VALID and m["micro_f1"] == 1
    assert score.predicted_spans[0].start == start and score.predicted_spans[0].end == start + len(word)


def test_12b_changing_an_apostrophe_breaks_exactness():
    source = "Ricoverata dell'Ospedale Verdeluna."
    start = source.index("Verdeluna")
    score, m = run(source, [(start, start + 9, "LUOGO/INDIRIZZO")], "inline", "Ricoverata dell’Ospedale [LUOGO/INDIRIZZO].")
    assert score.status != VALID and m["micro_f1"] == 0 and score.counts["modified_chars"] == 1


def test_13_truncated_output_is_invalid_and_reported():
    target = "La paziente seguita [NOME] riferisce dolore."
    cut, m = run(SOURCE, [NAME], "inline", target[:30])
    assert cut.status == TRUNCATED and tpfpfn(cut, "NOME") == (0, 1, 1) and m["micro_f1"] == 0 and m["invalid_output_rate"] == 1
    plain, _ = run(SOURCE, [NAME], "inline", SOURCE[:15])
    assert plain.status == TRUNCATED and tpfpfn(plain, "NOME") == (0, 0, 1)
    flagged, m2 = run(SOURCE, [NAME], "inline", target, truncation_flag=True)
    assert flagged.status == TRUNCATED and tpfpfn(flagged, "NOME") == (0, 1, 1) and m2["micro_f1"] == 0
    reason, _ = run(SOURCE, [NAME], "spans", spans=[list(NAME)], finish_reason="length")
    assert reason.status == TRUNCATED and tpfpfn(reason, "NOME") == (0, 1, 1)


def test_14_no_entity_note_has_no_false_positive_when_unchanged():
    source = "Nessun dato identificativo."
    score, m = run(source, [], "inline", source)
    assert score.status == VALID and tpfpfn(score, "NOME") == (0, 0, 0) and score.counts["fp_unknown"] == 0
    assert m["note_exact_rate"] == 1 and m["micro_f1"] is None and m["invalid_output_rate"] == 0


def test_15_overlapping_predictions_are_invalid_and_reported():
    score, m = run(SOURCE, [NAME], "spans", spans=[[20, 31, "NOME"], [25, 35, "DATA"]])
    assert score.status == OVERLAPPING and score.predicted_spans == ()
    assert tpfpfn(score, "NOME") == (0, 1, 1) and tpfpfn(score, "DATA") == (0, 1, 0)
    assert m["micro_f1"] == 0 and m["invalid_output_rate"] == 1


def test_16_unknown_placeholder_is_an_invalid_prediction():
    score, m = run(SOURCE, [NAME], "inline", "La paziente seguita [NAME] riferisce dolore.")
    assert score.status == VALID and score.predicted_spans[0].label == UNKNOWN
    assert score.counts["fp_unknown"] == 1 and tpfpfn(score, "NOME") == (0, 0, 1)
    assert m["micro_precision"] == 0 and m["micro_f1"] == 0 and m["macro_f1"] == 0
    assert m["removal_coverage"] == 1.0 and m["note_exact_rate"] == 0


def test_16b_unknown_placeholder_does_not_cancel_a_correct_one():
    source = "Zelda ha 45 anni."
    gold = [(0, 5, "NOME"), (9, 11, "ETÀ")]
    score, m = run(source, gold, "inline", "[NOME] ha [ETA] anni.")
    assert tpfpfn(score, "NOME") == (1, 0, 0) and tpfpfn(score, "ETÀ") == (0, 0, 1) and score.counts["fp_unknown"] == 1
    assert m["micro_precision"] == 0.5 and m["micro_recall"] == 0.5


def test_17_missed_entity_is_classified():
    source = "Zelda ha 45 anni."
    score, _ = run(source, [(0, 5, "NOME"), (9, 11, "ETÀ")], "inline", "[NOME] ha 45 anni.")
    assert score.errors[("ETÀ", "gold", MISSED)] == 1
