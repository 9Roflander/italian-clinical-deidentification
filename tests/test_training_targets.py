"""Segment targets: exact reconstruction, labels and rejection of malformed input. Invented text only."""
import json
import random

import pytest

from src.evaluation.alignment import CATEGORIES, OUTSIDE, VALID, resolve_prediction, resolve_segments
from src.evaluation.metrics import as_report, compute_metrics
from src.evaluation.scoring import score_note, totals
from src.training.targets import TargetError, build_segments, serialize_target, validate_segments


def text_of(segments):
    return "".join(s["text"] for s in segments)


def spans_read_back(text, segments):
    resolved = resolve_segments(text, serialize_target(segments))
    assert resolved.status == VALID
    return [(s.start, s.end, s.label) for s in resolved.spans]


def test_exact_segments_and_labels():
    text = "Zelinda ha 45 anni."
    assert build_segments(text, [(0, 7, "NOME"), (11, 13, "ETÀ")]) == [
        {"text": "Zelinda", "label": "NOME"}, {"text": " ha ", "label": OUTSIDE}, {"text": "45", "label": "ETÀ"}, {"text": " anni.", "label": OUTSIDE}]


@pytest.mark.parametrize("label", CATEGORIES)
def test_every_category_keeps_its_exact_label(label):
    assert build_segments("ab cd", [(0, 2, label)])[0]["label"] == label


def test_no_entities_gives_one_ordinary_segment():
    assert build_segments("Nessun dato.", []) == [{"text": "Nessun dato.", "label": OUTSIDE}]


def test_repeated_strings_are_separate_segments_at_their_own_offsets():
    text = "Zelinda vede Zelinda e Zelinda."
    spans = [(0, 7, "NOME"), (13, 20, "NOME")]
    segments = build_segments(text, spans)
    assert text_of(segments) == text and spans_read_back(text, segments) == spans
    assert segments[-1] == {"text": " e Zelinda.", "label": OUTSIDE}, "the unmarked third occurrence stays ordinary text"


def test_unicode_offsets_are_code_points():
    text = "Età già: Niccolò, 🙂 Città più."
    a = text.index("Niccolò")
    b = text.index("Città")
    spans = [(a, a + 7, "NOME"), (b, b + 5, "LUOGO/INDIRIZZO")]
    segments = build_segments(text, spans)
    assert text_of(segments) == text and spans_read_back(text, segments) == spans
    assert [s["text"] for s in segments if s["label"] != OUTSIDE] == ["Niccolò", "Città"]


def test_adjacent_entities_with_no_gap_produce_no_empty_segment():
    text = "Zelinda45 anni"
    segments = build_segments(text, [(0, 7, "NOME"), (7, 9, "ETÀ")])
    assert [s["text"] for s in segments] == ["Zelinda", "45", " anni"] and all(s["text"] for s in segments)


def test_entities_at_both_ends_and_whole_note():
    assert [s["label"] for s in build_segments("Zelinda ha Borgoverde", [(0, 7, "NOME"), (11, 21, "LUOGO/INDIRIZZO")])] == ["NOME", OUTSIDE, "LUOGO/INDIRIZZO"]
    assert build_segments("Zelinda", [(0, 7, "NOME")]) == [{"text": "Zelinda", "label": "NOME"}]


@pytest.mark.parametrize("text", ["Doppi  spazi   qui.", "Riga uno\nRiga due\r\n\tfine", ' Spazio iniziale e finale ', 'Virgolette "doppie" e \\ barra.', "dell'Aquilonia; (nota) [sic] <x>", "a b"])
def test_whitespace_and_punctuation_survive_serialization_exactly(text):
    segments = build_segments(text, [])
    assert text_of(json.loads(serialize_target(segments))) == text


def test_serialization_is_canonical():
    assert serialize_target(build_segments("Età", [])) == '[{"text":"Età","label":"O"}]'


@pytest.mark.parametrize("text,spans", [
    ("abcdef", [(0, 3, "NOME"), (2, 5, "DATA")]),
    ("abcdef", [(0, 3, "NOME"), (0, 3, "NOME")]),
    ("abcdef", [(-1, 3, "NOME")]),
    ("abcdef", [(3, 3, "NOME")]),
    ("abcdef", [(4, 2, "NOME")]),
    ("abcdef", [(0, 99, "NOME")]),
    ("abcdef", [(0, 3, "PERSONA")]),
    ("abcdef", [(0, 3, "O")]),
    ("abcdef", [(0.0, 3, "NOME")]),
    ("abcdef", [(True, 3, "NOME")]),
    ("abcdef", [("0", 3, "NOME")]),
    ("abcdef", [(0, 3)]),
    ("ab cd", [(1, 3, "NOME")]),
    ("ab cd", [(2, 4, "NOME")]),
    ("", []),
    ("Città", []),
])
def test_malformed_input_is_rejected(text, spans):
    with pytest.raises(TargetError):
        build_segments(text, spans)


@pytest.mark.parametrize("segments", [
    [],
    [{"text": "", "label": "O"}, {"text": "ab", "label": "O"}],
    [{"text": "ab", "label": "PERSONA"}],
    [{"text": "ab", "label": "O", "extra": 1}],
    [{"text": "ab"}],
    [{"text": "a", "label": "O"}],
    [{"text": "abc", "label": "O"}],
    [{"text": 1, "label": "O"}],
    "not a list",
])
def test_malformed_segments_are_rejected(segments):
    with pytest.raises(TargetError):
        validate_segments("ab", segments)


def test_rejection_messages_hold_no_note_text():
    with pytest.raises(TargetError) as e:
        build_segments("ZZSENTINEL resto", [(0, 99, "NOME")])
    assert "ZZSENTINEL" not in str(e.value)


def test_randomized_round_trip():
    rng = random.Random(11)
    pieces = ["Zelinda", " ", "  ", "45", ", ", "dell'", "Città", "è", "\n", '"', "\\", "🙂", "12/03/2015", "."]
    checked = 0
    for _ in range(1500):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 25)))
        spans, pos = [], 0
        for _ in range(rng.randint(0, 5)):
            start = pos + rng.randint(0, 5)
            end = start + rng.randint(1, 6)
            if end > len(text):
                break
            if text[start:end] == text[start:end].strip():
                spans.append((start, end, rng.choice(CATEGORIES)))
                pos = end
        segments = build_segments(text, spans)
        assert text_of(segments) == text and spans_read_back(text, segments) == spans and all(s["text"] for s in segments)
        checked += 1
    assert checked == 1500


def test_a_perfect_target_scores_one_in_the_evaluator():
    text = "Zelinda ha 45 anni, Borgoverde."
    spans = [(0, 7, "NOME"), (11, 13, "ETÀ"), (20, 30, "LUOGO/INDIRIZZO")]
    resolved = resolve_prediction(text, "segments", serialize_target(build_segments(text, spans)))
    metrics = as_report(compute_metrics(totals([score_note("n", 1, text, spans, resolved)])))
    assert metrics["micro_f1"] == 1 and metrics["note_exact_rate"] == 1
