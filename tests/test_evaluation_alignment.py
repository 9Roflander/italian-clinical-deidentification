"""The inline adapter and segment reader, checked against brute force and with randomized round trips."""
import inspect
import json
import random

import pytest

from src.data.loaders import GOLD_V2, load_gold
from src.evaluation.alignment import (AMBIGUOUS, CATEGORIES, EMPTY, INVALID_JSON, NO_ALIGNMENT, SEGMENTS_MISMATCH, SPAN_OUT_OF_RANGE, SOURCE_HAS_PLACEHOLDER, TRUNCATED,
                                      UNKNOWN, VALID, WHITESPACE_CHANGED, align_inline, count_alignments, render_inline, resolve_prediction, resolve_segments, resolve_spans, spans_to_segments)


def triples(resolution):
    return [(s.start, s.end, s.label) for s in resolution.spans]


def brute_force(source, literals):
    """Every placement of the literals in order, with a non-empty gap for each placeholder. Independent of the DP."""
    n = len(literals) - 1
    found = []

    def place(i, pos, gaps):
        if i == n:
            start = len(source) - len(literals[n])
            if start > pos and source[start:] == literals[n] and (n > 0):
                found.append(gaps + [(pos, start)])
            return
        for p in range(pos + 1, len(source) + 1):
            if source.startswith(literals[i], p):
                place(i + 1, p + len(literals[i]), gaps + [(pos, p)])
    if not source.startswith(literals[0]):
        return []
    if n == 0:
        return [[]] if source == literals[0] else []
    place(1, len(literals[0]), [])
    return found


def test_dp_agrees_with_brute_force_on_random_small_cases():
    rng = random.Random(1)
    for _ in range(6000):
        source = "".join(rng.choice("ab ") for _ in range(rng.randint(1, 12)))
        k = rng.randint(1, 4)
        literals = ["".join(rng.choice("ab ") for _ in range(rng.randint(0, 2))) for _ in range(k)]
        expected = brute_force(source, literals)
        count, gaps = count_alignments(source, literals)
        assert count == min(2, len(expected)), (source, literals)
        if count == 1:
            assert gaps == expected[0], (source, literals)


def test_rendered_spans_are_recovered_or_flagged_never_silently_wrong():
    rng = random.Random(2)
    outcomes = {VALID: 0, AMBIGUOUS: 0}
    for _ in range(6000):
        source = "".join(rng.choice("ab ,") for _ in range(rng.randint(5, 30)))
        cuts = sorted(rng.sample(range(len(source) + 1), min(2 * rng.randint(0, 3), len(source) // 2 * 2)))
        spans = [(cuts[i], cuts[i + 1], rng.choice(CATEGORIES)) for i in range(0, len(cuts) - 1, 2) if cuts[i] < cuts[i + 1]]
        if source != source.strip():
            continue
        resolved = align_inline(source, render_inline(source, spans))
        assert resolved.status in outcomes, (source, spans, resolved.status)
        outcomes[resolved.status] += 1
        if resolved.status == VALID:
            assert triples(resolved) == sorted(spans), (source, spans)
    assert outcomes[VALID] > 500 and outcomes[AMBIGUOUS] > 100, "the test must exercise both outcomes"


def test_segments_round_trip_on_random_spans_including_repeated_entities():
    rng = random.Random(3)
    for _ in range(3000):
        source = "".join(rng.choice(["Zelda", " ", "45", ", ", "dell'", "Città", "è"]) for _ in range(rng.randint(1, 20)))
        cuts = sorted(rng.sample(range(len(source) + 1), min(6, len(source) + 1) // 2 * 2))
        spans = [(cuts[i], cuts[i + 1], rng.choice(CATEGORIES)) for i in range(0, len(cuts) - 1, 2)]
        segments = spans_to_segments(source, spans)
        assert "".join(s["text"] for s in segments) == source
        resolved = resolve_segments(source, json.dumps(segments, ensure_ascii=False))
        assert resolved.status == VALID and triples(resolved) == sorted(spans)
        assert triples(resolve_spans(source, [list(s) for s in spans])) == sorted(spans)


def test_real_gold_notes_with_random_spans_are_never_silently_wrong():
    """Assertions report IDs only."""
    rng = random.Random(4)
    wrong, statuses = [], {VALID: 0, AMBIGUOUS: 0}
    for note in load_gold(GOLD_V2):
        for _ in range(10):
            spans, pos = [], 0
            while pos < len(note.text) - 3 and len(spans) < 12:
                start = pos + rng.randint(1, 400)
                end = start + rng.randint(1, 12)
                if end >= len(note.text):
                    break
                if note.text[start:end].strip() == note.text[start:end]:
                    spans.append((start, end, rng.choice(CATEGORIES)))
                pos = end + 1
            resolved = align_inline(note.text, render_inline(note.text, spans))
            statuses[resolved.status] = statuses.get(resolved.status, 0) + 1
            if resolved.status == VALID and triples(resolved) != spans:
                wrong.append(note.note_id)
            if resolved.status not in (VALID, AMBIGUOUS):
                wrong.append(note.note_id)
    assert not wrong, wrong
    assert statuses[VALID] > 700


def test_gold_targets_read_back_exactly():
    """A correct redaction of every real note is unique, so the inline ceiling on gold_v2 is 1.0."""
    bad = []
    for n in load_gold(GOLD_V2):
        got = align_inline(n.text, n.redacted_text)
        if got.status != VALID or triples(got) != sorted((e["start"], e["end"], e["type"]) for e in n.entities):
            bad.append(n.note_id)
    assert not bad, bad


def test_reader_cannot_receive_gold():
    for fn in (align_inline, resolve_segments, resolve_spans, resolve_prediction, count_alignments):
        names = set(inspect.signature(fn).parameters)
        assert not names & {"gold", "gold_spans", "entities", "annotations"}, fn.__name__


@pytest.mark.parametrize("output,status", [
    ("Preambolo: Zelda ha [ETÀ] anni.", NO_ALIGNMENT),                  # added text
    ("Zelda ha [ETÀ] anni. Fine.", NO_ALIGNMENT),                       # added text at the end
    ("Zelda [NOME] ha [ETÀ] anni.", NO_ALIGNMENT),                      # placeholder that replaced nothing
    ("Zeldo ha [ETÀ] anni.", NO_ALIGNMENT),                             # a typo: fuzzy matching is not allowed
    ("ha [ETÀ] anni.", NO_ALIGNMENT),                                   # silent deletion
    ("Luigi ha [ETÀ] anni.", NO_ALIGNMENT),                             # replaced by an invented name
    ("Zelda  ha [ETÀ] anni.", WHITESPACE_CHANGED),                      # whitespace differs: named, still invalid
    ("<NOME> ha [ETÀ] anni.", NO_ALIGNMENT),                            # wrong placeholder syntax
])
def test_any_change_to_literal_text_is_invalid(output, status):
    source = "Zelda ha 45 anni."
    got = align_inline(source, output)
    assert got.status == status and got.spans == ()


def test_outer_whitespace_only_is_normalized_and_recorded():
    got = align_inline("Zelda ha 45 anni.", "  Zelda ha [ETÀ] anni.\n")
    assert got.status == VALID and got.output_normalized and triples(got) == [(9, 11, "ETÀ")]
    kept = align_inline(" Zelda ha 45 anni.", "Zelda ha [ETÀ] anni.")
    assert kept.status != VALID, "a note with outer whitespace must match it exactly"


def test_adjacent_placeholders():
    assert triples(align_inline("xy", "[NOME][DATA]")) == [(0, 1, "NOME"), (1, 2, "DATA")]
    assert align_inline("xyz", "[NOME][DATA]").status == AMBIGUOUS
    assert triples(align_inline("Zelda Quaglia", "[NOME] [NOME]")) == [(0, 5, "NOME"), (6, 13, "NOME")]
    assert align_inline("Zelda Rina Quaglia", "[NOME] [NOME]").status == AMBIGUOUS


def test_partial_token_redaction_is_recovered_exactly():
    assert triples(align_inline("Zelda ha febbre.", "Zel[NOME] ha febbre.")) == [(3, 5, "NOME")]


def test_whole_note_replaced_by_one_placeholder_is_a_single_span():
    assert triples(align_inline("Zelda ha febbre.", "[NOME]")) == [(0, 16, "NOME")]


def test_bracketed_text_in_the_note_stays_literal():
    source = "Esito [sic] per Zelda."
    assert triples(align_inline(source, "Esito [sic] per [NOME].")) == [(16, 21, "NOME")]


def test_source_containing_a_placeholder_is_refused():
    assert align_inline("Campo [NOME] vuoto.", "Campo [NOME] vuoto.").status == SOURCE_HAS_PLACEHOLDER


def test_segments_invalid_inputs():
    source = "Zelda ha 45 anni."
    good = [{"text": "Zelda", "label": "NOME"}, {"text": " ha 45 anni.", "label": "O"}]
    assert triples(resolve_segments(source, good)) == [(0, 5, "NOME")]
    assert triples(resolve_segments(source, {"segments": good})) == [(0, 5, "NOME")]
    assert resolve_segments(source, '[{"text": "Zelda", "label": "NOME"},').status == INVALID_JSON
    assert resolve_segments(source, '[{"text": "Zelda", "label": "NOME"},').invalid_labels == ("NOME",)
    assert resolve_segments(source, '{"a": 1}').status == INVALID_JSON
    assert resolve_segments(source, [{"text": "Zelda", "label": "NOME", "extra": 1}]).status == INVALID_JSON
    assert resolve_segments(source, [{"text": "Zelda", "label": "NOME"}]).status == TRUNCATED
    assert resolve_segments(source, [{"text": "Zelda!", "label": "NOME"}, {"text": " ha 45 anni.", "label": "O"}]).status == SEGMENTS_MISMATCH
    assert resolve_segments(source, [{"text": "", "label": "NOME"}, *good]).status == SEGMENTS_MISMATCH
    assert resolve_segments(source, "").status == EMPTY
    unknown = resolve_segments(source, [{"text": "Zelda", "label": "PERSON"}, {"text": " ha 45 anni.", "label": "O"}])
    assert unknown.status == VALID and unknown.spans[0].label == UNKNOWN


@pytest.mark.parametrize("spans", [[[-1, 3, "NOME"]], [[3, 3, "NOME"]], [[5, 3, "NOME"]], [[0, 999, "NOME"]], [[0, "5", "NOME"]], [[0.0, 5, "NOME"]], [{"a": 1}], [[True, 5, "NOME"]]])
def test_spans_out_of_range_or_malformed_are_invalid(spans):
    assert resolve_spans("Zelda ha 45 anni.", spans).status == SPAN_OUT_OF_RANGE


def test_adjacent_spans_are_not_overlapping():
    assert resolve_spans("Zelda", [[0, 2, "NOME"], [2, 5, "NOME"]]).status == VALID


def test_generation_error_and_missing_spans_are_invalid():
    assert resolve_prediction("Zelda", "spans", predicted_spans=[[0, 5, "NOME"]], completion_status="error").status == "generation_error"
    assert resolve_prediction("Zelda", "spans", predicted_spans=None).status == "missing_prediction"
    assert resolve_prediction("Zelda", "xml", raw_output="Zelda").status == "unknown_output_format"
