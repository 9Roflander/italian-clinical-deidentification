"""E3-X extraction format and locator. Invented text only."""
import json

import pytest

from src.data.loaders import ROOT
from src.evaluation.alignment import VALID, resolve_prediction
from src.training.dataset import make_target
from src.training.experiments import EXPERIMENTS_CONFIG, load_yaml, require_training_decisions, resolve_experiment
from src.training.extraction import ExtractionTargetError, extraction_items, locate, parse_extraction, serialize_extraction_target
from src.training.formatting import read_system_prompt
from src.training.model_training import config_id, output_format_of

TEXT = "Paziente di 69 anni, visitato a Pisa il 3 maggio. Dimesso a Pisa."
AGE, CITY1, DATE, CITY2 = (12, 14, "ETÀ"), (32, 36, "LUOGO/INDIRIZZO"), (40, 48, "DATA"), (60, 64, "LUOGO/INDIRIZZO")


def test_fixture_offsets():
    assert [TEXT[a:b] for a, b, _ in (AGE, CITY1, DATE, CITY2)] == ["69", "Pisa", "3 maggio", "Pisa"]


def test_target_lists_items_in_order_and_reads_back():
    spans = [CITY2, AGE, DATE, CITY1]
    target = serialize_extraction_target(TEXT, spans)
    assert json.loads(target) == [{"text": "69", "label": "ETÀ"}, {"text": "Pisa", "label": "LUOGO/INDIRIZZO"}, {"text": "3 maggio", "label": "DATA"},
                                  {"text": "Pisa", "label": "LUOGO/INDIRIZZO"}]
    got, stats = locate(TEXT, parse_extraction(target))
    assert [(s["start"], s["end"], s["label"]) for s in got] == sorted(spans) and stats["unlocated"] == 0


def test_skip_item_only_where_an_unannotated_copy_comes_first():
    items = extraction_items(TEXT, [CITY2])   # the first "Pisa" is not annotated here
    assert items == [{"text": "Pisa", "label": "O"}, {"text": "Pisa", "label": "LUOGO/INDIRIZZO"}]
    got, stats = locate(TEXT, items)
    assert got == [{"start": 60, "end": 64, "label": "LUOGO/INDIRIZZO"}] and stats["skips"] == 1
    assert all(i["label"] != "O" for i in extraction_items(TEXT, [AGE, CITY1, DATE, CITY2]))


def test_negative_note_target_is_an_empty_list():
    assert serialize_extraction_target(TEXT, []) == "[]" and locate(TEXT, [])[0] == []


def test_entity_inside_a_longer_word_is_reachable():
    text = "Una 69enne con 69 battiti."
    assert serialize_extraction_target(text, [(4, 6, "ETÀ")]) == '[{"text":"69","label":"ETÀ"}]'


def test_unknown_category_and_overlap_are_refused():
    with pytest.raises(ExtractionTargetError):
        extraction_items(TEXT, [(12, 14, "FOO")])
    with pytest.raises(ExtractionTargetError):
        extraction_items(TEXT, [(32, 36, "LUOGO/INDIRIZZO"), (33, 40, "DATA")])


def test_unlocated_items_are_dropped_and_counted_without_moving_the_cursor():
    got, stats = locate(TEXT, [{"text": "Roma", "label": "LUOGO/INDIRIZZO"}, {"text": "", "label": "NOME"}, {"text": "69", "label": "ETÀ"}])
    assert got == [{"start": 12, "end": 14, "label": "ETÀ"}] and stats == {"items": 3, "located": 1, "unlocated": 2, "skips": 0}


def test_order_matters_an_item_listed_late_is_not_found_earlier():
    got, stats = locate(TEXT, [{"text": "3 maggio", "label": "DATA"}, {"text": "69", "label": "ETÀ"}])
    assert [s["label"] for s in got] == ["DATA"] and stats["unlocated"] == 1


@pytest.mark.parametrize("raw", ["", "nota", "{}", '{"text": "a", "label": "NOME"}', '[{"text": "a"}]', '[{"text": 1, "label": "NOME"}]',
                                 '[{"text": "a", "label": "NOME", "extra": 1}]', '["a"]', '[{"text": "a", "label": "NOME"}] coda'])
def test_parse_refuses_anything_but_an_item_array(raw):
    assert parse_extraction(raw) is None


def test_parse_accepts_whitespace_around_the_array():
    assert parse_extraction('\n [{"text": "a", "label": "NOME"}] \n') == [{"text": "a", "label": "NOME"}]


def test_evaluator_scores_located_spans_and_unknown_labels():
    spans, _ = locate(TEXT, [{"text": "69", "label": "ETÀ"}, {"text": "Pisa", "label": "CITTÀ"}])
    r = resolve_prediction(TEXT, "spans", predicted_spans=spans)
    assert r.status == VALID and [s.label for s in r.spans] == ["ETÀ", "UNKNOWN"]
    assert resolve_prediction(TEXT, "spans", predicted_spans=None).status != VALID   # unparseable output is an invalid output


def test_make_target_formats():
    assert make_target(TEXT, [AGE], "extraction") == '[{"text":"69","label":"ETÀ"}]'
    assert json.loads(make_target(TEXT, [AGE], "segments"))[1] == {"text": "69", "label": "ETÀ"}
    with pytest.raises(ValueError):
        make_target(TEXT, [AGE], "inline")


def test_e3x_config_differs_from_llama_only_in_prompt_format_and_folder():
    base, x = load_yaml(ROOT / "configs" / "training.yaml"), load_yaml(ROOT / "configs" / "training_llama_extraction.yaml")
    require_training_decisions(x)
    assert output_format_of(x) == "extraction" and output_format_of(base) == "segments"
    assert x["prompt_version"] == "redaction_extraction_v1" and read_system_prompt(x["prompt_version"])
    strip = lambda c: {k: ({kk: vv for kk, vv in v.items() if kk != "run_folder"} if k == "model" else v) for k, v in c.items() if k not in ("prompt_version", "output_format")}
    assert strip(x) == strip(base)
    e3 = resolve_experiment("E3", load_yaml(EXPERIMENTS_CONFIG))
    assert config_id(base, e3) == "137665080a72" and config_id(x, e3) != config_id(base, e3)
