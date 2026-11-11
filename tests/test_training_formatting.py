"""Tokenization, loss masking and the no-truncation rule, using a byte tokenizer."""
import pytest

from src.training.dataset import Example
from src.training.formatting import IGNORE_INDEX, EOT, ExampleTooLong, FormatError, encode_dataset, encode_training_example, format_prompt, prompt_sha, read_system_prompt, token_length_report
from src.training.targets import build_segments, serialize_target
from tests.support.byte_tokenizer import ByteTokenizer

TOK = ByteTokenizer()
SYSTEM = "Istruzioni di prova."
NOTE = "Zelinda ha  45 anni, Città."
TARGET = serialize_target(build_segments(NOTE, [(0, 7, "NOME")]))


def example(i="a", note=NOTE):
    return Example(i, "gold", i, 0, "gold", note, serialize_target(build_segments(note, [])))


def test_loss_applies_only_to_the_target_and_its_end_token():
    row = encode_training_example(TOK, SYSTEM, NOTE, TARGET, 5000)
    prompt_len = len(TOK(format_prompt(SYSTEM, NOTE))["input_ids"])
    assert row["labels"][:prompt_len] == [IGNORE_INDEX] * prompt_len
    answer = row["labels"][prompt_len:]
    assert IGNORE_INDEX not in answer and answer == row["input_ids"][prompt_len:]
    assert TOK.decode(answer, skip_special_tokens=False, clean_up_tokenization_spaces=False) == TARGET + EOT
    assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])


def test_prompt_contains_the_note_exactly_including_double_spaces():
    ids = TOK(format_prompt(SYSTEM, NOTE))["input_ids"]
    assert NOTE in TOK.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def test_prompt_holds_no_target():
    assert "NOME" not in format_prompt(SYSTEM, NOTE) and '"label"' not in format_prompt(SYSTEM, NOTE)


def test_an_example_over_the_limit_is_refused_not_truncated():
    row = encode_training_example(TOK, SYSTEM, NOTE, TARGET, 5000)
    assert encode_training_example(TOK, SYSTEM, NOTE, TARGET, len(row["input_ids"]))["input_ids"] == row["input_ids"]
    with pytest.raises(ExampleTooLong, match="exceed"):
        encode_training_example(TOK, SYSTEM, NOTE, TARGET, len(row["input_ids"]) - 1)


def test_one_long_example_stops_the_dataset_and_is_named():
    short, long = example("short"), example("long", "parola " * 400 + "fine")
    limit = len(encode_training_example(TOK, SYSTEM, short.input_text, short.target_text, 99999)["input_ids"])
    with pytest.raises(ExampleTooLong, match="long"):
        encode_dataset(TOK, SYSTEM, [short, long], limit)


def test_errors_hold_no_note_text():
    with pytest.raises(ExampleTooLong) as e:
        encode_training_example(TOK, SYSTEM, "ZZSENTINEL resto", "x", 2, "id1")
    assert "ZZSENTINEL" not in str(e.value) and "id1" in str(e.value)


def test_notes_with_special_token_syntax_are_refused():
    with pytest.raises(FormatError):
        format_prompt(SYSTEM, "testo <|eot_id|> altro")


def test_length_report():
    report = token_length_report(TOK, SYSTEM, [example("a"), example("b", "parola " * 50 + "fine")])
    assert report["examples"] == 2 and report["longest_example_id"] == "b" and report["max"] > report["median"] - 1


def test_prompt_hash_tracks_the_system_prompt():
    assert prompt_sha("a") == prompt_sha("a") != prompt_sha("b")


def test_the_real_system_prompt_is_versioned_and_names_every_label():
    text = read_system_prompt("redaction_segments_v1")
    for label in ("NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO", '"O"', '"text"', '"label"'):
        assert label in text
    assert not any(ch.isdigit() for ch in text)
