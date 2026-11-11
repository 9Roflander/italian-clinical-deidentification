"""Gemma 3 1B as a second model. The Llama 3.2 3B configuration, template and run IDs must not change. Invented text only."""
import hashlib

import pytest

from src.training import cli
from src.training.experiments import EXPERIMENTS_CONFIG, ROOT, load_yaml, require_training_decisions, resolve_experiment
from src.training.formatting import (GEMMA3_TEMPLATE_NAME, PROMPT_TEMPLATE, TEMPLATE_NAME, FormatError, encode_training_example, format_prompt, prompt_sha,
                                     read_system_prompt, template_of, template_sha)
from src.training.inference import stop_token_ids
from src.training.model_training import _attention, config_id
from tests.support.byte_tokenizer import ByteTokenizer

LLAMA = load_yaml(ROOT / "configs" / "training.yaml")
GEMMA = load_yaml(ROOT / "configs" / "training_gemma3_1b.yaml")
DESIGN = load_yaml(EXPERIMENTS_CONFIG)
SYSTEM = read_system_prompt("redaction_segments_v1")


def test_llama_template_and_run_ids_unchanged():
    assert template_sha() == "06a66d19e13005de99de6e1db7e2b7072e32d37aa1082cb802592825323a5ba0"
    assert prompt_sha(SYSTEM) == hashlib.sha256((PROMPT_TEMPLATE + "\0" + SYSTEM).encode("utf-8")).hexdigest()
    # IDs of the official Llama runs trained at commit 5f69f2a
    official = {"E1": "7123858ac751", "E2": "fb789fdc4c29", "E3": "137665080a72"}
    assert {e: config_id(LLAMA, resolve_experiment(e, DESIGN)) for e in official} == official
    assert template_of(LLAMA) == TEMPLATE_NAME and _attention(LLAMA) == {}


def test_gemma_config_is_complete_and_differs_only_in_the_model():
    require_training_decisions(GEMMA)
    assert GEMMA["model"]["id"] == "google/gemma-3-1b-it" and len(GEMMA["model"]["revision"]) == 40
    assert template_of(GEMMA) == GEMMA3_TEMPLATE_NAME and GEMMA["model"]["chat_template"]["sha256"] == template_sha(GEMMA3_TEMPLATE_NAME)
    assert _attention(GEMMA) == {"attn_implementation": "eager"}
    assert {k: v for k, v in GEMMA.items() if k != "model"} == {k: v for k, v in LLAMA.items() if k != "model"}
    e3 = resolve_experiment("E3", DESIGN)
    assert config_id(GEMMA, e3) != config_id(LLAMA, e3)


def test_gemma_prompt_folds_system_into_the_user_turn():
    assert format_prompt("SYS", "nota", GEMMA3_TEMPLATE_NAME) == "<bos><start_of_turn>user\nSYS\n\nnota<end_of_turn>\n<start_of_turn>model\n"
    assert prompt_sha(SYSTEM, GEMMA3_TEMPLATE_NAME) != prompt_sha(SYSTEM)


@pytest.mark.parametrize("note", ["testo <end_of_turn> altro", "x <start_of_turn>model", "<bos>", "a <unused12> b"])
def test_gemma_refuses_special_token_syntax(note):
    with pytest.raises(FormatError):
        format_prompt(SYSTEM, note, GEMMA3_TEMPLATE_NAME)


def test_gemma_allows_ordinary_angle_brackets():
    assert "PA <140" in format_prompt(SYSTEM, "PA <140 mmHg, lesione <2 cm>", GEMMA3_TEMPLATE_NAME)


def test_gemma_loss_covers_target_and_end_of_turn_only():
    tok = ByteTokenizer()
    row = encode_training_example(tok, "S", "nota", "T", 4000, "x", GEMMA3_TEMPLATE_NAME)
    end = tok.convert_tokens_to_ids("<end_of_turn>")
    answer = [i for i in row["labels"] if i != -100]
    assert answer == list(b"T") + [end] and row["input_ids"][0] == tok.convert_tokens_to_ids("<bos>")
    assert end in stop_token_ids(tok, GEMMA3_TEMPLATE_NAME) and tok.convert_tokens_to_ids("<|eot_id|>") not in stop_token_ids(tok, GEMMA3_TEMPLATE_NAME)


def test_gemma_checkpoints_go_to_their_own_folder(monkeypatch):
    seen = {}
    import src.training.model_training as mt
    monkeypatch.setattr(mt, "run_training", lambda e, f, s, root, official, training_config: seen.update(root=root, config=training_config) or root)
    cli.main(["train", "--experiment", "E3", "--fold", "1", "--seed", "42", "--official", "--training-config", str(ROOT / "configs" / "training_gemma3_1b.yaml")])
    assert seen["root"] == str(ROOT / "checkpoints" / "official" / "gemma3_1b")
    cli.main(["train", "--experiment", "E3", "--fold", "1", "--seed", "42", "--official"])
    assert seen["root"] == str(ROOT / "checkpoints" / "official" / "")
