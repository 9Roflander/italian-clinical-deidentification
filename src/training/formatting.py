"""Exact prompt format and token encoding.

The template is written out here rather than taken from a tokenizer, because the historical project never recorded which
one it used. The RTX preflight compares it with the tokenizer's own template. Nothing is truncated silently: an example
longer than the limit raises an error, so a target can never lose its end without anyone knowing.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import statistics
from dataclasses import dataclass

from src.data.loaders import ROOT

PROMPTS_DIR = ROOT / "prompts"
BEGIN = "<|begin_of_text|>"
EOT = "<|eot_id|>"
PROMPT_TEMPLATE = (BEGIN + "<|start_header_id|>system<|end_header_id|>\n\n{system}" + EOT
                   + "<|start_header_id|>user<|end_header_id|>\n\n{note}" + EOT + "<|start_header_id|>assistant<|end_header_id|>\n\n")
IGNORE_INDEX = -100
TEMPLATE_NAME = "project_llama3_segments_v1"


@dataclass(frozen=True)
class ChatTemplate:
    prompt: str                  # with {system} and {note}
    end: str                     # closes the answer; the loss covers it and inference stops on it
    markers: tuple[str, ...]     # must each be one token for this model
    forbidden: re.Pattern        # special-token syntax a note must not contain


# Gemma 3 has no system role: its own template puts the system text, a blank line and the user text in the first user turn.
GEMMA3_TEMPLATE_NAME = "project_gemma3_segments_v1"
TEMPLATES = {
    TEMPLATE_NAME: ChatTemplate(PROMPT_TEMPLATE, EOT, (BEGIN, "<|start_header_id|>", "<|end_header_id|>", EOT), re.compile(re.escape("<|"))),
    GEMMA3_TEMPLATE_NAME: ChatTemplate("<bos><start_of_turn>user\n{system}\n\n{note}<end_of_turn>\n<start_of_turn>model\n", "<end_of_turn>",
                                       ("<bos>", "<start_of_turn>", "<end_of_turn>"),
                                       re.compile(r"<(?:bos|eos|pad|unk|mask|start_of_turn|end_of_turn|start_of_image|end_of_image|image_soft_token|unused\d+)>")),
}


def get_template(name: str = TEMPLATE_NAME) -> ChatTemplate:
    if name not in TEMPLATES:
        raise ValueError(f"unknown chat template {name}")
    return TEMPLATES[name]


def template_of(cfg: dict) -> str:
    """The template a training configuration pins. Configurations without one predate the choice and used Llama 3."""
    return (cfg["model"].get("chat_template") or {}).get("name", TEMPLATE_NAME)


def template_sha(name: str = TEMPLATE_NAME) -> str:
    return hashlib.sha256(get_template(name).prompt.encode("utf-8")).hexdigest()


class FormatError(ValueError):
    """Messages hold IDs and counts, never text."""


class ExampleTooLong(FormatError):
    pass


def read_system_prompt(version: str, prompts_dir=PROMPTS_DIR) -> str:
    return (pathlib.Path(prompts_dir) / f"{version}.txt").read_text(encoding="utf-8").strip()


def prompt_sha(system: str, template: str = TEMPLATE_NAME) -> str:
    return hashlib.sha256((get_template(template).prompt + "\0" + system).encode("utf-8")).hexdigest()


def format_prompt(system: str, note: str, template: str = TEMPLATE_NAME) -> str:
    t = get_template(template)
    if t.forbidden.search(note):
        raise FormatError("note contains special-token syntax")
    return t.prompt.replace("{system}", system).replace("{note}", note)


def token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def encode_training_example(tokenizer, system: str, note: str, target: str, max_length: int, example_id: str = "?", template: str = TEMPLATE_NAME) -> dict:
    """Loss applies to the target and its end token only. Refuses to truncate."""
    prompt = token_ids(tokenizer, format_prompt(system, note, template))
    answer = token_ids(tokenizer, target + get_template(template).end)
    total = len(prompt) + len(answer)
    if total > max_length:
        raise ExampleTooLong(f"{example_id}: {total} tokens exceed the limit of {max_length}")
    return {"input_ids": prompt + answer, "labels": [IGNORE_INDEX] * len(prompt) + answer, "attention_mask": [1] * total}


def encode_dataset(tokenizer, system: str, examples, max_length: int, template: str = TEMPLATE_NAME) -> list[dict]:
    """All-or-nothing: one example over the limit stops the run and names every long example."""
    too_long, out = [], []
    for e in examples:
        try:
            out.append(encode_training_example(tokenizer, system, e.input_text, e.target_text, max_length, e.example_id, template))
        except ExampleTooLong:
            too_long.append(e.example_id)
    if too_long:
        raise ExampleTooLong(f"{len(too_long)} examples exceed {max_length} tokens: {too_long[:10]}")
    return out


def token_length_report(tokenizer, system: str, examples, template: str = TEMPLATE_NAME) -> dict:
    """Sets the sequence limit from data. IDs and counts only."""
    lengths, end = {}, get_template(template).end
    for e in examples:
        lengths[e.example_id] = len(token_ids(tokenizer, format_prompt(system, e.input_text, template))) + len(token_ids(tokenizer, e.target_text + end))
    values = sorted(lengths.values())
    longest = max(lengths, key=lengths.get) if lengths else None
    return {"examples": len(values), "median": statistics.median(values) if values else None, "p95": values[int(0.95 * (len(values) - 1))] if values else None,
            "max": values[-1] if values else None, "longest_example_id": longest}
