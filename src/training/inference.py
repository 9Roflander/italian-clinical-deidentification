"""Greedy inference on one fold's held-out notes.

The raw output is written and flushed before anything else looks at it. It is never stripped, cleaned or repaired. The
prediction records follow the evaluator's input format, and the evaluator alone decides validity and scores.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import pathlib

from src.data.loaders import DEFAULT_GOLD, file_sha256, load_gold
from src.data.splits import FoldSpec, LeakageError
from src.evaluation.alignment import resolve_segments
from src.training.formatting import TEMPLATE_NAME, format_prompt, get_template, token_ids
from src.training.extraction import FORMAT_NAME as EXTRACTION_FORMAT, locate, parse_extraction
from src.training.targets import text_sha256

RUN_FIELDS = ("experiment", "model", "model_revision", "checkpoint", "config_id", "seed")


class InferenceError(RuntimeError):
    """Messages hold IDs and counts, never text."""


def load_held_out_notes(spec: FoldSpec, gold_path=DEFAULT_GOLD) -> list:
    """The only training-package function that reads held-out text. Inference needs it, and nothing here feeds back into training."""
    if file_sha256(gold_path) != spec.gold_sha256:
        raise LeakageError("gold file does not match the frozen fold manifest")
    keep = set(spec.test_ids)
    notes = [n for n in load_gold(gold_path) if n.note_id in keep]
    if {n.note_id for n in notes} != keep:
        raise LeakageError(f"fold {spec.fold}: held-out notes are missing")
    return notes


def stop_token_ids(tokenizer, template: str = TEMPLATE_NAME) -> set[int]:
    ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(get_template(template).end)}
    return {i for i in ids if isinstance(i, int) and i >= 0}


def max_new_tokens_for(n_input_tokens: int, factor: float, margin: int) -> int:
    return math.ceil(n_input_tokens * factor) + margin


def check_limit_covers_targets(tokenizer, examples, settings: dict, template: str = TEMPLATE_NAME) -> dict:
    """Every training target must fit under the inference limit. Otherwise a correct answer would be cut off and scored as invalid.
    JSON overhead is per segment, so it is not proportional to note length. Uses training examples only."""
    tightest, tightest_id, over, end = None, None, [], get_template(template).end
    for e in examples:
        room = max_new_tokens_for(len(token_ids(tokenizer, e.input_text)), settings["max_new_tokens_factor"], settings["max_new_tokens_margin"]) - len(token_ids(tokenizer, e.target_text + end))
        if tightest is None or room < tightest:
            tightest, tightest_id = room, e.example_id
        if room < 0:
            over.append(e.example_id)
    if over:
        raise InferenceError(f"{len(over)} training targets exceed the inference token limit: {over[:10]}")
    return {"smallest_headroom_tokens": tightest, "tightest_example_id": tightest_id}


class HFGenerator:
    """Greedy decoding through Hugging Face, with sampling off."""

    def __init__(self, model, stop_ids):
        self.model, self.stop_ids = model, sorted(stop_ids)

    def generate(self, prompt_ids, max_new_tokens: int) -> list[int]:
        import torch
        inputs = torch.tensor([prompt_ids], device=next(self.model.parameters()).device)
        with torch.no_grad():
            out = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                      temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, eos_token_id=self.stop_ids, pad_token_id=self.stop_ids[0], use_cache=True)
        return out[0][inputs.shape[1] :].tolist()


def _append(path: pathlib.Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def predict_note(generator, tokenizer, system: str, note, fold: int, meta: dict, settings: dict, raw_path: pathlib.Path, template: str = TEMPLATE_NAME,
                 output_format: str = "segments") -> dict:
    prompt = token_ids(tokenizer, format_prompt(system, note.text, template))
    note_tokens = len(token_ids(tokenizer, note.text))
    limit = max_new_tokens_for(note_tokens, settings["max_new_tokens_factor"], settings["max_new_tokens_margin"])
    new = list(generator.generate(prompt, limit))
    stops = stop_token_ids(tokenizer, template)
    stopped = bool(new) and new[-1] in stops
    content = new[:-1] if stopped else new
    finish = "stop" if stopped else "length"
    raw = tokenizer.decode(content, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # Saved first, so a crash or a parsing bug cannot lose or alter what the model produced.
    _append(raw_path, {"note_id": note.note_id, "fold_id": fold, "generated_token_ids": new, "raw_model_output": raw, "finish_reason": finish})
    if output_format == "extraction":
        return _extraction_record(note, fold, meta, raw, new, prompt, limit, stopped, finish)
    resolved = resolve_segments(note.text, raw)
    return {**meta, "note_id": note.note_id, "fold_id": fold, "input_sha256": text_sha256(note.text), "output_format": "segments", "raw_model_output": raw,
            "generation_parameters": {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0, "max_new_tokens": limit, "whitespace_cleanup": False},
            "completion_status": "complete" if stopped else "truncated", "finish_reason": finish, "truncation_flag": not stopped,
            "prompt_tokens": len(prompt), "generated_tokens": len(new), "parse_status": resolved.status}


def _extraction_record(note, fold: int, meta: dict, raw: str, new: list, prompt: list, limit: int, stopped: bool, finish: str) -> dict:
    """E3-X: parse the item list and place it with the fixed locator. The evaluator scores the located spans."""
    items = parse_extraction(raw)
    spans, stats = locate(note.text, items) if items is not None else (None, None)
    return {**meta, "note_id": note.note_id, "fold_id": fold, "input_sha256": text_sha256(note.text), "output_format": "spans", "model_output_format": EXTRACTION_FORMAT,
            "raw_model_output": raw, "predicted_spans": spans, "locator": stats,
            "generation_parameters": {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0, "max_new_tokens": limit, "whitespace_cleanup": False},
            "completion_status": "complete" if stopped else "truncated", "finish_reason": finish, "truncation_flag": not stopped,
            "prompt_tokens": len(prompt), "generated_tokens": len(new),
            "parse_status": "unparseable" if items is None else ("unlocated_items" if stats["unlocated"] else "located")}


def run_fold_inference(generator, tokenizer, system: str, notes, spec: FoldSpec, meta: dict, settings: dict, out_dir, extra_manifest=None,
                       template: str = TEMPLATE_NAME, output_format: str = "segments") -> pathlib.Path:
    """Predict every held-out note in a new folder. Notes outside this fold's held-out set are refused."""
    absent = [k for k in RUN_FIELDS if k not in meta]
    if absent:
        raise InferenceError(f"missing metadata: {absent}")
    if {n.note_id for n in notes} != set(spec.test_ids):
        raise LeakageError(f"fold {spec.fold}: inference must cover exactly its held-out notes")
    out = pathlib.Path(out_dir)
    if out.exists():
        raise InferenceError(f"{out.name} already exists; predictions are immutable")
    out.mkdir(parents=True)
    raw_path, prediction_path = out / "raw_predictions.jsonl", out / "predictions.jsonl"
    statuses, finishes = collections.Counter(), collections.Counter()
    for note in sorted(notes, key=lambda n: n.note_id):
        record = predict_note(generator, tokenizer, system, note, spec.fold, meta, settings, raw_path, template, output_format)
        _append(prediction_path, record)
        statuses[record["parse_status"]] += 1
        finishes[record["finish_reason"]] += 1
    manifest = {**meta, "fold": spec.fold, "gold_sha256": spec.gold_sha256, "held_out_note_ids": sorted(spec.test_ids), "records": len(notes),
                "raw_predictions_sha256": file_sha256(raw_path), "predictions_sha256": file_sha256(prediction_path),
                "output_format": "spans" if output_format == "extraction" else "segments", **({"model_output_format": EXTRACTION_FORMAT} if output_format == "extraction" else {}),
                "parse_status_counts": dict(statuses), "finish_reason_counts": dict(finishes), "settings": settings, **(extra_manifest or {})}
    (out / "prediction_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return out


def merge_out_of_fold(fold_dirs, out_path) -> dict:
    """Combine fold predictions for the evaluator. Each note appears once, from its own fold, under one run's metadata."""
    rows, meta, folds = [], None, set()
    for d in fold_dirs:
        d = pathlib.Path(d)
        manifest = json.loads((d / "prediction_manifest.json").read_text(encoding="utf-8"))
        if file_sha256(d / "predictions.jsonl") != manifest["predictions_sha256"] or file_sha256(d / "raw_predictions.jsonl") != manifest["raw_predictions_sha256"]:
            raise InferenceError(f"fold {manifest['fold']}: files differ from the prediction manifest")
        if manifest["fold"] in folds:
            raise InferenceError(f"fold {manifest['fold']} supplied twice")
        folds.add(manifest["fold"])
        this = {k: manifest[k] for k in RUN_FIELDS}
        if meta is None:
            meta = this
        elif this != meta:
            raise InferenceError("all folds must come from one experiment, model, configuration and seed")
        for line in (d / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["fold_id"] != manifest["fold"] or row["note_id"] not in set(manifest["held_out_note_ids"]):
                raise LeakageError(f"{row['note_id']} is not a held-out note of fold {manifest['fold']}")
            rows.append(row)
    ids = [r["note_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise InferenceError("a note has more than one prediction")
    out = pathlib.Path(out_path)
    if out.exists():
        raise InferenceError(f"{out.name} already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: r["note_id"])
    out.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return {"folds": sorted(folds), "records": len(rows), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
