"""Train the encoder baseline on one fold, then label that fold's held-out notes. One process does both.

    python -m src.encoder.run --experiment E3 --fold 1 --seed 42 --official

Training examples are exactly the LLM arm's (src/training/dataset.py); their spans are read back from the segment targets
by the evaluator's own reader. Long notes are split into overlapping 512-token windows. Predictions are character spans in
the format the evaluator scores (`spans`), with the same manifest fields as the LLM folds, so merge and evaluation are shared.
Outputs hold IDs, positions, labels and numbers only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import shutil
import tempfile
import time

from src.data.loaders import ROOT, file_sha256
from src.data.splits import LeakageError, load_fold
from src.encoder import ENCODER_PIPELINE_VERSION
from src.encoder.bio import IGNORE, LABEL_ID, LABELS, decode, merge_windows, token_labels, windows
from src.evaluation.alignment import VALID, resolve_segments
from src.infrastructure import INFRASTRUCTURE_VERSION
from src.infrastructure.provenance import check_source_integrity, run_environment
from src.training.dataset import build_fold_dataset
from src.training.experiments import EXPERIMENTS_CONFIG, load_yaml, resolve_experiment
from src.training.inference import load_held_out_notes
from src.training.model_training import folder_sha256, package_versions
from src.training.targets import text_sha256

ENCODER_CONFIG = ROOT / "configs" / "encoder_bert_it.yaml"


class EncoderRunError(RuntimeError):
    """Messages hold IDs and counts, never text."""


def encoder_config_id(cfg: dict, experiment: dict) -> str:
    keep = {k: v for k, v in cfg.items() if k != "training"} | {"training": {k: v for k, v in cfg["training"].items() if k != "seeds"}}
    keep["model"] = {k: cfg["model"][k] for k in ("id", "revision")}
    return hashlib.sha256(json.dumps({"config": keep, "experiment": experiment, "kind": "encoder_bio"}, sort_keys=True).encode()).hexdigest()[:12]


def tokenize(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]


def example_spans(example) -> list[tuple[int, int, str]]:
    resolved = resolve_segments(example.input_text, example.target_text)
    if resolved.status != VALID:
        raise EncoderRunError(f"{example.example_id}: target does not read back ({resolved.status})")
    return [(s.start, s.end, s.label) for s in resolved.spans]


def training_items(tokenizer, examples, size: int, stride: int) -> tuple[list[dict], dict]:
    """One item per window: input ids with [CLS]/[SEP] and BIO labels (special tokens ignored by the loss)."""
    items, totals = [], {"examples": 0, "windows": 0, "spans": 0, "spans_misaligned": 0, "spans_without_tokens": 0, "max_tokens": 0}
    for e in examples:
        ids, offsets = tokenize(tokenizer, e.input_text)
        labels, stats = token_labels(offsets, example_spans(e))
        totals["examples"] += 1
        totals["max_tokens"] = max(totals["max_tokens"], len(ids))
        for k in ("spans", "spans_misaligned", "spans_without_tokens"):
            totals[k] += stats[k]
        for a, b in windows(len(ids), size, stride):
            items.append({"input_ids": [tokenizer.cls_token_id] + ids[a:b] + [tokenizer.sep_token_id], "labels": [IGNORE] + labels[a:b] + [IGNORE]})
            totals["windows"] += 1
    return items, totals


def _batch(rows, pad_id: int, device):
    import torch
    n = max(len(r["input_ids"]) for r in rows)
    ids = [r["input_ids"] + [pad_id] * (n - len(r["input_ids"])) for r in rows]
    mask = [[1] * len(r["input_ids"]) + [0] * (n - len(r["input_ids"])) for r in rows]
    labels = [r["labels"] + [IGNORE] * (n - len(r["labels"])) for r in rows]
    return {k: torch.tensor(v, device=device) for k, v in (("input_ids", ids), ("attention_mask", mask), ("labels", labels))}


def batch_order(n_items: int, t: dict, seed: int) -> list[list[int]]:
    """Batches of window indices. `epochs` (config v1): whole passes, a new shuffle per pass, the last batch of a pass may be short.
    `updates` (config v2): exactly that many full batches for every arm, drawn from passes with a new shuffle each, as in the LLM loop."""
    bs = t["batch_size"]
    if "updates" in t:
        flat, cycle = [], 0
        while len(flat) < t["updates"] * bs:
            order = list(range(n_items))
            random.Random(f"{seed}:{cycle}").shuffle(order)
            flat += order
            cycle += 1
        return [flat[k * bs:(k + 1) * bs] for k in range(t["updates"])]
    batches = []
    for epoch in range(t["epochs"]):
        order = list(range(n_items))
        random.Random(f"{seed}:{epoch}").shuffle(order)
        batches += [order[k:k + bs] for k in range(0, len(order), bs)]
    return batches


def train(model, items, t: dict, seed: int, pad_id: int, device: str, log=print) -> list[float]:
    """Updates over shuffled windows as set by `batch_order`. Returns the loss of each update."""
    import torch
    from transformers import get_linear_schedule_with_warmup
    if t["scheduler"] != "linear":
        raise ValueError("only the linear schedule is implemented, as configured")
    random.seed(seed)
    torch.manual_seed(seed)
    batches = batch_order(len(items), t, seed)
    total = len(batches)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": t["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=t["learning_rate"])
    schedule = get_linear_schedule_with_warmup(optimizer, round(t["warmup_ratio"] * total), total)
    amp = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    model.to(device).train()
    losses, started, step = [], time.time(), 0
    for indices in batches:
        batch = _batch([items[i] for i in indices], pad_id, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss = model(**batch).loss
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite loss at update {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(value)
        step += 1
        if step % 20 == 0 or step == total:
            log(f"step {step}/{total} loss {value:.4f} elapsed_min {(time.time() - started) / 60:.1f}")
    return losses


def predict_spans(model, tokenizer, text: str, size: int, stride: int, device: str) -> tuple[list[dict], dict]:
    import torch
    ids, offsets = tokenize(tokenizer, text)
    spans_w = windows(len(ids), size, stride)
    preds = []
    model.eval()
    with torch.no_grad():
        for a, b in spans_w:
            x = torch.tensor([[tokenizer.cls_token_id] + ids[a:b] + [tokenizer.sep_token_id]], device=device)
            preds.append(model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0, 1:-1].argmax(-1).tolist())
    labels = merge_windows(len(ids), spans_w, preds)
    spans = [{"start": s, "end": e, "label": c} for s, e, c in decode(offsets, labels)]
    return spans, {"tokens": len(ids), "windows": len(spans_w), "token_label_ids": labels}


def _write_jsonl(path: pathlib.Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(experiment_id: str, fold: int, seed: int, official: bool = False, config_path=ENCODER_CONFIG, model=None, tokenizer=None, device=None,
        checkpoint_root=None, predictions_root=None, dataset=None, notes=None, spec=None, log=print) -> dict:
    """Train, save, and predict one fold. `model`, `tokenizer`, `dataset`, `notes` and `spec` are injectable for tests."""
    import torch
    cfg = load_yaml(config_path)
    if experiment_id not in cfg["data"]["experiments"]:
        raise EncoderRunError(f"{experiment_id} is not an encoder arm: {cfg['data']['experiments']}")
    experiment = resolve_experiment(experiment_id, load_yaml(EXPERIMENTS_CONFIG))
    integrity = check_source_integrity(ROOT)
    if official and not integrity.clean:
        raise EncoderRunError(f"official runs need committed source. {integrity.summary()}")
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cid = encoder_config_id(cfg, experiment)
    name = f"{experiment_id}-BERT"
    ckpt = pathlib.Path(checkpoint_root or ROOT / "checkpoints" / ("official" if official else "development") / cfg["model"]["run_folder"]) / experiment_id / f"fold_{fold}" / f"seed_{seed}"
    pred = pathlib.Path(predictions_root or ROOT / "predictions_secure") / f"{name}__{cid}__seed{seed}" / f"fold_{fold}"
    for p in (ckpt, pred):
        if p.exists():
            raise EncoderRunError(f"{p} already exists; runs are immutable")
    spec = spec or load_fold(fold)
    dataset = dataset or build_fold_dataset(experiment_id, fold, cfg["data"]["gold_oversample"], experiment.get("synthetic"), require_official=official)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"], revision=cfg["model"]["revision"], use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise EncoderRunError("a fast tokenizer is needed for character offsets")
    w = cfg["windows"]
    items, data_stats = training_items(tokenizer, dataset.examples, w["size"], w["stride"])
    if model is None:
        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(cfg["model"]["id"], revision=cfg["model"]["revision"], num_labels=len(LABELS),
                                                                id2label=dict(enumerate(LABELS)), label2id=LABEL_ID)
    log(f"encoder {name} fold {fold} seed {seed}: {data_stats['examples']} examples, {data_stats['windows']} windows, "
        f"{data_stats['spans_misaligned']} of {data_stats['spans']} spans not on token edges")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    losses = train(model, items, cfg["training"], seed, tokenizer.pad_token_id, device, log)
    train_minutes = (time.time() - started) / 60
    ckpt.mkdir(parents=True)
    keep_seeds = cfg["training"].get("save_model_seeds")   # v2 keeps weights for listed seeds only; the hash is recorded for all
    saved = keep_seeds is None or seed in keep_seeds
    model_dir = ckpt / "model" if saved else pathlib.Path(tempfile.mkdtemp()) / "model"
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    model_sha = folder_sha256(model_dir)
    if not saved:
        shutil.rmtree(model_dir.parent)
    (ckpt / "dataset_manifest.json").write_text(json.dumps(dataset.manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    manifest = {"pipeline": "encoder_bio", "encoder_pipeline_version": ENCODER_PIPELINE_VERSION, "infrastructure_version": INFRASTRUCTURE_VERSION,
                "experiment": name, "data_experiment": experiment_id, "fold": fold, "seed": seed, "official_run": official, "config_id": cid, "config": cfg,
                "dataset_sha256": dataset.manifest["dataset_sha256"], "data": data_stats, "updates": len(losses), "losses": losses, "train_minutes": round(train_minutes, 2),
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None, "model_sha256": model_sha, "model_saved": saved,
                "source_integrity": integrity.to_manifest(), "packages": package_versions(), "environment": run_environment()}
    (ckpt / "training_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    notes = notes if notes is not None else load_held_out_notes(spec)
    if {n.note_id for n in notes} != set(spec.test_ids):
        raise LeakageError(f"fold {fold}: inference must cover exactly its held-out notes")
    meta = {"experiment": name, "model": cfg["model"]["id"], "model_revision": cfg["model"]["revision"], "checkpoint": f"{name}/seed_{seed}", "config_id": cid, "seed": seed}
    records, raws = [], []
    for n in sorted(notes, key=lambda n: n.note_id):
        spans, raw = predict_spans(model, tokenizer, n.text, w["size"], w["stride"], device)
        raws.append({"note_id": n.note_id, "fold_id": fold, **raw})
        records.append({**meta, "note_id": n.note_id, "fold_id": fold, "input_sha256": text_sha256(n.text), "output_format": "spans", "model_output_format": "bio_tokens",
                        "predicted_spans": spans, "completion_status": "complete", "finish_reason": "stop", "truncation_flag": False, "parse_status": "located",
                        "windows": raw["windows"]})
    pred.mkdir(parents=True)
    _write_jsonl(pred / "raw_predictions.jsonl", raws)
    _write_jsonl(pred / "predictions.jsonl", records)
    pm = {**meta, "fold": fold, "gold_sha256": spec.gold_sha256, "held_out_note_ids": sorted(spec.test_ids), "records": len(records),
          "raw_predictions_sha256": file_sha256(pred / "raw_predictions.jsonl"), "predictions_sha256": file_sha256(pred / "predictions.jsonl"),
          "output_format": "spans", "model_output_format": "bio_tokens", "model_sha256": manifest["model_sha256"], "packages": package_versions(), "environment": run_environment()}
    (pred / "prediction_manifest.json").write_text(json.dumps(pm, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    log(f"done: train {train_minutes:.1f} min, {len(records)} held-out notes labelled")
    return {"checkpoint": str(ckpt), "predictions": str(pred), "updates": len(losses), "train_minutes": round(train_minutes, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--config", default=str(ENCODER_CONFIG))
    a = ap.parse_args(argv)
    print(json.dumps(run(a.experiment, a.fold, a.seed, a.official, a.config, log=lambda s: print(s, flush=True)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
