"""End-to-end mechanical test of the encoder baseline on a tiny random BERT, a tiny tokenizer and invented notes, on CPU.

    .venv/bin/python scripts/smoke_tiny_encoder.py
    .venv/bin/python scripts/smoke_tiny_encoder.py --budget updates   # config v2: fixed updates, weights not kept for this seed

Exercises windows (notes longer than the window), training, saving, held-out labelling, the shared fold merge and the
evaluator. It learns nothing useful and needs no download. Prints counts only.
"""
import argparse
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast

from src.data.loaders import GoldNote
from src.data.splits import FoldSpec
from src.encoder.bio import LABEL_ID, LABELS
from src.encoder.run import run
from src.evaluation.alignment import render_inline
from src.evaluation.gold_verification import VerifiedGold
from src.evaluation.runner import evaluate_records
from src.training.dataset import Example, FoldDataset, make_target
from src.training.inference import merge_out_of_fold

CFG = {"model": {"id": "tiny", "revision": "none", "run_folder": "bert_tiny"}, "data": {"experiments": ["E1"], "gold_oversample": 1},
       "windows": {"size": 12, "stride": 4},
       "training": {"epochs": 3, "batch_size": 2, "learning_rate": 1e-3, "weight_decay": 0.01, "warmup_ratio": 0.1, "scheduler": "linear", "max_grad_norm": 1.0, "seeds": [42]}}


def tiny_tokenizer():
    chars = sorted(set("abcdefghijklmnopqrstuvwxyzàèéìòùABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,'"))
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars])}
    tok = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=False, clean_text=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def note(i):
    text = f"Zelinda ha {30 + i}  anni, vive a Pisa dal 3 maggio."
    at = lambda s, t: (text.index(s), text.index(s) + len(s), t)
    return text, [at("Zelinda", "NOME"), at(str(30 + i), "ETÀ"), at("Pisa", "LUOGO/INDIRIZZO"), at("3 maggio", "DATA")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", choices=["epochs", "updates"], default="epochs")
    budget = ap.parse_args(argv).budget
    cfg = json.loads(json.dumps(CFG))
    if budget == "updates":
        cfg["training"].pop("epochs")
        cfg["training"] |= {"updates": 7, "save_model_seeds": [1]}
    tokenizer = tiny_tokenizer()
    torch.manual_seed(0)
    model = BertForTokenClassification(BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                                                  max_position_embeddings=64, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL_ID))
    train_ids, test_ids = ("t0", "t1", "t2"), ("h0", "h1")
    data = {k: note(i) for i, k in enumerate(train_ids + test_ids)}
    examples = [Example(k, "gold", k, 0, "gold", data[k][0], make_target(*data[k])) for k in train_ids]
    dataset = FoldDataset("E1", 1, examples, {"dataset_sha256": "tiny"})
    notes = [GoldNote(k, n, data[k][0], tuple({"start": a, "end": b, "type": c, "text": data[k][0][a:b], "annotated_text": data[k][0][a:b]} for a, b, c in data[k][1]),
                      redacted_text=render_inline(*data[k])) for n, k in enumerate(test_ids)]
    spec = FoldSpec(1, train_ids, test_ids, "tiny", 42)
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        cfg_path = base / "encoder.yaml"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")   # JSON is valid YAML
        out = run("E1", 1, 42, False, cfg_path, model, tokenizer, "cpu", base / "ckpt", base / "pred", dataset, notes, spec, log=lambda s: None)
        manifest = json.loads((pathlib.Path(out["checkpoint"]) / "training_manifest.json").read_text())
        merged = base / "merged.jsonl"
        summary = merge_out_of_fold([out["predictions"]], merged)
        rows = [json.loads(x) for x in merged.read_text().splitlines()]
        gold = VerifiedGold({n.note_id: n for n in notes}, test_ids, {i: 1 for i in test_ids}, {i: data[i][1] for i in test_ids}, "tiny", "v2", {1: "x"})
        result = evaluate_records(rows, gold)
        try:
            run("E1", 1, 42, False, cfg_path, model, tokenizer, "cpu", base / "ckpt", base / "pred", dataset, notes, spec, log=lambda s: None)
            immutable = False
        except RuntimeError:
            immutable = True
        checks = {"windows_used_for_long_notes": manifest["data"]["windows"] > manifest["data"]["examples"], "spans_on_token_edges": manifest["data"]["spans_misaligned"] == 0,
                  "updates_as_planned": manifest["updates"] == (7 if budget == "updates" else 3 * -(-manifest["data"]["windows"] // 2)),
                  "losses_finite": all(x == x for x in manifest["losses"]),
                  "model_saved_as_configured": (pathlib.Path(out["checkpoint"]) / "model" / "config.json").exists() == (budget == "epochs") == manifest["model_saved"]
                                               and len(manifest["model_sha256"]) == 64, "merge_accepts_folds": summary["records"] == 2,
                  "evaluator_scores_every_note": len(result.scores) == 2, "outputs_are_valid_spans": all(s.status == "valid" for s in result.scores),
                  "runs_are_immutable": immutable}
    print(json.dumps({"budget": budget, "data": manifest["data"], "updates": manifest["updates"], "statuses": [s.status for s in result.scores], "checks": checks}, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
