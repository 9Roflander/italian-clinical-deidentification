"""End-to-end mechanical test on a tiny, randomly initialized model and invented notes. It learns nothing useful.

    .venv/bin/python scripts/smoke_tiny_pipeline.py
    python scripts/smoke_tiny_pipeline.py --template project_gemma3_segments_v1   # tiny Gemma 3; needs transformers >= 4.50
    .venv/bin/python scripts/smoke_tiny_pipeline.py --output-format extraction      # E3-X item list and locator

Exercises real PyTorch, Transformers and PEFT code on CPU: LoRA, the exact step budget, adapter saving and reloading, greedy
inference, raw output preservation, and the evaluator. It does not test quantization, CUDA or memory, which only the RTX
preflight can do. Prints counts and hashes only.
"""
import argparse
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import LlamaConfig, LlamaForCausalLM

from src.data.loaders import GoldNote
from src.data.splits import FoldSpec
from src.evaluation.alignment import render_inline
from src.evaluation.gold_verification import VerifiedGold
from src.evaluation.runner import evaluate_records
from src.training.formatting import GEMMA3_TEMPLATE_NAME, TEMPLATE_NAME, encode_training_example, read_system_prompt
from src.training.inference import HFGenerator, merge_out_of_fold, run_fold_inference, stop_token_ids
from src.training.model_training import attach_lora, budget_report, folder_sha256, train_steps, trainable_parameters
from src.training.dataset import make_target
from tests.support.byte_tokenizer import ByteTokenizer

LORA = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM", "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
SETTINGS = {"batch_size": 1, "gradient_accumulation_steps": 2, "learning_rate": 1e-3, "scheduler": "cosine", "warmup_steps": 1, "max_grad_norm": 1.0}
STEPS = 6


def tiny_model(template: str = TEMPLATE_NAME):
    torch.manual_seed(0)
    if template == GEMMA3_TEMPLATE_NAME:
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig
        return Gemma3ForCausalLM(Gemma3TextConfig(vocab_size=ByteTokenizer.vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                                  num_key_value_heads=1, head_dim=16, max_position_embeddings=4096, sliding_window=64, attn_implementation="eager",
                                                  eos_token_id=ByteTokenizer.eos_token_id, bos_token_id=256, pad_token_id=ByteTokenizer.eos_token_id))
    return LlamaForCausalLM(LlamaConfig(vocab_size=ByteTokenizer.vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=4096,
                                        eos_token_id=ByteTokenizer.eos_token_id, bos_token_id=256, pad_token_id=ByteTokenizer.eos_token_id))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=TEMPLATE_NAME, choices=[TEMPLATE_NAME, GEMMA3_TEMPLATE_NAME])
    parser.add_argument("--output-format", default="segments", choices=["segments", "extraction"])
    args = parser.parse_args(argv)
    template, output_format = args.template, args.output_format
    tokenizer, system = ByteTokenizer(), read_system_prompt("redaction_segments_v1")
    notes = {}
    for k in range(4):
        text = f"Zelinda{chr(97 + k)} ha {30 + k}  anni."
        spans = [(0, 8, "NOME"), (12, 14, "ETÀ")]
        notes[f"mini_{k}"] = (text, spans)
    encoded = [encode_training_example(tokenizer, system, t, make_target(t, s, output_format), 4000, i, template) for i, (t, s) in list(notes.items())[:2]]

    def train_once():
        model = attach_lora(tiny_model(template), LORA)
        trained, total = trainable_parameters(model)
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        steps = []
        losses = train_steps(model, encoded, SETTINGS, STEPS, 42, "cpu", on_step=lambda s, _: steps.append(s))
        moved = any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
        still = all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if not p.requires_grad)
        return model, losses, trained, total, moved, still, steps

    model, losses, trained, total, moved, still, steps = train_once()
    _, again, *_ = train_once()
    checks = {"exact_optimizer_steps": steps == list(range(1, STEPS + 1)) and len(losses) == STEPS, "only_lora_parameters_train": 0 < trained < total, "lora_parameters_moved": moved,
              "base_parameters_unchanged": still, "same_seed_same_losses": losses == again, "losses_finite": all(x == x for x in losses)}
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        model.save_pretrained(str(base / "adapter"))
        sha_first, sha_second = folder_sha256(base / "adapter"), folder_sha256(base / "adapter")
        reloaded = PeftModel.from_pretrained(tiny_model(template), str(base / "adapter"))
        reloaded.eval()
        test_ids = ("mini_2", "mini_3")
        spec = FoldSpec(1, ("mini_0", "mini_1"), test_ids, "tiny", 42)
        gold_notes = [GoldNote(i, n, notes[i][0], tuple({"start": a, "end": b, "type": c, "text": notes[i][0][a:b], "annotated_text": notes[i][0][a:b]} for a, b, c in notes[i][1]),
                               redacted_text=render_inline(*notes[i])) for n, i in enumerate(test_ids)]
        meta = {"experiment": "SMOKE", "model": "tiny", "model_revision": "none", "checkpoint": "SMOKE/seed_42", "config_id": "smoke", "seed": 42}
        out = run_fold_inference(HFGenerator(reloaded, stop_token_ids(tokenizer, template)), tokenizer, system, gold_notes, spec, meta, {"max_new_tokens_factor": 0.5, "max_new_tokens_margin": 4},
                                 base / "fold_1", template=template, output_format=output_format)
        merged = base / "merged.jsonl"
        summary = merge_out_of_fold([out], merged)
        rows = [json.loads(x) for x in merged.read_text(encoding="utf-8").splitlines()]
        raw = [json.loads(x) for x in (out / "raw_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        gold = VerifiedGold({n.note_id: n for n in gold_notes}, test_ids, {i: 1 for i in test_ids}, {i: notes[i][1] for i in test_ids}, "tiny", "v2", {1: "x"})
        result = evaluate_records(rows, gold)
        checks |= {"adapter_hash_stable": sha_first == sha_second, "raw_output_saved_for_every_note": len(raw) == 2 and all("generated_token_ids" in r for r in raw),
                   "raw_matches_prediction": [r["raw_model_output"] for r in raw] == [r["raw_model_output"] for r in rows],
                   "evaluator_accepts_every_record": len(result.scores) == 2 and summary["records"] == 2,
                   "untrained_output_is_invalid_not_credited": all(s.status != "valid" and s.counts["tp_NOME"] == 0 for s in result.scores)}
    print(json.dumps({"template": template, "output_format": output_format, "versions": {"torch": torch.__version__}, "budget": budget_report(2, STEPS, 2), "lora": [trained, total], "statuses": [s.status for s in result.scores], "checks": checks}, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
