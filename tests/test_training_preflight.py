"""The preflight must not report GPU readiness on a machine without CUDA. No check here reads clinical data."""
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

from tests.support.byte_tokenizer import ByteTokenizer

SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "preflight_rtx3070.py"


def load_script():
    spec = importlib.util.spec_from_file_location("preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(has_cuda(), reason="this test is for machines without CUDA")
def test_without_cuda_gpu_checks_fail_and_nothing_claims_readiness(tmp_path):
    out = tmp_path / "preflight.json"
    done = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], capture_output=True, text=True, timeout=120)
    report = json.loads(out.read_text(encoding="utf-8"))
    status = {c["check"]: c["status"] for c in report["checks"]}
    assert done.returncode == 1 and report["failed"] > 0
    assert status["CUDA available"] == "fail" and status["GPU is an RTX 3070"] == "fail" and status["bf16 support"] == "fail"
    assert status["gold file matches the approved hash"] == "skipped" and status["model, memory, stability and inference checks"] == "skipped"


def test_the_script_covers_the_required_checks():
    text = SCRIPT.read_text(encoding="utf-8")
    for needed in ("nvidia-smi", "CUDA available", "GPU is an RTX 3070", "VRAM", "bf16 support", "system RAM", "free disk", "bitsandbytes", "4-bit model load", "LoRA trainable parameters",
                   "gradient checkpointing enabled", "sequence length and peak memory", "training stability", "greedy inference", "preserves text exactly", "prompt template against"):
        assert needed in text, needed


def test_invented_stress_example_reaches_the_length_and_keeps_labels():
    module = load_script()
    encoded, total = module.invented_example(ByteTokenizer(), "Istruzioni.", 3000)
    assert total >= 3000 and len(encoded["input_ids"]) == total + 1 and encoded["labels"][0] == -100 and encoded["labels"][-1] != -100


def test_the_script_uses_only_invented_text_unless_asked_to_hash_gold():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "load_gold" not in text and "load_train_notes" not in text and "load_held_out_notes" not in text


def test_ram_can_be_measured_here():
    assert load_script().total_ram_gb() > 0
