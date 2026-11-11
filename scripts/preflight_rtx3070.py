"""Hardware preflight for the training machine. It trains no real experiment and uses invented text only.

    python scripts/preflight_rtx3070.py --out preflight.json
    python scripts/preflight_rtx3070.py --out preflight.json --model-id ID --revision SHA --check-gold

Results apply only to the machine that ran this. On a machine without CUDA the GPU checks fail, as they should: a Mac
result says nothing about the RTX 3070. Exit code 0 means no check failed. Checks that were not requested are `skipped`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_DISK_GB, MIN_RAM_GB, MAX_PEAK_GPU_GB = 100, 16, 7.5
ROUND_TRIP_SAMPLES = ["Testo  con  doppi  spazi.", "Città già più è, dell'Aquilonia; (nota) 12/03/2015.", "Riga uno\nRiga due\tfine", 'Virgolette "doppie" e \\ barra.']
INVENTED_SENTENCE = "La signora Zelinda Quagliarino, di 47 anni, residente a Borgoverde dell'Aquilonia, riferisce un lieve dolore toracico dal 12/03/2015. "


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name: str, status: str, detail="") -> None:
        self.rows.append({"check": name, "status": status, "detail": detail})
        print(f"{status.upper():7} {name}: {str(detail)[:150]}", flush=True)

    def run(self, name: str, fn) -> object:
        """A check function returns (passed, detail). An exception fails that check and the rest continue."""
        try:
            passed, detail = fn()
            self.add(name, "pass" if passed else "fail", detail)
            return passed
        except Exception as e:
            self.add(name, "fail", f"{type(e).__name__}: {str(e)[:200]}")
            return False

    def skip(self, name: str, why: str) -> None:
        self.add(name, "skipped", why)


def total_ram_gb() -> float:
    if platform.system() == "Windows":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = Status()
        s.dwLength = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullTotalPhys / 1024 ** 3
    import os
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3


def nvidia_smi() -> tuple[bool, str]:
    done = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total,memory.used", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    return done.returncode == 0, done.stdout.strip() or done.stderr.strip()


def environment_checks(r: Report) -> bool:
    from src.infrastructure.provenance import check_source_integrity, run_environment
    r.add("operating system", "pass", run_environment())
    r.run("NVIDIA driver and GPU, nvidia-smi", nvidia_smi)
    r.run("system RAM", lambda: (total_ram_gb() >= MIN_RAM_GB, f"{total_ram_gb():.1f} GB, minimum {MIN_RAM_GB}"))
    r.run("free disk", lambda: (shutil.disk_usage(ROOT).free / 1024 ** 3 >= MIN_DISK_GB, f"{shutil.disk_usage(ROOT).free / 1024 ** 3:.0f} GB, minimum {MIN_DISK_GB}"))
    integrity = check_source_integrity(ROOT)
    r.add("repository at a committed state", "pass" if integrity.clean else "fail", integrity.summary())
    eol = subprocess.run(["git", "ls-files", "--eol"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    converted = [line.split("\t")[-1] for line in eol if "w/crlf" in line and "attr/-text" not in line]
    r.add("line endings, no conversion on this machine", "fail" if converted else "pass", f"{len(converted)} files converted" if converted else f"{len(eol)} files checked")
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        r.run(f"package {package}", lambda p=package: (True, __import__("importlib.metadata").metadata.version(p)))
    try:
        import torch
    except ImportError:
        r.add("PyTorch import", "fail", "not installed")
        for name in ("CUDA available", "CUDA tensor operation", "GPU is an RTX 3070", "VRAM", "bf16 support"):
            r.add(name, "fail", "PyTorch is not installed")
        return False
    ok = bool(torch.cuda.is_available())
    r.add("CUDA available", "pass" if ok else "fail", f"torch {torch.__version__}, CUDA build {torch.version.cuda}")
    if not ok:
        for name in ("CUDA tensor operation", "GPU is an RTX 3070", "VRAM", "bf16 support"):
            r.add(name, "fail", "CUDA is not available on this machine")
        return False
    r.run("CUDA tensor operation", lambda: (float((torch.ones(4, device="cuda:0") * 2).sum()) == 8.0, "4 elements summed on cuda:0"))
    r.run("GPU is an RTX 3070", lambda: ("3070" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)))
    r.add("VRAM", "pass", f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    r.run("bf16 support", lambda: (torch.cuda.is_bf16_supported(), "torch.cuda.is_bf16_supported"))
    return True


def invented_example(tokenizer, system: str, target_tokens: int, template: str = "project_llama3_segments_v1"):
    """An invented note and target whose token total reaches the requested length."""
    from src.training.formatting import encode_training_example, format_prompt, token_ids
    from src.training.targets import build_segments, serialize_target
    note = INVENTED_SENTENCE
    while True:
        target = serialize_target(build_segments(note, []))
        total = len(token_ids(tokenizer, format_prompt(system, note, template))) + len(token_ids(tokenizer, target))
        if total >= target_tokens:
            return encode_training_example(tokenizer, system, note, target, total + 8, "invented", template), total
        note += INVENTED_SENTENCE


def model_checks(r: Report, args) -> None:
    import torch
    from src.training.experiments import TRAINING_CONFIG, load_yaml
    from src.training.formatting import format_prompt, get_template, read_system_prompt, template_of, token_ids
    from src.training.inference import HFGenerator, stop_token_ids
    from src.training.model_training import load_for_training, load_tokenizer, train_steps, trainable_parameters
    cfg = load_yaml(args.training_config or TRAINING_CONFIG)
    template = template_of(cfg)
    cfg["model"]["id"], cfg["model"]["revision"] = args.model_id, args.revision
    system = read_system_prompt(cfg["prompt_version"])
    tokenizer = load_tokenizer(cfg)
    r.add("tokenizer loads at the pinned revision", "pass", args.revision)

    def round_trip():
        bad = [i for i, s in enumerate(ROUND_TRIP_SAMPLES) if tokenizer.decode(token_ids(tokenizer, s), skip_special_tokens=True, clean_up_tokenization_spaces=False) != s]
        return not bad, f"samples that changed: {bad}"
    r.run("tokenizer preserves text exactly, including double spaces", round_trip)

    import hashlib
    from src.training.formatting import template_sha
    if not getattr(tokenizer, "chat_template", None):
        r.add("prompt template against the tokenizer template", "info", "the tokenizer has no template")
    else:
        own = format_prompt(system, "NOTA", template)
        theirs = tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": "NOTA"}], tokenize=False, add_generation_prompt=True)
        # Recorded, not failed: the project template is pinned, and the tokenizer's may insert a current date.
        r.add("prompt template against the tokenizer template", "info", {"identical": own == theirs, "project_template": template, "project_sha256": template_sha(template)[:16],
                                                                       "tokenizer_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()[:16]})
    markers = get_template(template).markers
    r.run("template markers are single tokens", lambda: (all(len(token_ids(tokenizer, m)) == 1 for m in markers), f"{len(markers)} markers"))
    r.run("stop tokens found", lambda: (len(stop_token_ids(tokenizer, template)) >= 1, sorted(stop_token_ids(tokenizer, template))))

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model, dtype = load_for_training(cfg)
    r.add("4-bit model load with LoRA and gradient checkpointing", "pass", f"{time.time() - started:.0f} s, {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB allocated")
    trained, total = trainable_parameters(model)
    r.add("LoRA trainable parameters", "pass" if 0 < trained < total else "fail", f"{trained:,} of {total:,}")
    r.add("gradient checkpointing enabled", "pass" if getattr(model, "is_gradient_checkpointing", False) else "fail", "")
    settings = {**cfg["training"], "warmup_steps": 2}

    def stress(tokens: int):
        encoded, length = invented_example(tokenizer, system, tokens, template)
        torch.cuda.reset_peak_memory_stats()
        started_at = time.time()
        train_steps(model, [encoded], {**settings, "gradient_accumulation_steps": 1, "warmup_steps": 0}, 2, 42, "cuda:0", dtype)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        return peak <= MAX_PEAK_GPU_GB, f"{length} tokens, peak {peak:.2f} GB, limit {MAX_PEAK_GPU_GB} GB, {(time.time() - started_at) / 2:.1f} s per step"
    for tokens in args.stress_tokens:
        r.run(f"sequence length and peak memory at about {tokens} tokens", lambda t=tokens: stress(t))

    def stability():
        encoded, _ = invented_example(tokenizer, system, min(args.stress_tokens), template)
        losses = train_steps(model, [encoded], {**settings, "gradient_accumulation_steps": 1}, args.stability_steps, 7, "cuda:0", dtype)
        return all(x == x for x in losses), f"{args.stability_steps} updates, first loss {losses[0]:.3f}, last {losses[-1]:.3f}, allocated {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
    r.run("training stability on invented text", stability)

    def greedy():
        model.eval()
        model.config.use_cache = True
        generator = HFGenerator(model, stop_token_ids(tokenizer, template))
        prompt = token_ids(tokenizer, format_prompt(system, INVENTED_SENTENCE, template))
        first, second = generator.generate(prompt, 40), generator.generate(prompt, 40)
        model.config.use_cache = False
        model.train()
        return first == second, f"{len(first)} tokens, two greedy runs identical: {first == second}"
    r.run("greedy inference repeats exactly", greedy)
    r.add("peak GPU memory, whole preflight", "pass", f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--training-config", help="default configs/training.yaml; configs/training_gemma3_1b.yaml for Gemma 3 1B")
    parser.add_argument("--stress-tokens", type=int, nargs="+", default=[1536, 2600, 3200])
    parser.add_argument("--stability-steps", type=int, default=200)
    parser.add_argument("--check-gold", action="store_true", help="hash the copied gold file; reads no text into the report")
    args = parser.parse_args(argv)
    r = Report()
    gpu_ready = environment_checks(r)
    if args.check_gold:
        from src.data.loaders import GOLD_V2, file_sha256
        import yaml
        approved = yaml.safe_load((ROOT / "configs" / "evaluation.yaml").read_text(encoding="utf-8"))["gold"]["sha256"]
        r.run("gold file matches the approved hash", lambda: (file_sha256(GOLD_V2) == approved, approved[:16]))
    else:
        r.skip("gold file matches the approved hash", "pass --check-gold after copying the secure data")
    if args.model_id and args.revision and gpu_ready:
        try:
            model_checks(r, args)
        except Exception as e:
            r.add("model checks", "fail", f"{type(e).__name__}: {str(e)[:300]}")
    else:
        r.skip("model, memory, stability and inference checks", "need CUDA plus --model-id and --revision")
    failed = [x for x in r.rows if x["status"] == "fail"]
    report = {"note": "Results apply only to this machine.", "failed": len(failed), "passed": sum(x["status"] == "pass" for x in r.rows), "skipped": sum(x["status"] == "skipped" for x in r.rows), "checks": r.rows}
    pathlib.Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{report['passed']} passed, {report['failed']} failed, {report['skipped']} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
