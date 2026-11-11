"""Portable QLoRA training with an exact optimizer-step budget.

A small custom loop rather than a trainer class: compared arms must get the same number of optimizer steps, and the loop
has few version-sensitive parts on an untested machine. Real training requires CUDA, because bitsandbytes does. CPU
runs exist only for tiny, randomly initialized test models. There is no Apple-specific path.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import time

from src.data.loaders import ROOT
from src.infrastructure import INFRASTRUCTURE_VERSION
from src.infrastructure.provenance import check_source_integrity, run_environment
from src.training import TRAINING_PIPELINE_VERSION
from src.training.dataset import build_fold_dataset
from src.training.experiments import EXPERIMENTS_CONFIG, TRAINING_CONFIG, load_yaml, require_training_decisions, resolve_experiment
from src.training.formatting import TEMPLATE_NAME, encode_dataset, prompt_sha, read_system_prompt, template_of, token_length_report
from src.training.inference import check_limit_covers_targets


class TrainingEnvironmentError(RuntimeError):
    pass


def micro_batch_order(n_examples: int, n_micro_batches: int, seed: int) -> list[int]:
    """Deterministic order: a new shuffle for each pass, cycling until the budget is used."""
    if n_examples < 1:
        raise ValueError("no training examples")
    order, cycle = [], 0
    while len(order) < n_micro_batches:
        indices = list(range(n_examples))
        random.Random(f"{seed}:{cycle}").shuffle(indices)
        order += indices
        cycle += 1
    return order[:n_micro_batches]


def budget_report(n_examples: int, optimizer_steps: int, accumulation: int) -> dict:
    """What a fixed budget means for one arm. Arms sharing a budget see different numbers of passes over their data."""
    micro = optimizer_steps * accumulation
    return {"examples": n_examples, "optimizer_steps": optimizer_steps, "accumulation": accumulation, "micro_batches": micro, "passes_over_data": round(micro / n_examples, 2)}


def presentation_report(examples, encoded, optimizer_steps: int, accumulation: int, seed: int) -> dict:
    """Exactly what training will see: unique examples, presentations, tokens and effective epochs.
    Uses the same order as train_steps, so it describes the real run."""
    order = micro_batch_order(len(encoded), optimizer_steps * accumulation, seed)
    tokens = sum(len(encoded[i]["input_ids"]) for i in order)
    loss_tokens = sum(sum(1 for x in encoded[i]["labels"] if x != -100) for i in order)
    sources = {e.source_id for e in examples}
    by_source = {}
    for i in order:
        by_source[examples[i].source] = by_source.get(examples[i].source, 0) + 1
    return {"optimizer_steps": optimizer_steps, "accumulation": accumulation, "examples_including_repeats": len(examples), "unique_source_records": len(sources),
            "presentations": len(order), "presentations_by_source": dict(sorted(by_source.items())), "tokens_processed": tokens, "loss_tokens_processed": loss_tokens,
            "effective_epochs": round(len(order) / len(examples), 3), "effective_epochs_per_unique_record": round(len(order) / len(sources), 3)}


def package_versions() -> dict:
    from importlib import metadata
    out = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "tokenizers", "safetensors", "numpy"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def attach_lora(model, lora: dict):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(**lora))


def trainable_parameters(model) -> tuple[int, int]:
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


def train_steps(model, encoded, settings: dict, optimizer_steps: int, seed: int, device: str = "cuda", amp_dtype=None, on_step=None) -> list[float]:
    """Run exactly `optimizer_steps` updates. Returns the mean loss per update. A non-finite loss stops the run."""
    import torch
    from transformers import get_cosine_schedule_with_warmup
    if settings["batch_size"] != 1:
        raise ValueError("only batch size 1 is implemented, as configured")
    if settings["scheduler"] != "cosine":
        raise ValueError("only the cosine schedule is implemented, as configured")
    accumulation = settings["gradient_accumulation_steps"]
    _set_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=settings["learning_rate"], weight_decay=0.0)
    schedule = get_cosine_schedule_with_warmup(optimizer, settings["warmup_steps"], optimizer_steps)
    order = micro_batch_order(len(encoded), optimizer_steps * accumulation, seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses, running = [], 0.0
    for i, index in enumerate(order, 1):
        row = encoded[index]
        batch = {k: torch.tensor([row[k]], device=device) for k in ("input_ids", "attention_mask", "labels")}
        if amp_dtype is not None:
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype):
                loss = model(**batch).loss
        else:
            loss = model(**batch).loss
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite loss at micro-batch {i}")
        (loss / accumulation).backward()
        running += value
        if i % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(params, settings["max_grad_norm"])
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(running / accumulation)
            running = 0.0
            if on_step:
                on_step(len(losses), losses[-1])
    return losses


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg["model"]["id"], revision=cfg["model"]["revision"])


def _quantization(cfg: dict):
    import torch
    from transformers import BitsAndBytesConfig
    q = dict(cfg["quantization"])
    q["bnb_4bit_compute_dtype"] = getattr(torch, q["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(**q), q["bnb_4bit_compute_dtype"]


def require_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        raise TrainingEnvironmentError("real training and inference need CUDA with bitsandbytes. Run on the training machine after the preflight")


def _attention(cfg: dict) -> dict:
    """Only set when the configuration names one (eager for Gemma 3, as its model card advises for training); otherwise the library default."""
    name = cfg["model"].get("attn_implementation")
    return {"attn_implementation": name} if name else {}


def load_for_training(cfg: dict):
    """4-bit base model with LoRA attached and gradient checkpointing on."""
    import torch  # noqa: F401
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM
    require_cuda()
    quantization, dtype = _quantization(cfg)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["id"], revision=cfg["model"]["revision"], quantization_config=quantization, device_map={"": 0},
                                                 **_attention(cfg))
    checkpointing = cfg["training"]["gradient_checkpointing"]
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return attach_lora(model, cfg["lora"]), dtype


def load_for_inference(cfg: dict, adapter_dir):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    require_cuda()
    quantization, _ = _quantization(cfg)
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["id"], revision=cfg["model"]["revision"], quantization_config=quantization, device_map={"": 0},
                                                **_attention(cfg))
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model


def folder_sha256(folder) -> str:
    h = hashlib.sha256()
    for p in sorted(pathlib.Path(folder).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(folder)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def config_id(cfg: dict, experiment: dict) -> str:
    keep = {k: cfg[k] for k in ("prompt_version", "quantization", "lora", "training")} | {"model": {k: cfg["model"].get(k) for k in ("id", "revision")}}
    # Added for Gemma 3. Absent from the Llama configuration, so Llama run IDs stay as they were.
    if cfg["model"].get("attn_implementation"):
        keep["model"]["attn_implementation"] = cfg["model"]["attn_implementation"]
    if template_of(cfg) != TEMPLATE_NAME:
        keep["model"]["chat_template"] = cfg["model"]["chat_template"]
    if output_format_of(cfg) != "segments":   # E3-X; absent from the original configurations, whose IDs stay as they were
        keep["output_format"] = output_format_of(cfg)
    keep["training"] = {k: v for k, v in keep["training"].items() if k != "seeds"}
    return hashlib.sha256(json.dumps({"config": keep, "experiment": experiment}, sort_keys=True).encode()).hexdigest()[:12]


def output_format_of(cfg: dict) -> str:
    return cfg.get("output_format", "segments")


def run_training(experiment_id: str, fold: int, seed: int, output_root, official: bool = False, training_config=TRAINING_CONFIG, experiments_config=EXPERIMENTS_CONFIG) -> pathlib.Path:
    """Train one fold. Official runs need committed source and official generated data."""
    import torch
    cfg, design = load_yaml(training_config), load_yaml(experiments_config)
    experiment = resolve_experiment(experiment_id, design)
    require_training_decisions(cfg)
    integrity = check_source_integrity(ROOT)
    if official and not integrity.clean:
        raise TrainingEnvironmentError(f"official training needs committed source. {integrity.summary()}")
    require_cuda()
    out = pathlib.Path(output_root) / experiment_id / f"fold_{fold}" / f"seed_{seed}"
    if out.exists():
        raise TrainingEnvironmentError(f"{out} already exists; runs are immutable")
    t = cfg["training"]
    dataset = build_fold_dataset(experiment_id, fold, t["gold_oversample"], experiment.get("synthetic"), require_official=official, output_format=output_format_of(cfg))
    system, template = read_system_prompt(cfg["prompt_version"]), template_of(cfg)
    tokenizer = load_tokenizer(cfg)
    lengths = token_length_report(tokenizer, system, dataset.examples, template)
    encoded = encode_dataset(tokenizer, system, dataset.examples, t["max_length"], template)
    headroom = check_limit_covers_targets(tokenizer, dataset.examples, cfg["inference"], template)
    model, dtype = load_for_training(cfg)
    trained, total = trainable_parameters(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    total_steps = t["optimizer_steps"]

    def progress(step: int, loss: float) -> None:
        """One line per optimizer step: numbers only, flushed so a log file shows progress during a long run."""
        elapsed = time.time() - started
        print(f"step {step}/{total_steps} loss {loss:.4f} elapsed_min {elapsed / 60:.1f} eta_min {elapsed / step * (total_steps - step) / 60:.1f}",
              flush=True)

    print(f"training {experiment_id} fold {fold} seed {seed}: {len(encoded)} examples, {total_steps} optimizer steps x "
          f"{t['gradient_accumulation_steps']} micro-batches", flush=True)
    losses = train_steps(model, encoded, t, total_steps, seed, "cuda:0", dtype, on_step=progress)
    out.mkdir(parents=True)
    model.save_pretrained(str(out / "adapter"))
    (out / "dataset_manifest.json").write_text(json.dumps(dataset.manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    manifest = {"pipeline_version": TRAINING_PIPELINE_VERSION, "infrastructure_version": INFRASTRUCTURE_VERSION, "experiment": experiment_id, "fold": fold, "seed": seed, "official_run": official,
                "config_id": config_id(cfg, experiment), "config": {k: cfg[k] for k in ("model", "quantization", "lora", "training", "prompt_version")}, "experiment_definition": experiment,
                "prompt_sha256": prompt_sha(system, template), "dataset_sha256": dataset.manifest["dataset_sha256"], "budget": presentation_report(dataset.examples, encoded, t["optimizer_steps"], t["gradient_accumulation_steps"], seed),
                "token_lengths": lengths, "inference_limit_headroom": headroom, "trainable_parameters": trained, "total_parameters": total, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "losses": losses, "adapter_sha256": folder_sha256(out / "adapter"), "source_integrity": integrity.to_manifest(), "packages": package_versions(), "environment": run_environment()}
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return out
