"""Command line for datasets, training and inference.

    python -m src.training.cli design
    python -m src.training.cli dataset --experiment E1 --fold 1 --out dataset_manifest.json
    python -m src.training.cli train --experiment E1 --fold 1 --seed 42
    python -m src.training.cli infer --experiment E1 --fold 1 --seed 42 --run checkpoints/development/E1/fold_1/seed_42
    python -m src.training.cli merge --folds DIR1 DIR2 DIR3 DIR4 DIR5 --out predictions.jsonl

train and infer take --training-config (default configs/training.yaml, Llama 3.2 3B); Gemma 3 1B uses configs/training_gemma3_1b.yaml.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from src.data.loaders import ROOT
from src.data.splits import load_fold
from src.infrastructure.provenance import run_environment
from src.training.dataset import build_fold_dataset
from src.training.experiments import EXPERIMENTS_CONFIG, TRAINING_CONFIG, load_yaml, require_training_decisions, resolve_experiment, validate_design
from src.training.formatting import prompt_sha, read_system_prompt, template_of
from src.training.inference import HFGenerator, load_held_out_notes, merge_out_of_fold, run_fold_inference, stop_token_ids


def infer(experiment_id: str, fold: int, seed: int, run_dir, output_root, training_config=TRAINING_CONFIG) -> pathlib.Path:
    from src.training.model_training import config_id, folder_sha256, load_for_inference, load_tokenizer, output_format_of, package_versions
    cfg, design = load_yaml(training_config), load_yaml(EXPERIMENTS_CONFIG)
    experiment = resolve_experiment(experiment_id, design)
    require_training_decisions(cfg)
    run_dir = pathlib.Path(run_dir)
    trained = json.loads((run_dir / "training_manifest.json").read_text(encoding="utf-8"))
    if (trained["experiment"], trained["fold"], trained["seed"]) != (experiment_id, fold, seed):
        raise ValueError("this adapter was trained for a different experiment, fold or seed")
    if trained["config_id"] != config_id(cfg, experiment):
        raise ValueError("this adapter was trained with a different training configuration (check --training-config)")
    adapter_sha = folder_sha256(run_dir / "adapter")
    if adapter_sha != trained["adapter_sha256"]:
        raise ValueError("adapter files differ from the training manifest")
    spec = load_fold(fold)
    system, template = read_system_prompt(cfg["prompt_version"]), template_of(cfg)
    if prompt_sha(system, template) != trained["prompt_sha256"]:
        raise ValueError("the prompt differs from the one used in training")
    tokenizer = load_tokenizer(cfg)
    model = load_for_inference(cfg, run_dir / "adapter")
    identifier = config_id(cfg, experiment)
    meta = {"experiment": experiment_id, "model": cfg["model"]["id"], "model_revision": cfg["model"]["revision"], "checkpoint": f"{experiment_id}/seed_{seed}", "config_id": identifier, "seed": seed}
    out = pathlib.Path(output_root) / f"{experiment_id}__{identifier}__seed{seed}" / f"fold_{fold}"
    return run_fold_inference(HFGenerator(model, stop_token_ids(tokenizer, template)), tokenizer, system, load_held_out_notes(spec), spec, meta, cfg["inference"], out,
                              {"adapter_sha256": adapter_sha, "prompt_sha256": prompt_sha(system, template), "packages": package_versions(), "environment": run_environment()},
                              template, output_format_of(cfg))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("design")
    d = sub.add_parser("dataset")
    d.add_argument("--experiment", required=True)
    d.add_argument("--fold", type=int, required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--official", action="store_true")
    d.add_argument("--training-config", default=str(TRAINING_CONFIG))
    t = sub.add_parser("train")
    t.add_argument("--experiment", required=True)
    t.add_argument("--fold", type=int, required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--official", action="store_true")
    t.add_argument("--output-root")
    t.add_argument("--training-config", default=str(TRAINING_CONFIG))
    i = sub.add_parser("infer")
    i.add_argument("--experiment", required=True)
    i.add_argument("--fold", type=int, required=True)
    i.add_argument("--seed", type=int, required=True)
    i.add_argument("--run", required=True)
    i.add_argument("--output-root", default=str(ROOT / "predictions_secure"))
    i.add_argument("--training-config", default=str(TRAINING_CONFIG))
    m = sub.add_parser("merge")
    m.add_argument("--folds", nargs="+", required=True)
    m.add_argument("--out", required=True)
    a = parser.parse_args(argv)
    if a.command == "design":
        problems = validate_design(load_yaml(EXPERIMENTS_CONFIG))
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    if a.command == "dataset":
        from src.training.model_training import output_format_of
        cfg, design = load_yaml(a.training_config), load_yaml(EXPERIMENTS_CONFIG)
        experiment = resolve_experiment(a.experiment, design)
        dataset = build_fold_dataset(a.experiment, a.fold, cfg["training"]["gold_oversample"], experiment.get("synthetic"), require_official=a.official,
                                     output_format=output_format_of(cfg))
        pathlib.Path(a.out).write_text(json.dumps(dataset.manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({"experiment": a.experiment, "fold": a.fold, "counts": dataset.manifest["counts"], "dataset_sha256": dataset.manifest["dataset_sha256"]}, indent=1))
        return 0
    if a.command == "train":
        from src.training.model_training import run_training
        # A model other than the original Llama gets its own subfolder, so its runs can never collide with the Llama ones.
        folder = load_yaml(a.training_config)["model"].get("run_folder", "")
        root = a.output_root or str(ROOT / "checkpoints" / ("official" if a.official else "development") / folder)
        print(run_training(a.experiment, a.fold, a.seed, root, a.official, training_config=a.training_config))
        return 0
    if a.command == "infer":
        print(infer(a.experiment, a.fold, a.seed, a.run, a.output_root, a.training_config))
        return 0
    print(json.dumps(merge_out_of_fold(a.folds, a.out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
