"""Experiment definitions and comparison checks. A blocked experiment cannot be run, and a comparison declared
single-factor must differ in exactly one factor, so causal language can be checked against the design."""
from __future__ import annotations

import pathlib

import yaml

from src.data.loaders import ROOT

EXPERIMENTS_CONFIG = ROOT / "configs" / "experiments.yaml"
TRAINING_CONFIG = ROOT / "configs" / "training.yaml"


class ExperimentBlocked(RuntimeError):
    pass


class ConfigurationIncomplete(RuntimeError):
    """A value the owner must decide is still unset."""


def load_yaml(path) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


def factors(experiment: dict) -> dict:
    """The variables that define an arm."""
    s = experiment.get("synthetic") or {}
    return {"domain_adaptation": bool(experiment.get("domain_adaptation")), "uses_synthetic": bool(s), "synthetic_arm": s.get("arm"),
            "entity_mix": (s.get("n_named"), s.get("n_name_free")) if s else None, "negatives": s.get("n_negative", 0) if s else 0}


def differing_factors(a: dict, b: dict) -> list[str]:
    fa, fb = factors(a), factors(b)
    return sorted(k for k in fa if fa[k] != fb[k])


def validate_design(config: dict) -> list[str]:
    """Problems in the design. An empty list means every declared comparison is what it claims to be."""
    problems = []
    experiments = config["experiments"]
    for c in config.get("comparisons", []):
        if c["a"] not in experiments or c["b"] not in experiments:
            problems.append(f"{c['id']}: unknown experiment")
            continue
        changed = differing_factors(experiments[c["a"]], experiments[c["b"]])
        if c.get("single_factor"):
            names = {"synthetic_arm": {"synthetic_arm"}, "negatives": {"negatives"}, "entity_mix": {"entity_mix"}, "domain_adaptation": {"domain_adaptation"}}
            if set(changed) != names.get(c.get("factor"), {None}):
                problems.append(f"{c['id']}: declared single-factor on {c.get('factor')}, but differs in {changed}")
        elif len(changed) < 2 and not c.get("allow_causal_review"):
            problems.append(f"{c['id']}: marked multi-factor, but differs in {changed}")
    return problems


def resolve_experiment(experiment_id: str, config: dict) -> dict:
    if experiment_id not in config["experiments"]:
        raise KeyError(f"unknown experiment {experiment_id}")
    experiment = config["experiments"][experiment_id]
    if experiment.get("blocked"):
        raise ExperimentBlocked(f"{experiment_id} is blocked: {experiment['blocked']}")
    return experiment


def require_training_decisions(training: dict) -> None:
    """Training refuses to start until these are decided. They are not given defaults on purpose."""
    missing = []
    if not training["model"].get("id") or not training["model"].get("revision"):
        missing.append("model.id and model.revision (base or Instruct)")
    t = training["training"]
    if not t.get("optimizer_steps"):
        missing.append("training.optimizer_steps")
    if not t.get("max_length"):
        missing.append("training.max_length")
    if missing:
        raise ConfigurationIncomplete("decide these first: " + "; ".join(missing))
    if t["optimizer_steps"] <= t["warmup_steps"]:
        raise ConfigurationIncomplete("training.optimizer_steps must exceed warmup_steps, or warmup never finishes")
    from src.training.formatting import TEMPLATES, template_sha
    pinned = training["model"].get("chat_template") or {}
    if pinned.get("name") not in TEMPLATES or pinned.get("sha256") != template_sha(pinned["name"]):
        raise ConfigurationIncomplete("model.chat_template does not match the template in src/training/formatting.py")
