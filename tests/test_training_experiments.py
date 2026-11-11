"""Experiment design, required decisions, the step budget and portability."""
import ast
import pathlib

import pytest

from src.training.experiments import (EXPERIMENTS_CONFIG, TRAINING_CONFIG, ConfigurationIncomplete, ExperimentBlocked, differing_factors, load_yaml, require_training_decisions, resolve_experiment, validate_design)
from src.training.model_training import budget_report, micro_batch_order

DESIGN, TRAINING = load_yaml(EXPERIMENTS_CONFIG), load_yaml(TRAINING_CONFIG)
PACKAGE = pathlib.Path(__file__).parents[1] / "src" / "training"


def test_requested_experiments_are_defined_and_runnable():
    for name in ("E1", "E2", "E3", "E8", "E9", "E10"):
        assert resolve_experiment(name, DESIGN)["domain_adaptation"] is False


@pytest.mark.parametrize("name", ["E4", "E5", "E7"])
def test_domain_adaptation_arms_are_blocked(name):
    with pytest.raises(ExperimentBlocked, match="DART"):
        resolve_experiment(name, DESIGN)


def test_every_domain_adaptation_arm_is_blocked():
    assert all(e.get("blocked") for e in DESIGN["experiments"].values() if e.get("domain_adaptation"))


def test_arms_have_the_intended_composition():
    e = DESIGN["experiments"]
    assert e["E1"]["synthetic"] is None
    assert e["E2"]["synthetic"]["arm"] == "non_anchored" and e["E3"]["synthetic"]["arm"] == "style_anchored"
    assert e["E8"]["synthetic"]["n_negative"] == 0 and e["E3"]["synthetic"]["n_negative"] == 20
    assert (e["E3"]["synthetic"]["n_named"], e["E3"]["synthetic"]["n_name_free"]) == (202, 78), "the reduced plan of 2026-11-06"
    assert e["E9"]["synthetic"]["n_named"] + e["E9"]["synthetic"]["n_name_free"] == e["E10"]["synthetic"]["n_named"] + e["E10"]["synthetic"]["n_name_free"] == 202


def test_the_design_is_consistent():
    assert validate_design(DESIGN) == []


@pytest.mark.parametrize("a,b,factor", [("E2", "E3", "synthetic_arm"), ("E9", "E10", "entity_mix"), ("E8", "E3", "negatives")])
def test_single_factor_comparisons_change_one_factor(a, b, factor):
    assert differing_factors(DESIGN["experiments"][a], DESIGN["experiments"][b]) == [factor]


def test_gold_only_comparisons_are_not_marked_single_factor():
    for c in DESIGN["comparisons"]:
        if c["a"] == "E1":
            assert c["single_factor"] is False and len(differing_factors(DESIGN["experiments"][c["a"]], DESIGN["experiments"][c["b"]])) > 1


def test_a_false_single_factor_claim_is_caught():
    bad = {"experiments": DESIGN["experiments"], "comparisons": [{"id": "x", "a": "E1", "b": "E3", "single_factor": True, "factor": "synthetic_arm"}]}
    assert validate_design(bad)


def test_owner_decisions_are_recorded():
    """Decided 2026-11-06: Instruct at a pinned revision. Decided 2026-11-09: 79 shared steps, warmup 8, max_length 3072."""
    from src.training.formatting import TEMPLATE_NAME, template_sha
    assert TRAINING["model"]["id"] == "meta-llama/Llama-3.2-3B-Instruct" and TRAINING["model"]["revision"] == "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    assert TRAINING["model"]["chat_template"] == {"name": TEMPLATE_NAME, "sha256": template_sha()}
    t = TRAINING["training"]
    assert t["optimizer_steps"] == 79 and t["warmup_steps"] == 8 and t["max_length"] == 3072
    assert (t["batch_size"], t["gradient_accumulation_steps"], t["learning_rate"], t["scheduler"]) == (1, 16, 5.0e-5, "cosine")
    assert t["gradient_checkpointing"] is True and TRAINING["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert (TRAINING["lora"]["r"], TRAINING["lora"]["lora_alpha"]) == (16, 32)
    assert "Base" not in TRAINING["model"]["id"] and not TRAINING["model"]["id"].endswith("3B")


def complete():
    import copy
    c = copy.deepcopy(TRAINING)
    c["model"].update(id="m", revision="r")
    c["training"].update(optimizer_steps=263, max_length=3000)
    return c


def test_training_refuses_until_decisions_are_made():
    require_training_decisions(TRAINING)    # decided 2026-11-09: the committed configuration is complete
    require_training_decisions(complete())
    for section, key in (("model", "id"), ("model", "revision"), ("training", "optimizer_steps"), ("training", "max_length")):
        c = complete()
        c[section][key] = None
        with pytest.raises(ConfigurationIncomplete):
            require_training_decisions(c)


def test_a_changed_template_is_refused():
    c = complete()
    c["model"]["chat_template"] = {"name": "project_llama3_segments_v1", "sha256": "0" * 64}
    with pytest.raises(ConfigurationIncomplete, match="chat_template"):
        require_training_decisions(c)


def test_presentation_report_counts_what_training_sees():
    from src.training.dataset import Example
    from src.training.model_training import presentation_report
    examples = [Example(f"g{i}#r{k}", "gold", f"g{i}", k, "gold", "x", "y") for i in range(3) for k in range(2)] + [Example("s0", "synthetic", "F1-N00000", 0, "synthetic_named", "x", "y")]
    encoded = [{"input_ids": [1] * (10 + i), "labels": [-100] * 5 + [1] * (5 + i), "attention_mask": [1] * (10 + i)} for i in range(7)]
    r = presentation_report(examples, encoded, 3, 4, 42)
    order = micro_batch_order(7, 12, 42)
    assert r["presentations"] == 12 and r["examples_including_repeats"] == 7 and r["unique_source_records"] == 4
    assert r["tokens_processed"] == sum(10 + i for i in order) and r["loss_tokens_processed"] == sum(5 + i for i in order)
    assert r["effective_epochs"] == round(12 / 7, 3) and r["effective_epochs_per_unique_record"] == 3.0
    assert sum(r["presentations_by_source"].values()) == 12


def test_a_budget_that_never_finishes_warmup_is_refused():
    c = complete()
    c["training"]["optimizer_steps"] = c["training"]["warmup_steps"]
    with pytest.raises(ConfigurationIncomplete, match="warmup"):
        require_training_decisions(c)


def test_provisional_settings_match_the_request():
    t, l, q = TRAINING["training"], TRAINING["lora"], TRAINING["quantization"]
    assert (l["r"], l["lora_alpha"], l["lora_dropout"]) == (16, 32, 0.05) and len(l["target_modules"]) == 7
    assert t["batch_size"] == 1 and t["gradient_accumulation_steps"] == 16 and t["gradient_checkpointing"] is True and t["learning_rate"] == 5e-5
    assert q["load_in_4bit"] is True and q["bnb_4bit_quant_type"] == "nf4"
    assert TRAINING["inference"]["do_sample"] is False and TRAINING["inference"]["num_beams"] == 1


def test_candidate_models_are_pinned():
    assert all(len(c["revision"]) == 40 for c in TRAINING["model"]["candidates"].values())


def test_step_budget_is_the_same_across_arm_sizes():
    gold_only, with_synthetic = budget_report(325, 263, 16), budget_report(2105, 263, 16)
    assert gold_only["micro_batches"] == with_synthetic["micro_batches"] == 4208
    assert gold_only["passes_over_data"] == 12.95 and with_synthetic["passes_over_data"] == 2.0


def test_micro_batch_order_is_exact_and_deterministic():
    order = micro_batch_order(5, 23, 42)
    assert len(order) == 23 and order == micro_batch_order(5, 23, 42) and order != micro_batch_order(5, 23, 43)
    assert sorted(order[:5]) == list(range(5)) and sorted(order[5:10]) == list(range(5)) and order[:5] != order[5:10]


def test_micro_batch_order_refuses_an_empty_dataset():
    with pytest.raises(ValueError):
        micro_batch_order(0, 4, 42)


def modules():
    return [(p.name, ast.parse(p.read_text(encoding="utf-8"))) for p in PACKAGE.glob("*.py")]


def test_gpu_libraries_are_imported_only_inside_functions():
    for name, tree in modules():
        for node in tree.body:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.split(".")[0] in {"torch", "transformers", "peft", "bitsandbytes", "accelerate"} for n in names), (name, names)


def test_there_is_no_apple_specific_training_path():
    for p in PACKAGE.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert '"mps"' not in text and "backends.mps" not in text, p.name


def test_evaluator_package_is_not_changed_by_this_pipeline():
    from src.evaluation import EVALUATOR_VERSION
    from src.evaluation.runner import evaluator_fingerprint
    assert EVALUATOR_VERSION == "1.0.1" and evaluator_fingerprint().startswith("6e72b75b4d50cde2")
