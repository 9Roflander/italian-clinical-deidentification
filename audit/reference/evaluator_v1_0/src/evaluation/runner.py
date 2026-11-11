"""Evaluate one model/configuration/seed from its saved out-of-fold predictions and write the required files.

Raw outputs go to predictions_secure/ (git-ignored). Every other file holds IDs, offsets and numbers only.
Run folders are never overwritten. `verify` recomputes a run from its saved predictions and compares byte for byte.

    python -m src.evaluation.runner baseline
    python -m src.evaluation.runner evaluate --predictions predictions.jsonl
    python -m src.evaluation.runner compare --a RUN_A --b RUN_B
    python -m src.evaluation.runner verify --run RUN
    python -m src.evaluation.runner controls
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
import pathlib
import platform
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import yaml

from src.data.loaders import GOLD_V1, GOLD_V2, ROOT, file_sha256, load_gold
from src.data.splits import FOLD_CSV, SPLITS_DIR, LeakageError
from src.evaluation import EVALUATOR_VERSION
from src.evaluation.alignment import (CATEGORIES, MISSING, PLACEHOLDER_OF, UNALIGNABLE, UNKNOWN, VALID, Resolution, resolve_prediction)
from src.evaluation.gold_verification import VerifiedGold, verify_gold
from src.evaluation.historical import historical_scores
from src.evaluation.metrics import COUNT_COLUMNS, as_report, compute_metrics
from src.evaluation.scoring import NoteScore, score_note, totals
from src.evaluation.statistics import SUMMARY_METRICS, bootstrap_intervals, mean_sd, paired_difference, per_note_table, seed_summary

CONFIG = ROOT / "configs" / "evaluation.yaml"
SECURE_ROOT = ROOT / "predictions_secure"
PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REQUIRED_FIELDS = ("experiment", "model", "model_revision", "checkpoint", "config_id", "seed", "note_id", "fold_id", "input_sha256", "output_format")
RUN_FIELDS = ("experiment", "model", "model_revision", "checkpoint", "config_id", "seed")
GENERATED_FILES = ("alignment_report.json", "per_note_metrics.csv", "per_fold_metrics.csv", "per_category_metrics.csv",
                   "pooled_metrics.json", "error_breakdown.csv", "bootstrap_results.json", "evaluation_summary.md")


class EvaluationInputError(RuntimeError):
    """Predictions cannot be evaluated as supplied. Messages name IDs, never text."""


class RunExistsError(RuntimeError):
    """A run folder already exists. Results are immutable."""


@dataclass
class EvaluationResult:
    meta: dict
    scores: list
    outputs: dict


def load_config(path=CONFIG) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluator_fingerprint(config_path=CONFIG) -> str:
    h = hashlib.sha256()
    for p in sorted(PACKAGE_DIR.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update(pathlib.Path(config_path).read_bytes())
    return h.hexdigest()


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _output_text(resolved: Resolution, source: str, record: dict | None) -> str:
    """The redacted text seen by the historical metrics."""
    if resolved.is_valid:
        out, pos = [], 0
        for s in sorted(set(resolved.spans)):
            out += [source[pos : s.start], PLACEHOLDER_OF.get(s.label, f"[{UNKNOWN}]")]
            pos = s.end
        return "".join(out) + source[pos:]
    raw = record.get("raw_model_output") if record else None
    return raw if isinstance(raw, str) else ""


def evaluate_records(records: list[dict], gold: VerifiedGold) -> EvaluationResult:
    """Each gold note must have one out-of-fold prediction. A missing one counts as an invalid output and stays reported."""
    if not records:
        raise EvaluationInputError("no prediction records")
    by_note, meta = {}, None
    for r in records:
        absent = [k for k in REQUIRED_FIELDS if k not in r]
        if absent:
            raise EvaluationInputError(f"record {r.get('note_id', '?')} lacks {absent}")
        this = {k: r[k] for k in RUN_FIELDS}
        if meta is None:
            meta = this
        elif this != meta:
            raise EvaluationInputError("one file must hold one model, configuration and seed")
        i = r["note_id"]
        if i not in gold.notes:
            raise EvaluationInputError(f"{i} is not a gold note")
        if i in by_note:
            raise EvaluationInputError(f"{i} has more than one prediction")
        if r["fold_id"] != gold.fold_of[i]:
            raise LeakageError(f"{i} was predicted by the model of fold {r['fold_id']}, but it is held out in fold {gold.fold_of[i]}")
        if r["input_sha256"] != text_sha256(gold.notes[i].text):
            raise EvaluationInputError(f"{i}: the prediction was made on different input text")
        by_note[i] = r
    scores, outputs = [], {}
    for i in gold.order:
        source, r = gold.notes[i].text, by_note.get(i)
        if r is None:
            resolved = Resolution(MISSING)
        else:
            resolved = resolve_prediction(source, r["output_format"], r.get("raw_model_output"), r.get("predicted_spans"),
                                          r.get("completion_status", "complete"), bool(r.get("truncation_flag", False)), r.get("finish_reason"))
        scores.append(score_note(i, gold.fold_of[i], source, gold.spans[i], resolved))
        outputs[i] = _output_text(resolved, source, r)
    return EvaluationResult(meta, scores, outputs)


def _csv_text(fieldnames, rows) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _json_text(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def _fmt(x) -> str:
    return "undefined" if x is None else f"{x:.3f}"


def _ci(entry: dict) -> str:
    return f"{_fmt(entry['point'])} [{_fmt(entry['lower'])}, {_fmt(entry['upper'])}]"


def _pooled_counts(t: dict) -> dict:
    return {c: {"tp": t[f"tp_{c}"], "fp": t[f"fp_{c}"], "fn": t[f"fn_{c}"]} for c in CATEGORIES}


def build_outputs(result: EvaluationResult, gold: VerifiedGold, cfg: dict) -> dict:
    """Every generated file as text. Deterministic for fixed predictions, evaluator and NumPy version."""
    scores = result.scores
    folds = sorted({s.fold for s in scores})
    pooled_t = totals(scores)
    pooled = as_report(compute_metrics(pooled_t))
    fold_metrics = {k: as_report(compute_metrics(totals([s for s in scores if s.fold == k]))) for k in folds}
    fold_totals = {k: totals([s for s in scores if s.fold == k]) for k in folds}

    status_counts = collections.Counter(s.status for s in scores)
    invalid = [s for s in scores if s.status != VALID]
    alignment = {
        "notes": len(scores), "valid_outputs": status_counts[VALID], "invalid_outputs": len(invalid),
        "outputs_not_alignable_unambiguously": sum(s.status in UNALIGNABLE for s in scores),
        "missing_predictions": status_counts[MISSING], "by_status": dict(sorted(status_counts.items())),
        "outputs_normalized": sum(s.output_normalized for s in scores),
        "invalid_note_ids": [s.note_id for s in invalid],
        "per_note": [{"note_id": s.note_id, "fold": s.fold, "status": s.status, "n_predictions": s.n_predictions,
                      "output_normalized": s.output_normalized,
                      "gold_entity_spans": [list(x) for x in gold.spans[s.note_id]],
                      "predicted_entity_spans": [[p.start, p.end, p.label] for p in s.predicted_spans]} for s in scores]}

    note_fields = ["note_id", "fold", "status", "n_predictions", *COUNT_COLUMNS]
    per_note = _csv_text(note_fields, [s.row() for s in scores])

    fold_fields = ["scope", "notes", *SUMMARY_METRICS, *[f"f1_{c}" for c in CATEGORIES], "invalid_output_rate", "removal_coverage", "over_redaction_rate"]
    keys = fold_fields[2:]
    fold_rows = [{"scope": f"fold_{k}", "notes": fold_totals[k]["notes"], **{m: fold_metrics[k][m] for m in keys}} for k in folds]
    summary = {m: mean_sd([fold_metrics[k][m] for k in folds]) for m in keys}
    fold_rows.append({"scope": "mean_of_folds", "notes": "", **{m: summary[m]["mean"] for m in keys}})
    fold_rows.append({"scope": "sd_of_folds", "notes": "", **{m: summary[m]["sd"] for m in keys}})
    per_fold = _csv_text(fold_fields, fold_rows)

    boot_cfg = cfg["bootstrap"]
    table = per_note_table([s.row() for s in scores])
    fold_ids = [s.fold for s in scores]
    intervals = bootstrap_intervals(table, fold_ids, boot_cfg["iterations"], boot_cfg["seed"], boot_cfg["confidence"], boot_cfg["minimum_iterations"])
    bootstrap = {"configuration": {**boot_cfg, "resampling_unit": "document", "numpy_version": np.__version__},
                 "scope": "Pooled out-of-fold results. Intervals are conditional on the trained fold models.",
                 "note_order_sha256": text_sha256("\n".join(s.note_id for s in scores)), "metrics": intervals}

    category_fields = ["scope", "category", "tp", "fp", "fn", "precision", "recall", "f1", "f1_ci_lower", "f1_ci_upper"]
    category_rows = []
    for c in CATEGORIES:
        category_rows.append({"scope": "pooled", "category": c, "tp": pooled_t[f"tp_{c}"], "fp": pooled_t[f"fp_{c}"], "fn": pooled_t[f"fn_{c}"],
                              "precision": pooled[f"precision_{c}"], "recall": pooled[f"recall_{c}"], "f1": pooled[f"f1_{c}"],
                              "f1_ci_lower": intervals[f"f1_{c}"]["lower"], "f1_ci_upper": intervals[f"f1_{c}"]["upper"]})
    for k in folds:
        for c in CATEGORIES:
            category_rows.append({"scope": f"fold_{k}", "category": c, "tp": fold_totals[k][f"tp_{c}"], "fp": fold_totals[k][f"fp_{c}"], "fn": fold_totals[k][f"fn_{c}"],
                                  "precision": fold_metrics[k][f"precision_{c}"], "recall": fold_metrics[k][f"recall_{c}"], "f1": fold_metrics[k][f"f1_{c}"]})
    for c in CATEGORIES:
        for name, stat_key in (("mean_of_folds", "mean"), ("sd_of_folds", "sd")):
            category_rows.append({"scope": name, "category": c, **{m: mean_sd([fold_metrics[k][f"{m}_{c}"] for k in folds])[stat_key] for m in ("precision", "recall", "f1")}})
    per_category = _csv_text(category_fields, category_rows)

    error_rows = []
    pooled_errors = collections.Counter()
    for s in scores:
        pooled_errors.update(s.errors)
    for (category, side, kind), n in sorted(pooled_errors.items()):
        error_rows.append({"category": category, "side": side, "kind": kind, "count": n})
    errors = _csv_text(["category", "side", "kind", "count"], error_rows)

    historical = {"purpose": "Historical comparison with Miranda et al. only. Not a quality measure: empty output scores 1.0.",
                  "gold_v2": historical_scores(result.outputs, {i: gold.notes[i].entities for i in gold.order}, gold.fold_of)}
    if pathlib.Path(GOLD_V1).exists() and file_sha256(GOLD_V1) == cfg["historical_gold_v1_sha256"]:
        v1 = {n.note_id: n.entities for n in load_gold(GOLD_V1)}
        if set(gold.order) <= set(v1):
            historical["gold_v1"] = historical_scores(result.outputs, v1, gold.fold_of)
    pooled_json = {
        "gold": {"version": gold.version, "sha256": gold.sha256, "notes": len(gold.order), "entities": sum(len(v) for v in gold.spans.values())},
        "distinction": "Pooled metrics combine TP, FP and FN from all held-out folds. They differ from the arithmetic mean of fold scores in per_fold_metrics.csv.",
        "counts": _pooled_counts(pooled_t), "unknown_placeholder_false_positives": pooled_t["fp_unknown"],
        "metrics": pooled,
        "correct_redaction_recall": pooled["micro_recall"],
        "correct_redaction_recall_note": "Protocol 4.1. Identical to strict micro recall by definition, so it is not a separate result.",
        "output_copy_fidelity": {"notes_with_text_exactly_preserved": pooled_t["notes_text_preserved"], "share_of_notes": pooled["text_preserved_rate"],
                                 "inserted_characters": pooled_t["inserted_chars"], "deleted_characters": pooled_t["deleted_chars"], "modified_characters": pooled_t["modified_chars"],
                                 "notes_with_unmeasurable_fidelity": pooled_t["fidelity_unmeasurable"],
                                 "outputs_not_alignable_unambiguously": alignment["outputs_not_alignable_unambiguously"]},
        "historical_not_for_quality_claims": historical}

    meta = result.meta
    category_table = "\n".join(
        f"| {c} | {pooled_t[f'tp_{c}']} | {pooled_t[f'fp_{c}']} | {pooled_t[f'fn_{c}']} | {_fmt(pooled[f'precision_{c}'])} | {_fmt(pooled[f'recall_{c}'])} | {_ci(intervals[f'f1_{c}'])} |"
        for c in CATEGORIES)
    fold_table = "\n".join(f"| {k} | {fold_totals[k]['notes']} | {_fmt(fold_metrics[k]['micro_f1'])} | {_fmt(fold_metrics[k]['macro_f1'])} |" for k in folds)
    limit_text = "\n".join(f"- {x}" for x in cfg["limitations"])
    summary_md = f"""# Evaluation summary: {meta['experiment']}

| Item | Value |
|---|---|
| Gold dataset | gold_{gold.version}, SHA-256 `{gold.sha256[:16]}` |
| Notes and entities | {len(gold.order)} notes, {sum(len(v) for v in gold.spans.values())} entities |
| Fold assignments | Frozen, held-out sizes {list(gold.fold_sizes.values())} |
| Model | {meta['model']}, revision {meta['model_revision']} |
| Checkpoint | {meta['checkpoint']} |
| Configuration | {meta['config_id']} |
| Random seed | {meta['seed']} |

## Pooled out-of-fold results

Intervals are {boot_cfg['confidence']:.0%}, from a document-level bootstrap stratified by fold, {boot_cfg['iterations']:,} iterations, seed {boot_cfg['seed']}. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **{_ci(intervals['micro_f1'])}** |
| Macro-F1 | {_ci(intervals['macro_f1'])} |
| Micro precision | {_ci(intervals['micro_precision'])} |
| Micro recall, also correct redaction recall | {_ci(intervals['micro_recall'])} |
| Removal coverage | {_ci(intervals['removal_coverage'])} |
| Complete-entity redaction recall | {_ci(intervals['complete_entity_recall'])} |
| Over-redaction rate, relative to current gold | {_ci(intervals['over_redaction_rate'])} |
| Non-sensitive characters preserved | {_ci(intervals['non_sensitive_preservation'])} |
| Note-level exact redaction rate | {_ci(intervals['note_exact_rate'])} |
| Invalid-output rate | {_ci(intervals['invalid_output_rate'])} |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
{category_table}

Unknown placeholders counted as false positives without a category: {pooled_t['fp_unknown']}. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
{fold_table}
| Mean | | {_fmt(summary['micro_f1']['mean'])} | {_fmt(summary['macro_f1']['mean'])} |
| SD | | {_fmt(summary['micro_f1']['sd'])} | {_fmt(summary['macro_f1']['sd'])} |

Fold means are arithmetic means of fold scores, not the pooled results. The folds overlap in training data, so they are not independent experiments.

## Output-copy fidelity

| Item | Value |
|---|---:|
| Notes with non-redacted text exactly preserved | {pooled_t['notes_text_preserved']} of {pooled_t['notes']} |
| Inserted characters | {pooled_t['inserted_chars']} |
| Deleted characters | {pooled_t['deleted_chars']} |
| Modified characters | {pooled_t['modified_chars']} |
| Outputs with unmeasurable fidelity | {pooled_t['fidelity_unmeasurable']} |
| Outputs that cannot be aligned unambiguously | {alignment['outputs_not_alignable_unambiguously']} |
| Invalid outputs, all causes | {len(invalid)} |

## Known limitations

{limit_text}
"""
    return {"alignment_report.json": _json_text(alignment), "per_note_metrics.csv": per_note, "per_fold_metrics.csv": per_fold,
            "per_category_metrics.csv": per_category, "pooled_metrics.json": _json_text(pooled_json), "error_breakdown.csv": errors,
            "bootstrap_results.json": _json_text(bootstrap), "evaluation_summary.md": summary_md}


def run_id_of(meta: dict) -> str:
    return f"{meta['experiment']}__{meta['config_id']}__seed{meta['seed']}"


def _lock(path: pathlib.Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def write_run(records: list[dict], gold: VerifiedGold, cfg: dict, output_root, secure_root=SECURE_ROOT, stage: str = "development_cross_validation",
              require_clean_tree: bool = False) -> pathlib.Path:
    """Write one immutable run: nine public files plus predictions_oof.jsonl in the secure folder."""
    git = git_state()
    if require_clean_tree and git["dirty"]:
        raise EvaluationInputError("uncommitted changes: the recorded commit would not identify the evaluator that produced this run")
    result = evaluate_records(records, gold)
    run_id = run_id_of(result.meta)
    run_dir, secure_dir = pathlib.Path(output_root) / run_id, pathlib.Path(secure_root) / run_id
    if run_dir.exists() or secure_dir.exists():
        raise RunExistsError(f"{run_id} already exists; results are immutable")
    outputs = build_outputs(result, gold, cfg)
    position = {i: n for n, i in enumerate(gold.order)}
    ordered = sorted(records, key=lambda r: position[r["note_id"]])
    secure_dir.mkdir(parents=True)
    predictions_path = secure_dir / "predictions_oof.jsonl"
    predictions_path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in ordered), encoding="utf-8", newline="\n")
    _lock(predictions_path)
    pending = pathlib.Path(output_root) / f".{run_id}.pending"
    pending.mkdir(parents=True)
    for name, text in outputs.items():
        (pending / name).write_text(text, encoding="utf-8", newline="\n")
    manifest = {
        "run_id": run_id, "stage": stage, **result.meta,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gold": {"version": gold.version, "sha256": gold.sha256, "notes": len(gold.order), "entities": sum(len(v) for v in gold.spans.values()),
                 "verification": "passed: hash, counts per category and fold, offsets, overlaps, NFC, stored targets"},
        "folds": {"source": "splits/fold_*.json", "held_out_sizes": gold.fold_sizes, "file_sha256": gold.fold_file_sha256},
        "evaluator": {"version": EVALUATOR_VERSION, "fingerprint_sha256": evaluator_fingerprint(), "git_commit": git["commit"], "git_dirty": git["dirty"]},
        "predictions": {"path": str(predictions_path.relative_to(ROOT)) if predictions_path.is_relative_to(ROOT) else predictions_path.name,
                        "sha256": file_sha256(predictions_path), "records": len(records)},
        "bootstrap": cfg["bootstrap"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "files": {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in outputs.items()}}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    (pending / "run_manifest.json").write_text(_json_text(manifest), encoding="utf-8", newline="\n")
    for p in pending.iterdir():
        _lock(p)
    os.replace(pending, run_dir)
    return run_dir


def read_predictions(path) -> list[dict]:
    return [json.loads(x) for x in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def verify_run(run_dir, gold: VerifiedGold, cfg: dict, secure_root=SECURE_ROOT) -> dict:
    """Recompute a run from its saved predictions. It reproduces only if every file matches byte for byte."""
    run_dir = pathlib.Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    stated = manifest.pop("manifest_sha256")
    actual = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    predictions_path = pathlib.Path(secure_root) / manifest["run_id"] / "predictions_oof.jsonl"
    problems = []
    if stated != actual:
        problems.append("run_manifest.json was changed")
    if file_sha256(predictions_path) != manifest["predictions"]["sha256"]:
        problems.append("predictions_oof.jsonl was changed")
    if manifest["gold"]["sha256"] != gold.sha256:
        problems.append("gold file differs")
    outputs = build_outputs(evaluate_records(read_predictions(predictions_path), gold), gold, cfg)
    for name in GENERATED_FILES:
        if (run_dir / name).read_text(encoding="utf-8") != outputs[name]:
            problems.append(f"{name} does not reproduce")
    return {"run_id": manifest["run_id"], "reproduced": not problems, "problems": problems,
            "evaluator_changed_since_run": manifest["evaluator"]["fingerprint_sha256"] != evaluator_fingerprint()}


def _load_rows(run_dir) -> tuple[list[dict], dict]:
    run_dir = pathlib.Path(run_dir)
    with open(run_dir / "per_note_metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))


def compare_runs(run_a, run_b, cfg: dict) -> dict:
    """Paired comparison of B minus A from saved per-note counts, resampling the same documents for both."""
    rows_a, man_a = _load_rows(run_a)
    rows_b, man_b = _load_rows(run_b)
    if man_a["gold"]["sha256"] != man_b["gold"]["sha256"]:
        raise EvaluationInputError("the runs used different gold files")
    if [(r["note_id"], r["fold"]) for r in rows_a] != [(r["note_id"], r["fold"]) for r in rows_b]:
        raise EvaluationInputError("the runs do not cover the same notes in the same folds")
    if man_a["seed"] != man_b["seed"]:
        raise EvaluationInputError("compare runs of the same training seed")
    b = cfg["bootstrap"]
    differences = paired_difference(per_note_table(rows_a), per_note_table(rows_b), [int(r["fold"]) for r in rows_a], b["iterations"], b["seed"], b["confidence"], b["minimum_iterations"])
    return {"a": man_a["run_id"], "b": man_b["run_id"], "difference": "b minus a", "gold_sha256": man_a["gold"]["sha256"],
            "configuration": {**b, "pairing": "same resampled documents in each replicate", "numpy_version": np.__version__},
            "evaluators": [man_a["evaluator"]["fingerprint_sha256"], man_b["evaluator"]["fingerprint_sha256"]],
            "note": "Intervals are conditional on the trained fold models and do not capture all training randomness.",
            "metrics": differences}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--output-root", default=str(ROOT / "experiments" / "clean_cv"))
    ev.add_argument("--stage", default="development_cross_validation")
    ev.add_argument("--allow-dirty-tree", action="store_true")
    ev.add_argument("--secure-root", default=str(SECURE_ROOT))
    base = sub.add_parser("baseline")
    base.add_argument("--output-root", default=str(ROOT / "experiments" / "baselines"))
    base.add_argument("--secure-root", default=str(SECURE_ROOT))
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--a", required=True)
    cmp_.add_argument("--b", required=True)
    cmp_.add_argument("--output-root", default=str(ROOT / "experiments" / "comparisons"))
    ver = sub.add_parser("verify")
    ver.add_argument("--run", required=True)
    ver.add_argument("--secure-root", default=str(SECURE_ROOT))
    seeds = sub.add_parser("seeds")
    seeds.add_argument("--runs", nargs="+", required=True)
    sub.add_parser("controls")
    args = parser.parse_args(argv)
    cfg = load_config()
    if args.command == "seeds":
        pooled = {}
        for run in args.runs:
            run = pathlib.Path(run)
            manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
            pooled[manifest["seed"]] = json.loads((run / "pooled_metrics.json").read_text(encoding="utf-8"))["metrics"]
        print(json.dumps(seed_summary(pooled), indent=1))
        return 0
    if args.command == "compare":
        report = compare_runs(args.a, args.b, cfg)
        out = pathlib.Path(args.output_root)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{report['a']}__vs__{report['b']}.json"
        if path.exists():
            raise RunExistsError(f"{path.name} already exists")
        path.write_text(_json_text(report), encoding="utf-8", newline="\n")
        print(f"wrote {path}")
        return 0
    gold = verify_gold(cfg)
    print(f"gold verified: {gold.version}, {len(gold.order)} notes, {sum(len(v) for v in gold.spans.values())} entities")
    if args.command == "evaluate":
        path = write_run(read_predictions(args.predictions), gold, cfg, args.output_root, args.secure_root, stage=args.stage, require_clean_tree=not args.allow_dirty_tree)
    elif args.command == "baseline":
        from src.evaluation.controls import baseline_e0_records
        path = write_run(baseline_e0_records(gold), gold, cfg, args.output_root, args.secure_root, stage="evaluator_control")
    elif args.command == "verify":
        report = verify_run(args.run, gold, cfg, args.secure_root)
        print(json.dumps(report, indent=1))
        return 0 if report["reproduced"] else 1
    else:
        from src.evaluation.controls import write_control_report
        path = write_control_report(gold)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
