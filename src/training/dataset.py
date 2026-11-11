"""Fold-specific training datasets.

Gold notes come only from `load_train_notes`, which never returns held-out text. Synthetic records must carry that fold's
identifiers, an accepted and audited manifest entry, and anchors from that fold's training notes only. Problems raise an
error, because silently dropping a record would change a dataset without anyone noticing.

The manifest holds IDs and hashes. Note text never leaves memory here.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import re
from dataclasses import dataclass, field

from src.data.loaders import DEFAULT_GOLD, ROOT, file_sha256
from src.data.splits import SPLITS_DIR, FoldSpec, LeakageError, load_fold, load_train_notes
from src.evaluation.alignment import render_inline
from src.infrastructure.provenance import is_official_output
from src.training.extraction import ExtractionTargetError, serialize_extraction_target
from src.training.targets import TargetError, build_segments, serialize_target, text_sha256

OUTPUT_FORMATS = ("segments", "extraction")


def make_target(text: str, spans, output_format: str = "segments") -> str:
    """The exact string a model is trained to produce, in the configured output format."""
    if output_format == "segments":
        return serialize_target(build_segments(text, spans))
    if output_format == "extraction":
        try:
            return serialize_extraction_target(text, spans)
        except ExtractionTargetError as e:
            raise TargetError(str(e)) from e
    raise ValueError(f"unknown output format {output_format}")

RECORD_TYPES = ("synthetic_named", "synthetic_name_free", "synthetic_negative")


class DatasetError(RuntimeError):
    """A dataset cannot be built safely. Messages hold IDs and counts, never text."""


@dataclass(frozen=True)
class Example:
    example_id: str
    source: str
    source_id: str
    repeat: int
    record_type: str
    input_text: str
    target_text: str


@dataclass
class FoldDataset:
    experiment: str
    fold: int
    examples: list
    manifest: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SyntheticSource:
    root: pathlib.Path
    arm: str
    records: tuple
    examples_sha256: str
    plan_sha256: str
    git_commit: str
    official: bool


def _jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_synthetic_source(root, fold: int, arm: str, spec: FoldSpec, require_official: bool = False, repo_root=ROOT, output_format: str = "segments") -> SyntheticSource:
    """Load and validate one fold's generated records against their manifest, audit result and fold."""
    run_dir = pathlib.Path(root) / f"fold_{fold}" / arm
    needed = ("examples.jsonl", "generation_manifest.csv", "validation_report.json", "run_config.json")
    absent = [n for n in needed if not (run_dir / n).exists()]
    if absent:
        raise DatasetError(f"fold {fold} {arm}: missing {absent}")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "validation_report.json").read_text(encoding="utf-8"))
    if config.get("fold") != fold or config.get("arm") != arm or report.get("fold") != fold or report.get("arm") != arm:
        raise LeakageError(f"run files in fold {fold} {arm} describe a different fold or arm")
    if config.get("gold_sha256") != spec.gold_sha256:
        raise DatasetError(f"fold {fold} {arm}: generated from a different gold file")
    if report.get("post_audit") != "passed" or report.get("permitted_for_training") is not True:
        raise DatasetError(f"fold {fold} {arm}: the post-generation audit has not passed")
    official = bool(config.get("official_run")) and not config.get("git_dirty", True) and is_official_output(root, repo_root)
    if require_official and not official:
        raise DatasetError(f"fold {fold} {arm}: official experiments need an official generation run from committed source")
    with open(run_dir / "generation_manifest.csv", newline="", encoding="utf-8") as f:
        manifest = {r["record_id"]: r for r in csv.DictReader(f)}
    permitted = {k for k, r in manifest.items() if r["status"] == "accepted" and r["permitted_for_training"] == "True"}
    id_pattern = re.compile(rf"^syn_f{fold}_{re.escape(arm)}_\d{{5}}$")
    case_pattern = re.compile(rf"^F{fold}-N\d{{5}}$")
    train_ids, test_ids = set(spec.train_ids), set(spec.test_ids)
    records, seen = [], set()
    for r in _jsonl(run_dir / "examples.jsonl"):
        rid = str(r.get("record_id"))
        if r.get("fold_id") != fold or not id_pattern.match(rid) or not case_pattern.match(str(r.get("source_record_id"))):
            raise LeakageError(f"{rid} does not belong to fold {fold} {arm}")
        if rid in seen:
            raise DatasetError(f"{rid} appears twice")
        seen.add(rid)
        if rid not in permitted:
            raise DatasetError(f"{rid} is not accepted and permitted in the manifest")
        anchors = set(r.get("anchor_ids", []))
        if anchors & test_ids or anchors - train_ids:
            raise LeakageError(f"{rid} used an anchor outside fold {fold}'s training notes")
        if arm == "non_anchored" and anchors:
            raise DatasetError(f"{rid} is in the non-anchored arm but has anchors")
        if r.get("record_type") not in RECORD_TYPES:
            raise DatasetError(f"{rid} has an unknown record type")
        if r["record_type"] == "synthetic_negative" and r.get("entities"):
            raise DatasetError(f"{rid} is a negative with entities")
        spans = [(e["start"], e["end"], e["type"]) for e in r.get("entities", [])]
        try:
            target = make_target(r["text"], spans, output_format)
            matches = render_inline(r["text"], spans) == r.get("redacted_text")
        except (TargetError, ValueError, KeyError) as e:
            raise DatasetError(f"{rid}: invalid target ({e})") from e
        if not matches:
            raise DatasetError(f"{rid}: stored redaction differs from its spans")
        records.append({**r, "_target": target})
    missing = permitted - seen
    if missing:
        raise DatasetError(f"fold {fold} {arm}: {len(missing)} permitted records are missing from examples.jsonl")
    records.sort(key=lambda x: x["record_id"])
    return SyntheticSource(pathlib.Path(root), arm, tuple(records), file_sha256(run_dir / "examples.jsonl"),
                           config.get("plan_sha256", ""), config.get("git_commit", ""), official)


def matched_case_ids(sources) -> set[str]:
    """Plan case IDs accepted in every arm, so compared arms can train on the same cases."""
    sets = [{r["source_record_id"] for r in s.records} for s in sources]
    return set.intersection(*sets) if sets else set()


def select_records(source: SyntheticSource, n_named: int, n_name_free: int, n_negative: int, matched_ids=None, allow_shortfall: bool = False) -> list[dict]:
    """Nested selection by plan order, so a smaller arm is always a subset of a larger one."""
    wanted = {"synthetic_named": n_named, "synthetic_name_free": n_name_free, "synthetic_negative": n_negative}
    chosen = []
    for kind, n in wanted.items():
        pool = sorted((r for r in source.records if r["record_type"] == kind and (matched_ids is None or r["source_record_id"] in matched_ids)), key=lambda r: r["type_order"])
        if len(pool) < n and not allow_shortfall:
            raise DatasetError(f"{source.arm}: requested {n} {kind} records, but only {len(pool)} are available")
        chosen += pool[:n]
    return sorted(chosen, key=lambda r: r["record_id"])


def assert_no_held_out(examples, spec: FoldSpec) -> None:
    held_out = set(spec.test_ids)
    bad = sorted({e.source_id for e in examples if e.source == "gold" and (e.source_id in held_out or e.source_id not in set(spec.train_ids))})
    if bad:
        raise LeakageError(f"fold {spec.fold}: held-out or unknown notes in training data: {bad}")
    for e in examples:
        if e.source == "synthetic" and not e.example_id.startswith(f"syn_f{spec.fold}_"):
            raise LeakageError(f"{e.example_id} crossed into fold {spec.fold}")


def build_fold_dataset(experiment: str, fold: int, gold_oversample: int = 5, synthetic=None, synthetic_root=None, gold_path=DEFAULT_GOLD, splits_dir=SPLITS_DIR,
                       require_official: bool = False, repo_root=ROOT, output_format: str = "segments") -> FoldDataset:
    """`synthetic` is None for gold-only, or a dict: arm, n_named, n_name_free, n_negative, and optionally matched_with."""
    if gold_oversample < 1:
        raise DatasetError("gold_oversample must be at least 1")
    if output_format not in OUTPUT_FORMATS:
        raise DatasetError(f"unknown output format {output_format}")
    spec = load_fold(fold, splits_dir)
    spec.check()
    notes = sorted(load_train_notes(spec, gold_path), key=lambda n: n.note_id)
    if {n.note_id for n in notes} != set(spec.train_ids):
        raise LeakageError(f"fold {fold}: training notes differ from the frozen manifest")
    examples = []
    for n in notes:
        if any(e.get("start") is None for e in n.entities):
            raise DatasetError(f"{n.note_id}: gold mention without an offset")
        try:
            target = make_target(n.text, [(e["start"], e["end"], e["type"]) for e in n.entities], output_format)
        except TargetError as e:
            raise DatasetError(f"{n.note_id}: invalid target ({e})") from e
        for k in range(gold_oversample):
            examples.append(Example(f"{n.note_id}#r{k}", "gold", n.note_id, k, "gold", n.text, target))
    synthetic_info = None
    if synthetic:
        root = synthetic_root if synthetic_root is not None else repo_root / "synthetic_data"
        source = load_synthetic_source(root, fold, synthetic["arm"], spec, require_official, repo_root, output_format)
        matched = None
        if synthetic.get("matched_with"):
            other = load_synthetic_source(root, fold, synthetic["matched_with"], spec, require_official, repo_root, output_format)
            matched = matched_case_ids([source, other])
        chosen = select_records(source, synthetic["n_named"], synthetic["n_name_free"], synthetic["n_negative"], matched, synthetic.get("allow_shortfall", False))
        for r in chosen:
            examples.append(Example(r["record_id"], "synthetic", r["source_record_id"], 0, r["record_type"], r["text"], r["_target"]))
        synthetic_info = {"arm": source.arm, "matched_with": synthetic.get("matched_with"), "matched_case_count": len(matched) if matched is not None else None,
                          "official_generation_run": source.official, "examples_file_sha256": source.examples_sha256, "plan_sha256": source.plan_sha256,
                          "generation_git_commit": source.git_commit, "requested": {k: synthetic[k] for k in ("n_named", "n_name_free", "n_negative")}}
    assert_no_held_out(examples, spec)
    unique_inputs = collections.defaultdict(set)
    for e in examples:
        unique_inputs[text_sha256(e.input_text)].add(e.source_id)
    duplicated = sorted(tuple(sorted(v)) for v in unique_inputs.values() if len(v) > 1)
    if duplicated:
        raise DatasetError(f"fold {fold}: different records share one input: {duplicated[:5]}")
    rows = [{"example_id": e.example_id, "source": e.source, "source_id": e.source_id, "repeat": e.repeat, "record_type": e.record_type,
             "input_sha256": text_sha256(e.input_text), "target_sha256": text_sha256(e.target_text)} for e in examples]
    manifest = {"experiment": experiment, "fold": fold, "gold_sha256": spec.gold_sha256, "gold_oversample": gold_oversample,
                **({"output_format": output_format} if output_format != "segments" else {}),
                "train_note_ids": list(spec.train_ids), "held_out_note_ids": list(spec.test_ids),
                "counts": dict(collections.Counter(e.record_type for e in examples)), "total_examples": len(examples),
                "synthetic": synthetic_info, "dataset_sha256": text_sha256(json.dumps(rows, sort_keys=True)), "examples": rows}
    return FoldDataset(experiment, fold, examples, manifest)
