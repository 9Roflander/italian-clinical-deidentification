"""Phase 2 audit: every dataset against the gold notes, the negative-corpus audit, and all manifests.

Outputs IDs, hashes and counts only.
    python -m src.data.leakage_audit
"""
from __future__ import annotations

import collections
import csv
import json

from src.data.annotations import canonical_type
from src.data.deduplication import compare_collections, rows_as_dicts
from src.data.loaders import DEFAULT_GOLD, GOLD_V1, HISTORICAL_GOLD, HISTORICAL_SYNTH, ROOT, file_sha256, load_gold, load_historical_synthetic
from src.data.manifests import RECORD_FIELDS, record, write_csv
from src.data.normalization import LEVELS, doc_hash
from src.data.splits import SPLITS_DIR, freeze_folds
from src.generation.privacy_screening import AGE_ANY, INITIALS, MONTH, NUMERIC_DATE, YEAR

MANIFESTS = ROOT / "data_manifests"
RESULTS = ROOT / "audit" / "results"


def crf_match_table(gold, crf_rows) -> dict:
    """Strongest match level between each CRF row and a gold note."""
    out = {}
    for name, fn in LEVELS.items():
        index = collections.defaultdict(list)
        for g in gold:
            index[fn(g.text)].append(g.note_id)
        for r, row in enumerate(crf_rows):
            hit = index.get(fn(row["clinical_note"]))
            if hit and r not in out:
                out[r] = {"gold_id": hit[0], "level": name}
    return out


def main() -> dict:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    gold = load_gold()
    specs = freeze_folds()
    test_fold = {i: s.fold for s in specs for i in s.test_ids}
    crf = list(load_dataset("NLP-FBK/synthetic-crf-train", split="it"))
    dys = list(load_dataset("NLP-FBK/dyspnea-clinical-notes", split="it"))
    hist = load_historical_synthetic()
    api = HfApi()
    versions = {n: api.dataset_info(n) for n in ("NLP-FBK/synthetic-crf-train", "NLP-FBK/dyspnea-clinical-notes", "praiselab-picuslab/DART")}

    docs = {"gold": {g.note_id: g.text for g in gold},
            "crf": {f"crf_{i:03d}": r["clinical_note"] for i, r in enumerate(crf)},
            "dyspnea": {f"dys_{i:04d}": r["clinical_note"] for i, r in enumerate(dys)}}
    for k, v in hist.items():
        docs[k] = {f"{k}_{i:04d}": r["text"] for i, r in enumerate(v)}

    comparisons = [("gold", "gold", True), ("gold", "crf", False), ("gold", "dyspnea", False), ("crf", "dyspnea", False)]
    comparisons += [("gold", k, False) for k in hist] + [("crf", k, False) for k in hist]
    all_rows, report = [], []
    for a, b, same in comparisons:
        rows = compare_collections(a, docs[a], b, docs[b], same_collection=same)
        c = collections.Counter(r.relation for r in rows)
        report.append({"collection_a": a, "n_a": len(docs[a]), "collection_b": b, "n_b": len(docs[b]), "pairs_sharing_8gram": len(rows),
                       "exact_duplicate": c["exact_duplicate"], "near_duplicate": c["near_duplicate"],
                       "partial_overlap_review": c["partial_overlap_review"], "shared_phrase_only": c["shared_phrase_only"],
                       "max_longest_run_tokens": max((r.longest_run_tokens for r in rows), default=0)})
        all_rows += [r for r in rows if r.relation != "shared_phrase_only"]
    # internal duplicates in historical synthetic data: exact hashes only (hash comparison stays linear)
    hist_hash = collections.Counter(doc_hash(t) for k in hist for t in docs[k].values())
    write_csv(MANIFESTS / "deduplication_report.csv", list(report[0]), report)
    write_csv(MANIFESTS / "duplicate_pairs_for_review.csv", list(rows_as_dicts(all_rows)[0]) if all_rows else [], rows_as_dicts(all_rows))

    # negative-corpus audit
    match = crf_match_table(gold, crf)
    near = {r.id_b: r.id_a for r in all_rows if r.source_a == "gold" and r.source_b == "crf" and r.relation in ("near_duplicate", "exact_duplicate")}
    by_id = {g.note_id: g for g in gold}
    neg_rows = []
    for i, row in enumerate(crf):
        n, rid = row["clinical_note"], f"crf_{i:03d}"
        twin = match.get(i, {}).get("gold_id")
        near_twin = None if twin else near.get(rid)
        year_hits = len(YEAR.findall(n)) + len(MONTH.findall(n)) + len(NUMERIC_DATE.findall(n))
        mentions = len(by_id[twin].entities) if twin else None
        if twin:
            decision, why = "excluded", "duplicate_of_gold_note"
        elif near_twin:
            decision, why = "excluded", "near_duplicate_of_gold_note"
        else:
            decision, why = "quarantined", "no_annotations_and_pii_status_unverified"
        neg_rows.append({"record_id": rid, "document_id": row["document_id"], "gold_twin": twin or near_twin or "",
                         "match": match.get(i, {}).get("level", "near_duplicate" if near_twin else "none"),
                         "twin_test_fold": test_fold.get(twin or near_twin, ""), "annotated_mentions_in_twin": mentions if mentions is not None else "",
                         "age_pattern_hits": len(AGE_ANY.findall(n)), "date_pattern_hits": year_hits, "initials_pattern_hits": len(INITIALS.findall(n)),
                         "historical_target": "output=input", "decision": decision, "reason": why, "permitted_for_training": False})
    write_csv(MANIFESTS / "negative_audit.csv", list(neg_rows[0]), neg_rows)

    # record manifests
    gold_twin_ids = {r["gold_twin"] for r in neg_rows if r["gold_twin"]}
    gold_records = {g.note_id: g for g in gold}
    global_rows = []
    for r in neg_rows:
        global_rows.append(record(r["record_id"], "NLP-FBK/synthetic-crf-train", r["document_id"], "historical_negative",
                                  "yes" if r["annotated_mentions_in_twin"] not in ("", 0) else "unknown", False,
                                  "duplicate_of_gold" if r["match"] not in ("none", "near_duplicate") else "none",
                                  "near_duplicate_of_gold" if r["match"] == "near_duplicate" else "none", permitted=False, reason=r["reason"]))
    for k, v in docs.items():
        if k in hist:
            for rid in v:
                global_rows.append(record(rid, f"historical_{k}", rid, "historical_synthetic", "yes" if "named" in k else "yes_no_names", True,
                                          "none", "see_duplicate_pairs", prompt="fixed anchors from all 80 gold notes", model="gemini-2.5-flash (reported)",
                                          permitted=False, reason="generated_from_prompts_exposing_test_notes"))
    write_csv(MANIFESTS / "record_manifest.csv", RECORD_FIELDS, global_rows)
    for s in specs:
        rows = []
        for g in gold:
            is_test = g.note_id in s.test_ids
            rows.append(record(g.note_id, "gold_v2", g.document_id, "gold", "yes" if g.entities else "no_annotated", True,
                               "unique", "none_flagged", s.fold, "test" if is_test else "train", permitted=not is_test,
                               reason="outer_test_note" if is_test else ""))
        for r in neg_rows:
            rows.append(record(r["record_id"], "NLP-FBK/synthetic-crf-train", r["document_id"], "historical_negative",
                               "yes" if r["annotated_mentions_in_twin"] not in ("", 0) else "unknown", False,
                               "duplicate_of_gold" if r["match"] not in ("none", "near_duplicate") else "none",
                               "near_duplicate_of_gold" if r["match"] == "near_duplicate" else "none", s.fold, "excluded", permitted=False, reason=r["reason"]))
        write_csv(SPLITS_DIR / f"fold_{s.fold}_manifest.csv", RECORD_FIELDS, rows)
        # overlap check on what this fold's manifest permits
        permitted = {x["record_id"] for x in rows if x["permitted_for_training"]}
        if permitted & set(s.test_ids):
            raise RuntimeError(f"fold {s.fold}: test note permitted for training")

    ds_fields = ["dataset", "version", "source", "records", "task_format", "io_format", "annotations", "pii_may_be_present", "data_use_authorized",
                 "duplicate_check_status", "permitted_uses"]
    dart = versions["praiselab-picuslab/DART"]
    manifest = [
        {"dataset": "gold_v2", "version": "sha256:" + file_sha256(DEFAULT_GOLD)[:16], "source": "gold_v1 plus the reviewed corrections in data_manifests/gold_v2_change_log.csv",
         "records": len(gold), "task_format": "de-identification annotation", "io_format": "text + typed mentions with character offsets and status", "annotations": "yes, reviewed", "pii_may_be_present": "yes",
         "data_use_authorized": "not documented; source repository has no licence file; owner to confirm", "duplicate_check_status": "checked", "permitted_uses": "train partition per fold; test partition evaluation only; development data overall"},
        {"dataset": "gold_v1", "version": "sha256:" + file_sha256(GOLD_V1)[:16], "source": "Built from 'Annotated clinical notes - samples.csv', github.com/michele17284/Italian-Clinical-Note-Deidentification at e8609c92",
         "records": len(gold), "task_format": "de-identification annotation", "io_format": "text + typed mentions with character offsets and status", "annotations": "yes, as published", "pii_may_be_present": "yes",
         "data_use_authorized": "as gold_v2", "duplicate_check_status": "same text as gold_v2", "permitted_uses": "comparison with Miranda et al. only"},
        {"dataset": "historical gold_standard_80", "version": "sha256:" + file_sha256(HISTORICAL_GOLD)[:16], "source": "original_work/data/gold_standard_80.json", "records": 80, "task_format": "de-identification annotation",
         "io_format": "text + (type,string), no offsets", "annotations": "yes", "pii_may_be_present": "yes", "data_use_authorized": "as above", "duplicate_check_status": "superseded", "permitted_uses": "none; replaced by gold_v1, which adds 2 absent notes, IDs and offsets"},
        {"dataset": "NLP-FBK/synthetic-crf-train (it)", "version": versions["NLP-FBK/synthetic-crf-train"].sha, "source": "Hugging Face", "records": len(crf), "task_format": "CRF filling",
         "io_format": "clinical_note + CRF annotations", "annotations": "not for PII", "pii_may_be_present": "yes", "data_use_authorized": "public dataset; licence to be confirmed",
         "duplicate_check_status": "checked: contains gold notes", "permitted_uses": "none for this task"}]
    for k, v in hist.items():
        manifest.append({"dataset": f"historical {k}", "version": "sha256:" + file_sha256(ROOT / "original_work" / "data" / HISTORICAL_SYNTH[k])[:16], "source": f"original_work/data/{HISTORICAL_SYNTH[k]}",
                         "records": len(v), "task_format": "inline redaction", "io_format": "text + redacted_text + entities", "annotations": "yes, unverified", "pii_may_be_present": "synthetic; not verified",
                         "data_use_authorized": "project-generated", "duplicate_check_status": "checked", "permitted_uses": "none for clean training; audit reference only"})
    manifest += [
        {"dataset": "NLP-FBK/dyspnea-clinical-notes (it)", "version": versions["NLP-FBK/dyspnea-clinical-notes"].sha, "source": "Hugging Face", "records": len(dys), "task_format": "unlabelled clinical notes",
         "io_format": "clinical_note", "annotations": "no", "pii_may_be_present": "reported anonymised; not verified", "data_use_authorized": "public dataset; licence to be confirmed",
         "duplicate_check_status": "checked against gold", "permitted_uses": "external domain adaptation (Phase 1) only"},
        {"dataset": "praiselab-picuslab/DART", "version": dart.sha, "source": "Hugging Face (gated: manual approval)", "records": "16029 (reported)", "task_format": "drug regulatory documents",
         "io_format": "structured fields", "annotations": "no", "pii_may_be_present": "unlikely; not verified", "data_use_authorized": "gated; owner access reported",
         "duplicate_check_status": "NOT CHECKED: gated, no authenticated access in this audit", "permitted_uses": "Phase 1 only after overlap check"},
        {"dataset": "synthetic_clinical_1000", "version": "absent", "source": "file not in repository", "records": "1000 (reported)", "task_format": "unknown", "io_format": "unknown", "annotations": "unknown",
         "pii_may_be_present": "unknown", "data_use_authorized": "unknown", "duplicate_check_status": "cannot be checked", "permitted_uses": "none"}]
    write_csv(MANIFESTS / "dataset_manifest.csv", ds_fields, manifest)

    summary = {"folds_frozen": len(specs), "report": report, "exact_duplicates_within_historical_synthetic": sum(v - 1 for v in hist_hash.values() if v > 1),
               "negative_decisions": dict(collections.Counter(r["decision"] for r in neg_rows)), "negative_reasons": dict(collections.Counter(r["reason"] for r in neg_rows)),
               "gold_notes_with_exact_crf_twin": len({r["gold_twin"] for r in neg_rows if r["match"] not in ("none", "near_duplicate")}),
               "gold_notes_with_near_crf_twin_only": sorted({r["gold_twin"] for r in neg_rows if r["match"] == "near_duplicate"} - {r["gold_twin"] for r in neg_rows if r["match"] not in ("none", "near_duplicate")}),
               "crf_rows_quarantined": [r["document_id"] for r in neg_rows if r["decision"] == "quarantined"],
               "gold_without_crf_twin": sorted(set(gold_records) - {r["gold_twin"] for r in neg_rows}),
               "test_notes_in_crf_per_fold": {s.fold: sum(i in gold_twin_ids for i in s.test_ids) for s in specs},
               "dataset_versions": {k: {"sha": v.sha, "last_modified": str(v.last_modified), "gated": str(v.gated)} for k, v in versions.items()}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "leakage_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=1, ensure_ascii=False))
