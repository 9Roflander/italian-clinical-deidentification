"""Record-level manifests. They contain IDs and statuses, never note text."""
from __future__ import annotations

import csv
import pathlib

RECORD_FIELDS = ["record_id", "source_dataset", "source_record_id", "record_type", "contains_target_pii",
                 "entity_annotations_available", "exact_duplicate_status", "near_duplicate_status", "fold_id",
                 "partition", "generation_prompt_version", "generation_model_version", "permitted_for_training",
                 "exclusion_reason"]


def write_csv(path, fields: list[str], rows: list[dict]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def record(record_id, source_dataset, source_record_id, record_type, contains_target_pii, annotations,
           exact="not_checked", near="not_checked", fold="", partition="", prompt="", model="", permitted=False,
           reason="") -> dict:
    return {"record_id": record_id, "source_dataset": source_dataset, "source_record_id": source_record_id,
            "record_type": record_type, "contains_target_pii": contains_target_pii,
            "entity_annotations_available": annotations, "exact_duplicate_status": exact,
            "near_duplicate_status": near, "fold_id": fold, "partition": partition,
            "generation_prompt_version": prompt, "generation_model_version": model,
            "permitted_for_training": permitted, "exclusion_reason": reason}
