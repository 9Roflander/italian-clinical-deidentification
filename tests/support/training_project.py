"""A miniature project on invented notes: gold with offsets, frozen folds, and generated runs for each fold and arm."""
import csv
import json

from src.data import splits
from src.evaluation.alignment import render_inline

SENTINEL = "ZZNOTETEXT"


def gold_row(i: int) -> dict:
    name, age = f"Zelinda{chr(97 + i % 26)}{chr(97 + (i * 3) % 26)}", str(20 + i)
    text = f"{SENTINEL}{i:03d} {name} ha {age} anni, vista a Borgoverde."
    spans = [(text.index(name), text.index(name) + len(name), "NOME"), (text.index(f"ha {age}") + 3, text.index(f"ha {age}") + 3 + len(age), "ETÀ"),
             (text.index("Borgoverde"), text.index("Borgoverde") + 10, "LUOGO/INDIRIZZO")]
    entities = [{"start": a, "end": b, "type": c, "text": text[a:b], "annotated_text": text[a:b], "status": "located", "origin": "source_annotation"} for a, b, c in spans]
    return {"note_id": f"gold_{i:03d}", "index": i, "text": text, "entities": entities, "document_id": f"IT{i}", "extra_occurrences": [], "redacted_text": render_inline(text, spans), "gold_version": "v2"}


def make_project(tmp_path, n_notes=25):
    gold = tmp_path / "gold.jsonl"
    gold.write_text("".join(json.dumps(gold_row(i), ensure_ascii=False) + "\n" for i in range(n_notes)), encoding="utf-8")
    split_dir = tmp_path / "splits"
    specs = splits.freeze_folds(gold, split_dir, tmp_path / "fold_assignments.csv")
    return {"gold": gold, "splits": split_dir, "specs": {s.fold: s for s in specs}, "root": tmp_path / "synthetic_data", "tmp": tmp_path}


def synthetic_record(fold: int, arm: str, order: int, kind: str, type_order: int, anchors=()) -> dict:
    tag = f"f{fold}{arm[0]}{order}"
    if kind == "synthetic_negative":
        text, spans = f"Caso {tag} senza dati identificativi, decorso regolare.", []
    else:
        age = str(30 + order % 50)
        text = f"Caso {tag}: di {age} anni, a Borgo{tag}."
        spans = [(text.index(f"di {age}") + 3, text.index(f"di {age}") + 3 + len(age), "ETÀ"), (text.index(f"Borgo{tag}"), text.index(f"Borgo{tag}") + len(f"Borgo{tag}"), "LUOGO/INDIRIZZO")]
        if kind == "synthetic_named":
            text = f"{text} Paziente Nominato{tag}."
            spans.append((text.index(f"Nominato{tag}"), text.index(f"Nominato{tag}") + len(f"Nominato{tag}"), "NOME"))
    return {"record_id": f"syn_f{fold}_{arm}_{order:05d}", "source_record_id": f"F{fold}-N{order:05d}", "record_type": kind, "fold_id": fold, "partition": "train", "type_order": type_order, "anchor_ids": list(anchors),
            "text": text, "redacted_text": render_inline(text, spans), "entities": [{"start": a, "end": b, "type": c, "text": text[a:b]} for a, b, c in spans], "permitted_for_training": False}


def write_run(project, fold: int, arm: str, counts=(6, 3, 2), skip_orders=(), official=False, audit="passed", root=None, mutate=None, anchors=None):
    """Generated run files. `skip_orders` models cases the validator rejected. `mutate` edits records to create faults."""
    spec = project["specs"][fold]
    run_dir = (root or project["root"]) / f"fold_{fold}" / arm
    run_dir.mkdir(parents=True, exist_ok=True)
    kinds = ["synthetic_named"] * counts[0] + ["synthetic_name_free"] * counts[1] + ["synthetic_negative"] * counts[2]
    seen, records = {}, []
    use = anchors if anchors is not None else (list(spec.train_ids[:3]) if arm == "style_anchored" else [])
    for order, kind in enumerate(kinds):
        seen[kind] = seen.get(kind, -1) + 1
        records.append(synthetic_record(fold, arm, order, kind, seen[kind], use))
    manifest = []
    kept = []
    for r in records:
        order = int(r["record_id"][-5:])
        ok = order not in skip_orders
        manifest.append({"record_id": r["record_id"], "case_id": r["source_record_id"], "status": "accepted" if ok else "rejected", "permitted_for_training": str(ok and audit == "passed")})
        if ok:
            kept.append(r)
    if mutate:
        mutate(kept, manifest)
    (run_dir / "examples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    with open(run_dir / "generation_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["record_id", "case_id", "status", "permitted_for_training"])
        w.writeheader()
        w.writerows(manifest)
    (run_dir / "validation_report.json").write_text(json.dumps({"fold": fold, "arm": arm, "post_audit": audit, "permitted_for_training": audit == "passed"}), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps({"fold": fold, "arm": arm, "gold_sha256": spec.gold_sha256, "plan_sha256": "planhash", "git_commit": "abc", "official_run": official, "git_dirty": not official}), encoding="utf-8")
    return run_dir
