"""Writes audit/gold_dataset_statistics.csv: aggregate statistics only, no note text."""
from __future__ import annotations

import collections
import csv

from src.data.annotations import ENTITY_TYPES, audit_note_annotations, canonical_type, word_boundary_regex
from src.data.loaders import DEFAULT_GOLD, ROOT, load_gold
from src.data.splits import load_fold


def main(gold_path=DEFAULT_GOLD, out=ROOT / "audit" / "gold_dataset_statistics.csv") -> list[tuple]:
    notes = load_gold(gold_path)
    rows: list[tuple] = []
    add = lambda section, metric, value: rows.append((section, metric, value))
    add("corpus", "notes", len(notes))
    add("corpus", "total_annotated_mentions", sum(len(n.entities) for n in notes))
    add("corpus", "mentions_with_verified_offsets", sum(e.get("start") is not None and n.text[e["start"] : e["end"]] == e["text"] for n in notes for e in n.entities))
    add("corpus", "notes_with_document_id", sum(bool(n.document_id) for n in notes))
    lengths = sorted(len(n.text) for n in notes)
    last = len(lengths) - 1
    for frac, name in [(0, "min"), (0.1, "p10"), (0.25, "p25"), (0.5, "median"), (0.75, "p75"), (0.9, "p90"), (1, "max")]:
        add("length_chars", name, lengths[round(frac * last)])
    add("length_chars", "notes_under_1000", sum(x < 1000 for x in lengths))
    per_note = collections.Counter()
    for n in notes:
        c = collections.Counter(canonical_type(e.get("type")) for e in n.entities)
        per_note[sum(v for k, v in c.items() if k)] += 1
    for k in sorted(per_note):
        add("mentions_per_note", f"notes_with_{k}_mentions", per_note[k])
    for t in ENTITY_TYPES:
        vals = [sum(canonical_type(e.get("type")) == t for e in n.entities) for n in notes]
        add(t, "mentions", sum(vals))
        add(t, "mentions_per_note", round(sum(vals) / len(notes), 3))
        add(t, "notes_with_none", sum(v == 0 for v in vals))
        add(t, "pct_notes_with_none", round(100 * sum(v == 0 for v in vals) / len(notes), 1))
    status = collections.Counter()
    repeats = nested = 0
    for n in notes:
        strs = collections.Counter(e.get("text", "") for e in n.entities)
        repeats += sum(v > 1 for v in strs.values())
        nested += sum(1 for a in strs for b in strs if a != b and a in b)
        for r in audit_note_annotations(n.text, n.entities):
            status[(r["type"], r["status"])] += 1
    add("annotation_strings", "strings_repeated_within_a_note", repeats)
    add("annotation_strings", "strings_nested_inside_another", nested)
    for (t, s), v in sorted(status.items(), key=str):
        add("annotation_vs_text", f"{t}|{s}", v)
    terciles = [lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]]
    by_len = collections.defaultdict(list)
    for n in notes:
        by_len["short" if len(n.text) <= terciles[0] else "medium" if len(n.text) <= terciles[1] else "long"].append(len(n.entities))
    for k, v in by_len.items():
        add("mentions_by_length_tercile", k, round(sum(v) / len(v), 2))
    index = {n.note_id: n for n in notes}
    status_counts = collections.Counter(e.get("status", "") for n in notes for e in n.entities)
    for k, v in sorted(status_counts.items()):
        add("mention_status", k, v)
    for k in range(1, 6):
        spec = load_fold(k)
        add(f"test_fold_{k}", "notes", len(spec.test_ids))
        c = collections.Counter(canonical_type(e.get("type")) for i in spec.test_ids for e in index[i].entities)
        for t in ENTITY_TYPES:
            add(f"test_fold_{k}", t, c[t])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "metric", "value"])
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    print(len(main()), "rows written")
