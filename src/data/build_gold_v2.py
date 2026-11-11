"""Builds gold_v2: gold_v1 plus the corrections approved by the project owner.

    python -m src.data.build_gold_v2

Input: data_secure/gold/gold_v1.jsonl and the two decision files in data_manifests/.
Output: data_secure/gold/gold_v2.jsonl (git-ignored), data_manifests/gold_v2_change_log.csv and
audit/results/gold_v2_build_summary.json (IDs, offsets and counts only).

gold_v1 is never modified. The note text is identical in both versions; only annotations differ.
The build stops if any flagged item lacks a decision, or if any change fails validation.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re

from src.data.annotations import word_boundary_regex
from src.data.build_gold import build_target, unannotated_patterns
from src.data.loaders import GOLD_V1, GOLD_V2, ROOT
from src.data.manifests import write_csv

PROPOSALS = {"review": ROOT / "data_manifests" / "gold_annotation_proposal.csv",
             "additional": ROOT / "data_manifests" / "gold_annotation_proposal_additional.csv"}
CHANGE_LOG = ROOT / "data_manifests" / "gold_v2_change_log.csv"
SUMMARY = ROOT / "audit" / "results" / "gold_v2_build_summary.json"
ADD_ACTIONS = {"add_nome", "add_luogo", "add_data", "add_place_consistent_with_other_note", "add_age_other_person_number_only", "add_age_number_only"}
# Decided not to be mentions: these numbers and words stay as ordinary text.
EXPECTED_RESIDUAL = {"date": 7, "initials": 2, "age": 0, "title_name": 0, "contact": 0}


class GoldBuildError(RuntimeError):
    """Messages carry IDs and offsets only."""


def read_decisions() -> list[dict]:
    rows = []
    for name, path in PROPOSALS.items():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["proposal"] = name
                rows.append(r)
    return rows


def span_of(r: dict) -> tuple[int, int]:
    return int(r["start"]), int(r["end"])


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and a[1] > b[0]


def validate_value(note_id: str, etype: str, value: str, span) -> None:
    ok = {"DATA": bool(re.search(r"\d", value)), "ETÀ": bool(re.fullmatch(r"\d{1,3}", value)),
          "NOME": bool(re.search(r"[^\W\d_]", value)) and not re.search(r"\d", value),
          "LUOGO/INDIRIZZO": bool(re.search(r"[^\W\d_]", value))}[etype]
    if not ok or value != value.strip():
        raise GoldBuildError(f"{note_id}: {etype} at {span} fails its format check")


def apply_decisions(records: dict, decisions: list[dict]) -> list[dict]:
    log, adds, locates = [], {}, {}
    for r in decisions:
        a = r["action"]
        if a in ADD_ACTIONS:
            adds.setdefault((r["note_id"], *span_of(r), r["type"]), []).append(r)
        elif a == "locate_variant_mention":
            locates.setdefault((r["note_id"], *span_of(r), r["type"]), []).append(r)

    def entry(note_id, action, etype, start, end, sources, reason):
        log.append({"note_id": note_id, "document_id": records[note_id]["document_id"], "action": action, "type": etype, "start": start, "end": end,
                    "decision_rows": ";".join(f"{x['proposal']}:{x['row']}" for x in sources), "reason": reason})

    for r in decisions:
        rec, a = records[r["note_id"]], r["action"]
        if a == "remove_age_is_a_duration":
            hit = [e for e in rec["entities"] if (e["start"], e["end"]) == span_of(r) and e["type"] == "ETÀ"]
            if len(hit) != 1:
                raise GoldBuildError(f"{r['note_id']}: age to remove not found at {span_of(r)}")
            rec["entities"].remove(hit[0])
            entry(r["note_id"], "remove_mention", "ETÀ", r["start"], r["end"], [r], "duration_not_an_age")
        elif a == "remove_surplus_listing":
            hit = [e for e in rec["entities"] if e["status"] == "annotated_more_often_than_occurs"]
            if len(hit) != 1:
                raise GoldBuildError(f"{r['note_id']}: expected exactly one surplus listing")
            rec["entities"].remove(hit[0])
            entry(r["note_id"], "remove_mention", hit[0]["type"], "", "", [r], "listed_more_often_than_it_occurs")
        elif a == "correct_source_typo":
            hit = [e for e in rec["entities"] if e["status"] == "located_fuzzy_variant"]
            if len(hit) != 1:
                raise GoldBuildError(f"{r['note_id']}: expected exactly one source typo")
            hit[0]["status"] = "corrected_source_typo"
            entry(r["note_id"], "correct_mention", hit[0]["type"], hit[0]["start"], hit[0]["end"], [r], "annotation_differs_from_note_by_one_character")

    for (nid, s, e, t), src in sorted(locates.items()):
        rec = records[nid]
        hit = [x for x in rec["entities"] if x["status"] == "annotated_more_often_than_occurs" and x["type"] == t]
        if len(hit) != 1:
            raise GoldBuildError(f"{nid}: expected one unlocated {t} listing")
        validate_value(nid, t, rec["text"][s:e], (s, e))
        hit[0].update(text=rec["text"][s:e], start=s, end=e, status="located_variant_in_review")
        entry(nid, "locate_mention", t, s, e, src, "same_entity_written_differently")

    for (nid, s, e, t), src in sorted(adds.items()):
        rec = records[nid]
        value = rec["text"][s:e]
        validate_value(nid, t, value, (s, e))
        for x in rec["entities"]:
            if x["start"] is not None and overlaps((s, e), (x["start"], x["end"])):
                raise GoldBuildError(f"{nid}: new {t} at {(s, e)} overlaps an existing mention")
        rec["entities"].append({"type": t, "text": value, "annotated_text": value, "start": s, "end": e, "status": "added_in_review"})
        entry(nid, "add_mention", t, s, e, src, "missing_from_source_annotations")
    return log


def main() -> dict:
    records = {}
    for line in GOLD_V1.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        records[r["note_id"]] = r
    v1_text = {k: hashlib.sha256(v["text"].encode("utf-8")).hexdigest() for k, v in records.items()}
    decisions = read_decisions()

    # Every extra occurrence in gold_v1 must have an explicit decision
    decided = collections.defaultdict(set)
    for r in decisions:
        if r["start"] and r["proposal"] == "review":
            decided[r["note_id"]].add(span_of(r)[0])
    undecided = [(k, x["start"]) for k, v in records.items() for x in v["extra_occurrences"] if x["start"] not in decided[k]]
    if undecided:
        raise GoldBuildError(f"extra occurrences without a decision: {undecided}")

    log = apply_decisions(records, decisions)
    reviewed = 0
    for k, r in records.items():
        for e in r["entities"]:
            if e["status"] == "located_ambiguous":
                e["status"] = "located_reviewed"
                reviewed += 1
            e["origin"] = "added_in_review" if e["status"] == "added_in_review" else "source_annotation"
        r["entities"].sort(key=lambda e: (e["start"] is None, e["start"] or 0))
        r["extra_occurrences"] = []
        r["redacted_text"] = build_target(r["text"], r["entities"], [])
        r["gold_version"] = "v2"
        spans = sorted((e["start"], e["end"]) for e in r["entities"] if e["start"] is not None)
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise GoldBuildError(f"{k}: overlapping mentions")
    if any(hashlib.sha256(r["text"].encode("utf-8")).hexdigest() != v1_text[k] for k, r in records.items()):
        raise GoldBuildError("note text changed")

    with open(GOLD_V2, "w", encoding="utf-8", newline="\n") as f:
        for r in records.values():
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    log.sort(key=lambda x: (x["note_id"], int(x["start"]) if x["start"] != "" else -1))
    write_csv(CHANGE_LOG, ["note_id", "document_id", "action", "type", "start", "end", "decision_rows", "reason"], log)

    # Post-build checks
    ents = [e for r in records.values() for e in r["entities"]]
    residual = collections.Counter()
    residual_notes = collections.defaultdict(list)
    for k, r in records.items():
        for name, v in unannotated_patterns(r["redacted_text"]).items():
            residual[name] += v
            if v:
                residual_notes[name].append(k)
    uncovered = []
    for k, r in records.items():
        spans = [(e["start"], e["end"]) for e in r["entities"]]
        for s in {e["text"] for e in r["entities"] if e["type"] in ("NOME", "LUOGO/INDIRIZZO") and len(e["text"]) >= 3}:
            for m in word_boundary_regex(s).finditer(r["text"]):
                if not any(overlaps(m.span(), sp) for sp in spans):
                    uncovered.append((k, m.start()))
    summary = {
        "notes": len(records), "mentions": len(ents), "by_type": dict(collections.Counter(e["type"] for e in ents)), "status": dict(collections.Counter(e["status"] for e in ents)),
        "origin": dict(collections.Counter(e["origin"] for e in ents)), "mentions_without_offsets": sum(e["start"] is None for e in ents),
        "offsets_verified": sum(r["text"][e["start"] : e["end"]] == e["text"] for r in records.values() for e in r["entities"]),
        "changes": dict(collections.Counter(x["action"] for x in log)), "changes_by_type": {f"{x}|{y}": v for (x, y), v in sorted(collections.Counter((c["action"], c["type"]) for c in log).items())},
        "notes_changed": len({c["note_id"] for c in log}), "ambiguous_mentions_now_reviewed": reviewed, "decisions_read": len(decisions),
        "decisions_by_action": dict(collections.Counter(d["action"] for d in decisions)),
        "notes_without_annotation": [k for k, r in records.items() if not r["entities"]],
        "residual_patterns": dict(residual), "residual_notes": dict(residual_notes), "residual_matches_rejected_rows": all(residual[k] == v for k, v in EXPECTED_RESIDUAL.items()),
        "name_or_place_string_left_unannotated": uncovered, "gold_v1_sha256": hashlib.sha256(GOLD_V1.read_bytes()).hexdigest(), "gold_v2_sha256": hashlib.sha256(GOLD_V2.read_bytes()).hexdigest()}
    SUMMARY.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=0, ensure_ascii=False))
