"""Builds the clean gold set from the original annotated CSV of Miranda et al. 2025.

    python -m src.data.build_gold

Input: the source CSV, pinned by commit and hash, downloaded if not already in data_secure/.
Output: data_secure/gold/gold_v1.jsonl and annotation_review.csv (both git-ignored), plus
data_manifests/gold_quality_report.csv and audit/results/gold_build_summary.json (IDs and counts only).

Source annotations are never edited. Each mention keeps its source string, and how it was located
is recorded in its status, so uncertain cases stay visible for review.
"""
from __future__ import annotations

import ast
import collections
import csv
import difflib
import hashlib
import io
import json
import pathlib
import re
import unicodedata
import urllib.parse
import urllib.request

from src.data.annotations import ENTITY_TYPES, PLACEHOLDERS, canonical_type, word_boundary_regex
from src.data.loaders import ROOT, gold_id
from src.data.manifests import write_csv
from src.generation.privacy_screening import AGE_CONTEXT, EMAIL, INITIALS, INITIALS_ALLOWED, MONTH, NUMERIC_DATE, PHONE, TITLE_NAME, YEAR

SOURCE_REPO = "michele17284/Italian-Clinical-Note-Deidentification"
SOURCE_COMMIT = "e8609c92f870eb6f0e0ca12e890667b7a22a0715"
SOURCE_FILE = "Annotated clinical notes - samples.csv"
SOURCE_SHA256 = "6be1cf9d0d2f9f199ba7c3e0a95636081c99023ae54fea878aa6202a36fea9f7"
GOLD_DIR = ROOT / "data_secure" / "gold"
RAW_CSV = GOLD_DIR / "raw" / SOURCE_FILE
OUT_JSONL = GOLD_DIR / "gold_v1.jsonl"
REVIEW_CSV = GOLD_DIR / "annotation_review.csv"
OLD_GOLD = ROOT / "original_work" / "data" / "gold_standard_80.json"

ID_PREFIX = re.compile(r"(\d+)\|t\|")
RELATION_HEAD = r"\s+([A-Za-z_]+)\s+(\d+)-(\d+)\s+(\d+)-(\d+)\s"   # separators are tabs in most rows, spaces in three
AGE_AFTER = re.compile(r"\s*-?\s*(?:anni|anno|aa\b|mesi|mese|settimane|giorni|enne)", re.I)
AGE_BEFORE = re.compile(r"(?:età\s+di|aa\.?|a\.)\s*$", re.I)
FUZZY_MIN = 0.9


def fetch_source(path=RAW_CSV) -> bytes:
    path = pathlib.Path(path)
    if not path.exists():
        url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/{urllib.parse.quote(SOURCE_FILE)}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("source CSV does not match the pinned hash")
    return raw


def split_input(cell: str) -> tuple[str, str, str]:
    """A cell is '<docid>|t|<narrative> <relation lines>'. Returns id, narrative, relation tail."""
    m = ID_PREFIX.match(cell)
    if not m:
        raise ValueError("cell has no document ID prefix")
    doc = m.group(1)
    body = cell[m.end():]
    tail = re.search(r"\s" + doc + RELATION_HEAD, body)
    return doc, body[: tail.start()] if tail else body, body[tail.start():] if tail else ""


def relation_offsets_ok(narrative: str, doc: str, tail: str) -> tuple[int, int, bool]:
    """Every relation's offsets must point at the two mentions listed after them, which proves where the
    narrative ends. The third value is False if any text in the tail lies outside a relation line."""
    if not tail:
        return 0, 0, True
    pattern = re.compile(doc + RELATION_HEAD)
    heads = list(pattern.finditer(tail))
    ok = 0
    explained = not tail[: heads[0].start()].strip() if heads else False
    for i, h in enumerate(heads):
        rest = tail[h.end() : heads[i + 1].start() if i + 1 < len(heads) else len(tail)]
        a = narrative[int(h.group(2)) : int(h.group(3))].strip()
        b = narrative[int(h.group(4)) : int(h.group(5))].strip()
        squeeze = lambda x: " ".join(x.split())
        listed = squeeze(rest)
        ok += bool(a and b) and listed in (squeeze(f"{a} {b}"), squeeze(f"{b} {a}"))
    return ok, len(heads), explained


def _has_age_context(text: str, start: int, end: int) -> bool:
    return bool(AGE_AFTER.match(text, end) or AGE_BEFORE.search(text[max(0, start - 12) : start]))


def _fuzzy_locate(text: str, s: str) -> list[tuple[int, int]]:
    """Single best word-aligned window, only if clearly similar and unique."""
    scored = collections.defaultdict(set)
    for L in range(max(1, len(s) - 3), len(s) + 4):
        for i in range(0, max(1, len(text) - L + 1)):
            if (i > 0 and text[i - 1].isalnum()) or (i + L < len(text) and text[i + L].isalnum()):
                continue
            window = text[i : i + L]
            if window != window.strip():
                continue
            r = difflib.SequenceMatcher(None, s, window, autojunk=False).ratio()
            if r >= FUZZY_MIN:
                scored[round(r, 6)].add((i, i + L))
    if not scored:
        return []
    best = sorted(scored[max(scored)])
    return best[:1] if len(best) == 1 else []


def recover_spans(text: str, annotations: list[dict]) -> tuple[list[dict], list[dict]]:
    """Map (type, string) annotations onto character offsets, longest string first.

    Returns one entity per source mention, in source count, plus extra occurrences:
    the same string appearing more often than it was annotated.
    """
    groups = collections.Counter((canonical_type(a.get("type")), a.get("text", "")) for a in annotations)
    entities, extras, claimed = [], [], []
    free = lambda sp: not any(sp[0] < e and sp[1] > s for s, e in claimed)
    for (t, s), k in sorted(groups.items(), key=lambda kv: (-len(kv[0][1]), str(kv[0]))):
        def add(span, status):
            entities.append({"type": t, "text": text[span[0] : span[1]] if span else s, "annotated_text": s,
                             "start": span[0] if span else None, "end": span[1] if span else None, "status": status})
        all_bnd = [m.span() for m in word_boundary_regex(s).finditer(text)]
        cands, status = [sp for sp in all_bnd if free(sp)], "located"
        nested = [sp for sp in all_bnd if not free(sp)]
        if not all_bnd:
            cands, status = [m.span() for m in re.finditer(re.escape(s), text) if free(m.span())], "located_inside_longer_token"
        if not all_bnd and not cands:
            pattern = r"\s+".join(map(re.escape, s.split()))
            cands, status = [m.span() for m in re.finditer(pattern, text) if free(m.span())], "located_whitespace_variant"
        if not all_bnd and not cands:
            cands, status = _fuzzy_locate(text, s), "located_fuzzy_variant"
        if len(cands) > k:
            if t == "ETÀ":
                cands = sorted(cands, key=lambda sp: (not _has_age_context(text, *sp), sp[0]))
            chosen, rest = sorted(cands[:k]), sorted(cands[k:])
            status = "located_ambiguous"
            extras += [{"type": t, "start": a, "end": b} for a, b in rest]
        else:
            chosen = cands
        for sp in chosen:
            claimed.append(sp)
            add(sp, status)
        missing = k - len(chosen)
        for sp in nested[: missing]:
            add(sp, "nested_in_longer_mention")
        for _ in range(missing - len(nested[: missing])):
            add(None, "annotated_more_often_than_occurs")
    return sorted(entities, key=lambda e: (e["start"] is None, e["start"] or 0)), sorted(extras, key=lambda e: e["start"])


def build_target(text: str, entities: list[dict], extras: list[dict]) -> str:
    """Redaction target. Extra occurrences of the same name, place or date are PII too; extra ages are ordinary numbers."""
    spans = [(e["start"], e["end"], e["type"]) for e in entities if e["start"] is not None and e["status"] != "nested_in_longer_mention"]
    spans += [(e["start"], e["end"], e["type"]) for e in extras if e["type"] != "ETÀ"]
    out, pos = [], 0
    for s, e, t in sorted(spans):
        if s < pos:
            continue
        out += [text[pos:s], PLACEHOLDERS[t]]
        pos = e
    return "".join(out) + text[pos:]


def unannotated_patterns(target: str) -> dict:
    return {"date": len(NUMERIC_DATE.findall(target)) + len(MONTH.findall(target)) + len(YEAR.findall(target)),
            "age": len(AGE_CONTEXT.findall(target)), "title_name": len(TITLE_NAME.findall(target)),
            "initials": sum(m.group(0).replace(" ", "") not in INITIALS_ALLOWED for m in INITIALS.finditer(target)),
            "contact": len(EMAIL.findall(target)) + len(PHONE.findall(target))}


def _context(text: str, start: int, end: int, width: int = 40) -> str:
    return text[max(0, start - width) : min(len(text), end + width)].replace("\n", " ")


def main() -> dict:
    csv.field_size_limit(10 ** 9)
    raw = fetch_source()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    # Public release: the historical file is not published; it only feeds the old_gold_relation column.
    old = json.loads(OLD_GOLD.read_text(encoding="utf-8")) if OLD_GOLD.exists() else None
    records, quality, review = [], [], []
    rel_ok = rel_total = 0
    tails_unexplained = []
    for i, row in enumerate(rows):
        doc, narrative, tail = split_input(row["input"])
        ok, total, explained = relation_offsets_ok(narrative, doc, tail)
        rel_ok += ok; rel_total += total
        if not explained:
            tails_unexplained.append(gold_id(i))
        nfc = unicodedata.normalize("NFC", narrative)
        annotations = json.loads(row["annotations_redacted"])
        entities, extras = recover_spans(nfc, annotations)
        target = build_target(nfc, entities, extras)
        patterns = unannotated_patterns(target)
        note_id = gold_id(i)
        records.append({"note_id": note_id, "index": i, "document_id": f"IT{doc}", "text": nfc, "entities": entities, "extra_occurrences": extras,
                        "redacted_text": target, "source_row": i, "source_commit": SOURCE_COMMIT, "normalization": "NFC" if nfc != narrative else "none",
                        "narrative_raw_sha256": hashlib.sha256(narrative.encode("utf-8")).hexdigest()})
        statuses = collections.Counter(e["status"] for e in entities)
        counts = collections.Counter(e["type"] for e in entities)
        if old is None:
            rel, lost = "historical_file_not_available", 0
        elif i < len(old):
            o = old[i]["text"]
            rel = "identical" if o == narrative else "narrative_longer_than_old_file" if narrative.startswith(o) else "other_difference"
            lost = len(narrative) - len(o)
        else:
            rel, lost = "absent_from_old_file", len(narrative)
        quality.append({"note_id": note_id, "document_id": f"IT{doc}", "chars": len(nfc), **{f"n_{t}": counts[t] for t in ENTITY_TYPES}, "extra_occurrences": len(extras),
                        "old_gold_relation": rel, "old_gold_chars_missing": lost, "non_nfc_source": nfc != narrative, "double_spaces": nfc.count("  "),
                        **{f"status_{k}": statuses[k] for k in ["located", "located_ambiguous", "located_inside_longer_token", "located_whitespace_variant", "located_fuzzy_variant", "nested_in_longer_mention", "annotated_more_often_than_occurs"]},
                        **{f"unannotated_{k}": v for k, v in patterns.items()}, "relation_lines": total, "relation_offsets_ok": ok})
        for e in entities:
            if e["status"] != "located":
                sp = (e["start"], e["end"]) if e["start"] is not None else (0, 0)
                review.append({"note_id": note_id, "document_id": f"IT{doc}", "issue": e["status"], "type": e["type"], "annotated": e["annotated_text"], "value_in_note": e["text"], "start": e["start"], "context": _context(nfc, *sp) if e["start"] is not None else ""})
        for e in extras:
            review.append({"note_id": note_id, "document_id": f"IT{doc}", "issue": "extra_occurrence", "type": e["type"], "annotated": "", "value_in_note": nfc[e["start"] : e["end"]], "start": e["start"], "context": _context(nfc, e["start"], e["end"])})
        for name, pats in [("unannotated_date", [NUMERIC_DATE, MONTH, YEAR]), ("unannotated_age", [AGE_CONTEXT]), ("unannotated_title_name", [TITLE_NAME]), ("unannotated_initials", [INITIALS]), ("unannotated_contact", [EMAIL, PHONE])]:
            for p in pats:
                for m in p.finditer(target):
                    if name == "unannotated_initials" and m.group(0).replace(" ", "") in INITIALS_ALLOWED:
                        continue
                    review.append({"note_id": note_id, "document_id": f"IT{doc}", "issue": name, "type": "", "annotated": "", "value_in_note": m.group(0), "start": "", "context": _context(target, m.start(), m.end())})

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    write_csv(REVIEW_CSV, ["note_id", "document_id", "issue", "type", "annotated", "value_in_note", "start", "context"], review)
    write_csv(ROOT / "data_manifests" / "gold_quality_report.csv", list(quality[0]), quality)

    ids = [r["document_id"] for r in records]
    all_ents = [e for r in records for e in r["entities"]]
    offsets_ok = sum(r["text"][e["start"] : e["end"]] == e["text"] for r in records for e in r["entities"] if e["start"] is not None)
    located = sum(e["start"] is not None for e in all_ents)
    summary = {
        "source": {"repo": SOURCE_REPO, "commit": SOURCE_COMMIT, "file": SOURCE_FILE, "sha256": SOURCE_SHA256, "licence_file_present": False},
        "csv_lines_including_header": len(rows) + 1, "notes": len(records),
        "document_ids_unique": len(set(ids)) == len(ids), "document_ids_ascending": all(int(a[2:]) < int(b[2:]) for a, b in zip(ids, ids[1:])),
        "first_last_document_id": [ids[0], ids[-1]],
        "relation_lines": rel_total, "relation_offsets_ok": rel_ok, "tails_not_fully_explained": tails_unexplained,
        "old_gold_relation": dict(collections.Counter(q["old_gold_relation"] for q in quality)),
        "narrative_differs_from_old_file": [{"note_id": q["note_id"], "relation": q["old_gold_relation"], "chars_difference": q["old_gold_chars_missing"]} for q in quality if q["old_gold_relation"] not in ("identical", "absent_from_old_file")],
        "notes_added": [q["note_id"] for q in quality if q["old_gold_relation"] == "absent_from_old_file"],
        "mentions": len(all_ents), "by_type": dict(collections.Counter(e["type"] for e in all_ents)), "mention_status": dict(collections.Counter(e["status"] for e in all_ents)),
        "located_mentions": located, "located_offsets_verified": offsets_ok, "extra_occurrences_by_type": dict(collections.Counter(e["type"] for r in records for e in r["extra_occurrences"])),
        "notes_without_annotation": [r["note_id"] for r in records if not r["entities"]], "non_nfc_notes": [r["note_id"] for r in records if r["normalization"] == "NFC"],
        "notes_needing_review": sorted({x["note_id"] for x in review}), "review_rows_by_issue": dict(collections.Counter(x["issue"] for x in review)),
        "unannotated_pattern_totals": {k: sum(q[f"unannotated_{k}"] for q in quality) for k in ["date", "age", "title_name", "initials", "contact"]},
        "unannotated_pattern_notes": {k: sum(q[f"unannotated_{k}"] > 0 for q in quality) for k in ["date", "age", "title_name", "initials", "contact"]},
        "annotations_column": "not used; 516 items with over 25 labels, and eval.py uses annotations_redacted as the gold standard",
        "output_sha256": hashlib.sha256(OUT_JSONL.read_bytes()).hexdigest()}
    p = ROOT / "audit" / "results" / "gold_build_summary.json"
    p.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=1, ensure_ascii=False))
