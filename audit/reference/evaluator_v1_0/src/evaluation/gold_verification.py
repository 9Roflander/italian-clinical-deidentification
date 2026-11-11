"""Verify the approved gold set before any result is calculated. Errors name IDs and counts, never text."""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import unicodedata
from dataclasses import dataclass

from src.data.loaders import GOLD_V2, GoldNote, file_sha256, load_gold
from src.data.splits import FOLD_CSV, N_FOLDS, SPLITS_DIR, load_fold
from src.evaluation.alignment import CATEGORIES, CATEGORY_OF, render_inline


class GoldVerificationError(RuntimeError):
    """The gold set or folds differ from what was approved."""


@dataclass(frozen=True)
class VerifiedGold:
    notes: dict
    order: tuple
    fold_of: dict
    spans: dict
    sha256: str
    version: str
    fold_file_sha256: dict

    @property
    def fold_sizes(self) -> dict:
        return dict(sorted(collections.Counter(self.fold_of.values()).items()))


def spans_of(note: GoldNote) -> list[tuple[int, int, str]]:
    return sorted((e["start"], e["end"], e["type"]) for e in note.entities)


def verify_gold(cfg: dict, path=GOLD_V2, splits_dir=SPLITS_DIR, fold_csv=FOLD_CSV) -> VerifiedGold:
    expected = cfg["gold"]
    path = pathlib.Path(path)
    sha = file_sha256(path)
    if sha != expected["sha256"]:
        raise GoldVerificationError(f"{path.name}: SHA-256 {sha[:12]} is not the approved {expected['sha256'][:12]}")
    versions = {json.loads(x)["gold_version"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}
    if versions != {expected["version"]}:
        raise GoldVerificationError(f"gold_version is {sorted(versions)}, expected {expected['version']}")
    notes = load_gold(path)
    ids = [n.note_id for n in notes]
    if len(ids) != expected["notes"] or len(set(ids)) != len(ids):
        raise GoldVerificationError(f"{len(ids)} notes ({len(set(ids))} unique), expected {expected['notes']}")

    spans, counts = {}, collections.Counter()
    for n in notes:
        s = spans_of(n)
        for start, end, label in s:
            if label not in CATEGORIES:
                raise GoldVerificationError(f"{n.note_id}: unknown category")
            if not 0 <= start < end <= len(n.text):
                raise GoldVerificationError(f"{n.note_id}: offsets out of range")
            if n.text[start:end] != n.text[start:end].strip():
                raise GoldVerificationError(f"{n.note_id}: span at {start} has outer whitespace")
            counts[label] += 1
        if any(a[1] > b[0] for a, b in zip(s, s[1:])):
            raise GoldVerificationError(f"{n.note_id}: overlapping gold spans")
        if any(n.text[e["start"] : e["end"]] != e["text"] for e in n.entities):
            raise GoldVerificationError(f"{n.note_id}: entity text does not match its offsets")
        if not unicodedata.is_normalized("NFC", n.text):
            raise GoldVerificationError(f"{n.note_id}: text is not NFC")
        if any(p in n.text for p in CATEGORY_OF):
            raise GoldVerificationError(f"{n.note_id}: source contains a placeholder")
        if render_inline(n.text, s) != n.redacted_text:
            raise GoldVerificationError(f"{n.note_id}: stored target differs from the rendered spans")
        spans[n.note_id] = s
    if dict(counts) != expected["mentions"] or sum(counts.values()) != expected["total_mentions"]:
        raise GoldVerificationError(f"mention counts {dict(counts)} differ from the approved {expected['mentions']}")
    empty = sorted(k for k, v in spans.items() if not v)
    if empty != expected["notes_without_mentions"]:
        raise GoldVerificationError(f"notes without mentions {empty}, expected {expected['notes_without_mentions']}")

    fold_of, fold_sha = {}, {}
    for k in range(1, N_FOLDS + 1):
        spec = load_fold(k, splits_dir)
        if spec.gold_sha256 != sha:
            raise GoldVerificationError(f"fold {k} was frozen on a different gold file")
        if set(spec.train_ids) | set(spec.test_ids) != set(ids):
            raise GoldVerificationError(f"fold {k} does not partition the notes")
        for i in spec.test_ids:
            if i in fold_of:
                raise GoldVerificationError(f"{i} is held out in two folds")
            fold_of[i] = k
        fold_sha[k] = file_sha256(pathlib.Path(splits_dir) / f"fold_{k}.json")
    if set(fold_of) != set(ids):
        raise GoldVerificationError("some notes are held out in no fold")
    with open(fold_csv, newline="", encoding="utf-8") as f:
        listed = {r["note_id"]: int(r["test_fold"]) for r in csv.DictReader(f)}
    if listed != fold_of:
        raise GoldVerificationError("fold_assignments.csv differs from the frozen fold files")
    for k, want in expected["folds"].items():
        members = [i for i in ids if fold_of[i] == k]
        got = {"notes": len(members), **{c: sum(s[2] == c for i in members for s in spans[i]) for c in CATEGORIES}}
        if got != want:
            raise GoldVerificationError(f"fold {k} counts {got} differ from the approved {want}")
    return VerifiedGold({n.note_id: n for n in notes}, tuple(ids), fold_of, spans, sha, expected["version"], fold_sha)
