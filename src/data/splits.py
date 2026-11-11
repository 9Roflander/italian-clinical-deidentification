"""Frozen cross-validation folds.

The first 80 notes keep the historical assignment (KFold, shuffle=True, random_state=42). Later
notes are added to folds by a seeded draw, without moving any existing note. This was fixed before
any clean score existed. Files are written once and never changed afterwards.

Generation code must use `load_train_notes`, which never returns test-note text.
"""
from __future__ import annotations

import csv
import json
import pathlib
from dataclasses import dataclass

from src.data.loaders import DEFAULT_GOLD, ROOT, GoldNote, file_sha256, gold_id, load_gold

SPLITS_DIR = ROOT / "splits"
FOLD_CSV = ROOT / "data_manifests" / "fold_assignments.csv"
N_FOLDS = 5
SEED = 42
HISTORICAL_NOTES = 80


class LeakageError(RuntimeError):
    """A test note or its derivative reached a step where it must not exist."""


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    gold_sha256: str
    seed: int

    def check(self) -> None:
        if set(self.train_ids) & set(self.test_ids):
            raise LeakageError(f"fold {self.fold}: train and test IDs overlap")


def compute_assignments(n_notes: int, seed: int = SEED) -> list[list[int]]:
    import random

    import numpy as np
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    base = min(n_notes, HISTORICAL_NOTES)
    folds = [sorted(int(i) for i in te) for _, te in kf.split(np.arange(base))]
    rng = random.Random(f"{seed}:added_notes")
    order = rng.sample(range(N_FOLDS), N_FOLDS)
    for k, note in enumerate(range(base, n_notes)):
        folds[order[k % N_FOLDS]].append(note)
    return folds


def freeze_folds(gold_path=DEFAULT_GOLD, splits_dir=SPLITS_DIR, fold_csv=FOLD_CSV) -> list[FoldSpec]:
    """Write fold files. Refuses to replace existing files with a different assignment."""
    import sklearn
    notes = load_gold(gold_path)
    sha = file_sha256(gold_path)
    assignments = compute_assignments(len(notes))
    splits_dir, fold_csv = pathlib.Path(splits_dir), pathlib.Path(fold_csv)
    splits_dir.mkdir(parents=True, exist_ok=True)
    fold_csv.parent.mkdir(parents=True, exist_ok=True)
    specs = []
    for k, te in enumerate(assignments, 1):
        test_ids = tuple(gold_id(i) for i in te)
        train_ids = tuple(n.note_id for n in notes if n.note_id not in set(test_ids))
        spec = FoldSpec(k, train_ids, test_ids, sha, SEED)
        spec.check()
        payload = {"fold": k, "train_ids": list(train_ids), "test_ids": list(test_ids), "gold_sha256": sha,
                   "seed": SEED, "method": "KFold(n_splits=5, shuffle=True, random_state=42) on the first 80 notes; later notes by seeded draw",
                   "sklearn_version": sklearn.__version__}
        path = splits_dir / f"fold_{k}.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old["test_ids"] != payload["test_ids"] or old["gold_sha256"] != sha:
                raise LeakageError(f"{path.name} already exists with a different assignment")
        else:
            path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        specs.append(spec)
    with open(fold_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["note_id", "gold_index", "test_fold"])
        fold_of = {n: s.fold for s in specs for n in s.test_ids}
        for n in notes:
            w.writerow([n.note_id, n.index, fold_of[n.note_id]])
    return specs


def load_fold(fold: int, splits_dir=SPLITS_DIR) -> FoldSpec:
    d = json.loads((pathlib.Path(splits_dir) / f"fold_{fold}.json").read_text(encoding="utf-8"))
    spec = FoldSpec(d["fold"], tuple(d["train_ids"]), tuple(d["test_ids"]), d["gold_sha256"], d["seed"])
    spec.check()
    return spec


def load_train_notes(spec: FoldSpec, gold_path=DEFAULT_GOLD) -> list[GoldNote]:
    """Training-partition notes only. Test-note text is dropped before this returns."""
    if file_sha256(gold_path) != spec.gold_sha256:
        raise LeakageError("gold file does not match the frozen fold manifest")
    keep = set(spec.train_ids)
    return [n for n in load_gold(gold_path) if n.note_id in keep]
