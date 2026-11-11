"""Guide test 4: split integrity, on the real frozen folds and through the loader."""
import json

import pytest

from src.data.loaders import DEFAULT_GOLD, file_sha256, gold_id
from src.data.splits import LeakageError, SPLITS_DIR, compute_assignments, freeze_folds, load_fold, load_train_notes


def test_real_folds_are_disjoint_and_cover_all_notes():
    seen = []
    for k in range(1, 6):
        s = load_fold(k)
        assert len(s.train_ids) + len(s.test_ids) == 82 and len(s.test_ids) in (16, 17)
        assert not set(s.train_ids) & set(s.test_ids)
        seen += s.test_ids
    assert len(seen) == 82 and len(set(seen)) == 82


def test_original_80_notes_keep_their_historical_folds():
    import numpy as np
    from sklearn.model_selection import KFold
    expected = [sorted(int(i) for i in te) for _, te in KFold(5, shuffle=True, random_state=42).split(np.arange(80))]
    for k in range(1, 6):
        kept = [i for i in load_fold(k).test_ids if int(i.split("_")[1]) < 80]
        assert kept == [gold_id(i) for i in expected[k - 1]]


def test_added_notes_are_each_in_exactly_one_fold():
    for note in ("gold_080", "gold_081"):
        assert sum(note in load_fold(k).test_ids for k in range(1, 6)) == 1


def test_small_datasets_use_plain_kfold():
    assert sorted(i for f in compute_assignments(25) for i in f) == list(range(25))


def test_frozen_folds_match_current_gold_file():
    assert load_fold(1).gold_sha256 == file_sha256(DEFAULT_GOLD)


@pytest.mark.parametrize("k", range(1, 6))
def test_loader_never_returns_test_notes(k):
    s = load_fold(k)
    ids = {n.note_id for n in load_train_notes(s)}
    assert ids == set(s.train_ids)


def test_loader_refuses_changed_gold_file(project, tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps([{"text": "x", "entities": []}]), encoding="utf-8")
    with pytest.raises(LeakageError):
        load_train_notes(project["specs"][0], other)


def test_overlap_in_a_fold_spec_is_rejected(project):
    from dataclasses import replace
    s = project["specs"][0]
    with pytest.raises(LeakageError):
        replace(s, train_ids=s.train_ids + (s.test_ids[0],)).check()


def test_freeze_refuses_a_different_assignment(project):
    path = project["splits"] / "fold_1.json"
    d = json.loads(path.read_text())
    d["test_ids"] = d["test_ids"][::-1]
    path.write_text(json.dumps(d))
    with pytest.raises(LeakageError):
        freeze_folds(project["gold"], project["splits"], project["tmp"] / "x.csv")
