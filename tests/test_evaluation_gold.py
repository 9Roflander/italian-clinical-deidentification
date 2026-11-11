"""Gold verification runs on the real file, then on tampered copies. Failures report IDs and counts only."""
import copy
import json
import shutil

import pytest

from src.data.loaders import GOLD_V2, ROOT, file_sha256
from src.data.splits import FOLD_CSV, SPLITS_DIR
from src.evaluation.alignment import render_inline
from src.evaluation.gold_verification import GoldVerificationError, verify_gold
from src.evaluation.runner import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def tampered(tmp_path, cfg):
    """Rewrites a copy of gold_v2 and its fold files, so the check under test is the one that fires."""
    def make(change):
        rows = [json.loads(x) for x in GOLD_V2.read_text(encoding="utf-8").splitlines() if x.strip()]
        change(rows)
        path = tmp_path / "gold_v2.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        new_cfg = copy.deepcopy(cfg)
        new_cfg["gold"]["sha256"] = file_sha256(path)
        splits = tmp_path / "splits"
        splits.mkdir(exist_ok=True)
        for k in range(1, 6):
            d = json.loads((SPLITS_DIR / f"fold_{k}.json").read_text(encoding="utf-8"))
            d["gold_sha256"] = new_cfg["gold"]["sha256"]
            (splits / f"fold_{k}.json").write_text(json.dumps(d), encoding="utf-8")
        return new_cfg, path, splits
    return make


def test_the_approved_gold_set_passes(cfg):
    gold = verify_gold(cfg)
    assert len(gold.order) == 82 and gold.version == "v2"
    assert sum(len(v) for v in gold.spans.values()) == 252
    assert gold.fold_sizes == {1: 16, 2: 17, 3: 17, 4: 16, 5: 16}


def test_approved_counts_match_the_protocol(cfg):
    assert cfg["gold"]["mentions"] == {"NOME": 54, "ETÀ": 102, "DATA": 59, "LUOGO/INDIRIZZO": 37} and cfg["gold"]["total_mentions"] == 252


def test_a_changed_file_is_refused(cfg, tmp_path):
    path = tmp_path / "gold_v2.jsonl"
    shutil.copyfile(GOLD_V2, path)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(GoldVerificationError, match="SHA-256"):
        verify_gold(cfg, path)


def test_the_tamper_fixture_alone_passes(tampered):
    new_cfg, path, splits = tampered(lambda rows: None)
    verify_gold(new_cfg, path, splits, FOLD_CSV)


def drop_entity_keeping_target_consistent(rows):
    """Removes a mention and rewrites the stored target too, so only the count check can catch it."""
    row = next(r for r in rows if len(r["entities"]) > 1)
    row["entities"].pop()
    row["redacted_text"] = render_inline(row["text"], [(e["start"], e["end"], e["type"]) for e in row["entities"]])


@pytest.mark.parametrize("change,message", [
    (lambda r: r[0]["entities"].pop(), "stored target"),
    (drop_entity_keeping_target_consistent, "mention counts"),
    (lambda r: r[1]["entities"][0].update(start=r[1]["entities"][0]["start"] + 1), "does not match|outer whitespace"),
    (lambda r: r[1]["entities"][0].update(type="PERSONA"), "unknown category"),
    (lambda r: r[1]["entities"].append(dict(r[1]["entities"][0])), "overlapping"),
    (lambda r: r[1].update(redacted_text=r[1]["redacted_text"] + "x"), "stored target"),
    (lambda r: [x.update(gold_version="v3") for x in r], "gold_version"),
    (lambda r: r.pop(), "notes"),
])
def test_tampered_content_is_refused(tampered, change, message):
    new_cfg, path, splits = tampered(change)
    with pytest.raises(GoldVerificationError, match=message):
        verify_gold(new_cfg, path, splits, FOLD_CSV)


def test_folds_frozen_on_another_file_are_refused(cfg, tampered):
    new_cfg, path, _ = tampered(lambda rows: rows[0].update(unused_field=1))
    with pytest.raises(GoldVerificationError, match="frozen on a different gold file"):
        verify_gold(new_cfg, path, SPLITS_DIR, FOLD_CSV)


def test_changed_fold_counts_are_refused(cfg, tampered):
    new_cfg, path, splits = tampered(lambda rows: None)
    new_cfg["gold"]["folds"][1]["NOME"] += 1
    with pytest.raises(GoldVerificationError, match="fold 1 counts"):
        verify_gold(new_cfg, path, splits, FOLD_CSV)


def test_a_changed_fold_assignment_csv_is_refused(cfg, tampered, tmp_path):
    new_cfg, path, splits = tampered(lambda rows: None)
    lines = FOLD_CSV.read_text(encoding="utf-8").splitlines()
    first = lines[1].split(",")
    first[2] = "1" if first[2] != "1" else "2"
    lines[1] = ",".join(first)
    csv_path = tmp_path / "fold_assignments.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(GoldVerificationError, match="fold_assignments"):
        verify_gold(new_cfg, path, splits, csv_path)
