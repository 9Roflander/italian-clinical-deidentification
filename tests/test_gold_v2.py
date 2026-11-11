"""gold_v2: reviewed annotations. Failures report IDs and offsets only."""
import collections
import csv
import json
import re

import pytest

from src.data.build_gold import build_target, unannotated_patterns
from src.data.build_gold_v2 import EXPECTED_RESIDUAL, GoldBuildError, apply_decisions, read_decisions
from src.data.loaders import GOLD_V1, GOLD_V2, ROOT, file_sha256, load_gold
from src.data.splits import load_fold

TAG = re.compile(r"\[(?:NOME|ETÀ|DATA|LUOGO/INDIRIZZO)\]")


@pytest.fixture(scope="module")
def v1():
    return {n.note_id: n for n in load_gold(GOLD_V1)}


@pytest.fixture(scope="module")
def v2():
    return {n.note_id: n for n in load_gold(GOLD_V2)}


@pytest.fixture(scope="module")
def log():
    with open(ROOT / "data_manifests" / "gold_v2_change_log.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def spans(note):
    return {(e["start"], e["end"], e["type"]) for e in note.entities if e["start"] is not None}


def test_mention_counts(v2):
    c = collections.Counter(e["type"] for n in v2.values() for e in n.entities)
    assert len(v2) == 82 and c == {"NOME": 54, "ETÀ": 102, "DATA": 59, "LUOGO/INDIRIZZO": 37}


def test_every_mention_has_an_exact_offset(v2):
    bad = [n.note_id for n in v2.values() for e in n.entities if e["start"] is None or n.text[e["start"] : e["end"]] != e["text"]]
    assert not bad, bad


def test_no_mentions_overlap(v2):
    bad = []
    for n in v2.values():
        s = sorted((e["start"], e["end"]) for e in n.entities)
        if any(a[1] > b[0] for a, b in zip(s, s[1:])):
            bad.append(n.note_id)
    assert not bad, bad


def test_text_order_and_ids_are_unchanged_from_v1(v1, v2):
    assert list(v1) == list(v2)
    assert all(v1[k].text == v2[k].text and v1[k].document_id == v2[k].document_id for k in v1)


def test_v1_file_is_untouched():
    a = json.loads((ROOT / "audit/results/gold_build_summary.json").read_text(encoding="utf-8"))["output_sha256"]
    b = json.loads((ROOT / "audit/results/gold_v2_build_summary.json").read_text(encoding="utf-8"))["gold_v1_sha256"]
    assert a == b == file_sha256(GOLD_V1)


def test_v2_file_matches_its_recorded_hash():
    d = json.loads((ROOT / "audit/results/gold_v2_build_summary.json").read_text(encoding="utf-8"))
    assert d["gold_v2_sha256"] == file_sha256(GOLD_V2)


def test_differences_from_v1_match_the_change_log_exactly(v1, v2, log):
    added = {(r["note_id"], int(r["start"]), int(r["end"]), r["type"]) for r in log if r["action"] == "add_mention"}
    located = {(r["note_id"], int(r["start"]), int(r["end"]), r["type"]) for r in log if r["action"] == "locate_mention"}
    removed = {(r["note_id"], int(r["start"]), int(r["end"]), r["type"]) for r in log if r["action"] == "remove_mention" and r["start"]}
    new, gone = set(), set()
    for k in v1:
        new |= {(k, *s) for s in spans(v2[k]) - spans(v1[k])}
        gone |= {(k, *s) for s in spans(v1[k]) - spans(v2[k])}
    assert new == added | located
    assert gone == removed


def test_net_change_is_accounted_for(v1, v2, log):
    a = collections.Counter(r["action"] for r in log)
    assert a == {"add_mention": 27, "remove_mention": 3, "locate_mention": 1, "correct_mention": 1}
    assert sum(len(n.entities) for n in v2.values()) == sum(len(n.entities) for n in v1.values()) + 27 - 3


def test_rejected_rows_did_not_become_mentions(v2):
    bad = []
    for r in read_decisions():
        if r["action"].startswith("reject") and r["start"]:
            s, e = int(r["start"]), int(r["end"])
            if any(s < x["end"] and e > x["start"] for x in v2[r["note_id"]].entities):
                bad.append((r["proposal"], r["row"]))
    assert not bad, bad


def test_every_flagged_row_has_a_decision():
    d = read_decisions()
    assert sorted(int(r["row"]) for r in d if r["proposal"] == "review") == list(range(1, 90))
    assert sorted(int(r["row"]) for r in d if r["proposal"] == "additional") == list(range(1, 209))
    assert all(r["action"] for r in d)


def test_no_extra_occurrences_remain(v2):
    assert all(not n.extra_occurrences for n in v2.values())


def test_targets_are_rebuilt_from_offsets(v2):
    bad = [n.note_id for n in v2.values() if n.redacted_text != build_target(n.text, list(n.entities), [])]
    assert not bad, bad


def test_target_has_one_placeholder_per_mention(v2):
    bad = [n.note_id for n in v2.values() if len(TAG.findall(n.redacted_text)) != len(n.entities)]
    assert not bad, bad


def test_only_rejected_patterns_remain_in_targets(v2):
    c = collections.Counter()
    for n in v2.values():
        c.update(unannotated_patterns(n.redacted_text))
    assert {k: c[k] for k in EXPECTED_RESIDUAL} == EXPECTED_RESIDUAL


def test_no_annotated_name_or_place_string_is_left_in_its_target(v2):
    from src.data.annotations import word_boundary_regex
    bad = []
    for n in v2.values():
        for e in n.entities:
            if e["type"] in ("NOME", "LUOGO/INDIRIZZO") and len(e["text"]) >= 3 and word_boundary_regex(e["text"]).search(n.redacted_text):
                bad.append(n.note_id)
    assert not bad, bad


def test_added_ages_are_number_only(v2):
    bad = [n.note_id for n in v2.values() for e in n.entities if e["type"] == "ETÀ" and e["status"] == "added_in_review" and not e["text"].isdigit()]
    assert not bad, bad


def test_age_convention_is_unchanged_from_source(v1, v2):
    """Ages kept from the source were not widened to include units."""
    for k in v1:
        kept = {(e["start"], e["end"]) for e in v2[k].entities if e["type"] == "ETÀ" and e["origin"] == "source_annotation"}
        assert kept <= {(e["start"], e["end"]) for e in v1[k].entities if e["type"] == "ETÀ"}


def test_mentions_record_their_origin(v2):
    c = collections.Counter(e["origin"] for n in v2.values() for e in n.entities)
    assert c == {"source_annotation": 225, "added_in_review": 27}


def test_change_log_has_no_text_column():
    head = (ROOT / "data_manifests/gold_v2_change_log.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert not {"text", "value", "value_in_note", "context", "annotated"} & set(head)


@pytest.mark.parametrize("k", range(1, 6))
def test_fold_membership_is_unchanged_and_points_at_v2(k):
    old = json.loads((ROOT / f"splits/gold_v1/fold_{k}.json").read_text(encoding="utf-8"))
    s = load_fold(k)
    assert list(s.test_ids) == old["test_ids"] and list(s.train_ids) == old["train_ids"]
    assert s.gold_sha256 == file_sha256(GOLD_V2) != old["gold_sha256"]


# build safety checks, on invented records
def rec(text, entities=()):
    return {"text": text, "entities": list(entities), "extra_occurrences": [], "document_id": "IT1"}


def decision(action, typ, start="", end="", row="1"):
    return {"action": action, "type": typ, "start": str(start), "end": str(end), "row": row, "note_id": "n", "proposal": "review"}


def test_build_refuses_an_addition_that_overlaps_a_mention():
    r = {"n": rec("Zelio Rossini oggi", [{"type": "NOME", "text": "Zelio", "annotated_text": "Zelio", "start": 0, "end": 5, "status": "located"}])}
    with pytest.raises(GoldBuildError):
        apply_decisions(r, [decision("add_nome", "NOME", 0, 13)])


@pytest.mark.parametrize("action,typ,start,end", [("add_data", "DATA", 0, 4), ("add_age_number_only", "ETÀ", 0, 4), ("add_nome", "NOME", 5, 7)])
def test_build_refuses_a_value_with_the_wrong_format(action, typ, start, end):
    with pytest.raises(GoldBuildError):
        apply_decisions({"n": rec("oggi 12 mg")}, [decision(action, typ, start, end)])


def test_build_refuses_to_remove_a_mention_that_is_not_there():
    with pytest.raises(GoldBuildError):
        apply_decisions({"n": rec("Testo 6 mesi")}, [decision("remove_age_is_a_duration", "ETÀ", 6, 7)])


def test_build_errors_never_contain_note_text():
    try:
        apply_decisions({"n": rec("Zelindo Segreto")}, [decision("add_data", "DATA", 0, 7)])
    except GoldBuildError as e:
        assert "Zelindo" not in str(e)
