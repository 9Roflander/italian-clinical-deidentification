from dataclasses import fields

from src.data.deduplication import PairRow, classify, compare_collections, longest_common_run

LONG = " ".join(f"parola{i}" for i in range(120))


def test_longest_run():
    assert longest_common_run(list("abcd"), list("xbcz")) == 2
    assert longest_common_run([], list("abc")) == 0


def test_exact_after_normalisation_is_found():
    rows = compare_collections("a", {"x": LONG}, "b", {"y": LONG.upper() + "  "})
    assert [r.relation for r in rows] == ["exact_duplicate"]


def test_near_duplicate_is_found():
    edited = LONG.replace("parola60", "diversa")
    rows = compare_collections("a", {"x": LONG}, "b", {"y": edited})
    assert rows[0].relation == "near_duplicate" and rows[0].token_seq_ratio > 0.9


def test_short_shared_phrase_is_not_a_duplicate():
    shared = "alfa beta gamma delta epsilon zeta eta theta iota"
    rows = compare_collections("a", {"x": "uno due " + shared + " tre"}, "b", {"y": "sei sette " + shared + " otto"})
    assert rows[0].relation == "shared_phrase_only"


def test_no_overlap_gives_no_rows():
    assert compare_collections("a", {"x": "uno due tre"}, "b", {"y": "quattro cinque sei"}) == []


def test_same_collection_skips_self_pairs():
    assert compare_collections("a", {"x": LONG, "y": "altro testo"}, "a", {"x": LONG, "y": "altro testo"}, same_collection=True) == []


def test_classify_thresholds():
    assert classify(False, 0.5, 0, 0, 0) == "near_duplicate"
    assert classify(False, 0, 0.9, 0, 0) == "near_duplicate"
    assert classify(False, 0, 0, 12, 0) == "partial_overlap_review"
    assert classify(False, 0, 0, 0, 3) == "partial_overlap_review"
    assert classify(False, 0, 0, 8, 1) == "shared_phrase_only"


def test_pair_rows_cannot_carry_text():
    assert {f.name for f in fields(PairRow)}.isdisjoint({"text", "text_a", "text_b", "snippet"})
