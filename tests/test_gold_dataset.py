"""The gold set: source parsing, ordering, offsets and status handling. Real-data checks report IDs and counts only."""
import json

import pytest

from src.data.build_gold import build_target, recover_spans, relation_offsets_ok, split_input, unannotated_patterns
from src.data.loaders import GOLD_V1, GOLD_V2, ROOT, load_gold


def test_split_separates_id_narrative_and_relation_tail():
    cell = "100002|t|Testo clinico qui. 100002\tREL\t0-5\t6-13\tTesto\tclinico 100002\tREL\t0-5\t6-13\tTesto\tclinico"
    doc, narrative, tail = split_input(cell)
    assert doc == "100002" and narrative == "Testo clinico qui." and tail.count("REL") == 2
    assert relation_offsets_ok(narrative, doc, tail) == (2, 2, True)


def test_relation_lines_separated_by_spaces_are_recognised():
    """Three source rows use spaces instead of tabs; these lines must not be kept as narrative."""
    cell = "100010|t|Testo clinico qui. 100010 REL 0-5 6-13 Testo clinico 100010 REL 0-5 6-13 Testo clinico"
    doc, narrative, tail = split_input(cell)
    assert narrative == "Testo clinico qui."
    assert relation_offsets_ok(narrative, doc, tail) == (2, 2, True)


def test_split_handles_a_note_without_a_tail():
    assert split_input("100003|t|Solo testo.") == ("100003", "Solo testo.", "")


def test_a_cell_without_an_id_is_refused():
    with pytest.raises(ValueError):
        split_input("testo senza identificativo")


def test_bad_relation_offsets_are_detected():
    ok, total, explained = relation_offsets_ok("Testo clinico", "1", " 1\tREL\t0-3\t6-9\tTesto\tclinico")
    assert ok == 0 and total == 1 and explained


def test_text_outside_relation_lines_is_detected():
    assert relation_offsets_ok("Testo", "1", " parole estranee 1 REL 0-5 0-5 Testo Testo")[2] is False


def spans(text, *pairs):
    return recover_spans(text, [{"type": t, "text": s} for t, s in pairs])


def test_each_repeated_mention_gets_its_own_offset():
    e, x = spans("Zelio vede Zelio.", ("NOME", "Zelio"), ("NOME", "Zelio"))
    assert [(a["start"], a["end"], a["status"]) for a in e] == [(0, 5, "located"), (11, 16, "located")] and not x


def test_age_prefers_the_occurrence_in_age_context():
    e, x = spans("Dose 8 mg, bambino di 8 anni.", ("ETÀ", "8"))
    assert e[0]["start"] == 22 and e[0]["status"] == "located_ambiguous"
    assert [a["start"] for a in x] == [5]


def test_digits_inside_longer_numbers_are_not_mentions():
    e, x = spans("Valore 18 e 80, poi 8 anni.", ("ETÀ", "8"))
    assert e[0]["start"] == 20 and e[0]["status"] == "located" and not x


def test_age_attached_to_a_word_is_located_and_flagged():
    e, _ = spans("Ragazzo 15enne arriva.", ("ETÀ", "15"))
    assert e[0]["status"] == "located_inside_longer_token" and e[0]["text"] == "15"


def test_whitespace_variant_keeps_both_strings():
    e, _ = spans("Vive in Via  Roma qui.", ("LUOGO/INDIRIZZO", "Via Roma"))
    assert e[0]["status"] == "located_whitespace_variant" and e[0]["text"] == "Via  Roma" and e[0]["annotated_text"] == "Via Roma"


def test_fuzzy_variant_is_flagged_not_silently_accepted():
    e, _ = spans("Accesso nel dicembre del 2015 in reparto.", ("DATA", "dicembre dell 2015"))
    assert e[0]["status"] == "located_fuzzy_variant" and e[0]["text"].startswith("dicembre")


def test_unlocatable_annotation_stays_in_the_count_without_offsets():
    e, _ = spans("Niente di simile qui.", ("NOME", "Xqwzzz"))
    assert len(e) == 1 and e[0]["start"] is None and e[0]["status"] == "annotated_more_often_than_occurs"


def test_longer_mention_claims_its_text_first():
    e, _ = spans("Presso Ospedale Civile di Lucca.", ("LUOGO/INDIRIZZO", "Lucca"), ("LUOGO/INDIRIZZO", "Ospedale Civile di Lucca"))
    assert {a["status"] for a in e} == {"located", "nested_in_longer_mention"}


def test_target_redacts_extra_names_but_not_extra_ages():
    text = "Zelio 8 mg, Zelio ha 8 anni."
    e, x = spans(text, ("NOME", "Zelio"), ("ETÀ", "8"))
    target = build_target(text, e, x)
    assert target == "[NOME] 8 mg, [NOME] ha [ETÀ] anni."


def test_target_never_splits_numbers():
    text = "Valore 18, ha 8 anni."
    e, x = spans(text, ("ETÀ", "8"))
    assert build_target(text, e, x) == "Valore 18, ha [ETÀ] anni."


def test_unannotated_patterns_are_counted():
    c = unannotated_patterns("Nel marzo 2015 e il 12/03/2016, visto da Q.Z. a Dott. Neri.")
    assert c["date"] >= 2 and c["initials"] == 1 and c["title_name"] == 1


# real data
@pytest.fixture(scope="module")
def gold():
    return load_gold()


def test_real_gold_has_82_notes_in_document_order(gold):
    ids = [n.document_id for n in gold]
    assert len(gold) == 82 and len(set(ids)) == 82
    assert all(int(a[2:]) < int(b[2:]) for a, b in zip(ids, ids[1:]))
    assert [n.note_id for n in gold] == [f"gold_{i:03d}" for i in range(82)]


def test_gold_v1_mention_counts_match_the_source():
    c = {}
    for n in load_gold(GOLD_V1):
        for e in n.entities:
            c[e["type"]] = c.get(e["type"], 0) + 1
    assert c == {"NOME": 46, "ETÀ": 101, "DATA": 47, "LUOGO/INDIRIZZO": 34}


def test_real_gold_offsets_are_exact(gold):
    bad = [n.note_id for n in gold for e in n.entities if e["start"] is not None and n.text[e["start"] : e["end"]] != e["text"]]
    assert not bad, bad


def test_real_gold_located_spans_do_not_overlap_except_marked_nesting(gold):
    bad = []
    for n in gold:
        s = sorted((e["start"], e["end"]) for e in n.entities if e["start"] is not None and e["status"] != "nested_in_longer_mention")
        if any(a[1] > b[0] for a, b in zip(s, s[1:])):
            bad.append(n.note_id)
    assert not bad, bad


def test_real_gold_text_is_clean(gold):
    import re
    import unicodedata
    bad = [n.note_id for n in gold if "\t" in n.text or "|t|" in n.text or n.text != n.text.strip() or unicodedata.normalize("NFC", n.text) != n.text
           or re.search(r"\bREL\b\s+\d+-\d+", n.text)]
    assert not bad, bad


def test_real_gold_matches_the_old_narratives_exactly():
    """After NFC, the first 80 notes must equal the old file. This caught a parsing error once."""
    import json as _j
    import unicodedata
    old = _j.loads((ROOT / "original_work" / "data" / "gold_standard_80.json").read_text(encoding="utf-8"))
    new = load_gold()
    bad = [new[i].note_id for i in range(80) if new[i].text != unicodedata.normalize("NFC", old[i]["text"])]
    assert not bad, bad


def test_real_relation_offsets_all_verify():
    d = json.loads((ROOT / "audit" / "results" / "gold_build_summary.json").read_text(encoding="utf-8"))
    assert d["relation_lines"] > 0 and d["relation_offsets_ok"] == d["relation_lines"]
    assert d["tails_not_fully_explained"] == []


def test_real_targets_remove_every_located_mention(gold):
    bad = []
    for n in gold:
        if n.redacted_text.count("[") < sum(e["start"] is not None and e["status"] != "nested_in_longer_mention" for e in n.entities):
            bad.append(n.note_id)
    assert not bad, bad


def test_quality_report_contains_no_text():
    head = (ROOT / "data_manifests" / "gold_quality_report.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "text" not in head.split(",") and "context" not in head


def test_build_is_deterministic():
    a = json.loads((ROOT / "audit" / "results" / "gold_build_summary.json").read_text(encoding="utf-8"))["output_sha256"]
    from src.data.loaders import file_sha256
    assert a == file_sha256(GOLD_V1)
