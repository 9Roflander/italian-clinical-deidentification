"""Guide test 9: generated entities, redacted text and annotations must agree."""
import copy

import pytest

from src.data.loaders import GoldNote
from src.generation.generation_plan import CaseSpec
from src.generation.synthetic_validation import ValidationContext, build_context, build_note, validate_case

CFG = {"validation": {"min_chars": 500, "max_chars": 4200, "length_tolerance": [0.6, 1.5], "min_entities": 2, "count_tolerance_share": 0.34,
                      "count_tolerance_min": 1, "max_verbatim_run_anchor": 10, "max_verbatim_run_train": 12, "near_duplicate_containment": 0.2,
                      "max_entity_chars": 80}}
PAD = "la paziente viene sottoposta a controllo cardiologico periodico con esito favorevole e buona tolleranza al trattamento "


def spec(kind="synthetic_named", counts=(1, 1, 0, 1), chars=900):
    return CaseSpec("F1-N00001", 1, kind, 1, 0, "cardiologia", "donna", "41-65", "Toscana", "nome proprio", "M", *counts, chars, "media")


def seg(text, kind="NESSUNA"):
    return {"text": text, "entity_type": kind}


def good_case():
    return {"case_id": "F1-N00001", "segments": [
        seg("Accede al reparto "), seg("Mirella", "NOME"), seg(", donna di "), seg("54", "ETÀ"), seg(" anni, ricoverata a "),
        seg("Pontedera", "LUOGO/INDIRIZZO"), seg(". " + PAD * 8)]}


def ctx(train=(), anchors=(), seen=None):
    c = build_context(list(train), CFG)
    return ValidationContext(c["train_hashes"], c["train_token_docs"], c["train_index"], list(anchors), seen or set(), CFG)


def reasons(case, s=None, **kw):
    return validate_case(case, s or spec(), ctx(**kw)).reasons


def test_good_case_is_accepted_with_correct_offsets():
    r = validate_case(good_case(), spec(), ctx())
    assert r.accepted, r.reasons
    n = r.note
    assert all(n["text"][e["start"] : e["end"]] == e["text"] for e in n["entities"])
    assert "[NOME]" in n["redacted_text"] and "Mirella" not in n["redacted_text"] and "54" not in n["redacted_text"]
    assert [e["type"] for e in n["entities"]] == ["NOME", "ETÀ", "LUOGO/INDIRIZZO"]


def test_whitespace_moves_outside_the_entity():
    n, why = build_note({"segments": [seg("Vista da"), seg(" Mirella ", "NOME"), seg("oggi")]})
    assert not why and n["entities"][0]["text"] == "Mirella" and n["text"] == "Vista da Mirella oggi"
    assert n["redacted_text"] == "Vista da [NOME] oggi"


def test_repeated_mentions_get_separate_offsets():
    n, _ = build_note({"segments": [seg("A", "NOME"), seg(" e "), seg("A", "NOME")]})
    assert [(e["start"], e["end"]) for e in n["entities"]] == [(0, 1), (4, 5)]


@pytest.mark.parametrize("case", [{}, {"segments": []}, {"segments": "x"}, {"segments": [{"text": 3, "entity_type": "NOME"}]}, {"segments": [{"text": "a", "entity_type": "ALTRO"}]}])
def test_malformed_cases_are_rejected(case):
    assert validate_case(case, spec(), ctx()).accepted is False


def edited(fn):
    c = copy.deepcopy(good_case())
    fn(c)
    return c


@pytest.mark.parametrize("code,fn", [
    ("entity_left_in_redacted", lambda c: c["segments"].append(seg(" Di nuovo Mirella riferisce dolore."))),
    ("place_part_left_in_redacted", lambda c: c["segments"].append(seg(" Trasferita a Pontedera."))),
    ("residual_date", lambda c: c["segments"].append(seg(" Visita del 12/03/2015."))),
    ("residual_date", lambda c: c["segments"].append(seg(" Rivista a marzo."))),
    ("residual_date", lambda c: c["segments"].append(seg(" Nel 2015 peggiora."))),
    ("residual_age", lambda c: c["segments"].append(seg(" Il marito, uomo di 60 anni, assiste."))),
    ("residual_title_name", lambda c: c["segments"].append(seg(" Seguita dal Dott. Verdi."))),
    ("residual_initials", lambda c: c["segments"].append(seg(" Paziente nota come Q.Z. in reparto."))),
    ("residual_contact", lambda c: c["segments"].append(seg(" Scrivere a nome@esempio.it per informazioni."))),
    ("eta_not_numeric", lambda c: c["segments"].__setitem__(3, seg("54 anni", "ETÀ"))),
    ("name_malformed", lambda c: c["segments"].__setitem__(1, seg("Mirella3", "NOME"))),
    ("date_malformed", lambda c: c["segments"].append(seg("ieri", "DATA"))),
    ("entity_too_long", lambda c: c["segments"].__setitem__(5, seg("Pontedera " * 10, "LUOGO/INDIRIZZO"))),
    ("placeholder_in_text", lambda c: c["segments"].append(seg(" Valore [NOME] nel testo."))),
    ("contradictory_types", lambda c: c["segments"].extend([seg(" poi "), seg("Pontedera", "NOME")])),
    ("adjacent_entities", lambda c: c["segments"].extend([seg("Mirella", "NOME"), seg("Fabbri", "NOME")])),
    ("empty_entity", lambda c: c["segments"].append(seg("  ", "NOME"))),
    ("case_id_mismatch", lambda c: c.update(case_id="F9-N99999")),
])
def test_each_rule_fires(code, fn):
    assert code in reasons(edited(fn))


def test_surname_left_after_full_name_is_caught():
    c = {"case_id": "F1-N00001", "segments": [seg("Accede "), seg("Mirella Fabbri", "NOME"), seg(", di "), seg("54", "ETÀ"), seg(" anni, a "),
                                             seg("Pontedera", "LUOGO/INDIRIZZO"), seg(". " + PAD * 8 + " La signora Fabbri migliora.")]}
    assert "name_part_left_in_redacted" in reasons(c)


def test_digits_inside_a_longer_number_do_not_count_as_a_left_entity():
    c = edited(lambda c: c["segments"].append(seg(" Emoglobina 154 g/L e 2547 UI.")))
    assert "entity_left_in_redacted" not in reasons(c)


def test_generic_clinical_abbreviations_are_not_initials():
    c = edited(lambda c: c["segments"].append(seg(" Seguita dalla U.O. di cardiologia.")))
    assert "residual_initials" not in reasons(c)


def test_durations_are_not_ages():
    c = edited(lambda c: c["segments"].append(seg(" Terapia di 3 mesi con follow-up di 5 anni.")))
    assert "residual_age" not in reasons(c)


def test_dose_like_numbers_are_not_years():
    c = edited(lambda c: c["segments"].append(seg(" Somministrati 2000 mg di farmaco.")))
    assert "residual_date" not in reasons(c)


def test_length_rules():
    short = {"case_id": "F1-N00001", "segments": [seg("Accede "), seg("Mirella", "NOME"), seg(", di "), seg("54", "ETÀ"), seg(" anni, a "), seg("Pontedera", "LUOGO/INDIRIZZO"), seg(".")]}
    assert "length_out_of_range" in reasons(short)
    assert "length_off_target" in reasons(good_case(), spec(chars=4000))


def test_entity_count_rules():
    assert "entity_count_off_plan" in reasons(good_case(), spec(counts=(6, 1, 0, 1)))
    assert "too_few_entities" in reasons({"case_id": "F1-N00001", "segments": [seg("Accede "), seg("Mirella", "NOME"), seg(". " + PAD * 8)]})
    assert "name_in_name_free" in reasons(good_case(), spec("synthetic_name_free", (0, 1, 0, 1)))
    assert "name_missing_in_named" in reasons({"case_id": "F1-N00001", "segments": [seg("Donna di "), seg("54", "ETÀ"), seg(" anni, a "), seg("Pontedera", "LUOGO/INDIRIZZO"), seg(". " + PAD * 8)]}, spec(counts=(1, 1, 0, 1)))


def test_negative_rules():
    clean = {"case_id": "F1-N00001", "segments": [seg(PAD * 8)]}
    assert validate_case(clean, spec("synthetic_negative", (0, 0, 0, 0)), ctx()).accepted
    assert "entity_in_negative" in reasons(good_case(), spec("synthetic_negative", (0, 0, 0, 0)))


def test_exact_duplicate_against_training_and_run():
    built = build_note(good_case())[0]["text"]
    train = GoldNote("gold_001", 1, built, ())
    assert "exact_duplicate" in reasons(good_case(), train=[train])
    from src.data.normalization import doc_hash
    assert "exact_duplicate" in reasons(good_case(), seen={doc_hash(built)})


def test_near_duplicate_of_training_note():
    built = build_note(good_case())[0]["text"]
    train = GoldNote("gold_001", 1, built.replace("Mirella", "Altro"), ())
    r = reasons(good_case(), train=[train])
    assert "near_duplicate_of_training_note" in r and "long_verbatim_run_with_training_note" in r


def test_anchor_copying_and_anchor_pii():
    anchor = GoldNote("gold_002", 2, PAD, ())
    assert "anchor_text_copied" in reasons(good_case(), anchors=[anchor])
    pii = GoldNote("gold_003", 3, "altro testo", ({"type": "LUOGO/INDIRIZZO", "text": "Pontedera"},))
    assert "anchor_pii_copied" in reasons(good_case(), anchors=[pii])
