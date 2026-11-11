"""Age-aware repeat check (owner decision of 2026-11-06, Option B). Invented sentences only.

When a marked age's digits reappear unmarked, the occurrence is rejected if it reads as an age or is ambiguous, and
allowed only when it is clearly a measurement, dose, angle, percentage, vital sign or score. v1 keeps the blanket rule.
"""
import copy

import pytest

from src.generation.synthetic_validation import age_repeat_occurrences, classify_age_repeat
from tests.test_generation_v2 import BASE, FILLER, V1, V2, reasons, spec_for


def kinds(sentence, value="54"):
    return [classify_age_repeat(sentence, s, e) for s, e in age_repeat_occurrences(value, sentence)]


# Unit level: one occurrence of 54 in each invented sentence

AGE_REFERENCES = [
    "riferisce che a 54 anni aveva già avuto un episodio simile",
    "la madre, 54 anni, è in buona salute",
    "la madre di 54 aa. è in buona salute",
    "la madre di 54 a. è in buona salute",
    "il fratello 54enne è iperteso",
    "il fratello 54-enne è iperteso",
    "il fratello 54enni",
    "all'età di 54 aveva subito un intervento",
    "età: 54",
    "età 54, sesso maschile",
    "di anni 54, residente in zona",
    "un bambino di 54 mesi",
    "un neonato di 54 giorni",
    "un lattante di 54 settimane",
    "aged 54",
    "54 Anni compiuti",
    "ha iniziato la dose a 54 anni",
    "pressione alta da quando aveva 54 anni",
]

QUANTITIES = [
    "emoglobina 12,4 g/dL e PCR 54 mg/L",
    "creatinina 54 µmol/L",
    "dose di 54 mg al giorno",
    "infusione di 54 ml/h",
    "saturazione 54%",
    "saturazione al 54 %",
    "flessione del ginocchio limitata a 54°",
    "pressione arteriosa 120/54 mmHg",
    "pressione 154/54",
    "rapporto 54/60",
    "frequenza cardiaca 54 bpm",
    "FC 54",
    "FC: 54",
    "Hb 54",
    "VAS 54",
    "punteggio di 54",
    "MMSE 54",
    "glicemia pari a 54",
    "valore di 1,54 alla prova",
    "valore di 54,3 alla prova",
    "intervallo 50-54 mg/dL",
    "intervallo 54-60 mg/dL",
    "alle ore 10:54",
    "piastrine 54 x 10^9/L",
    "durata 54 minuti",
    "durata 54 min",
    "per 54 ore",
    "QTc 54",
    "eseguite 54 compresse",
    "ripetuto 54 volte",
    "lesione di 54 mm",
    "54 UI di insulina",
    "temperatura a 54",
    "flessione limitata a 54 gradi",
    "perdita di 54 grammi",
    "diuresi di 54 millilitri",
    "marcia per 54 metri",
    "fuma 54 sigarette",
    "ciclo di 54 sedute",
]

AMBIGUOUS = [
    "ne aveva contati 54 in tutto",
    "sono stati 54 nell'ultimo controllo",
    "letto numero 54 del reparto",
    "riferisce 54 episodi",
    "il valore era 54 l'anno precedente",
    "aveva 54, ma non lo ricordava",
    "ne riferisce 54 e/o più",
    "ca 54 in tutto",
    "figli di 12, 54 e 60 anni",
    "età compresa tra 50-54",
    "pazienti ≥ 54",
    "il 54 del mese",
]


@pytest.mark.parametrize("sentence", AGE_REFERENCES)
def test_italian_age_expressions_are_read_as_ages(sentence):
    assert kinds(sentence) == ["age"]


@pytest.mark.parametrize("sentence", QUANTITIES)
def test_measurements_doses_angles_percentages_and_vitals_are_quantities(sentence):
    assert kinds(sentence) == ["quantity"]


@pytest.mark.parametrize("sentence", AMBIGUOUS)
def test_ambiguous_occurrences_are_not_waved_through(sentence):
    assert kinds(sentence) == ["ambiguous"]


def test_digits_inside_longer_numbers_are_not_occurrences():
    for s in ("anno 1954", "codice 5432", "valore 154 mg", "L54", "54esimo"):
        assert age_repeat_occurrences("54", s) == [], s


def test_every_occurrence_is_classified():
    s = "PCR 54 mg/L; la sorella di 54 anni; ne aveva contati 54"
    assert kinds(s) == ["quantity", "age", "ambiguous"]


# Integration: whole notes through validate_case

def with_tail(tail):
    """BASE (patient age 52, sister 49) plus one more invented sentence."""
    parts = copy.deepcopy(BASE)
    parts[-1] = (parts[-1][0] + tail, "NESSUNA")
    return parts


def r(parts, cfg=V2):
    return reasons(parts, cfg, spec_for(chars=len("".join(t for t, _ in parts))))


def test_v2_enables_the_context_policy_and_v1_does_not():
    assert V2["validation"]["age_repeat_policy"] == "context"
    assert "age_repeat_policy" not in V1["validation"]
    assert V2["validation"]["max_attempts_per_case"] == 5
    assert V1["validation"]["max_attempts_per_case"] == 3


@pytest.mark.parametrize("tail", ["Il padre è morto a 52 anni.", "Il cugino 52enne sta bene.", "Riferisce che all'età di 52 era già iperteso.",
                                  "Anche il fratello di 49 anni è iperteso."])
def test_a_repeated_unmarked_age_is_rejected(tail):
    assert "age_repeated_unmarked" in r(with_tail(tail))


@pytest.mark.parametrize("tail", ["Glicemia 52 mg/dL a digiuno.", "Saturazione 49% in aria ambiente.", "Abduzione della spalla fino a 52°.",
                                  "Pressione 130/52 mmHg.", "FC 52 bpm, ritmo sinusale.", "Dose di 52 mg al giorno.", "Creatinina 0,52 mg/dL."])
def test_a_measurement_equal_to_an_age_is_accepted_and_counted(tail):
    parts = with_tail(tail)
    assert r(parts) == []
    from src.generation.synthetic_validation import validate_case
    from tests.test_generation_v2 import case_from, context
    res = validate_case(case_from(parts), spec_for(chars=len("".join(t for t, _ in parts))), context(V2))
    assert res.flags["age_value_as_quantity"] >= 1


@pytest.mark.parametrize("tail", ["Ne aveva contati 52 in tutto.", "Letto 52 del reparto."])
def test_an_ambiguous_repeat_is_rejected(tail):
    assert "age_value_repeated_ambiguous" in r(with_tail(tail))


@pytest.mark.parametrize("tail", ["Glicemia 52 mg/dL a digiuno.", "Il padre è morto a 52 anni.", "Ne aveva contati 52 in tutto."])
def test_v1_keeps_the_blanket_rule(tail):
    got = r(with_tail(tail), V1)
    assert "entity_left_in_redacted" in got
    assert not {"age_repeated_unmarked", "age_value_repeated_ambiguous"} & set(got)


def test_non_age_entities_keep_the_blanket_rule_under_v2():
    parts = [("Paziente ", "NESSUNA"), ("Ugo", "NOME"), (" di ", "NESSUNA"), ("52", "ETÀ"),
             (" anni, la sorella di ", "NESSUNA"), ("49", "ETÀ"), (" anni sta bene; Ugo riferisce dolore. " + FILLER * 2, "NESSUNA")]
    spec = spec_for(n_nome=1, kind="synthetic_named", chars=len("".join(t for t, _ in parts)))
    assert "entity_left_in_redacted" in reasons(parts, V2, spec)


def test_a_clean_note_has_no_quantity_flag():
    from src.generation.synthetic_validation import validate_case
    from tests.test_generation_v2 import case_from, context
    res = validate_case(case_from(BASE), spec_for(chars=len("".join(t for t, _ in BASE))), context(V2))
    assert res.accepted and res.flags["age_value_as_quantity"] == 0


def test_an_unknown_policy_fails_loudly():
    cfg = copy.deepcopy(V2)
    cfg["validation"]["age_repeat_policy"] = "lenient"
    with pytest.raises(ValueError):
        r(BASE, cfg)
