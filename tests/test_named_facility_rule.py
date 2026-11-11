"""Unmarked named-facility rule, proposed 2026-11-07 after the final v2 pilot. Invented sentences only.

The generation prompt defines named hospitals and healthcare facilities as LUOGO/INDIRIZZO. A facility word followed by a
proper name outside every entity is therefore an unmarked identifier and the case is rejected. Generic descriptors stay allowed.
"""
import copy

import pytest

from src.generation.synthetic_validation import unmarked_named_facilities
from tests.test_generation_v2 import BASE, V1, V2, reasons, spec_for

NAMED = [
    "ricoverato presso l'Ospedale San Zefirello per accertamenti",
    "trasferita all'Arcispedale Santa Brunilde",
    "seguito presso il Policlinico Orvaldi",
    "visitato all'Istituto Pernasca",
    "presso la Clinica Villa Quarzana",
    "inviato al Centro Tullio Varnesi",
    "ricovero alla Casa di Cura San Fulberto",
    "valutato all'IRCCS Garzoli",
    "presso l'Ospedale Sant'Eulogia",
    "presso l'Ospedale S. Ermete",
    "presso la Fondazione Brusotti",
    "presso l'Azienda Ospedaliera Valtorbe",
    "presso il Presidio Ospedaliero Santo Ilarino",
    "presso l'Ospedale dei Viandoli",
    "presso l'ospedale Santo Ilarino di [LUOGO/INDIRIZZO]",
    "ricoverata al policlinico Orvaldi",
    "seguito dal centro Tullio Varnesi",
]
GENERIC = [
    "presso l'Ospedale Civile di [LUOGO/INDIRIZZO]",
    "presso l'ospedale di [LUOGO/INDIRIZZO]",
    "presso l'Ospedale [LUOGO/INDIRIZZO]",
    "presso il Policlinico Universitario di [LUOGO/INDIRIZZO]",
    "presso il Centro Trasfusionale",
    "presso il Centro di Riferimento regionale",
    "presso la Clinica Medica",
    "presso il Presidio Ospedaliero di [LUOGO/INDIRIZZO]",
    "seguito dal centro diabetologico",
    "presso l'Istituto Oncologico di [LUOGO/INDIRIZZO]",
    "presso il centro ospedaliero di zona",
    "presso l'Ospedale Maggiore di [LUOGO/INDIRIZZO]",
    "rientrata in ospedale per controllo",
    "nel centro storico",
    "presso il presidio ospedaliero di [LUOGO/INDIRIZZO]",
]


@pytest.mark.parametrize("s", NAMED)
def test_named_facilities_are_found(s):
    assert unmarked_named_facilities(s) == 1


@pytest.mark.parametrize("s", GENERIC)
def test_generic_facility_descriptions_are_not(s):
    assert unmarked_named_facilities(s) == 0


def with_tail(tail):
    parts = copy.deepcopy(BASE)
    parts[-1] = (parts[-1][0] + tail, "NESSUNA")
    return parts


def r(parts, cfg=V2):
    return reasons(parts, cfg, spec_for(chars=len("".join(t for t, _ in parts))))


def test_v2_enables_the_rule_and_v1_does_not():
    assert V2["validation"]["reject_unmarked_named_facility"] is True
    assert "reject_unmarked_named_facility" not in V1["validation"]


def test_an_unmarked_named_hospital_rejects_the_case():
    parts = with_tail("Trasferita all'Ospedale San Zefirello.")
    assert "named_facility_unmarked" in r(parts)
    assert "named_facility_unmarked" not in r(parts, V1), "v1 behaviour is unchanged"


def test_a_marked_named_hospital_is_accepted():
    parts = with_tail("Trasferita all'") + [("Ospedale San Zefirello", "LUOGO/INDIRIZZO"), (".", "NESSUNA")]
    assert "named_facility_unmarked" not in reasons(parts, V2, spec_for(chars=len("".join(t for t, _ in parts))))


def test_a_generic_hospital_with_a_marked_city_is_accepted():
    parts = with_tail("Trasferita all'Ospedale Civile di ") + [("Borgoquarzo", "LUOGO/INDIRIZZO"), (".", "NESSUNA")]
    assert "named_facility_unmarked" not in reasons(parts, V2, spec_for(chars=len("".join(t for t, _ in parts))))
