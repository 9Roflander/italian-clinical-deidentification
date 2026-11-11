"""Medical eponyms, used to make the capitalized-word flag less noisy without hiding possible names.

A capitalized word counts as an eponym only when it is on this list AND follows a clinical head word, as in a disease,
sign, test or scale named after a person. The same word standing alone, or after a title, still counts. The flag is
only a warning for reviewers, so this never changes which notes are accepted or rejected.
"""
from __future__ import annotations

import re

EPONYMS = frozenset("""
Addison Alzheimer Apgar Babinski Barrett Barré Basedow Bazett Bell Billroth Blumberg Brugada Charcot Child Conn Crohn
Cushing Danlos Down Duchenne Duke Ehlers Eisenmenger Fallot Forrest Fowler Gilbert Giordano Glasgow Gleason Graves Guillain
Hartmann Hashimoto Hawkins Henoch Hodgkin Holter Horner Huntington Jobe Kaposi Kawasaki Kennedy Kussmaul Lachman Lasègue Lewy
Mallory Marfan Meckel Meniere Ménière Mobitz Murphy Neer Nissen Parkinson Phalen Pugh Ranson Raynaud Reynolds Romberg Rovsing
Roux Schönlein Seldinger Sjögren Steinmann Tesla Thompson Tinel Trendelenburg Turner Valsalva Virchow Weiss Wells Wenckebach
Whipple Willebrand Wilms Wilson Wolff Zollinger Doppler Hoffmann Lyme Ebstein Brown Sequard Stevens Johnson Tietze Takayasu
""".split())
HEAD_WORDS = ("malattia", "malattie", "morbo", "sindrome", "segno", "segni", "test", "manovra", "manovre", "scala", "classificazione", "criteri",
              "formula", "fenomeno", "porpora", "tiroidite", "corpi", "linfoma", "sarcoma", "tumore", "reazione", "riflesso", "prova", "prove",
              "legamento", "frattura", "ernia", "cisti", "ulcera", "monitoraggio", "metodo", "intervento", "procedura", "tecnica", "punteggio",
              "esofago", "diverticolo", "cellule", "aneurisma", "blocco", "onde", "onda", "tetralogia", "anomalia", "paralisi", "secondo",
              "elettrocardiogramma", "ecografia", "color", "magnetica", "tracciato", "anello", "nodo", "triade", "fase", "stadio", "grado", "tipo")
STANDALONE = frozenset({"Holter", "Doppler", "Tesla"})
CAPITALISED = re.compile(r"(?<=[a-zà-ù,] )[A-ZÀ-Ý][a-zà-ù]{2,}")
_CONTEXT = re.compile(r"\b(?:" + "|".join(HEAD_WORDS) + r")\b(?:\s+[a-zà-ù'’]+){0,3}?(?:\s+(?:di|del|della|dello|dei|degli|secondo|sec\.))?"
                      r"(?:\s+[A-ZÀ-Ý][\w'’-]*(?:,|\s+e|\s+ed)?){0,3}\s*$", re.I)


def is_eponym_in_context(text: str, start: int, word: str) -> bool:
    if word not in EPONYMS:
        return False
    if word in STANDALONE:
        return True
    return bool(_CONTEXT.search(text[max(0, start - 80) : start]))


def capitalised_word_counts(text: str) -> tuple[int, int]:
    """(all mid-sentence capitalized words, those not explained as an eponym in context)."""
    matches = list(CAPITALISED.finditer(text))
    return len(matches), sum(1 for m in matches if not is_eponym_in_context(text, m.start(), m.group(0)))
