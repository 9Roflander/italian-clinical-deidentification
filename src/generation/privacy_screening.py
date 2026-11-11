"""PII masking before text leaves the machine, and residual-PII screening of generated notes."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.data.annotations import PLACEHOLDERS, canonical_type, mask_annotated, word_boundary_regex

MONTHS = "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
_UNITS = r"(?!\s*(?:mg|ml|mL|g|kg|cc|mcg|µg|UI|U|ng|pg|mm|cm|mEq|mmol|/))"
YEAR = re.compile(r"(?<![\d.,/-])(?:19|20)\d{2}(?![\d.,]*\d)" + _UNITS)
NUMERIC_DATE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b")
MONTH = re.compile(rf"\b(?:{MONTHS})\b", re.I)
# Durations such as "terapia di 3 mesi" or "follow-up di 5 anni" are not ages, so the check needs a person or age word.
AGE_CONTEXT = re.compile(r"\b(?:età di|ha|aveva|compiuti)\s+\d{1,3}\s+anni\b"
                         r"|\b(?:uomo|donna|paziente|ragazz[oa]|bambin[oa]|signor[ae]?|anzian[oa]|soggetto|giovane|maschio|femmina)\s+di\s+\d{1,3}\s+anni\b", re.I)
AGE_ANY = re.compile(r"\b\d{1,3}\s*(?:anni|aa\.?)(?=\W|$)|\b(?:aa\.?|a\.)\s*\d{1,3}\b|\betà di \d{1,3}\b", re.I)
TITLE_NAME = re.compile(r"\b(?:Sig\.?(?:ra)?|Signor[ae]?|Dott\.?(?:ssa)?|Dr\.?(?:ssa)?|Prof\.?(?:ssa)?)\s+[A-ZÀ-Ý]")
# Masking must take the whole name, or the rest of the surname stays in the text.
TITLE_FULL_NAME = re.compile(r"\b(?:Sig\.?(?:ra)?|Signor[ae]?|Dott\.?(?:ssa)?|Dr\.?(?:ssa)?|Prof\.?(?:ssa)?)\s+[A-ZÀ-Ý][\w'’-]*(?:\s+[A-ZÀ-Ý][\w'’-]*)*")
INITIALS = re.compile(r"\b[A-Z]\.\s?[A-Z]\.")
# Common Italian clinical abbreviations that look like initials.
INITIALS_ALLOWED = {"U.O.", "P.S.", "A.O.", "T.C.", "R.M.", "A.S.", "S.S.", "C.T.", "E.V.", "I.V.", "I.M.", "S.C.", "F.C.", "D.H."}
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(?<!\d)(?:\+39\s?)?\d{2,4}[\s./-]\d{5,8}(?!\d)")


def residual_pii_reasons(redacted: str) -> list[str]:
    """Rejection codes for PII-like patterns left in a generated note after redaction."""
    reasons = []
    if NUMERIC_DATE.search(redacted) or MONTH.search(redacted) or YEAR.search(redacted):
        reasons.append("residual_date")
    if AGE_CONTEXT.search(redacted):
        reasons.append("residual_age")
    if TITLE_NAME.search(redacted):
        reasons.append("residual_title_name")
    if any(m.group(0).replace(" ", "") not in INITIALS_ALLOWED for m in INITIALS.finditer(redacted)):
        reasons.append("residual_initials")
    if EMAIL.search(redacted) or PHONE.search(redacted):
        reasons.append("residual_contact")
    return reasons


def mask_residual(text: str) -> tuple[str, int]:
    """Broad masking for text about to leave the machine. Over-masking is acceptable here."""
    n = 0
    keep_abbrev = lambda m: m.group(0) if m.group(0).replace(" ", "") in INITIALS_ALLOWED else PLACEHOLDERS["NOME"]
    for pattern, tag in [(EMAIL, "[RIMOSSO]"), (PHONE, "[RIMOSSO]"), (NUMERIC_DATE, PLACEHOLDERS["DATA"]),
                         (re.compile(rf"\b(?:\d{{1,2}}\s+)?(?:{MONTHS})(?:\s+(?:del\s+)?\d{{2,4}})?\b", re.I), PLACEHOLDERS["DATA"]),
                         (YEAR, PLACEHOLDERS["DATA"]), (TITLE_FULL_NAME, PLACEHOLDERS["NOME"]), (INITIALS, keep_abbrev),
                         (AGE_ANY, PLACEHOLDERS["ETÀ"])]:
        text, k = pattern.subn(tag, text)
        n += k
    return text, n


@dataclass(frozen=True)
class MaskedAnchor:
    note_id: str
    text: str
    annotated_masked: int
    patterns_masked: int


def mask_anchor(note) -> MaskedAnchor:
    masked, a = mask_annotated(note.text, note.entities)
    masked, b = mask_residual(masked)
    return MaskedAnchor(note.note_id, masked, a, b)


def mask_anchor_set(notes) -> list[MaskedAnchor]:
    """Mask each anchor, then remove every anchor's names and places from all anchors in the request.

    The second pass also catches case variants and surname-only mentions the annotation missed.
    """
    pairs = set()
    for n in notes:
        for e in n.entities:
            t, x = canonical_type(e.get("type")), e.get("text", "")
            if t == "NOME" and len(x) >= 3:
                pairs.add((t, x))
                pairs.update((t, part) for part in re.findall(r"[^\W\d_]{3,}", x))
            elif t == "LUOGO/INDIRIZZO" and len(x) >= 3:
                pairs.add((t, x))
    ordered = sorted(pairs, key=lambda p: -len(p[1]))
    out = []
    for n in notes:
        m = mask_anchor(n)
        text, extra = m.text, 0
        for t, x in ordered:
            text, k = word_boundary_regex(x, re.I).subn(PLACEHOLDERS[t], text)
            extra += k
        out.append(MaskedAnchor(n.note_id, text, m.annotated_masked, m.patterns_masked + extra))
    return out


def identifying_strings(notes) -> set[str]:
    """Annotated names and places (3+ characters), the strings that must never leave in masked mode."""
    out = set()
    for n in notes:
        for e in n.entities:
            t, x = canonical_type(e.get("type")), e.get("text", "")
            if t in ("NOME", "LUOGO/INDIRIZZO") and len(x) >= 3:
                out.add(x)
    return out


def count_identifying_strings(text: str, notes) -> int:
    """Run on the anchor text only: region names in a case spec can match a place by chance."""
    return sum(1 for s in identifying_strings(notes) if word_boundary_regex(s, re.I).search(text))


def scrub_secret(message: str, secret: str | None) -> str:
    """Remove the API key from any text that may reach a log."""
    return message.replace(secret, "[REDACTED_KEY]") if secret else message
