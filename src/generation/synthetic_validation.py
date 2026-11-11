"""Validation of one generated case.

The output format is a segment list, so text, redacted text and offsets all come from one
source and cannot disagree. Checks compare against training notes only: filtering against
test notes would itself leak test information. Test-note checks live in post_audit.py.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field

from src.data.annotations import ENTITY_TYPES, PLACEHOLDERS, canonical_type, word_boundary_regex
from src.data.deduplication import longest_common_run
from src.data.normalization import doc_hash, tokens, word_ngrams
from src.generation.medical_eponyms import capitalised_word_counts
from src.generation.privacy_screening import MONTH, residual_pii_reasons

PLACEHOLDER_RE = re.compile(r"\[(?:NOME|ETÀ|DATA|LUOGO/INDIRIZZO)\]")
BRACKET_RE = re.compile(r"\[[A-ZÀ-Ý][^\]]*\]")
LINE_BREAK_RE = re.compile(r"[\r\n\u2028\u2029\x0b\x0c\x85]")
# An age that follows one of these is a milestone or limit, not a person's actual age.
AGE_MILESTONE_RE = re.compile(r"(?:compimento|raggiungimento|compiere|raggiungere|compiuti i|fino a(?:i|gli|l)?|entro i|oltre i|superati i|prima de(?:i|gli|l)|al di (?:sopra|sotto) de(?:i|gli|l)|soglia de(?:i|gli|l)|limite de(?:i|gli|l))\s*(?:de(?:i|gli|l)\s*)?$", re.I)
LOOP_RE = re.compile(r"(?:\[(?:NOME|ETÀ|DATA|LUOGO/INDIRIZZO)\]\s*){4,}")

# v2 age-repeat policy "context". A marked age's digits may reappear unmarked in the redacted text. Each occurrence is
# read in its context: another age reference is rejected; a quantity that cannot be an age is allowed; anything else
# is ambiguous and also rejected, so only clear quantities escape the old blanket rule.
AGE_WORD_AFTER = re.compile(r"\s*-?\s*(?:anni|anno|aa\b\.?|a\.(?=\s|$)|enn[ei]|mesi|mese|settimane|settimana|giorni|giorno|primavere)", re.I)
AGE_WORD_BEFORE = re.compile(r"(?:\betà|\banni|\baa\.?|\baged?)\s*(?:[:=]|di|pari a|circa|apparente)?\s*(?:di\s*)?$", re.I)
# Units and quantity words that never follow a person's age. Time units shorter than a day count too. Whole-word units must
# end the word, so "54 l'esame" is not read as litres.
QUANTITY_PREFIX = r"%|‰|°|x\s*10|compress|spruzz|erogazion|batt|/\s*(?:die|min|h|kg|m²|mm|µl|μl|l|dl|ml)(?![\w'’])"
QUANTITY_WORD = (r"mg|mcg|µg|μg|ug|ng|pg|g|kg|ml|dl|cl|l|mmhg|cmh2o|mm|mm³|mmc|cm|m|km|mmol|µmol|μmol|nmol|pmol|meq|mosm|mui|ui|iu|u|ku"
                 r"|fl|bpm|atti|cicli|respiri|gtt|gocce|cp|fiala|fiale|flaconi|unità|puff|kcal|gy|msv|mbq|ms|msec|sec|secondi|s|min|minuti"
                 r"|ore|h|hz|cc|volte|punti|kpa|mv|db|ppm|gradi|grado|litri|litro|millilitri|grammi|grammo|milligrammi|microgrammi|chili"
                 r"|chilogrammi|metri|metro|centimetri|millimetri|sedute|sessioni|dosi|sigarette|pacchetti|millimoli|battiti")
QUANTITY_AFTER = re.compile(r"(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?\s*(?:(?:" + QUANTITY_PREFIX + r")|(?:" + QUANTITY_WORD + r")(?![\w'’]))", re.I)
# Laboratory, vital-sign, score and dose labels written directly before a value. Abbreviations are case-sensitive, so
# "ca" for circa is not read as calcium; words are not.
QUANTITY_LABEL_BEFORE = re.compile(
    r"(?<![\w-])(?:Hb|PCR|VES|FC|FR|PA|PAS|PAD|PAM|SpO2|SatO2|SaO2|Sat|GCS|NIHSS|EF|FE|FEVS|INR|TSH|FT3|FT4|BMI|HbA1c|Na|K|Cl|Ca|Mg|PLT|GB|GR"
    r"|WBC|RBC|Hct|MCV|LDH|CPK|CK|AST|ALT|GOT|GPT|GGT|ALP|pH|pCO2|pO2|HCO3|FEV1|FVC|DLCO|VAS|NRS|MMSE|MoCA|ECOG|PSA|CEA|CA\s?125|CA\s?19-9"
    r"|AFP|BNP|NT-proBNP|LDL|HDL|eGFR|VFG|QTc|QRS|SUV(?:max)?|Ki-?67|CD4"
    r"|(?i:glicemia|creatinin[ae]|azotemia|sodio|potassio|cloro|calcio|magnesio|fosforo|piastrine|leucociti|neutrofili|linfociti|emoglobina"
    r"|ematocrito|bilirubina|albumina|ferritina|troponina|lattati|score|punteggio|apgar|barthel|karnofsky|mallampati|gleason|hounsfield"
    r"|frequenza(?: cardiaca| respiratoria)?|temperatura|pressione(?: arteriosa)?|saturazione|peso|altezza|statura|circonferenza|diametro"
    r"|volume|dose|dosaggio|posologia|valore|valori|livell[oi]|conta|titolo|grado|stadio|d-dimero|fibrinogeno|procalcitonina|colesterolo"
    r"|trigliceridi|uricemia|sistolica|diastolica|frazione di eiezione))"
    r"\s*(?:[:=]|(?i:di|pari a|a|circa|fino a))?\s*$")
# A decimal or clock part ("1,54", "10:54") with no space, or the second term of a ratio or product ("120/54", "3 x 54").
NUMERIC_PART_BEFORE = re.compile(r"(?:\d[.,:]|\d\s*[/×x*]\s*|[/×*±]\s*)$")
NUMERIC_PART_AFTER = re.compile(r"(?:[.,:]\d|\s*/\s*\d|\s*×|\s*\*\s*\d)")


# v2 rule "reject_unmarked_named_facility". The prompt defines named hospitals and healthcare facilities as LUOGO/INDIRIZZO,
# so a facility word (any case, e.g. "ospedale Santo ...") followed by a capitalised proper name outside any entity is an
# unmarked identifier. Generic descriptors are allowed.
FACILITY_WORD = r"(?i:Ospedale|Arcispedale|Nosocomio|Policlinico|Presidio(?:\s+Ospedaliero)?|Istituto|Clinica|Fondazione|Casa\s+di\s+Cura|IRCCS|Centro|Poliambulatorio|Hospice|Azienda\s+Ospedaliera)"
NAMED_FACILITY_RE = re.compile(r"\b" + FACILITY_WORD + r"\s+(?:(?:di|del|della|dei|degli)\s+)?(San\b|Santa\b|Sant['’]|Santo\b|SS\.|S\.|[A-ZÀ-Ý][\w'’-]+)")
GENERIC_FACILITY_WORDS = {"Civile", "Maggiore", "Generale", "Regionale", "Provinciale", "Universitario", "Universitaria", "Ospedaliero", "Ospedaliera",
                          "Ospedaliero-Universitaria", "Territoriale", "Nazionale", "Locale", "Clinico", "Clinica", "Diagnostico", "Medica", "Medico",
                          "Chirurgica", "Chirurgico", "Specialistico", "Sanitario", "Sanitaria", "Pediatrico", "Oncologico", "Cardiologico",
                          "Neurologico", "Psichiatrico", "Ortopedico", "Riabilitativo", "Trasfusionale", "Antidiabetico", "Dialisi", "Ricerca",
                          "Riferimento", "Coordinamento", "Salute", "Cura", "Prelievi", "Traumatologico", "Ustioni", "Antiveleni", "Trapianti"}


def unmarked_named_facilities(redacted: str) -> int:
    """Named facilities left outside entities, such as 'Ospedale San ...' or 'Istituto Gaslini'."""
    return sum(1 for m in NAMED_FACILITY_RE.finditer(redacted) if m.group(1) not in GENERIC_FACILITY_WORDS)


def classify_age_repeat(redacted: str, start: int, end: int) -> str:
    """'age', 'quantity' or 'ambiguous' for an unmarked occurrence of a marked age's digits."""
    before, after = redacted[max(0, start - 40) : start], redacted[end : end + 30]
    if AGE_WORD_AFTER.match(after) or AGE_WORD_BEFORE.search(before):
        return "age"
    if NUMERIC_PART_BEFORE.search(before) or NUMERIC_PART_AFTER.match(after) or QUANTITY_AFTER.match(after) or QUANTITY_LABEL_BEFORE.search(before):
        return "quantity"
    return "ambiguous"


_IT_UNITS = {"uno": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6, "sette": 7, "otto": 8, "nove": 9}
_IT_TEENS = {"dieci": 10, "undici": 11, "dodici": 12, "tredici": 13, "quattordici": 14, "quindici": 15, "sedici": 16,
             "diciassette": 17, "diciotto": 18, "diciannove": 19}
_IT_TENS = {"venti": 20, "trenta": 30, "quaranta": 40, "cinquanta": 50, "sessanta": 60, "settanta": 70, "ottanta": 80, "novanta": 90}


def _italian_below_100(w: str) -> int | None:
    if w in _IT_UNITS:
        return _IT_UNITS[w]
    if w in _IT_TEENS:
        return _IT_TEENS[w]
    for word, value in _IT_TENS.items():
        if w == word:
            return value
        if w.startswith(word) and w[len(word):] in _IT_UNITS and w[len(word):] not in ("uno", "otto"):
            return value + _IT_UNITS[w[len(word):]]
        if w.startswith(word[:-1]) and w[len(word) - 1:] in ("uno", "otto"):   # elision: ventuno, trentotto
            return value + _IT_UNITS[w[len(word) - 1:]]
    return None


def italian_number_word(value: str) -> int | None:
    """An age written as one Italian number word, 1 to 130 ('settantadue', 'ventitré', 'centouno'). Articles such as
    'un' or 'una' are not read as numbers. Returns None for anything else."""
    w = value.strip().lower().replace("é", "e").replace("è", "e")
    if not re.fullmatch(r"[a-z]+", w):
        return None
    n = 0
    if w.startswith("cento"):
        n, w = 100, w[5:]
        if not w:
            return 100
        if w == "tto":                                      # centotto
            w = "otto"
    rest = _italian_below_100(w)
    if rest is None:
        return None
    n += rest
    return n if 1 <= n <= 130 else None


def age_value(text: str, accept_spelled: bool) -> int | None:
    """Numeric value of an ETÀ entity: digits, or one Italian number word when spelled ages are accepted."""
    if re.fullmatch(r"\d{1,3}", text):
        return int(text)
    return italian_number_word(text) if accept_spelled else None


def age_repeat_occurrences(value: str, redacted: str) -> list[tuple[int, int]]:
    """Unmarked occurrences of an age's digits: standalone, or glued to an age suffix such as '54enne'."""
    found = [(m.start(), m.end()) for m in word_boundary_regex(value).finditer(redacted)]
    found += [(m.start(), m.start() + len(value)) for m in re.finditer(r"(?<!\w)" + re.escape(value) + r"(?=enn[ei]\b|aa\b)", redacted, re.I)]
    return sorted(set(found))
GENERIC_PLACE_WORDS = {"ospedale", "ospedaliera", "ospedaliero", "policlinico", "istituto", "clinica", "azienda",
                       "sanitaria", "universitaria", "centro", "presidio", "casa", "cura", "reparto", "unità",
                       "dipartimento", "servizio", "fondazione", "provincia", "regione", "comune", "distretto",
                       "pronto", "soccorso", "medicina", "chirurgia", "via", "viale", "piazza", "corso", "santo",
                       "santa", "san", "generale", "civile", "maggiore", "nazionale", "regionale", "locale"}


@dataclass
class ValidationContext:
    train_hashes: set
    train_token_docs: dict
    train_index: dict
    anchor_notes: list
    seen_hashes: set
    cfg: dict
    accepted_tokens: dict = field(default_factory=dict)   # v2: notes already accepted in this fold and arm
    accepted_index: dict = field(default_factory=dict)


def build_context(train_notes, cfg: dict) -> dict:
    """Pre-computed training-partition data shared by all validation calls in a fold."""
    toks = {n.note_id: tokens(n.text) for n in train_notes}
    index = collections.defaultdict(set)
    for k, t in toks.items():
        for g in word_ngrams(t, 8):
            index[g].add(k)
    return {"train_hashes": {doc_hash(n.text) for n in train_notes}, "train_token_docs": toks, "train_index": index}


@dataclass
class ValidationResult:
    accepted: bool
    reasons: list = field(default_factory=list)
    note: dict | None = None
    flags: dict = field(default_factory=dict)


def build_note(case: dict) -> tuple[dict | None, list[str]]:
    """Turn segments into text, redacted text and character-offset entities."""
    segs = case.get("segments") if isinstance(case, dict) else None
    if not isinstance(segs, list) or not segs:
        return None, ["malformed_segments"]
    text, red, ents, reasons = [], [], [], []
    pos = 0
    prev_entity = False
    for s in segs:
        if not isinstance(s, dict) or not isinstance(s.get("text"), str):
            return None, ["malformed_segment"]
        raw, kind = s["text"], s.get("entity_type")
        if kind == "NESSUNA":
            text.append(raw); red.append(raw); pos += len(raw)
            prev_entity = prev_entity and raw == ""
            continue
        t = canonical_type(kind) if kind in ENTITY_TYPES else None
        if t is None:
            return None, ["invalid_entity_type"]
        value = raw.strip()
        if not value:
            reasons.append("empty_entity")
            continue
        if prev_entity:
            reasons.append("adjacent_entities")
        lead = raw[: len(raw) - len(raw.lstrip())]
        trail = raw[len(raw.rstrip()) :]
        text.append(lead); red.append(lead); pos += len(lead)
        ents.append({"start": pos, "end": pos + len(value), "type": t, "text": value})
        text.append(value); red.append(PLACEHOLDERS[t]); pos += len(value)
        text.append(trail); red.append(trail); pos += len(trail)
        prev_entity = not trail
    return {"text": "".join(text), "redacted_text": "".join(red), "entities": ents}, reasons


def _entity_format_reasons(e: dict, max_chars: int, accept_spelled_ages: bool = False) -> list[str]:
    x, t, out = e["text"], e["type"], []
    if len(x) > max_chars:
        out.append("entity_too_long")
    if t == "ETÀ" and age_value(x, accept_spelled_ages) is None:
        out.append("eta_not_numeric")
    if t == "DATA" and not (re.search(r"\d", x) or MONTH.search(x)):
        out.append("date_malformed")
    if t == "NOME" and (re.search(r"\d", x) or not re.search(r"[^\W\d_]", x)):
        out.append("name_malformed")
    if t == "LUOGO/INDIRIZZO" and not re.search(r"[^\W\d_]", x):
        out.append("place_malformed")
    return out


def _left_in_redacted(ents: list[dict], redacted: str, age_policy: str = "strict", flags: dict | None = None) -> list[str]:
    out = set()
    for e in ents:
        x, t = e["text"], e["type"]
        if t == "ETÀ" and age_policy == "context":
            for s, end in age_repeat_occurrences(x, redacted):
                kind = classify_age_repeat(redacted, s, end)
                if kind == "age":
                    out.add("age_repeated_unmarked")
                elif kind == "ambiguous":
                    out.add("age_value_repeated_ambiguous")
                elif flags is not None:
                    flags["age_value_as_quantity"] = flags.get("age_value_as_quantity", 0) + 1
        elif word_boundary_regex(x).search(redacted):
            out.add("entity_left_in_redacted")
        if t == "NOME":
            for part in re.findall(r"[^\W\d_]{3,}", x):
                if word_boundary_regex(part, re.I).search(redacted):
                    out.add("name_part_left_in_redacted")
        if t == "LUOGO/INDIRIZZO":
            for part in re.findall(r"[^\W\d_]{4,}", x):
                if part[0].isupper() and part.lower() not in GENERIC_PLACE_WORDS and word_boundary_regex(part).search(redacted):
                    out.add("place_part_left_in_redacted")
    return sorted(out)


def validate_case(case: dict, spec, ctx: ValidationContext) -> ValidationResult:
    v = ctx.cfg["validation"]
    note, reasons = build_note(case)
    if note is None:
        return ValidationResult(False, reasons)
    reasons = list(reasons)
    text, red, ents = note["text"], note["redacted_text"], note["entities"]
    spelled = bool(v.get("accept_spelled_ages"))   # owner decision 2026-11-09: an age written as an Italian number word is ETÀ
    if case.get("case_id") != spec.case_id:
        reasons.append("case_id_mismatch")
    for e in ents:
        if text[e["start"] : e["end"]] != e["text"]:
            reasons.append("offset_inconsistent")
        reasons += _entity_format_reasons(e, v["max_entity_chars"], spelled)
    age_flags = {}
    policy = v.get("age_repeat_policy", "strict")
    if policy not in ("strict", "context"):
        raise ValueError(f"unknown age_repeat_policy {policy!r}")
    reasons += _left_in_redacted(ents, red, policy, age_flags)
    counts = collections.Counter(e["type"] for e in ents)
    if any(PLACEHOLDERS[t] not in red for t in counts):
        reasons.append("placeholder_missing_for_present_type")
    if BRACKET_RE.search(text):
        reasons.append("placeholder_in_text")
    if LOOP_RE.search(red):
        reasons.append("placeholder_loop")
    by_str = collections.defaultdict(set)
    for e in ents:
        by_str[e["text"]].add(e["type"])
    if any(len(k) > 1 for k in by_str.values()):
        reasons.append("contradictory_types")

    # length and entity-count rules
    if not v["min_chars"] <= len(text) <= v["max_chars"]:
        reasons.append("length_out_of_range")
    lo, hi = v["length_tolerance"]
    if not lo * spec.target_chars <= len(text) <= hi * spec.target_chars:
        reasons.append("length_off_target")
    if spec.record_type == "synthetic_negative":
        if ents:
            reasons.append("entity_in_negative")
    else:
        if len(ents) < v["min_entities"]:
            reasons.append("too_few_entities")
        if spec.record_type == "synthetic_name_free" and counts["NOME"]:
            reasons.append("name_in_name_free")
        for t in ENTITY_TYPES:
            tol = max(v["count_tolerance_min"], round(v["count_tolerance_share"] * spec.counts[t]))
            if abs(counts[t] - spec.counts[t]) > tol:
                reasons.append("entity_count_off_plan")
                break
        if spec.record_type == "synthetic_named" and not counts["NOME"]:
            reasons.append("name_missing_in_named")
    reasons += residual_pii_reasons(red)
    # v2 rules, active only when configured, so v1 runs keep their recorded behaviour.
    if v.get("reject_line_breaks") and LINE_BREAK_RE.search(text):
        reasons.append("line_break_in_text")
    if v.get("reject_age_milestones"):
        for e in ents:
            if e["type"] == "ETÀ" and AGE_MILESTONE_RE.search(text[max(0, e["start"] - 40) : e["start"]]):
                reasons.append("age_milestone_marked")
    if v.get("reject_unmarked_named_facility") and unmarked_named_facilities(red):
        reasons.append("named_facility_unmarked")
    if v.get("require_patient_age") and getattr(spec, "patient_age", None) is not None:
        if int(spec.patient_age) not in {age_value(e["text"], spelled) for e in ents if e["type"] == "ETÀ"}:
            reasons.append("patient_age_missing")

    # duplicates and copying, against the training partition only
    h = doc_hash(text)
    if h in ctx.train_hashes or h in ctx.seen_hashes:
        reasons.append("exact_duplicate")
    toks = tokens(text)
    hits = collections.Counter()
    grams = word_ngrams(toks, 8)
    for g in grams:
        for k in ctx.train_index.get(g, ()):
            hits[k] += 1
    for k, shared in hits.items():
        other = ctx.train_token_docs[k]
        if shared / max(1, min(len(grams), len(other))) >= v["near_duplicate_containment"]:
            reasons.append("near_duplicate_of_training_note")
        if longest_common_run(toks, other) >= v["max_verbatim_run_train"]:
            reasons.append("long_verbatim_run_with_training_note")
    limit = v.get("max_verbatim_run_accepted")
    if limit:
        candidates = collections.Counter(k for g in grams for k in ctx.accepted_index.get(g, ()))
        if any(longest_common_run(toks, ctx.accepted_tokens[k]) >= limit for k in candidates):
            reasons.append("near_duplicate_of_accepted_note")
    for a in ctx.anchor_notes:
        if longest_common_run(toks, tokens(a.text)) >= v["max_verbatim_run_anchor"]:
            reasons.append("anchor_text_copied")
        for e in a.entities:
            x = e.get("text", "")
            if canonical_type(e.get("type")) in ("NOME", "LUOGO/INDIRIZZO") and len(x) >= 4 and word_boundary_regex(x).search(text):
                reasons.append("anchor_pii_copied")

    reasons = sorted(set(reasons))
    every, unexplained = capitalised_word_counts(red)
    flags = {"midsentence_capitalised_in_redacted": every, "capitalised_not_explained_by_eponym": unexplained}
    if policy == "context":
        flags["age_value_as_quantity"] = age_flags.get("age_value_as_quantity", 0)
    if reasons:
        return ValidationResult(False, reasons, note, flags)
    return ValidationResult(True, [], note, flags)


def validate_negative_example(input_text: str, output_text: str, annotated_mentions: int) -> list[str]:
    """Rules for any record trained with output = input. Empty list means valid."""
    reasons = []
    if input_text != output_text:
        reasons.append("output_differs_from_input")
    if annotated_mentions:
        reasons.append("annotated_pii_present")
    reasons += residual_pii_reasons(input_text)
    if PLACEHOLDER_RE.search(input_text):
        reasons.append("placeholder_in_text")
    return reasons
