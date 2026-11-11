"""Checks an external review of annotation_review.csv against the full notes. It does not change gold data.

Writes IDs, offsets and counts to audit/results/annotation_review_check.json, a no-text proposal to
data_manifests/gold_annotation_proposal.csv, and new candidates with context to the git-ignored
data_secure/gold/annotation_review_additional.csv."""
import collections, csv, io, json, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.annotations import word_boundary_regex
from src.data.build_gold import RAW_CSV, split_input
from src.data.loaders import GOLD_V1, load_gold
from src.data.manifests import write_csv
from src.generation.privacy_screening import MONTHS

csv.field_size_limit(10 ** 9)
gold = {n.note_id: n for n in load_gold(GOLD_V1)}   # this review was made against gold_v1
rows = list(csv.DictReader(open(ROOT / "data_secure/gold/annotation_review.csv", encoding="utf-8", newline="")))
src = list(csv.DictReader(io.StringIO(RAW_CSV.read_bytes().decode("utf-8"), newline="")))
mask = lambda s: re.sub(r"[^\W\d_]", "a", re.sub(r"\d", "9", s))
out = {}

# 1. Do the review's row numbers match this file?
ranges = {"gold_000": (1, 5), "gold_001": (6, 12), "gold_002": (13, 18), "gold_004": (19, 32), "gold_007": (33, 33), "gold_008": (34, 34), "gold_009": (35, 36),
          "gold_010": (37, 37), "gold_015": (38, 39), "gold_016": (40, 40), "gold_032": (41, 41), "gold_040": (42, 42), "gold_046": (43, 45), "gold_047": (46, 47),
          "gold_049": (48, 51), "gold_054": (52, 53), "gold_055": (54, 55), "gold_056": (56, 56), "gold_058": (57, 60), "gold_060": (61, 62), "gold_062": (63, 66),
          "gold_066": (67, 68), "gold_067": (69, 70), "gold_068": (71, 73), "gold_071": (74, 76), "gold_072": (77, 81), "gold_074": (82, 85), "gold_076": (86, 86), "gold_077": (87, 89)}
row_note = {r: k for k, (a, b) in ranges.items() for r in range(a, b + 1)}
out["rows_total"] = len(rows)
out["rows_where_note_id_differs"] = [i for i, r in enumerate(rows, 1) if row_note.get(i) != r["note_id"]]

# 2. Age convention across ALL age mentions
forms = collections.Counter(); after = collections.Counter()
for n in gold.values():
    for e in n.entities:
        if e["type"] != "ETÀ":
            continue
        x = e["text"]
        forms["digits_only" if x.isdigit() else "contains_unit_word" if re.search(r"ann|mes|settiman|giorn", x, re.I) else "spelled_or_other"] += 1
        if e["start"] is not None and x.isdigit():
            m = re.match(r"\s*-?\s*([^\W\d_]+)", n.text[e["end"]:e["end"] + 14])
            w = (m.group(1).lower() if m else "")
            after[w if w in ("anni", "anno", "mesi", "mese", "settimane", "giorni", "enne", "aa") else "other_or_none"] += 1
out["age_all_mentions"] = {"total": sum(forms.values()), "forms": dict(forms), "word_after_digit_mentions": dict(after)}

# 3. The review's six open items, plus the surplus entries
raw_cell = {f"gold_{i:03d}": split_input(r["input"]) for i, r in enumerate(src)}
surplus = []
for nid, n in gold.items():
    for s in sorted({e["annotated_text"] for e in n.entities if e["status"] == "annotated_more_often_than_occurs"}):
        tail = raw_cell[nid][2]
        surplus.append({"note_id": nid, "type": next(e["type"] for e in n.entities if e["annotated_text"] == s), "listed": sum(e["annotated_text"] == s for e in n.entities),
                        "in_narrative_exact": len(word_boundary_regex(s).findall(n.text)), "in_narrative_any_case": len(word_boundary_regex(s, re.I).findall(n.text)),
                        "in_relation_tail": len(word_boundary_regex(s).findall(tail)), "shape": mask(s)})
out["surplus_entries"] = surplus

n = gold["gold_062"]; hit = next(r for r in rows if r["note_id"] == "gold_062" and r["issue"] == "extra_occurrence" and r["value_in_note"].strip() == "1" and int(r["start"]) == 539) if any(r["note_id"] == "gold_062" and r["start"] == "539" for r in rows) else None
tok = re.search(r"[\w.,:/-]*$", n.text[max(0, 539 - 15):539]).group(0) + n.text[539:540] + re.match(r"[\w.,:/-]*", n.text[540:555]).group(0)
out["gold_062_position_539"] = {"row_found": bool(hit), "token_shape": mask(tok), "looks_like_marker_dash_1": bool(re.fullmatch(r"[^\W\d_]{2,6}-1", tok))}

n = gold["gold_077"]; dates = [e for e in n.entities if e["type"] == "DATA"]
out["gold_077_dates"] = {"data_mentions": len(dates), "shapes_with_offsets": [(e["start"], mask(e["text"])) for e in dates], "month_names_left_in_target": len(re.findall(rf"\b(?:{MONTHS})\b", n.redacted_text, re.I))}

n = gold["gold_010"]; e = next(e for e in n.entities if e["status"] == "located_fuzzy_variant")
cand = [i for i in range(len(n.text)) if n.text.startswith(e["text"], i)]
out["gold_010_date"] = {"annotated_digits_differ_at": [i for i, (a, b) in enumerate(zip(e["annotated_text"], e["text"])) if a != b], "annotated_string_in_note": e["annotated_text"] in n.text, "note_string_occurrences": len(cand), "other_data_mentions_in_note": sum(x["type"] == "DATA" for x in n.entities) - 1}

n = gold["gold_002"]; six = [e for e in n.entities if e["type"] == "ETÀ" and e["text"] == "6"]
FUNC = {"da", "dopo", "per", "di", "circa", "età", "all", "a", "nei", "negli", "ultimi", "ultime", "al", "dell", "ad", "tra", "entro", "in"}
words = [re.findall(r"[^\W\d_]+", n.text[e["start"] - 32:e["start"]])[-3:] for e in six]
out["gold_002_age_6"] = {"located_mentions": len(six), "preceding_words": [[w.lower() if w.lower() in FUNC else "other" for w in x] for x in words], "age_word_within_32_chars": [bool(re.search(r"\betà\b", n.text[e["start"] - 32:e["start"]], re.I)) for e in six], "following_word": [re.match(r"\s*(\w+)", n.text[e["end"]:e["end"] + 12]).group(1).lower() for e in six]}

# 4. Check the proposed complete dates
FULL = re.compile(rf"\b(?:\d{{1,2}}\s+)?(?:{MONTHS})(?:\s+(?:del|dell'|di)\s*|\s+)?\d{{4}}\b|(?<![\d.,/-])(?:19|20)\d{{2}}(?![\d.,]*\d)", re.I)
add_dates = []
for nid in ("gold_000", "gold_001", "gold_002", "gold_004"):
    n = gold[nid]; taken = [(e["start"], e["end"]) for e in n.entities if e["start"] is not None]
    for m in FULL.finditer(n.text):
        if not any(m.start() < b and m.end() > a for a, b in taken):
            add_dates.append({"note_id": nid, "start": m.start(), "end": m.end(), "shape": mask(m.group(0))})
out["proposed_date_spans"] = {"count": len(add_dates), "per_note": dict(collections.Counter(a["note_id"] for a in add_dates)), "spans": add_dates}

# 5. Coverage the 89 rows could not show
def covered(n):
    return [(e["start"], e["end"]) for e in n.entities if e["start"] is not None] + [(e["start"], e["end"]) for e in n.extra_occurrences]
extra_rows = []
own = collections.Counter(); cross = collections.Counter(); per_note_variant = {}
all_strings = {t: set() for t in ("NOME", "LUOGO/INDIRIZZO")}
for n in gold.values():
    for e in n.entities:
        if e["type"] in all_strings and len(e["annotated_text"]) >= 4 and not re.fullmatch(r"(?:\w\.\s?)+", e["annotated_text"]):
            all_strings[e["type"]].add(e["annotated_text"])
for nid, n in gold.items():
    cov = covered(n)
    annotated_here = {e["annotated_text"] for e in n.entities if e["type"] in all_strings}
    for t, strs in all_strings.items():
        for s in strs:
            for m in word_boundary_regex(s, re.I).finditer(n.text):
                if any(m.start() < b and m.end() > a for a, b in cov):
                    continue
                kind = "same_note_case_variant" if s in annotated_here else "string_annotated_in_another_note"
                (own if s in annotated_here else cross)[t] += 1
                if s in annotated_here:
                    per_note_variant[nid] = per_note_variant.get(nid, 0) + 1
                extra_rows.append({"note_id": nid, "issue": kind, "type": t, "value_in_note": m.group(0), "start": m.start(), "context": n.text[max(0, m.start() - 40):m.end() + 40]})
out["same_note_case_variants_uncovered"] = {"by_type": dict(own), "per_note": per_note_variant}
out["string_annotated_elsewhere_uncovered"] = dict(cross)

PLACES = """Agrigento Alessandria Ancona Aosta Arezzo Ascoli Asti Avellino Bari Barletta Belluno Benevento Bergamo Biella Bologna Bolzano Brescia Brindisi Cagliari Caltanissetta Campobasso Caserta Catania Catanzaro Chieti Como Cosenza Cremona Crotone Cuneo Enna Fermo Ferrara Firenze Foggia Forlì Frosinone Genova Gorizia Grosseto Imperia Isernia Latina Lecce Lecco Livorno Lodi Lucca Macerata Mantova Massa Matera Messina Milano Modena Monza Napoli Novara Nuoro Oristano Padova Palermo Parma Pavia Perugia Pesaro Pescara Piacenza Pisa Pistoia Pordenone Potenza Prato Ragusa Ravenna Reggio Rieti Rimini Roma Rovigo Salerno Sassari Savona Siena Siracusa Sondrio Taranto Teramo Terni Torino Trapani Trento Treviso Trieste Udine Urbino Varese Venezia Verbania Vercelli Verona Vicenza Viterbo Abruzzo Basilicata Calabria Campania Lazio Liguria Lombardia Marche Molise Piemonte Puglia Sardegna Sicilia Toscana Umbria Veneto Italia Francia Germania Spagna Svizzera Inghilterra Marocco Albania Romania Tunisia Egitto Londra Parigi Berlino Madrid Vienna Mosca Bruxelles Amsterdam Zurigo Ginevra""".split()
place = collections.Counter(); place_notes = set()
for nid, n in gold.items():
    cov = covered(n)
    for s in PLACES:
        for m in word_boundary_regex(s, re.I).finditer(n.text):
            if not any(m.start() < b and m.end() > a for a, b in cov):
                place["uncovered"] += 1; place_notes.add(nid)
                extra_rows.append({"note_id": nid, "issue": "place_name_not_annotated", "type": "LUOGO/INDIRIZZO", "value_in_note": m.group(0), "start": m.start(), "context": n.text[max(0, m.start() - 40):m.end() + 40]})
out["gazetteer_places_uncovered"] = {"mentions": place["uncovered"], "notes": sorted(place_notes), "gold_058_has_uncovered_place": "gold_058" in place_notes}

SPELLED = "uno|una|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|undici|dodici|tredici|quattordici|quindici|sedici|diciassette|diciotto|diciannove|\\w*vent\\w*|\\w*trent\\w*|\\w*quarant\\w*|\\w*cinquant\\w*|\\w*sessant\\w*|\\w*settant\\w*|\\w*ottant\\w*|\\w*novant\\w*"
AGE_FORMS = {"age_digit_enne": re.compile(r"\b\d{1,3}\s*-?\s*enne\b", re.I), "age_word_enne": re.compile(r"\b\w{3,}enne\b", re.I),
             "age_spelled_with_context": re.compile(rf"\b(?:età di|di|ha|aveva)\s+(?:{SPELLED})\s+(?:anni|mesi)\b", re.I), "age_apostrophe_anni": re.compile(r"\b\w+'anni\b", re.I)}
NOT_AGE = re.compile(r"\b(?:antenne|ipertenne|ventenne_placeholder|tenne|ottenne|mantenne|trattenne|sostenne|contenne|ritenne|venne|divenne|provenne|intervenne|pervenne|convenne|Vienne|penne|senne|enne|perenne)\b", re.I)
age_counts = collections.Counter(); age_notes = collections.defaultdict(set)
for nid, n in gold.items():
    cov = covered(n)
    for k, p in AGE_FORMS.items():
        for m in p.finditer(n.text):
            if NOT_AGE.fullmatch(m.group(0)):
                continue
            if any(m.start() < b and m.end() > a for a, b in cov):
                continue
            age_counts[k] += 1; age_notes[k].add(nid)
            extra_rows.append({"note_id": nid, "issue": k, "type": "ETÀ", "value_in_note": m.group(0), "start": m.start(), "context": n.text[max(0, m.start() - 40):m.end() + 40]})
out["unannotated_age_forms"] = {k: {"mentions": v, "notes": sorted(age_notes[k])} for k, v in age_counts.items()}

caps = collections.Counter(); cap_notes = collections.Counter()
for nid, n in gold.items():
    cov = covered(n)
    for m in re.finditer(r"(?<=[a-zà-ù,;:] )[A-ZÀ-Ý][a-zà-ù]{2,}", n.text):
        if not any(m.start() < b and m.end() > a for a, b in cov):
            caps[m.group(0)] += 1
out["midsentence_capitalised_words_uncovered"] = {"mentions": sum(caps.values()), "distinct_words": len(caps)}
for w, c in caps.items():
    if c == 1:
        for nid, n in gold.items():
            for m in re.finditer(r"(?<=[a-zà-ù,;:] )" + re.escape(w) + r"\b", n.text):
                if not any(m.start() < b and m.end() > a for a, b in covered(n)):
                    extra_rows.append({"note_id": nid, "issue": "capitalised_word_once_in_corpus", "type": "", "value_in_note": w, "start": m.start(), "context": n.text[max(0, m.start() - 40):m.end() + 40]})
out["capitalised_words_once_in_corpus"] = sum(c == 1 for c in caps.values())
write_csv(ROOT / "data_secure/gold/annotation_review_additional.csv", ["note_id", "issue", "type", "value_in_note", "start", "context"], extra_rows)
out["additional_rows_by_issue"] = dict(collections.Counter(r["issue"] for r in extra_rows))
(ROOT / "audit/results/annotation_review_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "proposed_date_spans"}, ensure_ascii=False, indent=0))
print("proposed dates:", out["proposed_date_spans"]["count"], out["proposed_date_spans"]["per_note"], sorted({a["shape"] for a in out["proposed_date_spans"]["spans"]}))

# 6. Row-by-row outcome, with offsets and no text
ACTIONS = {"keep": [6, 7, 8, 9, 43, 44, 46, 48, 49, 52, 54, 58, 77, 79, 82, 83], "add_nome": [10, 42, 47, 50, 51, 53, 55, 81, 84, 85], "add_luogo": [45, 60],
           "add_data": [2, 3, 4, 5, 11, 12, 18] + list(range(19, 33)), "reject_not_pii": [15, 16, 17, 34, 35, 36, 39, 40, 41, 62, 64, 65, 66, 68, 70, 72, 73, 75, 76, 80, 86, 88, 89],
           "remove_age_is_a_duration": [14], "remove_surplus_listing": [1, 56], "locate_variant_mention": [33], "correct_source_typo": [37], "already_handled": [59],
           "age_boundary_pending_decision": [13, 38, 57, 61, 63, 67, 69, 71, 74, 78, 87]}
row_action = {r: a for a, rs in ACTIONS.items() for r in rs}
assert sorted(row_action) == list(range(1, 90))
spans = collections.defaultdict(list)
for a in add_dates:
    spans[a["note_id"]].append(a)
date_rows = {2: ("gold_000", 0), 4: ("gold_000", 0), 3: ("gold_000", 1), 5: ("gold_000", 1), 11: ("gold_001", 0), 12: ("gold_001", 1), 18: ("gold_002", 0)}
date_rows.update({r: ("gold_004", r - 19) for r in range(19, 26)}); date_rows.update({r: ("gold_004", r - 26) for r in range(26, 33)})
proposal = []
for i, r in enumerate(rows, 1):
    s, e = "", ""
    if i in date_rows:
        d = spans[date_rows[i][0]][date_rows[i][1]]; s, e = d["start"], d["end"]
    elif r["start"] not in ("", None):
        s = int(r["start"]); e = s + len(r["value_in_note"])
    elif i == 33:
        # The second listing is the same hospital written without the linking word, found on reading the notes.
        parts = re.findall(r"[^\W\d_]+", r["annotated"])
        v = re.search(parts[0] + r"\s+" + parts[-1], gold["gold_007"].text)
        s, e = v.span()
    elif i == 42:
        m = re.match(r"\s*[A-Z]\.\s?[A-Z]\.", gold["gold_040"].text); s, e = (m.start(), m.end()) if m else ("", "")
    proposal.append({"row": i, "note_id": r["note_id"], "issue": r["issue"], "action": row_action[i], "type": {"add_nome": "NOME", "add_luogo": "LUOGO/INDIRIZZO", "add_data": "DATA"}.get(row_action[i], r["type"]), "start": s, "end": e})
write_csv(ROOT / "data_manifests/gold_annotation_proposal.csv", ["row", "note_id", "issue", "action", "type", "start", "end"], proposal)
bad = [p["row"] for p in proposal if p["action"] in ("add_nome", "add_luogo", "add_data") and p["start"] == ""]
n58 = gold["gold_058"]; lon = [m.span() for m in word_boundary_regex("Londra", re.I).finditer(n58.text)]
places = [x for x in extra_rows if x["issue"] == "place_name_not_annotated"]
out["proposal"] = {"rows": len(proposal), "by_action": {a: len(rs) for a, rs in ACTIONS.items()}, "add_rows_without_offsets": bad,
                   "gold_040_initials_at_note_start": proposal[41]["start"] != "", "initials_already_annotated_as_NOME_in_corpus": sum(e["type"] == "NOME" and bool(re.fullmatch(r"(?:[A-Za-z]\.\s?)+", e["annotated_text"])) for nn in gold.values() for e in nn.entities)}
out["gold_058_place_check"] = {"occurrences": len(lon), "all_covered_by_a_mention": all(any(a < y and b > x for x, y in [(q["start"], q["end"]) for q in n58.entities if q["start"] is not None]) for a, b in lon)}
out["gold_072_name_mentions"] = {"annotated": sum(e["type"] == "NOME" for e in gold["gold_072"].entities), "extra_occurrences": sum(e["type"] == "NOME" for e in gold["gold_072"].extra_occurrences)}
out["place_candidates_detail"] = {"capitalised": sum(x["value_in_note"][0].isupper() for x in places), "lowercase_likely_ordinary_words": sum(x["value_in_note"][0].islower() for x in places),
                                  "capitalised_by_note": dict(collections.Counter(x["note_id"] for x in places if x["value_in_note"][0].isupper()))}
c = collections.Counter(e["type"] for nn in gold.values() for e in nn.entities)
out["counts_after_proposal_if_age_boundaries_unchanged"] = {"NOME": c["NOME"] - 2 + 10, "ETÀ": c["ETÀ"] - 1, "DATA": c["DATA"] + 12, "LUOGO/INDIRIZZO": c["LUOGO/INDIRIZZO"] + 2}
(ROOT / "audit/results/annotation_review_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: out[k] for k in ["proposal", "gold_058_place_check", "gold_072_name_mentions", "place_candidates_detail", "counts_after_proposal_if_age_boundaries_unchanged"]}, ensure_ascii=False, indent=0))
