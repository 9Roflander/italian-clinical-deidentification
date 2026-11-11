"""Follow-up checks after reading annotation_review_additional.csv. Does not change gold data.
Output: offsets, shapes and word categories only, written to audit/results/additional_rows_check.json."""
import collections, difflib, json, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.loaders import GOLD_V1, load_gold

gold = {n.note_id: n for n in load_gold(GOLD_V1)}   # this review was made against gold_v1
mask = lambda s: re.sub(r"[^\W\d_]", "a", re.sub(r"\d", "9", s))
def cov(n):
    return [(e["start"], e["end"]) for e in n.entities if e["start"] is not None] + [(e["start"], e["end"]) for e in n.extra_occurrences]
hit = lambda n, a, b: any(a < y and b > x for x, y in cov(n))
out = {}

# 1. Is each "surplus" listing really surplus, or a variant the exact match missed?
FUNC = {"di", "del", "della", "dello", "dei", "degli", "delle", "de", "d"}
res = []
for nid, n in gold.items():
    for s in sorted({e["annotated_text"] for e in n.entities if e["status"] == "annotated_more_often_than_occurs"}):
        parts = re.findall(r"[^\W\d_]+", s)
        pat = r"\s+".join(("(?:" + re.escape(p) + r"\s+)?") if p.lower() in FUNC else re.escape(p) for p in parts).replace(r")?\s+", ")?")
        variants = [(m.start(), m.end(), mask(m.group(0))) for m in re.finditer(pat, n.text, re.I) if not hit(n, m.start(), m.end())]
        close = []
        if len(parts) == 1:
            for m in re.finditer(r"[^\W\d_]{2,}", n.text):
                r = difflib.SequenceMatcher(None, s.lower(), m.group(0).lower()).ratio()
                if r >= 0.7 and not hit(n, m.start(), m.end()):
                    close.append({"start": m.start(), "shape": mask(m.group(0)), "ratio": round(r, 2)})
        initial = [m.start() for m in re.finditer(r"(?<![\w.])" + re.escape(s[0]) + r"\.(?!\w)", n.text) if not hit(n, m.start(), m.end())] if len(parts) == 1 else []
        res.append({"note_id": nid, "shape": mask(s), "uncovered_variant_spans": variants, "similar_uncovered_words": close, "uncovered_single_initial_at": initial})
out["surplus_listings"] = res

# 2. Initials at the start of gold_065, spotted in a row's context
n = gold["gold_065"]
out["gold_065_start"] = {"entities_starting_before_8": [(e["start"], e["type"], mask(e["text"])) for e in n.entities if e["start"] is not None and e["start"] < 8],
                         "first_token_shapes": [mask(t) for t in n.text.split()[:3]]}

# 3. Place rows
out["italia"] = {"annotated_as_mention": [nid for nid, x in gold.items() if any(e["type"] == "LUOGO/INDIRIZZO" and e["annotated_text"].lower() == "italia" for e in x.entities)],
                 "total_occurrences": sum(len(re.findall(r"\bItalia\b", x.text)) for x in gold.values())}
n = gold["gold_073"]; sp = re.search(r"\bItalia\b", n.text).start()
out["gold_073"] = {"place_mentions": [(e["start"], mask(e["text"])) for e in n.entities if e["type"] == "LUOGO/INDIRIZZO"], "unannotated_place_at": sp}

# 4. Ages written in words or digits with no span, by what precedes them
SPELLED = "uno|una|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|undici|dodici|tredici|quattordici|quindici|sedici|diciassette|diciotto|diciannove|\\w*vent\\w*|\\w*trent\\w*|\\w*quarant\\w*|\\w*cinquant\\w*|\\w*sessant\\w*|\\w*settant\\w*|\\w*ottant\\w*|\\w*novant\\w*"
DUR = {"da", "di", "dopo", "per", "circa", "ultimi", "ultime", "tra", "entro", "ogni", "oltre", "in", "a", "al", "fa", "ai", "negli", "nei", "dei", "degli", "primi", "prime", "successivi", "successive", "dal", "due", "tre", "sei", "e", "o", "ulteriori", "altri", "altre", "gli", "i", "le"}
rows = []; tally = collections.Counter()
for nid, n in gold.items():
    for kind, p in (("spelled", rf"\b({SPELLED})\s+(anni|anno|mesi|mese)\b"), ("digits", r"\b(\d{1,3})\s+(anni|anno|mesi|mese)\b")):
        for m in re.finditer(p, n.text, re.I):
            if hit(n, m.start(1), m.end(1)):
                continue
            before = re.findall(r"[^\W\d_]+|[,;:.()]", n.text[max(0, m.start() - 25):m.start()])
            prev = before[-1].lower() if before else "<start>"
            cat = "duration_like_word" if prev in DUR else "punctuation" if prev in ",;:.()" else "note_start" if prev == "<start>" else "other_word"
            tally[(kind, cat)] += 1
            if cat != "duration_like_word":
                rows.append({"note_id": nid, "start": m.start(1), "end": m.end(1), "kind": kind, "category": cat, "number_shape": mask(m.group(1)), "unit": m.group(2).lower(), "prev_shape": prev if prev in ",;:.()" else mask(prev)})
out["uncovered_age_like"] = {"tally": {f"{a}|{b}": v for (a, b), v in sorted(tally.items())}, "to_inspect": rows}

# 5. Exact spans for the additions spotted in the file
spans = {}
n = gold["gold_037"]; m = re.search(r"\b(\d{1,3})enne\b", n.text); spans["gold_037_age_number"] = [m.start(1), m.end(1)] if m else None
out["additional_spans"] = spans
out["gold_037_other_mentions"] = len(n.entities)
(ROOT / "audit/results/additional_rows_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))

# 6. Add age-like candidates that need context to the private review file (git-ignored)
import csv
PERSON = re.compile(r"^(?:uomo|donna|paziente|pazienti|ragazz[oa]|bambin[oa]|signor[ae]?|anzian[oa]|soggetto|giovane|maschio|femmina|figli[oa]|figlie|figli|madre|padre|marito|moglie|sorell[ae]|fratell[oi]|nonn[oa]|lattante|neonat[oa]|piccol[oa]|nipote|zi[oa]|madri|padri|compagn[oa]|familiare|parente|caregiver|neonati|bimb[oa]|adolescente)$", re.I)
rows2 = []
for r in out["uncovered_age_like"]["to_inspect"]:
    rows2.append((r["note_id"], r["start"], r["end"]))
for nid, n in gold.items():
    for p in (rf"\b({SPELLED})\s+(anni|anno|mesi|mese)\b", r"\b(\d{1,3})\s+(anni|anno|mesi|mese)\b"):
        for m in re.finditer(p, n.text, re.I):
            if hit(n, m.start(1), m.end(1)):
                continue
            before = re.findall(r"[^\W\d_]+", n.text[max(0, m.start() - 40):m.start()])
            if len(before) >= 2 and before[-1].lower() in ("di", "età") and (PERSON.match(before[-2]) or before[-1].lower() == "età"):
                rows2.append((nid, m.start(1), m.end(1)))
rows2 = sorted(set(rows2))
path = ROOT / "data_secure/gold/annotation_review_additional.csv"
existing = [r for r in csv.DictReader(open(path, encoding="utf-8", newline="")) if r["issue"] != "age_like_context_review"]
for nid, a, b in rows2:
    t = gold[nid].text
    existing.append({"note_id": nid, "issue": "age_like_context_review", "type": "ETÀ", "value_in_note": t[a:b], "start": a, "context": t[max(0, a - 45):b + 30].replace("\n", " ")})
with open(path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["note_id", "issue", "type", "value_in_note", "start", "context"])
    w.writeheader(); w.writerows(existing)
out["age_like_context_review_rows"] = len(rows2)
(ROOT / "audit/results/additional_rows_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

# 7. Decision for every additional row. Rules use the issue, case and offsets; no note text is written.
from src.data.manifests import write_csv
rows3 = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
decisions = []
for i, r in enumerate(rows3, 1):
    issue, v, start, nid = r["issue"], r["value_in_note"], int(r["start"]), r["note_id"]
    typ, s, e = "", "", ""
    if issue == "string_annotated_in_another_note":
        act = "add_place_consistent_with_other_note" if v.isupper() else "reject_ordinary_word"
    elif issue == "place_name_not_annotated":
        act = "reject_ordinary_word" if v[0].islower() else "reject_country_of_study_never_annotated"
    elif issue == "age_spelled_with_context":
        act = "reject_duration"
    elif issue == "age_digit_enne":
        act = "add_age_other_person_number_only"
    elif issue == "age_word_enne":
        act = "reject_ordinary_word"
    elif issue == "capitalised_word_once_in_corpus":
        act = "locate_variant_mention" if (nid, start) == ("gold_007", 745) else "reject_medical_or_generic_term"
    elif (nid, start) == ("gold_058", 1539):
        act = "add_age_number_only"
    elif nid == "gold_004":
        act = "reject_population_threshold"
    elif (nid, start) == ("gold_019", 32):
        act = "reject_years_of_schooling"
    else:
        act = "reject_duration"
    if act == "add_place_consistent_with_other_note":
        typ, s, e = "LUOGO/INDIRIZZO", start, start + len(v)
    elif act == "add_age_other_person_number_only":
        typ, s, e = "ETÀ", spans["gold_037_age_number"][0], spans["gold_037_age_number"][1]
    elif act == "add_age_number_only":
        typ, s, e = "ETÀ", start, start + len(v)
    elif act == "locate_variant_mention":
        sp = next(x for x in out["surplus_listings"] if x["note_id"] == "gold_007")["uncovered_variant_spans"][0]
        typ, s, e = "LUOGO/INDIRIZZO", sp[0], sp[1]
    decisions.append({"row": i, "note_id": nid, "issue": issue, "action": act, "type": typ, "start": s, "end": e})
write_csv(ROOT / "data_manifests/gold_annotation_proposal_additional.csv", ["row", "note_id", "issue", "action", "type", "start", "end"], decisions)
tally2 = collections.Counter(d["action"] for d in decisions)
for d in decisions:
    if d["start"] != "" and d["action"] != "locate_variant_mention":
        assert not hit(gold[d["note_id"]], d["start"], d["end"]), d
out["additional_decisions"] = dict(tally2)
c = collections.Counter(e["type"] for x in gold.values() for e in x.entities)
out["counts_after_both_proposals"] = {"NOME": c["NOME"] - 2 + 10, "ETÀ": c["ETÀ"] - 1 + 2, "DATA": c["DATA"] + 12, "LUOGO/INDIRIZZO": c["LUOGO/INDIRIZZO"] + 2 + 1}
out["counts_after_both_proposals"]["total"] = sum(out["counts_after_both_proposals"].values())
(ROOT / "audit/results/additional_rows_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("ADDITIONAL", json.dumps({"decisions": out["additional_decisions"], "counts": out["counts_after_both_proposals"], "rows": len(decisions)}, ensure_ascii=False))
