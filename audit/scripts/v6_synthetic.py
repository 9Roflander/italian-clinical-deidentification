"""Problem C and generator claims: S1/S2 consistency, alignment of text vs redacted_text, leakage indicators."""
import sys, pathlib, re, copy, collections, json
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import *
from sklearn.feature_extraction.text import TfidfVectorizer

gold = load_gold(); F = folds(); FO = fold_of(F); sy = load_synth(); out = {}
v1 = json.loads((RES / "v1_folds_anchors.json").read_text()); anch = [a["gold_id"] for a in v1["anchors"]]
G = load_funcs(OW / "src/generate_synthetic_v2.py", ["sanitize_type", "validate_case"], ["VALID_TYPES", "SPECIALTIES", "GENDERS", "AGE_RANGES", "DENSITIES"])
PH = re.compile(r"\[(NOME|ETÀ|DATA|LUOGO/INDIRIZZO)\]")

def align(text, red):
    """Greedy literal alignment of redacted_text onto text. Returns (status, spans)."""
    parts = PH.split(red)
    if not text.startswith(parts[0]): return "prefix_mismatch", []
    pos = len(parts[0]); spans = []
    for k in range(1, len(parts), 2):
        t, lit = parts[k], parts[k + 1]
        last = k + 2 >= len(parts)
        if last:
            if lit and not text.endswith(lit): return "suffix_mismatch", spans
            end = len(text) - len(lit) if lit else len(text)
            if end < pos: return "suffix_mismatch", spans
            spans.append((pos, end, t)); pos = len(text)
        else:
            if not lit: return "adjacent_placeholders", spans
            j = text.find(lit, pos)
            if j < 0: return "segment_not_found", spans
            spans.append((pos, j, t)); pos = j + len(lit)
    if len(parts) == 1 and text != red: return "no_placeholder_but_text_differs", []
    return "consistent", spans

MON = "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
RES_P = {"age_pattern": re.compile(r"\b\d{1,3}\s+anni\b", re.I), "month_year": re.compile(rf"\b(?:{MON})\s+(?:del\s+)?\d{{4}}\b", re.I),
         "numeric_date": re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"), "bare_year": re.compile(r"\b(?:19|20)\d{2}\b"),
         "title_plus_cap": re.compile(r"\b(?:Sig\.?|Sig\.ra|Signor|Signora|Dott\.?|Dott\.ssa|Dr\.?|Prof\.?)\s+[A-Z]")}
files = {}
for k, recs in sy.items():
    st = collections.Counter(); ent = collections.Counter(); span_chk = collections.Counter(); ph = collections.Counter(); res_hits = collections.Counter(); res_notes = collections.Counter()
    lens = sorted(len(r["text"]) for r in recs); valid = 0; miss_type = collections.Counter(); len_mismatch = 0; long_span = 0; empty_span = 0
    for r in recs:
        if G["validate_case"](copy.deepcopy(r))[0]: valid += 1
        for e in r["entities"]: ent[e["type"]] += 1
        for t in V: ph[t] += r["redacted_text"].count(f"[{t}]")
        s, spans = align(r["text"], r["redacted_text"]); st[s] += 1
        if s == "consistent":
            listed = {(e["type"], e["text"]) for e in r["entities"]}
            for a, b, t in spans:
                x = r["text"][a:b]
                if not x: empty_span += 1; continue
                if len(x) > 60: long_span += 1
                span_chk["span_exact_in_entities" if (t, x) in listed else ("span_listed_under_other_type" if any(x == y for _, y in listed) else "span_not_listed")] += 1
            if len(spans) != len(r["entities"]): len_mismatch += 1
        for n, p in RES_P.items():
            h = len(p.findall(r["redacted_text"])); res_hits[n] += h; res_notes[n] += bool(h)
    files[k] = {"records": len(recs), "pass_historical_validator": valid, "chars_min_median_max": [lens[0], lens[len(lens) // 2], lens[-1]],
                "entity_list_counts": dict(ent), "placeholder_counts_in_redacted": dict(ph), "notes_with_any_NOME_placeholder": sum("[NOME]" in r["redacted_text"] for r in recs),
                "alignment_status": dict(st), "recovered_span_check": dict(span_chk), "empty_spans": empty_span, "spans_over_60_chars": long_span,
                "placeholder_count_differs_from_entity_list": len_mismatch, "pattern_hits_left_in_redacted": dict(res_hits), "notes_with_pattern_left": dict(res_notes)}
out["files"] = files
out["totals"] = {"records": sum(len(v) for v in sy.values()), "S1": len(sy["S1_named"]) + len(sy["S1_noname"]), "S2": len(sy["S2_named"]) + len(sy["S2_noname"])}
# duplicates
hashes = collections.Counter(doc_hash(r["text"]) for v in sy.values() for r in v); ghash = {doc_hash(g["text"]) for g in gold}
out["exact_duplicates_normalised"] = {"within_synthetic": sum(c - 1 for c in hashes.values() if c > 1), "synthetic_equal_to_gold": sum(h in ghash for h in hashes)}
# overlap with gold
gt = [tokens(g["text"]) for g in gold]; ov = {}
for n in (6, 8, 10, 12):
    idx = collections.defaultdict(set)
    for i, t in enumerate(gt):
        for g in grams(t, n): idx[g].add(i)
    tot = collections.Counter(); per = {}
    for k, recs in sy.items():
        c = collections.Counter(); anyh = anch_h = 0
        for r in recs:
            hit = set()
            for g in grams(tokens(r["text"]), n): hit |= idx.get(g, set())
            for h in hit: c[h] += 1
            anyh += bool(hit); anch_h += bool(hit & set(anch))
        tot.update(c); per[k] = {"notes_sharing_with_any_gold": anyh, "notes_sharing_with_anchor": anch_h}
    ov[f"{n}gram"] = {"per_file": per, "top_gold_ids": tot.most_common(6), "anchor_counts": {i: tot[i] for i in anch}}
out["gold_overlap"] = ov
def longest(a, b):
    pos = collections.defaultdict(list)
    for j, w in enumerate(b): pos[w].append(j)
    prev = {}; best = 0
    for w in a:
        cur = {}
        for j in pos.get(w, ()):
            l = prev.get(j - 1, 0) + 1; cur[j] = l; best = max(best, l)
        prev = cur
    return best
idx6 = collections.defaultdict(set)
for i, t in enumerate(gt):
    for g in grams(t, 6): idx6[g].add(i)
mx = 0; hist = collections.Counter(); mx_a = 0
for recs in sy.values():
    for r in recs:
        t = tokens(r["text"]); hit = set()
        for g in grams(t, 6): hit |= idx6.get(g, set())
        for h in hit:
            L = longest(t, gt[h]); mx = max(mx, L)
            if h in anch: mx_a = max(mx_a, L)
out["longest_verbatim_run_tokens"] = {"any_gold": mx, "anchors": mx_a}
# anchor and prompt-derived strings in synthetic (string hashes only)
ap = []
for i in anch:
    for t, x in sorted({(stype(e["type"]), e["text"]) for e in gold[i]["entities"]}):
        p = boundary(x)
        ap.append({"anchor": i, "type": t, "string_sha8": sha8(x), "chars": len(x), **{k: sum(bool(p.search(r["text"])) for r in v) for k, v in sy.items()}})
out["anchor_entity_strings_in_synthetic"] = ap
fp = []
for row in v1["hard_coded_prompt_strings_found_in_gold"]:
    if len(row["gold_ids_in_text"]) > 3: continue
    src = (OW / row["file"]).read_text(encoding="utf-8").splitlines()[row["line"] - 1]
    for m in re.finditer(r'"([^"\n{}]{2,60})"', src):
        if sha8(m.group(1)) == row["string_sha8"]:
            p = boundary(m.group(1))
            fp.append({"file": row["file"], "line": row["line"], "string_sha8": row["string_sha8"], "gold_ids": row["gold_ids_in_text"],
                       "pct_of_notes": {k: round(100 * sum(bool(p.search(r["text"])) for r in v) / len(v), 1) for k, v in sy.items()}})
out["prompt_string_fingerprints"] = fp
# TF-IDF closeness of synthetic to anchors vs other gold
gs = [g["text"] for g in gold]; tf = {}
for k, recs in sy.items():
    ts = [r["text"] for r in recs]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(gs + ts)
    sim = (vec.transform(ts) @ vec.transform(gs).T).toarray(); mean = sim.mean(0); order = list(np.argsort(-mean))
    tf[k] = {"anchor_rank_of_80": {i: order.index(i) + 1 for i in anch}, "max_note_similarity": round(float(sim.max()), 3), "median_gold_mean_similarity": round(float(np.median(mean)), 3)}
out["tfidf_similarity"] = tf
# prompt parameter rotation actually possible
S, GE, A, D = G["SPECIALTIES"], G["GENDERS"], G["AGE_RANGES"], G["DENSITIES"]
combos = {(i % 10, i % 2, i % 3, i % 3) for i in range(3000)}
out["named_prompt_rotation"] = {"distinct_combinations": len(combos), "of_full_factorial": 10 * 2 * 3 * 3,
    "specialty_index_to_gender_index": sorted({(i % 10, i % 2) for i in range(100)}), "age_density_locked": True,
    "specialty_gender_pairs": {S[i % 10]: GE[i % 2] for i in range(10)}}
out["acceptance_upper_bound_from_defaults"] = {"S1_named": {"requested_max": ((1000 + 2) // 3 + 10) * 3, "kept": 624}, "S1_noname": {"requested_max": ((400 + 2) // 3 + 30) * 3, "kept": 254},
    "note": "Assumes the documented --n values. Rejection logs are absent, so reported 77% and 71% cannot be reproduced."}
# gold through the same validator
out["gold_through_validator"] = {"pass": sum(G["validate_case"](copy.deepcopy({"text": g["text"], "redacted_text": load_funcs(OW / "src/train_evaluate_cv_v4.py", ["sanitize_type", "redact_gold"], ["VALID_TYPES"])["redact_gold"](g["text"], g["entities"]), "entities": g["entities"]}))[0] for g in gold), "of": 80}
save("v6_synthetic.json", out)
for k, v in files.items(): print(k, v)
for k in ["totals", "exact_duplicates_normalised", "longest_verbatim_run_tokens", "tfidf_similarity", "prompt_string_fingerprints", "gold_through_validator", "acceptance_upper_bound_from_defaults"]: print(k, "=", out[k])
print("overlap", {k: v["per_file"] for k, v in ov.items() if k in ("8gram", "10gram", "12gram")}); print("anchor strings", [(a["anchor"], a["type"], a["chars"], a["S1_named"], a["S2_named"], a["S1_noname"], a["S2_noname"]) for a in ap])
