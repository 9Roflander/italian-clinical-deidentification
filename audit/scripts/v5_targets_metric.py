"""New checks: are gold training targets correct, what does the metric reward, how complete are annotations."""
import sys, pathlib, re, collections
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import *

gold = load_gold(); F = folds(); out = {}
SRC = OW / "src"
ns = load_funcs(SRC / "train_evaluate_cv_v4.py", ["sanitize_type", "redact_gold", "paper_metric"], ["VALID_TYPES"])
redact_gold, paper_metric = ns["redact_gold"], ns["paper_metric"]
out["identical_function_copies"] = {
    fn: {f: func_dump(SRC / f, fn) == func_dump(SRC / "train_evaluate_cv_v4.py", fn)
         for f in ["train_evaluate_cv_v3.py", "train_evaluate_cv_gemma.py", "train_evaluate_cv_bert.py", "train_redact_v3.py"]}
    for fn in ["redact_gold", "paper_metric"]}

# A. gold training targets built by redact_gold (.replace = substring replacement)
sub_total = bnd_total = 0; bad_notes = set(); over_notes = set(); per = collections.Counter(); ph_over = collections.Counter()
for i, g in enumerate(gold):
    seen, uniq = set(), []
    for e in g["entities"]:
        t, x = stype(e.get("type")), e.get("text", "")
        if t and x and (t, x) not in seen: seen.add((t, x)); uniq.append((t, x))
    uniq.sort(key=lambda z: -len(z[1]))
    cur = g["text"]
    for t, x in uniq:
        s = cur.count(x); b = len(boundary(x).findall(cur))
        sub_total += s; bnd_total += b
        if s > b: bad_notes.add(i); per[t] += s - b
        cur = cur.replace(x, f"[{t}]")
    target = redact_gold(g["text"], g["entities"])
    n = collections.Counter(stype(e.get("type")) for e in g["entities"])
    for t in V:
        d = target.count(f"[{t}]") - n[t]
        if d > 0: ph_over[t] += d; over_notes.add(i)
out["gold_targets"] = {"substring_replacements": sub_total, "replacements_inside_a_longer_token": sub_total - bnd_total,
    "by_type": dict(per), "notes_with_corrupted_tokens": sorted(bad_notes), "n_notes_with_corrupted_tokens": len(bad_notes),
    "excess_placeholders_vs_annotations_by_type": dict(ph_over), "n_notes_with_excess_placeholders": len(over_notes)}

# B. annotations vs text
ann = {t: collections.Counter() for t in V}; form = collections.Counter(); both_types = 0; nested = 0
not_in_text = []; lower = 0
for i, g in enumerate(gold):
    per_str = collections.Counter((stype(e.get("type")), e.get("text", "")) for e in g["entities"])
    strs = {x for _, x in per_str}; by_str = collections.defaultdict(set)
    for (t, x) in per_str: by_str[x].add(t)
    both_types += sum(len(v) > 1 for v in by_str.values())
    nested += sum(1 for a in strs for b in strs if a != b and a in b)
    for (t, x), a in per_str.items():
        if not t or not x: continue
        s = g["text"].count(x); b = len(boundary(x).findall(g["text"]))
        c = ann[t]; c["unique_strings"] += 1; c["annotated_mentions"] += a
        if s == 0: c["string_not_in_note"] += 1; not_in_text.append(i)
        elif b == 0: c["only_inside_longer_token"] += 1
        elif b == a: c["occurrences_equal_annotations"] += 1
        elif b > a: c["more_occurrences_than_annotations"] += 1
        else: c["fewer_occurrences_than_annotations"] += 1
    for e in g["entities"]:
        t, x = stype(e.get("type")), e.get("text", "")
        if t == "ETÀ":
            form["age_digits_only" if x.strip().isdigit() else ("age_with_anni" if "anni" in x.lower() else "age_other")] += 1
        if t == "NOME":
            form["name_initials" if re.fullmatch(r"(?:[A-Za-z]\.\s?)+", x) else ("name_all_lower" if x.islower() else "name_other")] += 1
        if t == "DATA":
            form["date_year_only" if re.fullmatch(r"\d{4}", x) else "date_other"] += 1
out["annotations"] = {"has_character_offsets": any("start" in e or "offset" in e for g in gold for e in g["entities"]),
    "record_keys": sorted(gold[0].keys()), "has_document_ids": "id" in gold[0] or "document_id" in gold[0],
    "by_type": {t: dict(c) for t, c in ann.items()}, "forms": dict(form), "strings_with_two_types": both_types,
    "strings_nested_in_another": nested, "notes_with_annotation_absent_from_text": sorted(set(not_in_text)),
    "notes_without_any_annotation": [i for i, g in enumerate(gold) if not g["entities"]]}

# C. pattern hits left in the gold target: possible under-annotation (patterns can be non-PII, e.g. durations)
MON = "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
PATS = {"age_pattern": re.compile(r"\b\d{1,3}\s+anni\b", re.I),
        "month_year": re.compile(rf"\b(?:{MON})\s+(?:del\s+)?\d{{4}}\b", re.I),
        "numeric_date": re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"),
        "bare_year": re.compile(r"\b(?:19|20)\d{2}\b")}
out["pattern_hits_left_in_gold_targets"] = {k: {"hits": sum(len(p.findall(redact_gold(g["text"], g["entities"]))) for g in gold),
    "notes": sum(bool(p.search(redact_gold(g["text"], g["entities"]))) for g in gold)} for k, p in PATS.items()}
out["annotated_DATA_mentions"] = sum(stype(e.get("type")) == "DATA" for g in gold for e in g["entities"])

# D. what the metric gives to known outputs, aggregated the way the paper does (fold totals -> F1 -> mean)
def evaluate(fn):
    macros = []; pooled = {t: [0, 0, 0] for t in V}
    for te in F:
        tot = {t: [0, 0, 0] for t in V}
        for i in te:
            for t, (a, b, c) in paper_metric(fn(gold[i]), gold[i]["entities"]).items():
                for k, v in enumerate((a, b, c)): tot[t][k] += v; pooled[t][k] += v
        macros.append(sum(f1(*tot[t]) for t in V) / 4)
    return {"macro_f1_mean": round(float(np.mean(macros)), 4), "per_fold": [round(m, 3) for m in macros],
            "pooled_per_type_f1": {t: round(f1(*pooled[t]), 3) for t in V}, "pooled_counts_tp_fp_fn": pooled}
def uniq_sorted(g):
    seen, u = set(), []
    for e in g["entities"]:
        t, x = stype(e.get("type")), e.get("text", "")
        if t and x and (t, x) not in seen: seen.add((t, x)); u.append((t, x))
    return sorted(u, key=lambda z: -len(z[1]))
def oracle_boundary(g):
    cur = g["text"]
    for t, x in uniq_sorted(g): cur = boundary(x).sub(f"[{t}]", cur)
    return cur
def oracle_first_k(g):
    cur = g["text"]; n = collections.Counter((stype(e.get("type")), e.get("text", "")) for e in g["entities"])
    for t, x in uniq_sorted(g): cur = boundary(x).sub(f"[{t}]", cur, count=n[(t, x)])
    return cur
def wrong_tag(g):
    cur = g["text"]; nxt = dict(zip(V, V[1:] + V[:1]))
    for t, x in uniq_sorted(g): cur = cur.replace(x, f"[{nxt[t]}]")
    return cur
out["metric_on_known_outputs"] = {
    "historical_training_target": evaluate(lambda g: redact_gold(g["text"], g["entities"])),
    "oracle_word_boundary_all_occurrences": evaluate(oracle_boundary),
    "oracle_word_boundary_first_k_occurrences": evaluate(oracle_first_k),
    "empty_output": evaluate(lambda g: ""),
    "input_returned_unchanged": evaluate(lambda g: g["text"]),
    "every_entity_given_the_wrong_tag": evaluate(wrong_tag)}

# E. BERT label construction uses the same substring rule
b = load_funcs(SRC / "train_evaluate_cv_bert.py", ["sanitize_type", "entities_to_spans"], ["VALID_TYPES"])
inside = tot = 0; bt = collections.Counter(); nts = set()
for i, g in enumerate(gold):
    for s, e, t in b["entities_to_spans"](g["text"], g["entities"]):
        tot += 1; x = g["text"]
        if (s > 0 and (x[s - 1].isalnum() or x[s - 1] == "_")) or (e < len(x) and (x[e].isalnum() or x[e] == "_")):
            inside += 1; bt[t] += 1; nts.add(i)
out["bert_gold_spans"] = {"spans": tot, "spans_inside_a_longer_token": inside, "by_type": dict(bt), "notes": len(nts)}
# F. near-duplicates among gold notes (cross-fold leakage)
gg = [grams(tokens(g["text"]), 8) for g in gold]; FO = fold_of(F); pairs = []
for a in range(80):
    for c in range(a + 1, 80):
        n = len(gg[a] & gg[c])
        if n: pairs.append({"a": a, "b": c, "shared_8grams": n, "cross_fold": FO[a] != FO[c]})
out["gold_internal_8gram_pairs"] = sorted(pairs, key=lambda p: -p["shared_8grams"])[:10]
save("v5_targets_metric.json", out)
for k in ["identical_function_copies", "gold_targets", "annotations", "pattern_hits_left_in_gold_targets", "annotated_DATA_mentions", "bert_gold_spans", "gold_internal_8gram_pairs"]:
    print(k, "=", out[k])
for k, v in out["metric_on_known_outputs"].items(): print(k, v["macro_f1_mean"], v["pooled_per_type_f1"], v["per_fold"])
