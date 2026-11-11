"""Problem A: synthetic-crf-train vs gold with graded normalisation, near-duplicates, per-fold impact, PII status."""
import sys, pathlib, re, unicodedata, difflib, collections
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import *
from datasets import load_dataset
from huggingface_hub import HfApi

gold = load_gold(); F = folds(); FO = fold_of(F); gt = [g["text"] for g in gold]
crf = load_dataset("NLP-FBK/synthetic-crf-train", split="it")
notes = [r["clinical_note"] for r in crf]; ids = [r["document_id"] for r in crf]
LEVELS = [("byte_identical", lambda s: s), ("whitespace_trimmed", str.strip),
          ("whitespace_normalised", lambda s: " ".join(s.split())),
          ("unicode_case_whitespace", lambda s: " ".join(unicodedata.normalize("NFKC", s).casefold().split())),
          ("token_level_punct_insensitive", lambda s: " ".join(tokens(s)))]
out = {"crf_rows": len(notes), "unique_rows": len(set(notes))}
row_match = {}; cum = {}
for name, fn in LEVELS:
    idx = collections.defaultdict(list)
    for i, t in enumerate(gt): idx[fn(t)].append(i)
    hits = 0
    for r, n in enumerate(notes):
        if fn(n) in idx:
            hits += 1
            row_match.setdefault(r, {"crf_row": r, "document_id": ids[r], "gold_ids": idx[fn(n)], "first_match_level": name})
    cum[name] = hits
out["cumulative_rows_matched_by_level"] = cum
matched_gold = sorted({g for m in row_match.values() for g in m["gold_ids"]})
out["distinct_gold_matched"] = len(matched_gold)
out["gold_not_matched"] = sorted(set(range(80)) - set(matched_gold))
out["crf_rows_not_matched"] = [{"crf_row": r, "document_id": ids[r]} for r in range(len(notes)) if r not in row_match]
out["first_match_level_counts"] = dict(collections.Counter(m["first_match_level"] for m in row_match.values()))
out["gold_to_crf"] = sorted(({"gold_id": g, "crf_row": m["crf_row"], "document_id": m["document_id"], "level": m["first_match_level"]}
                             for m in row_match.values() for g in m["gold_ids"]), key=lambda x: x["gold_id"])

# near-duplicate stage for everything exact matching did not explain
gtok = [tokens(t) for t in gt]; ctok = [tokens(t) for t in notes]
def near(a, b):
    ga8, gb8 = grams(a, 8), grams(b, 8); ga5, gb5 = grams(a, 5), grams(b, 5)
    return {"shared_8grams": len(ga8 & gb8), "containment_8": round(len(ga8 & gb8) / max(1, min(len(ga8), len(gb8))), 3),
            "shared_5grams": len(ga5 & gb5), "containment_5": round(len(ga5 & gb5) / max(1, min(len(ga5), len(gb5))), 3)}
nr = []
for row in out["crf_rows_not_matched"]:
    r = row["crf_row"]; cand = []
    for g in range(80):
        n = near(ctok[r], gtok[g]); cand.append((n["shared_5grams"], g, n))
    cand.sort(key=lambda x: -x[0]); s5, g, n = cand[0]
    ratio = difflib.SequenceMatcher(None, ctok[r], gtok[g], autojunk=False).ratio()
    ops = [o for o in difflib.SequenceMatcher(None, ctok[r], gtok[g], autojunk=False).get_opcodes() if o[0] != "equal"]
    nr.append({**row, "best_gold_id": g, "token_seq_ratio": round(ratio, 4), "n_edit_ops": len(ops), **n,
               "length_tokens": len(ctok[r]), "best_gold_test_fold": FO[g]})
out["unmatched_crf_rows_best_gold_candidate"] = nr
gn = []
for g in out["gold_not_matched"]:
    best = max((near(gtok[g], ctok[r]) | {"crf_row": r} for r in range(len(ctok))), key=lambda d: d["shared_5grams"])
    gn.append({"gold_id": g, "test_fold": FO[g], **best})
out["unmatched_gold_best_crf_candidate"] = gn

# per-fold impact (strict: exact token-level twin only)
out["per_fold_test_notes_in_training"] = [{"fold": k + 1, "in_training": sum(i in matched_gold for i in te), "of": len(te),
                                           "absent_ids": [i for i in te if i not in matched_gold]} for k, te in enumerate(F)]
# PII status
t_mentions = sum(len(g["entities"]) for g in gold)
out["pii"] = {"total_annotated_mentions": t_mentions,
              "mentions_in_matched_notes": sum(len(gold[g]["entities"]) for g in matched_gold),
              "matched_notes_with_pii": sum(bool(gold[g]["entities"]) for g in matched_gold),
              "matched_notes_without_annotated_pii": [g for g in matched_gold if not gold[g]["entities"]]}
AGE = re.compile(r"\b\d{1,3}\s*(?:anni|aa\.?|a\.)(?=\W|$)|\betà di\b|\ba{1,2}\.?\s*\d{1,3}\b", re.I)
MON = "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
DATE = re.compile(rf"\b\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}\b|\b(?:{MON})\s+(?:del\s+)?\d{{4}}\b|\b(?:19|20)\d{{2}}\b", re.I)
INIT = re.compile(r"\b[A-Z]\.\s?[A-Z]\.")
rem = []
for x in out["unmatched_crf_rows_best_gold_candidate"]:
    n = notes[x["crf_row"]]
    rem.append({"crf_row": x["crf_row"], "document_id": x["document_id"], "age_pattern_hits": len(AGE.findall(n)),
                "date_pattern_hits": len(DATE.findall(n)), "initials_pattern_hits": len(INIT.findall(n)),
                "near_duplicate_of_gold": x["best_gold_id"] if x["token_seq_ratio"] >= 0.9 else None})
out["rows_remaining_after_exact_exclusion"] = rem
out["all_rows_pattern_positive"] = {"age": sum(bool(AGE.search(n)) for n in notes), "date": sum(bool(DATE.search(n)) for n in notes), "of": len(notes)}
# provenance of the public dataset version
api = HfApi(); info = api.dataset_info("NLP-FBK/synthetic-crf-train")
out["hf_provenance"] = {"sha": info.sha, "last_modified": str(info.last_modified), "created": str(info.created_at),
                        "commits": [{"id": c.commit_id[:8], "date": str(c.created_at.date()), "title": c.title[:60]} for c in api.list_repo_commits("NLP-FBK/synthetic-crf-train", repo_type="dataset")]}
# crf vs synthetic
sy = load_synth(); idx8 = collections.defaultdict(set)
for r, t in enumerate(ctok):
    for g in grams(t, 8): idx8[g].add(r)
out["crf_vs_synth_notes_sharing_8gram"] = {k: sum(any(g in idx8 for g in grams(tokens(x["text"]), 8)) for x in v) for k, v in sy.items()}
save("v2_crf_overlap.json", out)
for k in ["crf_rows", "cumulative_rows_matched_by_level", "distinct_gold_matched", "gold_not_matched", "first_match_level_counts", "per_fold_test_notes_in_training", "pii", "unmatched_crf_rows_best_gold_candidate", "unmatched_gold_best_crf_candidate", "rows_remaining_after_exact_exclusion", "all_rows_pattern_positive", "crf_vs_synth_notes_sharing_8gram"]:
    print(k, "=", out[k])
print("hf", out["hf_provenance"]["sha"][:8], out["hf_provenance"]["last_modified"], [c["date"] + ":" + c["title"] for c in out["hf_provenance"]["commits"]])
