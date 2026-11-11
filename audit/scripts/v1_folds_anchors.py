"""Problems B, C, F: fold reconstruction, anchors, gold access, hard-coded prompt strings, score recomputation."""
import ast, re, sys, pathlib, random, collections
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import *

gold = load_gold(); F = folds(); FO = fold_of(F); out = {}
ns = load_funcs(OW / "src/generate_synthetic_v2.py", ["load_style_anchors"], extra={"GOLD_PATH": str(GOLD)})
by_text = {g["text"]: i for i, g in enumerate(gold)}
anch = [by_text[a["text"]] for a in ns["load_style_anchors"](3)]
sidx = sorted(range(80), key=lambda i: len(gold[i]["text"]))
out["anchors"] = [{"gold_id": i, "sorted_position": sidx.index(i), "chars": len(gold[i]["text"]),
                   "n_annotated_mentions": len(gold[i]["entities"]), "test_fold": FO[i]} for i in anch]
out["folds"] = F

# fold reconstruction against every saved result file
def counts(ids):
    c = collections.Counter()
    for i in ids:
        for e in gold[i]["entities"]:
            if stype(e.get("type")): c[stype(e["type"])] += 1
    return c
res = load_results(); chk = {}; agg = {}
for k, r in res.items():
    ok = all(fr["scores"][t]["tp"] + fr["scores"][t]["fn"] == counts(te)[t]
             for fr, te in zip(r["folds"], F) for t in V)
    chk[k] = bool(ok)
    per = [sum(f1(fr["scores"][t]["tp"], fr["scores"][t]["fp"], fr["scores"][t]["fn"]) for t in V) / 4 for fr in r["folds"]]
    agg[k] = {"fold_macro_f1": [round(x, 4) for x in per], "mean": round(float(np.mean(per)), 4),
              "std_ddof0": round(float(np.std(per)), 4), "std_ddof1": round(float(np.std(per, ddof=1)), 4),
              "saved_keys_per_fold": sorted(r["folds"][0].keys()), "has_note_level_output": False,
              "config": r.get("config", {})}
out["fold_reconstruction_matches_results"] = chk
out["score_recomputation"] = agg
out["fold_entity_counts"] = {f"fold_{k+1}": dict(counts(te)) for k, te in enumerate(F)}

# historical single split
ids = list(range(80)); random.Random(42).shuffle(ids)
single = sorted(ids[:16])
saved = json.loads((OW / "data/test_indices_seed42.json").read_text())
out["single_split"] = {"test_ids": single, "matches_saved_file": saved["test"] == single,
                       "anchors_in_test": [i for i in anch if i in single]}

# which functions read the gold file, and who calls the anchor selector
gold_access = []
for p in sorted((OW / "src").glob("*.py")) + [OW / "thesis/make_figures.py"]:
    src = p.read_text(encoding="utf-8"); tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if names & {"GOLD_PATH"} :
                gold_access.append({"file": str(p.relative_to(OW)), "function": node.name, "line": node.lineno})
            if "load_style_anchors" in names:
                gold_access.append({"file": str(p.relative_to(OW)), "function": node.name, "line": node.lineno, "calls": "load_style_anchors"})
out["gold_access_points"] = gold_access

# hard-coded quoted strings inside the prompt builders, matched against gold (hashes only)
rows = []
for rel, fn in [("src/generate_synthetic_v2.py", "build_prompt"), ("src/generate_noname.py", "build_noname_prompt")]:
    src = (OW / rel).read_text(encoding="utf-8"); tree = ast.parse(src); lines = src.splitlines()
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
    for ln in range(node.lineno, node.end_lineno + 1):
        for m in re.finditer(r'"([^"\n{}]{2,60})"', lines[ln - 1]):
            s = m.group(1)
            if s in ("text", "type", "cases", "redacted_text", "entities") or s.startswith("..."): continue
            pat = boundary(s)
            in_text = [i for i, g in enumerate(gold) if pat.search(g["text"])]
            exact = [i for i, g in enumerate(gold) if any(e.get("text") == s for e in g["entities"])]
            parts = [w for w in re.findall(r"\w+", s) if w[0].isupper() and len(w) >= 4]
            comp = sorted({i for w in parts for i, g in enumerate(gold) if boundary(w).search(g["text"])})
            if in_text or exact:
                rows.append({"file": rel, "line": ln, "string_sha8": sha8(s), "gold_ids_in_text": in_text,
                             "gold_ids_exact_entity": exact, "test_folds": sorted({FO[i] for i in in_text})})
out["hard_coded_prompt_strings_found_in_gold"] = rows
save("v1_folds_anchors.json", out)
print("anchors:", [(a["gold_id"], a["test_fold"]) for a in out["anchors"]], "| reconstruction:", chk)
print("gold-matching prompt strings:", [(r["file"].split('/')[-1], r["line"], r["gold_ids_in_text"], r["gold_ids_exact_entity"]) for r in rows])
print("scores:", {k: (v["mean"], v["std_ddof0"], v["std_ddof1"]) for k, v in agg.items()})
print("gold access:", [(g["file"].split('/')[-1], g["function"], g["line"]) for g in gold_access])
