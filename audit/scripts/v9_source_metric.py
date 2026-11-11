"""Compares the historical paper_metric with Miranda et al.'s real evaluate_deidentification on known outputs.

    python audit/scripts/v9_source_metric.py <path to their eval.py> [v1|v2]

Only their function is executed, by AST extraction. Output: scores only."""
import ast, collections, json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from _common import OW, RES, V, f1, load_funcs, save
from src.data.loaders import GOLD_V1, GOLD_V2, load_gold
from src.data.splits import load_fold

tree = ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_deidentification"]
ns = {"re": re, "defaultdict": collections.defaultdict, "Dict": dict, "List": list, "Tuple": tuple}
exec(compile(ast.Module(keep, []), "their_eval", "exec"), ns)
theirs = lambda text, out, ents: ns["evaluate_deidentification"](text, out, [{"type": e["type"], "text": e["annotated_text"]} for e in ents])
pm = load_funcs(OW / "src/train_evaluate_cv_v4.py", ["sanitize_type", "paper_metric"], ["VALID_TYPES"])["paper_metric"]
ours = lambda text, out, ents: pm(out, [{"type": e["type"], "text": e["annotated_text"]} for e in ents])
version = sys.argv[2] if len(sys.argv) > 2 else "v1"
gold = {n.note_id: n for n in load_gold(GOLD_V2 if version == "v2" else GOLD_V1)}
folds = [load_fold(k) for k in range(1, 6)]

def score(metric, fn):
    macros = []; pooled = {t: [0, 0, 0] for t in V}
    for s in folds:
        tot = {t: [0, 0, 0] for t in V}
        for i in s.test_ids:
            n = gold[i]
            for t, vals in metric(n.text, fn(n), n.entities).items():
                for k, v in enumerate(vals):
                    tot[t][k] += v; pooled[t][k] += v
        macros.append(sum(f1(*tot[t]) for t in V) / 4)
    return {"macro_f1_mean": round(sum(macros) / 5, 4), "per_type": {t: round(f1(*pooled[t]), 3) for t in V}}

def wrong_tag(n):
    nxt = dict(zip(V, V[1:] + V[:1]))
    return re.sub(r"\[(NOME|ETÀ|DATA|LUOGO/INDIRIZZO)\]", lambda m: f"[{nxt[m.group(1)]}]", n.redacted_text)

outputs = {"correct_redaction_gold_v1_target": lambda n: n.redacted_text, "empty_output": lambda n: "", "input_unchanged": lambda n: n.text, "every_entity_wrong_tag": wrong_tag}
out = {k: {"paper_metric_as_used": score(ours, fn), "miranda_eval_py": score(theirs, fn)} for k, fn in outputs.items()}
# their worked example: two annotated mentions of one string, found once in the output
ex = [{"type": "NOME", "text": "Zelinda", "annotated_text": "Zelinda"}] * 2
out["two_mentions_one_left"] = {"paper_metric_as_used": ours("", "Zelinda [NOME]", ex)["NOME"], "miranda_eval_py": theirs("", "Zelinda [NOME]", ex)["NOME"], "format": "(tp, fp, fn)"}
ex = [{"type": "ETÀ", "text": "8", "annotated_text": "8"}]
out["short_age_digit_three_elsewhere"] = {"paper_metric_as_used": ours("", "[ETÀ] anni, 18 mg, 28 mg, 80 mg", ex)["ETÀ"], "miranda_eval_py": theirs("", "[ETÀ] anni, 18 mg, 28 mg, 80 mg", ex)["ETÀ"], "format": "(tp, fp, fn)"}
save("v9_source_metric.json" if version == "v1" else "v9_source_metric_v2.json", out)
print(json.dumps(out, ensure_ascii=False, indent=0)[:1800])
