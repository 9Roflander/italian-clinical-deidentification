"""Problem E: what the 700-token limit is, which mentions it silently credits, and training-side max_length."""
import sys, pathlib, collections, json
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import *
from transformers import AutoTokenizer

gold = load_gold(); F = folds(); res = load_results(); sy = load_synth(); out = {}
crf_rows = None
def corrected(r, free):
    base, rem, miss = [], [], []
    for fr, c in zip(r["folds"], free):
        b, x, y = [], [], []
        for t in V:
            s = fr["scores"][t]; n = c.get(t, 0)
            b.append(f1(s["tp"], s["fp"], s["fn"]))
            x.append(f1(max(0, s["tp"] - n), s["fp"], s["fn"]))
            y.append(f1(max(0, s["tp"] - n), s["fp"], s["fn"] + n))
        base.append(sum(b) / 4); rem.append(sum(x) / 4); miss.append(sum(y) / 4)
    return {"reported": round(float(np.mean(base)), 4), "excluding_unprocessed": round(float(np.mean(rem)), 4),
            "counting_unprocessed_as_missed": round(float(np.mean(miss)), 4),
            "per_fold_counting_as_missed": [round(v, 3) for v in miss]}
def per_model(tk, cut_fn):
    free, longn, lens, ids_long = [], 0, [], []
    for te in F:
        c = collections.Counter()
        for i in te:
            visible, n_tok, was_cut = cut_fn(tk, gold[i]["text"])
            lens.append(n_tok)
            if was_cut: longn += 1; ids_long.append(i)
            for e in gold[i]["entities"]:
                t = stype(e.get("type")); x = e.get("text", "")
                if t and x and x not in visible: c[t] += 1
        free.append(c)
    tot = sum(free, collections.Counter())
    return free, {"notes_cut": longn, "cut_note_ids": sorted(ids_long), "tokens_median": sorted(lens)[40], "tokens_max": max(lens),
                  "mentions_credited_without_being_processed": sum(tot.values()), "by_type": dict(tot),
                  "per_fold": [sum(c.values()) for c in free]}
def cut700(tk, text):
    ids = tk.encode(text, add_special_tokens=False)
    return (text if len(ids) <= 700 else tk.decode(ids[:700])), len(ids), len(ids) > 700
def cut512(tk, text):
    enc = tk(text, truncation=True, max_length=512, return_offsets_mapping=True)
    full = len(tk(text, add_special_tokens=False)["input_ids"])
    end = max(e for _, e in enc["offset_mapping"])
    return text[:end], full, full > 510
LL = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B"); GE = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it")
BE = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
for key, tk, fn, rk, limit in [("llama_v4", LL, cut700, "llama_v4", 700), ("llama_v3", LL, cut700, "llama_v3", 700),
                               ("gemma_1b", GE, cut700, "gemma_1b", 700), ("bert_truncated_old_code", BE, cut512, "bert_truncated", 512)]:
    free, s = per_model(tk, fn); s["limit_tokens"] = limit; s["score_effect"] = corrected(res[rk], free); out[key] = s
out["tokenizer_note"] = "Counted with public mirrors unsloth/Llama-3.2-3B and unsloth/gemma-3-1b-it, not the original checkpoints; bert uses dbmdz/bert-base-italian-xxl-cased."
# training side: example length vs max_length=1536
ns = load_funcs(OW / "src/train_evaluate_cv_v4.py", ["sanitize_type", "redact_gold"], ["VALID_TYPES", "SYSTEM_PROMPT"])
sys_len = len(LL.encode(ns["SYSTEM_PROMPT"], add_special_tokens=False)); OVER = 12  # role headers/eot, approximate
def over(rows):
    n = cut = 0; lost = []
    for a, b in rows:
        la, lb = len(LL.encode(a, add_special_tokens=False)), len(LL.encode(b, add_special_tokens=False))
        n += 1; tot = sys_len + OVER + la + lb
        if tot > 1536: cut += 1; lost.append(min(1.0, (tot - 1536) / max(1, lb)))
    return {"examples": n, "over_1536": cut, "pct": round(100 * cut / n, 1), "median_share_of_target_lost": round(sorted(lost)[len(lost) // 2], 3) if lost else 0}
tr = {"gold": over([(g["text"], ns["redact_gold"](g["text"], g["entities"])) for g in gold])}
for k, v in sy.items(): tr[k] = over([(r["text"], r["redacted_text"]) for r in v])
out["training_examples_over_max_length"] = tr
out["training_note"] = "Token sum is approximate (no chat-template date header). Whether truncation cuts the target depends on the unrecorded trl version."
# Prompt-independent sizing for full-document inference
out["full_note_generation_tokens_needed_llama"] = max(len(LL.encode(g["text"], add_special_tokens=False)) for g in gold)
save("v4_truncation.json", out)
for k in ["llama_v4", "llama_v3", "gemma_1b", "bert_truncated_old_code"]:
    print(k, {x: out[k][x] for x in out[k] if x != "cut_note_ids"})
print(tr)
