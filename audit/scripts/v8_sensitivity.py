"""Direct-effect bounds for anchor notes, per-category CV history, and LoRA parameter count.
Bounds hold false positives constant, because FP cannot be attributed to notes without per-note output."""
import sys, pathlib, collections, json
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import *
from huggingface_hub import hf_hub_download

gold = load_gold(); F = folds(); res = load_results(); anch = {34, 13, 72}
def scen(r, excl, mode):
    ms = []
    for fr, te in zip(r["folds"], F):
        rm = collections.Counter(stype(e["type"]) for i in te if i in excl for e in gold[i]["entities"] if stype(e["type"]))
        fs = []
        for t in V:
            s = fr["scores"][t]; tp, fp, fn = s["tp"], s["fp"], s["fn"]; n = rm[t]
            if mode == "all_correct": tp = max(0, tp - n)
            elif mode == "all_missed": fn = max(0, fn - n)
            fs.append(f1(tp, fp, fn))
        ms.append(sum(fs) / 4)
    return round(float(np.mean(ms)), 4)
out = {"note": "Removing notes assuming their mentions were all correct gives the low end; assuming all missed gives the high end.", "bounds": {}}
for k, r in res.items():
    out["bounds"][k] = {"reported": scen(r, set(), "none"),
                        "without_anchors": [scen(r, anch, "all_correct"), scen(r, anch, "all_missed")],
                        "without_anchors_and_gold_0_1": [scen(r, anch | {0, 1}, "all_correct"), scen(r, anch | {0, 1}, "all_missed")]}
out["cv_nome_f1"] = {k: round(float(np.mean([f1(fr["scores"]["NOME"]["tp"], fr["scores"]["NOME"]["fp"], fr["scores"]["NOME"]["fn"]) for fr in res[k]["folds"]])), 3) for k in ["llama_v3", "llama_v4"]}
cfg = json.load(open(hf_hub_download("unsloth/Llama-3.2-3B", "config.json")))
d, ff, L, h, kv = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
head = d // h; r = 16
per_layer = sum(r * (a + b) for a, b in [(d, d), (d, kv * head), (d, kv * head), (d, d), (d, ff), (d, ff), (ff, d)])
base = 128256 * d + L * (d * d * 2 + 2 * d * kv * head + 3 * d * ff)
out["lora_llama_3_2_3b"] = {"hidden": d, "intermediate": ff, "layers": L, "trainable": per_layer * L, "base_params_approx": base, "pct": round(100 * per_layer * L / base, 2)}
save("v8_sensitivity.json", out)
print(json.dumps(out, indent=0)[:1500])
