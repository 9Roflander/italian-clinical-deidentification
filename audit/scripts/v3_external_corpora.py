"""Problem D: Phase 1 corpora vs gold (dyspnea checked; DART access recorded, not assumed)."""
import sys, pathlib, collections
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import *
from datasets import load_dataset
from huggingface_hub import HfApi

gold = load_gold(); out = {}
gt = [tokens(g["text"]) for g in gold]; ghash = {doc_hash(g["text"]) for g in gold}
ds = load_dataset("NLP-FBK/dyspnea-clinical-notes", split="it")
notes = [r["clinical_note"] for r in ds]; dt = [tokens(n) for n in notes]
res = {"rows": len(notes), "exact_normalised_hash_matches_to_gold": sum(doc_hash(n) in ghash for n in notes)}
for n in (12, 10, 8, 6, 5):
    idx = set()
    for t in dt: idx |= grams(t, n)
    per = [len(grams(t, n) & idx) for t in gt]
    res[f"gold_notes_sharing_{n}gram"] = sum(bool(x) for x in per)
    res[f"max_shared_{n}grams_any_gold_note"] = max(per)
res["note"] = "Word n-gram overlap only. Short shared n-grams can be generic clinical wording; no text is reported."
api = HfApi(); info = api.dataset_info("NLP-FBK/dyspnea-clinical-notes")
res["hf_sha"] = info.sha; res["hf_last_modified"] = str(info.last_modified)
out["dyspnea"] = res
try:
    d = api.dataset_info("praiselab-picuslab/DART")
    out["dart"] = {"gated": str(d.gated), "sha": d.sha, "last_modified": str(d.last_modified), "overlap_check": "NOT PERFORMED"}
    try:
        load_dataset("praiselab-picuslab/DART", split="DART"); out["dart"]["data_access"] = "available"
    except Exception as e:
        out["dart"]["data_access"] = "denied: " + type(e).__name__
except Exception as e:
    out["dart"] = {"metadata_error": type(e).__name__, "overlap_check": "NOT PERFORMED"}
save("v3_external_corpora.json", out); print(out)
