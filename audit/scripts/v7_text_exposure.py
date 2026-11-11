"""Does real note text appear outside data/? Reports file paths, gold IDs and counts only."""
import sys, pathlib, subprocess, collections
sys.path.insert(0, str(pathlib.Path(__file__).parent))
if len(sys.argv) > 1: sys.path.insert(0, sys.argv[1])   # optional path to a pypdf install
from _common import *

gold = load_gold(); gt = [tokens(g["text"]) for g in gold]
gset = [grams(t, 8) for t in gt]; index = collections.defaultdict(set)
for i, g in enumerate(gset):
    for x in g: index[x].add(i)
TEXT = {".py", ".md", ".tex", ".html", ".css", ".json", ".txt", ".log", ".bib", ".csv"}
def read(p):
    if p.suffix == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
        except Exception as e:
            return None
    if p.suffix in TEXT: return p.read_text(encoding="utf-8", errors="ignore")
    return ""
tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines())
rows = []; skipped = []
for p in sorted(OW.rglob("*")):
    if not p.is_file() or "/data/" in str(p): continue
    t = read(p)
    if t is None: skipped.append(str(p.relative_to(ROOT))); continue
    if not t: continue
    hits = collections.Counter()
    for g in grams(tokens(t), 8):
        for i in index.get(g, ()): hits[i] += 1
    if hits:
        rel = str(p.relative_to(ROOT))
        rows.append({"file": rel, "in_git_index": rel in tracked, "gold_ids": sorted(hits), "shared_8grams": sum(hits.values()),
                     "max_share_of_one_note": round(max(hits[i] / max(1, len(gset[i])) for i in hits), 3)})
rows.sort(key=lambda r: -r["shared_8grams"])
out = {"files_with_gold_text": rows, "unreadable_pdfs": skipped,
       "gold_in_public_git": {"data/gold_standard_80.json": "tracked at HEAD on main"}}
save("v7_text_exposure.json", out)
for r in rows: print(r)
print("skipped", skipped)
