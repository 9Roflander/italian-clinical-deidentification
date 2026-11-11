"""How much uncertainty can 82 notes support? A planning simulation, not a result.

Uses per-note mention COUNTS from gold_v2 only, and hypothetical outcomes. It reports the expected 95% interval
half-width for recall, and the smallest paired recall difference whose interval excludes zero in 80% of simulations,
using the protocol's fold-stratified bootstrap. Assumed: recall 0.80 for the weaker model, 20% of outcome variance
at note level, and 40% shared between the two models at mention level. Other assumptions give other numbers.
"""
import json
import pathlib
import sys
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.evaluation.alignment import CATEGORIES
from src.evaluation.gold_verification import verify_gold
from src.evaluation.runner import load_config
from src.evaluation.statistics import stratified_weights

NOTE_SHARE, SHARED_SHARE, BASE, SIMS, BOOT = 0.2, 0.4, 0.80, 400, 2000
DELTAS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18)
gold = verify_gold(load_config())
folds = [gold.fold_of[i] for i in gold.order]
counts = {c: np.array([sum(s[2] == c for s in gold.spans[i]) for i in gold.order]) for c in CATEGORIES}
counts["ALL"] = sum(counts[c] for c in CATEGORIES)
rng = np.random.default_rng(42)
W = stratified_weights(folds, BOOT, 42)
inv = NormalDist().inv_cdf


def simulate(m, r1, r2):
    n, idx = m.size, np.repeat(np.arange(m.size), m)
    base = np.sqrt(NOTE_SHARE) * rng.standard_normal(n)[idx] + np.sqrt(SHARED_SHARE) * rng.standard_normal(idx.size)
    own = np.sqrt(1 - NOTE_SHARE - SHARED_SHARE)
    x = base + own * rng.standard_normal(idx.size) < inv(r1)
    y = base + own * rng.standard_normal(idx.size) < inv(r2)
    return np.bincount(idx, x, n), np.bincount(idx, y, n)


def recall(hits, m):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (W @ hits) / (W @ m)


out = {"note": __doc__, "assumptions": {"base_recall": BASE, "note_variance_share": NOTE_SHARE, "shared_variance_share": SHARED_SHARE, "simulations": SIMS, "bootstrap": BOOT}}
for name, m in counts.items():
    widths = []
    for _ in range(SIMS):
        x, _y = simulate(m, BASE, BASE)
        low, high = np.nanquantile(recall(x, m), [0.025, 0.975])
        widths.append((high - low) / 2)
    smallest = None
    for d in DELTAS:
        hits = 0
        for _ in range(SIMS):
            x, y = simulate(m, BASE, BASE + d)
            low, high = np.nanquantile(recall(y, m) - recall(x, m), [0.025, 0.975])
            hits += low > 0
        if hits / SIMS >= 0.8:
            smallest = d
            break
    out[name] = {"mentions": int(m.sum()), "notes_with_mentions": int((m > 0).sum()), "largest_note": int(m.max()),
                 "effective_notes_kish": round(float(m.sum() ** 2 / (m ** 2).sum()), 1),
                 "ci95_halfwidth_recall": round(float(np.mean(widths)), 3),
                 "smallest_detectable_difference_80pct": smallest if smallest else f"above {DELTAS[-1]}"}
path = pathlib.Path(__file__).resolve().parents[1] / "results" / "v12_evaluation_power.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("note",)}, ensure_ascii=False, indent=1))
