"""Writes a random sample of accepted notes for a person to read, as the guide requires.

    python -m src.generation.sample_review --fold 1 --arm style_anchored --n 30

The sample holds generated text, so it goes to the git-ignored data_secure/review/ folder.
Each record type is represented in proportion to its share, with at least one note each.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random

from src.data.loaders import ROOT
from src.data.manifests import write_csv


def write_sample(fold: int, arm: str, n: int = 30, output_root=None, out_dir=None, seed: int = 42) -> pathlib.Path:
    root = pathlib.Path(output_root) if output_root else ROOT / "synthetic_data"
    path = root / f"fold_{fold}" / arm / "examples.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    groups = collections.defaultdict(list)
    for r in sorted(rows, key=lambda x: x["record_id"]):
        groups[r["record_type"]].append(r)
    rng = random.Random(f"{seed}:{fold}:{arm}:review")
    chosen = []
    for kind in sorted(groups):
        take = min(len(groups[kind]), max(1, round(n * len(groups[kind]) / len(rows))))
        chosen += rng.sample(groups[kind], take)
    out = pathlib.Path(out_dir) if out_dir else ROOT / "data_secure" / "review"
    target = out / f"fold_{fold}_{arm}_sample.csv"
    write_csv(target, ["record_id", "record_type", "text", "redacted_text", "entities", "reviewer_ok", "reviewer_notes"],
              [{"record_id": r["record_id"], "record_type": r["record_type"], "text": r["text"], "redacted_text": r["redacted_text"],
                "entities": json.dumps(r["entities"], ensure_ascii=False), "reviewer_ok": "", "reviewer_notes": ""} for r in chosen])
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--output-root")
    a = ap.parse_args()
    print(write_sample(a.fold, a.arm, a.n, a.output_root))


if __name__ == "__main__":
    main()
