"""Independent post-generation audit. This is the only generation-stage code that reads test notes.

It never removes records. Any violation stops the fold and needs a person to look at it,
because using test notes to filter training data would itself be leakage.

    python -m src.generation.post_audit --fold 1 --arm style_anchored
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import json
import pathlib
import sys

from src.data.annotations import canonical_type, word_boundary_regex
from src.data.deduplication import longest_common_run
from src.data.loaders import DEFAULT_GOLD, ROOT, load_gold
from src.data.normalization import doc_hash, tokens, word_ngrams
from src.data.splits import SPLITS_DIR, load_fold, load_train_notes
from src.generation.anchor_selection import AnchorSelector
from src.generation.generation_plan import build_plan
from src.generation.prompt_templates import PROMPTS_DIR, prompt_sha
from src.generation.synthetic_generator import build_request_prompt, group_requests, load_config


ENTITY_MATCH_POLICY = "v1, owner 2026-11-09; terminology: candidate match, never leak"


def entity_class(entity_type: str, value: str) -> str:
    """Policy v1: an exact full-entity match is a candidate match, except single-token NOME, ETÀ and DATA, which are counted
    apart. A candidate match is not a proven leak; the non-anchored arm, which never sees gold text, is its comparison."""
    if len(value.split()) == 1 and entity_type in ("NOME", "ETÀ", "DATA"):
        return f"single_token_{entity_type}"
    return "candidate_match"


def entity_matches(examples, test_notes) -> dict:
    """L2 check: a full test-note entity string, whitespace collapsed and case kept, found at word boundaries anywhere in a
    synthetic record. Reports types, classes, note IDs and record IDs only, never the values."""
    ws = lambda s: re.sub(r"\s+", " ", str(s)).strip()
    texts = [(ex["record_id"], ws(ex["text"])) for ex in examples]
    found = collections.defaultdict(set)
    for n in test_notes:
        for e in n.entities:
            t, value = canonical_type(e.get("type")), ws(e.get("text", ""))
            if t and value:
                found[(t, value)].add(n.note_id)
    counts = collections.defaultdict(collections.Counter)
    matches = []
    for (t, value), notes in sorted(found.items(), key=lambda kv: (kv[0][0], sorted(kv[1]))):
        c = entity_class(t, value)
        rx = word_boundary_regex(value)
        records = [rid for rid, text in texts if rx.search(text)]
        counts[c]["test_entities"] += 1
        counts[c]["matched"] += bool(records)
        if records and c == "candidate_match":
            matches.append({"type": t, "n_tokens": len(value.split()), "test_notes": sorted(notes), "n_records": len(records), "record_ids": records[:10]})
    return {"policy": ENTITY_MATCH_POLICY, "by_class": {k: dict(v) for k, v in sorted(counts.items())}, "candidate_matches": matches}


def audit_fold(cfg, fold, arm, output_root=None, gold_path=DEFAULT_GOLD, splits_dir=SPLITS_DIR, prompts_dir=PROMPTS_DIR, max_run=12) -> dict:
    spec = load_fold(fold, splits_dir)
    run_dir = pathlib.Path(output_root) if output_root else ROOT / "synthetic_data"
    run_dir = run_dir / f"fold_{fold}" / arm
    rows = lambda name: ([json.loads(x) for x in (run_dir / name).read_text(encoding="utf-8").splitlines() if x.strip()]
                         if (run_dir / name).exists() else [])
    examples, requests = rows("examples.jsonl"), rows("requests.jsonl")
    test_ids = set(spec.test_ids)
    test_notes = [n for n in load_gold(gold_path) if n.note_id in test_ids]
    test_hash = {doc_hash(n.text): n.note_id for n in test_notes}
    test_tok = {n.note_id: tokens(n.text) for n in test_notes}
    test_grams = {k: word_ngrams(v, 8) for k, v in test_tok.items()}
    violations = []

    # 1. anchors logged per request must be training notes
    for r in requests:
        bad = set(r.get("anchor_ids", [])) & test_ids
        if bad:
            violations.append({"check": "test_note_used_as_anchor", "request_id": r["request_id"], "note_ids": sorted(bad)})

    # 2. re-render each request; hash must match the log, and no test-note text may be in the prompt
    train_notes = load_train_notes(spec, gold_path)
    selector = AnchorSelector(spec, train_notes, cfg["anchors"]["strategy"], cfg["seed"], cfg["anchors"]["n"])
    plan = {s.case_id: s for s in build_plan(fold, train_notes, cfg)}
    rerendered = hash_mismatch = 0
    for r in requests:
        specs = [plan[c] for c in r["case_ids"]]
        group = {"index": r["index"], "attempt": r["attempt"], "specs": specs}
        prompt, anchors = build_request_prompt(group, arm, selector, cfg, prompts_dir)
        rerendered += 1
        if prompt_sha(prompt) != r["prompt_sha256"]:
            hash_mismatch += 1
            violations.append({"check": "prompt_hash_mismatch", "request_id": r["request_id"]})
        # A training anchor may share a stock phrase with a test note. Only text its anchors cannot explain is a violation.
        ptoks = word_ngrams(tokens(prompt), 8)
        explained = set()
        for a in anchors:
            explained |= word_ngrams(tokens(a.text), 8)
        for k, g in test_grams.items():
            if (ptoks & g) - explained:
                violations.append({"check": "test_note_text_in_prompt", "request_id": r["request_id"], "note_ids": [k]})

    # 3. generated records against test notes
    near = 0
    longest = 0
    for ex in examples:
        h = doc_hash(ex["text"])
        if h in test_hash:
            violations.append({"check": "synthetic_equals_test_note", "record_id": ex["record_id"], "note_ids": [test_hash[h]]})
        t = tokens(ex["text"])
        grams = word_ngrams(t, 8)
        for k, g in test_grams.items():
            shared = len(grams & g)
            if shared:
                run_len = longest_common_run(t, test_tok[k])
                longest = max(longest, run_len)
                if run_len >= max_run:
                    near += 1
                    violations.append({"check": "long_verbatim_run_with_test_note", "record_id": ex["record_id"], "note_ids": [k], "run_tokens": run_len})
    # 4. entity-level check. Reported always; blocking only when configured, because chance matches of common places and
    # dates also occur in the non-anchored arm, which never sees gold text, and that arm is the comparison for this count.
    entities = entity_matches(examples, test_notes)
    if cfg.get("post_audit", {}).get("candidate_match_blocking", False):
        for x in entities["candidate_matches"]:
            violations.append({"check": "test_entity_in_synthetic", **{k: x[k] for k in ("type", "test_notes", "n_records")}})
    passed = not violations
    report = {"fold": fold, "arm": arm, "examples_checked": len(examples), "requests_checked": len(requests), "prompts_rerendered": rerendered,
              "prompt_hash_mismatches": hash_mismatch, "longest_verbatim_run_with_any_test_note": longest,
              "entity_check": entities, "entity_candidate_matches": len(entities["candidate_matches"]), "violations": violations, "passed": passed}
    (run_dir / "post_audit_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    vr = run_dir / "validation_report.json"
    if vr.exists():
        d = json.loads(vr.read_text(encoding="utf-8"))
        d["post_audit"], d["permitted_for_training"] = ("passed" if passed else "failed"), passed
        vr.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    mf = run_dir / "generation_manifest.csv"
    if mf.exists():
        with open(mf, newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        if data:
            for row in data:
                row["permitted_for_training"] = passed and row["status"] == "accepted"
            with open(mf, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "generation_v2.yaml"))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--output-root")
    a = ap.parse_args(argv)
    r = audit_fold(load_config(a.config), a.fold, a.arm, a.output_root)
    print(json.dumps({k: v for k, v in r.items() if k != "violations"} | {"n_violations": len(r["violations"])}, indent=1))
    return 0 if r["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
