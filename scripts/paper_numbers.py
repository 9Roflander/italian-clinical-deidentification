"""Recompute the numbers reported in the paper from the files in this repository.

Usage: python scripts/paper_numbers.py

Reads only released files: experiments/, synthetic_data/, audit/ and data_manifests/. It needs neither the gold notes nor
model outputs, and it prints aggregate numbers only. Each block names the paper section or table it reproduces.
"""
from __future__ import annotations

import csv
import collections
import glob
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
SEEDS = (42, 43, 44)

RUNS = {
    ("S1", "E1", 42): "clean_cv/E1__7123858ac751__seed42",
    ("S1", "E2", 42): "clean_cv/E2__fb789fdc4c29__seed42",
    ("S1", "E3", 42): "clean_cv/E3__137665080a72__seed42",
    ("S2", "E1", 42): "clean_cv/E1__f412cdf4c3d8__seed42",
    ("S2", "E3", 42): "clean_cv/E3__00a0cf19e492__seed42",
    ("S3v1", "E1", 42): "baselines/E1-BERT__cc92f090aa20__seed42",
    ("S3v1", "E3", 42): "baselines/E3-BERT__ccd62f5aee70__seed42",
}
BERT = {"E1": "ae91fdcd33e2", "E2": "9925c1187755", "E3": "afe03a16820e",
        "E8": "f2bc7e028da1", "E9": "09ef32838f7e", "E10": "cebb42a994d9"}
for s in SEEDS:
    RUNS[("S2", "E2", s)] = f"clean_cv/E2__ccc3fa5896f3__seed{s}"
    for arm, cfg in BERT.items():
        RUNS[("S3", arm, s)] = f"baselines/{arm}-BERT__{cfg}__seed{s}"
SYSTEM_OF_CONFIG = {"7123858ac751": "S1", "fb789fdc4c29": "S1", "137665080a72": "S1",
                    "f412cdf4c3d8": "S2", "ccc3fa5896f3": "S2", "00a0cf19e492": "S2",
                    "cc92f090aa20": "S3v1", "ccd62f5aee70": "S3v1", **{v: "S3" for v in BERT.values()}}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pooled(key):
    return load(EXP / RUNS[key] / "pooled_metrics.json")


def boot(key):
    return load(EXP / RUNS[key] / "bootstrap_results.json")["metrics"]


def f3(x):
    return f"{x:.3f}"


def header(title):
    print(f"\n== {title}")


def main_results():
    header("Tables 5 and 6: seed 42, pooled over the 82 out-of-fold notes")
    print("run    micro-F1 [95% CI]        macro-F1 [95% CI]        P      R      invalid  Cov    OR")
    for system in ("S1", "S2", "S3"):
        for arm in ("E1", "E2", "E3"):
            k = (system, arm, 42)
            m, b = pooled(k)["metrics"], boot(k)
            ci = lambda name: f"[{f3(b[name]['lower'])}, {f3(b[name]['upper'])}]"
            print(f"{system} {arm}  {f3(m['micro_f1'])} {ci('micro_f1')}   {f3(m['macro_f1'])} {ci('macro_f1')}   "
                  f"{f3(m['micro_precision'])}  {f3(m['micro_recall'])}  {100 * m['invalid_output_rate']:5.1f}%   "
                  f"{f3(m['removal_coverage'])}  {f3(m['over_redaction_rate'])}")
    header("Table 5 and Section 7.4: micro-F1 over seeds 42, 43, 44 (mean ± sample standard deviation)")
    for system, arm in [("S2", "E2")] + [("S3", a) for a in BERT]:
        v = [pooled((system, arm, s))["metrics"]["micro_f1"] for s in SEEDS]
        print(f"{system} {arm:3s}  {' '.join(f3(x) for x in v)}  ->  {f3(st.mean(v))} ± {f3(st.stdev(v))}")
    header("Section 5.4: first BERT configuration (seed 42)")
    for arm in ("E1", "E3"):
        print(f"S3v1 {arm}  micro-F1 {f3(pooled(('S3v1', arm, 42))['metrics']['micro_f1'])}")


def label(run_id):
    experiment, config, seed = run_id.split("__")
    return f"{SYSTEM_OF_CONFIG[config]} {experiment.replace('-BERT', '')} {seed[4:]}"


def comparisons():
    header("Section 7.2 and Fig. 1: paired bootstrap differences in micro-F1 (b minus a); * = interval excludes 0")
    for path in sorted(glob.glob(str(EXP / "comparisons" / "*.json"))):
        c = load(path)
        m = c["metrics"]["micro_f1"]
        flag = "*" if m["lower"] > 0 or m["upper"] < 0 else " "
        print(f"{label(c['b']):12s} - {label(c['a']):12s}  {m['difference']:+.3f} [{m['lower']:+.3f}, {m['upper']:+.3f}] {flag}")


def error_analysis():
    header("Table 7: outcome of the gold mentions and spurious predictions, arm E2, seed 42")
    for system in ("S1", "S2", "S3"):
        d = collections.Counter()
        with open(EXP / RUNS[(system, "E2", 42)] / "error_breakdown.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d[(r["side"], r["kind"])] += int(r["count"])
        g = lambda kind: d[("gold", kind)]
        print(f"{system}  correct {g('correct')}  missed {g('missed')}  boundary {g('wrong_boundary')}  "
              f"category {g('wrong_category') + g('wrong_boundary_and_category')}  invalid output {g('invalid_output')}  "
              f"spurious {d[('prediction', 'spurious')]}  false positives from invalid outputs {d[('prediction', 'invalid_output')]}")
    header("Table 8: per-category precision / recall / F1, arm E2, seed 42")
    for system in ("S1", "S2", "S3"):
        with open(EXP / RUNS[(system, "E2", 42)] / "per_category_metrics.csv", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["scope"] == "pooled"]
        print(system, "  ".join(f"{r['category']} {float(r['precision']):.2f}/{float(r['recall']):.2f}/{float(r['f1']):.2f}"
                                for r in rows))
    header("Section 7.2: invalid outputs of the generative systems by cause")
    for system, arm, s in [("S1", a, 42) for a in ("E1", "E2", "E3")] + [("S2", "E1", 42)] + [("S2", "E2", x) for x in SEEDS] + [("S2", "E3", 42)]:
        a = load(EXP / RUNS[(system, arm, s)] / "alignment_report.json")
        print(f"{system} {arm} {s}  invalid {a['invalid_outputs']} of {a['notes']}  {a['by_status']}")


def gold():
    header("Section 3.2 and Table 2: gold standard v2")
    stats = {}
    with open(ROOT / "audit" / "gold_dataset_statistics.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            stats[(r["section"], r["metric"])] = r["value"]
    for cat in ("NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"):
        print(f"{cat:16s} mentions {stats[(cat, 'mentions')]}  per note {float(stats[(cat, 'mentions_per_note')]):.2f}  "
              f"notes with none {stats[(cat, 'notes_with_none')]} ({float(stats[(cat, 'pct_notes_with_none')]):.0f}%)")
    print(f"total mentions {stats[('corpus', 'total_annotated_mentions')]}, notes without any {stats[('mentions_per_note', 'notes_with_0_mentions')]}")
    summary = load(ROOT / "audit" / "results" / "gold_v2_build_summary.json")
    print("review decisions", summary["decisions_read"], " changes", summary["changes"], " by type", summary["changes_by_type"])
    with open(EXP / RUNS[("S3", "E2", 42)] / "per_note_metrics.csv", encoding="utf-8") as f:
        lengths = sorted(int(r["gold_chars"]) + int(r["non_sensitive_chars"]) for r in csv.DictReader(f))
    print(f"note length: min {lengths[0]}, median {st.median(lengths)}, max {lengths[-1]} characters ({len(lengths)} notes)")
    with open(ROOT / "data_manifests" / "fold_assignments.csv", encoding="utf-8") as f:
        sizes = collections.Counter(r["test_fold"] for r in csv.DictReader(f))
    print("test fold sizes", [sizes[str(k)] for k in range(1, 6)])


def synthetic():
    header("Sections 4.1, 4.3 and 4.4: synthetic data")
    arms = ("non_anchored", "style_anchored")
    accepted = {a: {} for a in arms}
    for a in arms:
        requests = slots = attempts = 0
        finish, versions, sizes = collections.Counter(), collections.Counter(), collections.Counter()
        for k in range(1, 6):
            d = ROOT / "synthetic_data" / f"fold_{k}" / a
            for line in open(d / "requests.jsonl", encoding="utf-8"):
                r = json.loads(line)
                requests += 1
                slots += len(r["case_ids"])
                sizes[len(r["case_ids"])] += 1
                finish[r["finish_reason"]] += 1
                versions[r["model_version"]] += 1
            log = [json.loads(x) for x in open(d / "case_log.jsonl", encoding="utf-8") if x.strip()]
            attempts += len(log)
            accepted[a][k] = {x["case_id"] for x in log if x["status"] == "accepted"}
        n = sum(len(v) for v in accepted[a].values())
        print(f"{a}: {requests} requests {dict(finish)} {dict(versions)}; cases per request {dict(sizes)}; "
              f"{slots} cases in requests, {attempts} case attempts, {n} accepted ({100 * n / attempts:.0f}%), "
              f"per fold {[len(accepted[a][k]) for k in range(1, 6)]}")
    print("accepted in both arms, per fold", [len(accepted[arms[0]][k] & accepted[arms[1]][k]) for k in range(1, 6)])
    with open(ROOT / "synthetic_data" / "fold_1" / "non_anchored" / "generation_plan.csv", encoding="utf-8") as f:
        print("plan of fold 1", dict(collections.Counter(r["record_type"] for r in csv.DictReader(f))))
    total, tokens = 0, collections.Counter()
    for k in range(1, 6):
        for a in arms:
            r = load(ROOT / "synthetic_data" / f"fold_{k}" / a / "post_audit_report.json")
            matches = r["entity_check"]["candidate_matches"]
            total += len(matches)
            tokens.update(x["n_tokens"] for x in matches)
            print(f"fold {k} {a:14s} prompts re-rendered {r['prompts_rerendered']}, hash mismatches {r['prompt_hash_mismatches']}, "
                  f"longest run shared with a test note {r['longest_verbatim_run_with_any_test_note']}, candidate matches {len(matches)}")
    print(f"candidate entity matches: {total}, by number of words {dict(tokens)}")


def training():
    header("Section 5 and Table 4: training records")
    root = EXP / "training_records"
    if not root.exists():
        print("experiments/training_records/ not found")
        return
    by_run = collections.defaultdict(list)
    for path in sorted(root.glob("*/fold_*/training_manifest.json")):
        by_run[path.parts[-3]].append(load(path))
    for run, ms in sorted(by_run.items()):
        steps = sorted({m.get("updates", m.get("budget", {}).get("optimizer_steps")) for m in ms})
        peak = {m["fold"]: round(m["peak_gpu_memory_bytes"] / 1e9, 2) for m in ms}
        line = f"{label(run):10s} folds {len(ms)}  updates {steps}  peak GPU GB {peak}"
        if "budget" in ms[0]:
            line += (f"  trainable {sorted({m['trainable_parameters'] for m in ms})}"
                     f"  examples {sorted({m['budget']['examples_including_repeats'] for m in ms})}"
                     f"  passes {min(m['budget']['effective_epochs'] for m in ms):.2f}-{max(m['budget']['effective_epochs'] for m in ms):.2f}"
                     f"  max tokens {max(m['token_lengths']['max'] for m in ms)}")
            minutes = []
            for log in sorted((root / run).glob("fold_*/train.log")):
                hits = re.findall(r"^step (\d+)/(\d+) .*elapsed_min ([0-9.]+)", log.read_text(encoding="utf-8", errors="replace"), re.M)
                done = [float(t) for s, n, t in hits if s == n]
                if done:
                    minutes.append(done[-1])
            if minutes:
                line += f"  training minutes {min(minutes)}-{max(minutes)}"
        else:
            line += (f"  minutes {min(m['train_minutes'] for m in ms)}-{max(m['train_minutes'] for m in ms)}"
                     f"  windows {sorted({m['data']['windows'] for m in ms})}"
                     f"  misaligned spans {sum(m['data']['spans_misaligned'] for m in ms)}")
        print(line)


if __name__ == "__main__":
    main_results()
    comparisons()
    error_analysis()
    gold()
    synthetic()
    training()
