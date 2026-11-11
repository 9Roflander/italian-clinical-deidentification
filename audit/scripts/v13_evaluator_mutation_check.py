"""Do the evaluator tests detect real bugs? Plants one deliberate bug at a time in a temporary copy and runs the tests.

A mutation "survives" if all tests still pass, which would mean the suite cannot see that bug. The project is untouched.
Run with .venv-gen/bin/python audit/scripts/v13_evaluator_mutation_check.py
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
LINKS = ("data_secure", "splits", "configs", "data_manifests", "audit", "original_work", "prompts")
TESTS = sorted(p.name for p in (ROOT / "tests").glob("test_evaluation_*.py"))
MUTATIONS = [
    ("undefined metrics reported as 1.0", "metrics.py", "out=np.full(np.broadcast(a, b).shape, np.nan)", "out=np.full(np.broadcast(a, b).shape, 1.0)"),
    ("unknown placeholders ignored in micro", "metrics.py", 'for c in CATEGORIES) + t["fp_unknown"]', "for c in CATEGORIES)"),
    ("macro skips categories scoring zero", "metrics.py", "defined = np.sum(~np.isnan(stacked), axis=0)", "defined = np.sum(~np.isnan(stacked) & (stacked > 0), axis=0)"),
    ("duplicates earn extra true positives", "scoring.py", "        elif p in unmatched:\n            unmatched.remove(p)", "        elif p in gold:\n            unmatched.discard(p)"),
    ("category ignored when matching", "scoring.py", "gold = {Span(*g) for g in gold_spans}", 'gold = {Span(g[0], g[1], "NOME") for g in gold_spans}'),
    ("invalid outputs drop their false negatives", "scoring.py", '        counts[f"fn_{g.label}"] += 1\n', '        counts[f"fn_{g.label}"] += int(resolved.is_valid)\n'),
    ("invalid outputs drop their false positives", "scoring.py", 'counts[f"fp_{label}" if label in CATEGORIES else "fp_unknown"] += 1', 'counts[f"fp_{label}" if label in CATEGORIES else "fp_unknown"] += 0'),
    ("over-redaction always zero", "scoring.py", 'counts["over_redacted_chars"] = counts["predicted_chars"] - counts["covered_chars"]', 'counts["over_redacted_chars"] = 0'),
    ("note-exact ignores extra predictions", "scoring.py", "and len(predictions) == len(gold) and len(correct) == len(gold))", "and len(correct) == len(gold))"),
    ("empty gaps allowed in alignment", "alignment.py", "while pointer < len(ends) and ends[pointer] < p:", "while pointer < len(ends) and ends[pointer] <= p:"),
    ("ambiguous alignment guessed", "alignment.py", "    if total != 1:\n        return total, None", "    if total < 1:\n        return total, None"),
    ("whitespace silently collapsed", "alignment.py", '    text = unicodedata.normalize("NFC", raw)\n', '    text = " ".join(unicodedata.normalize("NFC", raw).split())\n'),
    ("truncation flag ignored", "alignment.py", 'if truncation_flag or completion_status == "truncated" or finish_reason in LENGTH_REASONS:', "if False:"),
    ("bootstrap not stratified", "statistics.py", "members = np.flatnonzero(folds == k)", "members = np.arange(folds.size)"),
    ("paired comparison resamples different documents", "statistics.py", "a, b = replicate_metrics(table_a, weights), replicate_metrics(table_b, weights)", "a, b = replicate_metrics(table_a, weights), replicate_metrics(table_b, weights[::-1])"),
    ("old metric equation changed", "historical.py", "fn = sum(1 for x in texts if x and x in redacted)", "fn = sum(1 for x in texts if x and x not in redacted)"),
    ("gold counts not checked", "gold_verification.py", 'if dict(counts) != expected["mentions"] or sum(counts.values()) != expected["total_mentions"]:', "if False:"),
    ("prediction from wrong fold model accepted", "runner.py", 'if r["fold_id"] != gold.fold_of[i]:', "if False:"),
    ("missing notes dropped", "runner.py", "        if r is None:\n            resolved = Resolution(MISSING)", "        if r is None:\n            continue"),
    ("input hash not checked", "runner.py", 'if r["input_sha256"] != text_sha256(gold.notes[i].text):', "if False:"),
]


def run_tests(workdir) -> bool:
    done = subprocess.run([sys.executable, "-m", "pytest", *[f"tests/{t}" for t in TESTS], "-x", "-q", "--no-header", "-p", "no:cacheprovider"],
                          cwd=workdir, capture_output=True, text=True, timeout=300)
    return done.returncode == 0


results = []
with tempfile.TemporaryDirectory() as tmp:
    work = pathlib.Path(tmp)
    shutil.copytree(ROOT / "src", work / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "tests", work / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    for name in LINKS:
        (work / name).symlink_to(ROOT / name)
    subprocess.run(["git", "init", "-q"], cwd=work)
    assert run_tests(work), "the unmodified copy must pass"
    for label, filename, old, new in MUTATIONS:
        path = work / "src" / "evaluation" / filename
        original = path.read_text(encoding="utf-8")
        assert original.count(old) >= 1, f"mutation does not apply: {label}"
        path.write_text(original.replace(old, new, 1), encoding="utf-8")
        survived = run_tests(work)
        path.write_text(original, encoding="utf-8")
        results.append({"bug": label, "detected": not survived})
        print(("SURVIVED " if survived else "detected ") + label, flush=True)
report = {"purpose": "Deliberate bugs planted in a temporary copy of the evaluator.", "planted": len(results), "detected": sum(r["detected"] for r in results), "results": results}
(ROOT / "audit" / "results" / "v13_evaluator_mutation_check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
print(f"{report['detected']} of {report['planted']} detected")
