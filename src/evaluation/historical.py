"""Historical placeholder-count metrics, kept only for comparison with Miranda et al.

These are not quality measures: empty output scores 1.0 on both. `paper_metric` is the project's old code, unchanged, and
a test compares its syntax tree with original_work/src/train_evaluate_cv_v4.py. `miranda_deterministic` restates the
equations of their eval.py in this project's own code, because their repository has no licence file. Tests check it
against results from running their function (audit/results/v9_source_metric*.json).
"""
from __future__ import annotations

import collections
import re

VALID_TYPES = ["NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"]


def sanitize_type(etype):
    etype = (etype or "").upper()
    if "LUOGO" in etype or "INDIRIZZO" in etype:
        return "LUOGO/INDIRIZZO"
    if etype in VALID_TYPES:
        return etype
    return None


def paper_metric(redacted, gold_entities):
    by_type = {t: [] for t in VALID_TYPES}
    for e in gold_entities:
        et = sanitize_type(e.get("type", ""))
        if et:
            by_type[et].append(e.get("text", ""))
    result = {}
    for t in VALID_TYPES:
        texts = by_type[t]
        n_gold = len(texts)
        fn = sum(1 for x in texts if x and x in redacted)
        tp = n_gold - fn
        fp = max(0, redacted.count(f"[{t}]") - n_gold)
        result[t] = (tp, fp, fn)
    return result


def miranda_deterministic(output: str, annotations) -> dict:
    """Group annotations by string. FN counts every occurrence left in the output, substrings included."""
    result = {t: [0, 0, 0] for t in VALID_TYPES}
    groups = collections.Counter((a["text"], a["type"]) for a in annotations)
    for (text, label), gold_count in groups.items():
        if label not in result:
            continue
        left = len(re.findall(re.escape(text), output))
        result[label][0] += max(gold_count - left, 0)
        result[label][2] += left
    for label in VALID_TYPES:
        placeholders = output.count(f"[{label}]")
        gold_count = sum(1 for a in annotations if a["type"] == label)
        result[label][1] += max(placeholders - gold_count, 0)
    return {k: tuple(v) for k, v in result.items()}


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def historical_scores(outputs: dict, entities_by_note: dict, fold_of: dict) -> dict:
    """Both metrics with the old aggregation: fold totals, then F1 per type, then the mean over folds."""
    annotations = {i: [{"type": e["type"], "text": e["annotated_text"]} for e in ents] for i, ents in entities_by_note.items()}
    metrics = {"paper_metric": lambda i: paper_metric(outputs[i], annotations[i]),
               "miranda_deterministic": lambda i: miranda_deterministic(outputs[i], annotations[i])}
    report = {}
    for name, fn in metrics.items():
        by_fold = collections.defaultdict(lambda: {t: [0, 0, 0] for t in VALID_TYPES})
        pooled = {t: [0, 0, 0] for t in VALID_TYPES}
        for i in outputs:
            for t, vals in fn(i).items():
                for k, v in enumerate(vals):
                    by_fold[fold_of[i]][t][k] += v
                    pooled[t][k] += v
        macros = [sum(_f1(*tot[t]) for t in VALID_TYPES) / 4 for tot in by_fold.values()]
        report[name] = {"macro_f1_mean_over_folds": sum(macros) / len(macros), "per_type_f1_pooled": {t: _f1(*pooled[t]) for t in VALID_TYPES}}
    return report
