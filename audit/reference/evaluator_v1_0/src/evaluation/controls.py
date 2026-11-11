"""Constructed outputs with known correct scores, run on the real gold notes through the real evaluator.

Gold spans are used here only to build test outputs. The reader in alignment.py never receives them. The report
contains aggregate numbers only. No model is involved, so nothing here is a model result.
"""
from __future__ import annotations

import json
import re

from src.data.loaders import ROOT
from src.evaluation.alignment import CATEGORIES, render_inline, spans_to_segments
from src.evaluation.gold_verification import VerifiedGold
from src.evaluation.metrics import as_report, compute_metrics
from src.evaluation.scoring import totals

NEXT_CATEGORY = dict(zip(CATEGORIES, CATEGORIES[1:] + CATEGORIES[:1]))
REPORT_METRICS = ("micro_f1", "macro_f1", "micro_recall", "removal_coverage", "over_redaction_rate", "non_sensitive_preservation", "note_exact_rate", "invalid_output_rate")


def _record(gold: VerifiedGold, note_id: str, experiment: str, output_format: str, raw=None, spans=None, **extra) -> dict:
    from src.evaluation.runner import text_sha256
    return {"experiment": experiment, "model": "none", "model_revision": "none", "checkpoint": "none", "config_id": "control", "seed": 0,
            "note_id": note_id, "fold_id": gold.fold_of[note_id], "input_sha256": text_sha256(gold.notes[note_id].text),
            "output_format": output_format, "raw_model_output": raw, "predicted_spans": spans, "generation_parameters": {},
            "completion_status": "complete", "finish_reason": "none", "truncation_flag": False, "prompt_tokens": 0, "generated_tokens": 0, **extra}


def baseline_e0_records(gold: VerifiedGold) -> list[dict]:
    """E0: the unchanged-input baseline."""
    return [_record(gold, i, "E0", "inline", raw=gold.notes[i].text) for i in gold.order]


def _inline(fn):
    def build(gold, i, name):
        return _record(gold, i, name, "inline", raw=fn(gold.notes[i].text, gold.spans[i]))
    return build


def _wrong_category(text, spans):
    return render_inline(text, [(a, b, NEXT_CATEGORY[c]) for a, b, c in spans])


def _unknown_tags(text, spans):
    out = render_inline(text, spans)
    return out.replace("[ETÀ]", "[ETA]").replace("[NOME]", "[NAME]").replace("[DATA]", "[DATE]").replace("[LUOGO/INDIRIZZO]", "[LUOGO]")


def _shrunk(text, spans):
    """Each span loses its last character, or gains the next one if it has only one."""
    return [(a, b - 1 if b - a > 1 else b + 1, c) for a, b, c in spans]


def _every_other(text, spans):
    return render_inline(text, spans[::2])


def _one_typo(text, spans):
    taken = {i for a, b, _ in spans for i in range(a, b)}
    i = next(k for k, ch in enumerate(text) if ch.isalpha() and k not in taken)
    changed = text[:i] + ("x" if text[i] != "x" else "y") + text[i + 1 :]
    return render_inline(changed, spans)


def _truncated(text, spans):
    full = render_inline(text, spans)
    return full[: len(full) // 2]


def _collapsed(text, spans):
    return re.sub(r" {2,}", " ", render_inline(text, spans))


def _preamble(text, spans):
    return "Testo redatto: " + render_inline(text, spans)


def _numbers_as_age(gold, i, name):
    spans = [[m.start(), m.end(), "ETÀ"] for m in re.finditer(r"\d+", gold.notes[i].text)]
    return _record(gold, i, name, "spans", spans=spans)


CONTROLS = {
    "perfect_segments": lambda g, i, n: _record(g, i, n, "segments", raw=json.dumps(spans_to_segments(g.notes[i].text, g.spans[i]), ensure_ascii=False)),
    "perfect_spans": lambda g, i, n: _record(g, i, n, "spans", spans=[list(s) for s in g.spans[i]]),
    "perfect_inline": _inline(render_inline),
    "empty_output": lambda g, i, n: _record(g, i, n, "inline", raw=""),
    "unchanged_input": lambda g, i, n: _record(g, i, n, "inline", raw=g.notes[i].text),
    "every_entity_wrong_category": _inline(_wrong_category),
    "every_boundary_off_by_one": lambda g, i, n: _record(g, i, n, "spans", spans=[list(s) for s in _shrunk(g.notes[i].text, g.spans[i])]),
    "every_other_entity_missed": _inline(_every_other),
    "unknown_placeholders": _inline(_unknown_tags),
    "all_numbers_tagged_as_age": _numbers_as_age,
    "single_placeholder_for_whole_note": lambda g, i, n: _record(g, i, n, "inline", raw="[NOME]"),
    "truncated_at_half_unflagged": _inline(_truncated),
    "complete_but_flagged_truncated": lambda g, i, n: _record(g, i, n, "inline", raw=render_inline(g.notes[i].text, g.spans[i]), truncation_flag=True, finish_reason="length"),
    "one_typo_outside_entities": _inline(_one_typo),
    "preamble_added": _inline(_preamble),
    "double_spaces_collapsed": _inline(_collapsed),
}


def run_control(name: str, gold: VerifiedGold) -> dict:
    from src.evaluation.runner import evaluate_records
    from src.evaluation.historical import historical_scores
    result = evaluate_records([CONTROLS[name](gold, i, name) for i in gold.order], gold)
    t = totals(result.scores)
    metrics = as_report(compute_metrics(t))
    old = historical_scores(result.outputs, {i: gold.notes[i].entities for i in gold.order}, gold.fold_of)
    return {**{m: metrics[m] for m in REPORT_METRICS}, "invalid_outputs": t["notes_invalid"], "fp_total": sum(t[f"fp_{c}"] for c in CATEGORIES) + t["fp_unknown"],
            "historical_paper_metric": old["paper_metric"]["macro_f1_mean_over_folds"], "historical_miranda": old["miranda_deterministic"]["macro_f1_mean_over_folds"]}


def write_control_report(gold: VerifiedGold):
    report = {"purpose": "Constructed outputs, not model results. Aggregate numbers only.", "gold_sha256": gold.sha256,
              "controls": {name: run_control(name, gold) for name in CONTROLS}}
    path = ROOT / "audit" / "results" / "evaluation_controls.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path
