"""Fold summaries and paired document-level bootstrap, stratified by the frozen folds.

Intervals quantify document-sampling uncertainty conditional on the trained fold models. They do not capture all
training randomness. There is deliberately no paired t-test over five folds: those scores are not independent.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import COUNT_COLUMNS, as_report, compute_metrics

SUMMARY_METRICS = ("micro_precision", "micro_recall", "micro_f1", "macro_f1")


def stratified_weights(folds, iterations: int, seed: int, minimum_iterations: int = 2000) -> np.ndarray:
    """B x N resampling counts. Each replicate draws each fold's notes with replacement, keeping fold sizes fixed.
    The same seed and note order give the same matrix, so two models are resampled on the same documents."""
    if iterations < minimum_iterations:
        raise ValueError(f"at least {minimum_iterations} iterations are required")
    folds = np.asarray(folds)
    rng = np.random.default_rng(seed)
    weights = np.zeros((iterations, folds.size), dtype=np.int32)
    for k in sorted(set(folds.tolist())):
        members = np.flatnonzero(folds == k)
        weights[:, members] = rng.multinomial(members.size, np.full(members.size, 1 / members.size), size=iterations)
    return weights


def per_note_table(rows) -> dict:
    return {c: np.array([float(r[c]) for r in rows]) for c in COUNT_COLUMNS}


def replicate_metrics(table: dict, weights: np.ndarray) -> dict:
    return compute_metrics({c: weights @ table[c] for c in COUNT_COLUMNS})


def _interval(values: np.ndarray, confidence: float) -> dict:
    undefined = int(np.isnan(values).sum())
    if undefined == values.size:
        return {"lower": None, "upper": None, "undefined_replicates": undefined}
    a = (1 - confidence) / 2
    low, high = np.nanquantile(values, [a, 1 - a])
    return {"lower": float(low), "upper": float(high), "undefined_replicates": undefined}


def bootstrap_intervals(table: dict, folds, iterations: int, seed: int, confidence: float = 0.95, minimum_iterations: int = 2000) -> dict:
    point = as_report(compute_metrics({c: float(table[c].sum()) for c in COUNT_COLUMNS}))
    reps = replicate_metrics(table, stratified_weights(folds, iterations, seed, minimum_iterations))
    return {k: {"point": point[k], **_interval(v, confidence)} for k, v in reps.items()}


def paired_difference(table_a: dict, table_b: dict, folds, iterations: int, seed: int, confidence: float = 0.95, minimum_iterations: int = 2000) -> dict:
    """Difference B minus A, with both models resampled on the same documents in each replicate."""
    weights = stratified_weights(folds, iterations, seed, minimum_iterations)
    a, b = replicate_metrics(table_a, weights), replicate_metrics(table_b, weights)
    point_a = as_report(compute_metrics({c: float(table_a[c].sum()) for c in COUNT_COLUMNS}))
    point_b = as_report(compute_metrics({c: float(table_b[c].sum()) for c in COUNT_COLUMNS}))
    out = {}
    for k in a:
        delta = None if point_a[k] is None or point_b[k] is None else point_b[k] - point_a[k]
        out[k] = {"a": point_a[k], "b": point_b[k], "difference": delta, **_interval(b[k] - a[k], confidence)}
    return out


def mean_sd(values) -> dict:
    """Mean and sample SD (ddof=1) of the defined values. These are not independent experiments."""
    defined = [v for v in values if v is not None]
    if not defined:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(np.mean(defined)), "sd": float(np.std(defined, ddof=1)) if len(defined) > 1 else None, "n": len(defined)}


def seed_summary(pooled_by_seed: dict, metrics=SUMMARY_METRICS) -> dict:
    """Spread of pooled results across training seeds. One seed means training randomness is unmeasured."""
    return {"seeds": sorted(pooled_by_seed), "metrics": {m: mean_sd([p.get(m) for p in pooled_by_seed.values()]) for m in metrics}}
