"""Bootstrap and paired comparison: known answers on invented counts."""
import numpy as np
import pytest

from src.evaluation.metrics import COUNT_COLUMNS, as_report, compute_metrics
from src.evaluation.statistics import bootstrap_intervals, mean_sd, paired_difference, replicate_metrics, seed_summary, stratified_weights


def table(correct: list[int], n_per_note: int = 1) -> dict:
    """One NOME entity per note, correct or missed."""
    n = len(correct)
    t = {c: np.zeros(n) for c in COUNT_COLUMNS}
    t["tp_NOME"] = np.array(correct, dtype=float) * n_per_note
    t["fn_NOME"] = (1 - np.array(correct, dtype=float)) * n_per_note
    t["notes"] = np.ones(n)
    return t


def test_weights_keep_each_fold_size_in_every_replicate():
    folds = [1] * 16 + [2] * 17 + [3] * 17 + [4] * 16 + [5] * 16
    w = stratified_weights(folds, 2000, 42)
    a = np.array(folds)
    for k, size in zip(range(1, 6), (16, 17, 17, 16, 16)):
        assert (w[:, a == k].sum(axis=1) == size).all()
    assert w.shape == (2000, 82)


def test_weights_are_reproducible_and_seed_dependent():
    folds = [1] * 5 + [2] * 5
    assert (stratified_weights(folds, 2000, 42) == stratified_weights(folds, 2000, 42)).all()
    assert (stratified_weights(folds, 2000, 42) != stratified_weights(folds, 2000, 43)).any()


def test_fewer_than_the_minimum_iterations_is_refused():
    with pytest.raises(ValueError):
        stratified_weights([1, 2], 1999, 42)


def test_point_matches_pooled_metrics_and_lies_in_the_interval():
    t = table([1] * 30 + [0] * 10)
    folds = [1] * 20 + [2] * 20
    out = bootstrap_intervals(t, folds, 2000, 42)
    assert out["micro_f1"]["point"] == pytest.approx(as_report(compute_metrics({c: float(t[c].sum()) for c in COUNT_COLUMNS}))["micro_f1"])
    assert out["micro_recall"]["point"] == pytest.approx(0.75)
    assert out["micro_recall"]["lower"] <= 0.75 <= out["micro_recall"]["upper"]


def test_perfect_system_has_a_degenerate_interval():
    out = bootstrap_intervals(table([1] * 20), [1] * 10 + [2] * 10, 2000, 42)
    assert out["micro_f1"]["lower"] == out["micro_f1"]["upper"] == 1.0


def test_interval_width_matches_the_binomial_expectation():
    """A proportion of 0.8 over 400 independent notes: half-width near 1.96 * sqrt(.8 * .2 / 400) = 0.039."""
    t = table([1] * 320 + [0] * 80)
    out = bootstrap_intervals(t, [1] * 400, 10000, 42)["micro_recall"]
    assert (out["upper"] - out["lower"]) / 2 == pytest.approx(0.0392, abs=0.004)


def test_stratification_removes_between_fold_variance():
    """A category held in one fold keeps a constant count when folds are strata."""
    t = table([1] * 10 + [0] * 10)
    out = bootstrap_intervals(t, [1] * 10 + [2] * 10, 2000, 42)["micro_recall"]
    assert out["lower"] == out["upper"] == 0.5


def test_undefined_replicates_are_counted_not_hidden():
    t = table([1] + [0] * 39)
    for c in COUNT_COLUMNS:
        t[c][1:] = 0
    t["tp_NOME"][0] = 1
    out = bootstrap_intervals(t, [1] * 40, 2000, 42)["micro_recall"]
    assert out["undefined_replicates"] > 0 and out["lower"] == 1.0


def test_identical_models_differ_by_exactly_zero():
    t = table([1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 4)
    out = paired_difference(t, t, [1] * 20 + [2] * 20, 2000, 42)["micro_f1"]
    assert out["difference"] == 0 and out["lower"] == 0 and out["upper"] == 0


def test_difference_is_b_minus_a_and_excludes_zero_when_b_dominates():
    a, b = table([0] * 20 + [1] * 20), table([1] * 38 + [0] * 2)
    folds = [1, 2] * 20
    out = paired_difference(a, b, folds, 2000, 42)["micro_recall"]
    assert out["difference"] == pytest.approx(0.45) and out["lower"] > 0
    flipped = paired_difference(b, a, folds, 2000, 42)["micro_recall"]
    assert flipped["difference"] == pytest.approx(-0.45) and flipped["upper"] < 0
    assert flipped["lower"] == pytest.approx(-out["upper"]) and flipped["upper"] == pytest.approx(-out["lower"])


def test_pairing_uses_the_same_documents_for_both_models():
    """Correlated models: the paired interval is narrower than treating the models as independent."""
    rng = np.random.default_rng(1)
    base = (rng.random(200) < 0.7).astype(int).tolist()
    tweaked = base.copy()
    for i in range(0, 20):
        tweaked[i] = 1
    a, b = table(base), table(tweaked)
    folds = [1] * 200
    paired = paired_difference(a, b, folds, 4000, 42)["micro_recall"]
    w1, w2 = stratified_weights(folds, 4000, 42), stratified_weights(folds, 4000, 7)
    unpaired = replicate_metrics(b, w1)["micro_recall"] - replicate_metrics(a, w2)["micro_recall"]
    assert paired["upper"] - paired["lower"] < 0.5 * (np.quantile(unpaired, .975) - np.quantile(unpaired, .025))


def test_mean_sd_uses_the_sample_sd():
    assert mean_sd([1, 2, 3]) == {"mean": 2.0, "sd": 1.0, "n": 3}
    assert mean_sd([None, None]) == {"mean": None, "sd": None, "n": 0}
    assert mean_sd([0.5])["sd"] is None


def test_seed_summary():
    out = seed_summary({1: {"micro_f1": 0.8}, 2: {"micro_f1": 0.9}, 3: {"micro_f1": 1.0}}, metrics=("micro_f1",))
    assert out["seeds"] == [1, 2, 3] and out["metrics"]["micro_f1"]["mean"] == pytest.approx(0.9) and out["metrics"]["micro_f1"]["sd"] == pytest.approx(0.1)
