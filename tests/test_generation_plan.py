import collections

import pytest

from src.data.loaders import load_gold
from src.data.splits import load_fold, load_train_notes
from src.generation.generation_plan import build_plan, note_profile, plan_sha, plan_summary
from src.generation.synthetic_generator import load_config


@pytest.fixture(scope="module")
def real():
    cfg = load_config("configs/generation.yaml")
    return cfg, {k: build_plan(k, load_train_notes(load_fold(k)), cfg) for k in range(1, 6)}


def test_plan_is_deterministic(real):
    cfg, plans = real
    assert plan_sha(build_plan(1, load_train_notes(load_fold(1)), cfg)) == plan_sha(plans[1])


def test_plans_differ_between_folds(real):
    assert plan_sha(real[1][1]) != plan_sha(real[1][2])


def test_pool_composition(real):
    for p in real[1].values():
        c = collections.Counter(s.record_type for s in p)
        assert c == {"synthetic_named": 1218, "synthetic_name_free": 482, "synthetic_negative": 80}
        assert c["synthetic_named"] + c["synthetic_name_free"] == 1700


def test_type_rules(real):
    for p in real[1].values():
        for s in p:
            if s.record_type == "synthetic_named":
                assert s.n_nome >= 1
            elif s.record_type == "synthetic_name_free":
                assert s.n_nome == 0 and sum(s.counts.values()) >= 2
            else:
                assert sum(s.counts.values()) == 0


def test_implausible_combinations_are_excluded(real):
    for p in real[1].values():
        for s in p:
            assert not (s.specialty == "ginecologia" and (s.gender == "uomo" or s.age_band == "0-17"))
            assert not (s.specialty == "geriatria" and s.age_band != "66-95")
            assert not (s.specialty == "pediatria" and s.age_band != "0-17")


def test_specialty_gender_age_are_not_locked_together(real):
    p = real[1][1]
    assert len({(s.specialty, s.gender) for s in p}) > 15
    assert {(s.age_band, s.density_level) for s in p} >= {("41-65", "bassa"), ("41-65", "media"), ("18-40", "bassa")}


def test_children_are_in_the_plan(real):
    assert any(s.age_band == "0-17" for s in real[1][1])


def test_entity_rates_are_far_below_the_historical_data(real):
    """Historical named notes averaged about 12 mentions; the plan follows gold notes with 2+ mentions."""
    for p in real[1].values():
        m = plan_summary(p)["mean_mentions_per_non_negative_note"]
        assert sum(m.values()) < 7
        assert m["DATA"] < 2


def test_subset_keeps_composition(real):
    """Volume ablation: the first 624 named and 254 name-free notes should resemble the pool."""
    p = real[1][1]
    pool = [s for s in p if s.record_type == "synthetic_named"]
    sub = [s for s in pool if s.type_order < 624]
    mean = lambda x: sum(sum(s.counts.values()) for s in x) / len(x)
    assert abs(mean(sub) - mean(pool)) < 0.5
    assert abs(sum(s.target_chars for s in sub) / len(sub) - sum(s.target_chars for s in pool) / len(pool)) < 150


def test_note_profile_counts():
    n = load_gold()[0]
    prof = note_profile(n)
    assert prof["total"] == len(n.entities)
