"""Encoder baseline: BIO labels, windows and decoding. Pure Python, invented positions only."""
import pytest

from src.data.loaders import ROOT
from src.encoder.bio import LABEL_ID, LABELS, decode, merge_windows, token_labels, windows
from src.encoder.run import encoder_config_id
from src.evaluation.alignment import CATEGORIES
from src.training.experiments import EXPERIMENTS_CONFIG, load_yaml, resolve_experiment

# "Mario Rossi ha 69 anni a Pisa." tokenized as words; offsets only
OFFSETS = [(0, 5), (6, 11), (12, 14), (15, 17), (18, 22), (23, 24), (25, 29), (29, 30)]
SPANS = [(0, 11, "NOME"), (15, 17, "ETÀ"), (25, 29, "LUOGO/INDIRIZZO")]


def test_labels_cover_every_category_with_b_and_i():
    assert LABELS[0] == "O" and len(LABELS) == 1 + 2 * len(CATEGORIES)


def test_round_trip_word_aligned_spans():
    labels, stats = token_labels(OFFSETS, SPANS)
    assert [LABELS[i] for i in labels] == ["B-NOME", "I-NOME", "O", "B-ETÀ", "O", "O", "B-LUOGO/INDIRIZZO", "O"]
    assert stats == {"spans": 3, "spans_misaligned": 0, "spans_without_tokens": 0}
    assert decode(OFFSETS, labels) == sorted(SPANS)


def test_span_edge_inside_a_token_is_counted():
    _, stats = token_labels([(0, 6)], [(0, 2, "ETÀ")])   # e.g. an age glued to a suffix in one token
    assert stats["spans_misaligned"] == 1


def test_adjacent_entities_of_the_same_type_stay_apart():
    labels = [LABEL_ID["B-DATA"], LABEL_ID["B-DATA"]]
    assert decode([(0, 2), (3, 5)], labels) == [(0, 2, "DATA"), (3, 5, "DATA")]


def test_i_without_b_starts_an_entity_and_type_change_splits():
    labels = [LABEL_ID["I-NOME"], LABEL_ID["I-DATA"], LABEL_ID["O"]]
    assert decode([(0, 2), (3, 5), (6, 7)], labels) == [(0, 2, "NOME"), (3, 5, "DATA")]


def test_empty_offsets_are_skipped():
    assert decode([(0, 0), (0, 3)], [LABEL_ID["B-NOME"], LABEL_ID["B-NOME"]]) == [(0, 3, "NOME")]


@pytest.mark.parametrize("n", [0, 1, 509, 510, 511, 1000, 1400, 5000])
def test_windows_cover_every_token_within_the_size(n):
    w = windows(n, 510, 128)
    covered = {i for a, b in w for i in range(a, b)}
    assert covered == set(range(n)) and all(b - a <= 510 for a, b in w)
    assert all(w[k][1] - w[k + 1][0] == 128 for k in range(len(w) - 1))


def test_window_overlap_must_be_smaller_than_the_window():
    with pytest.raises(ValueError):
        windows(100, 10, 10)


def test_merge_takes_each_token_from_its_most_central_window():
    w = [(0, 4), (2, 6)]
    first, second = [1, 1, 1, 1], [2, 2, 2, 2]
    # token 2: margin 1 in the first window, 0 in the second; token 3: margin 0 then 1
    assert merge_windows(6, w, [first, second]) == [1, 1, 1, 2, 2, 2]


def test_config_and_run_ids():
    cfg = load_yaml(ROOT / "configs" / "encoder_bert_it.yaml")
    assert cfg["model"]["id"] == "dbmdz/bert-base-italian-xxl-cased" and len(cfg["model"]["revision"]) == 40
    assert cfg["windows"]["size"] + 2 <= 512 and cfg["data"]["experiments"] == ["E1", "E3"]
    assert cfg["data"]["gold_oversample"] == load_yaml(ROOT / "configs" / "training.yaml")["training"]["gold_oversample"]
    design = load_yaml(EXPERIMENTS_CONFIG)
    ids = {e: encoder_config_id(cfg, resolve_experiment(e, design)) for e in ("E1", "E3")}
    assert len(set(ids.values())) == 2 and all(len(v) == 12 for v in ids.values())
    assert encoder_config_id({**cfg, "training": {**cfg["training"], "seeds": [7]}}, resolve_experiment("E1", design)) == ids["E1"]


def test_v1_epoch_batches_are_unchanged():
    from src.encoder.run import batch_order
    t = {"epochs": 2, "batch_size": 4}
    b = batch_order(10, t, 42)
    assert len(b) == 6 and [len(x) for x in b] == [4, 4, 2, 4, 4, 2]
    assert sorted(i for x in b[:3] for i in x) == list(range(10)) and sorted(i for x in b[3:] for i in x) == list(range(10))


@pytest.mark.parametrize("n", [5, 16, 475, 930])
def test_v2_gives_every_arm_the_same_number_of_full_batches(n):
    from src.encoder.run import batch_order
    b = batch_order(n, {"updates": 175, "batch_size": 16}, 42)
    assert len(b) == 175 and all(len(x) == 16 for x in b)
    flat = [i for x in b for i in x]
    assert sorted(flat[:n]) == list(range(n))   # the first pass sees every window once
    assert batch_order(n, {"updates": 175, "batch_size": 16}, 42) == b != batch_order(n, {"updates": 175, "batch_size": 16}, 43)


def test_v2_config():
    v1, v2 = (load_yaml(ROOT / "configs" / f) for f in ("encoder_bert_it.yaml", "encoder_bert_it_v2.yaml"))
    assert v2["data"]["experiments"] == ["E1", "E2", "E3", "E8", "E9", "E10"] and v2["training"]["updates"] == 175 and "epochs" not in v2["training"]
    assert v2["model"]["id"] == v1["model"]["id"] and v2["model"]["revision"] == v1["model"]["revision"] and v2["windows"] == v1["windows"]
    assert set(v2["training"]["save_model_seeds"]) <= set(v2["training"]["seeds"]) and 42 in v2["training"]["seeds"]
    e1 = resolve_experiment("E1", load_yaml(EXPERIMENTS_CONFIG))
    assert encoder_config_id(v2, e1) != encoder_config_id(v1, e1)


def test_v1_run_ids_of_the_finished_runs_are_unchanged():
    cfg, design = load_yaml(ROOT / "configs" / "encoder_bert_it.yaml"), load_yaml(EXPERIMENTS_CONFIG)
    assert {e: encoder_config_id(cfg, resolve_experiment(e, design)) for e in ("E1", "E3")} == {"E1": "cc92f090aa20", "E3": "ccd62f5aee70"}
