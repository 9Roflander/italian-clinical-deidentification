"""Fold isolation, membership, merging and manifest validation, on invented notes."""
import json

import pytest

from src.data.splits import LeakageError
from src.training.dataset import DatasetError, assert_no_held_out, build_fold_dataset, load_synthetic_source, matched_case_ids, select_records, Example
from tests.support.training_project import SENTINEL, make_project, write_run

SYN = {"arm": "non_anchored", "n_named": 4, "n_name_free": 2, "n_negative": 1}


@pytest.fixture()
def project(tmp_path):
    return make_project(tmp_path)


def build(project, fold=1, synthetic=None, **kw):
    return build_fold_dataset("T", fold, 5, synthetic, project["root"], project["gold"], project["splits"], repo_root=project["tmp"], **kw)


@pytest.mark.parametrize("fold", [1, 2, 3, 4, 5])
def test_gold_only_uses_exactly_the_training_notes_and_never_held_out_text(project, fold):
    spec = project["specs"][fold]
    data = build(project, fold)
    assert {e.source_id for e in data.examples} == set(spec.train_ids)
    assert not {e.source_id for e in data.examples} & set(spec.test_ids)
    for test_id in spec.test_ids:
        marker = f"{SENTINEL}{int(test_id[5:]):03d}"
        assert not any(marker in e.input_text or marker in e.target_text for e in data.examples), test_id


def test_every_note_is_held_out_once_and_trained_in_the_other_folds(project):
    for i in range(25):
        note = f"gold_{i:03d}"
        trained_in = [k for k in range(1, 6) if note in {e.source_id for e in build(project, k).examples}]
        assert len(trained_in) == 4


def test_gold_is_oversampled_after_the_split(project):
    data = build(project)
    counts = {}
    for e in data.examples:
        counts[e.source_id] = counts.get(e.source_id, 0) + 1
    assert set(counts.values()) == {5} and {e.repeat for e in data.examples} == set(range(5))
    assert data.manifest["counts"] == {"gold": 5 * len(project["specs"][1].train_ids)}


def test_oversample_value_is_recorded_and_validated(project):
    assert build_fold_dataset("T", 1, 2, None, project["root"], project["gold"], project["splits"]).manifest["gold_oversample"] == 2
    with pytest.raises(DatasetError):
        build_fold_dataset("T", 1, 0, None, project["root"], project["gold"], project["splits"])


def test_gold_plus_synthetic_merges_the_requested_records(project):
    write_run(project, 1, "non_anchored")
    data = build(project, synthetic=SYN)
    assert data.manifest["counts"]["synthetic_named"] == 4 and data.manifest["counts"]["synthetic_name_free"] == 2 and data.manifest["counts"]["synthetic_negative"] == 1
    assert data.manifest["total_examples"] == 5 * len(project["specs"][1].train_ids) + 7
    assert all(json.loads(e.target_text) for e in data.examples)


def test_negative_targets_have_no_entities(project):
    write_run(project, 1, "non_anchored")
    for e in build(project, synthetic=SYN).examples:
        if e.record_type == "synthetic_negative":
            assert {s["label"] for s in json.loads(e.target_text)} == {"O"}


def test_negative_ablation_changes_only_negatives(project):
    write_run(project, 1, "non_anchored")
    with_neg = {e.example_id for e in build(project, synthetic=SYN).examples}
    without = {e.example_id for e in build(project, synthetic={**SYN, "n_negative": 0}).examples}
    assert without < with_neg and all("00009" in x or "0001" in x for x in with_neg - without)


def test_selection_is_nested_by_plan_order(project):
    write_run(project, 1, "non_anchored")
    small = {e.example_id for e in build(project, synthetic={**SYN, "n_named": 2, "n_name_free": 1}).examples}
    large = {e.example_id for e in build(project, synthetic=SYN).examples}
    assert small < large


def test_entity_mix_arms_have_equal_volume_and_different_composition(project):
    write_run(project, 1, "non_anchored")
    named_only = build(project, synthetic={"arm": "non_anchored", "n_named": 6, "n_name_free": 0, "n_negative": 2}).manifest["counts"]
    mixed = build(project, synthetic={"arm": "non_anchored", "n_named": 4, "n_name_free": 2, "n_negative": 2}).manifest["counts"]
    assert named_only["synthetic_named"] == 6 and "synthetic_name_free" not in named_only and mixed["synthetic_named"] == 4 and mixed["synthetic_name_free"] == 2
    assert sum(v for k, v in named_only.items() if k != "gold") == sum(v for k, v in mixed.items() if k != "gold")


def test_matched_arms_train_on_the_same_cases(project):
    write_run(project, 1, "non_anchored", skip_orders={1})
    write_run(project, 1, "style_anchored", skip_orders={2, 7})
    n = {"n_named": 3, "n_name_free": 1, "n_negative": 1}
    a = build(project, synthetic={"arm": "non_anchored", "matched_with": "style_anchored", **n})
    b = build(project, synthetic={"arm": "style_anchored", "matched_with": "non_anchored", **n})
    ids = lambda d: {e.source_id for e in d.examples if e.source == "synthetic"}
    assert ids(a) == ids(b) and not {"F1-N00001", "F1-N00002", "F1-N00007"} & ids(a)
    assert a.manifest["synthetic"]["matched_case_count"] == b.manifest["synthetic"]["matched_case_count"] == 8


def test_shortfall_is_an_error_unless_allowed(project):
    write_run(project, 1, "non_anchored", skip_orders={0, 1, 2})
    with pytest.raises(DatasetError, match="only 3"):
        build(project, synthetic=SYN)
    assert build(project, synthetic={**SYN, "allow_shortfall": True}).manifest["counts"]["synthetic_named"] == 3


def fault(project, mutate, arm="non_anchored", fold=1, **kw):
    write_run(project, fold, arm, mutate=mutate, **kw)
    return build(project, fold, synthetic={**SYN, "arm": arm})


def test_a_record_from_another_fold_is_refused(project):
    def change(records, manifest):
        records[0]["fold_id"] = 2
    with pytest.raises(LeakageError):
        fault(project, change)


def test_a_record_id_from_another_fold_is_refused(project):
    def change(records, manifest):
        records[0]["record_id"] = "syn_f2_non_anchored_00000"
        manifest[0]["record_id"] = "syn_f2_non_anchored_00000"
    with pytest.raises(LeakageError):
        fault(project, change)


def test_a_case_id_from_another_fold_is_refused(project):
    def change(records, manifest):
        records[0]["source_record_id"] = "F2-N00000"
    with pytest.raises(LeakageError):
        fault(project, change)


def test_a_run_folder_describing_another_fold_is_refused(project):
    run = write_run(project, 1, "non_anchored")
    config = json.loads((run / "run_config.json").read_text())
    config["fold"] = 2
    (run / "run_config.json").write_text(json.dumps(config))
    with pytest.raises(LeakageError):
        build(project, synthetic=SYN)


def test_a_held_out_note_used_as_anchor_is_refused(project):
    test_note = project["specs"][1].test_ids[0]
    with pytest.raises(LeakageError, match="anchor"):
        fault(project, None, arm="style_anchored", anchors=[test_note])


def test_an_unknown_anchor_is_refused(project):
    with pytest.raises(LeakageError, match="anchor"):
        fault(project, None, arm="style_anchored", anchors=["gold_999"])


def test_anchors_in_the_non_anchored_arm_are_refused(project):
    with pytest.raises(DatasetError, match="non-anchored"):
        fault(project, None, arm="non_anchored", anchors=[project["specs"][1].train_ids[0]])


def test_a_failed_or_missing_audit_is_refused(project):
    write_run(project, 1, "non_anchored", audit="failed")
    with pytest.raises(DatasetError, match="audit"):
        build(project, synthetic=SYN)


def test_a_record_not_accepted_in_the_manifest_is_refused(project):
    def change(records, manifest):
        manifest[0]["status"] = "rejected"
    with pytest.raises(DatasetError, match="not accepted"):
        fault(project, change)


def test_a_permitted_record_missing_from_examples_is_refused(project):
    def change(records, manifest):
        del records[0]
    with pytest.raises(DatasetError, match="missing"):
        fault(project, change)


def test_duplicate_records_are_refused(project):
    def change(records, manifest):
        records.append(dict(records[0]))
    with pytest.raises(DatasetError, match="twice"):
        fault(project, change)


def test_two_records_with_identical_text_are_refused(project):
    def change(records, manifest):
        records[1]["text"], records[1]["redacted_text"], records[1]["entities"] = records[0]["text"], records[0]["redacted_text"], records[0]["entities"]
    with pytest.raises(DatasetError, match="share one input"):
        fault(project, change)


@pytest.mark.parametrize("edit,message", [
    (lambda r: r.update(redacted_text="altro"), "differs"),
    (lambda r: r["entities"][0].update(end=9999), "invalid target"),
    (lambda r: r["entities"][0].update(type="PERSONA"), "invalid target"),
    (lambda r: r.update(record_type="other"), "unknown record type"),
])
def test_malformed_records_are_refused_not_dropped(project, edit, message):
    def change(records, manifest):
        edit(records[0])
    with pytest.raises(DatasetError, match=message):
        fault(project, change)


def test_a_negative_with_entities_is_refused(project):
    def change(records, manifest):
        records[-1]["entities"] = records[0]["entities"]
    with pytest.raises(DatasetError, match="negative"):
        fault(project, change)


def test_a_run_from_another_gold_file_is_refused(project):
    run = write_run(project, 1, "non_anchored")
    config = json.loads((run / "run_config.json").read_text())
    config["gold_sha256"] = "other"
    (run / "run_config.json").write_text(json.dumps(config))
    with pytest.raises(DatasetError, match="different gold"):
        build(project, synthetic=SYN)


@pytest.mark.parametrize("name", ["examples.jsonl", "generation_manifest.csv", "validation_report.json", "run_config.json"])
def test_missing_run_files_are_refused(project, name):
    run = write_run(project, 1, "non_anchored")
    (run / name).unlink()
    with pytest.raises(DatasetError, match="missing"):
        build(project, synthetic=SYN)


def test_official_datasets_refuse_development_generation(project):
    write_run(project, 1, "non_anchored", official=False)
    with pytest.raises(DatasetError, match="official"):
        build(project, synthetic=SYN, require_official=True)


def test_official_datasets_refuse_a_development_folder_even_if_marked_official(project):
    write_run(project, 1, "non_anchored", official=True, root=project["tmp"] / "experiments/development/x")
    with pytest.raises(DatasetError, match="official"):
        build_fold_dataset("T", 1, 5, SYN, project["tmp"] / "experiments/development/x", project["gold"], project["splits"], require_official=True, repo_root=project["tmp"])


def test_official_generation_passes_for_official_datasets(project):
    write_run(project, 1, "non_anchored", official=True)
    assert build(project, synthetic=SYN, require_official=True).manifest["synthetic"]["official_generation_run"] is True


def test_a_changed_gold_file_is_refused(project):
    project["gold"].write_text(project["gold"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(LeakageError):
        build(project)


def test_assert_no_held_out_catches_late_additions(project):
    spec = project["specs"][1]
    with pytest.raises(LeakageError):
        assert_no_held_out([Example("x", "gold", spec.test_ids[0], 0, "gold", "t", "t")], spec)
    with pytest.raises(LeakageError):
        assert_no_held_out([Example("syn_f2_a_00000", "synthetic", "F2-N00000", 0, "synthetic_named", "t", "t")], spec)


def test_manifest_records_ids_and_provenance_without_text(project):
    write_run(project, 1, "non_anchored")
    manifest = build(project, synthetic=SYN).manifest
    assert manifest["gold_sha256"] == project["specs"][1].gold_sha256 and manifest["held_out_note_ids"] == list(project["specs"][1].test_ids)
    assert manifest["synthetic"]["arm"] == "non_anchored" and len(manifest["synthetic"]["examples_file_sha256"]) == 64 and manifest["synthetic"]["official_generation_run"] is False
    assert len(manifest["examples"]) == manifest["total_examples"] and all(len(x["input_sha256"]) == 64 and len(x["target_sha256"]) == 64 for x in manifest["examples"])
    dumped = json.dumps(manifest, ensure_ascii=False)
    assert SENTINEL not in dumped and "Zelinda" not in dumped and "Borgoverde" not in dumped


def test_rebuilding_gives_an_identical_dataset(project):
    write_run(project, 1, "non_anchored")
    a, b = build(project, synthetic=SYN), build(project, synthetic=SYN)
    assert a.manifest == b.manifest and a.examples == b.examples


def test_matched_case_ids_is_an_intersection(project):
    spec = project["specs"][1]
    write_run(project, 1, "non_anchored", skip_orders={0})
    write_run(project, 1, "style_anchored", skip_orders={1})
    a = load_synthetic_source(project["root"], 1, "non_anchored", spec, repo_root=project["tmp"])
    b = load_synthetic_source(project["root"], 1, "style_anchored", spec, repo_root=project["tmp"])
    assert matched_case_ids([a, b]) == {f"F1-N{i:05d}" for i in range(2, 11)}
    assert len(select_records(a, 2, 1, 1, matched_case_ids([a, b]))) == 4
