"""Guide test 1: no test note in any training source."""
import csv
import json

import pytest

from src.data.loaders import ROOT
from src.data.splits import load_fold
from src.generation.post_audit import audit_fold
from src.generation.synthetic_generator import run_generation
from tests.conftest import FakeClient, make_gold


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("k", range(1, 6))
def test_real_manifests_permit_no_test_note_for_training(k):
    s = load_fold(k)
    rows = read(ROOT / "splits" / f"fold_{k}_manifest.csv")
    permitted = {r["record_id"] for r in rows if r["permitted_for_training"] == "True"}
    assert not permitted & set(s.test_ids)
    assert permitted == set(s.train_ids)


def test_real_manifest_excludes_all_historical_negatives_and_synthetic():
    rows = read(ROOT / "data_manifests" / "record_manifest.csv")
    assert rows and all(r["permitted_for_training"] == "False" for r in rows)


def test_post_audit_passes_a_clean_run(project):
    run_generation(project["cfg"], 1, "style_anchored", execute=True, confirm_anchors=True, output_root=project["out"], client=FakeClient(),
                   gold_path=project["gold"], splits_dir=project["splits"], prompts_dir=project["prompts"])
    r = audit_fold(project["cfg"], 1, "style_anchored", project["out"], project["gold"], project["splits"], project["prompts"])
    assert r["passed"], r["violations"]
    d = json.loads((project["out"] / "fold_1" / "style_anchored" / "validation_report.json").read_text(encoding="utf-8"))
    assert d["permitted_for_training"] is True


def _append_example(project, text):
    p = project["out"] / "fold_1" / "style_anchored" / "examples.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"record_id": "planted", "text": text, "redacted_text": text, "entities": []}) + "\n")


def _run(project):
    run_generation(project["cfg"], 1, "style_anchored", execute=True, confirm_anchors=True, output_root=project["out"], client=FakeClient(),
                   gold_path=project["gold"], splits_dir=project["splits"], prompts_dir=project["prompts"])


def test_post_audit_fails_on_a_copied_test_note(project):
    _run(project)
    idx = int(project["specs"][0].test_ids[0].split("_")[1])
    _append_example(project, make_gold()[idx]["text"])
    r = audit_fold(project["cfg"], 1, "style_anchored", project["out"], project["gold"], project["splits"], project["prompts"])
    assert not r["passed"] and any(v["check"] == "synthetic_equals_test_note" for v in r["violations"])
    d = json.loads((project["out"] / "fold_1" / "style_anchored" / "validation_report.json").read_text(encoding="utf-8"))
    assert d["permitted_for_training"] is False


def test_post_audit_fails_on_a_test_note_used_as_anchor(project):
    _run(project)
    p = project["out"] / "fold_1" / "style_anchored" / "requests.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    rows[0]["anchor_ids"] = [project["specs"][0].test_ids[0]]
    p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    r = audit_fold(project["cfg"], 1, "style_anchored", project["out"], project["gold"], project["splits"], project["prompts"])
    assert any(v["check"] == "test_note_used_as_anchor" for v in r["violations"])


def test_post_audit_never_deletes_records(project):
    _run(project)
    p = project["out"] / "fold_1" / "style_anchored" / "examples.jsonl"
    idx = int(project["specs"][0].test_ids[0].split("_")[1])
    _append_example(project, make_gold()[idx]["text"])
    before = p.read_text(encoding="utf-8")
    audit_fold(project["cfg"], 1, "style_anchored", project["out"], project["gold"], project["splits"], project["prompts"])
    assert p.read_text(encoding="utf-8") == before


def test_post_audit_catches_test_text_a_prompt_cannot_explain(project, monkeypatch):
    """If a prompt contained test-note text not present in its anchors, the audit must fail."""
    _run(project)
    import src.generation.post_audit as pa
    idx = int(project["specs"][0].test_ids[0].split("_")[1])
    leaked = " ".join(f"unico{i}" for i in range(12))
    gold = make_gold()
    gold[idx]["text"] = leaked + " " + gold[idx]["text"]
    real = pa.build_request_prompt
    monkeypatch.setattr(pa, "build_request_prompt", lambda *a, **k: (real(*a, **k)[0] + " " + leaked, real(*a, **k)[1]))
    other = project["tmp"] / "gold_changed.json"
    other.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(pa, "load_train_notes", lambda spec, path: real_train(spec, project))
    r = pa.audit_fold(project["cfg"], 1, "style_anchored", project["out"], other, project["splits"], project["prompts"])
    assert any(v["check"] == "test_note_text_in_prompt" for v in r["violations"])


def real_train(spec, project):
    from src.data.splits import load_train_notes
    return load_train_notes(spec, project["gold"])


def _test_note_with_name(project):
    gold = make_gold()
    idx = next(int(t.split("_")[1]) for t in project["specs"][0].test_ids if int(t.split("_")[1]) % 2 == 0)
    ents = {e["type"]: e["text"] for e in gold[idx]["entities"]}
    return f"gold_{idx:03d}", ents["LUOGO/INDIRIZZO"], ents["NOME"]


def test_post_audit_reports_test_entities_by_policy(project):
    """Entity-match policy v1: a full place string is a candidate match; a single-token name is counted apart; not blocking."""
    _run(project)
    note_id, place, name = _test_note_with_name(project)
    _append_example(project, f"Ricovero presso {place} . Visita di controllo per {name} .")
    r = audit_fold(project["cfg"], 1, "style_anchored", project["out"], project["gold"], project["splits"], project["prompts"])
    assert r["passed"], r["violations"]
    assert r["entity_candidate_matches"] == 1
    match = r["entity_check"]["candidate_matches"][0]
    assert match["test_notes"] == [note_id] and match["record_ids"] == ["planted"] and match["type"] == "LUOGO/INDIRIZZO"
    assert r["entity_check"]["by_class"]["single_token_NOME"]["matched"] == 1
    assert place not in json.dumps(r, ensure_ascii=False) and name not in json.dumps(r, ensure_ascii=False)


def test_candidate_match_stops_the_fold_when_configured(project):
    _run(project)
    _, place, _ = _test_note_with_name(project)
    _append_example(project, f"Ricovero presso {place} .")
    cfg = dict(project["cfg"], post_audit={"candidate_match_blocking": True})
    r = audit_fold(cfg, 1, "style_anchored", project["out"], project["gold"], project["splits"], project["prompts"])
    assert not r["passed"] and any(v["check"] == "test_entity_in_synthetic" for v in r["violations"])
