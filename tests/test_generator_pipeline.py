"""End-to-end generation with a fake client. No network."""
import json

import pytest

from src.data.splits import load_fold, load_train_notes
from src.generation.anchor_selection import AnchorSelector
from src.generation.gemini_client import QuotaExhausted
from src.generation.generation_plan import build_plan
from src.generation.privacy_screening import mask_anchor_set
from src.generation.synthetic_generator import build_request_prompt, group_requests, request_seed, run_generation
from tests.conftest import FakeClient


def run(project, arm="style_anchored", client=None, **kw):
    client = client or FakeClient()
    kw.setdefault("confirm_anchors", True)
    out = run_generation(project["cfg"], 1, arm, execute=kw.pop("execute", True), output_root=project["out"], client=client,
                         gold_path=project["gold"], splits_dir=project["splits"], prompts_dir=project["prompts"], **kw)
    return out, client


def rows(project, name, arm="style_anchored"):
    p = project["out"] / "fold_1" / arm / name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def test_dry_run_makes_no_calls_and_writes_no_examples(project):
    class Explodes(FakeClient):
        def generate(self, *a):
            raise AssertionError("API called in dry run")
    out, _ = run(project, client=Explodes(), execute=False)
    assert out["mode"] == "dry_run" and out["guards_passed_on_all_prompts"]
    assert not rows(project, "examples.jsonl") and not rows(project, "requests.jsonl")


def test_execute_produces_all_required_artifacts(project):
    out, _ = run(project)
    assert out["cases_accepted"] == 15, out["rejection_reason_counts"]
    folder = project["out"] / "fold_1" / "style_anchored"
    for name in ["generation_manifest.csv", "examples.jsonl", "validation_report.json", "requests.jsonl", "case_log.jsonl", "raw_responses.jsonl", "run_config.json", "generation_plan.csv"]:
        assert (folder / name).exists(), name


def test_examples_carry_provenance_and_consistent_offsets(project):
    run(project)
    s = load_fold(1, project["splits"])
    for r in rows(project, "examples.jsonl"):
        assert r["generation_prompt_version"] == "v1" and r["fold_id"] == 1 and r["generation_model_version"] == "fake-001"
        assert not set(r["anchor_ids"]) & set(s.test_ids) and len(r["anchor_ids"]) == 3
        assert all(r["text"][e["start"] : e["end"]] == e["text"] for e in r["entities"])


def test_request_log_has_settings_but_no_text(project):
    run(project)
    for r in rows(project, "requests.jsonl"):
        assert {"model_id", "model_version", "seed", "prompt_sha256", "anchor_ids", "finish_reason", "prompt_tokens", "timestamp"} <= set(r)
        assert "prompt" not in r and "text" not in r and "raw_text" not in r


def test_case_log_has_no_note_text(project):
    run(project)
    for r in rows(project, "case_log.jsonl"):
        assert "text" not in r and "note" not in r


def test_rejected_cases_are_kept_and_retried(project):
    target = "F1-N00002"
    out, client = run(project, client=FakeClient(broken_ids=[target]))
    logs = [r for r in rows(project, "case_log.jsonl") if r["case_id"] == target]
    assert [r["status"] for r in logs] == ["rejected", "accepted"]
    assert "residual_date" in logs[0]["reasons"]
    assert [r["case_id"] for r in rows(project, "rejected.jsonl")] == [target]
    assert out["cases_accepted"] == 15 and out["rejection_reason_counts"]["residual_date"] == 1


def test_resume_skips_finished_cases(project):
    run(project, max_requests=2)
    done = len(rows(project, "requests.jsonl"))
    first_ids = [r["request_id"] for r in rows(project, "requests.jsonl")]
    out, client = run(project)
    assert out["cases_accepted"] == 15
    all_ids = [r["request_id"] for r in rows(project, "requests.jsonl")]
    assert all_ids[: done] == first_ids and len(set(all_ids)) == len(all_ids)
    third, c3 = run(project)
    assert c3.prompts == [] and third["cases_accepted"] == 15


def test_quota_stop_saves_work_and_can_resume(project):
    out, _ = run(project, client=FakeClient(raise_after=1, raise_with=QuotaExhausted("429")))
    assert "quota" in out["stop_reason"] and 0 < out["cases_accepted"] < 15
    out2, _ = run(project)
    assert out2["cases_accepted"] == 15


def test_api_error_stops_the_run_and_is_logged(project):
    out, _ = run(project, client=FakeClient(raise_after=0, raise_with=RuntimeError("boom 500")))
    assert out["stop_reason"] and rows(project, "errors.jsonl")


def test_cost_cap_stops_before_spending_more(project):
    project["cfg"]["limits"]["max_cost_usd"] = 0.0
    out, c = run(project)
    assert c.prompts == [] and "max_cost" in out["stop_reason"]


def test_max_requests_is_respected(project):
    _, c = run(project, max_requests=1)
    assert len(c.prompts) == 1


def test_non_json_and_truncated_responses_are_recorded_as_failures(project):
    from src.generation.gemini_client import GenerationResult

    class Bad(FakeClient):
        def generate(self, prompt, seed):
            self.prompts.append(prompt)
            return GenerationResult('{"cases": [{"case_id"', "MAX_TOKENS", "fake", "r", 1, 1, 0, 2, 0.1)
    out, _ = run(project, client=Bad(), max_requests=1)
    assert out["cases_accepted"] == 0 and out["rejection_reason_counts"]["output_truncated"] >= 1
    assert rows(project, "raw_responses.jsonl")


def test_arms_differ_only_by_the_anchor_block(project):
    s = load_fold(1, project["splits"])
    train = load_train_notes(s, project["gold"])
    sel = AnchorSelector(s, train)
    plan = build_plan(1, train, project["cfg"])
    g = group_requests(plan, 1, 3)[0]
    anchored, _ = build_request_prompt(g, "style_anchored", sel, project["cfg"], project["prompts"])
    plain, none = build_request_prompt(g, "non_anchored", sel, project["cfg"], project["prompts"])
    assert none == [] and "ESEMPI DI STILE" not in plain and "ESEMPI DI STILE" in anchored
    start, end = anchored.index("\nESEMPI DI STILE"), anchored.index("STILE E STRUTTURA")
    assert anchored[:start] + "\n" + anchored[end:] == plain


def test_both_arms_use_the_same_plan_and_seeds(project):
    _, a = run(project, "style_anchored")
    _, b = run(project, "non_anchored")
    assert a.seeds and sorted(a.seeds) == sorted(b.seeds)
    ida = [r["case_ids"] for r in rows(project, "requests.jsonl", "style_anchored")]
    idb = [r["case_ids"] for r in rows(project, "requests.jsonl", "non_anchored")]
    assert sorted(map(tuple, ida)) == sorted(map(tuple, idb))


def test_non_anchored_prompt_has_no_note_content(project):
    _, c = run(project, "non_anchored")
    assert c.prompts and not any("ZZSENTINEL" in p for p in c.prompts)
    assert all(r["anchor_ids"] == [] for r in rows(project, "requests.jsonl", "non_anchored"))


def test_masked_anchors_hide_names_and_places_in_sent_prompts(project):
    s = load_fold(1, project["splits"])
    train = {n.note_id: n for n in load_train_notes(s, project["gold"])}
    _, c = run(project)
    logged = rows(project, "requests.jsonl")
    assert len(c.prompts) == len(logged)
    for r in logged:
        for a in r["anchor_ids"]:
            for e in train[a].entities:
                assert sum(e["text"] in p for p in c.prompts) == 0 or e["type"] == "ETÀ"


def test_raw_anchors_require_explicit_confirmation(project):
    project["cfg"]["anchors"]["mode"] = "raw"
    with pytest.raises(PermissionError):
        run(project)
    out, _ = run(project, confirm_raw=True, max_requests=1)
    assert out["mode"] == "executed"


def test_mask_anchor_set_removes_cross_anchor_and_case_variants():
    from src.data.loaders import GoldNote
    a = GoldNote("gold_001", 1, "Referto di Mirella Fabbri a Pontedera.", ({"type": "NOME", "text": "Mirella Fabbri"}, {"type": "LUOGO/INDIRIZZO", "text": "Pontedera"}))
    b = GoldNote("gold_002", 2, "Nota: FABBRI trasferita a pontedera, poi Mirella.", ())
    texts = " ".join(m.text for m in mask_anchor_set([a, b])).lower()
    assert "fabbri" not in texts and "pontedera" not in texts and "mirella" not in texts


def test_seed_is_stable_and_independent_of_arm():
    assert request_seed(42, 1, 3, 1) == request_seed(42, 1, 3, 1)
    assert request_seed(42, 1, 3, 1) != request_seed(42, 1, 3, 2)


def test_model_echoing_the_spec_heading_is_tolerated_but_unknown_ids_are_not(project):
    from src.generation.gemini_client import GenerationResult

    class Echo(FakeClient):
        def generate(self, prompt, seed):
            r = super().generate(prompt, seed)
            d = json.loads(r.text)
            d["cases"][0]["case_id"] = "SCHEDA " + d["cases"][0]["case_id"]
            d["cases"][1]["case_id"] = "F9-N99999"
            r.text = json.dumps(d, ensure_ascii=False)
            return r
    out, _ = run(project, client=Echo(), max_requests=1)
    assert out["accepted_by_type"] and out["rejection_reason_counts"].get("case_missing_from_response", 0) == 1


def test_anchored_execution_requires_owner_approval(project):
    class Explodes(FakeClient):
        def generate(self, *a):
            raise AssertionError("API called without approval")
    with pytest.raises(PermissionError):
        run(project, client=Explodes(), confirm_anchors=False)
    assert not rows(project, "requests.jsonl")


def test_non_anchored_and_dry_runs_do_not_need_that_approval(project):
    out, _ = run(project, "non_anchored", confirm_anchors=False, max_requests=1)
    assert out["mode"] == "executed"
    out, _ = run(project, confirm_anchors=False, execute=False)
    assert out["mode"] == "dry_run"


def test_quota_errors_are_logged(project):
    out, _ = run(project, client=FakeClient(raise_after=0, raise_with=QuotaExhausted("429 daily")))
    assert "quota" in out["stop_reason"]
    kinds = {e["kind"] for e in rows(project, "errors.jsonl")}   # each concurrent worker may log its own quota error
    assert kinds == {"quota"}


def test_title_masking_removes_the_whole_name():
    from src.generation.privacy_screening import mask_residual
    text, _ = mask_residual("Seguita dal Dott. Verdi Neri in reparto, poi dalla Dott.ssa Bianchi.")
    assert "erdi" not in text and "Neri" not in text and "ianchi" not in text and text.count("[NOME]") == 2


def test_masking_keeps_clinical_abbreviations():
    from src.generation.privacy_screening import mask_residual
    assert "U.O." in mask_residual("Ricoverata in U.O. di cardiologia.")[0]
    assert "Q.Z." not in mask_residual("Vista da Q.Z. in reparto.")[0]


@pytest.mark.parametrize("k", range(1, 6))
def test_real_masked_anchors_leave_no_identifiers_or_patterns(k):
    """Reports counts only. Applies to every note a fold could use as an anchor."""
    from src.generation.anchor_selection import eligible_pool
    from src.generation.privacy_screening import count_identifying_strings, residual_pii_reasons
    pool = eligible_pool(load_train_notes(load_fold(k)))
    identifying = patterns = 0
    for n in pool:
        masked = mask_anchor_set([n])[0].text
        identifying += count_identifying_strings(masked, [n])
        patterns += bool(residual_pii_reasons(masked))
    assert identifying == 0, f"fold {k}: {identifying} identifying strings"
    assert patterns == 0, f"fold {k}: {patterns} notes with residual patterns"


def test_examples_carry_the_full_record_schema(project):
    run(project)
    need = {"record_id", "source_dataset", "source_record_id", "record_type", "contains_target_pii", "entity_annotations_available",
            "exact_duplicate_status", "near_duplicate_status", "fold_id", "partition", "generation_prompt_version", "generation_model_version", "permitted_for_training"}
    for r in rows(project, "examples.jsonl"):
        assert need <= set(r) and r["permitted_for_training"] is False


def test_run_config_records_sdk_model_and_tree_state(project):
    class WithInfo(FakeClient):
        def model_info(self):
            return {"name": "models/fake", "version": "1"}
    run(project, client=WithInfo())
    d = json.loads((project["out"] / "fold_1" / "style_anchored" / "run_config.json").read_text(encoding="utf-8"))
    assert d["sdk_version"] == "fake" and d["model_reported_by_api"]["version"] == "1"
    assert isinstance(d["git_dirty"], bool) and d["git_commit"] and d["template_sha256"] and d["plan_sha256"] and d["gold_sha256"]


def test_full_run_refuses_uncommitted_code(project, monkeypatch):
    import src.generation.synthetic_generator as g
    monkeypatch.setattr(g, "git_is_dirty", lambda: True)
    with pytest.raises(PermissionError):
        run(project, require_clean_tree=True)
    assert not rows(project, "requests.jsonl")


def test_cli_has_no_dirty_tree_bypass(project, monkeypatch):
    """Rewritten in v1.0.1. The old version asserted that --allow-dirty-tree and --max-requests skipped the guard."""
    import src.generation.synthetic_generator as g
    with pytest.raises(SystemExit):
        g.main(["--fold", "1", "--arm", "non_anchored", "--execute", "--allow-dirty-tree"])
    monkeypatch.setattr(g, "git_is_dirty", lambda: True)
    with pytest.raises(PermissionError):
        g.main(["--fold", "1", "--arm", "non_anchored", "--execute", "--max-requests", "5"])


def test_prompt_states_the_validation_rules(project):
    assert "VERIFICHE AUTOMATICHE" in (project["prompts"] / "generation_v1.txt").read_text(encoding="utf-8")


def test_manual_review_sample(project, tmp_path):
    import csv
    from src.generation.sample_review import write_sample
    run(project)
    a = write_sample(1, "style_anchored", n=6, output_root=project["out"], out_dir=tmp_path)
    with open(a, newline="", encoding="utf-8") as f:
        first = list(csv.DictReader(f))
    assert len(first) == 6 and {"reviewer_ok", "reviewer_notes", "text", "redacted_text"} <= set(first[0])
    assert len({r["record_type"] for r in first}) == 3
    with open(write_sample(1, "style_anchored", n=6, output_root=project["out"], out_dir=tmp_path), newline="", encoding="utf-8") as f:
        assert [r["record_id"] for r in csv.DictReader(f)] == [r["record_id"] for r in first]


def test_record_type_filter_requests_only_those_cases(project):
    report, client = run(project, client=FakeClient(), max_requests=1, record_types=["synthetic_negative"])
    asked = [c for p in client.prompts for c in __import__("re").findall(r"SCHEDA (F\d-N\d+)", p)]
    plan_types = {r["case_id"]: r["record_type"] for r in rows(project, "case_log.jsonl")}
    assert asked and all(plan_types[c] == "synthetic_negative" for c in asked)
    assert all("NOME 0, ETÀ 0, DATA 0, LUOGO/INDIRIZZO 0" in p for p in client.prompts)
    assert report["accepted_by_type"].keys() <= {"synthetic_negative"}


def test_record_type_filter_is_refused_for_official_runs(project, monkeypatch):
    import src.generation.synthetic_generator as g
    monkeypatch.setattr(g, "ROOT", project["tmp"])
    with pytest.raises(PermissionError, match="partial dataset"):
        g.run_generation(project["cfg"], 1, "non_anchored", execute=True, client=FakeClient(), output_root=project["tmp"] / "synthetic_data",
                         gold_path=project["gold"], splits_dir=project["splits"], prompts_dir=project["prompts"], record_types=["synthetic_negative"])


def test_unknown_record_types_are_refused(project):
    with pytest.raises(ValueError):
        run(project, record_types=["other"])
