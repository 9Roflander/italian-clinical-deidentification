"""D1: generated outputs must not block a run, and changed critical source must. Temporary git repositories, invented data."""
import json
import pathlib
import subprocess

import pytest

import src.evaluation.runner as runner
import src.generation.synthetic_generator as generator
from src.data.loaders import ROOT, GoldNote
from src.evaluation.alignment import render_inline
from src.evaluation.gold_verification import VerifiedGold
from src.evaluation.runner import EvaluationInputError, load_config, text_sha256, write_run
from src.generation.gemini_client import QuotaExhausted
from src.infrastructure.provenance import (ARCHIVE, CACHE, CRITICAL, DOCUMENTATION, OUTPUT, UNCLASSIFIED, check_source_integrity, classify_path, is_official_output)
from tests.conftest import FakeClient


def git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args], cwd=repo, capture_output=True, text=True, check=True)


def commit_all(repo, message="source"):
    git(repo, "add", "--", ".")
    git(repo, "commit", "-q", "-m", message)


def make_repo(path: pathlib.Path) -> pathlib.Path:
    files = {"src/module.py": "VALUE = 1\n", "prompts/template.txt": "prompt\n", "configs/settings.yaml": "a: 1\n", "splits/fold_1.json": "{}\n",
             "data_manifests/fold_assignments.csv": "a,b\n", "tests/test_x.py": "\n", "audit/scripts/check.py": "\n", "audit/reference/old.py": "\n",
             "requirements-training.txt": "x==1\n", ".gitattributes": "* text=auto eol=lf\n", "original_work/src/old.py": "\n", "original_work/notes.md": "old\n",
             "reports/report.md": "r\n", "README.md": "r\n", ".gitignore": "predictions_secure/\ndata_secure/\n__pycache__/\nprompts/rendered/\nkey.txt\n"}
    for name, text in files.items():
        p = path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():   # a miniature project may already hold real fold files for the run
            p.write_text(text, encoding="utf-8")
    git(path, "init", "-q")
    commit_all(path)
    return path


@pytest.fixture()
def repo(tmp_path):
    return make_repo(tmp_path)


def reasons(root):
    return {p: why for p, why in check_source_integrity(root).blocking}


@pytest.mark.parametrize("path,kind", [
    ("src/evaluation/scoring.py", CRITICAL), ("src/new/deep/file.py", CRITICAL), ("prompts/generation_v1.txt", CRITICAL), ("configs/evaluation.yaml", CRITICAL),
    ("splits/fold_1.json", CRITICAL), ("splits/gold_v1/fold_1_manifest.csv", CRITICAL), ("data_manifests/fold_assignments.csv", CRITICAL), ("tests/test_a.py", CRITICAL),
    ("audit/scripts/v1.py", CRITICAL), ("audit/reference/evaluator_v1_0/src/evaluation/scoring.py", CRITICAL), ("requirements-generation.txt", CRITICAL),
    ("requirements.lock", CRITICAL), (".gitattributes", CRITICAL), (".gitignore", CRITICAL), ("original_work/src/train_evaluate_cv_v4.py", CRITICAL),
    ("synthetic_data/fold_1/non_anchored/requests.jsonl", OUTPUT), ("synthetic_data/fold_1/non_anchored/generation_plan.csv", OUTPUT), ("synthetic_data/.gitkeep", OUTPUT),
    ("experiments/clean_cv/run/run_manifest.json", OUTPUT), ("experiments/clean_cv/run/evaluation_summary.md", OUTPUT), ("audit/results/x.json", OUTPUT),
    ("reports/PRE_EXPERIMENT_AUDIT.md", DOCUMENTATION), ("README.md", DOCUMENTATION), ("audit/KNOWN_PROBLEMS.md", DOCUMENTATION), ("audit/gold_dataset_statistics.csv", DOCUMENTATION),
    ("69_main_long.pdf", DOCUMENTATION), ("original_work/report.tex", ARCHIVE), ("original_work/data/old.json", ARCHIVE),
    ("src/__pycache__/x.pyc", CACHE), (".DS_Store", CACHE),
    ("synthetic_data/fold_1/run.py", UNCLASSIFIED), ("experiments/clean_cv/run/patch.sh", UNCLASSIFIED), ("helper.py", UNCLASSIFIED), ("data_secure/gold/gold.jsonl", UNCLASSIFIED),
    ("predictions_secure/run/predictions_oof.jsonl", UNCLASSIFIED), ("key.txt", UNCLASSIFIED), ("newfolder/data.json", UNCLASSIFIED),
])
def test_every_path_has_the_intended_class(path, kind):
    assert classify_path(path) == kind


def test_a_committed_repository_passes(repo):
    report = check_source_integrity(repo)
    assert report.clean and report.commit and len(report.critical_tree_sha256) == 64 and not report.blocking


@pytest.mark.parametrize("name", ["src/module.py", "prompts/template.txt", "configs/settings.yaml", "splits/fold_1.json", "data_manifests/fold_assignments.csv",
                                  "tests/test_x.py", "audit/scripts/check.py", "audit/reference/old.py", "requirements-training.txt", ".gitattributes", ".gitignore", "original_work/src/old.py"])
def test_a_modified_critical_file_blocks(repo, name):
    with open(repo / name, "a", encoding="utf-8") as f:
        f.write("changed\n")
    assert reasons(repo) == {name: "modified_critical_file"}


def test_a_staged_change_blocks(repo):
    (repo / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    git(repo, "add", "--", "src/module.py")
    assert reasons(repo) == {"src/module.py": "modified_critical_file"}


def test_a_deleted_critical_file_blocks(repo):
    (repo / "src/module.py").unlink()
    assert reasons(repo) == {"src/module.py": "modified_critical_file"}


def test_a_new_untracked_source_file_blocks(repo):
    (repo / "src/deep").mkdir()
    (repo / "src/deep/new.py").write_text("\n", encoding="utf-8")
    assert reasons(repo) == {"src/deep/new.py": "untracked_critical_file"}


def test_a_new_dependency_lock_blocks(repo):
    (repo / "requirements.lock").write_text("x==1\n", encoding="utf-8")
    assert reasons(repo) == {"requirements.lock": "untracked_critical_file"}


def test_not_every_untracked_file_passes(repo):
    (repo / "helper.py").write_text("\n", encoding="utf-8")
    (repo / "synthetic_data/fold_1").mkdir(parents=True)
    (repo / "synthetic_data/fold_1/requests.py").write_text("\n", encoding="utf-8")
    assert reasons(repo) == {"helper.py": "unclassified_path", "synthetic_data/fold_1/requests.py": "unclassified_path"}


def test_a_critical_file_hidden_by_gitignore_blocks(repo):
    (repo / ".git/info/exclude").write_text("src/hidden.py\n", encoding="utf-8")
    (repo / "src/hidden.py").write_text("\n", encoding="utf-8")
    assert reasons(repo) == {"src/hidden.py": "ignored_critical_file"}


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_disabling_change_tracking_cannot_hide_an_edit(repo, flag):
    git(repo, "update-index", flag, "src/module.py")
    (repo / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert reasons(repo) == {"src/module.py": "git_change_tracking_disabled"}


def test_expected_ignored_files_do_not_block(repo):
    (repo / "src/__pycache__").mkdir()
    (repo / "src/__pycache__/module.pyc").write_text("x", encoding="utf-8")
    (repo / "prompts/rendered").mkdir()
    (repo / "prompts/rendered/sample.txt").write_text("x", encoding="utf-8")
    assert check_source_integrity(repo).clean


def test_outputs_and_documents_do_not_block_and_are_counted(repo):
    (repo / "synthetic_data/fold_1").mkdir(parents=True)
    (repo / "synthetic_data/fold_1/requests.jsonl").write_text("{}\n", encoding="utf-8")
    (repo / "experiments/clean_cv/run").mkdir(parents=True)
    (repo / "experiments/clean_cv/run/run_manifest.json").write_text("{}\n", encoding="utf-8")
    (repo / "reports/report.md").write_text("changed\n", encoding="utf-8")
    (repo / "original_work/notes.md").write_text("changed\n", encoding="utf-8")
    report = check_source_integrity(repo)
    assert report.clean and report.non_blocking_counts == {"output": 2, "documentation": 1, "archive": 1}
    assert report.to_manifest()["non_blocking_changes"] == {"archive": 1, "documentation": 1, "output": 2}


def test_outside_git_or_without_a_commit_fails_closed(tmp_path):
    (tmp_path / "plain").mkdir()
    assert not check_source_integrity(tmp_path / "plain").clean
    (tmp_path / "empty").mkdir()
    git(tmp_path / "empty", "init", "-q")
    assert not check_source_integrity(tmp_path / "empty").clean


def test_tree_hash_follows_critical_files_only(repo):
    before = check_source_integrity(repo).critical_tree_sha256
    (repo / "reports/report.md").write_text("changed\n", encoding="utf-8")
    commit_all(repo, "docs")
    assert check_source_integrity(repo).critical_tree_sha256 == before
    (repo / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    commit_all(repo, "code")
    assert check_source_integrity(repo).critical_tree_sha256 != before


def test_official_output_roots(tmp_path):
    for inside in ("synthetic_data", "synthetic_data/fold_1", "experiments/clean_cv", "experiments/baselines/x", "experiments/ablations", "experiments/independent_test"):
        assert is_official_output(tmp_path / inside, tmp_path), inside
    for outside in ("experiments/development/x", "experiments", "other"):
        assert not is_official_output(tmp_path / outside, tmp_path), outside


def test_integrity_output_holds_paths_and_counts_only(repo):
    (repo / "src/module.py").write_text("ZZSECRETVALUE = 2\n", encoding="utf-8")
    assert "ZZSECRETVALUE" not in json.dumps(check_source_integrity(repo).to_manifest())


# Generator, through run_generation with a fake client.

@pytest.fixture()
def gen(project, monkeypatch):
    """The miniature project as a committed repository, so synthetic_data/ below it is an official output."""
    root = project["tmp"]
    make_repo_files = make_repo(root)
    monkeypatch.setattr(generator, "ROOT", root)
    return {**project, "root": make_repo_files, "official": root / "synthetic_data"}


def execute(gen, arm="non_anchored", client=None, output_root=None, **kw):
    return generator.run_generation(gen["cfg"], 1, arm, execute=True, confirm_anchors=True, client=client or FakeClient(), output_root=output_root or gen["official"],
                                    gold_path=gen["gold"], splits_dir=gen["splits"], prompts_dir=gen["prompts"], **kw)


def config_of(gen, arm="non_anchored", root=None):
    return json.loads(((root or gen["official"]) / "fold_1" / arm / "run_config.json").read_text(encoding="utf-8"))


def jsonl(gen, name, arm="non_anchored"):
    p = gen["official"] / "fold_1" / arm / name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def test_second_run_starts_with_existing_generated_outputs(gen):
    first = execute(gen, "non_anchored")
    assert first["cases_accepted"] == first["planned_cases"]
    assert subprocess.run(["git", "status", "--porcelain"], cwd=gen["root"], capture_output=True, text=True).stdout.strip(), "outputs must exist for this test to mean anything"
    second = execute(gen, "style_anchored")
    assert second["cases_accepted"] == second["planned_cases"]
    c = config_of(gen, "style_anchored")
    assert c["official_run"] is True and c["git_dirty"] is False and c["source_integrity"]["clean"] is True
    assert c["source_integrity"]["non_blocking_changes"]["output"] > 0


def test_modifying_source_still_blocks_an_official_run(gen):
    execute(gen, "non_anchored")
    (gen["root"] / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="src/module.py"):
        execute(gen, "style_anchored")
    assert not jsonl(gen, "requests.jsonl", "style_anchored")


def test_modifying_a_prompt_template_blocks_an_official_run(gen):
    (gen["root"] / "prompts/template.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="prompts/template.txt"):
        execute(gen)


def test_resume_works_after_a_quota_stop(gen):
    stopped = execute(gen, client=FakeClient(raise_after=2, raise_with=QuotaExhausted("quota")))
    assert "quota" in stopped["stop_reason"] and 0 < stopped["cases_accepted"] < stopped["planned_cases"]
    done_first = {r["case_id"] for r in jsonl(gen, "case_log.jsonl") if r["status"] == "accepted"}
    client = FakeClient()
    resumed = execute(gen, client=client)
    assert resumed["stop_reason"] is None and resumed["cases_accepted"] == resumed["planned_cases"]
    asked_again = {c for p in client.prompts for c in __import__("re").findall(r"SCHEDA (F\d-N\d+)", p)}
    assert not asked_again & done_first, "finished cases must not be requested again"
    examples = [r["source_record_id"] for r in jsonl(gen, "examples.jsonl")]
    assert len(examples) == len(set(examples)) == resumed["planned_cases"]
    assert len(jsonl(gen, "run_config_history.jsonl")) == 2


def test_resume_refuses_mixing_two_committed_versions(gen):
    execute(gen, client=FakeClient(raise_after=2, raise_with=QuotaExhausted("quota")))
    (gen["root"] / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    git(gen["root"], "add", "--", "src/module.py")
    git(gen["root"], "commit", "-q", "-m", "changed source")
    with pytest.raises(PermissionError, match="mix two versions"):
        execute(gen)


def test_resume_is_allowed_after_a_documentation_commit(gen):
    execute(gen, client=FakeClient(raise_after=2, raise_with=QuotaExhausted("quota")))
    (gen["root"] / "reports/report.md").write_text("changed\n", encoding="utf-8")
    git(gen["root"], "add", "--", "reports/report.md")
    git(gen["root"], "commit", "-q", "-m", "docs")
    assert execute(gen)["stop_reason"] is None


def test_a_request_cap_does_not_bypass_the_official_guard(gen):
    (gen["root"] / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        execute(gen, max_requests=1)


def test_development_run_with_changed_source_is_allowed_and_recorded(gen):
    (gen["root"] / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    dev = gen["root"] / "experiments/development/test"
    execute(gen, output_root=dev, max_requests=1)
    c = config_of(gen, root=dev)
    assert c["official_run"] is False and c["git_dirty"] is True and c["source_integrity"]["blocking_paths"] == [{"path": "src/module.py", "reason": "modified_critical_file"}]


def test_dry_run_is_never_blocked(gen):
    (gen["root"] / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert generator.run_generation(gen["cfg"], 1, "non_anchored", output_root=gen["official"], gold_path=gen["gold"], splits_dir=gen["splits"], prompts_dir=gen["prompts"])["mode"] == "dry_run"


# Evaluator, through write_run on invented notes.

@pytest.fixture()
def ev(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    monkeypatch.setattr(runner, "ROOT", root)
    notes, spans, fold_of = {}, {}, {}
    for k in range(10):
        text = f"Zelinda{chr(97 + k)} ha {30 + k} anni."
        s = [(0, 8, "NOME"), (12, 14, "ETÀ")]
        i = f"mini_{k:02d}"
        notes[i] = GoldNote(i, k, text, tuple({"start": a, "end": b, "type": c, "text": text[a:b], "annotated_text": text[a:b]} for a, b, c in s), redacted_text=render_inline(text, s))
        spans[i], fold_of[i] = s, k % 5 + 1
    gold = VerifiedGold(notes, tuple(notes), fold_of, spans, "mini-sha", "v2", {k: "x" for k in range(1, 6)})
    cfg = load_config()
    cfg["bootstrap"]["iterations"] = 2000
    return {"root": root, "gold": gold, "cfg": cfg, "official": root / "experiments/clean_cv", "secure": root / "predictions_secure"}


def records(ev, experiment):
    g = ev["gold"]
    return [{"experiment": experiment, "model": "m", "model_revision": "r", "checkpoint": "c", "config_id": "cfg", "seed": 1, "note_id": i, "fold_id": g.fold_of[i],
             "input_sha256": text_sha256(g.notes[i].text), "output_format": "inline", "raw_model_output": g.notes[i].redacted_text} for i in g.order]


def evaluate(ev, experiment, output_root=None):
    return write_run(records(ev, experiment), ev["gold"], ev["cfg"], output_root or ev["official"], ev["secure"])


def test_second_evaluation_starts_with_existing_results(ev):
    evaluate(ev, "E1")
    run = evaluate(ev, "E2")
    m = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["official_run"] is True and m["evaluator"]["git_dirty"] is False
    assert m["evaluator"]["source_integrity"]["clean"] is True and m["evaluator"]["source_integrity"]["non_blocking_changes"]["output"] > 0


def test_modifying_source_blocks_an_official_evaluation(ev):
    evaluate(ev, "E1")
    (ev["root"] / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="src/module.py"):
        evaluate(ev, "E2")


def test_official_evaluation_cannot_skip_the_guard_through_the_api(ev):
    (ev["root"] / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(EvaluationInputError):
        write_run(records(ev, "E1"), ev["gold"], ev["cfg"], ev["official"], ev["secure"], require_clean_tree=False)


def test_development_evaluation_with_changed_source_is_recorded(ev):
    (ev["root"] / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    run = evaluate(ev, "E1", ev["root"] / "experiments/development/x")
    m = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["official_run"] is False and m["evaluator"]["git_dirty"] is True and m["evaluator"]["source_integrity"]["blocking_count"] == 1


def test_a_secure_folder_that_is_no_longer_ignored_blocks(ev):
    evaluate(ev, "E1")
    (ev["root"] / ".git/info/exclude").write_text("!predictions_secure/\n", encoding="utf-8")
    (ev["root"] / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    assert any(p.startswith("predictions_secure/") or p == ".gitignore" for p, _ in check_source_integrity(ev["root"]).blocking)


def test_evaluator_cli_has_no_bypass(ev):
    with pytest.raises(SystemExit):
        runner.main(["evaluate", "--predictions", "x.jsonl", "--allow-dirty-tree"])


def test_real_repository_check_runs_without_reading_files():
    """Paths and git metadata only. Its result depends on the current state, so only the shape is asserted."""
    report = check_source_integrity(ROOT)
    assert report.git_available and isinstance(report.clean, bool) and "policy_version" in report.to_manifest()
