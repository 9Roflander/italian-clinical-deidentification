"""D3 and the v1.0.1 promise: provenance changed, and scoring did not.

The comparison runs archived v1.0 and current code in separate temporary trees with no access to real data.
Two tests use real notes and skip themselves when the secure files are absent.
"""
import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

import pytest

from src.data.loaders import GOLD_V2, ROOT
from src.evaluation import EVALUATOR_VERSION, HISTORICAL_FINGERPRINTS
from src.evaluation.runner import GENERATED_FILES, evaluator_fingerprint
from src.infrastructure import INFRASTRUCTURE_VERSION
from src.infrastructure.provenance import run_environment

ARCHIVE = ROOT / "audit" / "reference" / "evaluator_v1_0"
OLD_FINGERPRINT = "8ff823ed86c5525af514b33d212e6fb9e704989038142a8874f2139b8cc6609d"
CHANGED_FILES = {"__init__.py", "runner.py"}
CHANGED_RUNNER_FUNCTIONS = {"git_state", "write_run", "main"}
DRY_RUN = ROOT / "experiments" / "development" / "evaluator_dry_run" / "E0__control__seed0"
DRY_RUN_PREDICTIONS = ROOT / "predictions_secure" / "development" / "E0__control__seed0" / "predictions_oof.jsonl"


def fingerprint(files: dict, config: bytes) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode())
        h.update(files[name])
    h.update(config)
    return h.hexdigest()


def package_files(folder) -> dict:
    return {p.name: p.read_bytes() for p in folder.glob("*.py")}


def test_versions():
    assert EVALUATOR_VERSION == INFRASTRUCTURE_VERSION == "1.0.1"


def test_the_old_fingerprint_is_preserved_and_matches_the_archive():
    assert HISTORICAL_FINGERPRINTS["1.0.0"] == OLD_FINGERPRINT
    assert fingerprint(package_files(ARCHIVE / "src/evaluation"), (ARCHIVE / "configs/evaluation.yaml").read_bytes()) == OLD_FINGERPRINT


def test_the_old_e0_manifest_records_the_old_fingerprint():
    manifest = json.loads((DRY_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluator"]["version"] == "1.0.0" and manifest["evaluator"]["fingerprint_sha256"] == OLD_FINGERPRINT


def test_the_new_fingerprint_differs_only_because_of_the_infrastructure_files():
    old, new = package_files(ARCHIVE / "src/evaluation"), package_files(ROOT / "src/evaluation")
    assert set(old) == set(new)
    assert {n for n in old if old[n] != new[n]} == CHANGED_FILES
    assert (ARCHIVE / "configs/evaluation.yaml").read_bytes() == (ROOT / "configs/evaluation.yaml").read_bytes()
    assert evaluator_fingerprint() != OLD_FINGERPRINT
    swapped = {**old, **{n: new[n] for n in CHANGED_FILES}}
    assert fingerprint(swapped, (ROOT / "configs/evaluation.yaml").read_bytes()) == evaluator_fingerprint()


def definitions(path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign):
            out["=" + ast.dump(node.targets[0])] = ast.dump(node)
    return out


def test_only_provenance_functions_changed_in_the_runner():
    old, new = definitions(ARCHIVE / "src/evaluation/runner.py"), definitions(ROOT / "src/evaluation/runner.py")
    assert set(old) == set(new)
    assert {k for k in old if old[k] != new[k]} == CHANGED_RUNNER_FUNCTIONS
    assert "score_note" not in new["write_run"] and "compute_metrics" not in new["write_run"]


def evaluator_tree(base, evaluation_source, config_source, with_infrastructure):
    (base / "src").mkdir(parents=True)
    (base / "configs").mkdir()
    shutil.copyfile(ROOT / "src/__init__.py", base / "src/__init__.py")
    shutil.copytree(ROOT / "src/data", base / "src/data", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(evaluation_source, base / "src/evaluation", ignore=shutil.ignore_patterns("__pycache__"))
    if with_infrastructure:
        shutil.copytree(ROOT / "src/infrastructure", base / "src/infrastructure", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(config_source, base / "configs/evaluation.yaml")
    return base


def run_driver(tree):
    env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, str(ROOT / "tests/support/scoring_comparison_driver.py")], cwd=tree, env=env, capture_output=True, text=True, timeout=500)
    assert done.returncode == 0, done.stderr[-800:]
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def both_versions(tmp_path_factory):
    base = tmp_path_factory.mktemp("versions")
    old = evaluator_tree(base / "old", ARCHIVE / "src/evaluation", ARCHIVE / "configs/evaluation.yaml", False)
    new = evaluator_tree(base / "new", ROOT / "src/evaluation", ROOT / "configs/evaluation.yaml", True)
    return run_driver(old), run_driver(new), run_driver(new)


def test_the_comparison_ran_on_every_control(both_versions):
    from src.evaluation.controls import CONTROLS
    old, new, _ = both_versions
    assert set(old["controls"]) == set(new["controls"]) == set(CONTROLS) and len(CONTROLS) == 16


def test_control_scores_are_identical_between_versions(both_versions):
    old, new, _ = both_versions
    assert old["controls"] == new["controls"]


def test_every_generated_file_is_byte_identical_between_versions(both_versions):
    old, new, _ = both_versions
    assert old["control_files"] == new["control_files"]
    assert all(set(v) == set(GENERATED_FILES) for v in new["control_files"].values())


def test_the_controls_have_their_required_values(both_versions):
    _, new, _ = both_versions
    c = new["controls"]
    assert c["perfect_inline"]["micro_f1"] == c["perfect_segments"]["micro_f1"] == c["perfect_spans"]["micro_f1"] == 1
    assert c["empty_output"]["micro_f1"] == 0 and c["unchanged_input"]["micro_f1"] == 0 and c["every_entity_wrong_category"]["micro_f1"] == 0
    assert c["unchanged_input"]["invalid_outputs"] == 0 and c["empty_output"]["invalid_outputs"] == 20


def test_e0_reproduces_byte_for_byte_on_both_versions_and_across_them(both_versions):
    old, new, again = both_versions
    assert old["e0_reproduced"] and new["e0_reproduced"] and not old["e0_problems"] and not new["e0_problems"]
    assert old["e0_files"] == new["e0_files"] == again["e0_files"]


def test_the_manifest_gains_only_provenance_fields(both_versions):
    old, new, _ = both_versions
    assert set(new["e0_manifest_keys"]) - set(old["e0_manifest_keys"]) == {"official_run", "infrastructure_version"}


def test_run_environment_has_the_required_fields():
    env = run_environment()
    assert env["machine"] == platform.machine() and env["python_implementation"] == platform.python_implementation()
    assert env["python_version"] == platform.python_version() and env["os"] == platform.system() and env["os_version"] == platform.version() and env["os_release"] == platform.release()


def test_evaluation_manifest_records_the_platform(tmp_path):
    from tests.test_infrastructure_source_integrity import make_repo
    import src.evaluation.runner as runner
    from src.data.loaders import GoldNote
    from src.evaluation.gold_verification import VerifiedGold
    root = make_repo(tmp_path)
    original = runner.ROOT
    runner.ROOT = root
    try:
        text = "Zelinda ha 45 anni."
        s = [(0, 7, "NOME")]
        i = "mini_00"
        gold = VerifiedGold({i: GoldNote(i, 0, text, ({"start": 0, "end": 7, "type": "NOME", "text": "Zelinda", "annotated_text": "Zelinda"},), redacted_text="[NOME] ha 45 anni.")}, (i,), {i: 1}, {i: s}, "m", "v2", {1: "x"})
        cfg = runner.load_config()
        cfg["bootstrap"]["iterations"] = 2000
        rec = {"experiment": "T", "model": "m", "model_revision": "r", "checkpoint": "c", "config_id": "x", "seed": 1, "note_id": i, "fold_id": 1, "input_sha256": runner.text_sha256(text), "output_format": "inline", "raw_model_output": "[NOME] ha 45 anni."}
        run = runner.write_run([rec], gold, cfg, tmp_path / "experiments/development/x", tmp_path / "predictions_secure")
    finally:
        runner.ROOT = original
    m = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    for key, value in run_environment().items():
        assert m["environment"][key] == value
    assert "numpy" in m["environment"] and "python" in m["environment"] and m["infrastructure_version"] == "1.0.1" and m["evaluator"]["version"] == "1.0.1"
    for name in GENERATED_FILES:
        assert platform.version() not in (run / name).read_text(encoding="utf-8"), f"{name} must stay free of platform details"


def test_platform_details_cannot_change_a_result(monkeypatch, tmp_path):
    import src.evaluation.runner as runner
    from src.data.loaders import GoldNote
    from src.evaluation.gold_verification import VerifiedGold
    text = "Zelinda ha 45 anni."
    i = "mini_00"
    gold = VerifiedGold({i: GoldNote(i, 0, text, ({"start": 0, "end": 7, "type": "NOME", "text": "Zelinda", "annotated_text": "Zelinda"},), redacted_text="[NOME] ha 45 anni.")}, (i,), {i: 1}, {i: [(0, 7, "NOME")]}, "m", "v2", {1: "x"})
    cfg = runner.load_config()
    cfg["bootstrap"]["iterations"] = 2000
    rec = {"experiment": "T", "model": "m", "model_revision": "r", "checkpoint": "c", "config_id": "x", "seed": 1, "note_id": i, "fold_id": 1, "input_sha256": runner.text_sha256(text), "output_format": "inline", "raw_model_output": "[NOME] ha 45 anni."}
    before = runner.build_outputs(runner.evaluate_records([rec], gold), gold, cfg)
    monkeypatch.setattr(platform, "machine", lambda: "other-machine")
    monkeypatch.setattr(platform, "system", lambda: "OtherOS")
    assert runner.build_outputs(runner.evaluate_records([rec], gold), gold, cfg) == before


def test_generation_config_records_the_platform(project):
    from src.generation import synthetic_generator as g
    from tests.conftest import FakeClient
    g.run_generation(project["cfg"], 1, "non_anchored", execute=True, max_requests=1, output_root=project["out"], client=FakeClient(), gold_path=project["gold"], splits_dir=project["splits"], prompts_dir=project["prompts"])
    c = json.loads((project["out"] / "fold_1/non_anchored/run_config.json").read_text(encoding="utf-8"))
    assert c["environment"] == run_environment() and c["infrastructure_version"] == "1.0.1" and c["official_run"] is False
    assert "source_integrity" in c


@pytest.mark.skipif(not GOLD_V2.exists(), reason="needs the secure gold file")
def test_real_controls_match_the_recorded_v1_0_results():
    from src.evaluation.controls import CONTROLS, run_control
    from src.evaluation.gold_verification import verify_gold
    from src.evaluation.runner import load_config
    recorded = json.loads((ROOT / "audit/results/evaluation_controls.json").read_text(encoding="utf-8"))["controls"]
    gold = verify_gold(load_config())
    different = [name for name in CONTROLS if run_control(name, gold) != recorded[name]]
    assert different == []


@pytest.mark.skipif(not DRY_RUN_PREDICTIONS.exists(), reason="needs the secure E0 predictions")
def test_real_e0_from_v1_0_still_reproduces_byte_for_byte():
    from src.evaluation.gold_verification import verify_gold
    from src.evaluation.runner import load_config, verify_run
    cfg = load_config()
    report = verify_run(DRY_RUN, verify_gold(cfg), cfg, ROOT / "predictions_secure" / "development")
    assert report["reproduced"] and not report["problems"] and report["evaluator_changed_since_run"] is True
