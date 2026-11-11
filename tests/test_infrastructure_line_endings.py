"""D2: hashed files stay byte-identical after a checkout that would otherwise convert to CRLF."""
import hashlib
import shutil
import subprocess

import pytest

from src.data.loaders import ROOT

HASHED_FILES = [p.relative_to(ROOT) for pattern in ("prompts/generation_*.txt", "prompts/anchor_block_*.txt", "splits/fold_*.json", "splits/fold_*_manifest.csv", "src/evaluation/*.py", "configs/*.yaml")
                for p in sorted(ROOT.glob(pattern))] + [ROOT.joinpath("data_manifests/fold_assignments.csv").relative_to(ROOT)]


def run(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args], cwd=cwd, capture_output=True, text=True, check=True)


def digests(base) -> dict:
    return {str(p): hashlib.sha256((base / p).read_bytes()).hexdigest() for p in HASHED_FILES}


def windows_style_clone(tmp_path, with_attributes: bool):
    source = tmp_path / "source"
    for p in HASHED_FILES:
        (source / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, source / p)
    if with_attributes:
        shutil.copyfile(ROOT / ".gitattributes", source / ".gitattributes")
    run(source, "init", "-q")
    run(source, "add", "--", ".")
    run(source, "commit", "-q", "-m", "files")
    clone = tmp_path / "clone"
    subprocess.run(["git", "-c", "core.autocrlf=true", "-c", "core.eol=crlf", "clone", "-q", "--config", "core.autocrlf=true", "--config", "core.eol=crlf", str(source), str(clone)],
                   check=True, capture_output=True)
    return clone


def test_the_scope_covers_prompt_fold_and_evaluator_files():
    names = {str(p) for p in HASHED_FILES}
    assert "prompts/generation_v1.txt" in names and "splits/fold_1.json" in names and "src/evaluation/runner.py" in names and "configs/evaluation.yaml" in names
    assert not any(n.startswith(("data_secure", "predictions_secure", "prompts/rendered")) for n in names)


def test_hashes_survive_a_crlf_checkout_with_the_attributes(tmp_path):
    clone = windows_style_clone(tmp_path, True)
    expected, got = digests(ROOT), digests(clone)
    assert [k for k in expected if expected[k] != got[k]] == []


def test_without_the_attributes_the_same_checkout_changes_hashes(tmp_path):
    """Shows the defect is real and the test above has power."""
    clone = windows_style_clone(tmp_path, False)
    expected, got = digests(ROOT), digests(clone)
    changed = {k for k in expected if expected[k] != got[k]}
    assert "src/evaluation/scoring.py" in changed and "prompts/generation_v1.txt" in changed and "splits/fold_1.json" in changed


@pytest.mark.parametrize("path,attribute,value", [
    ("src/evaluation/scoring.py", "text", "auto"), ("src/evaluation/scoring.py", "eol", "lf"), ("prompts/generation_v1.txt", "eol", "lf"), ("splits/fold_1.json", "eol", "lf"),
    ("configs/evaluation.yaml", "eol", "lf"), ("splits/fold_1_manifest.csv", "text", "unset"), ("data_manifests/fold_assignments.csv", "text", "unset"),
    ("original_work/src/train_evaluate_cv_v4.py", "text", "unset"), ("original_work/data/old.json", "text", "unset"), ("paper.pdf", "text", "unset"), ("paper.pdf", "diff", "unset"), ("model.safetensors", "binary", "set"),
])
def test_attributes_resolve_as_intended(tmp_path, path, attribute, value):
    run(tmp_path, "init", "-q")
    shutil.copyfile(ROOT / ".gitattributes", tmp_path / ".gitattributes")
    assert run(tmp_path, "check-attr", attribute, "--", path).stdout.strip().endswith(f": {value}")


def test_no_file_under_lf_normalization_currently_has_crlf():
    """Nothing would be silently rewritten on the next add. Reads git's end-of-line metadata and prints only paths."""
    out = run(ROOT, "ls-files", "--eol", "--cached", "--others", "--exclude-standard").stdout
    changing = [line.split("\t")[-1] for line in out.splitlines() if ("w/crlf" in line or "w/mixed" in line) and "attr/-text" not in line]
    assert changing == []
