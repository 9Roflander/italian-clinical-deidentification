"""Path-aware source integrity and platform details for run manifests.

An official run must be tied to committed source, so `check_source_integrity` asks: has anything that could change
a result been altered since HEAD? Every path git reports falls into one class:

  critical       Source, prompts, configuration, folds, manifests, tests, scripts, dependency locks. Any change blocks.
  output         What runs produce, in known folders and with data suffixes. Recorded, never blocking.
  documentation  Reports, papers, editor settings. Recorded, never blocking.
  archive        The historical project, except its source code. Recorded, never blocking.
  unclassified   Anything else. Blocks, so a new kind of file cannot slip past the check.

The check reads paths and git metadata only. It never opens a file, and it fails closed if git cannot answer.
"""
from __future__ import annotations

import fnmatch
import hashlib
import pathlib
import platform
import subprocess
from dataclasses import dataclass, field

POLICY_VERSION = "1"
CRITICAL_PREFIXES = ("src/", "prompts/", "configs/", "splits/", "data_manifests/", "tests/", "scripts/",
                     "audit/scripts/", "audit/reference/", "original_work/src/")
CRITICAL_ROOT_FILES = (".gitattributes", ".gitignore", "pyproject.toml", "setup.py", "setup.cfg", "conftest.py", "pytest.ini", "tox.ini")
CRITICAL_ROOT_PATTERNS = ("requirements*.txt", "*.lock", "environment*.yml", "environment*.yaml")
OUTPUT_PREFIXES = ("synthetic_data/", "experiments/", "metrics/", "audit/results/")
OUTPUT_SUFFIXES = (".jsonl", ".json", ".csv", ".md", ".txt", ".log")
OUTPUT_NAMES = (".gitkeep",)
DOCUMENTATION_PREFIXES = ("reports/", "paper_revisions/", ".vscode/", ".idea/")
DOCUMENTATION_ROOT_FILES = ("README.md",)
AUDIT_DOCUMENT_SUFFIXES = (".md", ".csv")
ARCHIVE_PREFIXES = ("original_work/",)
# Files git ignores on purpose inside critical folders. No other ignored file may hide there.
IGNORED_WITHIN_CRITICAL = ("prompts/rendered/",)
CACHE_COMPONENTS = ("__pycache__", ".pytest_cache", ".DS_Store")
# Result locations that count as official. Anywhere else is a development run, recorded as such.
OFFICIAL_OUTPUT_ROOTS = ("synthetic_data", "experiments/clean_cv", "experiments/baselines", "experiments/ablations", "experiments/independent_test")
MAX_LISTED_PATHS = 50

CRITICAL, OUTPUT, DOCUMENTATION, ARCHIVE, UNCLASSIFIED, CACHE = "critical", "output", "documentation", "archive", "unclassified", "cache"


def classify_path(path: str) -> str:
    """Class of a repository-relative path, with forward slashes."""
    parts = path.rstrip("/").split("/")
    name = parts[-1]
    if any(p in CACHE_COMPONENTS for p in parts) or name.endswith(".pyc"):
        return CACHE
    if path.startswith(CRITICAL_PREFIXES):
        return CRITICAL
    if len(parts) == 1 and (name in CRITICAL_ROOT_FILES or any(fnmatch.fnmatch(name, p) for p in CRITICAL_ROOT_PATTERNS)):
        return CRITICAL
    if path.startswith(OUTPUT_PREFIXES):
        return OUTPUT if name.endswith(OUTPUT_SUFFIXES) or name in OUTPUT_NAMES else UNCLASSIFIED
    if path.startswith(ARCHIVE_PREFIXES):
        return ARCHIVE
    if path.startswith(DOCUMENTATION_PREFIXES) or path in DOCUMENTATION_ROOT_FILES or name.endswith(".pdf"):
        return DOCUMENTATION
    if len(parts) == 2 and parts[0] == "audit" and name.endswith(AUDIT_DOCUMENT_SUFFIXES):
        return DOCUMENTATION
    return UNCLASSIFIED


@dataclass(frozen=True)
class IntegrityReport:
    git_available: bool
    commit: str
    critical_tree_sha256: str | None
    blocking: tuple = ()
    non_blocking_counts: dict = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return self.git_available and bool(self.commit) and bool(self.critical_tree_sha256) and not self.blocking

    def summary(self) -> str:
        if self.clean:
            return "source integrity passed"
        if not self.git_available or not self.commit:
            return "git could not identify a commit"
        shown = ", ".join(f"{p} ({why})" for p, why in self.blocking[:10])
        more = len(self.blocking) - 10
        return f"{len(self.blocking)} blocking path(s): {shown}" + (f", and {more} more" if more > 0 else "")

    def to_manifest(self) -> dict:
        """Paths and counts only."""
        return {"policy_version": POLICY_VERSION, "clean": self.clean, "git_available": self.git_available, "commit": self.commit,
                "critical_tree_sha256": self.critical_tree_sha256, "blocking_count": len(self.blocking),
                "blocking_paths": [{"path": p, "reason": why} for p, why in self.blocking[:MAX_LISTED_PATHS]],
                "non_blocking_changes": dict(sorted(self.non_blocking_counts.items()))}


def _git(root, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True)


def _entries(data: bytes) -> list[str]:
    return [x.decode("utf-8", "surrogateescape") for x in data.split(b"\0") if x]


def _is_expected_ignored(path: str) -> bool:
    return classify_path(path) == CACHE or path.startswith(IGNORED_WITHIN_CRITICAL)


def check_source_integrity(root) -> IntegrityReport:
    """Compare critical paths with HEAD. Fails closed: if git cannot answer, the result is not clean."""
    root = pathlib.Path(root)
    try:
        head = _git(root, "rev-parse", "--verify", "HEAD")
        status = _git(root, "status", "--porcelain=v1", "-z", "--no-renames", "--untracked-files=all", "--ignored=matching")
        tree = _git(root, "ls-tree", "-r", "-z", "HEAD")
        flags = _git(root, "ls-files", "-v", "-z")
    except OSError:
        return IntegrityReport(False, "", None)
    if status.returncode != 0:
        return IntegrityReport(False, "", None)
    commit = head.stdout.decode().strip() if head.returncode == 0 else ""

    blocking, counts = [], {}
    for entry in _entries(status.stdout):
        code, path = entry[:2], entry[3:]
        kind = classify_path(path)
        if code == "!!":
            # A critical file git has been told to ignore would never be seen by the checks below.
            if kind == CRITICAL and not _is_expected_ignored(path):
                blocking.append((path, "ignored_critical_file"))
            continue
        if kind == CRITICAL:
            blocking.append((path, "untracked_critical_file" if code == "??" else "modified_critical_file"))
        elif kind == UNCLASSIFIED:
            blocking.append((path, "unclassified_path"))
        elif kind != CACHE:
            counts[kind] = counts.get(kind, 0) + 1
    # A flagged file (assume-unchanged, skip-worktree) can differ from HEAD while git status stays silent.
    for entry in _entries(flags.stdout) if flags.returncode == 0 else []:
        tag, path = entry[0], entry[2:]
        if (tag.islower() or tag == "S") and classify_path(path) == CRITICAL:
            blocking.append((path, "git_change_tracking_disabled"))

    tree_sha = None
    if commit and tree.returncode == 0:
        lines = [x for x in _entries(tree.stdout) if classify_path(x.split("\t", 1)[1]) == CRITICAL]
        tree_sha = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8", "surrogateescape")).hexdigest() if lines else None
    return IntegrityReport(True, commit, tree_sha, tuple(sorted(set(blocking))), counts)


def is_official_output(path, root) -> bool:
    """True when results written under `path` count as official and must pass the integrity check."""
    target = pathlib.Path(path).resolve()
    base = pathlib.Path(root).resolve()
    return any(target == base / r or (base / r) in target.parents for r in OFFICIAL_OUTPUT_ROOTS)


def run_environment() -> dict:
    """Machine details for manifests, kept apart from anything that affects scoring.
    `machine` describes the running process: x86_64 under Rosetta on Apple Silicon."""
    return {"machine": platform.machine(), "python_implementation": platform.python_implementation(), "python_version": platform.python_version(),
            "os": platform.system(), "os_release": platform.release(), "os_version": platform.version()}
