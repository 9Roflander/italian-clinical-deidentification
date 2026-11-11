"""Scans every committable file outside original_work/ for note text, gold identifiers and the API key.
Failures report file paths and counts only."""
import pathlib
import re
import subprocess

from src.data.annotations import canonical_type, word_boundary_regex
from src.data.loaders import ROOT, load_gold
from src.data.normalization import tokens, word_ngrams
from src.generation.gemini_client import load_api_key

SKIP_SUFFIX = {".pdf", ".png"}


def committable_files():
    args = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    files = []
    for name in out:
        p = ROOT / name
        if name.startswith("original_work/") or not p.is_file() or p.suffix in SKIP_SUFFIX:
            continue
        files.append(p)
    return files


def test_scan_covers_the_new_work():
    names = {str(p.relative_to(ROOT)) for p in committable_files()}
    assert "audit/KNOWN_PROBLEMS.md" in names and "src/generation/synthetic_generator.py" in names
    assert "key.txt" not in names


def test_no_gold_note_text_in_committable_files():
    index = set()
    for n in load_gold():
        index |= word_ngrams(tokens(n.text), 8)
    bad = [str(p.relative_to(ROOT)) for p in committable_files() if word_ngrams(tokens(p.read_text(encoding="utf-8", errors="ignore")), 8) & index]
    assert not bad, f"files with gold note text: {bad}"


def test_no_gold_person_names_in_committable_files():
    names = {e.get("text", "") for n in load_gold() for e in n.entities if canonical_type(e.get("type")) == "NOME" and len(e.get("text", "")) >= 4}
    pats = [word_boundary_regex(x) for x in names]
    bad = {}
    for p in committable_files():
        t = p.read_text(encoding="utf-8", errors="ignore")
        c = sum(bool(x.search(t)) for x in pats)
        if c:
            bad[str(p.relative_to(ROOT))] = c
    assert not bad, f"files containing gold names (count only): {bad}"


def test_no_specific_gold_places_in_committable_files():
    """Multi-word places and institutions. Region names in the config are generic and not tested."""
    places = {e.get("text", "") for n in load_gold() for e in n.entities
              if canonical_type(e.get("type")) == "LUOGO/INDIRIZZO" and (" " in e.get("text", "") or e.get("text", "").isupper()) and len(e.get("text", "")) >= 4}
    pats = [word_boundary_regex(x) for x in places]
    bad = {}
    for p in committable_files():
        t = p.read_text(encoding="utf-8", errors="ignore")
        c = sum(bool(x.search(t)) for x in pats)
        if c:
            bad[str(p.relative_to(ROOT))] = c
    assert not bad, f"files containing gold places (count only): {bad}"


def test_api_key_is_in_no_committable_file():
    if not (ROOT / "key.txt").exists():
        return
    key = load_api_key(ROOT / "key.txt")
    bad = [str(p.relative_to(ROOT)) for p in committable_files() if key in p.read_text(encoding="utf-8", errors="ignore")]
    assert not bad, f"files containing the key: {bad}"


def test_no_google_style_key_outside_original_work():
    pat = re.compile(r"AIza[0-9A-Za-z_-]{20,}")
    bad = [str(p.relative_to(ROOT)) for p in committable_files() if pat.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert not bad, f"files with a key pattern: {bad}"
