"""Shared helpers for the Phase 0 audit scripts.

Privacy rule for every script in this folder: output IDs, hashes and counts only.
Never write or print note text, entity strings, or prompt strings.
"""
import ast, hashlib, json, pathlib, re, unicodedata
import numpy as np
from sklearn.model_selection import KFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
OW = ROOT / "original_work"
RES = ROOT / "audit" / "results"
GOLD = OW / "data" / "gold_standard_80.json"
V = ["NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"]
SYNTH = {
    "S1_named": "synthetic_v2_1000.json",
    "S1_noname": "synthetic_v2_noname_400.json",
    "S2_named": "synthetic_v4_named_more.json",
    "S2_noname": "synthetic_v4_noname_more.json",
}
RESULT_FILES = {
    "llama_v3": "cv_v3_results.json",
    "llama_v4": "cv_v4_results.json",
    "gemma_1b": "cv_gemma1b_results.json",
    "bert_truncated": "cv_bert_results.json",
    "bert_chunked": "bert_cpt_and_chunking_findings/cv_bert_chunked_results.json",
    "bert_cpt_chunked": "bert_cpt_and_chunking_findings/cv_bert_cpt_results.json",
}


def load_gold():
    return json.loads(GOLD.read_text(encoding="utf-8"))


def load_synth():
    return {k: json.loads((OW / "data" / v).read_text(encoding="utf-8")) for k, v in SYNTH.items()}


def load_results():
    return {k: json.loads((OW / v).read_text(encoding="utf-8")) for k, v in RESULT_FILES.items()}


def folds(n=80):
    """Same call as the historical CV scripts. Returns sorted test IDs per fold."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return [sorted(int(i) for i in te) for _, te in kf.split(np.arange(n))]


def fold_of(F):
    return {i: k + 1 for k, te in enumerate(F) for i in te}


def stype(t):
    t = (t or "").upper()
    if "LUOGO" in t or "INDIRIZZO" in t:
        return "LUOGO/INDIRIZZO"
    return t if t in V else None


def tokens(s):
    """Unicode-, case- and punctuation-insensitive word tokens."""
    return re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold())


def grams(t, n):
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def doc_hash(s):
    return hashlib.sha256(" ".join(tokens(s)).encode("utf-8")).hexdigest()


def sha8(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def boundary(s):
    return re.compile(r"(?<!\w)" + re.escape(s) + r"(?!\w)")


def load_funcs(path, names, constants=(), extra=None):
    """Exec only the named top-level defs/constants from a historical script, without
    importing it (its imports need a GPU stack). This runs the real code, not a copy."""
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in constants for t in node.targets
        ):
            keep.append(node)
    ns = {"re": re, "json": json}
    ns.update(extra or {})
    exec(compile(ast.Module(keep, []), str(path), "exec"), ns)
    return ns


def func_dump(path, name):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.dump(node)
    return None


def save(name, obj):
    RES.mkdir(parents=True, exist_ok=True)
    p = RES / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=False), encoding="utf-8")
    print("wrote", p.relative_to(ROOT))
