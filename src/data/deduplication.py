"""Exact and near-duplicate detection between document collections.

Thresholds are documented in audit/LEAKAGE_REPORT.md. They have not been validated by a
person reading the candidate pairs, because that requires access to the note text.
"""
from __future__ import annotations

import collections
import difflib
from dataclasses import asdict, dataclass

from src.data.normalization import doc_hash, tokens, word_ngrams

NEAR_DUPLICATE_CONTAINMENT = 0.5
NEAR_DUPLICATE_SEQ_RATIO = 0.9
PARTIAL_OVERLAP_RUN = 12
PARTIAL_OVERLAP_SHARED = 3


def longest_common_run(a: list[str], b: list[str]) -> int:
    positions = collections.defaultdict(list)
    for j, w in enumerate(b):
        positions[w].append(j)
    prev: dict[int, int] = {}
    best = 0
    for w in a:
        cur: dict[int, int] = {}
        for j in positions.get(w, ()):
            length = prev.get(j - 1, 0) + 1
            cur[j] = length
            best = max(best, length)
        prev = cur
    return best


def char_ngram_jaccard(a: str, b: str, n: int = 5) -> float:
    sa = {a[i : i + n] for i in range(len(a) - n + 1)}
    sb = {b[i : i + n] for i in range(len(b) - n + 1)}
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


@dataclass
class PairRow:
    source_a: str
    id_a: str
    source_b: str
    id_b: str
    relation: str
    shared_8grams: int
    containment_8: float
    longest_run_tokens: int
    token_seq_ratio: float
    char5_jaccard: float


def classify(exact: bool, containment: float, ratio: float, run: int, shared: int) -> str:
    if exact:
        return "exact_duplicate"
    if containment >= NEAR_DUPLICATE_CONTAINMENT or ratio >= NEAR_DUPLICATE_SEQ_RATIO:
        return "near_duplicate"
    if run >= PARTIAL_OVERLAP_RUN or shared >= PARTIAL_OVERLAP_SHARED:
        return "partial_overlap_review"
    return "shared_phrase_only"


def compare_collections(name_a: str, docs_a: dict[str, str], name_b: str, docs_b: dict[str, str],
                        same_collection: bool = False, n: int = 8, detail_min_shared: int = 2) -> list[PairRow]:
    """Every pair sharing at least one word n-gram, or an identical normalised hash."""
    tok_a = {k: tokens(v) for k, v in docs_a.items()}
    tok_b = {k: tokens(v) for k, v in docs_b.items()}
    gram_b = {k: word_ngrams(t, n) for k, t in tok_b.items()}
    index: dict[tuple, set[str]] = collections.defaultdict(set)
    for k, grams in gram_b.items():
        for g in grams:
            index[g].add(k)
    hash_b = collections.defaultdict(set)
    for k, v in docs_b.items():
        hash_b[doc_hash(v)].add(k)

    rows: list[PairRow] = []
    for ka, ta in tok_a.items():
        ga = word_ngrams(ta, n)
        hits = collections.Counter()
        for g in ga:
            for kb in index.get(g, ()):
                hits[kb] += 1
        exact_hits = hash_b.get(doc_hash(docs_a[ka]), set())
        for kb in set(hits) | exact_hits:
            if same_collection and ka >= kb:
                continue
            shared = hits.get(kb, 0)
            gb = gram_b[kb]
            denom = max(1, min(len(ga), len(gb)))
            containment = shared / denom
            is_exact = kb in exact_hits
            do_detail = is_exact or shared >= detail_min_shared
            ratio = (1.0 if is_exact else
                     difflib.SequenceMatcher(None, ta, tok_b[kb], autojunk=False).ratio() if do_detail else 0.0)
            jac = char_ngram_jaccard(docs_a[ka], docs_b[kb]) if do_detail else 0.0
            run = longest_common_run(ta, tok_b[kb]) if do_detail or shared else 0
            rows.append(PairRow(name_a, ka, name_b, kb, classify(is_exact, containment, ratio, run, shared),
                                shared, round(containment, 4), run, round(ratio, 4), round(jac, 4)))
    return rows


def rows_as_dicts(rows: list[PairRow]) -> list[dict]:
    return [asdict(r) for r in rows]
