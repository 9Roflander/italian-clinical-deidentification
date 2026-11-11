# Leakage Report

IDs and counts only. Reproduce with `python -m src.data.leakage_audit`. Outputs: `data_manifests/deduplication_report.csv`, `duplicate_pairs_for_review.csv`, `negative_audit.csv`, and `audit/results/leakage_summary.json`.

Note text is identical in `gold_v1` and `gold_v2`, so these results apply to both.

**Gate result: the CRF corpus and the historical S1 and S2 sets are blocked from training.** Gold training partitions are permitted, fold by fold. DART is blocked until its overlap check is done.

## Method

### Exact duplicates

Each document is reduced by Unicode NFKC, case folding, and removal of punctuation and extra whitespace, then hashed with SHA-256. Originals are never altered. Graded levels (byte-identical up to token-level) show how strong each match is.

### Near-duplicates

Candidates are pairs sharing a word 8-gram. For these, the audit computes 8-gram containment, longest common token run, token sequence ratio, and character 5-gram Jaccard.

| Class | Rule |
|---|---|
| `exact_duplicate` | Same normalized hash |
| `near_duplicate` | Containment at least 0.5, or sequence ratio at least 0.9 |
| `partial_overlap_review` | Longest run at least 12 tokens, or at least 3 shared 8-grams |
| `shared_phrase_only` | Anything smaller; counted, not treated as a duplicate |

**Limitation:** the guide asks for manual validation of thresholds on a sample. That requires reading note text, which this assistant must not do. The owner should review `duplicate_pairs_for_review.csv`. At present it has 78 pairs: the 77 exact twins and one near-copy. No pair fell in the review class, so the thresholds did not separate any borderline case.

## Results

| A | B | Pairs sharing an 8-gram | Exact | Near | Review | Shared phrase | Longest run, tokens |
|---|---|---|---:|---:|---:|---:|---:|
| Gold (82) | Gold (82) | 1 | 0 | 0 | 0 | 1 | 8 |
| Gold (82) | CRF (80) | 80 | **77** | **1** | 0 | 2 | 636 |
| Gold (82) | Dyspnea (2,667) | 0 | 0 | 0 | 0 | 0 | 0 |
| CRF (80) | Dyspnea (2,667) | 0 | 0 | 0 | 0 | 0 | 0 |
| Gold (82) | S1 named (624) | 24 | 0 | 0 | 0 | 24 | 9 |
| Gold (82) | S1 name-free (254) | 10 | 0 | 0 | 0 | 10 | 8 |
| Gold (82) | S2 named (577) | 24 | 0 | 0 | 0 | 24 | 9 |
| Gold (82) | S2 name-free (221) | 14 | 0 | 0 | 0 | 14 | 9 |

Historical synthetic notes also have no exact duplicates among themselves.

## Negative corpus audit

Every `synthetic-crf-train` row was trained with output = input.

| Group | Rows | Decision | Why |
|---|---:|---|---|
| Exact twin of a gold note | 77 | Excluded | Test or training note with a wrong target |
| `crf_048`, near-copy of `gold_050` | 1 | Excluded | Same content with two edits |
| `crf_074` (`IT101195`), `crf_079` (`IT102068`) | 2 | Quarantined | Not in the gold set; no PII annotations; both match age-like patterns |

A clean identity target needs a note with no PII. **No row qualifies.** Near-copies were found by the same method, so other notes derived from gold could exist only below these thresholds.

The two zero-annotation notes (`gold_020`, `gold_037`) are gold notes, already in the gold pool, and stay subject to their fold assignment.

### Impact per fold

| Fold | Test notes in CRF | Not in CRF |
|---:|---:|---|
| 1 | 16 of 16 | None |
| 2 | 14 of 17 | `gold_055`, `gold_061`, `gold_079` |
| 3 | 15 of 17 | `gold_053`, `gold_066` |
| 4 | 16 of 16 | None |
| 5 | 16 of 16 | None |

Fold 3's `gold_050` also has the near-copy.

## Generation prompts

| Exposure | Notes | Test fold |
|---|---|---|
| Whole note as anchor | `gold_034`, `gold_013`, `gold_072` | 2, 3, 4 |
| Hard-coded strings, `generate_synthetic_v2.py:115` | Found in `gold_000`, `gold_001` | 1, 5 |

Other hard-coded strings look like chance matches: for example, generic ages and years that match several notes. That reading is an inference. Every fold therefore had at least one test note connected to the historical prompts.

Measured effect on the text: longest verbatim run 9 tokens (7 with anchors), and no notes equal to gold. Details are in `audit/results/v6_synthetic.json`. This limits surface copying. It says nothing about S2's selection on CV results, which no text comparison can detect.

## Note text outside the data folder

From `audit/results/v7_text_exposure.json`:

| File | Note | Share of its 8-grams | In git |
|---|---|---:|---|
| `original_work/src/generalization_test.py` | `gold_000` | 14.1% | Yes, public |
| Submitted paper PDF, Table 1 | `gold_042` | 7.8% | No, but submitted |

No other project document shares an 8-word sequence with a gold note. `original_work/data/` is tracked and public.

## Not checked

- **DART vs gold.** Gated; needs an authenticated run.
- **Patient-level linkage.** No identifiers exist in the gold file.
- **Paraphrased overlap.** Word n-grams cannot detect rewording.
- **Whether quarantined CRF rows contain names.** Patterns cover ages, dates and initials only; a person would have to read them.
- **Manual validation of thresholds**, described above.

## Controls for the rebuild

| Control | Location | Test |
|---|---|---|
| Generator can load training notes only | `load_train_notes`, `src/data/splits.py` | `tests/test_fold_separation.py` |
| Anchors must be training notes | `AnchorSelector` | `tests/test_anchor_selection.py` |
| No test-note text in any rendered prompt | Sentinel test across folds | `tests/test_anchor_selection.py` |
| Plan is unchanged when test notes change | Hash comparison | `tests/test_anchor_selection.py` |
| Fold manifests never permit a test note | `splits/fold_*_manifest.csv` | `tests/test_leakage.py` |
| Identity targets must be PII-free | `validate_negative_example` | `tests/test_negative_examples.py` |
| Generated data checked against test notes, never filtered | `src/generation/post_audit.py` | `tests/test_leakage.py` |
