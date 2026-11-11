# Evaluation summary: E0

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | none, revision none |
| Checkpoint | none |
| Configuration | control |
| Random seed | 0 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.000 [0.000, 0.000]** |
| Macro-F1 | 0.000 [0.000, 0.000] |
| Micro precision | undefined [undefined, undefined] |
| Micro recall, also correct redaction recall | 0.000 [0.000, 0.000] |
| Removal coverage | 0.000 [0.000, 0.000] |
| Complete-entity redaction recall | 0.000 [0.000, 0.000] |
| Over-redaction rate, relative to current gold | undefined [undefined, undefined] |
| Non-sensitive characters preserved | 1.000 [1.000, 1.000] |
| Note-level exact redaction rate | 0.012 [0.000, 0.037] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 0 | 0 | 54 | undefined | 0.000 | 0.000 [0.000, 0.000] |
| ETÀ | 0 | 0 | 102 | undefined | 0.000 | 0.000 [0.000, 0.000] |
| DATA | 0 | 0 | 59 | undefined | 0.000 | 0.000 [0.000, 0.000] |
| LUOGO/INDIRIZZO | 0 | 0 | 37 | undefined | 0.000 | 0.000 [0.000, 0.000] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.000 | 0.000 |
| 2 | 17 | 0.000 | 0.000 |
| 3 | 17 | 0.000 | 0.000 |
| 4 | 16 | 0.000 | 0.000 |
| 5 | 16 | 0.000 | 0.000 |
| Mean | | 0.000 | 0.000 |
| SD | | 0.000 | 0.000 |

Fold means are arithmetic means of fold scores, not the pooled results. The folds overlap in training data, so they are not independent experiments.

## Output-copy fidelity

| Item | Value |
|---|---:|
| Notes with non-redacted text exactly preserved | 82 of 82 |
| Inserted characters | 0 |
| Deleted characters | 0 |
| Modified characters | 0 |
| Outputs with unmeasurable fidelity | 0 |
| Outputs that cannot be aligned unambiguously | 0 |
| Invalid outputs, all causes | 0 |

## Known limitations

- The gold set is small: 82 notes and 252 mentions, with few mentions in the rarer categories.
- All 82 notes were used in earlier development, so these are development estimates, not an independent generalization estimate.
- No external validation set has been used.
- gold_v2 has no independent second human annotator, so no agreement figure exists yet.
- Gold-span metrics cannot reveal identifiers missed during gold annotation.
- Only four categories are evaluated. They do not cover every form of personally identifiable information.
- Over-redaction is relative to the current gold annotations, not proof that the extra text is non-sensitive.
- Confidence intervals quantify document-sampling uncertainty conditional on the trained fold models. They do not capture all training randomness.
- Each run reflects one training seed. Training randomness is measured only where several seeds are summarized.
