# Evaluation summary: E1-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E1-BERT/seed_44 |
| Configuration | ae91fdcd33e2 |
| Random seed | 44 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.518 [0.453, 0.585]** |
| Macro-F1 | 0.450 [0.359, 0.510] |
| Micro precision | 0.557 [0.489, 0.636] |
| Micro recall, also correct redaction recall | 0.484 [0.406, 0.564] |
| Removal coverage | 0.619 [0.473, 0.748] |
| Complete-entity redaction recall | 0.619 [0.536, 0.705] |
| Over-redaction rate, relative to current gold | 0.077 [0.039, 0.133] |
| Non-sensitive characters preserved | 1.000 [0.999, 1.000] |
| Note-level exact redaction rate | 0.329 [0.244, 0.427] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 21 | 8 | 33 | 0.724 | 0.389 | 0.506 [0.328, 0.659] |
| ETÀ | 70 | 37 | 32 | 0.654 | 0.686 | 0.670 [0.595, 0.739] |
| DATA | 25 | 41 | 34 | 0.379 | 0.424 | 0.400 [0.145, 0.560] |
| LUOGO/INDIRIZZO | 6 | 11 | 31 | 0.353 | 0.162 | 0.222 [0.000, 0.356] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.769 | 0.737 |
| 2 | 17 | 0.426 | 0.307 |
| 3 | 17 | 0.176 | 0.086 |
| 4 | 16 | 0.462 | 0.352 |
| 5 | 16 | 0.614 | 0.449 |
| Mean | | 0.489 | 0.386 |
| SD | | 0.222 | 0.237 |

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
