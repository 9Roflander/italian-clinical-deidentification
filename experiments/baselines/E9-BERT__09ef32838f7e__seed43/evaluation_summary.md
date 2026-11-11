# Evaluation summary: E9-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E9-BERT/seed_43 |
| Configuration | 09ef32838f7e |
| Random seed | 43 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.776 [0.707, 0.841]** |
| Macro-F1 | 0.743 [0.665, 0.814] |
| Micro precision | 0.745 [0.663, 0.822] |
| Micro recall, also correct redaction recall | 0.810 [0.732, 0.878] |
| Removal coverage | 0.899 [0.832, 0.948] |
| Complete-entity redaction recall | 0.869 [0.798, 0.929] |
| Over-redaction rate, relative to current gold | 0.102 [0.061, 0.156] |
| Non-sensitive characters preserved | 0.999 [0.999, 0.999] |
| Note-level exact redaction rate | 0.549 [0.451, 0.646] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 47 | 22 | 7 | 0.681 | 0.870 | 0.764 [0.632, 0.874] |
| ETÀ | 87 | 23 | 15 | 0.791 | 0.853 | 0.821 [0.750, 0.885] |
| DATA | 51 | 6 | 8 | 0.895 | 0.864 | 0.879 [0.729, 0.974] |
| LUOGO/INDIRIZZO | 19 | 19 | 18 | 0.500 | 0.514 | 0.507 [0.333, 0.687] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.852 | 0.815 |
| 2 | 17 | 0.760 | 0.739 |
| 3 | 17 | 0.606 | 0.595 |
| 4 | 16 | 0.814 | 0.759 |
| 5 | 16 | 0.854 | 0.808 |
| Mean | | 0.777 | 0.743 |
| SD | | 0.103 | 0.089 |

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
