# Evaluation summary: E10-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E10-BERT/seed_42 |
| Configuration | cebb42a994d9 |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.708 [0.635, 0.780]** |
| Macro-F1 | 0.662 [0.587, 0.729] |
| Micro precision | 0.691 [0.616, 0.766] |
| Micro recall, also correct redaction recall | 0.726 [0.647, 0.805] |
| Removal coverage | 0.840 [0.756, 0.908] |
| Complete-entity redaction recall | 0.837 [0.778, 0.895] |
| Over-redaction rate, relative to current gold | 0.122 [0.064, 0.190] |
| Non-sensitive characters preserved | 0.999 [0.998, 0.999] |
| Note-level exact redaction rate | 0.451 [0.341, 0.561] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 34 | 16 | 20 | 0.680 | 0.630 | 0.654 [0.512, 0.787] |
| ETÀ | 85 | 30 | 17 | 0.739 | 0.833 | 0.783 [0.722, 0.841] |
| DATA | 49 | 12 | 10 | 0.803 | 0.831 | 0.817 [0.633, 0.923] |
| LUOGO/INDIRIZZO | 15 | 24 | 22 | 0.385 | 0.405 | 0.395 [0.231, 0.556] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.579 | 0.505 |
| 2 | 17 | 0.817 | 0.830 |
| 3 | 17 | 0.699 | 0.670 |
| 4 | 16 | 0.706 | 0.591 |
| 5 | 16 | 0.776 | 0.714 |
| Mean | | 0.716 | 0.662 |
| SD | | 0.091 | 0.123 |

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
