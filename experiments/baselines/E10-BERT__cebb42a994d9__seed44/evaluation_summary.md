# Evaluation summary: E10-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E10-BERT/seed_44 |
| Configuration | cebb42a994d9 |
| Random seed | 44 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.767 [0.697, 0.830]** |
| Macro-F1 | 0.749 [0.649, 0.818] |
| Micro precision | 0.739 [0.665, 0.811] |
| Micro recall, also correct redaction recall | 0.798 [0.708, 0.874] |
| Removal coverage | 0.844 [0.752, 0.913] |
| Complete-entity redaction recall | 0.845 [0.765, 0.912] |
| Over-redaction rate, relative to current gold | 0.125 [0.077, 0.186] |
| Non-sensitive characters preserved | 0.999 [0.998, 0.999] |
| Note-level exact redaction rate | 0.488 [0.378, 0.598] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 43 | 17 | 11 | 0.717 | 0.796 | 0.754 [0.615, 0.874] |
| ETÀ | 87 | 38 | 15 | 0.696 | 0.853 | 0.767 [0.689, 0.842] |
| DATA | 51 | 5 | 8 | 0.911 | 0.864 | 0.887 [0.738, 0.967] |
| LUOGO/INDIRIZZO | 20 | 11 | 17 | 0.645 | 0.541 | 0.588 [0.298, 0.750] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.864 | 0.863 |
| 2 | 17 | 0.780 | 0.770 |
| 3 | 17 | 0.684 | 0.701 |
| 4 | 16 | 0.718 | 0.563 |
| 5 | 16 | 0.787 | 0.739 |
| Mean | | 0.767 | 0.727 |
| SD | | 0.069 | 0.110 |

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
