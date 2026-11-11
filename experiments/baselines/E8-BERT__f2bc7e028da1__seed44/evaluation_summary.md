# Evaluation summary: E8-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E8-BERT/seed_44 |
| Configuration | f2bc7e028da1 |
| Random seed | 44 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.765 [0.702, 0.823]** |
| Macro-F1 | 0.733 [0.651, 0.796] |
| Micro precision | 0.738 [0.663, 0.807] |
| Micro recall, also correct redaction recall | 0.794 [0.721, 0.860] |
| Removal coverage | 0.868 [0.784, 0.931] |
| Complete-entity redaction recall | 0.869 [0.797, 0.929] |
| Over-redaction rate, relative to current gold | 0.130 [0.071, 0.207] |
| Non-sensitive characters preserved | 0.999 [0.998, 0.999] |
| Note-level exact redaction rate | 0.476 [0.378, 0.573] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 44 | 23 | 10 | 0.657 | 0.815 | 0.727 [0.594, 0.847] |
| ETÀ | 88 | 27 | 14 | 0.765 | 0.863 | 0.811 [0.749, 0.871] |
| DATA | 49 | 6 | 10 | 0.891 | 0.831 | 0.860 [0.733, 0.935] |
| LUOGO/INDIRIZZO | 19 | 15 | 18 | 0.559 | 0.514 | 0.535 [0.300, 0.689] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.789 | 0.725 |
| 2 | 17 | 0.727 | 0.695 |
| 3 | 17 | 0.703 | 0.703 |
| 4 | 16 | 0.839 | 0.789 |
| 5 | 16 | 0.761 | 0.702 |
| Mean | | 0.764 | 0.723 |
| SD | | 0.053 | 0.039 |

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
