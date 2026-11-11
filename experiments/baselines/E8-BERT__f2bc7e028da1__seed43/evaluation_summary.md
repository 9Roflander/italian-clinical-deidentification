# Evaluation summary: E8-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E8-BERT/seed_43 |
| Configuration | f2bc7e028da1 |
| Random seed | 43 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.739 [0.664, 0.813]** |
| Macro-F1 | 0.713 [0.617, 0.797] |
| Micro precision | 0.711 [0.636, 0.786] |
| Micro recall, also correct redaction recall | 0.770 [0.684, 0.855] |
| Removal coverage | 0.885 [0.816, 0.939] |
| Complete-entity redaction recall | 0.873 [0.808, 0.930] |
| Over-redaction rate, relative to current gold | 0.104 [0.063, 0.160] |
| Non-sensitive characters preserved | 0.999 [0.999, 0.999] |
| Note-level exact redaction rate | 0.524 [0.427, 0.634] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 34 | 18 | 20 | 0.654 | 0.630 | 0.642 [0.485, 0.790] |
| ETÀ | 90 | 40 | 12 | 0.692 | 0.882 | 0.776 [0.711, 0.842] |
| DATA | 49 | 3 | 10 | 0.942 | 0.831 | 0.883 [0.735, 0.958] |
| LUOGO/INDIRIZZO | 21 | 18 | 16 | 0.538 | 0.568 | 0.553 [0.259, 0.772] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.627 | 0.481 |
| 2 | 17 | 0.800 | 0.829 |
| 3 | 17 | 0.804 | 0.811 |
| 4 | 16 | 0.819 | 0.749 |
| 5 | 16 | 0.653 | 0.580 |
| Mean | | 0.740 | 0.690 |
| SD | | 0.093 | 0.153 |

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
