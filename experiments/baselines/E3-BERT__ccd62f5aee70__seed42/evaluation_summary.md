# Evaluation summary: E3-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E3-BERT/seed_42 |
| Configuration | ccd62f5aee70 |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.669 [0.592, 0.744]** |
| Macro-F1 | 0.616 [0.524, 0.697] |
| Micro precision | 0.639 [0.560, 0.720] |
| Micro recall, also correct redaction recall | 0.702 [0.616, 0.786] |
| Removal coverage | 0.842 [0.761, 0.904] |
| Complete-entity redaction recall | 0.817 [0.753, 0.880] |
| Over-redaction rate, relative to current gold | 0.147 [0.098, 0.209] |
| Non-sensitive characters preserved | 0.999 [0.998, 0.999] |
| Note-level exact redaction rate | 0.402 [0.293, 0.512] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 37 | 28 | 17 | 0.569 | 0.685 | 0.622 [0.474, 0.766] |
| ETÀ | 82 | 29 | 20 | 0.739 | 0.804 | 0.770 [0.705, 0.834] |
| DATA | 46 | 17 | 13 | 0.730 | 0.780 | 0.754 [0.564, 0.874] |
| LUOGO/INDIRIZZO | 12 | 26 | 25 | 0.316 | 0.324 | 0.320 [0.107, 0.477] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.619 | 0.537 |
| 2 | 17 | 0.648 | 0.618 |
| 3 | 17 | 0.707 | 0.685 |
| 4 | 16 | 0.621 | 0.509 |
| 5 | 16 | 0.761 | 0.700 |
| Mean | | 0.671 | 0.610 |
| SD | | 0.061 | 0.086 |

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
