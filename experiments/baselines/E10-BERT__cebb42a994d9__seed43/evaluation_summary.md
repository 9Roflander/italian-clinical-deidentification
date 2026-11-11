# Evaluation summary: E10-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E10-BERT/seed_43 |
| Configuration | cebb42a994d9 |
| Random seed | 43 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.710 [0.646, 0.773]** |
| Macro-F1 | 0.678 [0.593, 0.744] |
| Micro precision | 0.671 [0.601, 0.741] |
| Micro recall, also correct redaction recall | 0.754 [0.678, 0.827] |
| Removal coverage | 0.865 [0.790, 0.925] |
| Complete-entity redaction recall | 0.853 [0.798, 0.906] |
| Over-redaction rate, relative to current gold | 0.141 [0.091, 0.203] |
| Non-sensitive characters preserved | 0.999 [0.998, 0.999] |
| Note-level exact redaction rate | 0.402 [0.293, 0.512] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 36 | 24 | 18 | 0.600 | 0.667 | 0.632 [0.492, 0.765] |
| ETÀ | 85 | 32 | 17 | 0.726 | 0.833 | 0.776 [0.711, 0.838] |
| DATA | 49 | 18 | 10 | 0.731 | 0.831 | 0.778 [0.586, 0.894] |
| LUOGO/INDIRIZZO | 20 | 19 | 17 | 0.513 | 0.541 | 0.526 [0.314, 0.660] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.667 | 0.545 |
| 2 | 17 | 0.748 | 0.748 |
| 3 | 17 | 0.683 | 0.662 |
| 4 | 16 | 0.718 | 0.612 |
| 5 | 16 | 0.747 | 0.679 |
| Mean | | 0.713 | 0.649 |
| SD | | 0.037 | 0.076 |

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
