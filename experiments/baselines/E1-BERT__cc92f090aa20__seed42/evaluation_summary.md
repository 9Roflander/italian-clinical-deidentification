# Evaluation summary: E1-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E1-BERT/seed_42 |
| Configuration | cc92f090aa20 |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.030 [0.007, 0.058]** |
| Macro-F1 | 0.018 [0.005, 0.032] |
| Micro precision | 0.364 [0.167, 1.000] |
| Micro recall, also correct redaction recall | 0.016 [0.004, 0.031] |
| Removal coverage | 0.009 [0.001, 0.021] |
| Complete-entity redaction recall | 0.020 [0.004, 0.040] |
| Over-redaction rate, relative to current gold | 0.067 [0.000, 0.143] |
| Non-sensitive characters preserved | 1.000 [1.000, 1.000] |
| Note-level exact redaction rate | 0.037 [0.000, 0.073] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 0 | 0 | 54 | undefined | 0.000 | 0.000 [0.000, 0.000] |
| ETÀ | 4 | 7 | 98 | 0.364 | 0.039 | 0.071 [0.018, 0.126] |
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
| 5 | 16 | 0.145 | 0.067 |
| Mean | | 0.029 | 0.013 |
| SD | | 0.065 | 0.030 |

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
