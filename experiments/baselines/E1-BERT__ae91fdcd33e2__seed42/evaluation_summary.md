# Evaluation summary: E1-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E1-BERT/seed_42 |
| Configuration | ae91fdcd33e2 |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.573 [0.495, 0.653]** |
| Macro-F1 | 0.507 [0.416, 0.577] |
| Micro precision | 0.556 [0.475, 0.643] |
| Micro recall, also correct redaction recall | 0.591 [0.504, 0.679] |
| Removal coverage | 0.783 [0.679, 0.870] |
| Complete-entity redaction recall | 0.762 [0.678, 0.841] |
| Over-redaction rate, relative to current gold | 0.074 [0.036, 0.128] |
| Non-sensitive characters preserved | 0.999 [0.999, 1.000] |
| Note-level exact redaction rate | 0.402 [0.305, 0.512] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 23 | 10 | 31 | 0.697 | 0.426 | 0.529 [0.361, 0.681] |
| ETÀ | 75 | 75 | 27 | 0.500 | 0.735 | 0.595 [0.519, 0.676] |
| DATA | 46 | 24 | 13 | 0.657 | 0.780 | 0.713 [0.500, 0.847] |
| LUOGO/INDIRIZZO | 5 | 10 | 32 | 0.333 | 0.135 | 0.192 [0.000, 0.338] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.739 | 0.677 |
| 2 | 17 | 0.586 | 0.463 |
| 3 | 17 | 0.358 | 0.224 |
| 4 | 16 | 0.727 | 0.580 |
| 5 | 16 | 0.477 | 0.420 |
| Mean | | 0.577 | 0.473 |
| SD | | 0.163 | 0.172 |

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
