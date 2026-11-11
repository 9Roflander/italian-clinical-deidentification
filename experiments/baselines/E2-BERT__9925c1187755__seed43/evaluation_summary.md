# Evaluation summary: E2-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E2-BERT/seed_43 |
| Configuration | 9925c1187755 |
| Random seed | 43 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.605 [0.528, 0.687]** |
| Macro-F1 | 0.550 [0.479, 0.610] |
| Micro precision | 0.568 [0.484, 0.661] |
| Micro recall, also correct redaction recall | 0.647 [0.556, 0.738] |
| Removal coverage | 0.830 [0.753, 0.893] |
| Complete-entity redaction recall | 0.802 [0.733, 0.868] |
| Over-redaction rate, relative to current gold | 0.157 [0.100, 0.223] |
| Non-sensitive characters preserved | 0.999 [0.998, 0.999] |
| Note-level exact redaction rate | 0.390 [0.293, 0.500] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 27 | 28 | 27 | 0.491 | 0.500 | 0.495 [0.365, 0.627] |
| ETÀ | 79 | 30 | 23 | 0.725 | 0.775 | 0.749 [0.670, 0.823] |
| DATA | 45 | 28 | 14 | 0.616 | 0.763 | 0.682 [0.467, 0.818] |
| LUOGO/INDIRIZZO | 12 | 38 | 25 | 0.240 | 0.324 | 0.276 [0.122, 0.383] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.522 | 0.449 |
| 2 | 17 | 0.590 | 0.558 |
| 3 | 17 | 0.565 | 0.583 |
| 4 | 16 | 0.740 | 0.653 |
| 5 | 16 | 0.636 | 0.503 |
| Mean | | 0.611 | 0.549 |
| SD | | 0.083 | 0.078 |

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
