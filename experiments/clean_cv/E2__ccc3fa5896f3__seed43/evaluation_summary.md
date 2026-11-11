# Evaluation summary: E2

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | meta-llama/Llama-3.2-3B-Instruct, revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 |
| Checkpoint | E2/seed_43 |
| Configuration | ccc3fa5896f3 |
| Random seed | 43 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.697 [0.620, 0.772]** |
| Macro-F1 | 0.642 [0.550, 0.722] |
| Micro precision | 0.834 [0.758, 0.903] |
| Micro recall, also correct redaction recall | 0.599 [0.510, 0.690] |
| Removal coverage | 0.596 [0.475, 0.704] |
| Complete-entity redaction recall | 0.611 [0.522, 0.702] |
| Over-redaction rate, relative to current gold | 0.104 [0.039, 0.198] |
| Non-sensitive characters preserved | 0.953 [0.889, 0.999] |
| Note-level exact redaction rate | 0.476 [0.366, 0.585] |
| Invalid-output rate | 0.037 [0.000, 0.085] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 24 | 1 | 30 | 0.960 | 0.444 | 0.608 [0.479, 0.742] |
| ETÀ | 77 | 13 | 25 | 0.856 | 0.755 | 0.802 [0.711, 0.885] |
| DATA | 37 | 8 | 22 | 0.822 | 0.627 | 0.712 [0.497, 0.852] |
| LUOGO/INDIRIZZO | 13 | 8 | 24 | 0.619 | 0.351 | 0.448 [0.263, 0.606] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.690 | 0.548 |
| 2 | 17 | 0.822 | 0.818 |
| 3 | 17 | 0.557 | 0.524 |
| 4 | 16 | 0.792 | 0.725 |
| 5 | 16 | 0.650 | 0.546 |
| Mean | | 0.702 | 0.632 |
| SD | | 0.108 | 0.132 |

Fold means are arithmetic means of fold scores, not the pooled results. The folds overlap in training data, so they are not independent experiments.

## Output-copy fidelity

| Item | Value |
|---|---:|
| Notes with non-redacted text exactly preserved | 79 of 82 |
| Inserted characters | 0 |
| Deleted characters | 0 |
| Modified characters | 0 |
| Outputs with unmeasurable fidelity | 3 |
| Outputs that cannot be aligned unambiguously | 3 |
| Invalid outputs, all causes | 3 |

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
