# Evaluation summary: E2

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | meta-llama/Llama-3.2-3B-Instruct, revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 |
| Checkpoint | E2/seed_42 |
| Configuration | ccc3fa5896f3 |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.722 [0.650, 0.790]** |
| Macro-F1 | 0.674 [0.581, 0.742] |
| Micro precision | 0.867 [0.795, 0.926] |
| Micro recall, also correct redaction recall | 0.619 [0.534, 0.706] |
| Removal coverage | 0.584 [0.463, 0.697] |
| Complete-entity redaction recall | 0.627 [0.541, 0.713] |
| Over-redaction rate, relative to current gold | 0.093 [0.032, 0.181] |
| Non-sensitive characters preserved | 0.952 [0.887, 0.999] |
| Note-level exact redaction rate | 0.476 [0.366, 0.585] |
| Invalid-output rate | 0.037 [0.000, 0.085] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 29 | 1 | 25 | 0.967 | 0.537 | 0.690 [0.582, 0.806] |
| ETÀ | 80 | 10 | 22 | 0.889 | 0.784 | 0.833 [0.772, 0.891] |
| DATA | 32 | 7 | 27 | 0.821 | 0.542 | 0.653 [0.378, 0.839] |
| LUOGO/INDIRIZZO | 15 | 6 | 22 | 0.714 | 0.405 | 0.517 [0.324, 0.667] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.667 | 0.599 |
| 2 | 17 | 0.853 | 0.844 |
| 3 | 17 | 0.660 | 0.626 |
| 4 | 16 | 0.774 | 0.700 |
| 5 | 16 | 0.675 | 0.542 |
| Mean | | 0.726 | 0.662 |
| SD | | 0.085 | 0.117 |

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
