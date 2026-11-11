# Evaluation summary: E3

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | meta-llama/Llama-3.2-3B-Instruct, revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 |
| Checkpoint | E3/seed_42 |
| Configuration | 137665080a72 |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.611 [0.512, 0.710]** |
| Macro-F1 | 0.560 [0.433, 0.666] |
| Micro precision | 0.705 [0.601, 0.803] |
| Micro recall, also correct redaction recall | 0.540 [0.433, 0.656] |
| Removal coverage | 0.429 [0.293, 0.584] |
| Complete-entity redaction recall | 0.540 [0.433, 0.656] |
| Over-redaction rate, relative to current gold | 0.085 [0.033, 0.163] |
| Non-sensitive characters preserved | 0.958 [0.900, 0.999] |
| Note-level exact redaction rate | 0.427 [0.317, 0.537] |
| Invalid-output rate | 0.183 [0.110, 0.268] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 29 | 16 | 25 | 0.644 | 0.537 | 0.586 [0.385, 0.782] |
| ETÀ | 72 | 25 | 30 | 0.742 | 0.706 | 0.724 [0.631, 0.814] |
| DATA | 23 | 6 | 36 | 0.793 | 0.390 | 0.523 [0.247, 0.788] |
| LUOGO/INDIRIZZO | 12 | 10 | 25 | 0.545 | 0.324 | 0.407 [0.111, 0.593] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.615 | 0.588 |
| 2 | 17 | 0.773 | 0.588 |
| 3 | 17 | 0.547 | 0.527 |
| 4 | 16 | 0.433 | 0.356 |
| 5 | 16 | 0.736 | 0.687 |
| Mean | | 0.621 | 0.549 |
| SD | | 0.139 | 0.122 |

Fold means are arithmetic means of fold scores, not the pooled results. The folds overlap in training data, so they are not independent experiments.

## Output-copy fidelity

| Item | Value |
|---|---:|
| Notes with non-redacted text exactly preserved | 67 of 82 |
| Inserted characters | 58 |
| Deleted characters | 2516 |
| Modified characters | 7 |
| Outputs with unmeasurable fidelity | 2 |
| Outputs that cannot be aligned unambiguously | 13 |
| Invalid outputs, all causes | 15 |

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
