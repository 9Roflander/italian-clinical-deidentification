# Evaluation summary: E3

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | meta-llama/Llama-3.2-3B-Instruct, revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 |
| Checkpoint | E3/seed_42 |
| Configuration | 00a0cf19e492 |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.668 [0.584, 0.746]** |
| Macro-F1 | 0.607 [0.508, 0.685] |
| Micro precision | 0.812 [0.735, 0.883] |
| Micro recall, also correct redaction recall | 0.567 [0.470, 0.667] |
| Removal coverage | 0.547 [0.415, 0.670] |
| Complete-entity redaction recall | 0.583 [0.484, 0.683] |
| Over-redaction rate, relative to current gold | 0.143 [0.059, 0.254] |
| Non-sensitive characters preserved | 0.935 [0.862, 0.995] |
| Note-level exact redaction rate | 0.439 [0.329, 0.549] |
| Invalid-output rate | 0.049 [0.012, 0.098] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 26 | 1 | 28 | 0.963 | 0.481 | 0.642 [0.512, 0.781] |
| ETÀ | 76 | 13 | 26 | 0.854 | 0.745 | 0.796 [0.697, 0.883] |
| DATA | 30 | 9 | 29 | 0.769 | 0.508 | 0.612 [0.345, 0.792] |
| LUOGO/INDIRIZZO | 11 | 10 | 26 | 0.524 | 0.297 | 0.379 [0.174, 0.529] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.690 | 0.638 |
| 2 | 17 | 0.811 | 0.728 |
| 3 | 17 | 0.574 | 0.569 |
| 4 | 16 | 0.745 | 0.675 |
| 5 | 16 | 0.533 | 0.329 |
| Mean | | 0.671 | 0.588 |
| SD | | 0.116 | 0.156 |

Fold means are arithmetic means of fold scores, not the pooled results. The folds overlap in training data, so they are not independent experiments.

## Output-copy fidelity

| Item | Value |
|---|---:|
| Notes with non-redacted text exactly preserved | 78 of 82 |
| Inserted characters | 0 |
| Deleted characters | 0 |
| Modified characters | 0 |
| Outputs with unmeasurable fidelity | 4 |
| Outputs that cannot be aligned unambiguously | 4 |
| Invalid outputs, all causes | 4 |

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
