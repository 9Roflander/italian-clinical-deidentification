# Evaluation summary: E1

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | meta-llama/Llama-3.2-3B-Instruct, revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 |
| Checkpoint | E1/seed_42 |
| Configuration | 7123858ac751 |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.588 [0.490, 0.686]** |
| Macro-F1 | 0.534 [0.408, 0.638] |
| Micro precision | 0.690 [0.594, 0.780] |
| Micro recall, also correct redaction recall | 0.512 [0.404, 0.624] |
| Removal coverage | 0.388 [0.254, 0.540] |
| Complete-entity redaction recall | 0.516 [0.407, 0.629] |
| Over-redaction rate, relative to current gold | 0.541 [0.039, 0.757] |
| Non-sensitive characters preserved | 0.957 [0.900, 0.999] |
| Note-level exact redaction rate | 0.415 [0.305, 0.524] |
| Invalid-output rate | 0.195 [0.122, 0.280] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 28 | 15 | 26 | 0.651 | 0.519 | 0.577 [0.373, 0.782] |
| ETÀ | 72 | 28 | 30 | 0.720 | 0.706 | 0.713 [0.621, 0.800] |
| DATA | 15 | 7 | 44 | 0.682 | 0.254 | 0.370 [0.123, 0.649] |
| LUOGO/INDIRIZZO | 14 | 8 | 23 | 0.636 | 0.378 | 0.475 [0.133, 0.636] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.574 | 0.568 |
| 2 | 17 | 0.757 | 0.528 |
| 3 | 17 | 0.660 | 0.658 |
| 4 | 16 | 0.396 | 0.312 |
| 5 | 16 | 0.571 | 0.520 |
| Mean | | 0.592 | 0.517 |
| SD | | 0.133 | 0.127 |

Fold means are arithmetic means of fold scores, not the pooled results. The folds overlap in training data, so they are not independent experiments.

## Output-copy fidelity

| Item | Value |
|---|---:|
| Notes with non-redacted text exactly preserved | 66 of 82 |
| Inserted characters | 5 |
| Deleted characters | 2644 |
| Modified characters | 12 |
| Outputs with unmeasurable fidelity | 1 |
| Outputs that cannot be aligned unambiguously | 15 |
| Invalid outputs, all causes | 16 |

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
