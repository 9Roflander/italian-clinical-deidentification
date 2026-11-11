# Evaluation summary: E2

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | meta-llama/Llama-3.2-3B-Instruct, revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 |
| Checkpoint | E2/seed_44 |
| Configuration | ccc3fa5896f3 |
| Random seed | 44 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.706 [0.639, 0.771]** |
| Macro-F1 | 0.658 [0.566, 0.728] |
| Micro precision | 0.837 [0.759, 0.907] |
| Micro recall, also correct redaction recall | 0.611 [0.532, 0.692] |
| Removal coverage | 0.570 [0.457, 0.675] |
| Complete-entity redaction recall | 0.623 [0.544, 0.705] |
| Over-redaction rate, relative to current gold | 0.113 [0.043, 0.209] |
| Non-sensitive characters preserved | 0.951 [0.886, 0.999] |
| Note-level exact redaction rate | 0.439 [0.341, 0.537] |
| Invalid-output rate | 0.037 [0.000, 0.085] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 26 | 2 | 28 | 0.929 | 0.481 | 0.634 [0.527, 0.750] |
| ETÀ | 81 | 15 | 21 | 0.844 | 0.794 | 0.818 [0.752, 0.882] |
| DATA | 31 | 7 | 28 | 0.816 | 0.525 | 0.639 [0.375, 0.815] |
| LUOGO/INDIRIZZO | 16 | 6 | 21 | 0.727 | 0.432 | 0.542 [0.345, 0.676] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.783 | 0.697 |
| 2 | 17 | 0.735 | 0.576 |
| 3 | 17 | 0.598 | 0.569 |
| 4 | 16 | 0.769 | 0.664 |
| 5 | 16 | 0.667 | 0.546 |
| Mean | | 0.710 | 0.611 |
| SD | | 0.077 | 0.066 |

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
