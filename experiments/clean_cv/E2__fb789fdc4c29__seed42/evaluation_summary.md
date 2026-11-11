# Evaluation summary: E2

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | meta-llama/Llama-3.2-3B-Instruct, revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 |
| Checkpoint | E2/seed_42 |
| Configuration | fb789fdc4c29 |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.614 [0.520, 0.706]** |
| Macro-F1 | 0.570 [0.453, 0.667] |
| Micro precision | 0.718 [0.626, 0.801] |
| Micro recall, also correct redaction recall | 0.536 [0.432, 0.650] |
| Removal coverage | 0.412 [0.279, 0.568] |
| Complete-entity redaction recall | 0.536 [0.432, 0.650] |
| Over-redaction rate, relative to current gold | 0.138 [0.060, 0.235] |
| Non-sensitive characters preserved | 0.954 [0.892, 0.999] |
| Note-level exact redaction rate | 0.402 [0.305, 0.512] |
| Invalid-output rate | 0.171 [0.098, 0.256] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 29 | 10 | 25 | 0.744 | 0.537 | 0.624 [0.434, 0.788] |
| ETÀ | 71 | 26 | 31 | 0.732 | 0.696 | 0.714 [0.620, 0.805] |
| DATA | 21 | 7 | 38 | 0.750 | 0.356 | 0.483 [0.194, 0.756] |
| LUOGO/INDIRIZZO | 14 | 10 | 23 | 0.583 | 0.378 | 0.459 [0.200, 0.605] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.615 | 0.588 |
| 2 | 17 | 0.757 | 0.517 |
| 3 | 17 | 0.545 | 0.538 |
| 4 | 16 | 0.500 | 0.459 |
| 5 | 16 | 0.690 | 0.650 |
| Mean | | 0.622 | 0.550 |
| SD | | 0.104 | 0.073 |

Fold means are arithmetic means of fold scores, not the pooled results. The folds overlap in training data, so they are not independent experiments.

## Output-copy fidelity

| Item | Value |
|---|---:|
| Notes with non-redacted text exactly preserved | 68 of 82 |
| Inserted characters | 55 |
| Deleted characters | 28 |
| Modified characters | 5 |
| Outputs with unmeasurable fidelity | 3 |
| Outputs that cannot be aligned unambiguously | 11 |
| Invalid outputs, all causes | 14 |

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
