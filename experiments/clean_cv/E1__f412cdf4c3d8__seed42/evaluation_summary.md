# Evaluation summary: E1

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | meta-llama/Llama-3.2-3B-Instruct, revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 |
| Checkpoint | E1/seed_42 |
| Configuration | f412cdf4c3d8 |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.652 [0.576, 0.724]** |
| Macro-F1 | 0.560 [0.440, 0.640] |
| Micro precision | 0.863 [0.800, 0.919] |
| Micro recall, also correct redaction recall | 0.524 [0.437, 0.616] |
| Removal coverage | 0.407 [0.281, 0.539] |
| Complete-entity redaction recall | 0.528 [0.440, 0.620] |
| Over-redaction rate, relative to current gold | 0.132 [0.044, 0.263] |
| Non-sensitive characters preserved | 0.943 [0.877, 0.999] |
| Note-level exact redaction rate | 0.439 [0.329, 0.549] |
| Invalid-output rate | 0.037 [0.000, 0.085] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 30 | 2 | 24 | 0.938 | 0.556 | 0.698 [0.583, 0.808] |
| ETÀ | 77 | 8 | 25 | 0.906 | 0.755 | 0.824 [0.754, 0.890] |
| DATA | 18 | 5 | 41 | 0.783 | 0.305 | 0.439 [0.049, 0.649] |
| LUOGO/INDIRIZZO | 7 | 6 | 30 | 0.538 | 0.189 | 0.280 [0.121, 0.406] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.553 | 0.447 |
| 2 | 17 | 0.761 | 0.597 |
| 3 | 17 | 0.689 | 0.641 |
| 4 | 16 | 0.705 | 0.617 |
| 5 | 16 | 0.550 | 0.382 |
| Mean | | 0.651 | 0.537 |
| SD | | 0.095 | 0.115 |

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
