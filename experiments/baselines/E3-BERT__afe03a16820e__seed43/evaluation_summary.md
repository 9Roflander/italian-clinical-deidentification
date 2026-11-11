# Evaluation summary: E3-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E3-BERT/seed_43 |
| Configuration | afe03a16820e |
| Random seed | 43 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.622 [0.541, 0.703]** |
| Macro-F1 | 0.576 [0.503, 0.639] |
| Micro precision | 0.624 [0.541, 0.709] |
| Micro recall, also correct redaction recall | 0.619 [0.532, 0.708] |
| Removal coverage | 0.802 [0.704, 0.883] |
| Complete-entity redaction recall | 0.794 [0.718, 0.865] |
| Over-redaction rate, relative to current gold | 0.090 [0.057, 0.139] |
| Non-sensitive characters preserved | 0.999 [0.999, 1.000] |
| Note-level exact redaction rate | 0.390 [0.293, 0.488] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 33 | 36 | 21 | 0.478 | 0.611 | 0.537 [0.400, 0.672] |
| ETÀ | 73 | 27 | 29 | 0.730 | 0.716 | 0.723 [0.648, 0.795] |
| DATA | 40 | 5 | 19 | 0.889 | 0.678 | 0.769 [0.545, 0.884] |
| LUOGO/INDIRIZZO | 10 | 26 | 27 | 0.278 | 0.270 | 0.274 [0.145, 0.431] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.374 | 0.357 |
| 2 | 17 | 0.783 | 0.749 |
| 3 | 17 | 0.531 | 0.477 |
| 4 | 16 | 0.863 | 0.811 |
| 5 | 16 | 0.591 | 0.479 |
| Mean | | 0.628 | 0.574 |
| SD | | 0.196 | 0.195 |

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
