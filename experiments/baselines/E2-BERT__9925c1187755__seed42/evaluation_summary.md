# Evaluation summary: E2-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E2-BERT/seed_42 |
| Configuration | 9925c1187755 |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.750 [0.682, 0.811]** |
| Macro-F1 | 0.727 [0.647, 0.787] |
| Micro precision | 0.723 [0.652, 0.791] |
| Micro recall, also correct redaction recall | 0.778 [0.698, 0.849] |
| Removal coverage | 0.880 [0.803, 0.939] |
| Complete-entity redaction recall | 0.861 [0.793, 0.922] |
| Over-redaction rate, relative to current gold | 0.091 [0.055, 0.138] |
| Non-sensitive characters preserved | 0.999 [0.999, 1.000] |
| Note-level exact redaction rate | 0.500 [0.402, 0.598] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 43 | 21 | 11 | 0.672 | 0.796 | 0.729 [0.596, 0.839] |
| ETÀ | 83 | 28 | 19 | 0.748 | 0.814 | 0.779 [0.700, 0.853] |
| DATA | 49 | 6 | 10 | 0.891 | 0.831 | 0.860 [0.679, 0.951] |
| LUOGO/INDIRIZZO | 21 | 20 | 16 | 0.512 | 0.568 | 0.538 [0.339, 0.685] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.850 | 0.825 |
| 2 | 17 | 0.835 | 0.819 |
| 3 | 17 | 0.504 | 0.515 |
| 4 | 16 | 0.824 | 0.753 |
| 5 | 16 | 0.769 | 0.697 |
| Mean | | 0.756 | 0.722 |
| SD | | 0.144 | 0.127 |

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
