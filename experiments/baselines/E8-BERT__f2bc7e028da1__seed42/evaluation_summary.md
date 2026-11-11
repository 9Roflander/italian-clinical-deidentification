# Evaluation summary: E8-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E8-BERT/seed_42 |
| Configuration | f2bc7e028da1 |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.748 [0.675, 0.814]** |
| Macro-F1 | 0.728 [0.640, 0.793] |
| Micro precision | 0.727 [0.651, 0.798] |
| Micro recall, also correct redaction recall | 0.770 [0.682, 0.845] |
| Removal coverage | 0.870 [0.799, 0.925] |
| Complete-entity redaction recall | 0.841 [0.764, 0.907] |
| Over-redaction rate, relative to current gold | 0.093 [0.056, 0.145] |
| Non-sensitive characters preserved | 0.999 [0.999, 0.999] |
| Note-level exact redaction rate | 0.476 [0.366, 0.585] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 43 | 16 | 11 | 0.729 | 0.796 | 0.761 [0.615, 0.870] |
| ETÀ | 80 | 26 | 22 | 0.755 | 0.784 | 0.769 [0.685, 0.849] |
| DATA | 49 | 6 | 10 | 0.891 | 0.831 | 0.860 [0.676, 0.949] |
| LUOGO/INDIRIZZO | 22 | 25 | 15 | 0.468 | 0.595 | 0.524 [0.333, 0.659] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.831 | 0.804 |
| 2 | 17 | 0.857 | 0.869 |
| 3 | 17 | 0.691 | 0.702 |
| 4 | 16 | 0.596 | 0.536 |
| 5 | 16 | 0.771 | 0.706 |
| Mean | | 0.749 | 0.723 |
| SD | | 0.107 | 0.126 |

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
