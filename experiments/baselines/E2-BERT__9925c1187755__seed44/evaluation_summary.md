# Evaluation summary: E2-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E2-BERT/seed_44 |
| Configuration | 9925c1187755 |
| Random seed | 44 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.775 [0.712, 0.832]** |
| Macro-F1 | 0.736 [0.652, 0.797] |
| Micro precision | 0.746 [0.672, 0.813] |
| Micro recall, also correct redaction recall | 0.806 [0.731, 0.874] |
| Removal coverage | 0.879 [0.807, 0.934] |
| Complete-entity redaction recall | 0.881 [0.826, 0.932] |
| Over-redaction rate, relative to current gold | 0.098 [0.054, 0.160] |
| Non-sensitive characters preserved | 0.999 [0.999, 1.000] |
| Note-level exact redaction rate | 0.524 [0.427, 0.634] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 39 | 19 | 15 | 0.672 | 0.722 | 0.696 [0.553, 0.818] |
| ETÀ | 90 | 17 | 12 | 0.841 | 0.882 | 0.861 [0.800, 0.918] |
| DATA | 53 | 9 | 6 | 0.855 | 0.898 | 0.876 [0.762, 0.950] |
| LUOGO/INDIRIZZO | 21 | 24 | 16 | 0.467 | 0.568 | 0.512 [0.275, 0.649] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.821 | 0.789 |
| 2 | 17 | 0.585 | 0.513 |
| 3 | 17 | 0.817 | 0.808 |
| 4 | 16 | 0.865 | 0.810 |
| 5 | 16 | 0.782 | 0.695 |
| Mean | | 0.774 | 0.723 |
| SD | | 0.110 | 0.126 |

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
