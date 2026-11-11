# Evaluation summary: E9-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E9-BERT/seed_42 |
| Configuration | 09ef32838f7e |
| Random seed | 42 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.658 [0.577, 0.733]** |
| Macro-F1 | 0.608 [0.518, 0.687] |
| Micro precision | 0.619 [0.540, 0.696] |
| Micro recall, also correct redaction recall | 0.702 [0.610, 0.791] |
| Removal coverage | 0.865 [0.783, 0.929] |
| Complete-entity redaction recall | 0.869 [0.808, 0.923] |
| Over-redaction rate, relative to current gold | 0.144 [0.092, 0.206] |
| Non-sensitive characters preserved | 0.999 [0.998, 0.999] |
| Note-level exact redaction rate | 0.354 [0.256, 0.451] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 32 | 30 | 22 | 0.516 | 0.593 | 0.552 [0.396, 0.693] |
| ETÀ | 85 | 37 | 17 | 0.697 | 0.833 | 0.759 [0.703, 0.814] |
| DATA | 48 | 9 | 11 | 0.842 | 0.814 | 0.828 [0.649, 0.926] |
| LUOGO/INDIRIZZO | 12 | 33 | 25 | 0.267 | 0.324 | 0.293 [0.100, 0.471] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.826 | 0.819 |
| 2 | 17 | 0.667 | 0.639 |
| 3 | 17 | 0.667 | 0.646 |
| 4 | 16 | 0.691 | 0.536 |
| 5 | 16 | 0.353 | 0.198 |
| Mean | | 0.641 | 0.568 |
| SD | | 0.174 | 0.230 |

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
