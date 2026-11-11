# Evaluation summary: E1-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E1-BERT/seed_43 |
| Configuration | ae91fdcd33e2 |
| Random seed | 43 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.585 [0.497, 0.675]** |
| Macro-F1 | 0.491 [0.396, 0.577] |
| Micro precision | 0.565 [0.472, 0.664] |
| Micro recall, also correct redaction recall | 0.607 [0.513, 0.701] |
| Removal coverage | 0.785 [0.681, 0.871] |
| Complete-entity redaction recall | 0.762 [0.683, 0.834] |
| Over-redaction rate, relative to current gold | 0.089 [0.046, 0.143] |
| Non-sensitive characters preserved | 0.999 [0.999, 1.000] |
| Note-level exact redaction rate | 0.402 [0.305, 0.500] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 22 | 18 | 32 | 0.550 | 0.407 | 0.468 [0.302, 0.615] |
| ETÀ | 84 | 47 | 18 | 0.641 | 0.824 | 0.721 [0.643, 0.798] |
| DATA | 42 | 42 | 17 | 0.500 | 0.712 | 0.587 [0.343, 0.772] |
| LUOGO/INDIRIZZO | 5 | 11 | 32 | 0.312 | 0.135 | 0.189 [0.045, 0.326] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.836 | 0.779 |
| 2 | 17 | 0.552 | 0.427 |
| 3 | 17 | 0.320 | 0.183 |
| 4 | 16 | 0.804 | 0.708 |
| 5 | 16 | 0.420 | 0.234 |
| Mean | | 0.586 | 0.466 |
| SD | | 0.229 | 0.270 |

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
