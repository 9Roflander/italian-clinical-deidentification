# Evaluation summary: E9-BERT

| Item | Value |
|---|---|
| Gold dataset | gold_v2, SHA-256 `c0f66a9639ffb9af` |
| Notes and entities | 82 notes, 252 entities |
| Fold assignments | Frozen, held-out sizes [16, 17, 17, 16, 16] |
| Model | dbmdz/bert-base-italian-xxl-cased, revision 9bdf84aedf00aa4e99b01a6f2c76f6070ac7240b |
| Checkpoint | E9-BERT/seed_44 |
| Configuration | 09ef32838f7e |
| Random seed | 44 |

## Pooled out-of-fold results

Intervals are 95%, from a document-level bootstrap stratified by fold, 10,000 iterations, seed 42. They are conditional on the trained fold models.

| Metric | Value [interval] |
|---|---|
| **Strict micro-F1 (primary)** | **0.734 [0.661, 0.805]** |
| Macro-F1 | 0.694 [0.612, 0.760] |
| Micro precision | 0.699 [0.623, 0.779] |
| Micro recall, also correct redaction recall | 0.774 [0.695, 0.848] |
| Removal coverage | 0.878 [0.818, 0.926] |
| Complete-entity redaction recall | 0.853 [0.793, 0.907] |
| Over-redaction rate, relative to current gold | 0.088 [0.049, 0.141] |
| Non-sensitive characters preserved | 0.999 [0.999, 1.000] |
| Note-level exact redaction rate | 0.524 [0.427, 0.622] |
| Invalid-output rate | 0.000 [0.000, 0.000] |

## Per category, strict

| Category | TP | FP | FN | Precision | Recall | F1 [interval] |
|---|---:|---:|---:|---:|---:|---|
| NOME | 46 | 21 | 8 | 0.687 | 0.852 | 0.760 [0.639, 0.866] |
| ETÀ | 84 | 14 | 18 | 0.857 | 0.824 | 0.840 [0.777, 0.901] |
| DATA | 48 | 14 | 11 | 0.774 | 0.814 | 0.793 [0.610, 0.908] |
| LUOGO/INDIRIZZO | 17 | 35 | 20 | 0.327 | 0.459 | 0.382 [0.203, 0.527] |

Unknown placeholders counted as false positives without a category: 0. They lower micro-F1 only.

## Per fold

| Fold | Notes | Micro-F1 | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 16 | 0.596 | 0.552 |
| 2 | 17 | 0.841 | 0.823 |
| 3 | 17 | 0.690 | 0.670 |
| 4 | 16 | 0.821 | 0.766 |
| 5 | 16 | 0.750 | 0.653 |
| Mean | | 0.740 | 0.693 |
| SD | | 0.100 | 0.105 |

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
