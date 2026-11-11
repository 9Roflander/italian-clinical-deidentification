# Experimental Protocol

Status: draft for project-owner review. Items marked **Decision** need the owner's choice before the method is frozen.

## 1. Task definition

Input: an original Italian clinical note. Output: the same note with each sensitive span replaced by its placeholder and all other text unchanged.

| Category | Placeholder | Covers |
|---|---|---|
| Name | `[NOME]` | People: patients, relatives, clinicians. Full names, single names, initials |
| Age | `[ETÀ]` | A person's age |
| Date | `[DATA]` | Calendar references |
| Location or address | `[LUOGO/INDIRIZZO]` | Cities, regions, hospitals, healthcare organizations, addresses |

### Annotation rules

These rules follow Miranda et al. 2025 where it is explicit (sections 3.1, 3.2, 4.3 and Appendix A), and otherwise the conventions visible in the gold data, as measured in `audit/gold_dataset_statistics.csv`. The paper publishes categories and examples, but no full guideline or agreement figures.

| Rule | Definition | Evidence or status |
|---|---|---|
| Span limits | A span excludes surrounding whitespace and punctuation | Applied to generated data by construction |
| Ages | The number only. "anni" stays outside the span. Covers other people's ages too. Durations and study population thresholds are not ages | Their Appendix A gold example is a bare number; 94 of 101 gold ages are digits only. Generated data is held to this form |
| Partial entities | A partial match is not a correct match. The remaining text counts as residual PII | Strict metric, Phase 10 |
| Several entities in a sentence | Each is its own span, even if adjacent | |
| Repeated mentions | Each occurrence is a separate span | Their section 4.3 says annotations may repeat the same string; gold lists them without positions |
| Names | Patient, relatives and caregivers, and treating clinicians | Their prompt lists all three; initials are annotated in gold (11 mentions) but not discussed in the paper |
| Dates | Any calendar reference, from a full date to the year alone, including linking words inside it. Durations are not dates. Years inside guideline or protocol names are not patient dates | Their section 3.2 defines the range. The exclusions follow gold, which leaves these unannotated. Real dates missing from the source were added in `gold_v2` |
| Locations and hospitals | Whole name is one span. Includes countries and geographic origin, except the country of the corpus (`Italia`, never annotated). National bodies and manufacturers are not annotated | Their prompt lists origin, birthplace, specific places and facilities, and Appendix A's examples include a country |
| Nested entities | No nesting in the output: the longest span wins | 6 strings in gold are nested inside another |
| Eponyms | Medical eponyms are not names | Stated in the generation prompt |

Offsets are recovered and labelled in `src/data/build_gold.py`. Ambiguous and unplaced mentions were reviewed and resolved in `gold_v2`, with every change in `data_manifests/gold_v2_change_log.csv`.

**All new experiments use `gold_v2`.** `gold_v1` holds the published annotations and is used only for comparison with Miranda et al.

## 2. Gold data statistics

Full table for `gold_v2`: `audit/gold_dataset_statistics.csv`. `gold_v1`: `audit/gold_dataset_statistics_v1.csv`. No note text appears in either.

| Measure | Value |
|---|---|
| Notes | 82 (the source file has 83 lines including its header) |
| Annotated mentions, `gold_v2` | 252: ETÀ 102, DATA 59, NOME 54, LUOGO/INDIRIZZO 37 |
| Annotated mentions, `gold_v1` | 228: ETÀ 101, DATA 47, NOME 46, LUOGO/INDIRIZZO 34 |
| Mentions per note | 3.07 mean; 28 notes have one mention and 1 has none |
| Length | 572 to 4,043 characters, median 1,831; 28 notes under 1,000 |
| Notes with no mention of a category | NOME 65%, ETÀ 2%, DATA 79%, LUOGO 80% |
| Character offsets | All 252 verified |
| Document IDs | All 82, from the source |
| Patient linkage | Unknown |

The rare categories have few test-fold mentions (`test_fold_*` rows in the statistics file), so per-category fold scores will be unstable.

## 3. Evaluation strategy

**All 82 notes are development data.** The two added notes were not used in historical development, but they joined the same pool for simplicity. They guided earlier design decisions, including the name-free data and S2. Corrected cross-validation on them is exploratory, even when no test note reaches a fold's training pipeline. It must not be described as an untouched generalization estimate.

### Outer folds

- Frozen in `splits/fold_*.json` and `data_manifests/fold_assignments.csv`. The first 80 notes keep the historical assignment. `gold_080` and `gold_081` were added to folds 3 and 2 by a seeded draw, before any clean score existed. The earlier 80-note files are in `splits/historical_80/`.
- Every model and ablation uses the same assignment.
- Assignments will not change for any reason related to performance.
- Note-level, since patient linkage is unknown. This limitation must be disclosed. Only one pair of gold notes shares any 8-word sequence, so no duplicate documents were found.

### What outer test notes may not touch

Prompts, synthetic generation, negative examples, early stopping, hyperparameters, prompt edits, data strategy, thresholds, and manual corrections that affect training.

The post-generation audit (`src/generation/post_audit.py`) reads test notes only to verify separation. It never filters records, because filtering would use test notes to shape training data. A violation stops the fold for review.

### Validation

**Decision D2.** Recommendation: keep the historical Phase 2 hyperparameters frozen and use no inner validation for tuning, apart from memory-related changes required by the RTX 3070. Changes for memory must be fixed before any outer score is seen and applied to all folds. If tuning is wanted, an inner split must come from the 64 training notes only.

### Independent test set

The strongest generalization claim needs a new, independently annotated set that has not been exposed to generation, training, or selection. Method, checkpoints, prompts and evaluation code are frozen before it is touched.

**Decision D3.** This cannot be obtained from the repository. Possible sources to investigate:

- Other Italian documents from the E3C corpus outside the 82 Miranda et al. rows. Availability and annotation effort are unverified.
- Two CRF documents outside the gold set (`IT101195`, `IT102068`). They are excluded from training, but far too few for evaluation and would need annotation.

If no new set is obtained, the paper must state that all results are development estimates.

## 4. Data composition

| Source | Per fold | Rule |
|---|---|---|
| Gold training notes | 65 or 66 unique, 325 or 330 after 5x oversampling | Oversampled only after the split; targets from corrected spans |
| Generated named notes | Up to 1,218 | Fold-specific |
| Generated name-free notes | Up to 482 | Fold-specific; together 1,700 per fold, by owner decision |
| Generated negative notes | Up to 80 | Zero entities and no residual patterns |
| `synthetic-crf-train` | **None** | 78 excluded, 2 quarantined; see `data_manifests/negative_audit.csv` |
| Historical S1 and S2 | **None** | Audit reference only |

**Decision D4.** Named share. The pool keeps the historical 72% named, as the guide specifies, but only 35% of gold notes contain a name. The entity-distribution ablation can test a lower share by selecting subsets.

## 5. Experiment plan

The experiment matrix E0 to E6 and research questions RQ1 to RQ4 are in `audit/EVALUATION_PROTOCOL.md`, section 9, which the owner approved on 2026-11-05. It replaces the guide's Phase 9 matrix A to F unless the owner decides otherwise. All use the same frozen folds.

- Style-anchoring ablation: both arms use the same generation plan, seeds and validator. The rendered prompts differ only by the anchor block, which a test enforces.
- Volume ablation: `type_order` gives nested subsets of the same distribution.
- Entity ablation: subsets chosen from the same pool.

Uncertainty is reported as the guide's Phase 11 requires. Five overlapping folds are not five independent experiments.

## 6. Privacy rules for the process

- Reports, manifests and logs contain IDs, hashes and counts only.
- Anchors are masked by default before leaving the machine. Raw mode needs an explicit flag.
- Sending masked anchors to Gemini was approved by the project owner on 2026-11-05. No institutional approval is documented, and the paper must report the actual arrangement.
- Generated notes are not assumed to be anonymous.
