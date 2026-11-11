# Evaluation Protocol

Approved by the project owner on 2026-11-05, before any clean model result existed. Part A condenses the approved protocol, keeping every requirement. Its example strings are left out or replaced by character positions, so this committable file quotes no note-like text. Part B records how each open choice was implemented. The code is `src/evaluation/`, version 1.0.1, with settings in `configs/evaluation.yaml`. Version 1.0.1 changes run provenance only, not scoring. See `reports/INFRASTRUCTURE_PATCH_REVIEW.md`.

Changing a scoring rule requires a new `EVALUATOR_VERSION` and a written reason. The evaluator must not be changed to improve reported model performance.

# Part A: The approved protocol, condensed

## 1. Existing project configuration

The primary evaluation uses `gold_v2`, the reviewed dataset of 82 notes, with these mentions:

| Entity category | Number of mentions |
|---|---:|
| NOME | 54 |
| ETÀ | 102 |
| DATA | 59 |
| LUOGO/INDIRIZZO | 37 |
| Total | 252 |

Use the existing frozen five-fold assignments. Do not regenerate the folds. Do not modify gold annotations or offsets without creating a new version and documenting the changes.

The task involves four categories only: NOME, ETÀ, DATA, LUOGO/INDIRIZZO. Do not claim that these cover every possible form of personally identifiable information.

## 2. Primary evaluation: strict entity-level span matching

Each gold or predicted entity is `(start_offset, end_offset, entity_type)`, a half-open interval `[start_offset, end_offset)` in Python Unicode string positions of the original source note.

A predicted entity is correct only when the beginning character, ending character and category all match exactly. Partial matches receive no credit in the primary metric.

For a gold NOME at characters 20–31:

| Prediction | Result |
|---|---|
| NOME, 20–31 | TP |
| NOME, 20–30 | FP + FN |
| ETÀ, 20–31 | FP + FN |
| NOME, 19–31 | FP + FN |
| No prediction | FN |
| Additional incorrect prediction | FP |

Matching is one-to-one. Duplicate predictions receive no additional true positives. A wrong category is a false positive for the predicted category and a false negative for the gold category.

## 3. Required primary metrics

For each category: Precision = TP / (TP + FP). Recall = TP / (TP + FN). F1 = 2 × Precision × Recall / (Precision + Recall).

Micro-F1 aggregates TP, FP and FN across the four categories first. Macro-F1 is the arithmetic mean of the four category F1 scores, which prevents the frequent ETÀ category from dominating.

Primary reported metrics: strict micro-F1, macro-F1, and per-category precision, recall and F1. Do not report only an overall F1-score.

## 4. Additional privacy-oriented evaluation

NER performance is not equivalent to successful de-identification.

- **4.1 Correct redaction recall.** A gold entity is redacted only when its exact original span is replaced with the correct category placeholder. Strict span-level recall.
- **4.2 Gold-span removal coverage.** With G the union of gold sensitive character positions and P the union of predicted redaction positions, Removal Coverage = |G ∩ P| / |G|. Supplementary; it does not replace strict F1.
- **4.3 Over-redaction rate.** |P \ G| / |P|, reported relative to the current gold annotations, not as proof that such text is non-sensitive.
- **4.4 Output-copy fidelity.** Percentage of notes with exactly preserved non-redacted text; inserted, deleted and modified characters; outputs that cannot be aligned unambiguously. The task is inline redaction, not rewriting or summarizing.
- **4.5 Note-level exact redaction rate.** Percentage of notes where all gold entities are correctly redacted, no additional entities are redacted, and all non-sensitive text is preserved. Supplementary.

## 5. Recovering predicted spans from model output

Placeholder counts alone cannot give strict span-level F1. The evaluator must know the source offsets of every predicted placeholder.

**Preferred for new experiments:** inference exposes structured segments of original text with a predicted label, such as `{"text": "...", "label": "O"}` and `{"text": "...", "label": "ETÀ"}`. Concatenated segment text must reproduce the input exactly. Offsets come from cumulative character lengths, and deterministic post-processing then replaces labelled segments with placeholders.

**If a checkpoint only produces inline placeholders,** use a conservative exact alignment adapter: recognize the four official placeholders, align unchanged literal text in order, recover spans from the gaps, accept an alignment only when it is unique and exact, never use gold annotations to force an alignment, do not use fuzzy matching to manufacture correct spans, and mark ambiguous or invalid outputs explicitly.

Unalignable placeholders receive no credit. Count their presence as invalid predictions, count unmatched gold entities as false negatives, and report the number of affected outputs. Do not silently discard invalid notes.

## 6. Unit tests for the evaluator

| Test | Expected outcome |
|---|---|
| Perfect redaction | Precision, recall and F1 = 1.0 |
| Empty output | Recall and F1 = 0 |
| Unchanged input | Recall and F1 = 0 |
| Wrong entity category | FP + FN |
| Wrong boundary | FP + FN |
| Partial boundary overlap | No strict TP |
| Spurious placeholder | FP |
| Duplicate prediction | One TP maximum; additional prediction FP |
| Repeated occurrence of same name | Evaluate each occurrence independently |
| Ambiguous alignment | Invalid/unalignable; no guessed TP |
| Unicode accents | Correct exact offsets |
| Punctuation and apostrophes | Correct exact offsets |
| Truncated output | Invalid output, explicitly reported |
| No-entity note | No false positive when output is unchanged |
| Overlapping predictions | Invalid output, explicitly reported |
| Unknown placeholder | Invalid prediction |

**Critical test:** an empty output must never achieve a perfect score. Add randomized tests for offset handling, rendering and repeated entities.

## 7. Five-fold cross-validation protocol

For each frozen fold: train only on training-fold gold notes; use only training-fold notes as synthetic-data anchors; generate synthetic data from training-fold statistics; keep held-out notes out of prompts, data generation, parameter selection and training; train with the frozen configuration; evaluate on held-out notes; save every prediction and its results.

Each of the 82 notes receives exactly one out-of-fold prediction per model, configuration and seed. Do not oversample test notes. Audit the Phase 1 domain-adaptation corpora for overlap with the gold notes before using them, and verify the outstanding DART provenance issue.

## 8. Statistical analysis

**8.1 Per fold.** For each model, macro-F1 and micro-F1 for each fold, the mean and standard deviation across folds, and per-category results.

**8.2 Pooled out-of-fold.** Combine TP, FP and FN from all held-out folds for pooled micro-precision, micro-recall, micro-F1, per-category F1 and macro-F1. Distinguish pooled results from the arithmetic mean of fold scores.

**8.3 Confidence intervals.** Paired document-level bootstrap: resample document IDs with replacement, within each of the five folds; use the same sampled documents when comparing two models; recalculate micro-F1 and macro-F1 per sample; at least 2,000 iterations, 10,000 preferred; 95% intervals, including for differences between models. Save the seed and configuration.

These intervals quantify document-sampling uncertainty conditional on the trained fold models and do not capture all training randomness. If feasible, run three fixed training seeds for the principal configurations. If only one seed is used, acknowledge this. Do not rely on an ordinary paired t-test over five folds as the primary significance test.

## 9. Experimental comparisons

| Experiment | Model configuration | Research question |
|---|---|---|
| E0 | Unchanged-input baseline | Does doing no redaction receive zero F1? |
| E1 | Base Llama + gold-only | Gold-only baseline |
| E2 | Base Llama + gold + non-anchored synthetic data | Does synthetic data help without style anchors? |
| E3 | Base Llama + gold + style-anchored synthetic data | Does style conditioning help? |
| E4 | Domain-adapted Llama + gold-only | Does domain adaptation help? |
| E5 | Domain-adapted Llama + gold + style-anchored synthetic data | Main proposed pipeline |
| E6 | Italian encoder-based token-classification model | Comparison with conventional NER |

- **RQ1, domain adaptation:** E1 against E4, holding task-specific training data constant.
- **RQ2, synthetic data:** gold-only against gold plus synthetic data, holding the model and other settings constant.
- **RQ3, style conditioning:** anchored against non-anchored data, with matched accepted case IDs wherever possible.
- **RQ4, entity-distribution balancing:** balanced against unbalanced synthetic data, holding other conditions constant.

If the full matrix is too expensive, prioritize comparisons that isolate the paper's central contributions. Do not claim a technique improves performance if several variables changed at once.

## 10. Manual privacy and annotation validation

Prefer independent second annotation of all 82 notes. If infeasible, select a documented subset representing all four categories, ambiguous boundaries, repeated entities and notes without identifiers. Report exact-span inter-annotator precision, recall and F1 by category, followed by adjudication.

Also conduct a blinded manual audit of the principal model's held-out outputs for residual sensitive information, incorrect categories, missing redactions, unnecessary redactions, and changes to non-sensitive clinical content.

Gold-span metrics cannot reveal identifiers missed during gold annotation. Do not claim complete anonymisation based only on the four-category evaluation.

## 11. Required evaluation output files

`run_manifest.json`, `predictions_oof.jsonl`, `alignment_report.json`, `per_note_metrics.csv`, `per_fold_metrics.csv`, `per_category_metrics.csv`, `pooled_metrics.json`, `error_breakdown.csv`, `bootstrap_results.json`, `evaluation_summary.md`.

The summary includes gold version, notes and entities, fold assignments, model and checkpoint, seed, strict micro-F1, macro-F1, per-category precision, recall and F1, confidence intervals, invalid-output rate, output-copy fidelity, removal coverage, over-redaction rate, and known limitations.

Do not put raw clinical notes, generated notes, outputs containing clinical text, sensitive prompts or API keys into public version control.

## 12. Acceptance criteria

Do not start full model training until all these hold:

1. Strict span-level precision, recall and F1 are implemented.
2. Empty output receives zero F1 when gold entities are present.
3. Perfect synthetic predictions achieve F1 = 1.0.
4. Wrong-type and wrong-boundary cases are handled correctly.
5. Repeated entities are evaluated by individual offsets.
6. Unicode and punctuation tests pass.
7. Ambiguous output alignment is detected rather than guessed.
8. Redaction coverage and over-redaction are implemented.
9. The five frozen folds are preserved.
10. Out-of-fold metrics can be reproduced from saved predictions.
11. Confidence intervals can be calculated.
12. All results are linked to an immutable experiment manifest and evaluator version.

If any criterion fails, stop and report the problem.

## 13. Expected scientific outcome

The paper should answer: whether compact language models can accurately identify Italian clinical PII; whether synthetic data improves on gold-only training; whether style-conditioned generation has measurable benefits over non-anchored data; and whether domain adaptation helps with limited annotated data.

The primary scientific result is strict span-level micro-F1, supported by macro-F1, per-category performance, privacy measures and statistical uncertainty. The paper must acknowledge the limited gold size, annotation reliability limits, and absent external validation if unresolved. **Do not make claims that exceed the evidence produced by these experiments.**

# Part B: Implementation decisions

These are choices the protocol left open. They were fixed before any model output existed.

| Topic | Decision | Reason |
|---|---|---|
| Invalid predictions | Each counts as a false positive: an official placeholder in its category, an unknown one in micro-F1 only | Leaving them out would inflate precision |
| Unknown placeholder | Treated as a gap, so the other placeholders in that output can still score. The span earns no credit | The protocol calls this an invalid prediction, while truncation and overlap are invalid outputs |
| Permitted normalization | NFC, and outer whitespace when the note has none. Recorded per note. Nothing else | NFC forms are the same text. Neither can change a span |
| Internal whitespace | Any change makes the output invalid. It is named as its own status for diagnosis and earns no credit | The protocol requires exact alignment |
| Span trimming | None. A span including a neighbouring space is a boundary error | Strict means exact |
| Placeholder that replaced nothing | The output is invalid, because every gap must hold at least one character | Zero-length spans would multiply readings |
| Duplicate and overlap | An identical span repeated is a false positive. Any other overlap makes the output invalid | The two protocol rows require this split |
| Undefined values | P, R and F1 stay undefined when the denominator is zero, and are never 1.0 | An empty output must never look perfect |
| Macro-F1 | Averages categories that have gold or predictions. A category with predictions but no gold scores 0 | A fold or replicate may lack a category |
| Removal coverage | Spans with wrong or unknown tags count, because the text is gone. Invalid outputs cover nothing | Keeps tag errors in F1 and text removal here, and gives invalid outputs no privacy credit |
| Fidelity on invalid outputs | Character diff, reported as damage only. It never creates spans or credit | The protocol forbids fuzzy matching for spans, not for describing errors |
| Reported once | Correct redaction recall (4.1) equals strict micro-recall, so it has one number with two labels. Residual PII equals 1 minus coverage, so it is not reported separately | Owner's instruction to avoid redundant metrics |
| Kept from the earlier guide | Complete-entity redaction recall and non-sensitive character preservation | Not redundant: coverage can be high while every entity leaks a remainder |
| Bootstrap | Percentile intervals, 10,000 iterations, seed 42. Undefined replicates are counted and reported | Protocol 8.3 |
| Fold SD | Sample SD (ddof = 1) | Five scores from one assignment |
| Historical metrics | `paper_metric` is copied unchanged, and a test compares its syntax tree with the original. Miranda et al.'s equations are restated in project code, because their repository has no licence file, and are tested against recorded runs of their function | Protocol scope: historical comparison only |
| File locations | `predictions_oof.jsonl` in `predictions_secure/<run>/` (git-ignored). The other nine in `experiments/<stage>/<run>/`, with IDs, offsets and numbers only | Protocol 11 |
| Immutability | Run folders cannot be overwritten, files are read-only, and the manifest holds hashes of files, predictions, gold, folds and evaluator code plus the git commit. Runs under official results folders refuse changed experiment-critical source, with no bypass. Runs elsewhere are recorded as development runs | Criterion 12 |

## Where the protocol and the repository differ

- **`gold_v1`.** The protocol calls it the 80-note historical dataset. In this repository, `gold_v1` has 82 notes with source annotations (228 mentions), and the 80-note file is `original_work/data/gold_standard_80.json`. The primary evaluation is unaffected.
- **RQ4 has no experiment row.** E0 to E6 contain no balanced and unbalanced pair.
- **Guide Phase 9.** Its matrix A to F includes a no-negatives comparison, which E0 to E6 omit. This protocol's matrix is used unless the owner decides otherwise.
