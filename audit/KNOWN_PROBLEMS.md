# Known Problems

Every problem below was checked independently against code and data. Paths are relative to `original_work/` unless noted. Numbers come from `audit/results/` and `data_manifests/`.

Figures describe the historical 80-note data unless they say otherwise. The rebuilt 82-note set is evaluated in `reports/GOLD_DATA_EVALUATION.md`.

Severity:

- **Critical**: invalidates reported scores or conclusions.
- **High**: materially affects a result or claim.
- **Medium**: weakens rigor or reproducibility.
- **Low**: a wording or documentation fix.

## Summary

| ID | Problem | Severity | Status |
|---|---|---|---|
| A | Test notes in the negative training corpus | Critical | Confirmed, with corrections |
| B | Fixed anchors and hard-coded strings from test notes | High | Confirmed, with corrections |
| C | S1 and S2 generated once, S2 after seeing CV results | High | Confirmed |
| D | Phase 1 not reproducible, and not applied to Gemma | High | Confirmed |
| E | Truncated input credited as correct redaction | Critical | Confirmed, with corrections |
| F | No per-note predictions | High | Confirmed |
| G | Style-anchoring control is confounded | Critical | Confirmed |
| H | Gold training targets corrupted by substring replacement | Critical | **New** |
| I | The metric rewards wrong output and penalizes correct output | Critical | **New** |
| J | Metric differs from the cited definition | Medium | **New**, confirmed in their code |
| K | Long training examples lose their target at `max_length` | High | **New**, trl version unverified |
| L | Synthetic distributions far from gold across all categories | High | **New** |
| M | Synthetic targets contain inconsistencies and unredacted PII patterns | High | **New** |
| N | BERT labels are built by substring matching on incomplete lists | High | **New** |
| O | "Chunking hurt BERT" is probably a scoring artifact | High | **New**, inferred |
| P | Gold annotations lack offsets and may be incomplete | High | **New** |
| Q | Generation parameters locked together; no children prompted | Medium | **New** |
| R | Other errors in the submitted paper | Medium | **New** in part |
| S | Secrets and note text in a public repository | Critical | **New** |
| T | Original generator model is being retired | High | **New** |
| U | Training hardware has less memory than the original | Medium | **New**, from project owner |
| V | Source file differs from the old gold file and from the paper's description | High | **New** |

## Corrections to the reconstruction guide

The guide asked for its findings to be verified, not assumed. These details changed:

| Guide says | Evidence shows |
|---|---|
| 700 tokens is an "output cutoff" | It is **input** truncation (`src/train_evaluate_cv_v4.py:176`). The output limit is `max_new_tokens=1024`. |
| Duplicates are "word for word" | 0 rows are byte-identical, 66 match after trimming whitespace, 75 after normalizing whitespace. A 76th row (`crf_048`) is a near-copy of `gold_050` with two edits. The rest of the claim holds. |
| "13 of 225 mentions" is general | That is the Llama tokenizer. Gemma's is 11 of 225, and old BERT's is 19 of 225. |
| Prompts contain "examples from notes #0 and #1" | They are entity strings, not note text. Two strings on `src/generate_synthetic_v2.py:115` occur in those notes. One is an exact annotation of `gold_000`. Derivation is inferred. |
| Target example `[NOME] ha [ETÀ] e vive a ...` and BIO `69 B-ETÀ / anni I-ETÀ` | 91 of 98 gold ages are digits only and "anni" stays outside the entity, as in the paper's Table 1. The BIO example needs the number alone. |
| BIO example tag `B-LUOGO` | The project's type is `LUOGO/INDIRIZZO`. |
| The source file has 83 notes (project owner) | 82 notes. The file has 83 lines including its header. |

## A. Test notes in the negative training corpus

- **Evidence:** `audit/results/v2_crf_overlap.json`, `data_manifests/negative_audit.csv`. 75 distinct gold notes have an exact twin in `NLP-FBK/synthetic-crf-train`; 73 of them contain annotated PII, holding 207 of 225 mentions. `crf_048` is a near-copy of `gold_050`. Test notes affected per fold: 16, 13, 14, 16, 16 of 16.
- **Not found in CRF:** `gold_053`, `gold_055`, `gold_061`, `gold_066`, `gold_079`. No shared 8-word sequence with any CRF row.
- **Remaining CRF rows:** `crf_074`, `crf_077`, `crf_078`, `crf_079` are separate documents. All four match age-like patterns, and none has PII annotations.
- **Code:** `get_train_dataset`, `src/train_evaluate_cv_v4.py:133-135`; `src/train_evaluate_cv_v3.py:118-120`; `src/train_evaluate_cv_gemma.py:126-128`; `load_crf_rows`, `src/train_evaluate_cv_bert.py:123-128`; `process_crf`, `src/train_redact_v3.py:101-106`.
- **Affected:** every reported LLM and BERT score, the single-split results, and C2.
- **Direction:** it probably lowers scores, since it teaches the model to copy those notes unredacted. Only a rerun can show this.
- **On the rebuilt 82 notes:** the two added notes are also in CRF, so 77 of 82 notes have exact twins. Per fold: 16 of 16, 14 of 17, 15 of 17, 16 of 16, 16 of 16. Rows `crf_074` (`IT101195`) and `crf_079` (`IT102068`) stay quarantined.
- **Correction:** no CRF row is permitted for training. `data_manifests/negative_audit.csv` now records 78 excluded and 2 quarantined. New negatives are generated and validated.

## B. Fixed anchors and hard-coded strings

- **Evidence:** `audit/results/v1_folds_anchors.json`. Anchors are `gold_034`, `gold_013`, `gold_072`, test notes in folds 2, 3, 4. `gold_072` is also in the historical 16-note split.
- **Code:** `load_style_anchors`, `src/generate_synthetic_v2.py:76-84`; called at `generate_synthetic_parallel.py:75`, `generate_noname.py:133`, `generate_synthetic_v2.py:217`; hard-coded strings at `generate_synthetic_v2.py:115`.
- **Other readers of gold:** `load_split` (`src/train_redact_v3.py:66`), `run_cv` in each CV script, and the two `main` functions in the eval scripts. None of these select anchors.
- **Measured exposure, which limits this problem:** the longest verbatim run between any of the 1,676 notes and any gold note is 9 words, and 7 for anchors. No synthetic note equals a gold note. Anchor name strings almost never appear. Synthetic notes are not measurably closer to anchors than to other gold notes (TF-IDF ranks 11 to 53 of 80). This shows little surface copying. It does not rule out subtler style influence.
- **Direct score bound for Llama v4, with false positives held constant:** 0.628 to 0.660 without the anchors; 0.577 to 0.678 also without `gold_000` and `gold_001`.
- **Correction:** `src/generation/anchor_selection.py` accepts training notes only, and templates contain no concrete values.

## C. S1 and S2 reused across folds

- **Evidence:** No CV script calls Gemini. Every fold reloads the same files (`src/train_evaluate_cv_v4.py:26-33`, `:127-129`), and BERT's comment says "shared across folds" (`src/train_evaluate_cv_bert.py:286`). The S2 motivation is at `src/train_evaluate_cv_v4.py:12`. The name-free data followed inspection of NOME errors on the 16-note split (`FINAL_REPORT.md` 3.4).
- **Consequence:** the 80 notes guided design decisions, so CV on them is development evaluation. This cannot be fixed by regeneration alone.
- **Correction:** per-fold generation in `synthetic_data/`, and a protocol that names the 80 notes as development data (`audit/EXPERIMENTAL_PROTOCOL.md`).

## D. Phase 1

- **Evidence:** `TEMP_MERGED_DIR` is loaded per fold (`src/train_evaluate_cv_v4.py:24`, `:150`, `:158`). `src/pretrain_cpt_bert.py:6-10` states the LLM script was never committed. `FINAL_REPORT.md:69` calls it inherited. The Gemma script loads Hugging Face weights directly (`src/train_evaluate_cv_gemma.py:143`, `:150`), while the paper's page 6 implies the same pipeline.
- **Reuse across folds is not itself the problem**, given an independent corpus. Dyspnea: no exact match and no shared sequence of 8 or more words with any gold note, and at most 2 shared 6-word sequences in 5 gold notes. DART: **not checked**, access denied without authentication.
- **Corpus differences:** BERT CPT used 6,000 of 16,029 DART rows; the LLM's use is unknown.
- **Correction:** Phase 6 decides between reconstructing, rerunning, or disclosing. See `audit/RECONSTRUCTION_PLAN.md`.

## E. Truncation credited as redaction

- **Evidence:** `audit/results/v4_truncation.json`. Token counts use public copies of the tokenizers.

| Model | Notes cut | Mentions credited unseen | Reported macro-F1 | Excluding them | Counting them as missed |
|---|---|---|---|---|---|
| Llama v4 | 29 of 80 | 13 of 225 | 0.649 | 0.596 | 0.575 |
| Llama v3 | 29 of 80 | 13 of 225 | 0.536 | 0.484 | 0.474 |
| Gemma 1B | 26 of 80 | 11 of 225 | 0.611 | 0.575 | 0.551 |
| BERT, 512 limit | 32 of 80 | 19 of 225 | 0.569 | 0.493 | 0.463 |

- **Mechanism:** `paper_metric` credits a mention when its string is absent from the output, so text the model never received counts as redacted.
- **Correction:** the longest gold note is 1,255 Llama tokens, which does not need chunking. Evaluation in Phase 10 must process whole notes and record truncation.

## F. No per-note predictions

- **Evidence:** `redacted` is scored and dropped (`src/train_evaluate_cv_v4.py:187-188`). `save_strategy="no"` (`:229`). Result files hold fold totals only.
- **Consequence:** span-level metrics cannot be computed from existing artifacts, so reruns are required.

## G. Style-anchoring control

- **Evidence:** C2 trained on unanchored synthetic notes plus CRF rows, with no gold (`FINAL_REPORT.md:164`, `:170`; paper section 4.2). 75 of its 80 test notes were training examples with an unchanged target, and those rows were the only real-register text it saw. Its reported behavior of "copying the input unchanged" (`FINAL_REPORT.md:172`) is what that predicts.
- **More confounds:** it differs from the main system in style anchoring and in gold inclusion, and is evaluated differently, despite "all other settings unchanged". Script and data are absent.
- **Correction:** remove 0.18 as evidence. The same generation plan drives both arms now, so style anchoring can be tested alone.

## H. Gold training targets are corrupted (new)

- **Evidence:** `audit/results/v5_targets_metric.json`. `redact_gold` uses `str.replace` (`src/train_evaluate_cv_v4.py:81`). 41 of 287 replacements fall inside longer tokens, 40 of them ages, in 11 notes. A short age string also changes digits inside other numbers.
- **Excess placeholders against annotations:** 55 ETÀ, 10 NOME, 1 LUOGO, in 20 notes.
- **Effect:** these targets are what the 5x oversampling repeats. They teach the model to break numbers. They are also why the targets score only 0.919 on the metric.
- **Correction:** word-boundary matching in `src/data/annotations.py`. Offset recovery for training targets belongs to Phase 7 and Phase 10.

## I. The metric rewards wrong output (new)

- **Evidence:** `metric_on_known_outputs` in `audit/results/v5_targets_metric.json`, running the real `paper_metric` with the paper's aggregation:

| Output | Macro-F1 |
|---|---|
| Empty string for every note | **1.000** |
| Every entity replaced with the **wrong** tag | **0.674** |
| Reported Llama v4 | 0.649 |
| Word-boundary oracle, first k occurrences | 0.907 |
| Word-boundary oracle, all occurrences | 0.934 |
| Input returned unchanged | 0.025 |

- **Meaning:** the headline score is below what a system with entirely wrong categories gets, and correct redaction cannot reach 1.0. A short age remains "present" whenever those digits occur anywhere in the output, which penalizes correct work.
- **The paper is also inaccurate here:** page 8 says a mention counts only "when it is fully matched and replaced with the corresponding placeholder". The code never looks at the replacement.
- **Correction:** Phase 10 keeps this metric as a labelled historical module and adds strict span evaluation. The reviewer's concern is confirmed.

## J. Metric differs from its citation (new)

- **Evidence:** the paper says it adopts Miranda et al.'s metric "without modification". Their worked example has two annotated mentions of one string, found once in the output, giving FN = 1 and TP = 1. `paper_metric` gives FN = 2 and TP = 0, because each listed mention tests string presence.
- **Confirmed in their code:** `eval.py` groups annotations by string and counts occurrences with `re.findall`. One age `8` with that digit inside three other numbers gives FN = 3 there and FN = 1 in `paper_metric`. Results are in `audit/results/v9_source_metric.json`. On correct redaction of the rebuilt notes, theirs gives 0.905 and `paper_metric` 0.941.
- **Correction:** remove "without modification". See also V for their different input text.

## K. Training truncation (new)

- **Evidence:** `audit/results/v4_truncation.json`. Examples over `max_length=1536` tokens: gold 29 of 80 (median 58.5% of target lost), S1 named 28.0%, S1 name-free 60.2%, S2 named 33.4%, S2 name-free 57.0%. Token sums are approximate.
- **Limit:** in trl 1.17.1, truncation defaults to `keep_start`, which cuts the end of the target and its end token. The version actually used is unknown, though `requirements.txt` asks for 1.4 or later.
- **Correction:** the new plan follows gold lengths, and Phase 7 sets a limit that fits the longest example.

## L. Synthetic distributions (new)

Per-note rates, counting placeholders:

| | Gold | S1 named | S2 named | S1 name-free | S2 name-free | New plan, fold 1 |
|---|---|---|---|---|---|---|
| NOME | 0.57 | 4.89 | 4.98 | 0 | 0 | 1.08 |
| ETÀ | 1.23 | 1.03 | 1.07 | 1.28 | 1.22 | 1.33 |
| DATA | 0.59 | 3.83 | 3.93 | 4.32 | 4.42 | 0.93 |
| LUOGO | 0.42 | 2.10 | 2.07 | 2.26 | 2.33 | 0.52 |
| Total | 2.81 | 11.85 | 12.06 | 7.86 | 7.98 | 3.86 |

- The paper's "fourfold" NOME figure uses unique entity-list items; counting mentions it is about 8.6 times. Only NOME was ever controlled, despite the claim of entity-distribution control.
- Length: 28 of 80 gold notes are under 1,000 characters, which the validator rejected. 44 of 80 gold notes fail the old validator overall.

## M. Synthetic target quality (new)

- **Evidence:** `audit/results/v6_synthetic.json`.
- 28 of 1,676 notes: `redacted_text` does not align with `text`.
- In the other 1,648: 414 of 17,832 replaced spans are absent from the entity list, 869 notes have a placeholder count differing from their entity list, and 46 spans exceed 60 characters.
- PII-like patterns left in `redacted_text`: 395 month-and-year, 109 numeric dates, 568 bare years, 217 age patterns. Some are not PII, such as durations, but this teaches the model to leave dates in place.
- **Correction:** the segment format makes text, target and offsets one source, and these patterns are rejected.

## N. BERT labels (new)

- **Evidence:** `entities_to_spans` (`src/train_evaluate_cv_bert.py:83-105`) has no word boundary: 43 of 287 gold spans fall inside longer tokens. It also uses the entity list alone, so the 414 unlisted spans above stay labelled as non-entities.
- **Correction:** use offsets from the new generator. Phase 8 must use recovered gold spans.

## O. "Chunking hurt BERT" (new, inferred)

- **Evidence:** the 0.569 baseline used a 512-token limit with the same credit artifact (19 mentions). Corrected, it is 0.463 to 0.493, below chunked BERT's 0.508. The −0.061 finding in `bert_cpt_and_chunking_findings/README.md` likely reflects removing the artifact, not fragmentation.
- **Related:** corrected Llama v4 (0.575 to 0.596) is about level with BERT + CPT (0.588). All of these are contaminated by Problem A, so neither claim stands until rerun.
- **Limit:** `cv_bert_results.json` cannot be reproduced from `HEAD`; it needs the code at `a2ba020`.

## P. Gold annotations (new)

- **Evidence:** `audit/results/v5_targets_metric.json`, `audit/gold_dataset_statistics.csv`.
- No offsets and no document IDs in the old file. **Now resolved:** the source supplies IDs for all 82 notes, and 225 of 228 mentions have verified offsets.
- 2 annotated strings were absent from their notes (`gold_010`, `gold_058`). The source has the same strings, so they are small source errors, one a typo and one a spacing difference. 1 occurs only inside a longer token.
- 22 strings occur more often than annotated, and 3 less often. Recovering spans for these is ambiguous.
- Patterns left in targets, on the rebuilt notes: 28 date-like strings in 10 notes and 3 initials in 3 notes, against 47 annotated dates. Annotations may be incomplete, or a convention may exclude these; a person must decide using `data_secure/gold/annotation_review.csv`. A correct model can be penalized if they are incomplete.
- **Resolved in `gold_v2`:** a row-by-row review added 27 mentions and removed 3, and kept ages as the number only. See `reports/GOLD_DATA_EVALUATION.md`. No independent second annotator was available.
- Patient-level grouping is unknown. Only one gold pair shares even one 8-word sequence, so no near-duplicates were found.

## Q. Generation parameters (new)

- **Evidence:** batch-index modulo (`generate_synthetic_parallel.py:38-41`) gives 30 of 180 combinations. Gender is locked to specialty, so gynecology is always requested with male patients. Age is locked to density. Paper page 6 says "systematically varied".
- 22% of numeric gold ages are under 18 (20 of 91), and no prompt asks for a child.
- **Correction:** `src/generation/generation_plan.py`.

## R. Other paper errors

| Claim | Evidence |
|---|---|
| Anchors "rotated across batches" | False; see B |
| Same pipeline applied to Gemma | False; see D |
| 52 million LoRA parameters, 1.7% | 24.3M, 0.75%, recalculated for Llama-3.2-3B dimensions; already in `report_phases_errata.tex` |
| Adding 475 notes raised CV NOME F1 from 0.36 to 0.71 | 0.36 is from a single split, and v4 also added 577 named notes. CV was 0.593 to 0.714 |
| DART is "clinical text across specialties" | Drug package inserts |
| Control keeps "all other settings unchanged" | False; see G |
| Validator rule 2 ensures "complete redaction" | Only listed strings are checked; see M |
| Standard deviations | Population (ddof=0). With five folds the sample values are larger: 0.082 and 0.122 |

`report_phases_errata.tex:22` says no experiments need rerunning. That is now wrong. `thesis/thesis.tex:424` and `:438` call the CRF set PII-free.

## S. Secrets and note text in public (new)

- **Secrets:** API keys of the earlier code were exposed in an earlier public repository. Their location is removed from this public copy. The new key is git-ignored and was never committed.
- **Note text:** `data/gold_standard_80.json` is tracked and public. `src/generalization_test.py` contains an edited copy of `gold_000` (14.1% of its 8-word sequences). The submitted paper's Table 1 prints an extract of `gold_042` (7.8%). `audit/results/v7_text_exposure.json` has the scan.
- **Sources may be public:** Miranda et al. describe the source as the openly available E3C corpus of clinical cases, which may make this acceptable. Licence and permitted redistribution are unverified, and guide rule 13 applies regardless.
- **API exposure:** three anchors were sent to Gemini in full in every historical call. Authorization is undocumented.
- **This session:** before the guide's rule 14 arrived, earlier analysis printed short excerpts and entity values from gold notes into this coding-assistant session, and read `generalization_test.py`. Later work prints IDs and counts only.
- **Correction needed from the owner:** revoke the three keys, and decide on repository visibility and history.

## T. Generator model retirement (new)

Google's deprecations page lists `gemini-2.5-flash` for shutdown on 2026-12-01, routing to another model from that date. The project owner chose the newest available model, so the new data uses `gemini-3.8-flash`. S1 and S2 therefore cannot be exactly regenerated, and the paper must name the model actually used.

## U. Training hardware (new)

The owner will train on a Windows RTX 3070 (typically 8 GB), not the original 12 GB RTX 4070 Ti. Original settings may not fit, and Gemma 4B already failed on 12 GB. See `audit/RECONSTRUCTION_PLAN.md`.

## V. Source file differences (new)

- **Evidence:** `audit/results/gold_build_summary.json`, `audit/results/v9_source_metric.json`. Source: `Annotated clinical notes - samples.csv`, commit `e8609c92`.
- **Notes:** the file has 82 notes, the paper and Miranda et al. say 80, and the old file dropped the last two (`IT102056`, `IT102057`), which hold 3 age mentions. That explains 98 vs 101 ETÀ.
- **Old text was correct:** all 80 old narratives match the source once its ID prefix and CLinkaRT relation lines are removed. All 649 relation lines verify against the narrative.
- **Baseline may not be like-for-like (inferred):** their paper says only the textual component was used. But `eval.py` compares each model input with the whole `input` cell, including ID and relation lines (16.8% of its characters), and skips notes that differ. Their model input file, `samples.csv`, is not published. The project evaluated on narrative only.
- **Licence:** the repository has no licence file.
- **Unused column:** `annotations` has 516 items under more than 25 labels and is not the gold standard; `eval.py` uses `annotations_redacted`.
- **Their table counts all 82 notes:** Table 1 reports 47, 101, 34 and 46 mentions, the totals over all 82 rows, although the text says 80 notes.
- **Annotation method:** the paper says the notes were manually annotated into predefined categories. It gives no guidelines, number of annotators, or agreement figures.
- **Correction:** `src/data/build_gold.py` builds `gold_v1` from the pinned source. Any comparison with their table must disclose both the input and metric differences.

## Historical experiments

None of these numbers can be used as results.

| Experiment | Reported | Script in repo | Result file | A | B/C | E | H |
|---|---|---|---|---|---|---|---|
| Extraction v0 | 0.13 | No | No | Unknown | No | Unknown | No |
| Redaction v1 | 0.60, on training data | No | No | Yes | No | Unknown | Yes |
| C2 control | 0.18 | No | No | Yes | No | Unknown | No |
| v2, 16-note split | 0.635 | No | No | Yes | Yes | Yes | Yes |
| v3, 16-note split | 0.692 | Yes | No | Yes | Yes | Yes | Yes |
| v3 CV | 0.536 ± 0.144 | Yes | Yes | Yes | Yes | Yes | Yes |
| v4 CV | 0.649 ± 0.073 | Yes | Yes | Yes | Yes | Yes | Yes |
| Gemma 1B CV | 0.611 ± 0.109 | Yes | Yes | Yes | Yes | Yes | Yes |
| BERT, truncated | 0.569 ± 0.143 | At `a2ba020` | Yes | Yes | Yes | Yes | Labels: N |
| BERT, chunked | 0.508 ± 0.125 | Yes | Yes | Yes | Yes | No | Labels: N |
| BERT, CPT + chunked | 0.588 ± 0.146 | Yes | Yes | Yes | Yes | No | Labels: N |
| Out-of-distribution test | Not saved | Yes | No | Yes | Yes | No | Yes |
