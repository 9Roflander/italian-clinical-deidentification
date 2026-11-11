# Data Provenance

Machine-readable versions: `data_manifests/dataset_manifest.csv` and `data_manifests/record_manifest.csv`. Statuses: Confirmed, Reported (a document says so, with no artifact to check), Not checked.

## Gold standard, 82 notes (`gold_v1`)

| Field | Value | Status |
|---|---|---|
| Source | `Annotated clinical notes - samples.csv`, `github.com/michele17284/Italian-Clinical-Note-Deidentification`, commit `e8609c92` | Confirmed |
| Source SHA-256 | `6be1cf9d0d2f9f199ba7c3e0a95636081c99023ae54fea878aa6202a36fea9f7` | Confirmed and pinned |
| Built file | `data_secure/gold/gold_v1.jsonl`, git-ignored, by `src/data/build_gold.py` | Confirmed |
| Origin of the notes | Clinical cases from CLinkaRT, derived from E3C | Reported by Miranda et al.; the source format matches CLinkaRT relation lines |
| Gold annotations | `annotations_redacted`, as used by their `eval.py` | Confirmed |
| Old file | `original_work/data/gold_standard_80.json` equals the first 80 notes' narrative and annotations | Confirmed |
| Licence and redistribution | No licence file in the repository | Owner to confirm |
| Authorization for external APIs | Not documented | Owner to confirm |

Internal IDs `gold_000` to `gold_081` follow source order, which is ascending document ID, so they match the historical `#n`. Document IDs `IT100002` to `IT102057` are stored per note.

## Reviewed gold standard (`gold_v2`)

| Field | Value | Status |
|---|---|---|
| File | `data_secure/gold/gold_v2.jsonl`, git-ignored, by `src/data/build_gold_v2.py` | Confirmed |
| Input | `gold_v1` and the decision files `data_manifests/gold_annotation_proposal*.csv` | Confirmed |
| Note text | Identical to `gold_v1` | Confirmed by test |
| Changes | 27 mentions added, 3 removed, 1 located, 1 corrected, per `data_manifests/gold_v2_change_log.csv` | Confirmed by test |
| Mentions | 252, each marked as source annotation (225) or added in review (27) | Confirmed |
| Reviewers | Project owner, an external AI assistant, Claude Code. No independent second annotator | Reported |
| Folds | Same membership as `gold_v1`. `gold_v1` fold files are in `splits/gold_v1/` | Confirmed by test |

## NLP-FBK/synthetic-crf-train, Italian split

| Field | Value | Status |
|---|---|---|
| Revision | `5c6a0098` | Confirmed |
| Data uploaded | 2025-12-02; later commits are README updates | Confirmed from commit titles |
| Rows | 80 | Confirmed |
| Relation to gold | 75 exact twins and 1 near-copy | Confirmed |
| Decision | Not permitted for training | See `data_manifests/negative_audit.csv` |

"Synthetic" appears to describe the generated case report forms, not the clinical notes. That reading comes from the cited paper's title and from these overlaps; it is an inference.

The dataset's content predates the May and August 2026 runs, so those runs used this data. Their logs record 80 rows.

## Historical synthetic data

| Set | File | Records | Status |
|---|---|---|---|
| S1 named | `synthetic_v2_1000.json` | 624 | Confirmed |
| S1 name-free | `synthetic_v2_noname_400.json` | 254 | Confirmed |
| S2 named | `synthetic_v4_named_more.json` | 577 | Confirmed |
| S2 name-free | `synthetic_v4_noname_more.json` | 221 | Confirmed |

What can be reconstructed:

- **Generator for S1:** `generate_synthetic_parallel.py --n 1000` and `generate_noname.py --n 400`, as documented in `FINAL_REPORT.md`. Reported, and consistent with default output paths.
- **Generator for S2:** the command was not recorded. Same prompts are **inferred**: one hard-coded string occurs in 1.9% and 2.3% of the two named files and 0% of both name-free files. All 1,676 notes pass the historical validator.
- **Anchors for S2:** inferred from the generator above, not proven. Verbatim overlap is too low to identify the anchors from the data.
- **Model:** `gemini-2.5-flash`, an alias. The exact version was not logged.
- **Not recoverable:** prompts as sent, seeds, rejection logs, timestamps, costs.

Decision: not reused as clean training data, because their prompts exposed test notes and S2 followed CV inspection.

## Control data

`synthetic_clinical_1000.json` is git-ignored and absent. Its script is absent too. It cannot be audited.

## Phase 1 corpora and checkpoint

| Item | Value | Status |
|---|---|---|
| Dyspnea notes | Revision `b58288bb`, 2,667 rows | Confirmed |
| Dyspnea vs gold | No exact match; no shared sequence of 8 or more words; 5 gold notes share a 6-word sequence (at most 2), 36 share a 5-word sequence (at most 5) | Confirmed. Short overlaps look like common wording, but no person has read them |
| DART | Revision `14ce22de`, gated with manual approval | Metadata confirmed |
| DART vs gold | **Not checked.** Anonymous access was denied | The BERT log shows the owner had access in August |
| DART content | Drug package inserts, per the dataset paper's title and `pretrain_cpt_bert.py` | Reported |
| LLM Phase 1 script, settings, logs | Absent | Cannot be verified |
| LLM adapter name | `llama-3.2-3b-dart-sft` | Reported only |
| Merged checkpoint | `temp_merged_phase1`, not found on this Mac | Other machines unknown |
| Base model revision | Unknown | Cannot be verified |

This project does not claim DART is free of overlap. Overlap is unlikely given its content, but likelihood is not a check.

## BERT CPT

`pretrain_cpt_bert.py` used 2,667 dyspnea notes and a seeded sample of 6,000 DART rows, confirmed by `original_work/cv_bert_cpt_outputs/cpt_run.log`. The checkpoint folder `bert_it_clinical_cpt/` is not present here.

## New generated data

`synthetic_data/fold_*/<arm>/` records, for every run:

- Exact model ID and returned model version, SDK version, all sampling parameters, and per-request seed.
- Prompt template hash, rendered prompt hash, and generation plan hash.
- Anchor IDs per request, and fold manifest hash.
- Tokens, finish reason, timestamp, and git commit.
- Every rejection and its reason.

`examples.jsonl`, `raw_responses.jsonl` and `rejected.jsonl` stay out of git.
