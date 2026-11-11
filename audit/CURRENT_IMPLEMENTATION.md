# Current Implementation (historical pipeline)

What the previous project actually did, reconstructed from its code, data, logs and result files. All paths are relative to `original_work/`.

Status labels used below:

- **Confirmed**: checked directly in code, data or result files.
- **Inferred**: the best explanation of the evidence, but not proven.
- **Reported**: stated in project documents, with no artifact to check it against.
- **Unverifiable**: the artifact needed to check it does not exist in this repository.

Reproducible checks are in `audit/scripts/`, with results in `audit/results/`. They output IDs and counts only.

## 1. Repository map

| Path | Contents | In git? |
|---|---|---|
| `src/` | 14 Python scripts, listed in section 2 | Yes |
| `data/` | Gold notes, four synthetic files, one single-split index | Yes, on a public remote |
| `cv_v3_results.json`, `cv_v4_results.json`, `cv_gemma1b_results.json`, `cv_bert_results.json` | Fold-level totals | Yes |
| `bert_cpt_and_chunking_findings/` | Two more BERT result files and a README | Yes |
| `cv_bert_outputs/`, `cv_bert_cpt_outputs/` | BERT logs and empty fold folders | No, git-ignored |
| `FINAL_REPORT.md`, `README.md`, `report*.tex`, `presentation*`, `thesis/` | Documents | Mostly yes; see `git status` for uncommitted edits |
| `REVIEW FORM-NJ148 (1).pdf`, `splnproc1703_...copy.pdf` | Review and submitted paper | No, untracked |

There are no notebooks, YAML configs, shell scripts, model weights, adapters, or `trainer_state.json` files anywhere in the project. All file contents were preserved byte for byte in the move to `original_work/`; hashes are in `audit/results/original_work_sha256.txt`.

## 2. Executable scripts

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `src/generate_synthetic_v2.py` | Single-threaded generator; defines prompts, schema, validator, anchor selection | Gold | `--output` JSON |
| `src/generate_synthetic_parallel.py` | Three-thread generator reusing the above; S1 named | Gold | `data/synthetic_v2_1000.json` by default |
| `src/generate_noname.py` | Name-free generator; S1 name-free | Gold | `data/synthetic_v2_noname_400.json` by default |
| `src/train_redact_v3.py` | Single-split Phase 2 training for Llama | `temp_merged_phase1`, gold, S1, CRF | `llama-3.2-3b-deid-redact-v3` adapter |
| `src/eval_redact_v3.py` | Evaluates that adapter on the 16-note split | Adapter, gold | stdout |
| `src/eval_paper_metric.py` | Evaluates the older JSON-extraction adapter on all 80 notes | `llama-3.2-3b-deid-sft-v2`, gold | stdout |
| `src/train_evaluate_cv_v3.py` | 5-fold CV for Llama, 878 synthetic notes | `temp_merged_phase1`, gold, S1, CRF | `cv_v3_results.json` |
| `src/train_evaluate_cv_v4.py` | 5-fold CV for Llama, 1,676 synthetic notes | `temp_merged_phase1`, gold, S1, S2, CRF | `cv_v4_results.json` |
| `src/train_evaluate_cv_gemma.py` | 5-fold CV for Gemma 1B or 4B | Hugging Face base model, gold, S1, S2, CRF | `cv_gemma1b_results.json` |
| `src/pretrain_cpt_bert.py` | MLM domain adaptation for BERT | Dyspnea, 6,000 of 16,029 DART rows | `bert_it_clinical_cpt/` |
| `src/train_evaluate_cv_bert.py` | BERT token-classification CV | BERT, gold, S1, S2, CRF | Results JSON |
| `src/train_evaluate_cv_bert_chunked_test.py` | BERT CV with chunking | As above | `cv_bert_chunked_results.json` |
| `src/train_evaluate_cv_bert_cpt.py` | BERT CV from the CPT checkpoint | CPT checkpoint, as above | `cv_bert_cpt_results.json` |
| `src/generalization_test.py` | Out-of-distribution check on one edited note | Adapter | stdout only, not saved |
| `thesis/make_figures.py` | Thesis figures; loads results but also has hard-coded values | Result files | `thesis/figures/` |

## 3. Execution order

Git cannot establish the order: all LLM work arrived in one commit on 2026-05-18 (`9b13423`). This order comes from `FINAL_REPORT.md` sections 3 and 8, and is **Inferred** unless marked.

1. Phase 1 adapter `llama-3.2-3b-dart-sft`, merged into `temp_merged_phase1`. **Unverifiable**; the report calls it "inherited from earlier work".
2. JSON-extraction format; macro-F1 about 0.13. **Reported**; script absent.
3. Redaction v1 on `synthetic_clinical_1000.json`, gold and CRF; 0.60 on its own training data. **Reported**; script and data absent.
4. Control C2: same synthetic data plus CRF, no gold, tested on all 80 notes; 0.18. **Reported**; script and data absent.
5. S1 named generation, then redaction v2; 0.635 on the 16-note split. **Reported**; v2 script absent.
6. S1 name-free generation, then `train_redact_v3.py`; 0.692 on the same split. **Reported**.
7. `train_evaluate_cv_v3.py`; 0.5355 mean macro-F1. **Confirmed** from results.
8. S2 generated after seeing v3 CV variance. **Confirmed** by the v4 docstring (`src/train_evaluate_cv_v4.py:12`). Command not recorded.
9. `train_evaluate_cv_v4.py`; 0.6486. **Confirmed** from results.
10. Gemma 1B CV; 0.6107. Gemma 4B ran out of memory. **Confirmed** from results and commit message.
11. BERT baseline at commit `a2ba020`, then chunking and CPT variants at `10efd31`. **Confirmed** from git and results.

## 4. Generated files

| File | Records | Status |
|---|---|---|
| `data/synthetic_v2_1000.json` | 624 | **Confirmed** count |
| `data/synthetic_v2_noname_400.json` | 254 | **Confirmed** count |
| `data/synthetic_v4_named_more.json` | 577 | **Confirmed** count; command not recorded |
| `data/synthetic_v4_noname_more.json` | 221 | **Confirmed** count; command not recorded |
| Six result JSON files | Fold-level TP, FP, FN only | **Confirmed**; no per-note output |
| `*_rejected.json`, generation logs, LLM training logs | None | Absent; git-ignored or never kept |

## 5. Data sources

| Source | Use | Status |
|---|---|---|
| `data/gold_standard_80.json` | Training and evaluation | 80 records, keys `text` and `entities` only. No offsets and no document IDs. **Confirmed**. It equals the first 80 rows of Miranda et al.'s CSV, without ID prefixes and relation lines. Their CSV has 82 notes |
| `NLP-FBK/synthetic-crf-train` (it) | "Negative" examples with output = input | 80 rows, data last uploaded 2025-12-02, so runs used this content. **Confirmed**; see `audit/LEAKAGE_REPORT.md` |
| `NLP-FBK/dyspnea-clinical-notes` (it) | Phase 1 and BERT CPT | 2,667 rows. **Confirmed** count |
| `praiselab-picuslab/DART` | Phase 1 and BERT CPT | Gated. BERT log shows 6,000 rows used; LLM use **Unverifiable** |
| `synthetic_clinical_1000.json` | v1 and C2 | Absent. **Unverifiable** |

## 6. Fold creation

`KFold(n_splits=5, shuffle=True, random_state=42)` over note positions in the gold file, at `src/train_evaluate_cv_v4.py:206` and the equivalent lines in the other CV scripts. **Confirmed**: the reconstructed folds match TP + FN in all six result files, for every fold and category. Frozen copies are in `splits/` and `data_manifests/fold_assignments.csv`.

## 7. Training example assembly

`get_train_dataset` in `src/train_evaluate_cv_v4.py:127-146` concatenates, per fold:

- All 1,676 synthetic records, reloaded unchanged. Target is `redacted_text`.
- 64 gold notes, each repeated five times (320). Target comes from `redact_gold`.
- 80 CRF rows. Target is `output = input`.

Total: 2,076 examples, confirmed by `total train samples: 2076` in the BERT log. Oversampling happens after the split, so that rule was followed. The CRF rows break the separation in a different way; see Problem A.

Training settings at `src/train_evaluate_cv_v4.py:154-165` and `:220-234`: 4-bit NF4 with double quantization, LoRA r=16, alpha=32, dropout 0.05 on seven projection layers, 2 epochs, batch 1 with 16 accumulation steps, learning rate 5e-5, cosine schedule, 50 warmup steps, `max_length=1536`, gradient clipping 1.0, `assistant_only_loss=True`, `save_strategy="no"`. These match the paper. **Confirmed**. Package versions used are **Unverifiable**: `requirements.txt` uses `>=`, and this Mac's `.venv` has no `trl` or `bitsandbytes`.

## 8. Synthetic generation

- Model string `gemini-2.5-flash`, temperature 0.7, max output 12,288 tokens, three cases per request, schema-constrained JSON (`src/generate_synthetic_v2.py:28-29`, `:192-197`). No seed, no `top_p`, no logged model version.
- `load_style_anchors` (`:76-84`) sorts all 80 gold notes by length and takes positions 20, 40 and 60, which are `gold_034`, `gold_013`, `gold_072`. It is called once per run. Every batch gets the same three notes in full, plus their entity lists.
- Rotation (`generate_synthetic_parallel.py:37-43`) covers only specialty, gender, age and density, each by batch index modulo list length.
- The validator (`validate_case`, `:149-186`) checks: length 1,000 to 4,000; every entity string in text; no entity string in `redacted_text`; a tag for each present type; at least two entities. It does not check that `redacted_text` equals `text` apart from tags, although the prompt demands it.
- Results come from `as_completed`, so record order depends on network timing.
- S2's exact command is **Unverifiable**. **Inferred** to use the same generator: a hard-coded prompt string appears in about 2% of both named files and neither name-free file, and S1 and S2 have the same overlap and entity profiles.

## 9. Inference

`evaluate_fold` (`src/train_evaluate_cv_v4.py:168-198`):

- Notes over 700 tokens are cut on the **input** side: `tokenizer.decode(ids[:700])` at line 176.
- `max_new_tokens=1024` is the separate output limit. It is not reached in normal use once input is capped at 700.
- Greedy decoding, with both Llama end tokens.
- The decoded output is scored and then discarded. Stop reason, truncation state and output length are never recorded.

## 10. Metric

`paper_metric` (`src/train_evaluate_cv_v4.py:85-99`), an identical copy in the v3, Gemma and BERT scripts:

- `FN` = gold mentions whose string appears anywhere in the output (substring test).
- `TP` = gold mentions minus `FN`.
- `FP` = placeholders of that type beyond the gold count, floored at zero.

Fold totals become per-category F1, macro-F1 is the unweighted mean, and the reported figures are the mean and population standard deviation (`np.std`, ddof=0) across folds. The test does not look at which tag replaced an entity, or whether output exists at all. See Problems E and I.

## 11. What cannot be verified

- Phase 1: script, settings, corpus versions, whether all of DART was used, logs, checkpoint. `temp_merged_phase1` was not found on this Mac; other machines have not been checked.
- The objective of Phase 1. Documents describe next-token pretraining, but the adapter name ends in `-sft`, which elsewhere in the project means supervised fine-tuning.
- v0, v1, v2 and C2: scripts, data and scores.
- Generation pass rates of 77% and 71%. With the documented arguments the most that could be requested was 1,032 and 492 cases, giving 60.5% and 51.6% kept, but API failures are unknown.
- Generation costs: both "$10 for about 880" and "$15 for about 1,700" appear, and no invoice is present.
- Which chat template trained the LLMs. The custom template applies only `if not tokenizer.chat_template`, and the checkpoint's tokenizer is unavailable.
- Package versions and training logs for all LLM runs. Timing of about 50 minutes per run is **Reported**.

## 12. Checkpoints

None exist in this repository. Spotlight on this Mac finds no `temp_merged_phase1`, `llama-3.2-3b-dart-sft`, `llama-3.2-3b-deid-redact-v3`, `llama-3.2-3b-deid-sft-v2`, or `cv_*_outputs` with contents. `save_strategy="no"` means fold adapters were never saved. The `cv_bert_outputs/fold_*` folders are empty.
