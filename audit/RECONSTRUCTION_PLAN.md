# Reconstruction Plan

Covers the code to replace, build order, resource estimates, and risks the repository cannot resolve. Estimates are labelled; do not read them as measurements.

## Code to replace

| Historical code | Problem | Replacement | Status |
|---|---|---|---|
| `load_style_anchors`, `generate_synthetic_v2.py:76-84` | Global anchors | `AnchorSelector`, `src/generation/anchor_selection.py` | Done |
| `build_prompt`, `:87-137`; `build_noname_prompt`, `generate_noname.py:20-71` | Hard-coded strings; two templates | `prompts/generation_v1.txt`, `src/generation/prompt_templates.py` | Done |
| `validate_case`, `:149-186` | Substring checks; no target consistency check | `src/generation/synthetic_validation.py` | Done |
| `run_one_batch`, `generate_synthetic_parallel.py:37-65` | Locked parameters; no seed or logs | `generation_plan.py`, `synthetic_generator.py` | Done |
| Hard-coded API keys (location removed from the public copy) | Exposed secrets | `load_api_key`, `src/generation/gemini_client.py` | Done; owner must revoke old keys |
| `KFold` inside each CV script | Implicit split | `src/data/splits.py`, frozen files | Done |
| `get_train_dataset`, CRF rows | Test notes as negatives | Fold manifests; validated negatives | Manifests done; training pending |
| `redact_gold` | Substring replacement | Span-based targets, now in `gold_v2` | Targets done; training use pending, Phase 7 |
| `evaluate_fold` | Input cut; output discarded | Full-note inference; saved outputs | **Pending**, Phase 10 |
| `paper_metric` | Weak metric | Preserved as historical module, plus strict span metric | **Pending**, Phase 10 |
| `entities_to_spans`, BERT | Substring labels | Offsets from recovered spans | **Pending**, Phase 8 |
| `get_model_and_tokenizer` | Unpinned model and unverified template | Pinned revision and explicit template | **Pending**, Phase 7 |
| Phase 1 script | Absent | See below | **Pending**, Phase 6 |

## Build order

1. Phases 0 to 5, audit through generation scripts: **done, awaiting review**.
2. Owner decisions in `reports/GENERATION_REVIEW.md`, then the first anchored test.
3. Full generation and post-audit for all folds and arms.
4. Phases 6 and 7: Phase 1 decision, then a training pipeline for Windows with span-based targets.
5. Phase 10 before any full training: inference, full-note handling, span alignment, and both metrics, tested on constructed examples. Evaluation must exist before results do.
6. Phase 8: BERT baseline.
7. Phase 9: ablation matrix.
8. Phase 11 and 13: statistics and privacy assessment.
9. Phase 12: independent test, if one can be obtained.
10. Phase 14: tables and paper revision, last.

## Resource estimates

### Gemini

**Measured** from the smoke tests in `reports/GENERATION_REVIEW.md`. Full estimates are given there, since they depend on measured acceptance.

**Rate limits:** per project, and visible at `aistudio.google.com/rate-limit`. They were not measured. The config assumes 10 requests per minute until the owner sets the real value.

**Price change:** the pricing page says `gemini-3.8-flash` doubles on 2027-01-01.

### GPU, on one RTX 3070

Nothing here has been measured on that card. The anchor is a **reported** figure of about 50 minutes for 1,278 examples on an RTX 4070 Ti. The estimate assumes the 3070 plus any memory-saving settings run about 1.5 times slower.

| Run | Examples per fold | Estimated train time per fold |
|---|---:|---:|
| A, B: gold only | 320 | 20 minutes |
| C, E: gold and synthetic | 1,996 | 2 hours |
| D, F: complete | 2,076 | 2 hours |

Adding about 10 minutes of inference per fold:

| Scope | Estimate |
|---|---|
| Matrix A to F, 5 folds, Llama | About 50 hours |
| Same for Gemma 1B, if about 0.6 times Llama | About 30 hours |
| Style, entity and volume ablations, both models | About 45 hours |
| BERT | Under 10 hours; measured 39 minutes per fold on an M4 Mac |
| **Total** | **Roughly 130 GPU hours, without Phase 1** |

A first useful stage is D and A for Llama only: about 12 hours.

### Phase 1

Not estimable, because the script and settings are unknown. It covers 18,696 documents at the reported corpus size, which is larger than any Phase 2 run.

## RTX 3070 considerations

- Memory: the RTX 3070 typically has 8 GB. The original work used 12 GB, and Gemma 4B already failed there.
- Historical `max_length=1536` truncated targets. Fully avoiding that for the longest gold note needs roughly 2,600 tokens at batch size 1, which may not fit on 8 GB without gradient checkpointing. This is an estimate, to be tested first.
- Any memory-saving change must be fixed before outer scores are seen, applied to all folds, and documented.
- Bitsandbytes on Windows and the `trl` version need checking before committing to the original stack.

The first training step should be a one-fold, gold-only run to measure peak memory and speed, including the longest note.

## Phase 1 options

| Option | Effect | When to choose |
|---|---|---|
| A. Find the checkpoint and its provenance | Keeps historical adaptation | If the old training machine still has it |
| B. Rerun with a documented script for Llama and Gemma | Cleanest, and lets both models be compared fairly | **Recommended** if the DART check can run and 8 GB is sufficient |
| C. Report as inherited, with narrower claims | No compute | Fallback |

The ablation's domain-adaptation comparison needs Option A or B for a meaningful result.

## Risks the repository cannot resolve

| Risk | Needed from |
|---|---|
| Authorization and licence for gold notes, including sending any derived text to Gemini | Owner, institution |
| Free or paid API tier: free-tier content may be read by reviewers and used for product development | Owner |
| Whether `temp_merged_phase1` survives anywhere | Owner |
| Original Phase 1 settings | Owner, or whoever ran it |
| Remaining annotation misses cannot be ruled out, and there is no second annotator or published guideline | Disclose; a second annotator on a sample would measure agreement |
| Independent test set | Owner, co-authors |
| Patient-level linkage | Corpus authors |
| DART overlap | Owner's authenticated run |
| Publicly exposed keys and note text | Owner |
| Model stability: `gemini-3.8-flash` may change or retire before a rerun is needed | Keep all generated data and logs |
