# Analysis plan for the system comparison, fixed 2026-11-10

Owner decision 2026-11-10: compare two Llama 3.2 3B variants with an Italian BERT baseline. Gemma 3 1B runs only if time
allows and is then exploratory. This file is committed before any run below starts; its commit identifies the plan.
Results already seen when it was written: Llama E1–E3 in the segments format, and BERT config v1 (see Deviations).

## Systems

| ID | System | Configuration | Output |
|---|---|---|---|
| S1 | Llama 3.2 3B Instruct, QLoRA, segments format | `configs/training.yaml` | the whole note as labelled segments |
| S2 | Llama 3.2 3B Instruct, QLoRA, extraction format ("E3-X") | `configs/training_llama_extraction.yaml` | only the sensitive strings, placed by the fixed locator in `src/training/extraction.py` |
| S3 | Italian BERT (`dbmdz/bert-base-italian-xxl-cased`), BIO token classification | `configs/encoder_bert_it_v2.yaml` | one label per token |

S1 and S2 differ only in prompt, output format and checkpoint folder. All three systems use the same folds, the same
training example lists (built by `src/training/dataset.py`, gold oversampled 5 times) and the same evaluator (v1.0.1).
No system is tuned: S1 and S2 use the settings fixed on 2026-11-09 (decisions A–D), S3 uses standard BERT settings.

## Runs

| System | Arms | Seeds |
|---|---|---|
| S1 | E1, E2, E3 | 42 (finished, commit 5f69f2a) |
| S2 | E1, E2, E3 | 42 |
| S3 | E1, E2, E3, E8, E9, E10 | 42, 43, 44 |

Arms are defined in `configs/experiments.yaml`. E8–E10 are run for S3 only.

## Metrics

- Primary: strict exact-span micro-F1, pooled over the 82 out-of-fold notes.
- Secondary: macro-F1, per-category precision, recall and F1, invalid-output rate, over-redaction rate.
- Uncertainty: paired document-level bootstrap, 10,000 replicates, stratified by fold, percentile 95% intervals
  (`configs/evaluation.yaml`). For S3, mean and standard deviation over the three seeds are also reported.

## Hypotheses and primary comparisons

Each is a paired difference in micro-F1 between seed-42 runs on the same notes.

| ID | Question | Comparisons |
|---|---|---|
| H1 | Does synthetic data help? | E2 − E1, within S1, S2 and S3 |
| H2 | Does style anchoring help? | E3 − E2, within S1, S2 and S3 |
| H3 | Does the extraction format help the LLM? | S2 − S1, on E1, E2 and E3 |
| H4 | Generative model or encoder? | S3 − S1 and S3 − S2, on E1, E2 and E3 |

Decision rule: a difference is called an effect only when its 95% interval excludes zero. Otherwise it is reported as
"no detectable difference", with the interval. For S3, a conclusion is also checked for the same sign on seeds 43 and 44,
and any disagreement is reported. Intervals are not adjusted for multiplicity; the number of comparisons is stated.

Exploratory, and labelled so: E8–E10 (negatives, named-only, 72/28 balance), per-category differences, and Gemma.

## Fixed rules

- No setting of S1, S2 or S3 changes after these runs start. A later change makes a new configuration version, and its
  results are exploratory.
- S2 locator: plain left-to-right search; an "O" item skips an unannotated earlier copy; an unlocated item is dropped and
  counted; output that is not a JSON array of {"text", "label"} objects is an invalid output.
- S3 budget: 175 updates of 16 windows for every arm, as S1 and S2 share 79 optimizer steps for every arm.
- Runs are immutable. A failed or interrupted fold is moved aside and run again from the start, never resumed.

## Deviations known at the time of writing

1. S2 was designed after the S1 E1–E3 results were seen. Its motivation: 15–20% of S1 outputs could not be scored,
   because the model changed the note while copying it.
2. S3 config v1 (`configs/encoder_bert_it.yaml`: 3 epochs, arms E1 and E3, seed 42) ran first. With 3 epochs, E1 got 90
   updates against 174 for E3, and the E1 training loss stayed on the all-"O" plateau. v2 sets one update budget for
   all arms, the rule already used for S1. The v1 runs stay unchanged and are reported in an appendix.
3. S1 and S2 have one seed. The bootstrap does not capture training randomness for them.
4. Gemma 3 1B is deferred.

## Limitations stated in advance

All 82 gold notes are development data; these are cross-validation estimates, and claims of generalization need an
independent test set. There is a single annotator. Domain adaptation (E4, E5, E7) is not run.
