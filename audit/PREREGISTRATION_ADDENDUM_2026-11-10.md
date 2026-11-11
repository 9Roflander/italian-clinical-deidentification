# Addendum to the analysis plan of 2026-11-10: extra seeds for S2 E2

Written 2026-11-10, after all runs of `audit/PREREGISTRATION_2026-11-10.md` were evaluated and before the runs below start.

## Why

S3 (BERT) showed a seed standard deviation of up to 0.09 micro-F1 (arm E2: 0.750, 0.605, 0.775 for seeds 42, 43, 44).
S1 and S2 have one seed (deviation 3 of the plan), so a single-seed difference between them, or against S3, may reflect
training randomness rather than the system. The owner decided to add seeds 43 and 44 for S2 on arm E2, the arm with the
best S2 and S3 results.

## Runs

| System | Arm | Seeds | Configuration |
|---|---|---|---|
| S2 | E2 | 43, 44 | `configs/training_llama_extraction.yaml`, unchanged (config id `ccc3fa5896f3`), code commit acb65c8 |

The seed sets LoRA initialization and the order of micro-batches; the fold data are the same as for seed 42.

## How they are used

- The preregistered conclusions for H1–H4 stay those of seed 42. They are not replaced or re-decided.
- The extra seeds are reported as a stability check: mean and standard deviation of S2 E2 over seeds 42–44, and whether
  the sign of S2−S1 (E2) and S3−S2 (E2) is the same for each seed pair with equal seeds (S2 vs S3 at seeds 43 and 44).
- These runs were chosen after the seed-42 results were known, and are labelled as such wherever they are reported.
- S1 E2 has no extra seeds, so S2−S1 at seeds 43 and 44 compares against S1 seed 42 on point estimates only.
