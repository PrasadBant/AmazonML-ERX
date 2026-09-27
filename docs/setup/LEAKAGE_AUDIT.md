# Leakage Audit — Amazon ML Challenge 2026

**Status: CONFIRMED** unless marked otherwise. Source: `workspace/src/data/leakage_and_quality.py`
(raw output in `workspace/src/data/leakage_output.txt`). No original files were modified.

## Checks performed

### 1. entity_id overlap between train and test

CONFIRMED **zero** overlap for all three sources:

| Check | Overlap |
| --- | ---: |
| S1 train ∩ S1 test | 0 |
| S2 train ∩ S2 test | 0 |
| S3 train ∩ S3 test | 0 |

IDs are independently assigned per file (numeric suffixes, no shared range). No direct
ID-based leakage path from train into test.

### 2. Duplicate / near-duplicate records within a source

- Source 1 (both splits): 0% exact name+address duplicates (deduplicated by
  construction, matches README claim).
- Source 2: 0.73% (train) / 0.66% (test) exact duplicate rows.
- Source 3: 0.45% (train) / 0.43% (test) exact duplicate rows.

These are **within-split** duplicates (e.g. two S2 rows with identical name+address),
not train/test duplicates. They are a data-quality note (see `DATA_PROFILE.md`), not a
leakage path, since the check was run separately per file.

### 3. Train/test overlap via identical name+address

**Not yet run** — UNKNOWN. The exact-duplicate check above was computed
per-file (train vs. train, test vs. test), not train-vs-test. A cross-split exact-match
check (e.g. "does any test Source 2 record have byte-identical name+address to a train
Source 2 record") has not been performed. This matters because if the *same*
real-world business appears in both train and test under different entity_ids, a model
that memorizes exact strings (rather than learning generalizable similarity) could look
artificially strong in validation. Recommended before serious modeling: a normalized
name+address anti-join between train and test per source, run via DuckDB in streaming
mode to stay memory-safe.

### 4. Ground-truth-derived leakage

- Ground truth only exists for train; test genuinely has no labels (CONFIRMED — no
  `test_ground_truth.tsv` or equivalent file exists anywhere in the resource tree).
- No column in any source file encodes match information directly (schema is just
  `entity_id, business_name, business_address, country` — no hidden foreign key to
  another source's ID).

### 5. Ordering / metadata leakage

- Row order was not found to encode useful information in the samples inspected (IDs
  appear in non-sequential, effectively random numeric order within each file) — INFERRED
  from sample rows, not exhaustively verified.
- No file timestamps, sequence numbers, or other metadata columns exist beyond the 4
  data columns per source file.

### 6. Country-based leakage

- France appears only in test, never in train (CONFIRMED, see `DATASET_INVENTORY.md`).
  This is an intentional distribution-shift test built into the challenge, not
  leakage — flagged here only so it isn't mistaken for a bug.

## Open items before modeling

1. Run the cross-split exact-match check described in §3.
2. Sanity-check the ground truth file itself for internal duplicate/self-referencing
   IDs within a `matched_entity_ids` list (see `GROUND_TRUTH_ANALYSIS.md`, "Malformed /
   duplicate labels").
3. Once a validation split is drawn from training data (see `VALIDATION_STRATEGY.md`),
   re-run an ID-overlap and exact-duplicate check between the drawn train and
   validation subsets, since intra-training leakage is possible even though
   train/test leakage was not found.

---

## Phase 2 additions (CONFIRMED)

- **Each S2/S3 record is matched to at most one S1:** all 7,638,365 matched IDs in the ground truth are distinct. This is a structural property, not leakage, and it motivates the optional one-owner assignment rule.
- **Country agreement within true pairs is 100%**, so blocking is restricted to the same country. Country is used only as an open-set key, never as a model feature.
- **Learned resources use training folds only:** the transliteration dictionary is learned from folds 0–7. Document frequencies and name-ambiguity counts are computed from unlabeled record text of the same split (train for validation, test for inference). This uses no labels and is the same transductive statistic available at test time.
- **Engineering incident (fixed):** concurrent DuckDB processes originally shared one spill directory, which corrupted a query in one run and was detected as an impossible value. Spill directories are now per process (`db.py`). All final outputs were produced after the fix, without concurrent spilling processes.
