# Setup Report — Amazon ML Challenge 2026

All findings below are CONFIRMED by direct inspection unless marked INFERRED or
UNKNOWN. See individual docs for full detail and raw evidence.

## 1. Workspace

- Project path: `<project root>`
- Official challenge data was **not** in this workspace — found at
  `<downloads>/student_resource/student_resource\` (see
  `RESOURCE_INVENTORY.md`). Referenced in place via `workspace/src/data/config.py`,
  never copied.
- Key directories:
  - `workspace/src/data/` — dataset config, forensics scripts, access test
  - `workspace/src/{blocking,features,models,evaluation,inference}/` — empty, ready for modeling
  - `docs/` — this report and the 7 companion documents
  - `workspace/{experiments,outputs,submissions,logs}/` — empty, ready for use

## 2. Official Resources

- `student_resource/README.md` — full problem statement (read in full)
- `student_resource/Documentation_template.md` — methodology template
- `student_resource/utils/validate_submission.py` — official validator (read in full, run successfully, not modified)
- No `requirements.txt`/`pyproject.toml` shipped officially (confirmed absent)

## 3. Dataset

```
TRAIN
Source 1: 2,206,821 rows x 4 cols   210.1 MB
Source 2: 5,034,616 rows x 4 cols   489.3 MB
Source 3: 5,285,603 rows x 4 cols   503.7 MB
Ground Truth: 2,206,821 rows x 2 cols   127.0 MB

TEST
Source 1: 1,732,544 rows x 4 cols   175.0 MB
Source 2: 4,887,273 rows x 4 cols   509.5 MB
Source 3: 5,082,316 rows x 4 cols   506.0 MB
```

All 7 expected files present. Nothing missing, nothing fabricated.

## 4. Schema

`*_source{1,2,3}.tsv`: `entity_id` (prefixed S1-/S2-/S3-), `business_name` (never
missing), `business_address` (missing 2.6-3.4% in S2/S3 only, never missing in S1),
`country` (never missing; train = {US, India}, test adds France).

`train_ground_truth.tsv`: `source1_entity_id`, `matched_entity_ids` (comma-separated
S2-/S3- IDs, empty = singleton).

## 5. Ground Truth

- Singleton rate: **5.59%** (123,247 / 2,206,821)
- Match-count distribution: median 3, right-skewed, max observed 11 (full table in
  `GROUND_TRUTH_ANALYSIS.md`)
- Source 2 vs Source 3 share of matches: 48.4% / 51.6% — roughly balanced

## 6. Data Quality

- Multi-script names (Latin + Devanagari) confirmed present — similarity features must
  not assume ASCII.
- Casing differs sharply by source: S1 0% all-caps, S2 ~28%, S3 ~8%.
- Exact name+address duplicate rate: 0% (S1, both splits), 0.43-0.73% (S2/S3).
- 0 duplicate entity_ids anywhere; 0 malformed rows encountered.

## 7. Train/Test Differences

- **Country mix is the dominant shift**: US 60%→38%, India 40%→47%, France 0%→15%.
- Source 2 address all-caps rate drops from 63.4% (train) to 50.2% (test) — plausibly
  explained by France's addition (UNKNOWN, not yet decomposed by country).
- Name/address length distributions and missingness rates are stable between splits.

## 8. Leakage

- Zero entity_id overlap between train and test across all 3 sources (confirmed).
- No hidden match-encoding columns in any file.
- **Not yet checked**: cross-split exact name+address duplicates (train vs. test) —
  flagged as an open item in `LEAKAGE_AUDIT.md` §3, to run before trusting any
  validation split built later.

## 9. Validation

Recommend: split at the Source 1 entity level, keep the full S2/S3 pool shared across
train/val folds (mirrors how test scoring works), stratify by country and by
match-count bucket, and report validation F_0.5 broken out by country given the
train/test country shift. Full rationale in `VALIDATION_STRATEGY.md`.

## 10. Compute

- Local CPU + DuckDB/Polars streaming is sufficient for all EDA, blocking, and
  classical-model feature engineering at this dataset's scale.
- GPU: not needed for a classical (string-similarity + gradient boosting) approach;
  only relevant if a transformer-embedding approach is later chosen.
- RAM strategy: never materialize a naive S1×S2/S3 cross join (~11 trillion pairs);
  stream large files, keep candidate tables scoped to blocking output.
- AWS: not recommended yet — no demonstrated need at this dataset scale. Revisit after
  a working baseline produces real timing numbers.

## 11. Environment

- **Correction (Phase 2):** `python` on PATH is **3.13.12** (`C:\Program Files\Python313`), and all
  packages are installed there. The 3.12.10 originally reported came from a separate `python3`
  WindowsApps shim that is not used by this project. Always invoke `python`.
- Installed and import-verified: pandas 2.2.3, numpy 2.2.3, polars 1.39.2,
  scikit-learn 1.6.1 (pre-existing), duckdb 1.5.5, pyarrow 25.0.1, rapidfuzz 3.14.6
  (newly installed this session)
- `workspace/requirements.txt` created, pinned to the verified versions above.
- Import check: all 7 packages imported successfully in one process (see session log).

## 12. Official Validator

- Location: `<downloads>/student_resource/student_resource\utils\validate_submission.py`
- Run from the `student_resource/student_resource/` directory:
  ```
  python3 utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir dataset/test
  ```
- Confirmed behavior by running it with no output files present: prints required S1
  entity count (1,732,544), two warnings (ID-existence check off by default;
  candidate_pairs.tsv missing), then fails cleanly with "File not found:
  output/matching_results.tsv" — exactly as documented, exit code 1.
- `--check-ids` is opt-in and memory-heavier (loads all S2/S3 test IDs); off by default.
  No discrepancy found between the README's description and the validator's actual
  behavior.

## 13. Submission Structure

Two output files, both TSV, both required conceptually (only `matching_results.tsv` is
leaderboard-scored; `candidate_pairs.tsv` is required in the final zip and used to
audit blocking quality):

- `matching_results.tsv`: `source1_entity_id`, `matched_entity_ids` — one row per test
  S1 entity, S2-/S3- IDs only, no duplicates, empty allowed for singletons.
- `candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids` — the exact
  candidate set fed into the final matching model at inference time (not an earlier,
  since-filtered blocking pass). Every ID in `matching_results.tsv` should also appear
  here.
- Final submission zip: `output/` (both TSVs) + `code/business_entity_resolution/`
  (runnable pipeline, `README.md`, `requirements.txt`) + filled-in
  `Documentation_template.md`.

## 14. Missing Information

- Cross-split (train vs. test) exact-duplicate leakage check — not yet run.
- PIN/postal-code presence and format by country — not yet scanned.
- Whether S2 train/test address-casing shift is a France-dilution effect or a real
  shift within US/India — not yet decomposed.
- Internal duplicate/self-reference sanity check on `train_ground_truth.tsv`'s own
  `matched_entity_ids` lists — not yet run.
- Contents of `Downloads/6ab674645103d_emails_comms_amazon_ml_challenge_2026-1.pdf` —
  not yet read (logistics-only PDF, judged non-blocking for setup).

## 15. READINESS

**CAN WE START MODELING? YES**

The dataset is fully located, verified, profiled, and safe to read at scale; the
environment is installed and import-checked; the official validator has been read and
run successfully; a validation strategy has been designed (not yet implemented).

First recommended experiment: implement the entity-level train/validation split
described in `VALIDATION_STRATEGY.md` (country- and match-count-stratified), run the
two open leakage checks from `LEAKAGE_AUDIT.md` (§3 cross-split duplicates, and the
ground-truth self-consistency check) against that split, and only then build a first
blocking pass (e.g. normalized-token or phonetic key blocking on `business_name`) to
establish a recall ceiling before any matching model is trained.
