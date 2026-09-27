# Compute Strategy — Amazon ML Challenge 2026

**Status: INFERRED recommendation**, grounded in CONFIRMED dataset sizes from
`DATASET_INVENTORY.md`. Machine: ASUS TUF F15, 16 GB RAM, Windows.

## What can run locally

- **All EDA / forensics performed so far** ran locally via DuckDB streaming, peak
  memory capped at 6 GB (`SET memory_limit='6GB'` in both scripts), against files up to
  510 MB each. CONFIRMED to complete without issue.
- **Blocking / candidate generation** (e.g. TF-IDF + approximate nearest neighbors,
  phonetic/sorted-token keys, LSH on n-grams) over Source 2/3 (~5M rows each) is well
  within local capability using DuckDB, Polars, or PyArrow for the join/grouping logic
  and scikit-learn or a sparse-matrix approach for any vectorization — none of these
  require loading the full 5M-row table as dense in-memory Python objects.
- **Feature engineering on candidate pairs** (string-similarity features via rapidfuzz,
  token overlap, etc.) is local-friendly *as long as the candidate set stays a small
  multiple of the number of S1 entities* (e.g. 5,000-50,000 candidates per entity would
  not be local-friendly; low tens of candidates per entity, which is what a competent
  blocking stage should produce, is).
- **A classical ML matching model** (logistic regression, gradient boosting on
  engineered features) trained on a candidate-pair feature table sized to the true
  match-count distribution (median 3-4 candidates/entity, per `GROUND_TRUTH_ANALYSIS.md`)
  is local-friendly — this implies roughly 2.2M S1 entities x ~10-30 candidates/entity
  (allowing headroom above the true median for blocking recall) = tens of millions of
  feature rows, which fits comfortably as a Parquet/Arrow table processed in chunks or
  via Polars' lazy engine.

## What requires care / optimized processing

- **Never materialize a naive S1 x S2/S3 cross join.** 2.2M x 5M is ~11 trillion pairs
  — infeasible at any scale, not just locally. This is exactly why the README treats
  blocking as the primary lever; it is a hard technical requirement here, not an
  optimization nicety.
- **Loading a full source file into pandas without dtype control** risks multi-GB RAM
  usage on a 16 GB machine (a 500 MB TSV of short strings can expand several-fold as
  Python object arrays). Prefer Polars (`scan_csv` + lazy `.collect()` on filtered/
  aggregated results) or DuckDB for anything touching a full source file; reserve
  pandas for already-reduced, sample, or candidate-pair-scale tables.
- **Any embedding-based approach** (e.g. sentence embeddings for semantic name
  matching) needs batched inference — encoding 5M+ strings at once would be memory- and
  time-heavy locally. Batch it, and consider whether a much cheaper lexical/phonetic
  blocking pass can do most of the work first.

## Does GPU help?

- **UNKNOWN / not yet needed.** Classical string-similarity + gradient-boosted-tree
  approaches (the README's suggested Jaccard/Levenshtein/TF-IDF direction) do not
  benefit meaningfully from GPU. GPU becomes relevant only if the approach shifts to
  embedding-based semantic matching with a transformer encoder (permitted up to 8B
  parameters, MIT/Apache-2.0 licensed, per the README's constraint #5) — in which case
  batched GPU inference for encoding ~2.2M + 5M + 5M ≈ 12M training-side strings (and a
  similar count on test) would be meaningfully faster than CPU. Not recommended to
  commit to this path before a classical baseline exists to compare against.

## Is AWS necessary?

**Not yet, and not recommended at this stage.** Nothing in the confirmed dataset
profile (total raw data ~2.5 GB, largest single file 510 MB, blocking-scale candidate
tables in the tens-of-millions-of-rows range) exceeds what DuckDB/Polars can process
locally within a 16 GB budget when full cross joins are avoided. AWS credits should be
held in reserve for a concrete, demonstrated need — e.g., if a chosen embedding model
turns out to require GPU throughput unavailable locally, or if hyperparameter search
over the classical pipeline becomes wall-clock-bound. Recommend revisiting this after
the first working blocking + baseline-matching pipeline produces real timing numbers.

## RAM strategy summary

- Stream, never fully materialize, the 4 large source files (>200 MB each).
- Keep candidate-pair tables scoped to blocking output size, not S1 x S2/S3.
- Use Parquet as an intermediate format for any derived table larger than a few hundred
  MB, to avoid re-parsing TSVs repeatedly during iteration.
