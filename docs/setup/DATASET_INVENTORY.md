# Dataset Inventory — Amazon ML Challenge 2026

**Status: CONFIRMED** — produced by `workspace/src/data/forensics.py`, a DuckDB
streaming query (never loads a full file into RAM). Raw output preserved at
`workspace/src/data/forensics_output.txt`.

## File-level facts

| File | Rows | Cols | Size |
| --- | ---: | ---: | ---: |
| train_source1.tsv | 2,206,821 | 4 | 210.1 MB |
| train_source2.tsv | 5,034,616 | 4 | 489.3 MB |
| train_source3.tsv | 5,285,603 | 4 | 503.7 MB |
| train_ground_truth.tsv | 2,206,821 | 2 | 127.0 MB |
| test_source1.tsv | 1,732,544 | 4 | 175.0 MB |
| test_source2.tsv | 4,887,273 | 4 | 509.5 MB |
| test_source3.tsv | 5,082,316 | 4 | 506.0 MB |

Delimiter: tab (`\t`), confirmed by successful DuckDB `read_csv(..., delim='\t')` parse
into the expected column count for every file. Encoding: UTF-8 (Devanagari names in
Source 2/3 parsed correctly once console output was forced to UTF-8; no decode errors
from DuckDB).

## Schema (identical across all `*_source{1,2,3}.tsv` files)

| Column | Type | Notes |
| --- | --- | --- |
| entity_id | VARCHAR | Prefixed `S1-`/`S2-`/`S3-` matching filename; CONFIRMED 100% prefix consistency per file, 0 duplicate IDs in any of the 7 files |
| business_name | VARCHAR | Never empty in any source (CONFIRMED, 0 missing across all 6 source files) |
| business_address | VARCHAR | Missing in Source 2/3 only (train: 168,967 / 175,916 rows; test: 129,408 / 136,098 rows), never missing in Source 1 |
| country | VARCHAR | Never missing |

`train_ground_truth.tsv` schema: `source1_entity_id` (VARCHAR), `matched_entity_ids`
(VARCHAR, comma-separated, empty string = singleton).

## Column-level details

- **entity_id**: purely numeric suffix after the source prefix (e.g. `S1-925783039`).
  No structural relationship found between S1/S2/S3 numeric suffixes (CONFIRMED no ID
  overlap across sources or across train/test — see `docs/LEAKAGE_AUDIT.md`).
- **business_name**: avg length ~24-26 chars across all sources/splits. Source 1 names
  are exclusively mixed/title case (0% all-caps). Source 2 names are ~28% all-caps;
  Source 3 names are ~8% all-caps. INFERRED: Source 1 (the curated reference source) has
  been normalized/cleaned more than Source 2/3, which look closer to raw ingested data.
- **business_address**: avg length ~48-57 chars. Source 2 addresses are majority
  all-caps in train (63.4%) but only half all-caps in test (50.2%) — see
  train/test shift note in `DATA_PROFILE.md`.
- **country**: train = `{US, India}` only (CONFIRMED, 2 distinct values in every train
  file). Test adds `France` (CONFIRMED, 3 distinct values in every test file) — matches
  the README's explicit statement that France is unseen in training.

## Row-count relationships (structural, CONFIRMED)

- `train_source1.tsv` rows == `train_ground_truth.tsv` rows == 2,206,821. Every Source 1
  training entity has exactly one ground-truth row (see `GROUND_TRUTH_ANALYSIS.md`).
- Source 2 and Source 3 are each roughly **2.3x** the size of Source 1 in both train and
  test — consistent with the README's framing of S1 as the deduplicated reference and
  S2/S3 as noisier, larger, less-deduplicated sources.
- Test set is smaller than train across all three sources (~79% of S1, ~97% of S2, ~96%
  of S3 by row count) despite adding a third country — INFERRED train is the larger
  split by design, not an artifact of missing data.
