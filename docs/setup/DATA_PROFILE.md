# Data Profile — Amazon ML Challenge 2026

**Status: CONFIRMED** unless marked otherwise. Source: `workspace/src/data/forensics.py`
and `workspace/src/data/leakage_and_quality.py` (raw output in
`workspace/src/data/forensics_output.txt` and `leakage_output.txt`).

## Business names

- Never missing in any of the 6 source files (CONFIRMED).
- Casing varies sharply by source: S1 is 0% all-caps, S2 is ~28% all-caps, S3 is
  ~8% all-caps (both train and test). INFERRED: S1 has undergone name normalization
  that S2/S3 have not.
- Contains multi-script text: Devanagari (Hindi) business names appear in Indian
  records (e.g. `राम मार्केटिंग प्राइवेट लिमिटेड`), alongside Latin-script names — CONFIRMED
  by direct sample inspection of S2/S3. Any name-similarity feature (Jaccard,
  Levenshtein, TF-IDF) must handle non-Latin scripts; naive ASCII-only normalization
  will silently break on these rows.
- Legal-suffix noise visible in samples (`Corp`, `Inc`, `Pvt Ltd`, `LLP`, `S.A.S`,
  `Sarl`) — CONFIRMED by sample inspection, matches the README's stated noise patterns.
- Leading punctuation/junk observed in a few samples (e.g. `-- Holloway Peak Inc
  Seafood`, `<< Team Ecole`) — CONFIRMED present, INFERRED to be a deliberate noise
  injection rather than a parsing artifact (values parsed cleanly as single fields).

## Addresses

- Missing only in Source 2 (train 3.4% / test 2.6%) and Source 3 (train 3.3% / test
  2.7%); never missing in Source 1 (CONFIRMED).
- Country-specific structure visible: US addresses are `street, city, state`; Indian
  addresses often include locality/district plus state name spelled out (no PIN code
  seen in the sampled rows — UNKNOWN whether PIN codes appear elsewhere in the corpus,
  needs a targeted regex scan before assuming absence); French addresses (test only)
  use `numéro rue, ville, région` with accented characters (e.g. `Boulevard du
  Président Franklin Roosevelt`).
- Address casing shift: Source 2 train addresses are 63.4% all-caps vs. 50.2% in test —
  see Train/Test Distribution below.

## Country

- Train: exactly `{US, India}` in all 3 train source files (CONFIRMED).
- Test: `{US, India, France}` in all 3 test source files (CONFIRMED) — France is
  entirely unseen in training, exactly as the README states. Any hard-coded
  country-conditional logic (e.g. an India-specific PIN-code regex applied only when
  `country == 'India'`) is safe; a global allowlist of `{US, India}` is not.
- Country distribution shift train→test: India's share rises from 40% (train S1) to
  47% (test S1) and US's share falls from 60% to 38%, with France taking the
  remaining ~15% — CONFIRMED by direct count, see `DATA_PROFILE` country tables in
  `DATASET_INVENTORY.md`. This is a material distribution shift a country-stratified
  validation split should account for (see `VALIDATION_STRATEGY.md`).

## IDs

- 100% prefix consistency (`S1-`/`S2-`/`S3-` matches filename) and 0 duplicate
  `entity_id` values within any of the 7 files (CONFIRMED).
- No shared numeric ID space between sources or between train/test (CONFIRMED via
  inner-join overlap counts, all zero) — IDs are independently assigned per file, not a
  meaningful blocking key on their own.

## Exact-duplicate rows (same normalized name + address)

| File | Duplicate rate |
| --- | ---: |
| train_source1 | 0.00% |
| train_source2 | 0.73% |
| train_source3 | 0.45% |
| test_source1 | 0.00% |
| test_source2 | 0.66% |
| test_source3 | 0.43% |

Source 1 has zero exact duplicates by construction (README calls it "deduplicated").
Source 2/3 carry a small but non-zero duplicate rate — INFERRED to be genuine repeated
listings (e.g. franchise locations or re-submitted records) rather than a parsing bug,
since the rate is consistent between train and test.

## Train/Test distribution differences

- **Country mix** shifts materially (see above) — the single largest distribution
  shift found.
- **Address casing** in Source 2 shifts from 63.4% all-caps (train) to 50.2%
  (test) — INFERRED to be partly explained by France's addition (French addresses
  are not all-caps in the samples seen), diluting the all-caps share rather than
  reflecting a change within the US/India population. Not yet decomposed
  by country — flagged as **UNKNOWN**, worth confirming before relying on
  casing as a feature.
- **Name/address length** distributions are close between train and test (name avg
  ~24-26 chars, address avg ~48-57 chars in both splits) — no material shift found.
- **Missingness** rates for address are close between train and test (within ~1pp per
  source) — no material shift found.
- **Row-count ratios** between S1/S2/S3 are stable between train and test (~1:2.3:2.4)
  — no evidence the test set changes the source-size relationship.

## What could not be determined at this stage (UNKNOWN)

- PIN/postal-code presence rate and format, by country — needs a dedicated regex pass,
  deferred (would require a full-column scan design, not done to keep this phase light).
- Whether France addresses in test ever mix in English tokens (transliteration).
- Whether India addresses in Source 2/3 systematically include PIN codes that Source 1
  lacks (would be a useful blocking signal if true).
