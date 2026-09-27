# Ground Truth Analysis — Amazon ML Challenge 2026

**Status: CONFIRMED** — produced by `workspace/src/data/forensics.py::profile_ground_truth`
via DuckDB streaming (no full-file load). Raw output in
`workspace/src/data/forensics_output.txt`. `train_ground_truth.tsv` was only read, never
modified.

## Structure

- 2,206,821 rows, matching `train_source1.tsv` exactly.
- `source1_entity_id` is unique across all rows (0 duplicates) — **every Source 1
  training entity has exactly one ground-truth row**. There is no entity missing a
  label and no entity labeled twice.
- `matched_entity_ids` is a comma-separated list of S2-/S3- IDs, empty string for
  singletons — matches the README's stated format exactly.

## Singleton rate

- **123,247 / 2,206,821 = 5.59%** of Source 1 training entities have zero matches
  (empty `matched_entity_ids`).
- This means **94.41%** of Source 1 entities match at least one Source 2/3 record.

## Match-count distribution (per Source 1 entity)

| # matches | # entities | % of total |
| ---: | ---: | ---: |
| 0 | 123,247 | 5.59% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.06% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10 | 534 | 0.02% |
| 11 | 37 | 0.002% |

Median is 3 matches per entity; the distribution is unimodal and right-skewed, capped
at 11 in this sample (no entity was observed with more than 11 matches). This is a
**one-to-many** relationship: a single Source 1 entity commonly maps to multiple
Source 2/3 records (a business appearing multiple times across the noisier sources),
not a strict 1:1 matching problem.

## Source 2 vs Source 3 contribution

Across all matched IDs in the ground truth: **3,944,746 are Source 3** and
**3,693,619 are Source 2** — roughly balanced (51.6% / 48.4%), with a slight lean
toward Source 3. INFERRED: this is consistent with Source 3 being marginally larger
than Source 2 overall (5.29M vs 5.03M rows), not evidence that Source 3 is
systematically easier or harder to match.

## Malformed / duplicate labels

- 0 duplicate `source1_entity_id` rows (CONFIRMED).
- No malformed rows encountered during parsing (DuckDB's CSV reader would have failed
  or misaligned columns on a broken row; none did across all 2,206,821 rows).
- Not yet checked: whether any individual `matched_entity_ids` list contains an
  internal duplicate ID or a self-referencing S1- ID (would violate the README's own
  submission rules if it existed in the labels) — **UNKNOWN**, worth a follow-up pass
  using the same duplicate-detection logic already written into
  `utils/validate_submission.py`, applied to the ground truth file itself as a sanity
  check rather than a submission.

## Implication for modeling (not yet acted on)

- Because ~94% of entities have at least one match and the modal count is 2-4, a
  classifier that defaults to "no match" for everything would score reasonably on
  precision for singletons but destroy recall on the majority; naive blocking needs to
  retrieve on the order of 3-5 true candidates per entity to keep recall high.
- The precision-heavy F_0.5 metric combined with a 5.6% true-singleton rate means
  correctly abstaining on those ~123K entities is worth real leaderboard points and
  should not be an afterthought.
