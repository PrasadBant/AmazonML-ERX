# Methodology — Business Entity Resolution (team ZeroTrust)

**Team:** Preranna Pandey (team leader), Prasad Bant  
**Final submission:** v5 (validation macro F0.5 0.9776; leaderboard 0.968 as reported by the team)

This is the maintained version of the methodology submitted with the challenge package
(`Documentation_template.md` inside `submissions/packages/ZeroTrust_submission.zip`). It contains two
corrections found during the repository clean-up; see [Errata](#errata).

## 1. Summary
A three-stage pipeline:
1. IDF-weighted *pair-key* blocking: pairs of each record's rarest name and address tokens, plus
   compact-name keys, with name- and address-channel top-K lists.
2. An XGBoost pairwise classifier over ~60 string-similarity, IDF-overlap, house-number and
   candidate-group-context features.
3. A macro-F0.5-tuned threshold combined with a **one-owner rule**: each S2/S3 record is given only to
   the Source-1 entity that scores it highest, mirroring the ground truth.

Key ideas are an Indic-script→Latin word dictionary learned only from training labels, and a
country-agnostic model (France appears only in test).

## 2. Data findings (all measured on the provided data; details in `docs/setup/`)
- Train: 2.21M Source-1 (S1), 5.03M S2 and 5.29M S3 records. Test: 1.73M S1, 4.89M S2 and 5.08M S3.
- Ground truth: 5.59% of S1 are singletons; the median is 3 matches and the maximum 11.
  **Every S2/S3 record is matched to at most one S1** (7,638,365 matched ids, all distinct).
  26% of S2/S3 records are unmatched distractors.
- Country agrees in 100% of true pairs, so blocking is within country. Country is an open-set string:
  France is test-only (15% of test S1).
- Name noise: typos, token shuffles, legal-suffix swaps, `M/s`/`Dr`/`Sri` prefixes, `(ID: …)` tags,
  domain-style names, and Indic-script names that are word-by-word transliterations of the English name
  (7.4% of true pairs).
- Address noise: abbreviations (St/Street, R./Rue), full vs. abbreviated state names, zero-padded house
  numbers (`00508`), small house-number perturbations, reordered components, `NULL` tokens, and a missing
  address in ~3.4% of S2/S3.

## 3. Normalization
- **Names** (`preprocessing/names.py`): NFKD with accents stripped, lower-case, and non-letter/digit runs
  collapsed to one space (any script). The *core* name drops legal forms and honorifics (EN/IN/FR); the
  *compact* name is the core tokens concatenated.
- **Indic scripts** (`preprocessing/transliteration.py`): a word dictionary learned from folds 0–7 label
  pairs with equal word counts (positional alignment, majority vote), with Unidecode romanization as the
  fallback. It holds 1,347 words and covers 97% of Indic words in train+test.
- **Addresses** (`preprocessing/address.py`): an abbreviation map, region name→code (a full name and its
  code map to the same token), leading zeros stripped, and noise tokens (`no`, `null`, …) dropped. The
  rules are identical for every country.

## 4. Candidate generation (blocking)
- **Keys** (`blocking/tokens.py`, `blocking/keys.py`): every unordered pair among a record's ≤3 rarest
  core-name tokens and ≤3 rarest address tokens (rarity = document frequency among S2/S3 of the same
  country), plus the compact name and its 8-character prefix. Single tokens were rejected: 640–22,900
  raw pairs per S1.
- **Scoring** (`blocking/candidates.py`): a candidate's score is the sum of ln(N/df) over shared keys,
  using only keys carried by ≤1,000 S2/S3 records. Per S1 we keep the union of the top-100 by total
  score, the top-20 by name-key score and the top-20 by address-key score.
- **Result:** 97.5% candidate recall on validation, 80 average / 116 P95 / ~135 max candidates per S1,
  and 141,240,118 test candidates (reduction ratio 0.99999 vs the full S1×(S2∪S3) product).
- **Blocking recall progression** (validation):

  | Change (settings) | Recall | Candidates / S1 |
  |---|---|---|
  | Pair keys (df cap 50, top-50) | 94.8% | 35 |
  | + transliteration (df cap 200, top-50) | 96.3% | 39 |
  | + address normalization and compact-prefix key (df cap 200, top-50) | 96.6% | 41 |
  | + df cap 1,000, top-100, name/address channel lists | 97.5% | 80 |

## 5. Matching model
- **Features** (`features/pairwise.py`):
  - *Name:* rapidfuzz ratio, token-sort, token-set and partial on the normalized, core and compact
    names; Jaro-Winkler; exact flags; length ratio.
  - *Address:* ratio, token-sort, token-set and partial; length ratio.
  - *Numbers:* counts, overlap and Jaccard; first-number equality and log |difference|.
  - *IDF-weighted shared token mass:* name and address, as a fraction of each side.
  - *Name ambiguity:* how many S1 and how many S2/S3 records share the compact name.
  - *Blocking:* scores and ranks per channel.
  - *Target flags:* missing address, Indic script, all caps.
  - *Candidate-group context:* best score among the S1's candidates and the gap to it, the candidate
    count, and the number of strong-name candidates.
- **Country is not a feature.** Removing it cost 0.0006 F0.5 on validation and keeps France safe.
- **Model** (`matching/model.py`): XGBoost hist with 1,500 trees, depth 10, lr 0.05, subsample and
  colsample 0.8, min_child_weight 5.
  - v5 is trained on blocked candidates of 180K fold 0–7 S1 entities (11.8M pairs).
  - Hard-negative sampling keeps every positive, every negative with name or address similarity ≥60 or
    blocking rank ≤10, and 20% of the remaining easy negatives.
- **Model ladder** (identical 1M-pair training subsample, v3 features):

  | Model | Macro F0.5 |
  |---|---|
  | Extra Trees | 0.955 |
  | Random Forest | 0.960 |
  | **XGBoost** | **0.970** |

  On the earlier v1 features, the best rule baseline scored 0.749 and logistic regression 0.910.

## 6. Decision rule and singleton handling
- **Threshold:** a coarse grid, then 0.01 steps, maximizing **per-entity macro F0.5** on validation.
- **One-owner** (`pipeline/submission.py`): a record goes only to its highest-scoring S1 among those
  with p ≥ threshold. Exact score ties go to the smallest S1 id.
- **Validation of the rule:** the 380,804 train S1 entities from other folds that compete for the same
  records were scored with the full pipeline (`evaluation/owner_validation.py`).
  - One-owner raised validation F0.5 from 0.9752 to 0.9770 (v4) and to 0.9776 with the v5 model.
  - Threshold chosen on one half of the entities and scored on the other: 0.9777 and 0.9774.
- **Singletons:** an S1 with no surviving candidate gets an empty list. Singleton accuracy is 0.976.
- **One-to-many:** every surviving candidate is returned. Top-1 decoding scored 0.690.

## 7. Validation protocol
- S1 entities are split into 10 folds by md5 of the id. The model trains on folds 0–7 and is evaluated
  on 50K S1 of fold 9; fold 8 is untouched.
- The S2/S3 pool is shared, as on test.
- Learned resources (the transliteration dictionary) use folds 0–7 only.
- The folds are balanced: 40% India, 5.6% singletons and 3.46 average matches in each.

## 8. Results

| Version | Change | Validation F0.5 | Leaderboard (reported) |
|---|---|---|---|
| v3 | 120K-S1 XGBoost, threshold 0.69 | 0.9752 | 0.963 |
| v4 | + one-owner, threshold 0.55 | 0.9770 | 0.966 |
| **v5** | + 180K-S1 training data, threshold 0.61 | **0.9776** | **0.968** |
| v6 (experimental) | + abstain on contested records (margin 0.10) | 0.9777 | lower than v5 |

- **Main remaining errors:**
  - Matcher misses on S2/S3 records without an address (oracle headroom +0.011).
  - Blocking misses on Indian and no-address records (+0.009).
- **Offline/leaderboard gap:** on test, France has 20–45× more contested records per S1 than India or
  the US (0.234 vs 0.005–0.011), a situation validation barely contains. v6 targeted it but scored
  lower on the leaderboard.

## 9. Reproducibility
- `configs/v3.json` … `configs/v6.json` record every setting and artifact name. The README explains how
  to run them.
- The repository clean-up checked the refactored code against the database that produced the
  submissions:
  - Normalization, transliteration and tokens match 100% on 30K sampled records.
  - The documented validation scores of v3–v6 are reproduced exactly from saved scores.
  - v3 and v6 match files are reproduced exactly from the saved test scores. v4 and v5 differ in 4 and
    5 pairs, all exact probability ties between two S1 entities that were originally broken
    arbitrarily; ties now go to the smallest S1 id.
- **Blocking determinism.** The original candidate ranking sorted on float sums aggregated in parallel,
  so mathematically tied candidates were ordered by rounding noise. About 2–5% of boundary candidates
  changed between runs, with recall unchanged (97.45–97.48% on the dev sample). Sort keys are now
  rounded, making reruns identical. As a result, **v3–v6 cannot be regenerated bit-for-bit**; the
  submitted files remain the reference.

## Errata
1. **The first-house-number blocking key was never active.** The submitted methodology and
   `submissions/v4/V4_CONFIG.md` list a "first house number" pair key. Its regex was written as `'\b…'` in a
   non-raw Python f-string, so `\b` became a backspace character and the key matched nothing: the token
   table holds zero such tokens. None of v3–v6 used it. The recall gain attributed to "first number +
   compact prefix" (96.4% → 96.6%) came from the compact prefix alone. The house-number *features*
   (`num_first_eq`, `num_first_absdiff`) were computed correctly. The dead code was removed.
2. The v4 match file's ties (see §9) were broken arbitrarily, not by smallest S1 id.

## Data policy
No external data, APIs, geocoding or business lookups were used. The only non-data resources are
open-source libraries (DuckDB, XGBoost, rapidfuzz, Unidecode, scikit-learn, pandas, pyarrow) and generic
normalization tables (street-type abbreviations, US and Indian state names).
