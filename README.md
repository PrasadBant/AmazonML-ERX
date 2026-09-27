# Business Entity Resolution — Amazon ML Challenge 2026

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-informational)

**Team ZeroTrust** (Preranna Pandey — lead, Prasad Bant). A blocking → feature-engineering →
gradient-boosted-tree pipeline that matches noisy, multi-source business records without any
common identifier. Final submission **v5**: validation macro F0.5 **0.97763**, public leaderboard
**0.968** (self-reported by the team — see [Evaluation](#evaluation) for what that qualifier means).

Full write-ups: [`docs/methodology.md`](docs/methodology.md) (design, errata, reproducibility) and
[`docs/experiments.md`](docs/experiments.md) (every run tried during the challenge).

---

## Table of contents
1. [Challenge context](#challenge-context)
2. [Problem definition](#problem-definition)
3. [Why the problem is hard](#why-the-problem-is-hard)
4. [Solution overview](#solution-overview)
5. [Candidate generation (blocking)](#candidate-generation-blocking)
6. [Feature engineering](#feature-engineering)
7. [Matching model](#matching-model)
8. [Evaluation](#evaluation)
9. [Submission artifacts](#submission-artifacts)
10. [Repository structure](#repository-structure)
11. [Installation](#installation)
12. [Usage](#usage)
13. [Validating outputs](#validating-outputs)
14. [Versions](#versions)
15. [Engineering decisions](#engineering-decisions)
16. [Limitations](#limitations)
17. [Reproducibility](#reproducibility)
18. [Data policy](#data-policy)
19. [License](#license)

---

## Challenge context

The Amazon ML Challenge 2026 business-entity-resolution task provides business records from three
independent sources that describe the same underlying set of businesses with **no shared key** —
only a name, an address and a country string, each written inconsistently across sources. Source 1
(S1) is pre-deduplicated; Source 2 (S2) and Source 3 (S3) contain the records to be linked back to
it, plus unmatched distractors. Each S1 entity can have zero, one, or many true matches in S2/S3.

## Problem definition

```text
Source 1 (2.21M train / 1.73M test)         Source 2 + Source 3 (10.3M train / 10.0M test)
  entity_id                                   entity_id
  business_name         ─────────────►        business_name
  business_address                            business_address
  country                                     country

For every S1 record, output the subset of S2/S3 records that refer to the same business.
```

Output is two files per S1 id: the predicted matches, and the candidate set they were drawn from
(see [Submission artifacts](#submission-artifacts)). The scoring metric rewards precision more than
recall (F0.5), so a false merge costs more than a missed match.

## Why the problem is hard

Measured directly from the provided training data (`docs/setup/`, `docs/methodology.md §2`):

- **26%** of S2/S3 records are unmatched distractors with no true S1 partner.
- **5.59%** of S1 entities are singletons (zero true matches); the median match count is 3, the max 11.
- **7.4%** of true pairs have an Indic-script name that is a word-by-word transliteration of the
  English name on the other side.
- Name noise: typos, shuffled word order, legal-suffix swaps (`Pvt Ltd` / `LLC` / …), `M/s` / `Dr` /
  `Sri` prefixes, `(ID: …)` tags, domain-style names.
- Address noise: abbreviations (`St`/`Street`, `R.`/`Rue`), full vs. abbreviated region names,
  zero-padded house numbers (`00508`), small house-number perturbations, reordered components,
  `NULL` tokens, and a missing address in **~3.4%** of S2/S3 records.
- **France appears only in the test set** (15% of test S1) — the country string itself is open-set,
  so any solution that keys on a fixed country vocabulary breaks at test time.
- Country agrees in 100% of true pairs, so it is a safe *blocking* signal but (per §7) a harmful
  *matching* feature.

## Solution overview

```text
Raw TSVs
   │  data/load.py            → DuckDB (streamed, never loaded whole into pandas)
   ▼
Normalization
   │  preprocessing/          → accent/case/punctuation folding, legal-form stripping,
   │                             learned Indic→Latin dictionary, address abbreviation rules
   ▼
Candidate generation (blocking)
   │  blocking/                → pair-key overlap, IDF-scored, within-country
   ▼
candidate_pairs.tsv            ← exactly the pairs the model scores
   │
   ▼
Feature engineering
   │  features/pairwise.py     → ~56 name/address/number/context features
   ▼
Matching
   │  matching/                → XGBoost pairwise classifier, country excluded
   ▼
Decision + one-owner
   │  pipeline/submission.py   → threshold + each record assigned to its single best S1
   ▼
matching_results.tsv           ← scored on the leaderboard
```

Every stage is a DuckDB SQL pipeline or a vectorized batch job; nothing iterates over individual
Python-level rows.

## Candidate generation (blocking)

All-pairs comparison is infeasible: the training set alone is 2.21M × 10.3M ≈ 2.3×10¹³ possible
pairs. Blocking reduces this to a manageable candidate set before any model runs.

**Keys** (`blocking/tokens.py`, `blocking/keys.py`): every unordered pair among a record's ≤3
rarest *core*-name tokens and ≤3 rarest address tokens (rarity = document frequency among S2/S3 of
the same country), plus the compact name and its 8-character prefix as single keys.

**Scoring** (`blocking/candidates.py`): a candidate's score is the sum of `ln(N/df)` over its shared
keys, restricted to keys carried by ≤1,000 S2/S3 records (`df_cap`). Per S1, the kept set is the
union of the **top-100 by total score**, the **top-20 by name-channel score** and the **top-20 by
address-channel score**.

**Result on validation:** 97.5% candidate recall, 80 average / 116 P95 candidates per S1. On the
full test set: **141,240,118 candidate pairs**, a reduction ratio of 0.99999 against the naive
S1 × (S2∪S3) product.

Recall progression as blocking was tightened (validation, `docs/methodology.md §4`):

| Change | Recall | Candidates / S1 |
|---|---:|---:|
| Pair keys (df cap 50, top-50) | 94.8% | 35 |
| + learned Indic transliteration | 96.3% | 39 |
| + address normalization, compact-prefix key | 96.6% | 41 |
| + df cap 1,000, top-100, name/address channel lists | **97.5%** | **80** |

`candidate_pairs.tsv` is the exact set the matching model scores; every predicted match in
`matching_results.tsv` is a subset of it.

## Feature engineering

`features/pairwise.py` computes ~56 columns per candidate pair (the real feature list is recorded
in `experiments/deep_optimization/CHAMPION_V3.json`):

| Group | Examples |
|---|---|
| Name similarity | rapidfuzz ratio, token-sort, token-set, partial, Jaro-Winkler — each on the normalized, *core* and *compact* name variants; exact-match flags; length ratio |
| Address similarity | same fuzzy-matching set as names, plus length ratio |
| House numbers | count, overlap, Jaccard of extracted numbers; first-number equality and log-absolute-difference |
| IDF-weighted overlap | shared-token mass (as a fraction of each side) for name and address separately |
| Name ambiguity | how many S1 records, and how many S2/S3 records, share the same compact name |
| Blocking context | per-channel blocking scores and ranks |
| Target flags | missing address, Indic script detected, all-caps |
| Candidate-group context | best blocking score among the S1's own candidates and the gap to it, total candidate count, count of strong-name candidates |

**Country is deliberately not a feature** — `matching/model.py` raises `ValueError` if it appears in
the feature set. Removing it cost 0.0006 F0.5 on validation but is required for a model that has to
generalize to France, a country never seen in training.

## Matching model

`matching/model.py`: `xgb.XGBClassifier(tree_method="hist", ...)`, hyperparameters set per config —
v5 uses **1,500 trees, depth 10, learning rate 0.05, subsample 0.8, colsample 0.8,
min_child_weight 5** (`configs/v5.json`).

- Trained on blocked candidates of **180K** fold 0–7 S1 entities (≈11.8M pairs).
- Hard-negative sampling: every positive is kept; negatives are kept if name or address similarity
  ≥60 or blocking rank ≤10, plus a random 20% of the remaining easy negatives.
- 10-fold split by `md5(entity_id) mod 10`: folds 0–7 train, fold 9 (50K S1) is validation, fold 8 is
  held untouched. All learned resources (the transliteration dictionary) use folds 0–7 only.

**Model choice was measured, not assumed** — identical 1M-pair subsample, identical features
(`docs/experiments.md`):

| Model | Macro F0.5 |
|---|---:|
| Extra Trees | 0.955 |
| Random Forest | 0.960 |
| **XGBoost** | **0.970** |

Earlier baselines on the same task: a fuzzy-matching rule (name ≥85, address ≥70) scored 0.749;
logistic regression scored 0.910.

**Decision rule** (`pipeline/submission.py`): a threshold on `p`, chosen by grid search to maximize
per-entity macro F0.5 on validation, combined with a **one-owner rule** — a candidate record is kept
only for the single S1 entity that scores it highest among those above threshold (exact ties go to
the smallest S1 id). This was validated against 380,804 real train-S1 entities from other folds that
compete for the same S2/S3 records (`evaluation/owner_validation.py`), where it raised validation
F0.5 from 0.9752 to 0.9770. Singletons (no surviving candidate) get an empty list; singleton
accuracy on v5 is 0.976.

## Evaluation

The official metric is F0.5, computed **per Source-1 entity** and macro-averaged over all entities
(`evaluation/metrics.py`):

```text
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Weighting precision at 2.5× recall reflects the task: a false merge (predicting a match that doesn't
exist) is penalized more than a missed match. A singleton scores 1.0 if predicted empty and 0.0 if
any match is predicted for it.

Validation results (`submissions/v5/V5_VALIDATION.md`, 50K fold-9 S1, full train S2/S3 pool):

| Version | Val F0.5 | Precision | Recall | Cand. recall | Singleton acc. | India | US | Leaderboard¹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 (plain, th 0.69) | 0.97520 | 0.98793 | 0.94766 | 0.97537 | 0.97316 | 0.96898 | 0.97940 | 0.963 |
| v4 (one-owner, th 0.55) | 0.97704 | 0.98838 | 0.95270 | 0.97537 | 0.97634 | 0.97178 | 0.98059 | 0.966 |
| **v5 (one-owner, th 0.61)** | **0.97763** | 0.98949 | 0.95150 | 0.97537 | 0.97599 | 0.97197 | 0.98145 | **0.968** |
| v6 (one-owner, th 0.61, margin 0.10) | 0.97771 | — | — | 0.97537 | — | — | — | lower than v5 |

¹ Leaderboard scores are the public-leaderboard numbers the team recorded during the challenge
(`configs/v*.json`, `submissions/README.md`). There is no CI-verifiable artifact for them in this
repository — no screenshot or exported log — so they are reported here with the same qualifier the
project's own docs use: **self-reported**.

Threshold-robustness check (threshold chosen on one half of validation entities, scored on the
other): v5 scores 0.97774 / 0.97741 (mean 0.97758), beating v4's 0.97726 / 0.97667 on **both**
halves.

## Submission artifacts

```text
matching_results.tsv    source1_entity_id \t matched_entity_ids   (comma-separated, scored on leaderboard)
candidate_pairs.tsv     source1_entity_id \t candidate_entity_ids (the full candidate set the model scored)
```

Every id in `matching_results.tsv` is guaranteed to also appear in `candidate_pairs.tsv` for the
same S1 — matches are always a subset of candidates. The final challenge package bundled both files
under `output/`, the source package under `code/business_entity_resolution/`, and a methodology
write-up as `Documentation_template.md`.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt / requirements-dev.txt
│
├── src/business_entity_resolution/
│   ├── data/            load.py, folds.py           — DuckDB ingestion, train/val split
│   ├── preprocessing/   names.py, address.py, transliteration.py
│   ├── blocking/        tokens.py, keys.py, candidates.py
│   ├── features/        pairwise.py                 — ~56 candidate-pair features
│   ├── matching/        model.py, training_sets.py, inference.py
│   ├── evaluation/      metrics.py, owner_validation.py, submission_checks.py, tracking.py
│   ├── pipeline/        stages.py, submission.py     — threshold + one-owner + output
│   ├── config.py, db.py, ids.py
│
├── scripts/             run_pipeline.py, generate_candidates.py, train_model.py,
│                        generate_matches.py, validate_submission.py
│   └── analysis/        blocking_sweep, model_ladder, error_analysis, decision_rules,
│                        one_owner_validation, data_profile, show_matches, check_reproduction
│
├── configs/              v3.json … v6.json (full run configs), local.example.json (machine paths)
├── tests/                 pytest suite: 4 files, 19 tests, synthetic data
├── docs/                  methodology.md, experiments.md, setup/ (initial data-forensics reports)
├── experiments/           experiment_log.csv, per-run configs/metrics, deep_optimization/
│
├── submissions/
│   ├── README.md          version table and notes (source of truth for reported scores)
│   └── v4/, v5/, v6/       V<n>_CONFIG.md, V<n>_VALIDATION.md, V<n>_HASHES.txt (SHA-256 of TSVs)
│
├── output/               README describing the required output directory shape
└── data_work/             working DuckDB DB, features, models, scores — not in git, ~36 GB
```

Note: `submissions/v1_nc/` and `submissions/v3/` are **not present as directories** — those two early
submissions exist only as historical entries in `submissions/README.md` and `experiments/`; their
output TSVs were never retained locally. `submissions/packages/` (the final zip and its staging
folder) is `.gitignore`-excluded and is also not present in this checkout. `v4/`, `v5/` and `v6/` are
the versions with folders actually on disk (their TSVs are themselves gitignored — 0.1–1.8 GB each;
SHA-256 hashes in `V<n>_HASHES.txt` identify them).

## Installation

```bash
git clone <this-repository-url>
cd AmazonML-ERX

python -m pip install -r requirements-dev.txt   # Python >= 3.11 (developed on 3.13)
cp configs/local.example.json configs/local.json
# edit configs/local.json: set data_dir to the challenge dataset location
```

`data_dir` may also be given via the `ER_DATA` environment variable. `ER_WORK`, `ER_MEM` (DuckDB
memory limit, default 3 GB) and `ER_THREADS` are optional overrides. The official challenge dataset
(`student_resource/dataset/`) is not part of this repository — see `.gitignore`.

Runtime dependencies (`requirements.txt`): `duckdb`, `numpy`, `pandas`, `pyarrow`, `rapidfuzz`,
`scikit-learn`, `xgboost`, `psutil`, `Unidecode`.

## Usage

```bash
# full pipeline for a given config (runtime depends on hardware; --stages runs a subset)
python scripts/run_pipeline.py --config v5 --output-dir submissions/v7/output

# or step by step
python scripts/generate_candidates.py --config v5   # build the DB + report blocking quality
python scripts/train_model.py --config v5            # build training/validation sets + train
python scripts/generate_matches.py --config v5 --output-dir submissions/v7/output
```

Output folders that already contain submission files are never overwritten unless `--overwrite` is
passed — this protects `submissions/v5/` from accidental regeneration.

## Validating outputs

```bash
python scripts/validate_submission.py submissions/v5/output              # local checks, both files
python scripts/validate_submission.py submissions/v5/output --official   # + externally configured validator
python -m pytest                                                          # unit tests
```

The local checks stream through DuckDB and cover both files: exact header and one tab per line; one
row per test S1 id with none missing, extra or duplicated; ids are S2-/S3-only, not repeated within a
list, and present in the test files; every matched id is also a candidate.

`--official` shells out to a validator script whose path is set by `official_validator` in
`configs/local.json` — that script is provided by the challenge, is **not included in this
repository**, and is run against `matching_results.tsv` only, since the ~141M-row candidate file
needs more than 10 GB of RAM for it.

## Versions

| Version | Change | Validation F0.5 | Leaderboard¹ | Status |
|---|---|---:|---:|---|
| v1_nc | XGBoost without `country`, threshold 0.69 | 0.9648 | not recorded | first safety submission |
| v3 | XGBoost, 120K S1, threshold 0.69 | 0.97520 | 0.963 | fallback |
| v4 | + one-owner rule, threshold 0.55 | 0.97704 | 0.966 | fallback |
| **v5** | + 180K training S1, threshold 0.61 | **0.97763** | **0.968** | **final / proven** |
| v6 | + abstain on contested records (margin 0.10) | 0.97771 | lower than v5 | experimental, not selected |

¹ self-reported, see [Evaluation](#evaluation).

v6 tested the observation that France has far more contested records on test than validation
contains (see [Limitations](#limitations)). It dropped 25,765 matches (2.28% of France's, ~0.1% of
India's/US's) and scored *lower* on the leaderboard despite a marginally higher validation score —
**v5 remains the final submission and is kept byte-for-byte in `submissions/v5/`.**

## Engineering decisions

- **Precision over recall, throughout.** F0.5 weights precision 2.5×. The one-owner rule, the hard
  threshold, and excluding `country` as a feature (which would have overfit to training-only
  countries) are all precision-preserving choices that cost small amounts of recall.
- **Blocking width vs. candidate recall.** df cap 1,000 / top-100+20+20 was chosen because tighter
  caps (df cap 50/200, top-50) plateaued around 96% recall — the last ~1.5 points of recall required
  roughly doubling candidates per S1 (41 → 80), a trade-off accepted because the matcher can reject
  bad candidates but can never recover ones blocking never produced.
- **Country in blocking, not in matching.** Safe as a blocking filter (100% of true pairs agree on
  country) but actively harmful as a model feature once France — unseen in training — appears at test
  time.
- **Validation vs. leaderboard gap is real and documented, not glossed over.** v6 improved validation
  F0.5 (0.97771 vs. 0.97763) but scored worse on the leaderboard; the team kept v5 rather than trust
  a marginal, direction-uncertain validation gain over a directly-observed leaderboard regression.

## Limitations

- **No test ground truth.** All F0.5 numbers except the leaderboard score are computed on a held-out
  validation fold of the *training* set, not the actual test labels.
- **Validation under-represents France's difficulty.** On test, France has 20–45× more contested
  records per S1 than India or the US (0.234 vs. 0.005–0.011 contested-record rate per
  `docs/methodology.md §8`) — a regime validation barely samples, which is the likely cause of the
  v5/v6 validation-vs-leaderboard disagreement.
- **Error headroom is concentrated, not eliminated.** Oracle analysis on v3 (`docs/experiments.md`)
  attributes +0.0107 F0.5 to matcher false negatives and +0.0087 to blocking misses, mostly on
  records with no address or Indian-locale noise.
- **Candidate ranking was non-deterministic during the challenge** (see
  [Reproducibility](#reproducibility)) — the exact submitted `v3`–`v6` files cannot be regenerated
  bit-for-bit from a fresh run, only their recorded scores.

## Reproducibility

- Each `configs/v*.json` records the model hyperparameters, threshold, and artifact names used to
  produce that version; saved models and scores live under `data_work/`.
- A repository clean-up cross-checked the refactored code against the database that produced the
  original submissions: normalization, transliteration and tokenization match 100% on 30K sampled
  records; the documented validation scores of v3–v6 reproduce exactly from saved scores; v3 and v6
  match files reproduce exactly; v4 and v5 differ by 4 and 5 pairs respectively, all exact
  probability ties originally broken arbitrarily (ties are now deterministic — smallest S1 id wins).
- **Known non-determinism, now fixed:** the original candidate ranking sorted on floating-point sums
  aggregated in parallel, so tied candidates could be ordered differently between runs (~2–5% of
  boundary candidates, recall unaffected at 97.45–97.48%). Sort keys are now rounded before
  comparison, making reruns identical — but this means `v3`–`v6` as submitted remain the reference
  artifacts and are not regenerable bit-for-bit going forward.
- **Documented erratum:** `submissions/v4/V4_CONFIG.md` lists a "first house number" blocking key
  that was never actually active — its regex used `\b` inside a non-raw Python f-string, which
  silently became a backspace character, so the key matched zero tokens in any run. The recall gain
  originally attributed to it came entirely from the compact-name prefix key added alongside it. The
  house-number *features* used by the model (`num_first_eq`, `num_first_absdiff`) were computed
  correctly and are unaffected; the dead blocking-key code has been removed.

## Data policy

Only the challenge-provided TSVs are used. No external business registries, geocoding services,
entity-resolution APIs, or other external data were used at any pipeline stage. Third-party
dependencies are limited to open-source libraries (DuckDB, XGBoost, rapidfuzz, Unidecode,
scikit-learn, pandas, pyarrow) and generic, non-proprietary normalization tables (street-type
abbreviations, US/Indian region-name-to-code maps).

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Team ZeroTrust (Preranna Pandey, Prasad Bant).
