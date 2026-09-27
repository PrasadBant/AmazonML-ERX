# Business Entity Resolution — Amazon ML Challenge 2026 (team ZeroTrust)

Given business records from three independent sources, find every Source-2 / Source-3 record that
refers to the same real-world business as each Source-1 record. Source 1 is deduplicated. An entity
can have zero, one or many matches.

**Final submission: v5.** Validation macro F0.5 0.9776; public leaderboard 0.968 (as reported by the
team). Methodology: [`docs/methodology.md`](docs/methodology.md). Experiments: [`docs/experiments.md`](docs/experiments.md).

## Dataset
The official `student_resource/dataset` folder is not part of this repository (see `.gitignore`).

| File | Rows | Content |
|---|---|---|
| `train/train_source{1,2,3}.tsv` | 2.21M / 5.03M / 5.29M | `entity_id, business_name, business_address, country` |
| `train/train_ground_truth.tsv` | 2.21M | `source1_entity_id, matched_entity_ids` (comma list, empty = no match) |
| `test/test_source{1,2,3}.tsv` | 1.73M / 4.89M / 5.08M | same columns; adds `France`, a country unseen in training |

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Load | `data/load.py` | Raw TSVs into a DuckDB database (`data_work/er.duckdb`); never loads a whole file into pandas |
| Folds | `data/folds.py` | md5(entity_id) mod 10 over train S1: folds 0–7 train, fold 9 validation |
| Normalization | `preprocessing/` | Names (accents, punctuation, legal forms → core / compact name); Indic→Latin word dictionary learned from training labels; addresses (abbreviations, region codes, leading zeros) |
| Candidate generation | `blocking/` | Keys = pairs of each record's rarest name/address tokens + compact-name keys; IDF-scored overlap within country; top-100 + 20 name-channel + 20 address-channel per S1 |
| Features | `features/pairwise.py` | ~60 features: name/address similarity, house numbers, IDF overlap, name ambiguity, blocking ranks, candidate-group context |
| Matching | `matching/` | XGBoost (1,500 trees), threshold chosen on validation macro F0.5; country-agnostic |
| Decision + output | `pipeline/submission.py` | p ≥ threshold, and one-owner: each S2/S3 record goes to its best S1 only. Writes both TSVs |
| Evaluation | `evaluation/` | Official-style macro F0.5, one-owner validation, local submission checks, experiment logging |

Numbers for v5: candidate recall 97.5%; 80 average / 116 P95 candidates per S1; precision 0.989,
recall 0.952 and singleton accuracy 0.976 on validation.

### Metric
F0.5 = 1.25·P·R / (0.25·P + R), computed **per Source-1 entity** and averaged over all entities. A
singleton scores 1.0 when predicted empty and 0.0 when any match is predicted
(`evaluation/metrics.py`).

## Repository layout
```
src/business_entity_resolution/   the package: data, preprocessing, blocking, features, matching, evaluation, pipeline
scripts/                          entry points (run_pipeline, generate_candidates, train_model, generate_matches, validate_submission)
scripts/analysis/                 profiling, blocking sweeps, model ladder, error analysis, decision rules, one-owner validation
configs/                          v3.json … v6.json run configs; local.example.json for machine paths
tests/                            pytest suite (synthetic data, runs in seconds)
docs/                             methodology, experiments, docs/setup/ (initial data forensics reports)
experiments/                      experiment records (CSV/JSON) from the challenge
submissions/                      v1_nc, v3–v6 outputs + records; packages/ holds the final zip (TSVs/zips not in git)
output/                           README on the required output structure
data_work/                        working DuckDB database, features, models, scores (not in git, ~36 GB)
```

## Setup
```bash
python -m pip install -r requirements-dev.txt        # Python ≥ 3.11 (developed on 3.13)
cp configs/local.example.json configs/local.json     # set data_dir (and official_validator)
```
`data_dir` can also be given as the `ER_DATA` environment variable. `ER_WORK`, `ER_MEM` (DuckDB memory,
default 3GB) and `ER_THREADS` are optional.

## Running
```bash
# full pipeline for a config (~5 h on a 16 GB laptop; stages can be run individually with --stages)
python scripts/run_pipeline.py --config v5 --output-dir submissions/v7/output

# or step by step
python scripts/generate_candidates.py --config v5            # build DB + report blocking quality
python scripts/train_model.py --config v5                    # training/validation sets + model
python scripts/generate_matches.py --config v5 --output-dir submissions/v7/output
```
Output folders that already contain submission files are never overwritten unless `--overwrite` is
given.

## Validating outputs
```bash
python scripts/validate_submission.py submissions/v5/output             # local checks, both files
python scripts/validate_submission.py submissions/v5/output --official  # + official validator
python -m pytest                                                        # unit tests
```
The local checks stream through DuckDB and cover both files:
- exact header and exactly one tab per line;
- one row per test Source-1 id, with none missing, none extra and no duplicates;
- ids are S2-/S3- only, not repeated within a list, and present in the test files;
- every matched id is also a candidate.

The official `utils/validate_submission.py` is run on `matching_results.tsv` only: on the ~141M-id
candidate file it needs more than 10 GB of RAM.

## Submission structure
```
<team>_submission.zip
├── output/
│   ├── matching_results.tsv    source1_entity_id \t matched_entity_ids    (scored on the leaderboard)
│   └── candidate_pairs.tsv     source1_entity_id \t candidate_entity_ids  (final candidate set fed to the model)
├── code/business_entity_resolution/   src/, README.md, requirements.txt
└── Documentation_template.md          methodology
```
`candidate_pairs.tsv` is exactly the set of pairs the model scored. Matches are always a subset of it.

## Versions
| Version | Change | Validation F0.5 | Leaderboard |
|---|---|---|---|
| v3 | XGBoost on 120K entities, threshold 0.69 | 0.9752 | 0.963 |
| v4 | + one-owner rule, threshold 0.55 | 0.9770 | 0.966 |
| **v5** | + 180K training entities, threshold 0.61 | **0.9776** | **0.968** |
| v6 | experimental: + abstain on contested records | 0.9777 | lower than v5 |

`submissions/v5/` is the proven submission and is kept byte-for-byte. v6 is kept separately for reference.

## Reproducibility notes
- Each config records its model, thresholds and artifact names. Saved models and scores are in `data_work/`.
- The candidate ranking used during the challenge was not deterministic (tied float scores), so the
  submitted files cannot be regenerated bit-for-bit. The ranking is deterministic now; see
  methodology §9.
- An intended first-house-number blocking key never generated any keys, due to a regex escaping bug.
  It is documented in the methodology errata and removed from the code.

## Data policy
Only challenge-provided data is used. No external business data, registries, geocoding or entity-resolution APIs.
