# Resource Inventory — Amazon ML Challenge 2026

**Status: CONFIRMED** (verified by direct inspection on 2026-09-27)

## Key finding

The challenge resources were **not** in the Claude Code workspace
(`<project root>`) — that directory was empty. They were
found extracted at:

```
<downloads>/student_resource/student_resource\
```

(originally `<downloads>/student_resource.zip`, macOS-zipped — a sibling
`__MACOSX/` folder with `._*` AppleDouble files and a stray `.DS_Store` sit alongside
the real content and should be ignored / not treated as data).

All paths below are referenced in place from `workspace/src/data/config.py` in this
repo — nothing has been copied, moved, renamed, or modified.

## Documentation

| File | Size | Purpose |
| --- | --- | --- |
| `student_resource/README.md` | 14.1 KB | Official problem statement: data schema, output format, scoring (F_0.5), fair-play rules |
| `student_resource/Documentation_template.md` | 2.2 KB | Methodology write-up template for final submission |
| `Downloads/6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf` | 69 KB | Logistics: challenge window (2026-09-25 to 2026-09-27 IST), 5 submissions/day, top-100 team follow-up docs |
| `Downloads/6ab674645103d_emails_comms_amazon_ml_challenge_2026-1.pdf` | not yet read | Likely email comms — not inspected (not blocking for setup) |

## Code / Utilities

| File | Purpose |
| --- | --- |
| `student_resource/utils/validate_submission.py` | Official, stdlib-only validator for `matching_results.tsv` / `candidate_pairs.tsv`. Read in full — see `docs/SETUP_REPORT.md` §12 for exact behavior. **Not modified.** |

No `requirements.txt`, `pyproject.toml`, or `environment.yml` was shipped with the
official resources — CONFIRMED absent (searched `student_resource/` recursively).

## Dataset

See `docs/DATASET_INVENTORY.md` for full per-file forensics. Summary:

```
dataset/train/train_source1.tsv        210.1 MB
dataset/train/train_source2.tsv        489.3 MB
dataset/train/train_source3.tsv        503.7 MB
dataset/train/train_ground_truth.tsv   127.0 MB
dataset/test/test_source1.tsv          175.0 MB
dataset/test/test_source2.tsv          509.5 MB
dataset/test/test_source3.tsv          506.0 MB
```

All 7 expected files are present. Nothing is missing.

## Other

- `student_resource/student_resource/.DS_Store`, `dataset/.DS_Store` — macOS artifacts, harmless, ignored.
- No sample submission file was shipped in the official resources (CONFIRMED absent).
