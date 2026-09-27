# Submissions

Each version folder holds `output/matching_results.tsv` and `output/candidate_pairs.tsv`, plus
`V<n>_CONFIG.md`, `V<n>_VALIDATION.md` and `V<n>_HASHES.txt` (SHA-256) where they were recorded.
The TSVs are 0.1–1.8 GB each and are **not tracked in git**. Keep them locally; the hashes identify them.

| Folder | Status | Validation F0.5 | Leaderboard (reported) | Run config |
|---|---|---|---|---|
| `v1_nc/` | first safety submission | 0.9648 | not recorded | – |
| `v3/` | fallback | 0.97520 | 0.963 | `configs/v3.json` |
| `v4/` | fallback | 0.97704 | 0.966 | `configs/v4.json` |
| **`v5/`** | **final / proven** | **0.97763** | **0.968** | `configs/v5.json` |
| `v6/` | experimental, not selected | 0.97771 | lower than v5 | `configs/v6.json` |
| `packages/` | final zip (`ZeroTrust_submission.zip` = v5), its staging folder, and the earlier v3 package | | | |

- **v5 vs v6:** the model, candidates and threshold (0.61) are the same. v6 adds a margin: a record claimed by
  several S1 entities is dropped unless the best one leads by ≥ 0.10. This removed 25,765 matches
  (2.28% of France's, ~0.1% of India's and the US's) and lowered the leaderboard score.
- **Do not edit the version folders.** New outputs go to a new folder; the scripts refuse to overwrite
  existing submission files.
- **Erratum for v4/v5:** `v4/V4_CONFIG.md` lists a "first house number" blocking key. That key never
  produced any keys (see `docs/methodology.md`, Errata); the v4/v5/v6 outputs are unaffected as recorded.
