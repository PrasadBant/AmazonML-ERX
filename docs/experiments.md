# Experiments

The machine-readable records are the source of truth:
- `experiments/experiment_log.csv`, plus `experiments/configs/<id>.json` and `experiments/metrics/<id>.json`:
  every blocking, baseline, model and post-processing run during the challenge.
- `experiments/deep_optimization/`: the champion record, error analysis, oracle gains, decision-rule
  sweeps, one-owner validation and the internal scoreboard (`leaderboard.csv`).

All scores are per-entity macro F0.5 on the same validation set (50K fold-9 S1, full train S2/S3 pool)
unless noted. "Leaderboard" values are the public-leaderboard scores reported by the team.

## Blocking (20K dev entities of fold 9)
| Id | Change | Candidate recall | Avg candidates |
|---|---|---|---|
| B00 | single name/address tokens | rejected: 640–22,900 raw pairs / S1 | – |
| B02 | pair keys, df cap 50, top-50 | 0.9475 | 34.5 |
| B05 | + learned Indic transliteration | 0.9605 | 35.1 |
| B06 | same, df cap 200 | 0.9628 | 39.5 |
| – | + address normalization, compact-prefix key | 0.9661 | 41.4 |
| – | + df cap 1,000, top-100, 20 name + 20 address channel | 0.9748 | 80.2 |

On the 50K validation set the final blocking gives 0.9754 recall (S2 0.984, S3 0.968), with 92.2% of
entities having all their true matches among the candidates.

## Matcher
| Id | Change | Validation F0.5 |
|---|---|---|
| rules | best rule: fuzzy name ≥85 and address ≥70 | 0.749 |
| v1 M1 | logistic regression | 0.910 |
| v1 M3 | XGBoost, 600 trees | 0.9654 |
| v1_nc | XGBoost without `country` (kept for France) | 0.9648 |
| v2 | address normalization + name-ambiguity features, 80K S1 | 0.9699 |
| v2_p2 | same, 160K S1 | 0.9711 |
| v3_p2 | new blocking + house-number distance + hard negatives, 120K S1 | 0.9746 |
| v3_p2_big | 1,500 trees, depth 10, lr 0.05 | 0.9752 |
| v3_p3_big | same, 180K S1 | 0.9760 (plain threshold 0.78) |

Model ladder on an identical 1M-pair subsample (v3 features): Extra Trees 0.955, Random Forest 0.960,
XGBoost 0.970.

## Decision rules (on v3_p2_big scores)
| Rule | F0.5 | Status |
|---|---|---|
| global threshold 0.69 | 0.97520 | baseline |
| top-1 only | 0.69039 | rejected |
| per-source thresholds | 0.97524 | rejected (noise) |
| best-plus-second | 0.97522 | rejected |
| entity gate / margin to best | 0.97520 | rejected |
| no-address threshold | 0.97550 | rejected (in-sample tuned) |
| isotonic calibration, half-split | 0.97504 vs raw 0.97499 | rejected |
| **one-owner, threshold 0.55** | **0.97704** (half-split 0.97726 / 0.97667) | accepted → v4 |

## Submissions
| Version | Model | Rule | Validation F0.5 | Leaderboard |
|---|---|---|---|---|
| v1_nc | v1_nc | threshold 0.69 | 0.9648 | not recorded |
| v3 | v3_p2_big | threshold 0.69 | 0.97520 | 0.963 |
| v4 | v3_p2_big | one-owner, threshold 0.55 | 0.97704 | 0.966 |
| **v5** | v3_p3_big | one-owner, threshold 0.61 | **0.97763** (half-split 0.97774 / 0.97741) | **0.968** |
| v6 | v3_p3_big | one-owner, threshold 0.61, margin 0.10 | 0.97771 (contested subset 0.900 → 0.910) | lower than v5 |

v6 tested the finding that France has many more contested records on test than validation contains.
It did not improve the leaderboard, so v5 remains the final submission.

## Error headroom (v3 at threshold 0.69)
The oracle gain is the F0.5 increase if every error in the class were fixed:

| Error class | Oracle gain |
|---|---|
| Matcher false negatives | +0.0107 |
| Blocking misses | +0.0087 |
| All false positives | +0.0060 |
| False positives on records owned by another S1 | +0.0030 (addressed by one-owner) |
