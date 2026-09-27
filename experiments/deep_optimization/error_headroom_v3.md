# Error headroom (validation fold 9, 50K S1, CHAMPION_V3 scores at threshold 0.69)

"Oracle gain" is the macro-F0.5 increase if every error in the class were fixed and nothing else changed.

| Error class | Count (pairs) | Oracle gain | Fix needs | Cost | Risk |
|---|---|---|---|---|---|
| Matcher false negatives | 5,274 (2,290 target-no-address, 810 name-changed-same-address, 683 near-threshold) | +0.0107 | new features / model | test feature recompute 2.5 h | high |
| Blocking misses | 4,255 (1,621 India, 1,557 target-no-address, 547 transliterated, 530 US) | +0.0087 | new keys | full test re-inference 2.5 h+ | high |
| False positives | 1,175 (595 owned by another S1) | +0.0060 | one-owner rule | assembly only | low -> **done in V4** (+0.0018) |
| Singleton false matches | 76 of 2,832 singletons | incl. above | | | |
| Decision-rule variants (source, country, no-address thresholds, gates, calibration) | n/a | ≤ +0.0003 | rules | minutes | rejected (noise) |

A score of 0.985 would need almost all matcher false negatives recovered with no precision loss. That is not supported
within the remaining time.
