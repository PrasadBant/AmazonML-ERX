# V5 validation

Same framework as v3/v4: 50,000 fold-9 S1, full train S2/S3 pool, per-entity macro F0.5. The one-owner
competition is scored with the same 380,804 competing train S1, re-scored with the v5 model.

| Version | Val F0.5 | Precision | Recall | Cand. recall | Avg / P95 cands | Singleton acc | India | US |
|---|---|---|---|---|---|---|---|---|
| v3 (plain, th 0.69) | 0.97520 | 0.98793 | 0.94766 | 0.97537 | 79.7 / 116 | 0.97316 | 0.96898 | 0.97940 |
| v4 (one-owner, th 0.55) | 0.97704 | 0.98838 | 0.95270 | 0.97537 | 79.7 / 116 | 0.97634 | 0.97178 | 0.98059 |
| **v5** (one-owner, th 0.61) | **0.97763** | 0.98949 | 0.95150 | 0.97537 | 79.7 / 116 | 0.97599 | 0.97197 | 0.98145 |

Robustness: threshold chosen on one half of the entities, scored on the other half.
- v4: 0.97726 / 0.97667 (mean 0.97697)
- v5: 0.97774 / 0.97741 (mean 0.97758). v5 wins on **both** halves, by +0.0006 mean.
- At fixed thresholds 0.55/0.60/0.65/0.70/0.75: v5 scores 0.97742/0.97761/0.97761/0.97742/0.97718 and v4 scores 0.97704/0.97693/0.97694/0.97681/0.97655. v5 is higher at every point.

The gain is small (+0.0006) but consistent, precision-led, and comes from more training data only.
Details are in `experiments/deep_optimization/v5_vs_v4_owner_validation.json`.
