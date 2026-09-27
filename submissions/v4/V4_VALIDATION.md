# V4 validation

Same validation framework as CHAMPION_V3: 50,000 S1 entities from fold 9, full train S2/S3 pool, per-entity macro F0.5.
For one-owner, the 380,804 train S1 entities from other folds that rank a contested record in their top-20 were scored
with the same pipeline and model, so competition is realistic. On test, the rule sees all 1.73M S1.

| Version | Val F0.5 | Precision | Recall | Candidate recall | Avg cands | P95 cands | Singleton acc | India | US |
|---|---|---|---|---|---|---|---|---|---|
| v3 (th 0.69) | 0.97520 | 0.98793 | 0.94766 | 0.97537 | 79.7 | 116 | 0.97316 | 0.96898 | 0.97940 |
| **v4** (one-owner, th 0.55) | **0.97704** | 0.98838 | 0.95270 | 0.97537 | 79.7 | 116 | 0.97634 | 0.97178 | 0.98059 |

Robustness (threshold chosen on one half of the entities, scored on the other half):
- v3: 0.97513 / 0.97499 (mean 0.97506)
- v4: 0.97726 / 0.97667 (mean 0.97696), i.e. **+0.0019**, positive on both halves.
- v4 beats v3's best at every threshold from 0.50 to 0.75 (0.9768–0.9770 vs 0.9752).

Output checks:
- Official `validate_submission.py --check-ids` on `matching_results.tsv`: **PASS**. There are 1,732,544 rows, of which 99,091 are empty.
- `check_candidates.py`, which applies the official rules to the 141M-id candidate file: **ALL PASS**, including matches ⊆ candidates.
- No S2/S3 id appears in more than one S1's match list.

Not changed from v3: the candidate set, and therefore blocking recall and candidate counts. The detailed record is in
`experiments/deep_optimization/one_owner_validation.json`.
