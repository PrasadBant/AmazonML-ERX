# V4 configuration

v4 = CHAMPION_V3 scores + the one-owner assignment rule + a re-tuned threshold. No model, feature,
normalization or blocking change was made.

| Component | Setting |
|---|---|
| Blocking | Identical to v3. Pair keys (3 rarest core-name tokens, 3 rarest address tokens, first house number; all pairs), compact name, 8-char compact prefix. Per country, df ≤ 1000, IDF-sum score, union of top-100 combined, top-20 name-channel and top-20 address-channel. |
| Normalization | Identical to v3. |
| Features / model | Identical to v3: `xgb_v3_p2_big` (XGBoost, 1500 trees, depth 10, lr 0.05). Test scores reused from `data_work/infer_test_v3`. |
| Decision rule | Keep candidate (s1, t) if p ≥ **0.55** **and** s1 is the highest-scoring Source-1 entity for record t among all test S1 (ties broken by smallest s1 id). |
| Why one-owner | In the training ground truth, every S2/S3 record belongs to at most one S1 (7,638,365 matched ids, all distinct). |
| Singleton logic | An S1 with no surviving candidate gets an empty list. |
| One-to-many | Unchanged: all surviving candidates are returned. |

Reproduce:
```
python assemble.py test infer_test_v3 ../../submissions/v4/output --threshold 0.55 --one-owner
```
