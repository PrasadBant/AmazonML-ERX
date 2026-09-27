# V5 configuration

v5 = v4 with a model trained on 1.5x more data, plus a re-selected threshold. Blocking, normalization, features,
candidate set and decision rule are identical to v4.

| Component | Setting |
|---|---|
| Blocking / normalization / features | Identical to v3/v4. Test features reused from `data_work/feat_test_infer_test_v3` (cached, not recomputed). |
| Model | `xgb_v3_p3_big`: XGBoost hist, 1500 trees, depth 10, lr 0.05, subsample 0.8, colsample 0.8, min_child_weight 5. Trained on blocked candidates of **180K** fold 0-7 S1 entities (3 disjoint 60K parts, 11.77M pairs after hard-negative sampling). v3/v4 used 120K. |
| Decision rule | One-owner: keep (s1, t) if p ≥ **0.61** and s1 is the highest-scoring test S1 for record t. |
| Singleton / one-to-many | Unchanged. An empty list if nothing survives; otherwise all surviving candidates. |

Reproduce:
```
python make_sets.py v3                            # N_TR=60000 NEGSAMP=0.2, OFFSET = 0, 60000, 120000, 180000 (SETS=q_tr)
NOCOUNTRY=1 SKIPBASE=1 TRPARTS=3 NEST=1500 LR=0.05 DEPTH=10 MSUFFIX=_p3_big python train.py v3
python rescore.py feat_test_infer_test_v3 v3_p3_big infer_test_v5
python assemble.py test infer_test_v5 ../../submissions/v5/output --threshold 0.61 --one-owner
```
TRPARTS=3 uses the first 3 part files in sorted name order: offsets 0, 120000 and 180000.
