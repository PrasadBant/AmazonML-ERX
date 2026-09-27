# V6 configuration

v6 = v5 plus a contested-record abstention margin. The model, features, blocking, candidates and test scores are
identical to v5 (`data_work/infer_test_v5`).

Decision rule. For each S2/S3 record t, consider only S1 entities with p ≥ 0.61:
- if exactly one S1 qualifies, it gets t (same as v5);
- if several qualify, t goes to the top S1 **only if p_top − p_second ≥ 0.10**; otherwise **no S1** gets t.

Why: on test, France has 20–45× more contested records than India, the US or validation
(0.234 vs 0.005–0.011 per S1). On validation, contested S1 entities score F0.5 0.900 against 0.978 overall.
The rule is country-agnostic: no country value is used.

Reproduce:
```
python assemble.py test infer_test_v5 ../../submissions/v6/output --threshold 0.61 --one-owner --margin 0.1
```
