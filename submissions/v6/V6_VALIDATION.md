# V6 validation

Same framework as v5 (50K fold-9 S1, 380,804 competing train S1 scored with the v5 model).

| Rule | All validation F0.5 | Contested S1 subset (n=424) F0.5 | Contested precision / recall |
|---|---|---|---|
| v5 (margin 0) | 0.97763 | 0.90045 | n/a |
| margin 0.05 | 0.97768 | 0.90618 | 0.928 / 0.887 |
| **v6: margin 0.10** | **0.97771** | **0.90991** | 0.944 / 0.846 |
| margin 0.20 | 0.97773 | 0.91204 | 0.964 / 0.795 |
| margin 0.30 | 0.97759 | 0.89469 | 0.960 / 0.751 |

Standard validation gain is +0.0001, because contested entities are rare there (0.85% of S1). On the contested subset,
which is the test-like population, the gain is +0.0095. Margin 0.10 was chosen over the 0.20 peak because the subset
is small (424 entities) and 0.10 costs half the recall.

Effect on test: 25,765 fewer matches (0.45%). 19,985 of them are France (2.28% of France matches); India loses 0.15%
and the US 0.08%.

Caveat: the contested subset is small, and France cannot be validated offline. **This is a targeted, evidence-based
bet, not a validated large gain.** It does not meet the "≥ +0.0013 on standard validation" bar.
