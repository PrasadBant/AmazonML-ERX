import pandas as pd
import pytest

from business_entity_resolution.evaluation.metrics import macro_f05
from business_entity_resolution.evaluation.owner_validation import owner_selection


def test_readme_example():
    # predicted 3, 2 correct out of 2 true -> P = 2/3, R = 1 -> F0.5 = 0.714
    m, _ = macro_f05(pd.Series({1: 2}), pd.DataFrame({"s1": [1, 1, 1], "y": [1, 1, 0]}))
    assert m["macro_f0_5"] == pytest.approx(0.714, abs=1e-3)


def test_singletons_score_one_when_empty_and_zero_when_matched():
    n_true = pd.Series({1: 0, 2: 0})
    m, _ = macro_f05(n_true, pd.DataFrame({"s1": [2], "y": [0]}))
    assert m["macro_f0_5"] == pytest.approx(0.5)
    assert m["singleton_accuracy"] == pytest.approx(0.5)


def test_missing_prediction_for_matched_entity_scores_zero():
    m, _ = macro_f05(pd.Series({1: 3, 2: 0}), pd.DataFrame({"s1": [], "y": []}))
    assert m["macro_f0_5"] == pytest.approx(0.5)  # entity 1 -> 0, singleton 2 -> 1


def test_owner_selection_assigns_each_record_once_and_respects_margin():
    scores = pd.DataFrame({"s1": [1, 2, 3, 1], "t": [10, 10, 11, 12], "p": [0.9, 0.85, 0.7, 0.3]})
    chosen = owner_selection(scores, threshold=0.5)
    assert set(map(tuple, chosen[["s1", "t"]].values)) == {(1, 10), (3, 11)}
    # record 10 is contested (0.90 vs 0.85): dropped when the winner must lead by >= 0.1
    chosen = owner_selection(scores, threshold=0.5, margin=0.1)
    assert set(map(tuple, chosen[["s1", "t"]].values)) == {(3, 11)}
