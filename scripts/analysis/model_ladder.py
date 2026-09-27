"""Model ladder on identical data: rule baselines, logistic regression, random forest, extra trees, XGBoost.
Every model is scored with per-entity macro F0.5 on the same validation entities at its best threshold.

    python scripts/analysis/model_ladder.py --feature-tag v3 --train-rows 1000000
"""
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _path  # noqa: F401
from business_entity_resolution.db import connect
from business_entity_resolution.evaluation.metrics import macro_f05
from business_entity_resolution.matching.model import feature_matrix, select_threshold
from business_entity_resolution.matching.training_sets import training_part_path, validation_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-tag", default="v3")
    ap.add_argument("--train-rows", type=int, default=1_000_000)
    args = ap.parse_args()
    train = pd.read_parquet(training_part_path(args.feature_tag, 0)).sample(args.train_rows, random_state=0)
    valid = pd.read_parquet(validation_path(args.feature_tag))
    n_true = connect(read_only=True).execute("SELECT s1, n_match FROM ntrue_q_va").df().set_index("s1").n_match

    rules = {
        "rule_exact_normalized_name": valid.n_exact == 1,
        "rule_exact_compact_name": valid.cp_exact == 1,
        "rule_compact_name_and_address_tokens": (valid.cp_exact == 1) & (valid.a_tset >= 99.9),
        "rule_fuzzy_name85_address70": (valid.c_tsort >= 85) & ((valid.a_tset >= 70) | (valid.t_noaddr == 1)),
    }
    for name, mask in rules.items():
        print(f"{name}: {macro_f05(n_true, valid.loc[mask, ['s1', 'y']])[0]['macro_f0_5']:.5f}")

    x_train, x_valid = feature_matrix(train).fillna(-1), feature_matrix(valid).fillna(-1)
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
        "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=4, random_state=0),
        "extra_trees": ExtraTreesClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=4, random_state=0),
        "xgboost": xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.08, subsample=0.8,
                                     colsample_bytree=0.8, tree_method="hist", n_jobs=4),
    }
    for name, model in models.items():
        scores = model.fit(x_train, train.y).predict_proba(x_valid)[:, 1]
        threshold, metrics, _ = select_threshold(valid, np.asarray(scores), n_true)
        print(f"{name}: macro F0.5 {metrics['macro_f0_5']:.5f} at threshold {threshold}")


if __name__ == "__main__":
    main()
