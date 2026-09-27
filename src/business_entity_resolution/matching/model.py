"""XGBoost pair matcher: training, threshold selection on macro F0.5, and model persistence.

Models are country-agnostic: `country` is never a feature because France appears only in test.
Artifacts in WORK_DIR: xgb_<name>.json (model), xgb_<name>_meta.pkl ({"features", "threshold"}),
pred_va_<name>.parquet (validation scores s1, t, y, p).
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from ..config import WORK_DIR
from ..evaluation.metrics import macro_f05
from .training_sets import training_part_path, validation_path

NON_FEATURE_COLUMNS = ("s1", "t", "y", "country")


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])


def select_threshold(pairs: pd.DataFrame, scores: np.ndarray, n_true: pd.Series) -> tuple[float, dict, list]:
    """Coarse grid 0.05..0.95, then 0.01 steps around the best, maximizing per-entity macro F0.5."""
    def evaluate(grid):
        results = [(float(th), macro_f05(n_true, pairs.loc[scores >= th, ["s1", "y"]])[0]) for th in grid]
        return max(results, key=lambda r: r[1]["macro_f0_5"]), [(th, m["macro_f0_5"]) for th, m in results]

    (coarse_th, _), coarse = evaluate(np.round(np.arange(0.05, 0.96, 0.05), 3))
    (th, metrics), _ = evaluate(np.round(np.arange(max(coarse_th - 0.06, 0.01), min(coarse_th + 0.06, 0.99), 0.01), 3))
    return th, metrics, coarse


def load_training_data(tag: str, offsets: list[int]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(training_part_path(tag, o)) for o in offsets], ignore_index=True)


def train_xgboost(con, feature_tag: str, model_name: str, offsets: list[int], params: dict) -> dict:
    train = load_training_data(feature_tag, offsets)
    valid = pd.read_parquet(validation_path(feature_tag))
    n_true = con.execute("SELECT s1, n_match, country FROM ntrue_q_va").df().set_index("s1")
    x_train = feature_matrix(train)
    t0 = time.time()
    model = xgb.XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=8, **params).fit(x_train, train.y)
    train_secs = time.time() - t0
    scores = model.predict_proba(feature_matrix(valid))[:, 1]
    threshold, _, coarse = select_threshold(valid, scores, n_true["n_match"])
    metrics, _ = macro_f05(n_true["n_match"], valid.loc[scores >= threshold, ["s1", "y"]], by=n_true["country"])
    save_model(model, model_name, list(x_train.columns), threshold)
    valid.assign(p=scores)[["s1", "t", "y", "p"]].to_parquet(WORK_DIR / f"pred_va_{model_name}.parquet")
    return {"threshold": threshold, "train_rows": len(train), "train_secs": round(train_secs, 1),
            "coarse_sweep": coarse, **metrics}


def save_model(model: xgb.XGBClassifier, name: str, features: list[str], threshold: float) -> None:
    model.save_model(WORK_DIR / f"xgb_{name}.json")
    with open(WORK_DIR / f"xgb_{name}_meta.pkl", "wb") as f:
        pickle.dump({"features": features, "threshold": threshold}, f)


def load_model(name: str) -> tuple[xgb.XGBClassifier, dict]:
    model = xgb.XGBClassifier()
    model.load_model(WORK_DIR / f"xgb_{name}.json")
    with open(WORK_DIR / f"xgb_{name}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, meta


def predict(model: xgb.XGBClassifier, meta: dict, feats: pd.DataFrame) -> np.ndarray:
    x = feats[meta["features"]]
    if "country" in x:
        raise ValueError("models must be country-agnostic (France is unseen in training)")
    return model.predict_proba(x)[:, 1]
