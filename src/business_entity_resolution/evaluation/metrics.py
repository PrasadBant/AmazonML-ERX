"""The official metric: F0.5 computed per Source-1 entity, then averaged over ALL entities.

A singleton (no true matches) scores 1.0 when predicted empty and 0.0 when anything is predicted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def macro_f05(n_true: pd.Series, pred: pd.DataFrame, by: pd.Series | None = None) -> tuple[dict, pd.DataFrame]:
    """n_true: index s1 -> number of true matches (0 = singleton); must cover every evaluated S1.
    pred: one row per predicted pair with columns (s1, y), y = 1 if the pair is a true match.
    by: optional s1 -> group label (e.g. country) for a per-group breakdown."""
    agg = pred.groupby("s1")["y"].agg(["size", "sum"]).rename(columns={"size": "npred", "sum": "tp"})
    df = pd.DataFrame({"ntrue": n_true}).join(agg).fillna(0)
    p = np.where(df.npred > 0, df.tp / df.npred.clip(lower=1), 0)
    r = np.where(df.ntrue > 0, df.tp / df.ntrue.clip(lower=1), 0)
    f = np.where((p + r) > 0, 1.25 * p * r / (0.25 * p + r + 1e-12), 0.0)
    f = np.where((df.ntrue == 0) & (df.npred == 0), 1.0, f)
    df["f"] = f
    single = df.ntrue == 0
    out = dict(
        macro_f0_5=float(df.f.mean()),
        macro_precision=float(np.where(df.npred > 0, p, (df.ntrue == 0)).mean()),
        macro_recall=float(np.where(df.ntrue > 0, r, 1.0).mean()),
        micro_precision=float(df.tp.sum() / max(df.npred.sum(), 1)),
        micro_recall=float(df.tp.sum() / max(df.ntrue.sum(), 1)),
        singleton_accuracy=float((df.npred[single] == 0).mean()) if single.any() else float("nan"),
        false_merge_rate=float(((df.npred - df.tp).sum()) / max(df.npred.sum(), 1)),
        n_s1=int(len(df)),
    )
    if by is not None:
        out["by_group"] = {str(k): float(v) for k, v in df.f.groupby(by.reindex(df.index)).mean().items()}
    return out, df
