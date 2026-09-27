"""Validating the one-owner decision rule with realistic competition.

On test, every S1 competes for the same S2/S3 records. On validation, the 50K fold-9 entities (V)
compete with train S1 from other folds, so those competitors must be scored too:
  1. competitor scan (blocking only): train S1 outside V whose candidate list ranks a record that
     the model scores >= p_min for some V entity within its top-`top_rank` (combined or name channel);
  2. score the competitors with the full pipeline (keeping only rows for those contested records);
  3. evaluate plain thresholding vs one-owner (optionally with an abstention margin) on V only.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..blocking.candidates import BlockingParams, generate_candidates
from ..features.pairwise import build_features
from ..matching.model import load_model, predict
from .metrics import macro_f05


def find_competitors(con, val_scores: pd.DataFrame, params: BlockingParams,
                     p_min: float = 0.45, top_rank: int = 20, chunk: int = 100_000) -> np.ndarray:
    con.register("contested_targets", val_scores.loc[val_scores.p >= p_min, ["t"]].drop_duplicates())
    con.execute(f"""CREATE OR REPLACE TEMP TABLE allq AS
        SELECT rid, (row_number() OVER (ORDER BY rid) - 1) // {chunk} c FROM fold
        WHERE rid NOT IN (SELECT rid FROM q_va)""")
    found = []
    for c in range(con.execute("SELECT max(c) + 1 FROM allq").fetchone()[0]):
        con.execute(f"CREATE OR REPLACE TEMP TABLE qc AS SELECT rid FROM allq WHERE c={c}")
        generate_candidates(con, "qc", "train", params)
        found.append(con.execute(f"""SELECT DISTINCT s1 FROM cand JOIN contested_targets USING (t)
                                     WHERE brank <= {top_rank} OR rn <= {top_rank}""").df())
    return pd.concat(found).s1.unique()


def score_competitors(con, competitors: np.ndarray, model_name: str, params: BlockingParams,
                      targets: set, chunk: int = 25_000) -> pd.DataFrame:
    model, meta = load_model(model_name)
    con.register("competitors", pd.DataFrame({"rid": competitors}))
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cq AS
        SELECT rid, (row_number() OVER (ORDER BY rid) - 1) // {chunk} c FROM competitors""")
    scored, t0 = [], time.time()
    for c in range(con.execute("SELECT coalesce(max(c) + 1, 0) FROM cq").fetchone()[0]):
        con.execute(f"CREATE OR REPLACE TEMP TABLE qc AS SELECT rid FROM cq WHERE c={c}")
        generate_candidates(con, "qc", "train", params)
        feats = build_features(con, "train", label=False)
        feats["p"] = predict(model, meta, feats)
        scored.append(feats.loc[feats.t.isin(targets), ["s1", "t", "p"]])
        print(f"  competitors chunk {c}: {time.time() - t0:.0f}s", flush=True)
    return pd.concat(scored, ignore_index=True)


def owner_selection(all_scores: pd.DataFrame, threshold: float, margin: float = 0.0) -> pd.DataFrame:
    """Winner per record among S1 with p >= threshold; contested records whose winner leads the
    runner-up by less than `margin` are dropped. Returns (s1, t) pairs."""
    above = all_scores[all_scores.p >= threshold].sort_values(["t", "p", "s1"], ascending=[True, False, True])
    rank = above.groupby("t").cumcount()
    top = above[rank == 0].set_index("t")
    runner_up = above[rank == 1].set_index("t").p
    lead = top.p - runner_up.reindex(top.index)
    keep = lead.isna() | (lead >= margin)
    return top[keep].reset_index()[["s1", "t"]]


def evaluate_rule(val_scores: pd.DataFrame, n_true: pd.Series, chosen: pd.DataFrame,
                  subset: pd.Index | None = None) -> dict:
    pairs = chosen.merge(val_scores[["s1", "t", "y"]], on=["s1", "t"])
    truth = n_true if subset is None else n_true[n_true.index.isin(subset)]
    return macro_f05(truth, pairs[pairs.s1.isin(truth.index)][["s1", "y"]])[0]


def half_split_score(val_scores, n_true, choose, grid) -> list[tuple[float, float]]:
    """Honest estimate: pick the threshold on even s1 ids, score on odd ids, and vice versa.
    choose(threshold) -> chosen (s1, t) pairs."""
    cache = {th: choose(th) for th in grid}
    out = []
    for fit_half, eval_half in ((0, 1), (1, 0)):
        fit_ids = n_true.index[n_true.index % 2 == fit_half]
        eval_ids = n_true.index[n_true.index % 2 == eval_half]
        best = max(grid, key=lambda th: evaluate_rule(val_scores, n_true, cache[th], fit_ids)["macro_f0_5"])
        out.append((float(best), evaluate_rule(val_scores, n_true, cache[best], eval_ids)["macro_f0_5"]))
    return out
