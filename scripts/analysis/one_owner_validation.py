"""Validate the one-owner rule (v4/v5) and the contested-record margin (v6) with realistic competition.

    python scripts/analysis/one_owner_validation.py --model v3_p3_big
    python scripts/analysis/one_owner_validation.py --model v3_p3_big --test-profile --scores-dir infer_test_v5

Competitor scores are cached in data_work/owner_scores_<model>.pkl (competitor scan: ~12 min,
scoring ~380K competitors: ~40 min). --test-profile compares contested-record rates per country
between test and validation (the evidence behind v6).
"""
import argparse
import json

import numpy as np
import pandas as pd

import _path  # noqa: F401
from business_entity_resolution.blocking.candidates import BlockingParams
from business_entity_resolution.config import WORK_DIR, load_run_config
from business_entity_resolution.db import connect
from business_entity_resolution.evaluation.owner_validation import (evaluate_rule, find_competitors, half_split_score,
                                                                    owner_selection, score_competitors)


def test_profile(con, scores_dir: str, threshold: float) -> None:
    glob_path = (WORK_DIR / scores_dir / "chunk_*.parquet").as_posix()
    print(con.execute(f"""
        WITH above AS (SELECT s1, t, p FROM read_parquet('{glob_path}') WHERE p >= {threshold}),
        claims AS (SELECT t, count(*) n FROM above GROUP BY t),
        per_s1 AS (SELECT a.s1, count(*) FILTER (c.n > 1) contested, count(*) n FROM above a JOIN claims c USING (t) GROUP BY 1)
        SELECT r.country, count(*) n_s1, round(avg(coalesce(contested, 0)), 4) contested_records_per_s1
        FROM rec r LEFT JOIN per_s1 ON per_s1.s1 = r.rid WHERE r.split='test' AND r.src=1 GROUP BY 1 ORDER BY 1""").df())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v3_p3_big")
    ap.add_argument("--config", default="v5", help="blocking parameters")
    ap.add_argument("--test-profile", action="store_true")
    ap.add_argument("--scores-dir", default="infer_test_v5")
    args = ap.parse_args()
    cfg = load_run_config(args.config)
    con = connect(read_only=True)
    if args.test_profile:
        test_profile(con, args.scores_dir, cfg["decision"]["threshold"])
        return
    params = BlockingParams.from_config(cfg["blocking"])
    val = pd.read_parquet(WORK_DIR / f"pred_va_{args.model}.parquet")
    n_true = con.execute("SELECT s1, n_match FROM ntrue_q_va").df().set_index("s1").n_match
    cache = WORK_DIR / f"owner_scores_{args.model}.pkl"
    if cache.exists():
        all_scores = pd.read_pickle(cache)
    else:
        competitors = find_competitors(con, val, params)
        targets = set(val.loc[val.p >= 0.45, "t"])
        all_scores = pd.concat([val[["s1", "t", "p"]], score_competitors(con, competitors, args.model, params, targets)])
        all_scores.to_pickle(cache)

    grid = np.round(np.arange(0.45, 0.85, 0.01), 2)
    plain = lambda th: val.loc[val.p >= th, ["s1", "t"]]  # noqa: E731
    owner = lambda th: owner_selection(all_scores, th)  # noqa: E731
    report = {}
    for name, choose in (("plain", plain), ("one_owner", owner)):
        full = {float(th): evaluate_rule(val, n_true, choose(th))["macro_f0_5"] for th in grid}
        best = max(full, key=full.get)
        report[name] = {"best_threshold": best, "macro_f0_5": full[best],
                        "half_split": half_split_score(val, n_true, choose, grid)}
    # contested = validation S1 that win a record which another S1 also scores above the threshold
    th = cfg["decision"]["threshold"]
    above = all_scores[all_scores.p >= th].sort_values(["t", "p", "s1"], ascending=[True, False, True])
    winners = above[(above.groupby("t").s1.transform("size") > 1) & (above.groupby("t").cumcount() == 0)]
    contested_s1 = pd.Index(sorted(set(winners.s1) & set(n_true.index)))
    report["margin"] = {str(m): {"all": evaluate_rule(val, n_true, owner_selection(all_scores, th, m))["macro_f0_5"],
                                 "contested_subset": evaluate_rule(val, n_true, owner_selection(all_scores, th, m),
                                                                   contested_s1)["macro_f0_5"]}
                        for m in (0.0, 0.05, 0.1, 0.2, 0.3)}
    report["n_contested_validation_s1"] = len(contested_s1)
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
