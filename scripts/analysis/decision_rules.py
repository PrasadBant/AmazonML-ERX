"""Decision-rule experiments on saved validation scores (no retraining): global threshold, top-1,
per-source thresholds, best-plus-second, entity gate, margin to best, missing-address threshold,
per-country threshold (diagnostic only) and isotonic calibration (honest half-split).

    python scripts/analysis/decision_rules.py --model v3_p2_big
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import _path  # noqa: F401
from business_entity_resolution.config import WORK_DIR
from business_entity_resolution.db import connect
from business_entity_resolution.evaluation.metrics import macro_f05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v3_p2_big")
    ap.add_argument("--feature-tag", default="v3")
    args = ap.parse_args()
    con = connect(read_only=True)
    d = pd.read_parquet(WORK_DIR / f"feat_{args.feature_tag}_q_va.parquet", columns=["s1", "t", "src", "t_noaddr"]) \
        .merge(pd.read_parquet(WORK_DIR / f"pred_va_{args.model}.parquet"), on=["s1", "t"])
    nt = con.execute("SELECT s1, n_match, country FROM ntrue_q_va").df().set_index("s1")
    d = d.merge(nt[["country"]], left_on="s1", right_index=True)
    d["pmax"] = d.groupby("s1").p.transform("max")
    d["rank"] = d.groupby("s1").p.rank(ascending=False, method="first")
    score = lambda mask: macro_f05(nt.n_match, d.loc[mask, ["s1", "y"]])[0]["macro_f0_5"]  # noqa: E731
    grid = np.round(np.arange(0.40, 0.90, 0.01), 2)
    pairs = lambda lo, hi, step: [(a, b) for a in np.arange(lo, hi, step) for b in np.arange(lo, hi, step)]  # noqa: E731

    experiments = {
        "global_threshold": (lambda th: d.p >= th, [(t,) for t in grid]),
        "top1_only": (lambda th: (d.p >= th) & (d["rank"] == 1), [(t,) for t in grid]),
        "per_source_thresholds": (lambda a, b: ((d.src == 2) & (d.p >= a)) | ((d.src == 3) & (d.p >= b)),
                                  pairs(0.55, 0.85, 0.02)),
        "best_plus_second_rule": (lambda a, b: ((d["rank"] == 1) & (d.p >= a)) | ((d["rank"] > 1) & (d.p >= b)),
                                  [(a, b) for a in np.arange(0.45, 0.85, 0.03) for b in np.arange(0.55, 0.95, 0.03)]),
        "entity_gate": (lambda th, g: (d.p >= th) & (d.pmax >= g),
                        [(th, g) for th in (0.65, 0.69, 0.73) for g in (0.69, 0.75, 0.8, 0.85, 0.9)]),
        "margin_to_best": (lambda th, a: (d.p >= th) & (d.p >= a * d.pmax),
                           [(th, a) for th in (0.6, 0.65, 0.69) for a in (0.3, 0.5, 0.7, 0.8)]),
        "missing_address_threshold": (lambda a, b: ((d.t_noaddr == 0) & (d.p >= a)) | ((d.t_noaddr == 1) & (d.p >= b)),
                                      [(a, b) for a in np.arange(0.6, 0.8, 0.02) for b in np.arange(0.4, 0.9, 0.04)]),
        "per_country_threshold_diagnostic": (
            lambda a, b: ((d.country == "US") & (d.p >= a)) | ((d.country == "India") & (d.p >= b)),
            pairs(0.6, 0.8, 0.02)),
    }
    for name, (rule, params) in experiments.items():
        best = max(params, key=lambda p: score(rule(*p)))
        print(f"{name}: {score(rule(*best)):.5f} at {tuple(round(float(x), 2) for x in best)}")

    # calibration: fit on even s1 ids, pick threshold there, score on odd ids
    even = d.s1 % 2 == 0
    d["pc"] = IsotonicRegression(out_of_bounds="clip").fit(d.p[even], d.y[even]).predict(d.p)
    for col in ("p", "pc"):
        th = max(grid, key=lambda t: macro_f05(nt.n_match[nt.index % 2 == 0], d.loc[even & (d[col] >= t), ["s1", "y"]])[0]["macro_f0_5"])
        f = macro_f05(nt.n_match[nt.index % 2 == 1], d.loc[~even & (d[col] >= th), ["s1", "y"]])[0]["macro_f0_5"]
        print(f"{'isotonic' if col == 'pc' else 'raw'}_probability_half_split: {f:.5f} (threshold {th})")


if __name__ == "__main__":
    main()
