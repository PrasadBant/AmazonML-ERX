"""Validation error analysis for a trained model: error categories and oracle F0.5 gains.

    python scripts/analysis/error_analysis.py --model v3_p2_big --threshold 0.69

Oracle gain = macro F0.5 increase if every error of a class were fixed and nothing else changed.
FN_blocking: true pair never became a candidate. FN_model: candidate scored below threshold.
"""
import argparse
import json

import numpy as np
import pandas as pd

import _path  # noqa: F401
from business_entity_resolution.config import WORK_DIR
from business_entity_resolution.db import connect
from business_entity_resolution.evaluation.metrics import macro_f05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v3_p2_big")
    ap.add_argument("--threshold", type=float, default=0.69)
    ap.add_argument("--feature-tag", default="v3")
    args = ap.parse_args()
    th = args.threshold
    con = connect(read_only=True)
    feats = pd.read_parquet(WORK_DIR / f"feat_{args.feature_tag}_q_va.parquet",
                            columns=["s1", "t", "n_tsort", "a_tset", "num_first_eq", "t_noaddr", "t_indic"])
    d = feats.merge(pd.read_parquet(WORK_DIR / f"pred_va_{args.model}.parquet"), on=["s1", "t"])
    n_true = con.execute("SELECT s1, n_match FROM ntrue_q_va").df().set_index("s1").n_match
    truth = con.execute("""SELECT f.rid s1, r.rid t FROM ntrue_q_va n JOIN fold f ON f.rid = n.s1
        JOIN gt_pairs g ON g.s1 = f.s1 JOIN rec r ON r.split='train' AND r.entity_id = g.tid""").df()
    owned = set(con.execute("SELECT r.rid FROM gt_pairs g JOIN rec r ON r.split='train' AND r.entity_id = g.tid").df().rid)
    singletons = set(n_true[n_true == 0].index)

    fn = truth.merge(d[["s1", "t", "p"]], on=["s1", "t"], how="left")
    fn = fn[fn.p.isna() | (fn.p < th)]
    fn_blocking = fn[fn.p.isna()]
    fn_model = fn[fn.p.notna()].merge(d, on=["s1", "t"], suffixes=("", "_d"))
    fp = d[(d.p >= th) & (d.y == 0)]

    fp_cat = np.select([fp.s1.isin(singletons), (fp.n_tsort >= 90) & (fp.a_tset < 70),
                        (fp.a_tset >= 90) & (fp.n_tsort < 60), (fp.num_first_eq == 0) & (fp.n_tsort >= 80),
                        fp.t_noaddr == 1],
                       ["FP_on_true_singleton", "FP_same_name_diff_address", "FP_same_address_diff_name",
                        "FP_similar_name_diff_house_number", "FP_target_no_address"], "FP_other")
    fn_cat = np.select([fn_model.t_noaddr == 1, fn_model.t_indic == 1,
                        (fn_model.n_tsort < 50) & (fn_model.a_tset >= 80),
                        (fn_model.a_tset < 60) & (fn_model.n_tsort >= 80), fn_model.p >= th - 0.2],
                       ["FN_model_target_no_address", "FN_model_transliterated", "FN_model_name_changed_same_address",
                        "FN_model_address_variation", "FN_model_near_threshold"], "FN_model_other")
    categories = pd.concat([pd.Series(fp_cat), pd.Series(fn_cat)]).value_counts().to_dict()

    base_pred = d[d.p >= th]
    f = lambda pred: macro_f05(n_true, pred[["s1", "y"]])[0]["macro_f0_5"]  # noqa: E731
    base = f(base_pred)
    missing_pairs = fn_blocking[["s1"]].assign(y=1)
    report = {
        "macro_f0_5": base, "n_true_pairs": len(truth), "n_FP": len(fp), "n_FN_model": len(fn_model),
        "n_FN_blocking": len(fn_blocking), "FP_target_owned_by_other_S1": int(fp.t.isin(owned).sum()),
        "categories": categories,
        "oracle_gain": {
            "remove_all_FP": f(base_pred[base_pred.y == 1]) - base,
            "remove_FP_owned_elsewhere": f(base_pred[~((base_pred.y == 0) & base_pred.t.isin(owned))]) - base,
            "add_FN_model": f(pd.concat([base_pred, d[(d.p < th) & (d.y == 1)]])) - base,
            "add_FN_blocking": f(pd.concat([base_pred[["s1", "y"]], missing_pairs])) - base,
        },
    }
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
