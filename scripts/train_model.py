"""Build training/validation feature sets and train the matcher for a run config.

    python scripts/train_model.py --config v5                  # build sets (if missing) + train
    python scripts/train_model.py --config v5 --rebuild-sets

Training parts are disjoint 60K slices of folds 0-7; validation is 50K entities of fold 9.
The selected threshold (max macro F0.5 on validation) is stored with the model.
"""
import argparse
import json

import _bootstrap  # noqa: F401
from business_entity_resolution.blocking.candidates import BlockingParams
from business_entity_resolution.config import load_run_config
from business_entity_resolution.db import connect
from business_entity_resolution.evaluation.tracking import log_experiment
from business_entity_resolution.matching.model import train_xgboost
from business_entity_resolution.matching.training_sets import (build_training_part, build_validation_set,
                                                               training_part_path, validation_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="v5")
    ap.add_argument("--rebuild-sets", action="store_true")
    args = ap.parse_args()
    cfg = load_run_config(args.config)
    training, model = cfg["training"], cfg["model"]
    tag = training["feature_tag"]
    params = BlockingParams.from_config(cfg["blocking"])
    missing = [o for o in training["part_offsets"] if not training_part_path(tag, o).exists()]
    if args.rebuild_sets or missing or not validation_path(tag).exists():
        con = connect()
        print("validation:", build_validation_set(con, tag, params, training["validation_entities"]))
        for offset in (training["part_offsets"] if args.rebuild_sets else missing):
            print(f"train part @{offset}:", build_training_part(
                con, tag, params, training["entities_per_part"], offset,
                training["validation_entities"], training["easy_negative_rate"]))
        con.close()
    result = train_xgboost(connect(read_only=True), tag, model["name"], training["part_offsets"], model["params"])
    log_experiment(f"train_{model['name']}", cfg, result, stage="matching", model=model["name"],
                   hyperparameters=json.dumps(model["params"]), threshold=result["threshold"],
                   precision=result["macro_precision"], recall=result["macro_recall"],
                   macro_f0_5=result["macro_f0_5"], singleton_accuracy=result["singleton_accuracy"],
                   runtime_s=result["train_secs"])
    print({k: v for k, v in result.items() if k != "coarse_sweep"})


if __name__ == "__main__":
    main()
