"""End-to-end pipeline for one run config: raw TSVs -> working DB -> training sets -> model -> test scores
-> submission files.

    python scripts/run_pipeline.py --config v5 --output-dir submissions/v7/output
    python scripts/run_pipeline.py --config v5 --stages train_sets,train      # a subset of stages

Stages (in order): load folds transliterate addresses tokens keys | train_sets train | infer submit
Full run: ~5 h on a 16 GB laptop (test feature computation dominates). Output folders that already
contain submission files are never overwritten unless --overwrite is given.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from business_entity_resolution.blocking.candidates import BlockingParams
from business_entity_resolution.config import WORK_DIR, load_run_config
from business_entity_resolution.db import connect
from business_entity_resolution.matching.inference import score_split
from business_entity_resolution.matching.model import train_xgboost
from business_entity_resolution.matching.training_sets import build_training_part, build_validation_set
from business_entity_resolution.pipeline.stages import DATABASE_STAGES, run_database_stage
from business_entity_resolution.pipeline.submission import DecisionRule, score_files_in, write_submission

ALL_STAGES = DATABASE_STAGES + ("train_sets", "train", "infer", "submit")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="v5")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--output-dir", help="where to write the two TSVs (required for the submit stage)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_run_config(args.config)
    stages = args.stages.split(",")
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")
    if "submit" in stages and not args.output_dir:
        ap.error("--output-dir is required for the submit stage")
    params = BlockingParams.from_config(cfg["blocking"])
    training, model = cfg["training"], cfg["model"]

    writes = any(s in stages for s in DATABASE_STAGES + ("train_sets",))
    con = connect(read_only=not writes)  # one connection: DuckDB rejects mixed modes in one process
    for stage in (s for s in DATABASE_STAGES if s in stages):
        run_database_stage(con, stage, cfg["blocking"])
    if "train_sets" in stages:
        print("validation:", build_validation_set(con, training["feature_tag"], params, training["validation_entities"]))
        for offset in training["part_offsets"]:
            print(f"train part @{offset}:", build_training_part(
                con, training["feature_tag"], params, training["entities_per_part"], offset,
                training["validation_entities"], training["easy_negative_rate"]))
    if "train" in stages:
        result = train_xgboost(con, training["feature_tag"], model["name"], training["part_offsets"], model["params"])
        print({k: v for k, v in result.items() if k != "coarse_sweep"})
    if "infer" in stages:
        score_split(con, "test", model["name"], cfg["inference"]["scores_dir"], params,
                    cfg["inference"]["chunk_size"], save_features=True)
    if "submit" in stages:
        rids = con.execute("SELECT rid FROM rec WHERE split='test' AND src=1").df().rid.tolist()
        stats = write_submission(score_files_in(WORK_DIR / cfg["inference"]["scores_dir"]), rids,
                                 Path(args.output_dir), DecisionRule.from_config(cfg["decision"]), args.overwrite)
        print(stats)


if __name__ == "__main__":
    main()
