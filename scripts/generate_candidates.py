"""Build the blocking tables (preprocessing + token/pair keys) and report candidate quality.

    python scripts/generate_candidates.py --config v5                 # build DB stages, then evaluate
    python scripts/generate_candidates.py --config v5 --evaluate-only --n 20000

The evaluation blocks a sample of validation-fold S1 entities against the full train S2/S3 pool
and reports candidate recall, candidate counts and reduction ratio.
"""
import argparse
import json

import _bootstrap  # noqa: F401
from business_entity_resolution.blocking.candidates import BlockingParams, evaluate_candidates, generate_candidates
from business_entity_resolution.config import load_run_config
from business_entity_resolution.data.folds import VALIDATION_FOLD
from business_entity_resolution.db import connect
from business_entity_resolution.pipeline.stages import DATABASE_STAGES, run_database_stage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="v5")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--n", type=int, default=20000, help="validation S1 entities to evaluate")
    args = ap.parse_args()
    cfg = load_run_config(args.config)
    con = connect(read_only=args.evaluate_only)
    if not args.evaluate_only:
        for stage in DATABASE_STAGES:
            run_database_stage(con, stage, cfg["blocking"])
    con.execute(f"""CREATE TEMP TABLE eval_q AS SELECT rid FROM fold WHERE f = {VALIDATION_FOLD}
                    ORDER BY md5(s1) LIMIT {args.n}""")
    generate_candidates(con, "eval_q", "train", BlockingParams.from_config(cfg["blocking"]))
    print(json.dumps(evaluate_candidates(con, "eval_q"), indent=1, default=float))


if __name__ == "__main__":
    main()
