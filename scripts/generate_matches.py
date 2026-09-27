"""Score the test set with a config's model (if needed) and write matching_results.tsv + candidate_pairs.tsv.

    python scripts/generate_matches.py --config v5 --output-dir submissions/v7/output

Reuses existing test scores (`inference.scores_dir`) when present; otherwise rescores cached test
features (`inference.rescored_from_features`) or runs full test inference. Existing submission folders
are never overwritten unless --overwrite is given.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from business_entity_resolution.blocking.candidates import BlockingParams
from business_entity_resolution.config import WORK_DIR, load_run_config
from business_entity_resolution.db import connect
from business_entity_resolution.matching.inference import rescore_cached_features, score_split
from business_entity_resolution.pipeline.submission import DecisionRule, score_files_in, write_submission

TEST_CHUNKS = 35  # 1,732,544 test S1 / 50,000 per chunk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_run_config(args.config)
    inference = cfg["inference"]
    scores_dir = WORK_DIR / inference["scores_dir"]
    if len(score_files_in(scores_dir)) < TEST_CHUNKS:
        if inference.get("rescored_from_features") and (WORK_DIR / inference["rescored_from_features"]).exists():
            rescore_cached_features(inference["rescored_from_features"], cfg["model"]["name"], inference["scores_dir"])
        else:
            score_split(connect(read_only=True), "test", cfg["model"]["name"], inference["scores_dir"],
                        BlockingParams.from_config(cfg["blocking"]), inference["chunk_size"], save_features=True)
    rids = connect(read_only=True).execute("SELECT rid FROM rec WHERE split='test' AND src=1").df().rid.tolist()
    print(write_submission(score_files_in(scores_dir), rids, Path(args.output_dir),
                           DecisionRule.from_config(cfg["decision"]), args.overwrite))


if __name__ == "__main__":
    main()
