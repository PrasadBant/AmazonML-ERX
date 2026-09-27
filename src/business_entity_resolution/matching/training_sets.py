"""Blocked candidates + features for the matcher's training parts (folds 0-7) and validation (fold 9)."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..blocking.candidates import BlockingParams, evaluate_candidates, generate_candidates
from ..config import WORK_DIR
from ..data.folds import create_query_sets
from ..features.pairwise import build_features


def training_part_path(tag: str, offset: int) -> Path:
    return WORK_DIR / f"feat_{tag}_q_tr{'_o' + str(offset) if offset else ''}.parquet"


def validation_path(tag: str) -> Path:
    return WORK_DIR / f"feat_{tag}_q_va.parquet"


def sample_hard_negatives(feats: pd.DataFrame, easy_rate: float) -> pd.DataFrame:
    """Keep every positive, every hard negative (similar name or address, or top-10 by blocking score),
    and a deterministic `easy_rate` fraction of the remaining easy negatives."""
    hard = (feats.c_tsort >= 60) | (feats.a_tset >= 60) | (feats.brank <= 10)
    # multiplicative hash of the pair ids (int64 wrap-around is fine: it only needs to be deterministic)
    easy_kept = (feats.s1 * 2654435761 + feats.t) % 100 < int(easy_rate * 100)
    return feats[(feats.y == 1) | hard | easy_kept].reset_index(drop=True)


def _build(con, query_table: str, params: BlockingParams) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    generate_candidates(con, query_table, "train", params)
    report = evaluate_candidates(con, query_table)
    feats = build_features(con, "train")
    report["secs"] = round(time.time() - t0, 1)
    return feats, report


def build_validation_set(con, tag: str, params: BlockingParams, n_validation: int) -> dict:
    create_query_sets(con, n_train=0, n_validation=n_validation)
    feats, report = _build(con, "q_va", params)
    feats.to_parquet(validation_path(tag))
    con.execute("""CREATE OR REPLACE TABLE ntrue_q_va AS
                   SELECT f.rid s1, f.n_match, f.country FROM fold f JOIN q_va USING (rid)""")
    return report


def build_training_part(con, tag: str, params: BlockingParams, n_train: int, offset: int,
                        n_validation: int, easy_negative_rate: float | None) -> dict:
    create_query_sets(con, n_train=n_train, n_validation=n_validation, train_offset=offset)
    feats, report = _build(con, "q_tr", params)
    if easy_negative_rate is not None:
        feats = sample_hard_negatives(feats, easy_negative_rate)
    report["rows_written"] = len(feats)
    feats.to_parquet(training_part_path(tag, offset))
    return report
