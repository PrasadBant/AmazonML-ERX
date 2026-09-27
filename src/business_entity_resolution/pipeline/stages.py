"""Database build stages, in dependency order.

load -> folds -> transliterate (needs training folds) -> addresses -> tokens -> keys
The resulting tables (rec, fold, tok, df, pk, pdf, ...) are what blocking, features and inference read.
"""
from __future__ import annotations

import time

from ..blocking.keys import build_pair_keys
from ..blocking.tokens import add_name_columns, build_token_tables
from ..config import WORK_DIR
from ..data.folds import create_folds
from ..data.load import load_ground_truth, load_records
from ..preprocessing.address import normalize_addresses_in_db
from ..preprocessing.transliteration import apply_transliteration

DATABASE_STAGES = ("load", "folds", "transliterate", "addresses", "tokens", "keys")


def run_database_stage(con, stage: str, blocking_cfg: dict) -> None:
    t0 = time.time()
    if stage == "load":
        load_records(con)
        load_ground_truth(con)
    elif stage == "folds":
        create_folds(con)
    elif stage == "transliterate":
        print("  ", apply_transliteration(con, WORK_DIR / "indic_dict.json"))
    elif stage == "addresses":
        normalize_addresses_in_db(con)
    elif stage == "tokens":
        add_name_columns(con)
        build_token_tables(con)
    elif stage == "keys":
        build_pair_keys(con, blocking_cfg.get("rarest_name_tokens", 3), blocking_cfg.get("rarest_address_tokens", 3))
    else:
        raise ValueError(f"unknown stage {stage!r}; expected one of {DATABASE_STAGES}")
    print(f"stage {stage}: {time.time() - t0:.0f}s", flush=True)
