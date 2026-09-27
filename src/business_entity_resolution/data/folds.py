"""Deterministic entity-level folds for train Source-1 entities.

fold = md5(entity_id) mod 10. Labels are split by S1 entity; the S2/S3 pool stays shared, as on test.
Folds 0-7 train the matcher, fold 9 is validation, fold 8 is an untouched spare.
"""
from __future__ import annotations

TRAIN_FOLDS_MAX = 7
VALIDATION_FOLD = 9


def create_folds(con) -> None:
    con.execute("""CREATE OR REPLACE TABLE fold AS
        SELECT r.rid, r.entity_id s1, r.country, g.n_match,
               (('0x' || substr(md5(r.entity_id), 1, 8))::UBIGINT % 10)::TINYINT AS f
        FROM rec r JOIN gt g ON g.s1 = r.entity_id WHERE r.split='train' AND r.src=1""")


def create_query_sets(con, n_train: int, n_validation: int, train_offset: int = 0) -> None:
    """q_tr: a slice of training-fold S1 (disjoint slices via offset); q_va: validation S1 sample."""
    con.execute(f"""CREATE OR REPLACE TABLE q_tr AS SELECT rid FROM fold WHERE f <= {TRAIN_FOLDS_MAX}
                    ORDER BY md5(s1 || 'tr') LIMIT {n_train} OFFSET {train_offset}""")
    con.execute(f"""CREATE OR REPLACE TABLE q_va AS SELECT rid FROM fold WHERE f = {VALIDATION_FOLD}
                    ORDER BY md5(s1 || 'va') LIMIT {n_validation}""")
