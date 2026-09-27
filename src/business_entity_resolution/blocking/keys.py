"""Pair blocking keys.

Single tokens are too common (names come from a small vocabulary), so each record is keyed by every
unordered pair of its rarest tokens: up to `rarest_name` core-name tokens and `rarest_address`
address tokens (rarity = df among S2/S3 of the same country). Compact-name and compact-prefix
tokens are added as single keys.

pk(split, rid, src, country, kt, h)   kt: 0 name-name, 1 compact/prefix, 2 name-address, 3 address-address
pdf(split, country, h, df)            number of S2/S3 records carrying key h
"""
from __future__ import annotations

from ..data.load import SOURCES, SPLITS
from .tokens import KIND_ADDRESS, KIND_COMPACT, KIND_NAME, KIND_PREFIX


def build_pair_keys(con, rarest_name: int = 3, rarest_address: int = 3) -> None:
    con.execute("""CREATE OR REPLACE TABLE pk (split VARCHAR, rid BIGINT, src TINYINT,
                   country VARCHAR, kt TINYINT, h UBIGINT)""")
    for split in SPLITS:
        for s in SOURCES:
            con.execute(f"""INSERT INTO pk
            WITH ranked AS (
                SELECT q.rid, q.src, q.country, q.kind, q.h,
                       row_number() OVER (PARTITION BY q.rid, q.kind ORDER BY d.df, q.h) rn
                FROM tok q JOIN df d ON d.split=q.split AND d.country=q.country
                                     AND d.kind=q.kind AND d.h=q.h
                WHERE q.split='{split}' AND q.src={s} AND q.kind IN ({KIND_NAME}, {KIND_ADDRESS})),
            chosen AS (SELECT DISTINCT rid, src, country, kind, h FROM ranked
                       WHERE (kind={KIND_NAME} AND rn<={rarest_name})
                          OR (kind={KIND_ADDRESS} AND rn<={rarest_address}))
            SELECT '{split}', a.rid, a.src, a.country,
                   (CASE WHEN a.kind={KIND_NAME} AND b.kind={KIND_NAME} THEN 0
                         WHEN a.kind={KIND_ADDRESS} AND b.kind={KIND_ADDRESS} THEN 3 ELSE 2 END)::TINYINT,
                   hash(a.h, b.h)
            FROM chosen a JOIN chosen b ON a.rid=b.rid AND a.h < b.h
            UNION ALL
            SELECT split, rid, src, country, 1, h FROM tok
            WHERE split='{split}' AND src={s} AND kind IN ({KIND_COMPACT}, {KIND_PREFIX})""")
    con.execute("CREATE OR REPLACE TABLE pdf AS SELECT split, country, h, count(*)::INT df FROM pk WHERE src > 1 GROUP BY ALL")
