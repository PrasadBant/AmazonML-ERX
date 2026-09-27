"""Token tables and document frequencies used by blocking and by the IDF features.

rec gains: core (name tokens minus legal forms) and compact (core tokens concatenated).
tok(split, rid, src, country, kind, h), with kind:
    0 = core-name token, 1 = compact name, 2 = address token, 5 = 8-char compact-name prefix
df(split, country, kind, h, df)      document frequency over the S2+S3 records of a split
ntg(split, country, n)               number of S2+S3 records per split/country (IDF numerator)
s1cp / tgcp(split, country, compact, cnt)   how many S1 / S2+S3 records share a compact name
"""
from __future__ import annotations

from ..data.load import SOURCES, SPLITS
from ..preprocessing.names import create_core_tokens_macro

KIND_NAME, KIND_COMPACT, KIND_ADDRESS, KIND_PREFIX = 0, 1, 2, 5


def add_name_columns(con) -> None:
    create_core_tokens_macro(con)
    cols = {r[0] for r in con.execute("DESCRIBE rec").fetchall()}
    exclude = "EXCLUDE (core, compact)" if "core" in cols else ""
    con.execute(f"""CREATE OR REPLACE TABLE rec AS
        SELECT * {exclude}, core_toks(nn) AS core, array_to_string(core_toks(nn), '') AS compact FROM rec""")


def build_token_tables(con) -> None:
    con.execute("""CREATE OR REPLACE TABLE tok (split VARCHAR, rid BIGINT, src TINYINT,
                   country VARCHAR, kind TINYINT, h UBIGINT)""")
    for split in SPLITS:
        for s in SOURCES:
            w = f"WHERE split='{split}' AND src={s}"
            con.execute(f"""INSERT INTO tok SELECT split, rid, src, country, {KIND_NAME},
                            hash('n:' || unnest(list_distinct(core))) FROM rec {w}""")
            con.execute(f"""INSERT INTO tok SELECT split, rid, src, country, {KIND_COMPACT},
                            hash('c:' || compact) FROM rec {w} AND length(compact) >= 3""")
            con.execute(f"""INSERT INTO tok SELECT split, rid, src, country, {KIND_ADDRESS}, hash('a:' || t) FROM
                (SELECT split, rid, src, country,
                        unnest(list_distinct(list_filter(string_split(an, ' '), x -> x <> ''))) t FROM rec {w})""")
            # domain-style names ("statenislandyouth.com") keep the start of the compact name
            con.execute(f"""INSERT INTO tok SELECT split, rid, src, country, {KIND_PREFIX},
                            hash('p:' || substr(compact, 1, 8)) FROM rec {w} AND length(compact) >= 8""")
    con.execute("""CREATE OR REPLACE TABLE df AS
        SELECT split, country, kind, h, count(*)::INT df FROM tok WHERE src > 1 GROUP BY ALL""")
    con.execute("CREATE OR REPLACE TABLE ntg AS SELECT split, country, count(*) n FROM rec WHERE src > 1 GROUP BY ALL")
    con.execute("""CREATE OR REPLACE TABLE s1cp AS
        SELECT split, country, compact, count(*)::INT cnt FROM rec WHERE src = 1 GROUP BY ALL""")
    con.execute("""CREATE OR REPLACE TABLE tgcp AS
        SELECT split, country, compact, count(*)::INT cnt FROM rec WHERE src > 1 GROUP BY ALL""")
