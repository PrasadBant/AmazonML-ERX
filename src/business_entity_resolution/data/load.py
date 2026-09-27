"""Load the official TSVs (read-only) into the working DuckDB database.

Tables created:
  rec(split, src, entity_id, name, addr, country, nn, an, rid)
      nn = normalized name; an = normalized address (filled by preprocessing.address)
  gt(s1, n_match)       one row per train Source-1 entity
  gt_pairs(s1, tid)     exploded train labels
"""
from __future__ import annotations

from ..config import ground_truth_path, source_path
from ..ids import RID_FROM_ENTITY_SQL
from ..preprocessing.names import normalize_name_sql

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def load_records(con) -> None:
    con.execute("DROP TABLE IF EXISTS rec")
    con.execute("""CREATE TABLE rec (split VARCHAR, src TINYINT, entity_id VARCHAR, name VARCHAR,
                   addr VARCHAR, country VARCHAR, nn VARCHAR, an VARCHAR, rid BIGINT)""")
    for split in SPLITS:
        for n in SOURCES:
            con.execute(f"""
                INSERT INTO rec
                SELECT '{split}', {n}, entity_id, business_name, business_address, country,
                       {normalize_name_sql('business_name')}, NULL,
                       {RID_FROM_ENTITY_SQL.format(src=n, col='entity_id')}
                FROM read_csv('{source_path(split, n)}', delim='\t', header=true, quote='',
                              columns={{'entity_id':'VARCHAR','business_name':'VARCHAR',
                                        'business_address':'VARCHAR','country':'VARCHAR'}})""")


def load_ground_truth(con) -> None:
    con.execute("DROP TABLE IF EXISTS gt; DROP TABLE IF EXISTS gt_pairs")
    con.execute(f"""CREATE TABLE gt AS
        SELECT source1_entity_id s1,
               CASE WHEN coalesce(trim(matched_entity_ids), '') = '' THEN 0
                    ELSE len(string_split(matched_entity_ids, ',')) END n_match,
               matched_entity_ids m
        FROM read_csv('{ground_truth_path()}', delim='\t', header=true, quote='', all_varchar=true)""")
    con.execute("CREATE TABLE gt_pairs AS SELECT s1, trim(unnest(string_split(m, ','))) tid FROM gt WHERE n_match > 0")
    con.execute("ALTER TABLE gt DROP COLUMN m")
