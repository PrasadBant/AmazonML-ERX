"""Dataset forensics on the raw TSVs (streamed with DuckDB): per-file schema, row counts, missingness,
cardinality, countries, id prefixes, text lengths/casing, exact-duplicate rates and train/test id overlap.
Outputs of the original run are in docs/setup/eda_outputs/.

    python scripts/analysis/data_profile.py
"""
import argparse

import duckdb

import _path  # noqa: F401
from business_entity_resolution.config import ground_truth_path, source_path

SOURCE_FILES = [(split, n) for split in ("train", "test") for n in (1, 2, 3)]


def read(path: str) -> str:
    return f"read_csv('{path}', delim='\t', header=true, quote='')"


def profile_source(con, split: str, n: int) -> None:
    p = source_path(split, n)
    stats = con.execute(f"""SELECT count(*),
            count(*) FILTER (WHERE business_address IS NULL OR trim(business_address) = ''),
            count(DISTINCT entity_id), count(DISTINCT business_name), count(DISTINCT country),
            avg(len(business_name)), avg(len(business_address)),
            avg((business_name = upper(business_name))::INT), avg((business_address = upper(business_address))::INT)
        FROM {read(p)}""").fetchone()
    countries = con.execute(f"SELECT country, count(*) c FROM {read(p)} GROUP BY 1 ORDER BY c DESC").fetchall()
    dup = con.execute(f"""WITH t AS (SELECT lower(trim(business_name)) n, lower(trim(coalesce(business_address, ''))) a
                                     FROM {read(p)})
                          SELECT count(*) - count(DISTINCT (n, a)) FROM t""").fetchone()[0]
    print(f"{split}_source{n}: rows={stats[0]} addr_missing={stats[1]} dup_ids={stats[0] - stats[2]} "
          f"uniq_names={stats[3]} countries={countries} name_len={stats[5]:.1f} addr_len={stats[6]:.1f} "
          f"all_caps_name={stats[7]:.3f} all_caps_addr={stats[8]:.3f} exact_dup_rows={dup}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    for split, n in SOURCE_FILES:
        profile_source(con, split, n)
    gt = f"read_csv('{ground_truth_path()}', delim='\t', header=true, quote='', all_varchar=true)"
    print("ground truth match-count distribution:", con.execute(f"""
        SELECT n, count(*) FROM (SELECT CASE WHEN coalesce(trim(matched_entity_ids), '') = '' THEN 0
               ELSE len(string_split(matched_entity_ids, ',')) END n FROM {gt}) GROUP BY 1 ORDER BY 1""").fetchall())
    for n in (1, 2, 3):
        overlap = con.execute(f"""SELECT count(*) FROM {read(source_path('train', n))} a
                                  JOIN {read(source_path('test', n))} b USING (entity_id)""").fetchone()[0]
        print(f"S{n} train/test entity_id overlap: {overlap}")


if __name__ == "__main__":
    main()
