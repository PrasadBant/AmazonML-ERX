"""Check that an existing submission's matches equal what the current code derives from the saved test
scores and a config's decision rule. Read-only: nothing is written.

    python scripts/analysis/check_reproduction.py --config v5 --submission submissions/v5/output
"""
import argparse

import duckdb

import _path  # noqa: F401
from business_entity_resolution.config import WORK_DIR, load_run_config
from business_entity_resolution.ids import ENTITY_FROM_RID_SQL
from business_entity_resolution.pipeline.submission import DecisionRule, owner_table_sql, score_files_in


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--submission", required=True, help="folder containing matching_results.tsv")
    args = ap.parse_args()
    cfg = load_run_config(args.config)
    rule = DecisionRule.from_config(cfg["decision"])
    files = "[" + ",".join(f"'{f}'" for f in (p.replace("\\", "/") for p in
                                                score_files_in(WORK_DIR / cfg["inference"]["scores_dir"]))) + "]"
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=6")
    if rule.one_owner:
        con.execute(owner_table_sql(files, rule))
        derived = f"SELECT x.s1, x.t FROM read_parquet({files}) x JOIN owner o ON o.t = x.t AND o.s1 = x.s1 WHERE x.p >= {rule.threshold}"
    else:
        derived = f"SELECT s1, t FROM read_parquet({files}) WHERE p >= {rule.threshold}"
    con.execute(f"""CREATE TABLE derived AS SELECT {ENTITY_FROM_RID_SQL.format(col='s1')} s1,
                    {ENTITY_FROM_RID_SQL.format(col='t')} t FROM ({derived})""")
    path = f"{args.submission}/matching_results.tsv".replace("\\", "/")
    con.execute(f"""CREATE TABLE submitted AS SELECT s1, unnest(string_split(m, ',')) t
        FROM read_csv('{path}', delim='\t', header=true, quote='', all_varchar=true,
                      columns={{'s1': 'VARCHAR', 'm': 'VARCHAR'}}) WHERE m IS NOT NULL AND m <> ''""")
    n_derived = con.execute("SELECT count(*) FROM derived").fetchone()[0]
    n_submitted = con.execute("SELECT count(*) FROM submitted").fetchone()[0]
    only_derived = con.execute("SELECT count(*) FROM (SELECT * FROM derived EXCEPT SELECT * FROM submitted)").fetchone()[0]
    only_submitted = con.execute("SELECT count(*) FROM (SELECT * FROM submitted EXCEPT SELECT * FROM derived)").fetchone()[0]
    print(f"derived={n_derived} submitted={n_submitted} only_in_derived={only_derived} only_in_submitted={only_submitted}")
    print("IDENTICAL MATCH SETS" if only_derived == only_submitted == 0 else "MATCH SETS DIFFER")


if __name__ == "__main__":
    main()
