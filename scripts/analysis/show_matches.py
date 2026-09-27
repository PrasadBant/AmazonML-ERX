"""Print random train Source-1 entities with their ground-truth S2/S3 records (to eyeball noise patterns).

    python scripts/analysis/show_matches.py --n 12
"""
import argparse

import duckdb

import _path  # noqa: F401
from business_entity_resolution.config import ground_truth_path, source_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    read = lambda p: f"read_csv('{p}', delim='\t', header=true, quote='')"  # noqa: E731
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    rows = con.execute(f"""
        WITH gt AS (SELECT * FROM {read(ground_truth_path())} USING SAMPLE {args.n} ROWS (reservoir, 7)),
        m AS (SELECT source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) mid FROM gt),
        o AS (SELECT * FROM {read(source_path('train', 2))} UNION ALL SELECT * FROM {read(source_path('train', 3))})
        SELECT s.entity_id, s.business_name, s.business_address, o.entity_id, o.business_name, o.business_address
        FROM gt JOIN {read(source_path('train', 1))} s ON s.entity_id = gt.source1_entity_id
        LEFT JOIN m ON m.s1 = s.entity_id LEFT JOIN o ON o.entity_id = m.mid ORDER BY s.entity_id""").fetchall()
    last = None
    for s_id, s_name, s_addr, t_id, t_name, t_addr in rows:
        if s_id != last:
            print(f"\n{s_id} | {s_name} | {s_addr}")
            last = s_id
        if t_id:
            print(f"   {t_id:14s} | {t_name} | {t_addr}")


if __name__ == "__main__":
    main()
