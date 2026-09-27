"""Local, memory-safe submission validation (streams through DuckDB).

Mirrors the official utils/validate_submission.py rules for BOTH files - the official script needs
10+ GB of RAM for a 141M-id candidate file - and adds format checks:
  exact header; exactly one tab per line; one row per test Source-1 id (none missing, none extra,
  no duplicates); ids are S2-/S3- only, non-empty, not repeated within a list, and exist in the test
  S2/S3 files; every matched id is also a candidate of the same Source-1 entity.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

HEADERS = {"matching_results.tsv": ["source1_entity_id", "matched_entity_ids"],
           "candidate_pairs.tsv": ["source1_entity_id", "candidate_entity_ids"]}


def _lines_sql(path: str) -> str:
    # one VARCHAR column per physical line; \x01 never occurs in the files, so nothing is split
    return (f"read_csv('{path}', delim='\x01', header=false, skip=1, quote='', escape='', "
            f"columns={{'line': 'VARCHAR'}}, all_varchar=true)")


def check_submission(output_dir: Path, test_dir: Path, check_ids: bool = True) -> dict[str, dict]:
    output_dir, test_dir = Path(output_dir), Path(test_dir)
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=6")
    q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    s1_path = (test_dir / "test_source1.tsv").as_posix()
    con.execute(f"""CREATE TABLE s1 AS SELECT entity_id FROM read_csv('{s1_path}', delim='\t', header=true,
                    quote='', all_varchar=true)""")
    if check_ids:
        paths = [(test_dir / f"test_source{n}.tsv").as_posix() for n in (2, 3)]
        con.execute(f"""CREATE TABLE targets AS SELECT entity_id FROM read_csv(['{paths[0]}', '{paths[1]}'],
                        delim='\t', header=true, quote='', all_varchar=true)""")
    results: dict[str, dict] = {}

    def record(name: str, bad: int, detail: str = "") -> None:
        results[name] = {"passed": bad == 0, "violations": int(bad), "detail": detail}

    for fname, header in HEADERS.items():
        path = output_dir / fname
        tag = fname.split("_")[0]
        if not path.exists():
            record(f"{tag}:file_exists", 1, str(path))
            continue
        with open(path, encoding="utf-8") as f:
            first = f.readline().rstrip("\n").split("\t")
        record(f"{tag}:header", int(first != header), str(first))
        table = f"{tag}_rows"
        con.execute(f"""CREATE TABLE {table} AS
            SELECT split_part(line, '\t', 1) s1, coalesce(split_part(line, '\t', 2), '') ids,
                   length(line) - length(replace(line, '\t', '')) n_tabs
            FROM {_lines_sql(path.as_posix())}""")
        record(f"{tag}:one_tab_per_line", q(f"SELECT count(*) FROM {table} WHERE n_tabs <> 1"))
        record(f"{tag}:no_duplicate_s1_rows", q(f"SELECT count(*) - count(DISTINCT s1) FROM {table}"))
        record(f"{tag}:no_missing_s1", q(f"SELECT count(*) FROM s1 ANTI JOIN {table} t ON t.s1 = s1.entity_id"))
        record(f"{tag}:no_extra_s1", q(f"SELECT count(*) FROM {table} t ANTI JOIN s1 ON t.s1 = s1.entity_id"))
        con.execute(f"""CREATE TABLE {tag}_ids AS
            SELECT s1, unnest(string_split(ids, ',')) t FROM {table} WHERE ids <> ''""")
        record(f"{tag}:ids_s2_s3_only", q(f"SELECT count(*) FROM {tag}_ids WHERE NOT (t LIKE 'S2-%' OR t LIKE 'S3-%')"))
        record(f"{tag}:no_duplicate_ids_in_list",
               q(f"SELECT count(*) FROM (SELECT s1, t FROM {tag}_ids GROUP BY ALL HAVING count(*) > 1)"))
        if check_ids:
            record(f"{tag}:ids_exist_in_test",
                   q(f"SELECT count(*) FROM {tag}_ids i ANTI JOIN targets g ON g.entity_id = i.t"))
    if "matching:header" in results and "candidate:header" in results:
        record("matches_subset_of_candidates",
               q("SELECT count(*) FROM matching_ids m ANTI JOIN candidate_ids c ON c.s1 = m.s1 AND c.t = m.t"))
        results["info:records_matched_to_several_s1"] = {
            "passed": True, "violations": 0,
            "detail": str(q("SELECT count(*) FROM (SELECT t FROM matching_ids GROUP BY t HAVING count(*) > 1)"))}
        results["info:counts"] = {"passed": True, "violations": 0, "detail": str({
            "rows": q("SELECT count(*) FROM matching_rows"),
            "empty_match_rows": q("SELECT count(*) FROM matching_rows WHERE ids = ''"),
            "matches": q("SELECT count(*) FROM matching_ids"),
            "candidates": q("SELECT count(*) FROM candidate_ids")})}
    return results
