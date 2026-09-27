"""Write the two official TSVs from scored candidate chunks (streamed chunk by chunk).

candidate_pairs.tsv  = every candidate the model scored, i.e. the final candidate set.
matching_results.tsv = candidates that pass the decision rule:
    p >= threshold, and with one_owner each S2/S3 record goes only to its highest-scoring S1
    (the training labels assign every record to at most one S1). With margin > 0 a contested record
    (a second S1 also >= threshold) is dropped entirely unless the winner leads by >= margin.
Every Source-1 entity gets exactly one row in both files (empty list = no match / no candidates),
and matches are a subset of candidates by construction.
"""
from __future__ import annotations

import glob
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

from ..ids import ENTITY_FROM_RID_SQL, rid_to_entity

MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids\n"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids\n"


@dataclass(frozen=True)
class DecisionRule:
    threshold: float
    one_owner: bool = False
    margin: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "DecisionRule":
        return cls(cfg["threshold"], cfg.get("one_owner", False), cfg.get("margin", 0.0))


def owner_table_sql(parquet_list: str, rule: DecisionRule) -> str:
    return f"""CREATE TABLE owner AS
        SELECT t, s1 FROM (
            SELECT t, s1, p, lead(p) OVER (PARTITION BY t ORDER BY p DESC, s1) p2,
                   row_number() OVER (PARTITION BY t ORDER BY p DESC, s1) r
            FROM read_parquet({parquet_list}) WHERE p >= {rule.threshold})
        WHERE r = 1 AND (p2 IS NULL OR p - p2 >= {rule.margin})"""


def write_submission(score_files: list[str], all_s1_rids: list[int], out_dir: Path, rule: DecisionRule,
                     overwrite: bool = False) -> dict:
    out_dir = Path(out_dir)
    targets = [out_dir / "candidate_pairs.tsv", out_dir / "matching_results.tsv"]
    if not overwrite and any(p.exists() for p in targets):
        raise FileExistsError(f"{out_dir} already has submission files; refusing to overwrite (use overwrite=True)")
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=4")
    files = [str(f).replace("\\", "/") for f in score_files]
    if rule.one_owner:
        con.execute(owner_table_sql("[" + ",".join(f"'{f}'" for f in files) + "]", rule))
    keep = f"p >= {rule.threshold}" + (" AND o.s1 = x.s1" if rule.one_owner else "")
    join = "LEFT JOIN owner o ON o.t = x.t" if rule.one_owner else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen, n_candidates, n_matches = set(), 0, 0
    with open(targets[0], "w", encoding="utf-8", newline="\n") as fc, \
            open(targets[1], "w", encoding="utf-8", newline="\n") as fm:
        fc.write(CANDIDATE_HEADER)
        fm.write(MATCHING_HEADER)
        for f in files:
            rows = con.execute(f"""
                SELECT x.s1, {ENTITY_FROM_RID_SQL.format(col='x.s1')} e1,
                       string_agg({ENTITY_FROM_RID_SQL.format(col='x.t')}, ',' ORDER BY p DESC) cands,
                       coalesce(string_agg({ENTITY_FROM_RID_SQL.format(col='x.t')}, ',' ORDER BY p DESC)
                                FILTER ({keep}), '') matches,
                       count(*) nc, count(*) FILTER ({keep}) nm
                FROM read_parquet('{f}') x {join} GROUP BY x.s1""").fetchall()
            for s1, e1, cands, matches, nc, nm in rows:
                seen.add(s1)
                fc.write(f"{e1}\t{cands}\n")
                fm.write(f"{e1}\t{matches}\n")
                n_candidates += nc
                n_matches += nm
        without_candidates = [r for r in all_s1_rids if r not in seen]
        for s1 in without_candidates:
            fc.write(f"{rid_to_entity(s1)}\t\n")
            fm.write(f"{rid_to_entity(s1)}\t\n")
    return dict(n_s1=len(all_s1_rids), s1_without_candidates=len(without_candidates),
                n_candidates=n_candidates, n_matches=n_matches, secs=round(time.time() - t0, 1))


def score_files_in(directory: Path) -> list[str]:
    return sorted(glob.glob(str(Path(directory) / "chunk_*.parquet")))
