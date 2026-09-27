"""Candidate generation: IDF-weighted pair-key overlap, within country, df-capped, multi-channel top-K."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockingParams:
    df_cap: int = 1000          # ignore keys carried by more than this many S2/S3 records
    top_k: int = 100            # best candidates by combined key score
    top_k_name: int = 20        # plus best by name-key score (kt 0/1)
    top_k_address: int = 20     # plus best by address-key score (kt 2/3)

    @classmethod
    def from_config(cls, cfg: dict) -> "BlockingParams":
        return cls(**{k: cfg[k] for k in ("df_cap", "top_k", "top_k_name", "top_k_address") if k in cfg})


def generate_candidates(con, query_table: str, split: str, params: BlockingParams, out: str = "cand") -> None:
    """query_table has a column rid (Source-1 ids). Creates temp table `out`
    (s1, t, bscore, nkeys, sn, sa, brank, rn, ra) - the final candidate set scored by the model."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE {out} AS
        WITH qk AS (SELECT p.rid, p.country, p.kt, p.h FROM pk p JOIN {query_table} q USING (rid)
                    WHERE p.split='{split}'),
        w AS (SELECT qk.rid, qk.country, qk.kt, qk.h, ln(n.n / d.df) w FROM qk
              JOIN pdf d ON d.split='{split}' AND d.country=qk.country AND d.h=qk.h
              JOIN ntg n ON n.split='{split}' AND n.country=qk.country
              WHERE d.df <= {params.df_cap}),
        m AS (SELECT w.rid s1, t.rid t, sum(w.w) bscore, count(*)::SMALLINT nkeys,
                     coalesce(sum(w.w) FILTER (w.kt IN (0, 1)), 0) sn,
                     coalesce(sum(w.w) FILTER (w.kt IN (2, 3)), 0) sa
              FROM w JOIN pk t ON t.split='{split}' AND t.src > 1 AND t.country=w.country AND t.h=w.h
              GROUP BY ALL),
        -- Scores are float sums aggregated in parallel, so mathematically equal scores (same key set)
        -- differ by ~1e-15 from run to run; rounding the sort keys lets exact ties fall through to
        -- the deterministic `t` tie-break. (Submissions v3-v6 predate this and are not bit-reproducible.)
        r AS (SELECT *, row_number() OVER (PARTITION BY s1 ORDER BY round(bscore, 6) DESC, t) brank,
                        row_number() OVER (PARTITION BY s1 ORDER BY round(sn, 6) DESC, round(bscore, 6) DESC, t) rn,
                        row_number() OVER (PARTITION BY s1 ORDER BY round(sa, 6) DESC, round(bscore, 6) DESC, t) ra
              FROM m)
        SELECT * FROM r WHERE brank <= {params.top_k}
                           OR (rn <= {params.top_k_name} AND sn > 0)
                           OR (ra <= {params.top_k_address} AND sa > 0)""")


def evaluate_candidates(con, query_table: str, cand: str = "cand") -> dict:
    """Pair recall against train labels plus candidate-count statistics for the query S1 set."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _truth AS
        SELECT f.rid s1, r.rid t FROM {query_table} q JOIN fold f USING (rid)
        JOIN gt_pairs g ON g.s1 = f.s1 JOIN rec r ON r.split='train' AND r.entity_id = g.tid""")
    n_pairs, hits = con.execute(f"""SELECT count(*), count(c.t) FROM _truth x
        LEFT JOIN {cand} c ON c.s1=x.s1 AND c.t=x.t""").fetchone()
    n_query = con.execute(f"SELECT count(*) FROM {query_table}").fetchone()[0]
    avg, med, p95, mx, total = con.execute(f"""
        WITH c AS (SELECT q.rid, count(c.t) n FROM {query_table} q LEFT JOIN {cand} c ON c.s1=q.rid GROUP BY 1)
        SELECT avg(n), median(n), quantile_cont(n, 0.95), max(n), sum(n) FROM c""").fetchone()
    n_targets = con.execute("SELECT sum(n) FROM ntg WHERE split='train'").fetchone()[0]
    return dict(candidate_recall=hits / n_pairs, true_pairs=n_pairs, n_s1=n_query, avg_cands=avg,
                median_cands=med, p95_cands=p95, max_cands=mx, total_cands=total,
                reduction_ratio=1 - total / (n_query * n_targets))
