"""Pairwise features for a candidate table (see blocking.candidates).

Feature names are part of the saved models' contract - do not rename without retraining.
Groups: name similarity, address similarity, house-number evidence, IDF-weighted token overlap,
name ambiguity counts, blocking scores/ranks, target flags, and candidate-group context
(how a candidate compares with the best candidate of the same S1).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

from ..preprocessing.transliteration import INDIC_SQL

_NUMBER = re.compile(r"\d+")
_BLOCKING_COLUMNS = ("bscore", "nkeys", "brank", "sn", "sa", "rn", "ra")
_PASSTHROUGH_COLUMNS = _BLOCKING_COLUMNS + (
    "src", "sh_n", "sh_a", "sh_ncnt", "t_noaddr", "t_indic", "t_caps",
    "s_cp_s1cnt", "t_cp_s1cnt", "s_cp_tgcnt", "t_cp_tgcnt")
_GROUP_COLUMNS = ("n_tsort", "a_tset", "bscore", "c_ratio")


def fetch_pair_data(con, split: str, cand: str = "cand") -> pd.DataFrame:
    """Pair text plus SQL-side features (IDF token overlap, name ambiguity counts)."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _rids AS
        SELECT DISTINCT s1 rid FROM {cand} UNION SELECT DISTINCT t FROM {cand}""")
    # token weights w = ln(N / df) for the name (kind 0) and address (kind 2) tokens of involved records
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _tk AS
        SELECT q.rid, q.kind, q.h, ln(n.n / coalesce(d.df, 1)) w
        FROM tok q JOIN _rids USING (rid)
        LEFT JOIN df d ON d.split=q.split AND d.country=q.country AND d.kind=q.kind AND d.h=q.h
        JOIN ntg n ON n.split=q.split AND n.country=q.country
        WHERE q.split='{split}' AND q.kind IN (0, 2)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE _tot AS
        SELECT rid, sum(w) FILTER (kind=0) nw, sum(w) FILTER (kind=2) aw,
               max(w) FILTER (kind=0) nmax FROM _tk GROUP BY rid""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _sh AS
        SELECT c.s1, c.t, sum(a.w) FILTER (a.kind=0) sh_n, sum(a.w) FILTER (a.kind=2) sh_a,
               max(a.w) FILTER (a.kind=0) sh_nmax, count(*) FILTER (a.kind=0) sh_ncnt
        FROM {cand} c JOIN _tk a ON a.rid=c.s1 JOIN _tk b ON b.rid=c.t AND b.kind=a.kind AND b.h=a.h
        GROUP BY ALL""")
    return con.execute(f"""
        SELECT c.s1, c.t, c.bscore, c.nkeys, c.brank, c.sn, c.sa, c.rn, c.ra, s.country, t.src,
               s.nn s_nn, t.nn t_nn, s.an s_an, t.an t_an, s.compact s_cp, t.compact t_cp,
               array_to_string(s.core, ' ') s_core, array_to_string(t.core, ' ') t_core,
               t.addr IS NULL t_noaddr, regexp_matches(t.name, {INDIC_SQL}) t_indic,
               t.name = upper(t.name) t_caps,
               coalesce(x.sh_n, 0) sh_n, coalesce(x.sh_a, 0) sh_a, coalesce(x.sh_nmax, 0) sh_nmax,
               coalesce(x.sh_ncnt, 0) sh_ncnt,
               coalesce(ts.nw, 0) s_nw, coalesce(ts.aw, 0) s_aw, coalesce(tt.nw, 0) t_nw,
               coalesce(tt.aw, 0) t_aw, coalesce(ts.nmax, 0) s_nmax,
               coalesce(a1.cnt, 0) s_cp_s1cnt, coalesce(a2.cnt, 0) t_cp_s1cnt,
               coalesce(b1.cnt, 0) s_cp_tgcnt, coalesce(b2.cnt, 0) t_cp_tgcnt
        FROM {cand} c
        JOIN rec s ON s.split='{split}' AND s.rid=c.s1
        JOIN rec t ON t.split='{split}' AND t.rid=c.t
        LEFT JOIN _sh x ON x.s1=c.s1 AND x.t=c.t
        LEFT JOIN _tot ts ON ts.rid=c.s1 LEFT JOIN _tot tt ON tt.rid=c.t
        LEFT JOIN s1cp a1 ON a1.split='{split}' AND a1.country=s.country AND a1.compact=s.compact
        LEFT JOIN s1cp a2 ON a2.split='{split}' AND a2.country=s.country AND a2.compact=t.compact
        LEFT JOIN tgcp b1 ON b1.split='{split}' AND b1.country=s.country AND b1.compact=s.compact
        LEFT JOIN tgcp b2 ON b2.split='{split}' AND b2.country=s.country AND b2.compact=t.compact""").df()


def _similarity(a: list, b: list, scorer) -> np.ndarray:
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def string_features(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    sn, tn, sc, tc, sa, ta, scp, tcp = (d[c].fillna("").tolist() for c in
                                        ("s_nn", "t_nn", "s_core", "t_core", "s_an", "t_an", "s_cp", "t_cp"))
    f["n_ratio"] = _similarity(sn, tn, fuzz.ratio)
    f["n_tsort"] = _similarity(sn, tn, fuzz.token_sort_ratio)
    f["n_tset"] = _similarity(sn, tn, fuzz.token_set_ratio)
    f["n_partial"] = _similarity(sn, tn, fuzz.partial_ratio)
    f["c_ratio"] = _similarity(sc, tc, fuzz.ratio)
    f["c_tsort"] = _similarity(sc, tc, fuzz.token_sort_ratio)
    f["c_jw"] = _similarity(sc, tc, distance.JaroWinkler.normalized_similarity)
    f["cp_ratio"] = _similarity(scp, tcp, fuzz.ratio)
    f["cp_partial"] = _similarity(scp, tcp, fuzz.partial_ratio)
    f["a_ratio"] = _similarity(sa, ta, fuzz.ratio)
    f["a_tsort"] = _similarity(sa, ta, fuzz.token_sort_ratio)
    f["a_tset"] = _similarity(sa, ta, fuzz.token_set_ratio)
    f["a_partial"] = _similarity(sa, ta, fuzz.partial_ratio)
    f["n_exact"] = (d["s_nn"] == d["t_nn"]).astype(np.int8)
    f["cp_exact"] = (d["s_cp"] == d["t_cp"]).astype(np.int8)
    f["n_lenratio"] = (d["t_nn"].str.len() / d["s_nn"].str.len().clip(lower=1)).astype(np.float32)
    f["a_lenratio"] = (d["t_an"].fillna("").str.len() / d["s_an"].str.len().clip(lower=1)).astype(np.float32)
    return f.join(number_features(sa, ta))


def number_features(s_addresses: list, t_addresses: list) -> pd.DataFrame:
    """House/unit number evidence from the normalized addresses (leading zeros already stripped)."""
    s_nums = [set(_NUMBER.findall(a or "")) for a in s_addresses]
    t_nums = [set(_NUMBER.findall(a or "")) for a in t_addresses]
    s_first = [_NUMBER.search(a or "") for a in s_addresses]
    t_first = [_NUMBER.search(a or "") for a in t_addresses]
    f = pd.DataFrame({
        "num_s": np.array([len(x) for x in s_nums], np.int16),
        "num_t": np.array([len(x) for x in t_nums], np.int16),
        "num_shared": np.array([len(a & b) for a, b in zip(s_nums, t_nums)], np.int16),
        "num_jacc": np.array([len(a & b) / len(a | b) if (a or b) else -1 for a, b in zip(s_nums, t_nums)],
                             np.float32),
        # log |first number difference|: true matches are perturbed by small amounts (177 vs 175)
        "num_first_absdiff": np.array([np.log1p(abs(int(a.group()) - int(b.group()))) if a and b else -1
                                       for a, b in zip(s_first, t_first)], np.float32),
        "num_first_eq": np.array([-1 if (a is None or b is None) else int(a.group() == b.group())
                                  for a, b in zip(s_first, t_first)], np.int8),
    })
    return f


def build_features(con, split: str, cand: str = "cand", label: bool = True) -> pd.DataFrame:
    d = fetch_pair_data(con, split, cand)
    f = string_features(d)
    f.index = d.index
    f["sh_n_frac_s"] = (d["sh_n"] / d["s_nw"].clip(lower=1e-6)).astype(np.float32)
    f["sh_n_frac_t"] = (d["sh_n"] / d["t_nw"].clip(lower=1e-6)).astype(np.float32)
    f["sh_a_frac_s"] = (d["sh_a"] / d["s_aw"].clip(lower=1e-6)).astype(np.float32)
    f["sh_a_frac_t"] = (d["sh_a"] / d["t_aw"].clip(lower=1e-6)).astype(np.float32)
    f["sh_nmax_frac"] = (d["sh_nmax"] / d["s_nmax"].clip(lower=1e-6)).astype(np.float32)
    for c in _PASSTHROUGH_COLUMNS:
        f[c] = d[c].astype(np.float32)
    grp = pd.concat([d[["s1"]], f[list(_GROUP_COLUMNS)]], axis=1).groupby("s1")
    for c in _GROUP_COLUMNS:
        f[f"{c}_gmax"] = grp[c].transform("max").astype(np.float32)
        f[f"{c}_gap"] = (f[c] - f[f"{c}_gmax"]).astype(np.float32)
    f["g_ncand"] = grp["bscore"].transform("size").astype(np.float32)
    f["g_n_strong"] = grp["n_tsort"].transform(lambda x: (x >= 85).sum()).astype(np.float32)
    out = pd.concat([d[["s1", "t"]], f], axis=1)
    if label:
        out = attach_labels(con, out, cand)
    return out


def attach_labels(con, pairs: pd.DataFrame, cand: str = "cand") -> pd.DataFrame:
    """y = 1 when (s1, t) is a ground-truth pair (train split only)."""
    positives = con.execute(f"""SELECT c.s1, c.t FROM {cand} c JOIN fold fo ON fo.rid=c.s1
        JOIN gt_pairs g ON g.s1=fo.s1 JOIN rec r ON r.split='train' AND r.rid=c.t AND r.entity_id=g.tid""").df()
    positives["y"] = 1
    out = pairs.merge(positives, on=["s1", "t"], how="left")
    out["y"] = out["y"].fillna(0).astype(np.int8)
    return out
