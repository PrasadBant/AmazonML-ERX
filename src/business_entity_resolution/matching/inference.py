"""Chunked scoring of every Source-1 entity of a split: block -> features -> model score.

Each chunk parquet holds ALL candidates the model scored (s1, t, p); together they are exactly the
candidate set written to candidate_pairs.tsv. Finished chunks are skipped, so runs are resumable.
"""
from __future__ import annotations

import glob
import time
from pathlib import Path

import pandas as pd
import psutil

from ..blocking.candidates import BlockingParams, generate_candidates
from ..config import WORK_DIR
from ..features.pairwise import build_features
from .model import load_model, predict


def score_split(con, split: str, model_name: str, out_name: str, params: BlockingParams,
                chunk_size: int = 50_000, save_features: bool = False) -> Path:
    out = WORK_DIR / out_name
    out.mkdir(exist_ok=True)
    feature_dir = WORK_DIR / f"feat_{split}_{out_name}"
    model, meta = load_model(model_name)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE inf_q AS
        SELECT rid, (row_number() OVER (ORDER BY rid) - 1) // {chunk_size} c FROM rec WHERE split='{split}' AND src=1""")
    n_chunks = con.execute("SELECT max(c) + 1 FROM inf_q").fetchone()[0]
    t0 = time.time()
    for c in range(n_chunks):
        target = out / f"chunk_{c:03d}.parquet"
        if target.exists():
            continue
        con.execute(f"CREATE OR REPLACE TEMP TABLE qc AS SELECT rid FROM inf_q WHERE c={c}")
        generate_candidates(con, "qc", split, params)
        feats = build_features(con, split, label=False)
        if save_features:
            feature_dir.mkdir(exist_ok=True)
            feats.to_parquet(feature_dir / target.name, compression="zstd")
        feats["p"] = predict(model, meta, feats)
        feats[["s1", "t", "p"]].to_parquet(target)
        print(f"  chunk {c + 1}/{n_chunks}: {len(feats)} pairs, {time.time() - t0:.0f}s, "
              f"rss {psutil.Process().memory_info().rss / 1e9:.2f} GB", flush=True)
    return out


def rescore_cached_features(feature_dir: str, model_name: str, out_name: str) -> Path:
    """Score previously saved test features with another model (no feature recomputation)."""
    model, meta = load_model(model_name)
    out = WORK_DIR / out_name
    out.mkdir(exist_ok=True)
    for path in sorted(glob.glob(str(WORK_DIR / feature_dir / "chunk_*.parquet"))):
        target = out / Path(path).name
        if target.exists():
            continue
        d = pd.read_parquet(path, columns=["s1", "t"] + meta["features"])
        d["p"] = predict(model, meta, d)
        d[["s1", "t", "p"]].to_parquet(target)
    return out
