"""DuckDB connection to the working database (all heavy data stays on disk)."""
from __future__ import annotations

import os

import duckdb

from .config import DB_PATH, WORK_DIR

MEMORY_LIMIT = os.environ.get("ER_MEM", "3GB")
THREADS = int(os.environ.get("ER_THREADS", "8"))


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"SET threads={THREADS}")
    # One spill directory per process: concurrent DuckDB processes sharing a spill directory
    # corrupt each other's temp files (observed as impossible values in query results).
    con.execute(f"SET temp_directory='{(WORK_DIR / 'tmp' / str(os.getpid())).as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    return con
