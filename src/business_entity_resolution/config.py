"""Filesystem locations and run configuration.

Machine-specific paths come from environment variables or an untracked `configs/local.json`
({"data_dir": ..., "work_dir": ..., "official_validator": ...}); everything else lives in the
versioned run configs (`configs/v5.json`, ...).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
EXPERIMENTS_DIR = ROOT / "experiments"
SUBMISSIONS_DIR = ROOT / "submissions"


def _local_settings() -> dict:
    path = CONFIG_DIR / "local.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


LOCAL = _local_settings()
# folder that contains train/ and test/ from the official student_resource/dataset
DATA_DIR = Path(os.environ.get("ER_DATA") or LOCAL.get("data_dir") or ROOT / "data" / "dataset")
# working DuckDB database, cached features, models and scored candidate chunks
WORK_DIR = Path(os.environ.get("ER_WORK") or LOCAL.get("work_dir") or ROOT / "data_work")
DB_PATH = WORK_DIR / "er.duckdb"
OFFICIAL_VALIDATOR = LOCAL.get("official_validator")


def source_path(split: str, source: int) -> str:
    return (DATA_DIR / split / f"{split}_source{source}.tsv").as_posix()


def ground_truth_path() -> str:
    return (DATA_DIR / "train" / "train_ground_truth.tsv").as_posix()


def load_run_config(name_or_path: str) -> dict:
    """Load `configs/<name>.json` (or an explicit path)."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.json"
    return json.loads(path.read_text(encoding="utf-8"))
