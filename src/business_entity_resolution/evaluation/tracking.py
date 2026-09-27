"""Machine-readable experiment records.

experiments/experiment_log.csv (+ configs/<id>.json, metrics/<id>.json): one row per run.
experiments/deep_optimization/leaderboard.csv: internal scoreboard, upserted by experiment id.
"""
from __future__ import annotations

import csv
import datetime as dt
import json

import psutil

from ..config import EXPERIMENTS_DIR

LOG_FIELDS = ["experiment_id", "date", "stage", "blocking_strategy", "features", "model", "hyperparameters",
              "threshold", "candidate_recall", "avg_candidates", "p95_candidates", "max_candidates",
              "reduction_ratio", "precision", "recall", "macro_f0_5", "singleton_accuracy",
              "runtime_s", "peak_rss_gb", "output", "notes"]
BOARD_FIELDS = ["experiment_id", "description", "blocking", "normalization", "features", "model", "threshold",
                "candidate_recall", "avg_candidates", "p95_candidates", "precision", "recall", "macro_f0_5",
                "singleton_accuracy", "runtime", "memory", "status"]


def log_experiment(exp_id: str, config: dict, metrics: dict, **row) -> None:
    for sub in ("configs", "metrics"):
        (EXPERIMENTS_DIR / sub).mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_DIR / "configs" / f"{exp_id}.json").write_text(json.dumps(config, indent=2, default=str))
    (EXPERIMENTS_DIR / "metrics" / f"{exp_id}.json").write_text(json.dumps(metrics, indent=2, default=str))
    row = {"experiment_id": exp_id, "date": dt.datetime.now().isoformat(timespec="seconds"),
           "peak_rss_gb": round(psutil.Process().memory_info().rss / 1e9, 2), **row}
    path = EXPERIMENTS_DIR / "experiment_log.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, LOG_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})


def upsert_scoreboard(**row) -> None:
    path = EXPERIMENTS_DIR / "deep_optimization" / "leaderboard.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["experiment_id"] != row["experiment_id"]]
    rows.append({c: row.get(c, "") for c in BOARD_FIELDS})
    rows.sort(key=lambda r: -float(r["macro_f0_5"] or 0))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, BOARD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
