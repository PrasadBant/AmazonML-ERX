"""Sweep blocking parameters on the 20K-entity dev sample of fold 9 (candidate recall vs candidate count).

    python scripts/analysis/blocking_sweep.py --caps 200,1000 --k 50,100 --channels 0:0,20:20
"""
import argparse
import json
import time

import _path  # noqa: F401
from business_entity_resolution.blocking.candidates import BlockingParams, evaluate_candidates, generate_candidates
from business_entity_resolution.data.folds import VALIDATION_FOLD
from business_entity_resolution.db import connect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="1000")
    ap.add_argument("--k", default="100")
    ap.add_argument("--channels", default="20:20", help="comma list of name_k:address_k")
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    con = connect(read_only=True)
    con.execute(f"CREATE TEMP TABLE dev_q AS SELECT rid FROM fold WHERE f = {VALIDATION_FOLD} ORDER BY md5(s1) LIMIT {args.n}")
    for cap in map(int, args.caps.split(",")):
        for k in map(int, args.k.split(",")):
            for channel in args.channels.split(","):
                kn, ka = map(int, channel.split(":"))
                t0 = time.time()
                generate_candidates(con, "dev_q", "train", BlockingParams(cap, k, kn, ka))
                r = evaluate_candidates(con, "dev_q")
                r.update(df_cap=cap, top_k=k, top_k_name=kn, top_k_address=ka, secs=round(time.time() - t0, 1))
                print(json.dumps({a: round(b, 4) if isinstance(b, float) else b for a, b in r.items()}), flush=True)


if __name__ == "__main__":
    main()
