"""Validate a submission folder (matching_results.tsv + candidate_pairs.tsv) before uploading.

    python scripts/validate_submission.py submissions/v5/output
    python scripts/validate_submission.py submissions/v5/output --official

Runs the local streaming checks on both files (see evaluation/submission_checks.py). With --official
it also runs the challenge's utils/validate_submission.py on matching_results.tsv with --check-ids
(path from configs/local.json "official_validator"). The official script is not run on
candidate_pairs.tsv by default because it needs 10+ GB of RAM for a ~141M-id file.
Exit code 0 = every check passed.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from business_entity_resolution.config import DATA_DIR, OFFICIAL_VALIDATOR
from business_entity_resolution.evaluation.submission_checks import check_submission


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir")
    ap.add_argument("--test-dir", default=str(DATA_DIR / "test"))
    ap.add_argument("--no-id-check", action="store_true")
    ap.add_argument("--official", action="store_true")
    args = ap.parse_args()
    results = check_submission(Path(args.output_dir), Path(args.test_dir), check_ids=not args.no_id_check)
    ok = True
    for name, r in results.items():
        status = "INFO" if name.startswith("info:") else ("PASS" if r["passed"] else "FAIL")
        ok &= r["passed"]
        print(f"{status:4s} {name}" + (f" ({r['violations']} violations)" if r["violations"] else "")
              + (f"  {r['detail']}" if r["detail"] and (status != "PASS") else ""))
    if args.official:
        if not OFFICIAL_VALIDATOR:
            print("FAIL official validator path not configured (configs/local.json: official_validator)")
            return 1
        proc = subprocess.run([sys.executable, OFFICIAL_VALIDATOR,
                               "--matching", str(Path(args.output_dir) / "matching_results.tsv"),
                               "--candidate", str(Path(args.output_dir) / "_not_checked_by_official.tsv"),
                               "--test-dir", args.test_dir, "--check-ids"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr)
        ok &= proc.returncode == 0
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
