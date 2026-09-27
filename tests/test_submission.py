"""Submission writer + local validator on synthetic scores and a synthetic test directory."""
import pandas as pd
import pytest

from business_entity_resolution.evaluation.submission_checks import check_submission
from business_entity_resolution.ids import entity_to_rid
from business_entity_resolution.pipeline.submission import DecisionRule, write_submission

S1 = ["S1-1", "S1-2", "S1-3", "S1-4"]
TARGETS = ["S2-10", "S3-11", "S2-12", "S3-13"]


@pytest.fixture()
def test_dir(tmp_path):
    d = tmp_path / "test"
    d.mkdir()
    for n, ids in ((1, S1), (2, [t for t in TARGETS if t.startswith("S2")]), (3, [t for t in TARGETS if t.startswith("S3")])):
        rows = "".join(f"{i}\tname\taddr\tUS\n" for i in ids)
        (d / f"test_source{n}.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n" + rows)
    return d


@pytest.fixture()
def score_files(tmp_path):
    scores = pd.DataFrame({
        "s1": [entity_to_rid(x) for x in ["S1-1", "S1-1", "S1-2", "S1-2", "S1-3"]],
        "t": [entity_to_rid(x) for x in ["S2-10", "S3-11", "S2-10", "S2-12", "S3-13"]],
        "p": [0.95, 0.40, 0.80, 0.70, 0.30],
    })
    files = []
    for i, part in enumerate((scores.iloc[:2], scores.iloc[2:])):  # two chunks, S1 never split across chunks
        path = tmp_path / f"chunk_{i:03d}.parquet"
        part.to_parquet(path)
        files.append(str(path))
    return files


def read_tsv(path):
    lines = path.read_text().splitlines()
    return lines[0], dict(line.split("\t") for line in lines[1:])


def test_writes_one_row_per_source1_and_matches_are_candidates(tmp_path, score_files, test_dir):
    out = tmp_path / "out"
    stats = write_submission(score_files, [entity_to_rid(x) for x in S1], out, DecisionRule(threshold=0.5))
    header, matches = read_tsv(out / "matching_results.tsv")
    _, cands = read_tsv(out / "candidate_pairs.tsv")
    assert header == "source1_entity_id\tmatched_entity_ids"
    assert set(matches) == set(cands) == set(S1)
    assert matches["S1-1"] == "S2-10" and matches["S1-3"] == "" and matches["S1-4"] == ""
    assert set(matches["S1-2"].split(",")) == {"S2-10", "S2-12"}   # plain threshold: S2-10 given twice
    for s1, m in matches.items():
        assert set(filter(None, m.split(","))) <= set(filter(None, cands[s1].split(",")))
    assert stats["s1_without_candidates"] == 1
    results = check_submission(out, test_dir)
    assert all(r["passed"] for r in results.values())


def test_one_owner_gives_each_record_to_the_best_source1(tmp_path, score_files):
    out = tmp_path / "owner"
    write_submission(score_files, [entity_to_rid(x) for x in S1], out, DecisionRule(0.5, one_owner=True))
    _, matches = read_tsv(out / "matching_results.tsv")
    assert matches["S1-1"] == "S2-10" and matches["S1-2"] == "S2-12"   # S2-10 stays with S1-1 (0.95 > 0.80)


def test_margin_drops_close_contests(tmp_path, score_files):
    out = tmp_path / "margin"
    write_submission(score_files, [entity_to_rid(x) for x in S1], out, DecisionRule(0.5, True, margin=0.2))
    _, matches = read_tsv(out / "matching_results.tsv")
    assert matches["S1-1"] == ""   # 0.95 vs 0.80 is closer than 0.2 -> nobody gets S2-10


def test_refuses_to_overwrite_existing_submission(tmp_path, score_files):
    out = tmp_path / "protected"
    write_submission(score_files, [entity_to_rid(x) for x in S1], out, DecisionRule(0.5))
    with pytest.raises(FileExistsError):
        write_submission(score_files, [entity_to_rid(x) for x in S1], out, DecisionRule(0.9))


def test_validator_detects_broken_files(tmp_path, test_dir):
    out = tmp_path / "broken"
    out.mkdir()
    (out / "candidate_pairs.tsv").write_text(
        "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-10,S2-10\nS1-1\tS3-11\nS1-2\tS1-3\nS1-3\tS2-99\n")
    (out / "matching_results.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS3-13\nS1-2\t\nS1-3\t\textra\nS1-4\t\n")
    r = check_submission(out, test_dir)
    failed = {k for k, v in r.items() if not v["passed"]}
    assert {"candidate:no_duplicate_ids_in_list", "candidate:no_duplicate_s1_rows", "candidate:no_missing_s1",
            "candidate:ids_s2_s3_only", "candidate:ids_exist_in_test", "matching:one_tab_per_line",
            "matches_subset_of_candidates"} <= failed
