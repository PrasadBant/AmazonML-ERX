"""Blocking on a tiny synthetic database (same SQL as the real pipeline)."""
import duckdb
import pytest

from business_entity_resolution.blocking.candidates import BlockingParams, generate_candidates
from business_entity_resolution.blocking.keys import build_pair_keys
from business_entity_resolution.blocking.tokens import add_name_columns, build_token_tables
from business_entity_resolution.ids import entity_to_rid
from business_entity_resolution.preprocessing.address import normalize_address

RECORDS = [
    # (entity_id, name, address, country)
    ("S1-1", "Trinity Ministries Inc", "1552 Park Avenue, Indianapolis, IN", "US"),
    ("S1-2", "Harbor Holdings Group", "469 Oak Road, Ogden, UT", "US"),
    ("S1-3", "Trinity Ministries", "12 Rue Nationale, Lille", "France"),
    ("S2-10", "TRINITY MINISTRIES", "1552 PARK AVE, INDIANAPOLIS, IN", "US"),
    ("S3-11", "Trinity Mínistries Corporation", "1552 Park Ave, Indianapolis, Indiana", "US"),
    ("S2-12", "Harb0r Holdings Group LLC", "469 OAK RD, OGDEN, UT", "US"),
    ("S3-13", "Unrelated Bakery", "9 Elm Street, Boston, MA", "US"),
    ("S2-14", "Trinity Ministries", "12 R. Nationale, Lille", "France"),
]


@pytest.fixture()
def con():
    c = duckdb.connect()
    c.execute("""CREATE TABLE rec (split VARCHAR, src TINYINT, entity_id VARCHAR, name VARCHAR, addr VARCHAR,
                 country VARCHAR, nn VARCHAR, an VARCHAR, rid BIGINT)""")
    for eid, name, addr, country in RECORDS:
        nn = " ".join("".join(ch if ch.isalnum() else " " for ch in name.lower()).split())
        c.execute("INSERT INTO rec VALUES ('test', ?, ?, ?, ?, ?, ?, ?, ?)",
                  [int(eid[1]), eid, name, addr, country, nn, normalize_address(addr), entity_to_rid(eid)])
    add_name_columns(c)
    build_token_tables(c)
    build_pair_keys(c)
    c.execute("CREATE TABLE q AS SELECT rid FROM rec WHERE src = 1")
    return c


def candidates_of(con, entity_id, params=BlockingParams(df_cap=1000, top_k=100, top_k_name=20, top_k_address=20)):
    generate_candidates(con, "q", "test", params)
    rows = con.execute("SELECT t FROM cand WHERE s1 = ?", [entity_to_rid(entity_id)]).fetchall()
    return {r[0] for r in rows}


def test_true_variants_become_candidates(con):
    assert {entity_to_rid("S2-10"), entity_to_rid("S3-11")} <= candidates_of(con, "S1-1")
    assert entity_to_rid("S2-12") in candidates_of(con, "S1-2")


def test_blocking_stays_within_country(con):
    us_candidates = candidates_of(con, "S1-1")
    assert entity_to_rid("S2-14") not in us_candidates           # same name, France
    assert entity_to_rid("S2-14") in candidates_of(con, "S1-3")  # found from the French S1


def test_candidates_are_only_source2_and_source3(con):
    generate_candidates(con, "q", "test", BlockingParams())
    assert con.execute("SELECT count(*) FROM cand WHERE t // 10000000000 NOT IN (2, 3)").fetchone()[0] == 0


def test_top_k_limits_candidate_count(con):
    generate_candidates(con, "q", "test", BlockingParams(df_cap=1000, top_k=1, top_k_name=0, top_k_address=0))
    assert con.execute("SELECT max(n) FROM (SELECT s1, count(*) n FROM cand GROUP BY 1)").fetchone()[0] == 1
