import duckdb
import pytest

from business_entity_resolution.ids import entity_to_rid, rid_to_entity, source_of
from business_entity_resolution.preprocessing.address import normalize_address
from business_entity_resolution.preprocessing.names import create_core_tokens_macro, normalize_name_sql
from business_entity_resolution.preprocessing.transliteration import learn_dictionary, transliterate


@pytest.mark.parametrize("entity_id", ["S1-925783039", "S2-5", "S3-22467283"])
def test_entity_id_round_trip(entity_id):
    rid = entity_to_rid(entity_id)
    assert rid_to_entity(rid) == entity_id
    assert source_of(rid) == int(entity_id[1])


@pytest.mark.parametrize("raw, expected", [
    ("00508 ALAMEDA RD, AZLE, TX", "508 alameda road azle tx"),
    ("175 Madison St, Malden, Massachusetts", "175 madison street malden ma"),
    ("0030 Rte. Du Cap Ferret", "30 route du cap ferret"),
    ("20 R. DU MILLAU, NANTES", "20 rue du millau nantes"),
    ("H.no B3/1a, Hyderabad, TG", "b3 1a hyderabad tg"),
    ("1 Main St, <NULL>, Albany, New York", "1 main street albany ny"),
    ("Chennai, Tamil Nadu", "chennai tn"),
    (None, ""),
])
def test_normalize_address(raw, expected):
    assert normalize_address(raw) == expected


def test_region_name_and_code_agree():
    # "CT" and "Connecticut" must land on the same token even though "ct" is also "court"
    assert normalize_address("Hartford, CT").split()[-1] == normalize_address("Hartford, Connecticut").split()[-1]


def test_name_normalization_and_core_tokens():
    con = duckdb.connect()
    create_core_tokens_macro(con)
    nn = con.execute(f"SELECT {normalize_name_sql('?')}", ["Punia (India) Investment Pvt. Ltd."]).fetchone()[0]
    assert nn == "punia india investment pvt ltd"
    assert con.execute("SELECT core_toks(?)", [nn]).fetchone()[0] == ["punia", "india", "investment"]
    accented = con.execute(f"SELECT {normalize_name_sql('?')}", ["Moncada Léarning & Co"]).fetchone()[0]
    assert accented == "moncada learning co"


def test_transliteration_dictionary_is_learned_from_aligned_pairs():
    pairs = [("North Foods Private Limited", "नॉर्थ फूड्स प्राइवेट लिमिटेड"),
             ("North Global Foods Pvt Ltd", "नॉर्थ ग्लोबल फूड्स प्रा. लि."),
             ("Mismatched Word Count", "नॉर्थ")]  # different word counts -> ignored
    dictionary = learn_dictionary(pairs)
    assert dictionary["नॉर्थ"] == "north"
    assert dictionary["फूड्स"] == "foods"
    assert transliterate("नॉर्थ फूड्स", dictionary) == "north foods"
    assert transliterate("Latin Name", dictionary) == "latin name"
    assert transliterate("अज्ञात", dictionary)  # unseen word falls back to romanization, never empty
