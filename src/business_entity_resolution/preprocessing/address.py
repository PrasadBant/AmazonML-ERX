"""Address normalization, applied identically to every source and country (country is an open set).

lower + strip accents -> multi-word region names to codes -> tokenize -> abbreviation map ->
strip leading zeros from numbers -> drop noise tokens. A full region name maps to exactly what its
two-letter code maps to, so "CT" and "Connecticut" end up as the same token.
"""
from __future__ import annotations

import re
import time
import unicodedata
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import WORK_DIR

ABBREVIATIONS = {
    # street types (EN / FR / IN)
    "st": "street", "str": "street", "rd": "road", "av": "avenue", "ave": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "boul": "boulevard", "dr": "drive", "drv": "drive",
    "ln": "lane", "ct": "court", "crt": "court", "pl": "place", "plz": "plaza", "sq": "square",
    "hwy": "highway", "hway": "highway", "pkwy": "parkway", "pky": "parkway", "cir": "circle",
    "ter": "terrace", "terr": "terrace", "trl": "trail", "tr": "trail", "cres": "crescent",
    "expy": "expressway", "fwy": "freeway", "mt": "mount", "ft": "fort", "pt": "point",
    "r": "rue", "rte": "route", "all": "allee", "che": "chemin", "chem": "chemin", "imp": "impasse",
    "fbg": "faubourg", "qu": "quai",
    "nr": "near", "opp": "opposite", "flr": "floor", "fl": "floor", "apt": "apartment",
    "appt": "apartment", "bldg": "building", "bldng": "building", "extn": "extension", "ext": "extension",
    "ngr": "nagar", "mkt": "market", "stn": "station", "rly": "railway", "ste": "suite",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "1st": "1", "2nd": "2", "3rd": "3", "4th": "4", "5th": "5", "6th": "6", "7th": "7", "8th": "8", "9th": "9",
}
NOISE_TOKENS = {"no", "hno", "h", "null", "na", "nos"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "or": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "ts": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py",
}

_NULL = re.compile(r"<?null>?|n/a")
_SPLIT = re.compile(r"[\W_]+")
_MULTI_WORD_REGIONS = sorted(((k, v) for d in (US_STATES, IN_STATES) for k, v in d.items() if " " in k),
                             key=lambda kv: -len(kv[0]))
_WORD_MAP = dict(ABBREVIATIONS)
_WORD_MAP["ts"] = "tg"
for _regions in (US_STATES, IN_STATES):
    for _name, _code in _regions.items():
        if " " not in _name:
            _WORD_MAP[_name] = ABBREVIATIONS.get(_code, _code)


def normalize_address(address: str | None) -> str:
    if not address:
        return ""
    s = unicodedata.normalize("NFKD", address.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = " " + " ".join(t for t in _SPLIT.split(_NULL.sub(" ", s)) if t) + " "
    for name, code in _MULTI_WORD_REGIONS:
        if name in s:
            s = s.replace(f" {name} ", f" {ABBREVIATIONS.get(code, code)} ")
    out = []
    for token in s.split():
        if token in NOISE_TOKENS:
            continue
        out.append((token.lstrip("0") or "0") if token.isdigit() else _WORD_MAP.get(token, token))
    return " ".join(out)


def _normalize_batch(batch: list) -> list:
    return [normalize_address(a) for a in batch]


def normalize_addresses_in_db(con, workers: int = 6) -> None:
    """Fill rec.an for all records (streamed in 1M-row batches, parallel over processes)."""
    t0 = time.time()
    out = WORK_DIR / "addr_norm.parquet"
    reader = con.execute("SELECT split, rid, addr FROM rec").fetch_record_batch(1_000_000)
    writer = None
    with Pool(workers) as pool:
        for rb in reader:
            addrs = rb.column(2).to_pylist()
            parts = [addrs[i:i + 50_000] for i in range(0, len(addrs), 50_000)]
            normalized = [x for part in pool.map(_normalize_batch, parts) for x in part]
            tbl = pa.table({"split": rb.column(0), "rid": rb.column(1), "an2": pa.array(normalized, pa.string())})
            writer = writer or pq.ParquetWriter(out, tbl.schema)
            writer.write_table(tbl)
            print(f"  addresses normalized ({time.time() - t0:.0f}s)", flush=True)
    writer.close()
    con.execute(f"""CREATE OR REPLACE TABLE rec AS SELECT r.* REPLACE (m.an2 AS an)
                    FROM rec r JOIN read_parquet('{out.as_posix()}') m USING (split, rid)""")
