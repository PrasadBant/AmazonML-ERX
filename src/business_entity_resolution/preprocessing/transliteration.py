"""Indic-script business names -> Latin, using a word dictionary learned from training labels only.

Indic names in S2/S3 are word-by-word transliterations of the English S1 name. For label pairs from
training folds 0-7 whose word counts agree, each Indic word is aligned positionally with the Latin
word and the majority translation wins. Unseen words fall back to Unidecode romanization.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
from unidecode import unidecode

from ..data.folds import TRAIN_FOLDS_MAX

INDIC = re.compile(r"[ऀ-෿]")
INDIC_SQL = r"'[\x{0900}-\x{0DFF}]'"
_INDIC_SPLIT = re.compile(r"[\s,.()\-/&+#\[\]]+")
_LATIN_SPLIT = re.compile(r"[^\w]+")


def latin_words(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return [w for w in _LATIN_SPLIT.split(s) if w]


def indic_words(s: str) -> list[str]:
    return [w for w in _INDIC_SPLIT.split(unicodedata.normalize("NFC", s)) if w]


def romanize(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unidecode(word).lower())


def learn_dictionary(pairs) -> dict[str, str]:
    """pairs: iterable of (latin_name, indic_name). Ties are broken alphabetically (deterministic)."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for latin_name, indic_name in pairs:
        a, b = latin_words(latin_name), indic_words(indic_name)
        if len(a) != len(b):
            continue
        for indic, latin in zip(b, a):
            if INDIC.search(indic):
                votes[indic][latin] += 1
    return {w: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for w, c in votes.items()}


def transliterate(name: str, dictionary: dict[str, str]) -> str:
    out = []
    for word in indic_words(name):
        if INDIC.search(word):
            out.append(dictionary.get(word) or romanize(word))
        else:
            out.extend(latin_words(word))
    return " ".join(x for x in out if x)


def learn_dictionary_from_db(con) -> dict[str, str]:
    rows = con.execute(f"""
        SELECT s.name, t.name FROM fold f JOIN gt_pairs g ON g.s1 = f.s1
        JOIN rec s ON s.split='train' AND s.rid=f.rid
        JOIN rec t ON t.split='train' AND t.entity_id=g.tid
        WHERE f.f <= {TRAIN_FOLDS_MAX} AND f.country <> 'US' AND regexp_matches(t.name, {INDIC_SQL})""").fetchall()
    return learn_dictionary(rows)


def apply_transliteration(con, dictionary_path: Path) -> dict:
    """Rewrite rec.nn for every Indic-script name. Reuses a saved dictionary when present so the
    submitted versions stay reproducible; otherwise learns one and saves it."""
    if dictionary_path.exists():
        dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    else:
        dictionary = learn_dictionary_from_db(con)
        dictionary_path.write_text(json.dumps(dictionary, ensure_ascii=False), encoding="utf-8")
    rows = con.execute(f"SELECT split, rid, name FROM rec WHERE regexp_matches(name, {INDIC_SQL})").fetchall()
    words = [w for _, _, name in rows for w in indic_words(name) if INDIC.search(w)]
    table = pa.table({"split": [r[0] for r in rows], "rid": [r[1] for r in rows],
                      "nn2": [transliterate(r[2], dictionary) for r in rows]})
    con.register("translit_rows", table)
    con.execute("CREATE OR REPLACE TABLE translit_map AS SELECT * FROM translit_rows")
    con.execute("UPDATE rec SET nn = m.nn2 FROM translit_map m WHERE rec.split = m.split AND rec.rid = m.rid")
    return {"dictionary_words": len(dictionary), "records": len(rows),
            "word_coverage": sum(w in dictionary for w in words) / max(len(words), 1)}
