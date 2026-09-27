"""Business-name normalization (DuckDB SQL, applied to every source and country)."""
from __future__ import annotations

# Legal forms, honorifics and connector words (EN / IN / FR / generic) removed to form the "core" name.
# Single letters appear because punctuation splitting turns "L.L.C." / "M/s" into letters.
LEGAL_FORMS = """limited private llc inc ltd pvt corp co corporation company llp lp pc pllc plc
l c s p d a m the and of dr mr mrs ms sri shri smt com www incorporated
sarl sas sasu eurl sa sci ei cie et de du des la le les ets etablissements societe
gmbh ag bv nv pte sdn bhd pty""".split()


def normalize_name_sql(col: str) -> str:
    """lower-case, strip accents, non-letter/digit runs -> single space (any script)."""
    return (rf"trim(regexp_replace(regexp_replace(lower(strip_accents(coalesce({col}, ''))), "
            rf"'[^\p{{L}}\p{{N}}]+', ' ', 'g'), '\s+', ' ', 'g'))")


def create_core_tokens_macro(con) -> None:
    """core_toks(normalized_name) -> list of tokens without legal forms / honorifics."""
    stop = "[" + ",".join(f"'{w}'" for w in LEGAL_FORMS) + "]"
    con.execute(f"CREATE OR REPLACE MACRO core_toks(s) AS "
                f"list_filter(string_split(s, ' '), x -> x <> '' AND NOT list_contains({stop}, x))")
