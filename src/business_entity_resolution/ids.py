"""Compact integer ids for entity ids.

`S{source}-{number}` <-> source * 1e10 + number. The mapping is lossless for all 24.2M challenge
records (no zero-padded numbers), which was verified against the raw files.
"""
from __future__ import annotations

SOURCE_FACTOR = 10_000_000_000

# SQL expressions (DuckDB) for the same mapping; {col} is the column to convert
RID_FROM_ENTITY_SQL = "({src}::BIGINT * 10000000000 + CAST(substr({col}, 4) AS BIGINT))"
ENTITY_FROM_RID_SQL = "('S' || ({col} // 10000000000)::VARCHAR || '-' || ({col} % 10000000000)::VARCHAR)"


def entity_to_rid(entity_id: str) -> int:
    prefix, number = entity_id.split("-", 1)
    return int(prefix[1:]) * SOURCE_FACTOR + int(number)


def rid_to_entity(rid: int) -> str:
    return f"S{rid // SOURCE_FACTOR}-{rid % SOURCE_FACTOR}"


def source_of(rid: int) -> int:
    return rid // SOURCE_FACTOR
