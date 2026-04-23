"""
Hybrid search over sources.json data in DuckDB.
Combines vector similarity, FTS, and name matching.
"""

import json
import logging
import sys
from pathlib import Path

import duckdb
from sentence_transformers import SentenceTransformer

from src.query import SOURCES_QUERY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/duck_db/dbt_sources.duckdb"

if not Path(DB_PATH).exists():
    logger.error(f"Database not found: {DB_PATH}. Run ingestion script first.")
    sys.exit(1)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
con = duckdb.connect(DB_PATH)
emb = model.encode(SOURCES_QUERY).tolist()

sql = """
WITH
vector_scores AS (
    SELECT
        s.id,
        s.source_name,
        s.table_name,
        s.unique_id,
        s.status,
        s.max_loaded_at,
        s.snapshotted_at,
        s.max_loaded_at_time_ago_in_s,
        s.criteria,
        list_cosine_similarity(s.embedding, ?) AS vector_score
    FROM source_freshness s
    ORDER BY vector_score DESC
    LIMIT 100
),
fts_scores AS (
    SELECT
        s.id,
        s.source_name,
        s.table_name,
        s.unique_id,
        s.status,
        s.max_loaded_at,
        s.snapshotted_at,
        s.max_loaded_at_time_ago_in_s,
        s.criteria,
        fts_main_source_freshness.match_bm25(s.id, ?) AS fts_score
    FROM source_freshness s
    WHERE fts_main_source_freshness.match_bm25(s.id, ?) IS NOT NULL
    ORDER BY fts_score DESC
    LIMIT 100
),
all_candidates AS (
    SELECT id, source_name, table_name, unique_id, status, max_loaded_at, snapshotted_at, max_loaded_at_time_ago_in_s, criteria FROM vector_scores
    UNION
    SELECT id, source_name, table_name, unique_id, status, max_loaded_at, snapshotted_at, max_loaded_at_time_ago_in_s, criteria FROM fts_scores
),
candidate_scores AS (
    SELECT
        c.id,
        c.source_name,
        c.table_name,
        c.unique_id,
        c.status,
        c.max_loaded_at,
        c.snapshotted_at,
        c.max_loaded_at_time_ago_in_s,
        c.criteria,
        COALESCE(v.vector_score, 0.0) AS vector_score,
        COALESCE(f.fts_score, 0.0) AS fts_score
    FROM all_candidates c
    LEFT JOIN vector_scores v ON c.id = v.id
    LEFT JOIN fts_scores f ON c.id = f.id
),
normalized_scores AS (
    SELECT
        id,
        source_name,
        table_name,
        unique_id,
        status,
        max_loaded_at,
        snapshotted_at,
        max_loaded_at_time_ago_in_s,
        criteria,
        vector_score,
        fts_score,
        CASE
            WHEN MAX(vector_score) OVER () - MIN(vector_score) OVER () > 0
            THEN (vector_score - MIN(vector_score) OVER ()) / (MAX(vector_score) OVER () - MIN(vector_score) OVER ())
            ELSE CASE WHEN vector_score > 0 THEN 1.0 ELSE 0.0 END
        END AS normalized_vector_score,
        CASE
            WHEN MAX(fts_score) OVER () - MIN(fts_score) OVER () > 0
            THEN (fts_score - MIN(fts_score) OVER ()) / (MAX(fts_score) OVER () - MIN(fts_score) OVER ())
            ELSE CASE WHEN fts_score > 0 THEN 1.0 ELSE 0.0 END
        END AS normalized_fts_score,
        CASE
            WHEN table_name = ? THEN 1.0
            WHEN table_name LIKE ('%' || ? || '%') THEN 0.5
            WHEN source_name LIKE ('%' || ? || '%') THEN 0.3
            ELSE 0.0
        END AS name_match_boost
    FROM candidate_scores
)
SELECT
    source_name,
    table_name,
    unique_id,
    status,
    max_loaded_at,
    snapshotted_at,
    CAST(max_loaded_at_time_ago_in_s AS DOUBLE) AS max_loaded_at_time_ago_in_s,
    criteria,
    CAST(normalized_vector_score AS DOUBLE) AS normalized_vector_score,
    CAST(normalized_fts_score AS DOUBLE) AS normalized_fts_score,
    CAST(name_match_boost AS DOUBLE) AS name_match_boost,
    CAST((0.4 * normalized_vector_score + 0.4 * normalized_fts_score + 0.2 * name_match_boost) AS DOUBLE) AS hybrid_score
FROM normalized_scores
ORDER BY hybrid_score DESC
LIMIT 10;
"""

# Parameters: emb, QUERY (FTS x2), QUERY (name match x3)
res = con.execute(
    sql,
    [emb, SOURCES_QUERY, SOURCES_QUERY, SOURCES_QUERY, SOURCES_QUERY, SOURCES_QUERY],
)
cols = [c[0] for c in res.description]
rows = [dict(zip(cols, row)) for row in res.fetchall()]

print(json.dumps(rows, indent=2))
