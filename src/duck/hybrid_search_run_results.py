"""
Hybrid search over run_results.json data in DuckDB.
Combines vector similarity, FTS, name matching, and status boosting.
"""

import json
import logging
import sys
from pathlib import Path

import duckdb
from sentence_transformers import SentenceTransformer

from src.query import RUN_RESULTS_QUERY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/duck_db/dbt_run_results.duckdb"

if not Path(DB_PATH).exists():
    logger.error(f"Database not found: {DB_PATH}. Run ingestion script first.")
    sys.exit(1)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
con = duckdb.connect(DB_PATH)
emb = model.encode(RUN_RESULTS_QUERY).tolist()

sql = """
WITH
-- Get top 100 vector search candidates
vector_scores AS (
    SELECT
        r.id,
        r.name,
        r.unique_id,
        r.resource_type,
        r.status,
        r.message,
        r.compiled_code,
        r.execution_time,
        list_cosine_similarity(r.embedding, ?) AS vector_score
    FROM run_results r
    ORDER BY vector_score DESC
    LIMIT 100
),
-- Get top 100 FTS candidates
fts_scores AS (
    SELECT
        r.id,
        r.name,
        r.unique_id,
        r.resource_type,
        r.status,
        r.message,
        r.compiled_code,
        r.execution_time,
        fts_main_run_results.match_bm25(r.id, ?) AS fts_score
    FROM run_results r
    WHERE fts_main_run_results.match_bm25(r.id, ?) IS NOT NULL
    ORDER BY fts_score DESC
    LIMIT 100
),
-- Combine all candidates
all_candidates AS (
    SELECT id, name, unique_id, resource_type, status, message, compiled_code, execution_time FROM vector_scores
    UNION
    SELECT id, name, unique_id, resource_type, status, message, compiled_code, execution_time FROM fts_scores
),
-- Join back to get both scores for each candidate
candidate_scores AS (
    SELECT
        c.id,
        c.name,
        c.unique_id,
        c.resource_type,
        c.status,
        c.message,
        c.compiled_code,
        c.execution_time,
        COALESCE(v.vector_score, 0.0) AS vector_score,
        COALESCE(f.fts_score, 0.0) AS fts_score
    FROM all_candidates c
    LEFT JOIN vector_scores v ON c.id = v.id
    LEFT JOIN fts_scores f ON c.id = f.id
),
-- Normalize scores within the candidate set
normalized_scores AS (
    SELECT
        id,
        name,
        unique_id,
        resource_type,
        status,
        message,
        compiled_code,
        execution_time,
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
        -- Name match boost
        CASE
            WHEN name = ? THEN 1.0
            WHEN name LIKE ('%' || ? || '%') THEN 0.5
            ELSE 0.0
        END AS name_match_boost,
        -- Status boost: reward when query intent matches result status
        CASE
            WHEN status IN ('error', 'fail') AND (? ILIKE '%fail%' OR ? ILIKE '%error%') THEN 1.0
            WHEN status = 'warn' AND ? ILIKE '%warn%' THEN 1.0
            WHEN status = 'success' AND ? ILIKE '%success%' THEN 0.5
            ELSE 0.0
        END AS status_boost
    FROM candidate_scores
)
SELECT
    name,
    unique_id,
    resource_type,
    status,
    message,
    LEFT(compiled_code, 200) AS compiled_code_preview,
    CAST(execution_time AS DOUBLE) AS execution_time,
    CAST(normalized_vector_score AS DOUBLE) AS normalized_vector_score,
    CAST(normalized_fts_score AS DOUBLE) AS normalized_fts_score,
    CAST(name_match_boost AS DOUBLE) AS name_match_boost,
    CAST(status_boost AS DOUBLE) AS status_boost,
    CAST((0.35 * normalized_vector_score + 0.35 * normalized_fts_score + 0.15 * name_match_boost + 0.15 * status_boost) AS DOUBLE) AS hybrid_score
FROM normalized_scores
ORDER BY hybrid_score DESC
LIMIT 10;
"""

# Parameters: emb, QUERY (FTS WHERE), QUERY (FTS SELECT),
# QUERY (exact match), QUERY (LIKE match),
# QUERY (status x4 for ILIKE checks)
res = con.execute(
    sql,
    [
        emb,
        RUN_RESULTS_QUERY,
        RUN_RESULTS_QUERY,
        RUN_RESULTS_QUERY,
        RUN_RESULTS_QUERY,
        RUN_RESULTS_QUERY,
        RUN_RESULTS_QUERY,
        RUN_RESULTS_QUERY,
        RUN_RESULTS_QUERY,
    ],
)
cols = [c[0] for c in res.description]
rows = [dict(zip(cols, row)) for row in res.fetchall()]

print(json.dumps(rows, indent=2))
