"""
Hybrid search over catalog.json data in DuckDB.
Combines vector similarity, FTS, and name matching.
"""

import json
import logging
import sys
from pathlib import Path

import duckdb
from sentence_transformers import SentenceTransformer

from src.query import CATALOG_QUERY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/duck_db/dbt_catalog.duckdb"

if not Path(DB_PATH).exists():
    logger.error(f"Database not found: {DB_PATH}. Run ingestion script first.")
    sys.exit(1)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
con = duckdb.connect(DB_PATH)
emb = model.encode(CATALOG_QUERY).tolist()

sql = """
WITH
vector_scores AS (
    SELECT
        c.id,
        c.name,
        c.unique_id,
        c.node_type,
        c.schema_name,
        c.database_name,
        c.owner,
        c.comment,
        c.columns_json,
        c.row_count,
        list_cosine_similarity(c.embedding, ?) AS vector_score
    FROM catalog_nodes c
    ORDER BY vector_score DESC
    LIMIT 100
),
fts_scores AS (
    SELECT
        c.id,
        c.name,
        c.unique_id,
        c.node_type,
        c.schema_name,
        c.database_name,
        c.owner,
        c.comment,
        c.columns_json,
        c.row_count,
        fts_main_catalog_nodes.match_bm25(c.id, ?) AS fts_score
    FROM catalog_nodes c
    WHERE fts_main_catalog_nodes.match_bm25(c.id, ?) IS NOT NULL
    ORDER BY fts_score DESC
    LIMIT 100
),
all_candidates AS (
    SELECT id, name, unique_id, node_type, schema_name, database_name, owner, comment, columns_json, row_count FROM vector_scores
    UNION
    SELECT id, name, unique_id, node_type, schema_name, database_name, owner, comment, columns_json, row_count FROM fts_scores
),
candidate_scores AS (
    SELECT
        c.id,
        c.name,
        c.unique_id,
        c.node_type,
        c.schema_name,
        c.database_name,
        c.owner,
        c.comment,
        c.columns_json,
        c.row_count,
        COALESCE(v.vector_score, 0.0) AS vector_score,
        COALESCE(f.fts_score, 0.0) AS fts_score
    FROM all_candidates c
    LEFT JOIN vector_scores v ON c.id = v.id
    LEFT JOIN fts_scores f ON c.id = f.id
),
normalized_scores AS (
    SELECT
        id,
        name,
        unique_id,
        node_type,
        schema_name,
        database_name,
        owner,
        comment,
        columns_json,
        row_count,
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
            WHEN name = ? THEN 1.0
            WHEN name LIKE ('%' || ? || '%') THEN 0.5
            ELSE 0.0
        END AS name_match_boost
    FROM candidate_scores
)
SELECT
    name,
    unique_id,
    node_type,
    schema_name,
    database_name,
    owner,
    comment,
    columns_json,
    row_count,
    CAST(normalized_vector_score AS DOUBLE) AS normalized_vector_score,
    CAST(normalized_fts_score AS DOUBLE) AS normalized_fts_score,
    CAST(name_match_boost AS DOUBLE) AS name_match_boost,
    CAST((0.4 * normalized_vector_score + 0.4 * normalized_fts_score + 0.2 * name_match_boost) AS DOUBLE) AS hybrid_score
FROM normalized_scores
ORDER BY hybrid_score DESC
LIMIT 10;
"""

# Parameters: emb, QUERY (FTS x2), QUERY (name match x2)
res = con.execute(
    sql,
    [emb, CATALOG_QUERY, CATALOG_QUERY, CATALOG_QUERY, CATALOG_QUERY],
)
cols = [c[0] for c in res.description]
rows = [dict(zip(cols, row)) for row in res.fetchall()]

print(json.dumps(rows, indent=2))
