"""
Example of hybrid search with DuckDB as a single paramerized query.
"""
import json
from sentence_transformers import SentenceTransformer
import duckdb

from src.query import QUERY

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

con = duckdb.connect("data/duck_db/dbt_models.duckdb")

emb = model.encode(QUERY).tolist()  # MUST BE a Python list of floats

sql = """
WITH
-- Get top 100 vector search candidates
vector_scores AS (
    SELECT
        m.id,
        m.name,
        m.unique_id,
        m.description,
        list_cosine_similarity(m.embedding, ?) AS vector_score
    FROM models m
    ORDER BY vector_score DESC
    LIMIT 100
),
-- Get top 100 FTS candidates
fts_scores AS (
    SELECT
        m.id,
        m.name,
        m.unique_id,
        m.description,
        fts_main_models.match_bm25(m.id, ?) AS fts_score
    FROM models m
    WHERE fts_main_models.match_bm25(m.id, ?) IS NOT NULL
    ORDER BY fts_score DESC
    LIMIT 100
),
-- Combine all candidates (FULL OUTER JOIN via UNION)
all_candidates AS (
    SELECT id, name, unique_id, description FROM vector_scores
    UNION
    SELECT id, name, unique_id, description FROM fts_scores
),
-- Join back to get both scores for each candidate
candidate_scores AS (
    SELECT
        c.id,
        c.name,
        c.unique_id,
        c.description,
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
        description,
        vector_score,
        fts_score,
        -- Normalize vector scores to [0, 1]
        CASE
            WHEN MAX(vector_score) OVER () - MIN(vector_score) OVER () > 0
            THEN (vector_score - MIN(vector_score) OVER ()) / (MAX(vector_score) OVER () - MIN(vector_score) OVER ())
            ELSE CASE WHEN vector_score > 0 THEN 1.0 ELSE 0.0 END
        END AS normalized_vector_score,
        -- Normalize FTS scores to [0, 1]
        CASE
            WHEN MAX(fts_score) OVER () - MIN(fts_score) OVER () > 0
            THEN (fts_score - MIN(fts_score) OVER ()) / (MAX(fts_score) OVER () - MIN(fts_score) OVER ())
            ELSE CASE WHEN fts_score > 0 THEN 1.0 ELSE 0.0 END
        END AS normalized_fts_score,
        -- Boost score for exact name matches or substring matches
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
    description,
    CAST(vector_score AS DOUBLE) AS vector_score,
    CAST(fts_score AS DOUBLE) AS fts_score,
    CAST(normalized_vector_score AS DOUBLE) AS normalized_vector_score,
    CAST(normalized_fts_score AS DOUBLE) AS normalized_fts_score,
    CAST(name_match_boost AS DOUBLE) AS name_match_boost,
    -- Weighted hybrid score: 40% vector, 40% FTS, 20% name match boost
    CAST((0.4 * normalized_vector_score + 0.4 * normalized_fts_score + 0.2 * name_match_boost) AS DOUBLE) AS hybrid_score
FROM normalized_scores
ORDER BY hybrid_score DESC
LIMIT 10;
"""

# Make it pretty
# Parameters: emb, QUERY (FTS WHERE), QUERY (FTS SELECT), QUERY (exact match), QUERY (LIKE match)
res = con.execute(sql, [emb, QUERY, QUERY, QUERY, QUERY])
cols = [c[0] for c in res.description]
rows = [dict(zip(cols, row)) for row in res.fetchall()]

print(json.dumps(rows, indent=2))
