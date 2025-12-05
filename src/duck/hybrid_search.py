"""
Example of hybrid search with DuckDB as a single paramerized query.
"""
import json
from sentence_transformers import SentenceTransformer
import duckdb

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

con = duckdb.connect("data/duck_db/dbt_models.duckdb")

query = "user_accounts_to_salesforce_accounts__bridge"

emb = model.encode(query).tolist()  # MUST BE a Python list of floats

sql = """
WITH
vector_scores AS (
    SELECT
        m.id,
        m.name,
        m.unique_id,
        m.description,
        list_cosine_similarity(m.embedding, ?) AS vector_score
    FROM models m
),
fts_scores AS (
    SELECT
        m.id,
        fts_main_models.match_bm25(m.id, ?) AS fts_score
    FROM models m
)
SELECT
    v.name,
    v.unique_id,
    v.description,
    v.vector_score,
    COALESCE(f.fts_score, 0.0) AS fts_score,
    (0.5 * v.vector_score + 0.5 * COALESCE(f.fts_score, 0.0)) AS hybrid_score
FROM vector_scores v
LEFT JOIN fts_scores f ON v.id = f.id
ORDER BY hybrid_score DESC
LIMIT 10;
"""

# Make it pretty
res = con.execute(sql, [emb, query])
cols = [c[0] for c in res.description]
rows = [dict(zip(cols, row)) for row in res.fetchall()]

print(json.dumps(rows, indent=2))
