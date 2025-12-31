"""
Example of hybrid search with LanceDB combining vector similarity and FTS.
Generated w/ Claude (pointed at DuckDB implementation but plan on using this
as a learning script and deepen my personal understanding)
"""
import json
import logging
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import lancedb
import pyarrow as pa
import pyarrow.compute as pc

from src.query import QUERY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/lance_db"

# Check database exists
if not Path(DB_PATH).exists():
    logger.error(f"Database not found: {DB_PATH}. Run ingestion script first.")
    sys.exit(1)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
db = lancedb.connect(DB_PATH)
table = db.open_table("models")

query_vector = model.encode(QUERY).tolist()

# 1. Perform vector search
vector_results = (
    table.search(query_vector)
    .limit(100)  # Get more candidates for reranking
    .to_arrow()
)

# 2. Perform FTS search
fts_results = (
    table.search(QUERY, query_type="fts")
    .limit(100)
    .to_arrow()
)

# 3. Combine results with hybrid scoring
# Normalize vector scores (distance to similarity: smaller distance = higher similarity)
if vector_results.num_rows > 0:
    distances = vector_results["_distance"]
    max_distance = pc.max(distances).as_py()
    min_distance = pc.min(distances).as_py()

    if max_distance > min_distance:
        # Normalize: 1 - (distance - min) / (max - min)
        normalized = pc.divide(
            pc.subtract(distances, min_distance),
            max_distance - min_distance
        )
        vector_scores = pc.subtract(1.0, normalized)
    else:
        vector_scores = pa.array([1.0] * vector_results.num_rows)

    # Add vector_score column
    vector_results = vector_results.append_column("vector_score", vector_scores)

# Normalize FTS scores
if fts_results.num_rows > 0:
    # Find the score column (try different possible names)
    score_col = None
    for possible_name in ["_relevance_score", "score", "_score", "relevance_score"]:
        if possible_name in fts_results.column_names:
            score_col = possible_name
            break

    if score_col:
        scores = fts_results[score_col]
        max_score = pc.max(scores).as_py()
        min_score = pc.min(scores).as_py()

        if max_score > min_score:
            # Normalize: (score - min) / (max - min)
            fts_scores = pc.divide(
                pc.subtract(scores, min_score),
                max_score - min_score
            )
        else:
            fts_scores = pa.array([1.0] * fts_results.num_rows)
    else:
        logger.warning(f"No score column found. Available: {fts_results.column_names}")
        fts_scores = pa.array([1.0] * fts_results.num_rows)

    # Add fts_score column
    fts_results = fts_results.append_column("fts_score", fts_scores)

# Prepare tables for merging - select and cast id columns
vector_table = vector_results.select(["id", "name", "unique_id", "description", "vector_score"])
vector_table = vector_table.set_column(0, "id", pc.cast(vector_table["id"], pa.int64()))

fts_table = fts_results.select(["id", "fts_score"])
fts_table = fts_table.set_column(0, "id", pc.cast(fts_table["id"], pa.int64()))

# Perform full outer join on id
# PyArrow doesn't have a direct merge, so we'll use a dictionary-based approach
vector_dict = {}
for i in range(vector_table.num_rows):
    id_val = vector_table["id"][i].as_py()
    vector_dict[id_val] = {
        "name": vector_table["name"][i].as_py(),
        "unique_id": vector_table["unique_id"][i].as_py(),
        "description": vector_table["description"][i].as_py(),
        "vector_score": vector_table["vector_score"][i].as_py(),
    }

fts_dict = {}
for i in range(fts_table.num_rows):
    id_val = fts_table["id"][i].as_py()
    fts_dict[id_val] = fts_table["fts_score"][i].as_py()

# Merge dictionaries
all_ids = set(vector_dict.keys()) | set(fts_dict.keys())
merged_data = []
for id_val in all_ids:
    vector_data = vector_dict.get(id_val, {})
    fts_score = fts_dict.get(id_val, 0.0)
    vector_score = vector_data.get("vector_score", 0.0)

    # Calculate hybrid score (0.5 weight for each)
    hybrid_score = 0.5 * vector_score + 0.5 * fts_score

    merged_data.append({
        "name": vector_data.get("name", ""),
        "unique_id": vector_data.get("unique_id", ""),
        "description": vector_data.get("description", ""),
        "vector_score": vector_score,
        "fts_score": fts_score,
        "hybrid_score": hybrid_score,
    })

# Sort by hybrid score descending and take top 10
merged_data.sort(key=lambda x: x["hybrid_score"], reverse=True)
top_results = merged_data[:10]

print(json.dumps(top_results, indent=2))
