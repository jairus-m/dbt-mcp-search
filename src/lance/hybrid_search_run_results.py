"""
Hybrid search over run_results.json data in LanceDB.
Combines vector similarity, FTS, name matching, and status boosting.
"""

import json
import logging
import sys
from pathlib import Path

import lancedb
import pyarrow as pa
import pyarrow.compute as pc
from sentence_transformers import SentenceTransformer

from src.query import RUN_RESULTS_QUERY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/lance_db"

if not Path(DB_PATH).exists():
    logger.error(f"Database not found: {DB_PATH}. Run ingestion script first.")
    sys.exit(1)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
db = lancedb.connect(DB_PATH)
table = db.open_table("run_results")

query_vector = model.encode(RUN_RESULTS_QUERY).tolist()

# 1. Perform vector search
vector_results = table.search(query_vector).limit(100).to_arrow()

# 2. Perform FTS search
fts_results = table.search(RUN_RESULTS_QUERY, query_type="fts").limit(100).to_arrow()

# 3. Normalize vector scores (distance to similarity)
if vector_results.num_rows > 0:
    distances = vector_results["_distance"]
    max_distance = pc.max(distances).as_py()
    min_distance = pc.min(distances).as_py()

    if max_distance > min_distance:
        normalized = pc.divide(
            pc.subtract(distances, min_distance), max_distance - min_distance
        )
        vector_scores = pc.subtract(1.0, normalized)
    else:
        vector_scores = pa.array([1.0] * vector_results.num_rows)

    vector_results = vector_results.append_column("vector_score", vector_scores)

# 4. Normalize FTS scores
if fts_results.num_rows > 0:
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
            fts_scores = pc.divide(
                pc.subtract(scores, min_score), max_score - min_score
            )
        else:
            fts_scores = pa.array([1.0] * fts_results.num_rows)
    else:
        logger.warning(f"No score column found. Available: {fts_results.column_names}")
        fts_scores = pa.array([1.0] * fts_results.num_rows)

    fts_results = fts_results.append_column("fts_score", fts_scores)

# 5. Build dictionaries for merging
select_cols = [
    "id", "name", "unique_id", "resource_type", "status",
    "message", "compiled_code", "execution_time", "vector_score",
]
vector_table = vector_results.select(select_cols)
vector_table = vector_table.set_column(
    0, "id", pc.cast(vector_table["id"], pa.int64())
)

fts_table = fts_results.select(["id", "fts_score"])
fts_table = fts_table.set_column(0, "id", pc.cast(fts_table["id"], pa.int64()))

vector_dict = {}
for i in range(vector_table.num_rows):
    id_val = vector_table["id"][i].as_py()
    vector_dict[id_val] = {
        "name": vector_table["name"][i].as_py(),
        "unique_id": vector_table["unique_id"][i].as_py(),
        "resource_type": vector_table["resource_type"][i].as_py(),
        "status": vector_table["status"][i].as_py(),
        "message": vector_table["message"][i].as_py(),
        "compiled_code": vector_table["compiled_code"][i].as_py(),
        "execution_time": vector_table["execution_time"][i].as_py(),
        "vector_score": vector_table["vector_score"][i].as_py(),
    }

fts_dict = {}
for i in range(fts_table.num_rows):
    id_val = fts_table["id"][i].as_py()
    fts_dict[id_val] = fts_table["fts_score"][i].as_py()

# 6. Merge and compute hybrid scores
query_lower = RUN_RESULTS_QUERY.lower()
all_ids = set(vector_dict.keys()) | set(fts_dict.keys())
merged_data = []

for id_val in all_ids:
    vector_data = vector_dict.get(id_val, {})
    fts_score = fts_dict.get(id_val, 0.0)
    vector_score = vector_data.get("vector_score", 0.0)

    name = vector_data.get("name", "")
    status = vector_data.get("status", "")

    # Name match boost
    if name == RUN_RESULTS_QUERY:
        name_boost = 1.0
    elif name and name.lower() in query_lower:
        name_boost = 0.5
    else:
        name_boost = 0.0

    # Status boost
    status_boost = 0.0
    if status in ("error", "fail") and ("fail" in query_lower or "error" in query_lower):
        status_boost = 1.0
    elif status == "warn" and "warn" in query_lower:
        status_boost = 1.0
    elif status == "success" and "success" in query_lower:
        status_boost = 0.5

    hybrid_score = (
        0.35 * vector_score + 0.35 * fts_score
        + 0.15 * name_boost + 0.15 * status_boost
    )

    compiled_code = vector_data.get("compiled_code", "") or ""
    merged_data.append({
        "name": name,
        "unique_id": vector_data.get("unique_id", ""),
        "resource_type": vector_data.get("resource_type", ""),
        "status": status,
        "message": vector_data.get("message", ""),
        "compiled_code_preview": compiled_code[:200] if compiled_code else "",
        "execution_time": vector_data.get("execution_time", 0.0),
        "vector_score": vector_score,
        "fts_score": fts_score,
        "name_boost": name_boost,
        "status_boost": status_boost,
        "hybrid_score": hybrid_score,
    })

merged_data.sort(key=lambda x: x["hybrid_score"], reverse=True)
top_results = merged_data[:10]

print(json.dumps(top_results, indent=2))
