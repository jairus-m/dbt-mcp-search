"""
1. Loads sources.json artifact data (source freshness results)
2. Generates text embeddings
3. Stores data in LanceDB with both vector embeddings and FTS index
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import lancedb
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

JSON_PATH = "data/admin_output/sources.json"
DB_PATH = "data/lance_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class SourceFreshnessIngestor:
    """Handles ingestion of sources.json data into LanceDB with embeddings."""

    def __init__(self, db_path: str, embedding_model: str):
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.model: Optional[SentenceTransformer] = None
        self.db: Optional[lancedb.DBConnection] = None

    def load_embedding_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(self.embedding_model_name)
        logger.info("Embedding model loaded successfully")

    def load_json_data(self, json_path: str) -> List[Dict[str, Any]]:
        """Load source freshness data from JSON file."""
        if not Path(json_path).exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        logger.info(f"Loading data from {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)

        results = data.get("results", [])
        logger.info(f"Loaded {len(results)} source freshness results")
        return results

    def _parse_unique_id(self, unique_id: str) -> tuple[str, str]:
        """Extract source_name and table_name from unique_id.

        Format: 'source.project.source_name.table_name'
        """
        parts = unique_id.split(".")
        source_name = parts[2] if len(parts) > 2 else ""
        table_name = parts[3] if len(parts) > 3 else ""
        return source_name, table_name

    def create_text_for_embedding(self, result: Dict[str, Any]) -> str:
        """Create a combined text representation for embedding."""
        unique_id = result.get("unique_id", "")
        source_name, table_name = self._parse_unique_id(unique_id)
        status = result.get("status", "")
        max_loaded_at = result.get("max_loaded_at", "") or "unknown"

        return (
            f"Source: {source_name}.{table_name}. "
            f"Freshness status: {status}. "
            f"Last loaded: {max_loaded_at}"
        )

    def generate_embeddings(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")

        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        return embeddings.tolist()

    def setup_database(self):
        """Set up LanceDB database connection."""
        logger.info(f"Setting up database at {self.db_path}")

        db_dir = Path(self.db_path)
        db_dir.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(self.db_path)
        logger.info("Database connection established")

    def insert_data(
        self, results: List[Dict[str, Any]], embeddings: List[List[float]]
    ):
        """Insert source freshness data and embeddings into LanceDB."""
        if self.db is None:
            raise RuntimeError("Database connection not established")

        logger.info(f"Inserting {len(results)} source freshness results into database")

        data = []
        for idx, (result, embedding) in enumerate(zip(results, embeddings)):
            unique_id = result.get("unique_id", "")
            source_name, table_name = self._parse_unique_id(unique_id)
            criteria = result.get("criteria")
            criteria_str = json.dumps(criteria) if criteria else ""

            data.append({
                "id": idx,
                "unique_id": unique_id,
                "source_name": source_name,
                "table_name": table_name,
                "status": result.get("status", ""),
                "max_loaded_at": result.get("max_loaded_at", "") or "",
                "snapshotted_at": result.get("snapshotted_at", "") or "",
                "max_loaded_at_time_ago_in_s": result.get(
                    "max_loaded_at_time_ago_in_s", 0.0
                )
                or 0.0,
                "criteria": criteria_str,
                "vector": embedding,
            })

        try:
            self.db.drop_table("source_freshness")
            logger.info("Dropped existing 'source_freshness' table")
        except Exception:
            pass

        table = self.db.create_table("source_freshness", data=data)
        logger.info("Data inserted successfully")

        logger.info("Creating FTS index for hybrid search")
        table.create_fts_index(
            ["source_name", "table_name", "status"], use_tantivy=True
        )
        logger.info("FTS index created successfully")

    def print_statistics(self):
        """Print statistics about the ingested data."""
        if self.db is None:
            raise RuntimeError("Database connection not established")

        table = self.db.open_table("source_freshness")
        count = table.count_rows()
        logger.info(f"Total source freshness results: {count}")

        arrow_table = table.to_arrow()
        if arrow_table.num_rows > 0:
            vector_col = arrow_table["vector"]
            first_vector = vector_col[0].as_py()
            logger.info(f"Embedding dimension: {len(first_vector)}")

    def run(self, json_path: str):
        """Run the complete ingestion pipeline."""
        try:
            self.load_embedding_model()
            results = self.load_json_data(json_path)
            texts = [self.create_text_for_embedding(r) for r in results]
            embeddings = self.generate_embeddings(texts)
            self.setup_database()
            self.insert_data(results, embeddings)
            self.print_statistics()
            logger.info("Ingestion completed successfully!")
        except Exception as e:
            logger.error(f"Error during ingestion: {e}", exc_info=True)
            raise


def main():
    ingestor = SourceFreshnessIngestor(
        db_path=DB_PATH, embedding_model=EMBEDDING_MODEL
    )
    ingestor.run(JSON_PATH)


if __name__ == "__main__":
    main()
