"""
1. Loads run_results.json artifact data
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

JSON_PATH = "data/admin_output/run_results.json"
DB_PATH = "data/lance_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class RunResultsIngestor:
    """Handles ingestion of run_results.json data into LanceDB with embeddings."""

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
        """Load run results data from JSON file."""
        if not Path(json_path).exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        logger.info(f"Loading data from {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)

        results = data.get("results", [])
        logger.info(f"Loaded {len(results)} run results")
        return results

    def _parse_unique_id(self, unique_id: str) -> tuple[str, str]:
        """Extract resource_type and name from unique_id like 'model.project.name'."""
        parts = unique_id.split(".")
        resource_type = parts[0] if parts else ""
        name = parts[-1] if parts else ""
        return resource_type, name

    def create_text_for_embedding(self, result: Dict[str, Any]) -> str:
        """Create a combined text representation for embedding.

        Excludes compiled_code since SQL distorts semantic embeddings
        and is better served by FTS.
        """
        unique_id = result.get("unique_id", "")
        resource_type, name = self._parse_unique_id(unique_id)
        status = result.get("status", "")
        message = result.get("message", "") or ""

        return f"{resource_type}: {name}. Status: {status}. Message: {message}"

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
        """Insert run results data and embeddings into LanceDB."""
        if self.db is None:
            raise RuntimeError("Database connection not established")

        logger.info(f"Inserting {len(results)} run results into database")

        data = []
        for idx, (result, embedding) in enumerate(zip(results, embeddings)):
            unique_id = result.get("unique_id", "")
            resource_type, name = self._parse_unique_id(unique_id)

            data.append({
                "id": idx,
                "unique_id": unique_id,
                "resource_type": resource_type,
                "name": name,
                "status": result.get("status", ""),
                "message": result.get("message", "") or "",
                "relation_name": result.get("relation_name", "") or "",
                "compiled_code": result.get("compiled_code", "") or "",
                "execution_time": result.get("execution_time", 0.0) or 0.0,
                "vector": embedding,
            })

        try:
            self.db.drop_table("run_results")
            logger.info("Dropped existing 'run_results' table")
        except Exception:
            pass

        table = self.db.create_table("run_results", data=data)
        logger.info("Data inserted successfully")

        logger.info("Creating FTS index for hybrid search")
        table.create_fts_index(
            ["name", "status", "message", "compiled_code"], use_tantivy=True
        )
        logger.info("FTS index created successfully")

    def print_statistics(self):
        """Print statistics about the ingested data."""
        if self.db is None:
            raise RuntimeError("Database connection not established")

        table = self.db.open_table("run_results")
        count = table.count_rows()
        logger.info(f"Total run results: {count}")

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
    ingestor = RunResultsIngestor(db_path=DB_PATH, embedding_model=EMBEDDING_MODEL)
    ingestor.run(JSON_PATH)


if __name__ == "__main__":
    main()
