"""
1. Loads dbt model data from JSON
2. Generates text embeddings
3. Stores data in DuckDB with both vector embeddings and FTS index
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import duckdb
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

JSON_PATH = "data/gql_output/all_models.json"
DB_PATH = "data/duck_db/dbt_models.duckdb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ModelDataIngestor:
    """Handles ingestion of dbt model data into DuckDB with embeddings."""

    def __init__(
        self,
        db_path: str,
        embedding_model: str,
    ):
        """
        Initialize the ingestor.

        Args:
            db_path: Path to the DuckDB database file
            embedding_model: Name of the sentence-transformers model to use
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.model: Optional[SentenceTransformer] = None
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def load_embedding_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(self.embedding_model_name)
        logger.info("Embedding model loaded successfully")

    def load_json_data(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load dbt models data from JSON file.

        Args:
            json_path: Path to the JSON file

        Returns:
            List of model dictionaries
        """
        logger.info(f"Loading data from {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} models")
        return data

    def create_text_for_embedding(self, model: Dict[str, Any]) -> str:
        """
        Create a combined text representation for embedding.

        Args:
            model: Model dictionary with name, uniqueId, and description

        Returns:
            Combined text string
        """
        # Combine name + description for embessing (arbitrary choice)
        name = model.get("name", "")
        description = model.get("description", "")

        # Format: "Model: {name}. Description: {description}"
        text = f"Model: {name}. Description: {description}"
        return text

    def generate_embeddings(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            List of embedding vectors
        """
        assert self.model is not None, (
            "Model must be loaded before generating embeddings"
        )
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        logger.info("Embeddings generated successfully")
        return embeddings.tolist()

    def setup_database(self):
        """Set up DuckDB database and create tables."""
        logger.info(f"Setting up database at {self.db_path}")

        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self.conn = duckdb.connect(self.db_path)

        self.conn.execute("INSTALL fts;")
        self.conn.execute("LOAD fts;")

        self.conn.execute("DROP TABLE IF EXISTS models;")
        self.conn.execute("DROP TABLE IF EXISTS fts_models;")

        # table models == main table w/ embeddings/data
        self.conn.execute("""
            CREATE TABLE models (
                id INTEGER PRIMARY KEY,
                name VARCHAR,
                unique_id VARCHAR,
                description TEXT,
                embedding FLOAT[384],  -- all-MiniLM-L6-v2 produces 384-dim vectors
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        logger.info("Database tables created successfully")

    def insert_data(self, models: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Insert models data and embeddings into DuckDB.

        Args:
            models: List of model dictionaries
            embeddings: List of embedding vectors
        """
        assert self.conn is not None, "Database connection must be established"
        logger.info(f"Inserting {len(models)} models into database")

        data_to_insert = []
        for idx, (model, embedding) in enumerate(zip(models, embeddings)):
            data_to_insert.append(
                {
                    "id": idx,
                    "name": model.get("name", ""),
                    "unique_id": model.get("uniqueId", ""),
                    "description": model.get("description", ""),
                    "embedding": embedding,
                }
            )

        self.conn.executemany(
            """
            INSERT INTO models (id, name, unique_id, description, embedding)
            VALUES (?, ?, ?, ?, ?)
        """,
            [
                (d["id"], d["name"], d["unique_id"], d["description"], d["embedding"])
                for d in data_to_insert
            ],
        )

        logger.info("Data inserted successfully")

        logger.info("Creating FTS index for hybrid search")
        self.conn.execute("""
            PRAGMA create_fts_index(
                'models',
                'id',
                'name',
                'description',
                overwrite=1
            );
        """)
        logger.info("FTS index created successfully")

    def create_indexes(self):
        """Create additional indexes for performance."""
        assert self.conn is not None, "Database connection must be established"
        logger.info("Creating additional indexes")

        # Index on name for faster lookups
        self.conn.execute("CREATE INDEX idx_name ON models(name);")

        # Index on unique_id for faster lookups (when to choose what field to index?)
        self.conn.execute("CREATE INDEX idx_unique_id ON models(unique_id);")

        logger.info("Indexes created successfully")

    def print_statistics(self):
        """Print statistics about the ingested data."""
        assert self.conn is not None, "Database connection must be established"
        logger.info("=" * 60)
        logger.info("DATABASE STATISTICS")
        logger.info("=" * 60)

        result = self.conn.execute("SELECT COUNT(*) FROM models").fetchone()
        assert result is not None
        logger.info(f"Total models: {result[0]}")

        result = self.conn.execute(
            "SELECT array_length(embedding) FROM models LIMIT 1"
        ).fetchone()
        assert result is not None
        logger.info(f"Embedding dimension: {result[0]}")

        result = self.conn.execute(
            "SELECT AVG(LENGTH(description)) FROM models"
        ).fetchone()
        assert result is not None
        logger.info(f"Average description length: {result[0]:.2f} characters")

        logger.info("\nSample records:")
        results = self.conn.execute("""
            SELECT name, LEFT(description, 100) as desc_preview
            FROM models
            LIMIT 3
        """).fetchall()

        for name, desc in results:
            logger.info(f"  - {name}: {desc}...")

        logger.info("=" * 60)

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def run(self, json_path: str):
        """
        Run the complete ingestion pipeline.

        Args:
            json_path: Path to the JSON file containing model data
        """
        try:
            self.load_embedding_model()

            models = self.load_json_data(json_path)

            # Create combined texts for embedding
            texts = [self.create_text_for_embedding(model) for model in models]

            embeddings = self.generate_embeddings(texts)

            self.setup_database()

            self.insert_data(models, embeddings)

            self.create_indexes()

            self.print_statistics()

            logger.info("Ingestion completed successfully!")

        except Exception as e:
            logger.error(f"Error during ingestion: {e}", exc_info=True)
            raise
        finally:
            self.close()


def main():
    ingestor = ModelDataIngestor(db_path=DB_PATH, embedding_model=EMBEDDING_MODEL)
    ingestor.run(JSON_PATH)


if __name__ == "__main__":
    main()
