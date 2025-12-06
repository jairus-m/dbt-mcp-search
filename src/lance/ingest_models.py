"""
1. Loads dbt model data from JSON
2. Generates text embeddings
3. Stores data in LanceDB with both vector embeddings and FTS index
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

JSON_PATH = "data/gql_output/all_models.json"
DB_PATH = "data/lance_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ModelDataIngestor:
    """Handles ingestion of dbt model data into LanceDB with embeddings."""

    def __init__(
        self,
        db_path: str,
        embedding_model: str,
    ):
        """
        Initialize the ingestor.

        Args:
            db_path: Path to the LanceDB database directory
            embedding_model: Name of the sentence-transformers model to use
        """
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
        name = model.get("name", "")
        description = model.get("description", "")

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
        """Set up LanceDB database connection."""
        logger.info(f"Setting up database at {self.db_path}")

        db_dir = Path(self.db_path)
        db_dir.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(self.db_path)
        logger.info("Database connection established")

    def insert_data(self, models: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Insert models data and embeddings into LanceDB.

        Args:
            models: List of model dictionaries
            embeddings: List of embedding vectors
        """
        assert self.db is not None, "Database connection must be established"
        logger.info(f"Inserting {len(models)} models into database")

        # Prepare data for LanceDB
        data = []
        for idx, (model, embedding) in enumerate(zip(models, embeddings)):
            data.append({
                "id": idx,
                "name": model.get("name", ""),
                "unique_id": model.get("uniqueId", ""),
                "description": model.get("description", ""),
                "vector": embedding,  # LanceDB uses 'vector' as the default field name
            })

        try:
            self.db.drop_table("models")
            logger.info("Dropped existing 'models' table")
        except Exception:
            pass

        table = self.db.create_table("models", data=data)
        logger.info("Data inserted successfully")

        # Create FTS index for hybrid search
        logger.info("Creating FTS index for hybrid search")
        table.create_fts_index(["name", "description"], use_tantivy=True)
        logger.info("FTS index created successfully")

    def print_statistics(self):
        """Print statistics about the ingested data."""
        assert self.db is not None, "Database connection must be established"
        logger.info("=" * 60)
        logger.info("DATABASE STATISTICS")
        logger.info("=" * 60)

        table = self.db.open_table("models")

        count = table.count_rows()
        logger.info(f"Total models: {count}")

        arrow_table = table.to_arrow()

        if arrow_table.num_rows > 0:
            # 1st row embedding dim
            vector_col = arrow_table["vector"]
            first_vector = vector_col[0].as_py()
            embedding_dim = len(first_vector)
            logger.info(f"Embedding dimension: {embedding_dim}")

            desc_col = arrow_table["description"]
            desc_lengths = [len(desc.as_py()) if desc.as_py() else 0 for desc in desc_col]
            avg_desc_len = sum(desc_lengths) / len(desc_lengths)
            logger.info(f"Average description length: {avg_desc_len:.2f} characters")

        logger.info("\nSample records:")
        sample_size = min(3, arrow_table.num_rows)
        for i in range(sample_size):
            name = arrow_table["name"][i].as_py()
            description = arrow_table["description"][i].as_py()
            desc_preview = description[:100] if description else ""
            logger.info(f"  - {name}: {desc_preview}...")

        logger.info("=" * 60)

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

            self.print_statistics()

            logger.info("Ingestion completed successfully!")

        except Exception as e:
            logger.error(f"Error during ingestion: {e}", exc_info=True)
            raise


def main():
    ingestor = ModelDataIngestor(db_path=DB_PATH, embedding_model=EMBEDDING_MODEL)
    ingestor.run(JSON_PATH)


if __name__ == "__main__":
    main()
