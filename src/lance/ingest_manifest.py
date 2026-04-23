"""
1. Loads manifest.json artifact data (project nodes)
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

JSON_PATH = "data/admin_output/manifest.json"
DB_PATH = "data/lance_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ManifestIngestor:
    """Handles ingestion of manifest.json nodes into LanceDB with embeddings."""

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
        """Load manifest nodes from JSON file."""
        if not Path(json_path).exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        logger.info(f"Loading data from {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)

        nodes = list(data.get("nodes", {}).values())
        logger.info(f"Loaded {len(nodes)} manifest nodes")
        return nodes

    def _get_column_names(self, node: Dict[str, Any]) -> str:
        """Extract comma-separated column names from a node."""
        columns = node.get("columns", {})
        if isinstance(columns, dict):
            return ", ".join(columns.keys())
        return ""

    def create_text_for_embedding(self, node: Dict[str, Any]) -> str:
        """Create a combined text representation for embedding.

        Includes column names for semantic matches but excludes
        raw/compiled code (better served by FTS).
        """
        resource_type = node.get("resource_type", "")
        name = node.get("name", "")
        description = node.get("description", "") or ""
        column_names = self._get_column_names(node)

        text = f"{resource_type}: {name}. Description: {description}"
        if column_names:
            text += f". Columns: {column_names}"
        return text

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
        self, nodes: List[Dict[str, Any]], embeddings: List[List[float]]
    ):
        """Insert manifest node data and embeddings into LanceDB."""
        if self.db is None:
            raise RuntimeError("Database connection not established")

        logger.info(f"Inserting {len(nodes)} manifest nodes into database")

        data = []
        for idx, (node, embedding) in enumerate(zip(nodes, embeddings)):
            columns = node.get("columns", {})
            columns_json = json.dumps(columns) if columns else ""
            depends_on = node.get("depends_on", {})
            depends_on_json = json.dumps(depends_on) if depends_on else ""

            data.append({
                "id": idx,
                "unique_id": node.get("unique_id", ""),
                "resource_type": node.get("resource_type", ""),
                "name": node.get("name", ""),
                "description": node.get("description", "") or "",
                "raw_code": node.get("raw_code", "")
                or node.get("raw_sql", "")
                or "",
                "compiled_code": node.get("compiled_code", "")
                or node.get("compiled_sql", "")
                or "",
                "columns_json": columns_json,
                "depends_on_json": depends_on_json,
                "database_name": node.get("database", "") or "",
                "schema_name": node.get("schema", "") or "",
                "package_name": node.get("package_name", "") or "",
                "vector": embedding,
            })

        try:
            self.db.drop_table("manifest_nodes")
            logger.info("Dropped existing 'manifest_nodes' table")
        except Exception:
            pass

        table = self.db.create_table("manifest_nodes", data=data)
        logger.info("Data inserted successfully")

        logger.info("Creating FTS index for hybrid search")
        table.create_fts_index(
            ["name", "description", "raw_code", "compiled_code", "columns_json"],
            use_tantivy=True,
        )
        logger.info("FTS index created successfully")

    def print_statistics(self):
        """Print statistics about the ingested data."""
        if self.db is None:
            raise RuntimeError("Database connection not established")

        table = self.db.open_table("manifest_nodes")
        count = table.count_rows()
        logger.info(f"Total manifest nodes: {count}")

        arrow_table = table.to_arrow()
        if arrow_table.num_rows > 0:
            vector_col = arrow_table["vector"]
            first_vector = vector_col[0].as_py()
            logger.info(f"Embedding dimension: {len(first_vector)}")

    def run(self, json_path: str):
        """Run the complete ingestion pipeline."""
        try:
            self.load_embedding_model()
            nodes = self.load_json_data(json_path)
            texts = [self.create_text_for_embedding(n) for n in nodes]
            embeddings = self.generate_embeddings(texts)
            self.setup_database()
            self.insert_data(nodes, embeddings)
            self.print_statistics()
            logger.info("Ingestion completed successfully!")
        except Exception as e:
            logger.error(f"Error during ingestion: {e}", exc_info=True)
            raise


def main():
    ingestor = ManifestIngestor(db_path=DB_PATH, embedding_model=EMBEDDING_MODEL)
    ingestor.run(JSON_PATH)


if __name__ == "__main__":
    main()
