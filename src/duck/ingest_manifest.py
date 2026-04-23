"""
1. Loads manifest.json artifact data (project nodes)
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

JSON_PATH = "data/admin_output/manifest.json"
DB_PATH = "data/duck_db/dbt_manifest.duckdb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ManifestIngestor:
    """Handles ingestion of manifest.json nodes into DuckDB with embeddings."""

    def __init__(self, db_path: str, embedding_model: str):
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
        """Set up DuckDB database and create tables."""
        logger.info(f"Setting up database at {self.db_path}")

        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self.conn = duckdb.connect(self.db_path)

        self.conn.execute("INSTALL fts;")
        self.conn.execute("LOAD fts;")

        self.conn.execute("DROP TABLE IF EXISTS manifest_nodes;")

        self.conn.execute("""
            CREATE TABLE manifest_nodes (
                id INTEGER PRIMARY KEY,
                unique_id VARCHAR,
                resource_type VARCHAR,
                name VARCHAR,
                description TEXT,
                raw_code TEXT,
                compiled_code TEXT,
                columns_json TEXT,
                depends_on_json TEXT,
                database_name VARCHAR,
                schema_name VARCHAR,
                package_name VARCHAR,
                embedding FLOAT[384],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        logger.info("Database tables created successfully")

    def insert_data(
        self, nodes: List[Dict[str, Any]], embeddings: List[List[float]]
    ):
        """Insert manifest node data and embeddings into DuckDB."""
        if self.conn is None:
            raise RuntimeError("Database connection not established")

        logger.info(f"Inserting {len(nodes)} manifest nodes into database")

        data_to_insert = []
        for idx, (node, embedding) in enumerate(zip(nodes, embeddings)):
            columns = node.get("columns", {})
            columns_json = json.dumps(columns) if columns else ""
            depends_on = node.get("depends_on", {})
            depends_on_json = json.dumps(depends_on) if depends_on else ""

            data_to_insert.append((
                idx,
                node.get("unique_id", ""),
                node.get("resource_type", ""),
                node.get("name", ""),
                node.get("description", "") or "",
                node.get("raw_code", "") or node.get("raw_sql", "") or "",
                node.get("compiled_code", "") or node.get("compiled_sql", "") or "",
                columns_json,
                depends_on_json,
                node.get("database", "") or "",
                node.get("schema", "") or "",
                node.get("package_name", "") or "",
                embedding,
            ))

        self.conn.executemany(
            """
            INSERT INTO manifest_nodes (
                id, unique_id, resource_type, name, description,
                raw_code, compiled_code, columns_json, depends_on_json,
                database_name, schema_name, package_name, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            data_to_insert,
        )

        logger.info("Data inserted successfully")

        logger.info("Creating FTS index for hybrid search")
        self.conn.execute("""
            PRAGMA create_fts_index(
                'manifest_nodes',
                'id',
                'name',
                'description',
                'raw_code',
                'compiled_code',
                'columns_json',
                overwrite=1
            );
        """)
        logger.info("FTS index created successfully")

    def create_indexes(self):
        """Create additional indexes for performance."""
        logger.info("Creating additional indexes")
        self.conn.execute("CREATE INDEX idx_mn_name ON manifest_nodes(name);")
        self.conn.execute(
            "CREATE INDEX idx_mn_unique_id ON manifest_nodes(unique_id);"
        )
        self.conn.execute(
            "CREATE INDEX idx_mn_resource_type ON manifest_nodes(resource_type);"
        )
        logger.info("Indexes created successfully")

    def print_statistics(self):
        """Print statistics about the ingested data."""
        result = self.conn.execute("SELECT COUNT(*) FROM manifest_nodes").fetchone()
        logger.info(f"Total manifest nodes: {result[0]}")

        result = self.conn.execute(
            "SELECT resource_type, COUNT(*) as cnt FROM manifest_nodes GROUP BY resource_type ORDER BY cnt DESC"
        ).fetchall()
        for resource_type, count in result:
            logger.info(f"  {resource_type}: {count}")

        result = self.conn.execute(
            "SELECT array_length(embedding) FROM manifest_nodes LIMIT 1"
        ).fetchone()
        if result:
            logger.info(f"Embedding dimension: {result[0]}")

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def run(self, json_path: str):
        """Run the complete ingestion pipeline."""
        try:
            self.load_embedding_model()
            nodes = self.load_json_data(json_path)
            texts = [self.create_text_for_embedding(n) for n in nodes]
            embeddings = self.generate_embeddings(texts)
            self.setup_database()
            self.insert_data(nodes, embeddings)
            self.create_indexes()
            self.print_statistics()
            logger.info("Ingestion completed successfully!")
        except Exception as e:
            logger.error(f"Error during ingestion: {e}", exc_info=True)
            raise
        finally:
            self.close()


def main():
    ingestor = ManifestIngestor(db_path=DB_PATH, embedding_model=EMBEDDING_MODEL)
    ingestor.run(JSON_PATH)


if __name__ == "__main__":
    main()
