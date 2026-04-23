"""
1. Loads catalog.json artifact data (table/column metadata)
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

JSON_PATH = "data/admin_output/catalog.json"
DB_PATH = "data/duck_db/dbt_catalog.duckdb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class CatalogIngestor:
    """Handles ingestion of catalog.json data into DuckDB with embeddings."""

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
        """Load catalog nodes from JSON file."""
        if not Path(json_path).exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        logger.info(f"Loading data from {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)

        nodes = list(data.get("nodes", {}).values())
        sources = list(data.get("sources", {}).values())
        all_entries = nodes + sources
        logger.info(
            f"Loaded {len(nodes)} catalog nodes and {len(sources)} catalog sources"
        )
        return all_entries

    def _get_columns_summary(self, node: Dict[str, Any]) -> str:
        """Extract column names and types as a summary string."""
        columns = node.get("columns", {})
        if not columns:
            return ""
        parts = []
        for col_name, col_data in columns.items():
            col_type = col_data.get("type", "") if isinstance(col_data, dict) else ""
            if col_type:
                parts.append(f"{col_name} ({col_type})")
            else:
                parts.append(col_name)
        return ", ".join(parts)

    def _get_row_count(self, node: Dict[str, Any]) -> Optional[int]:
        """Extract row count from stats if available."""
        stats = node.get("stats", {})
        if isinstance(stats, dict):
            row_count_stat = stats.get("row_count", {}) or stats.get(
                "num_rows", {}
            )
            if isinstance(row_count_stat, dict):
                value = row_count_stat.get("value")
                if value is not None:
                    try:
                        return int(float(value))
                    except (ValueError, TypeError):
                        pass
        return None

    def create_text_for_embedding(self, node: Dict[str, Any]) -> str:
        """Create a combined text representation for embedding."""
        metadata = node.get("metadata", {})
        name = metadata.get("name", "") if isinstance(metadata, dict) else ""
        node_type = metadata.get("type", "") if isinstance(metadata, dict) else ""
        comment = metadata.get("comment", "") if isinstance(metadata, dict) else ""
        columns_summary = self._get_columns_summary(node)

        text = f"Table: {name}. Type: {node_type}"
        if comment:
            text += f". Comment: {comment}"
        if columns_summary:
            text += f". Columns: {columns_summary}"
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

        self.conn.execute("DROP TABLE IF EXISTS catalog_nodes;")

        self.conn.execute("""
            CREATE TABLE catalog_nodes (
                id INTEGER PRIMARY KEY,
                unique_id VARCHAR,
                name VARCHAR,
                node_type VARCHAR,
                schema_name VARCHAR,
                database_name VARCHAR,
                owner VARCHAR,
                comment TEXT,
                columns_json TEXT,
                row_count BIGINT,
                embedding FLOAT[384],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        logger.info("Database tables created successfully")

    def insert_data(
        self, nodes: List[Dict[str, Any]], embeddings: List[List[float]]
    ):
        """Insert catalog node data and embeddings into DuckDB."""
        if self.conn is None:
            raise RuntimeError("Database connection not established")

        logger.info(f"Inserting {len(nodes)} catalog entries into database")

        data_to_insert = []
        for idx, (node, embedding) in enumerate(zip(nodes, embeddings)):
            metadata = node.get("metadata", {}) or {}
            columns = node.get("columns", {})
            columns_json = json.dumps(columns) if columns else ""

            data_to_insert.append((
                idx,
                node.get("unique_id", ""),
                metadata.get("name", "") if isinstance(metadata, dict) else "",
                metadata.get("type", "") if isinstance(metadata, dict) else "",
                metadata.get("schema", "") if isinstance(metadata, dict) else "",
                metadata.get("database", "") if isinstance(metadata, dict) else "",
                metadata.get("owner", "") if isinstance(metadata, dict) else "",
                metadata.get("comment", "") if isinstance(metadata, dict) else "",
                columns_json,
                self._get_row_count(node),
                embedding,
            ))

        self.conn.executemany(
            """
            INSERT INTO catalog_nodes (
                id, unique_id, name, node_type, schema_name, database_name,
                owner, comment, columns_json, row_count, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            data_to_insert,
        )

        logger.info("Data inserted successfully")

        logger.info("Creating FTS index for hybrid search")
        self.conn.execute("""
            PRAGMA create_fts_index(
                'catalog_nodes',
                'id',
                'name',
                'comment',
                'columns_json',
                overwrite=1
            );
        """)
        logger.info("FTS index created successfully")

    def create_indexes(self):
        """Create additional indexes for performance."""
        logger.info("Creating additional indexes")
        self.conn.execute("CREATE INDEX idx_cn_name ON catalog_nodes(name);")
        self.conn.execute(
            "CREATE INDEX idx_cn_unique_id ON catalog_nodes(unique_id);"
        )
        self.conn.execute(
            "CREATE INDEX idx_cn_node_type ON catalog_nodes(node_type);"
        )
        logger.info("Indexes created successfully")

    def print_statistics(self):
        """Print statistics about the ingested data."""
        result = self.conn.execute("SELECT COUNT(*) FROM catalog_nodes").fetchone()
        logger.info(f"Total catalog entries: {result[0]}")

        result = self.conn.execute(
            "SELECT node_type, COUNT(*) as cnt FROM catalog_nodes GROUP BY node_type ORDER BY cnt DESC"
        ).fetchall()
        for node_type, count in result:
            logger.info(f"  {node_type}: {count}")

        result = self.conn.execute(
            "SELECT array_length(embedding) FROM catalog_nodes LIMIT 1"
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
    ingestor = CatalogIngestor(db_path=DB_PATH, embedding_model=EMBEDDING_MODEL)
    ingestor.run(JSON_PATH)


if __name__ == "__main__":
    main()
