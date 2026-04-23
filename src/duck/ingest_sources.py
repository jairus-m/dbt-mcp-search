"""
1. Loads sources.json artifact data (source freshness results)
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

JSON_PATH = "data/admin_output/sources.json"
DB_PATH = "data/duck_db/dbt_sources.duckdb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class SourceFreshnessIngestor:
    """Handles ingestion of sources.json data into DuckDB with embeddings."""

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
        """Set up DuckDB database and create tables."""
        logger.info(f"Setting up database at {self.db_path}")

        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self.conn = duckdb.connect(self.db_path)

        self.conn.execute("INSTALL fts;")
        self.conn.execute("LOAD fts;")

        self.conn.execute("DROP TABLE IF EXISTS source_freshness;")

        self.conn.execute("""
            CREATE TABLE source_freshness (
                id INTEGER PRIMARY KEY,
                unique_id VARCHAR,
                source_name VARCHAR,
                table_name VARCHAR,
                status VARCHAR,
                max_loaded_at VARCHAR,
                snapshotted_at VARCHAR,
                max_loaded_at_time_ago_in_s FLOAT,
                criteria TEXT,
                embedding FLOAT[384],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        logger.info("Database tables created successfully")

    def insert_data(
        self, results: List[Dict[str, Any]], embeddings: List[List[float]]
    ):
        """Insert source freshness data and embeddings into DuckDB."""
        if self.conn is None:
            raise RuntimeError("Database connection not established")

        logger.info(f"Inserting {len(results)} source freshness results into database")

        data_to_insert = []
        for idx, (result, embedding) in enumerate(zip(results, embeddings)):
            unique_id = result.get("unique_id", "")
            source_name, table_name = self._parse_unique_id(unique_id)
            criteria = result.get("criteria")
            criteria_str = json.dumps(criteria) if criteria else ""

            data_to_insert.append((
                idx,
                unique_id,
                source_name,
                table_name,
                result.get("status", ""),
                result.get("max_loaded_at", ""),
                result.get("snapshotted_at", ""),
                result.get("max_loaded_at_time_ago_in_s", 0.0),
                criteria_str,
                embedding,
            ))

        self.conn.executemany(
            """
            INSERT INTO source_freshness (
                id, unique_id, source_name, table_name, status,
                max_loaded_at, snapshotted_at, max_loaded_at_time_ago_in_s,
                criteria, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            data_to_insert,
        )

        logger.info("Data inserted successfully")

        logger.info("Creating FTS index for hybrid search")
        self.conn.execute("""
            PRAGMA create_fts_index(
                'source_freshness',
                'id',
                'source_name',
                'table_name',
                'status',
                overwrite=1
            );
        """)
        logger.info("FTS index created successfully")

    def create_indexes(self):
        """Create additional indexes for performance."""
        logger.info("Creating additional indexes")
        self.conn.execute(
            "CREATE INDEX idx_sf_source_name ON source_freshness(source_name);"
        )
        self.conn.execute(
            "CREATE INDEX idx_sf_table_name ON source_freshness(table_name);"
        )
        self.conn.execute("CREATE INDEX idx_sf_status ON source_freshness(status);")
        logger.info("Indexes created successfully")

    def print_statistics(self):
        """Print statistics about the ingested data."""
        result = self.conn.execute(
            "SELECT COUNT(*) FROM source_freshness"
        ).fetchone()
        logger.info(f"Total source freshness results: {result[0]}")

        result = self.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM source_freshness GROUP BY status ORDER BY cnt DESC"
        ).fetchall()
        for status, count in result:
            logger.info(f"  {status}: {count}")

        result = self.conn.execute(
            "SELECT array_length(embedding) FROM source_freshness LIMIT 1"
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
            results = self.load_json_data(json_path)
            texts = [self.create_text_for_embedding(r) for r in results]
            embeddings = self.generate_embeddings(texts)
            self.setup_database()
            self.insert_data(results, embeddings)
            self.create_indexes()
            self.print_statistics()
            logger.info("Ingestion completed successfully!")
        except Exception as e:
            logger.error(f"Error during ingestion: {e}", exc_info=True)
            raise
        finally:
            self.close()


def main():
    ingestor = SourceFreshnessIngestor(
        db_path=DB_PATH, embedding_model=EMBEDDING_MODEL
    )
    ingestor.run(JSON_PATH)


if __name__ == "__main__":
    main()
