"""In-memory DuckDB artifact store.

Provides a generic loader driven by ArtifactConfig definitions, plus
read-only SQL query and full-text search capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Any

import duckdb

from src.mcp_server.artifacts import ALL_ARTIFACTS, ArtifactConfig

logger = logging.getLogger(__name__)

READONLY_BLOCKED = frozenset({
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "TRUNCATE", "REPLACE", "MERGE", "UPSERT", "COPY", "LOAD",
    "INSTALL", "ATTACH", "DETACH", "EXPORT", "IMPORT", "VACUUM",
    "PRAGMA",
})

MAX_RESULT_ROWS = 500


class ArtifactStore:
    """Manages an in-memory DuckDB database loaded with dbt artifacts."""

    def __init__(self, data_dir: str = "data/admin_output") -> None:
        self.data_dir = Path(data_dir)
        self.conn = duckdb.connect()
        self.conn.execute("INSTALL fts;")
        self.conn.execute("LOAD fts;")
        self._loaded_tables: set[str] = set()

    # ── Loading ──────────────────────────────────────────────────────

    def load_all(self) -> dict[str, int]:
        """Load all available artifact files. Returns {table_name: row_count}."""
        counts: dict[str, int] = {}
        for config in ALL_ARTIFACTS:
            path = self.data_dir / config.filename
            if path.exists():
                try:
                    count = self._load_artifact(config, path)
                    counts[config.table_name] = count
                except Exception:
                    logger.exception(f"Failed to load {config.filename}")
            else:
                logger.warning(
                    f"Skipping {config.filename}: not found at {path}"
                )
        return counts

    def _load_artifact(self, config: ArtifactConfig, path: Path) -> int:
        """Generic loader: parse JSON, create table, insert rows, build indexes."""
        with open(path) as f:
            data = json.load(f)

        entries = config.extract_entries(data)
        if not entries:
            logger.warning(f"No entries found in {config.filename}")
            return 0

        # Create table
        self.conn.execute(f"DROP TABLE IF EXISTS {config.table_name};")
        self.conn.execute(config.table_ddl)

        # Insert rows
        rows = [config.map_row(idx, entry) for idx, entry in enumerate(entries)]
        placeholders = ", ".join(["?"] * len(rows[0]))
        self.conn.executemany(
            f"INSERT INTO {config.table_name} VALUES ({placeholders})", rows
        )

        # FTS index
        fts_cols = ", ".join(f"'{c}'" for c in config.fts_columns)
        self.conn.execute(
            f"PRAGMA create_fts_index('{config.table_name}', 'id', "
            f"{fts_cols}, overwrite=1);"
        )

        # B-tree indexes
        for col in config.index_columns:
            idx_name = f"idx_{config.table_name[:2]}_{col}"
            self.conn.execute(
                f'CREATE INDEX {idx_name} ON {config.table_name}("{col}");'
            )

        self._loaded_tables.add(config.table_name)
        logger.info(f"Loaded {len(rows)} rows into {config.table_name}")
        return len(rows)

    # ── Query methods ────────────────────────────────────────────────

    def list_tables(self) -> list[dict[str, Any]]:
        """List all loaded artifact tables with row counts."""
        result = self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()

        tables = []
        for (table_name,) in result:
            if table_name.startswith("fts_"):
                continue
            count = self.conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
            tables.append({"table_name": table_name, "row_count": count})
        return tables

    def describe_table(self, table_name: str) -> list[dict[str, str]]:
        """Describe the schema of a loaded table."""
        self._validate_table_name(table_name)
        rows = self.conn.execute(f'DESCRIBE "{table_name}"').fetchall()
        return [
            {"column_name": row[0], "column_type": row[1]} for row in rows
        ]

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a read-only SQL query. Results capped at 500 rows."""
        first_word = sql.strip().split()[0].upper() if sql.strip() else ""
        if first_word in READONLY_BLOCKED:
            raise ValueError(
                f"Only read-only queries are allowed. "
                f"Blocked keyword: {first_word}"
            )

        result = self.conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchmany(MAX_RESULT_ROWS)
        return [
            {col: _serialize(val) for col, val in zip(columns, row)}
            for row in rows
        ]

    def search(
        self, table_name: str, query_text: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Full-text BM25 search on a loaded table."""
        self._validate_table_name(table_name)
        limit = min(limit, MAX_RESULT_ROWS)

        fts_table = f"fts_main_{table_name}"
        sql = f"""
            SELECT t.*, {fts_table}.match_bm25(t.id, ?) AS fts_score
            FROM "{table_name}" t
            WHERE {fts_table}.match_bm25(t.id, ?) IS NOT NULL
            ORDER BY fts_score DESC
            LIMIT ?
        """
        result = self.conn.execute(sql, [query_text, query_text, limit])
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [
            {col: _serialize(val) for col, val in zip(columns, row)}
            for row in rows
        ]

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self.conn:
            self.conn.close()
            logger.info("In-memory database closed")

    # ── Helpers ───────────────────────────────────────────────────────

    def _validate_table_name(self, table_name: str) -> None:
        if table_name not in self._loaded_tables:
            available = ", ".join(sorted(self._loaded_tables)) or "(none)"
            raise ValueError(
                f"Unknown table '{table_name}'. Available: {available}"
            )


def _serialize(val: Any) -> Any:
    """Ensure values are JSON-serializable."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    return str(val)
