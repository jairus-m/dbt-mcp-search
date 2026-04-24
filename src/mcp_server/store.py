"""In-memory DuckDB artifact store.

Provides a multi-table loader driven by extraction pipelines (one artifact
file can produce many tables), plus read-only SQL query and full-text search.
"""

import json
import logging
from pathlib import Path
from typing import Any

import duckdb

from src.mcp_server.artifacts import ARTIFACT_PIPELINE, TABLES, TableConfig

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
        # Create all tables up front
        for config in TABLES.values():
            self.conn.execute(f"DROP TABLE IF EXISTS {config.table_name};")
            self.conn.execute(config.table_ddl)

        counts: dict[str, int] = {}

        for filename, extractor in ARTIFACT_PIPELINE:
            path = self.data_dir / filename
            if not path.exists():
                logger.warning(f"Skipping {filename}: not found at {path}")
                continue

            try:
                with open(path) as f:
                    data = json.load(f)

                tables_data = extractor(data)

                for table_name, rows in tables_data.items():
                    if not rows:
                        continue

                    # Special: catalog column merge into node_columns
                    if table_name == "_node_columns_update":
                        self._merge_node_columns(rows)
                        continue

                    config = TABLES[table_name]
                    self._insert_rows(config, rows)
                    counts[table_name] = counts.get(table_name, 0) + len(rows)
                    self._loaded_tables.add(table_name)
                    logger.info(f"Loaded {len(rows)} rows into {table_name}")

            except Exception:
                logger.exception(f"Failed to load {filename}")

        # Build indexes after all data is loaded
        for table_name in self._loaded_tables:
            self._build_indexes(TABLES[table_name])

        return counts

    def _insert_rows(self, config: TableConfig, rows: list[tuple]) -> None:
        """Bulk insert rows into a table."""
        if not rows:
            return
        placeholders = ", ".join(["?"] * len(rows[0]))
        self.conn.executemany(
            f"INSERT INTO {config.table_name} VALUES ({placeholders})", rows
        )

    def _merge_node_columns(self, rows: list[tuple]) -> None:
        """Merge catalog column data into node_columns via batch update.

        Each row is (unique_id, column_name, column_index, catalog_type, catalog_comment).
        Uses a temp table + batch UPDATE/INSERT for performance.
        """
        if not rows:
            return

        self.conn.execute("""
            CREATE TEMP TABLE _catalog_cols (
                unique_id VARCHAR, column_name VARCHAR,
                column_index INTEGER, catalog_type VARCHAR,
                catalog_comment TEXT
            )
        """)
        placeholders = ", ".join(["?"] * 5)
        self.conn.executemany(
            f"INSERT INTO _catalog_cols VALUES ({placeholders})", rows
        )

        # Update existing node_columns with catalog data
        self.conn.execute("""
            UPDATE node_columns SET
                column_index = COALESCE(cc.column_index, node_columns.column_index),
                catalog_type = COALESCE(cc.catalog_type, node_columns.catalog_type),
                data_type = COALESCE(cc.catalog_type, node_columns.data_type),
                catalog_comment = COALESCE(cc.catalog_comment, node_columns.catalog_comment)
            FROM _catalog_cols cc
            WHERE node_columns.unique_id = cc.unique_id
              AND node_columns.column_name = cc.column_name
        """)

        # Insert catalog-only columns not already in node_columns
        row = self.conn.execute(
            "SELECT COALESCE(MAX(id), -1) + 1 FROM node_columns"
        ).fetchone()
        assert row is not None
        next_id = row[0]
        self.conn.execute(f"""
            INSERT INTO node_columns (id, unique_id, column_name, column_index,
                catalog_type, data_type, catalog_comment)
            SELECT {next_id} + row_number() OVER () - 1,
                   cc.unique_id, cc.column_name, cc.column_index,
                   cc.catalog_type, cc.catalog_type, cc.catalog_comment
            FROM _catalog_cols cc
            WHERE NOT EXISTS (
                SELECT 1 FROM node_columns nc
                WHERE nc.unique_id = cc.unique_id
                  AND nc.column_name = cc.column_name
            )
        """)

        self.conn.execute("DROP TABLE _catalog_cols")
        self._loaded_tables.add("node_columns")
        logger.info(f"Merged {len(rows)} catalog columns into node_columns")

    def _build_indexes(self, config: TableConfig) -> None:
        """Build FTS and B-tree indexes for a table."""
        if config.fts_columns:
            fts_cols = ", ".join(f"'{c}'" for c in config.fts_columns)
            self.conn.execute(
                f"PRAGMA create_fts_index('{config.table_name}', 'id', "
                f"{fts_cols}, overwrite=1);"
            )

        for col in config.index_columns:
            idx_name = f"idx_{config.table_name[:4]}_{col}"
            self.conn.execute(
                f'CREATE INDEX IF NOT EXISTS {idx_name} '
                f'ON {config.table_name}("{col}");'
            )

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
            count_row = self.conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()
            assert count_row is not None
            count = count_row[0]
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
        config = TABLES.get(table_name)
        if not config or not config.fts_columns:
            raise ValueError(f"Table '{table_name}' does not support full-text search")
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
