"""FastMCP server exposing dbt artifact query tools over an in-memory DuckDB.

Start with:  python -m src.mcp_server
Or via MCP:  mcp dev src/mcp_server/server.py
"""

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from src.mcp_server.store import ArtifactStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Start the server immediately, load artifacts in a background thread."""
    store = ArtifactStore()
    ready = threading.Event()

    def _load():
        try:
            counts = store.load_all()
            logger.info(f"Artifacts loaded: {counts}")
        except Exception:
            logger.exception("Failed to load artifacts")
        finally:
            ready.set()

    loader = threading.Thread(target=_load, daemon=True)
    loader.start()

    try:
        yield {"store": store, "ready": ready}
    finally:
        loader.join(timeout=5)
        store.close()


mcp = FastMCP(
    name="dbt-artifacts",
    instructions=(
        "Query dbt Cloud job run artifacts loaded in an in-memory DuckDB. "
        "Tables: nodes (models/tests/seeds/snapshots/sources with rich config), "
        "node_columns (column definitions with catalog types), "
        "edges (dependency graph: parent→child), "
        "test_metadata (test configs and attached nodes), "
        "exposures (BI dashboards/notebooks), metrics (metric definitions), "
        "groups (access groups), macros (macro SQL and metadata), "
        "catalog_tables (warehouse table metadata), "
        "catalog_stats (table statistics like row counts), "
        "invocations (dbt run metadata), run_results (per-node execution status), "
        "source_freshness (source staleness with criteria). "
        "Use list_tables first, then describe_table to understand schemas, "
        "then query or search to find what you need. "
        "Use edges to trace dependencies (e.g. impact analysis). "
        "Join nodes with node_columns on unique_id for full column info."
    ),
    lifespan=lifespan,
)


def _get_store(ctx: Context) -> ArtifactStore:
    lc = ctx.request_context.lifespan_context
    ready: threading.Event = lc["ready"]
    if not ready.is_set():
        ready.wait(timeout=120)
        if not ready.is_set():
            raise RuntimeError("Artifact loading timed out")
    return lc["store"]


@mcp.tool(
    description=(
        "List all loaded dbt artifact tables and their row counts. "
        "Call this first to see what data is available."
    )
)
def list_tables(ctx: Context) -> list[dict[str, Any]]:
    return _get_store(ctx).list_tables()


@mcp.tool(
    description=(
        "Show column names and types for a table. "
        "Use after list_tables to understand the schema before querying."
    )
)
def describe_table(table_name: str, ctx: Context) -> list[dict[str, str]]:
    return _get_store(ctx).describe_table(table_name)


@mcp.tool(
    description=(
        "Execute a read-only SQL query against the in-memory database. "
        "Supports JOINs, aggregations, window functions, CTEs, etc. "
        "Results capped at 500 rows."
    )
)
def query(sql: str, ctx: Context) -> list[dict[str, Any]]:
    return _get_store(ctx).query(sql)


@mcp.tool(
    description=(
        "Full-text keyword search (BM25) on a specific table. "
        "Searches across indexed text columns (names, descriptions, code, etc). "
        "Use when you need fuzzy keyword matching rather than exact SQL filters."
    )
)
def search(
    table_name: str, search_query: str, limit: int = 20, ctx: Context = None
) -> list[dict[str, Any]]:
    return _get_store(ctx).search(table_name, search_query, limit)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
