"""FastMCP server exposing dbt artifact query tools over an in-memory DuckDB.

Start with:  python -m src.mcp_server
Or via MCP:  mcp dev src/mcp_server/server.py
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from src.mcp_server.store import ArtifactStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Load artifacts into in-memory DuckDB on startup."""
    store = ArtifactStore()
    counts = store.load_all()
    logger.info(f"Artifacts loaded: {counts}")
    try:
        yield store
    finally:
        store.close()


mcp = FastMCP(
    name="dbt-artifacts",
    instructions=(
        "Query dbt Cloud job run artifacts loaded in an in-memory DuckDB. "
        "Tables: manifest_nodes (project nodes with SQL code and columns), "
        "catalog_nodes (table/column metadata), run_results (execution status), "
        "source_freshness (source staleness). "
        "Use list_tables first, then describe_table to understand schemas, "
        "then query or search to find what you need."
    ),
    lifespan=lifespan,
)


def _get_store(ctx: Context) -> ArtifactStore:
    return ctx.request_context.lifespan_context


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
