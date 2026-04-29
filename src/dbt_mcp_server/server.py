"""Standalone FastMCP server for development and testing.

Reads Admin API credentials from environment variables and registers
artifact search tools. Run with ``python -m src.dbt_mcp_server``.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from dbt_mcp.config.config_providers import AdminApiConfig, StaticConfigProvider
from dbt_mcp.config.headers import AdminApiHeadersProvider
from dbt_mcp.oauth.token_provider import StaticTokenProvider

from src.dbt_mcp_server.store import ArtifactStore
from src.dbt_mcp_server.tools import register_artifact_search_tools

logger = logging.getLogger(__name__)


def _build_config_provider() -> StaticConfigProvider[AdminApiConfig]:
    """Build an Admin API config provider from environment variables.

    Required env vars:
        DBT_HOST: e.g. "https://cloud.getdbt.com"
        DBT_TOKEN: Service token or PAT
        DBT_ACCOUNT_ID: Numeric account ID
    """
    host = os.environ.get("DBT_HOST")
    token = os.environ.get("DBT_TOKEN")
    account_id_str = os.environ.get("DBT_ACCOUNT_ID")

    if not all([host, token, account_id_str]):
        missing = []
        if not host:
            missing.append("DBT_HOST")
        if not token:
            missing.append("DBT_TOKEN")
        if not account_id_str:
            missing.append("DBT_ACCOUNT_ID")
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Set DBT_HOST, DBT_TOKEN, and DBT_ACCOUNT_ID to use the artifact search server."
        )

    token_provider = StaticTokenProvider(token)
    headers_provider = AdminApiHeadersProvider(token_provider)

    config = AdminApiConfig(
        url=host,  # type: ignore[arg-type]
        headers_provider=headers_provider,
        account_id=int(account_id_str),  # type: ignore[arg-type]
    )

    return StaticConfigProvider(config)


INSTRUCTIONS = """Query dbt Cloud job run artifacts loaded in an in-memory DuckDB.

Workflow:
1. Use load_artifacts to fetch artifacts for a run — clears any previous run first.
   Pass artifact_types to load a subset (e.g. ["manifest.json", "run_results.json"]).
2. Use list_artifact_tables to see what's loaded
3. Use describe_artifact_table to understand schemas
4. Use query_artifacts for precise SQL queries or search_artifacts for keyword search

One run at a time: loading a new run automatically replaces the previous one.
Use clear_store to explicitly wipe the store without loading a new run.

Tables (after loading all 4 artifacts): nodes, node_columns, edges, test_metadata,
exposures, metrics, groups, macros, catalog_tables, catalog_stats, invocations,
run_results, source_freshness.

Use edges to trace dependencies (impact analysis). Edge types: ref (model→model),
exposure_ref (exposure depends on model), metric_ref (metric depends on model).
Join nodes with node_columns on unique_id for full column info.
"""


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Server lifespan: build config, create store, register tools."""
    config_provider = _build_config_provider()
    store = ArtifactStore()
    register_artifact_search_tools(server, config_provider, store=store)
    try:
        yield {"store": store}
    finally:
        store.close()


mcp = FastMCP(
    name="dbt-artifact-search",
    instructions=INSTRUCTIONS,
    lifespan=lifespan,
)


def main() -> None:
    """Entry point for the standalone MCP server."""
    mcp.run(transport="stdio")
