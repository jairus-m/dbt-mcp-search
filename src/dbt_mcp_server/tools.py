"""Artifact search tool definitions, context, and registration.

Mirrors dbt_mcp/dbt_admin/tools.py: one ToolContext dataclass, N tool
functions decorated with the dbt-mcp tool decorator, a flat tool list,
and a register function.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from dbt_mcp.config.config_providers import AdminApiConfig, ConfigProvider
from dbt_mcp.dbt_admin.client import DbtAdminAPIClient
from dbt_mcp.tools.register import generic_register_tools
from dbt_mcp.tools.toolsets import Toolset

from src.dbt_mcp_server.artifacts.extractors import ArtifactType
from src.dbt_mcp_server.client import ArtifactFetchClient
from src.dbt_mcp_server.prompts import get_prompt
from src.dbt_mcp_server.store import ArtifactStore
from src.dbt_mcp_server.tool_names import (
    TOOL_TO_TOOLSET,
    ArtifactSearchToolName,
    ArtifactSearchToolset,
    artifact_search_tool,
)

logger = logging.getLogger(__name__)


# ── Context ──────────────────────────────────────────────────────────────


@dataclass
class ArtifactSearchToolContext:
    """Context for artifact search tools.

    Holds both the Admin API client (for on-demand artifact fetching)
    and the ArtifactStore (in-memory DuckDB for querying).

    Mirrors ``AdminToolContext`` from ``dbt_mcp.dbt_admin.tools``.
    """

    admin_client: DbtAdminAPIClient
    fetch_client: ArtifactFetchClient
    admin_api_config_provider: ConfigProvider[AdminApiConfig]
    store: ArtifactStore

    def __init__(
        self,
        admin_api_config_provider: ConfigProvider[AdminApiConfig],
        store: ArtifactStore,
    ):
        self.admin_api_config_provider = admin_api_config_provider
        self.admin_client = DbtAdminAPIClient(admin_api_config_provider)
        self.fetch_client = ArtifactFetchClient(self.admin_client)
        self.store = store


# ── Tools ────────────────────────────────────────────────────────────────


@artifact_search_tool(
    description=(
        "Load dbt artifacts for a run into the searchable in-memory database. "
        "Clears any previously loaded run first — only one run is held at a time. "
        "Defaults to all four artifact types; pass artifact_types to load a subset "
        "(e.g. [\"manifest.json\", \"run_results.json\"] for a quick failure diagnosis). "
        "Indexes are built once after all artifacts are loaded for efficiency."
    ),
    title="Load Artifacts",
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
)
async def load_artifacts(
    context: ArtifactSearchToolContext,
    run_id: int = Field(description="The dbt Cloud job run ID"),
    artifact_types: list[str] = Field(
        default=["manifest.json", "catalog.json", "run_results.json", "sources.json"],
        description=(
            "Which artifacts to load. Defaults to all four: "
            "manifest.json, catalog.json, run_results.json, sources.json"
        ),
    ),
) -> dict[str, Any]:
    """Clear the store, fetch artifacts from the API, and load them into DuckDB."""
    config = await context.admin_api_config_provider.get_config()

    # Clear any previously loaded run before inserting
    context.store.reset()

    all_tables: dict[str, int] = {}
    all_timing: dict[str, dict[str, int]] = {}
    errors: dict[str, str] = {}

    for artifact_str in artifact_types:
        try:
            art_type = ArtifactType(artifact_str)
        except ValueError:
            errors[artifact_str] = f"Invalid artifact type: {artifact_str}"
            continue
        try:
            data = await context.fetch_client.fetch_artifact(
                account_id=config.account_id,
                run_id=run_id,
                artifact_path=art_type.value,
            )
            result = context.store.load_artifact(run_id, art_type, data, reindex=False)
            all_tables.update(result["tables"])
            all_timing[artifact_str] = result["timing"]
        except Exception as e:
            errors[artifact_str] = str(e)

    # Build all indexes once
    t_index = time.perf_counter()
    indexed = context.store.build_all_indexes()
    index_build_ms = round((time.perf_counter() - t_index) * 1000)

    return {
        "status": "loaded",
        "run_id": run_id,
        "tables_loaded": all_tables,
        "indexes_built": indexed,
        "timing_ms": all_timing,
        "index_build_ms": index_build_ms,
        "errors": errors,
    }


@artifact_search_tool(
    description=(
        "Drop all loaded artifact tables and reset the in-memory store to empty. "
        "Use this to free memory or wipe the store without loading a new run. "
        "Note: load_artifacts already clears the previous run automatically."
    ),
    title="Clear Store",
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
)
async def clear_store(
    context: ArtifactSearchToolContext,
) -> dict[str, Any]:
    """Reset the in-memory store to empty by dropping all artifact tables."""
    dropped = context.store.reset()
    return {"status": "cleared", "tables_dropped": dropped}


@artifact_search_tool(
    description=get_prompt("list_artifact_tables"),
    title="List Artifact Tables",
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
)
async def list_artifact_tables(
    context: ArtifactSearchToolContext,
) -> list[dict[str, Any]]:
    """List all loaded dbt artifact tables and their row counts."""
    return context.store.list_tables()


@artifact_search_tool(
    description=get_prompt("describe_artifact_table"),
    title="Describe Artifact Table",
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
)
async def describe_artifact_table(
    context: ArtifactSearchToolContext,
    table_name: str = Field(description="Name of the artifact table to describe"),
) -> list[dict[str, str]]:
    """Show column names and types for a loaded artifact table."""
    return context.store.describe_table(table_name)


@artifact_search_tool(
    description=get_prompt("query_artifacts"),
    title="Query Artifacts",
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
)
async def query_artifacts(
    context: ArtifactSearchToolContext,
    sql: str = Field(
        description=(
            "Read-only SQL query (SELECT only) against the in-memory artifact database. "
            "Supports JOINs, aggregations, CTEs, window functions. Results capped at 500 rows."
        )
    ),
) -> list[dict[str, Any]]:
    """Execute a read-only SQL query against loaded artifact data."""
    return context.store.query(sql)


@artifact_search_tool(
    description=get_prompt("search_artifacts"),
    title="Search Artifacts",
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
)
async def search_artifacts(
    context: ArtifactSearchToolContext,
    table_name: str = Field(
        description="Table to search (e.g. nodes, macros, run_results)"
    ),
    search_query: str = Field(
        description="Search terms for BM25 full-text search across indexed columns"
    ),
    limit: int = Field(default=20, description="Maximum number of results to return"),
) -> list[dict[str, Any]]:
    """Full-text BM25 keyword search on a loaded artifact table."""
    return context.store.search(table_name, search_query, limit)


# ── Tool list ────────────────────────────────────────────────────────────

ARTIFACT_SEARCH_TOOLS = [
    load_artifacts,
    list_artifact_tables,
    describe_artifact_table,
    query_artifacts,
    search_artifacts,
    clear_store,
]


# ── Registration ─────────────────────────────────────────────────────────


def register_artifact_search_tools(
    dbt_mcp: FastMCP,
    admin_config_provider: ConfigProvider[AdminApiConfig],
    *,
    store: ArtifactStore | None = None,
    disabled_tools: set[ArtifactSearchToolName] | None = None,
    enabled_tools: set[ArtifactSearchToolName] | None = None,
    enabled_toolsets: set[Toolset | ArtifactSearchToolset] | None = None,
    disabled_toolsets: set[Toolset | ArtifactSearchToolset] | None = None,
) -> ArtifactStore:
    """Register artifact search tools with the MCP server.

    Mirrors ``register_admin_api_tools()`` from ``dbt_mcp.dbt_admin.tools``.

    Returns the shared ``ArtifactStore`` instance so callers can
    reference it (e.g. for cleanup).
    """
    shared_store = store or ArtifactStore()

    def bind_context() -> ArtifactSearchToolContext:
        return ArtifactSearchToolContext(
            admin_api_config_provider=admin_config_provider,
            store=shared_store,
        )

    generic_register_tools(
        dbt_mcp,
        tool_definitions=[
            tool.adapt_context(bind_context) for tool in ARTIFACT_SEARCH_TOOLS
        ],
        disabled_tools=disabled_tools or set(),
        enabled_tools=enabled_tools,
        enabled_toolsets=enabled_toolsets or set(),
        disabled_toolsets=disabled_toolsets or set(),
        tool_to_toolset=TOOL_TO_TOOLSET,
    )

    return shared_store
