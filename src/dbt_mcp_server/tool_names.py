"""Tool name enum and toolset mapping for artifact search.

Uses a local enum + generic_dbt_mcp_tool so the scaffold works standalone
without modifying dbt-mcp's ToolName. When porting to dbt-mcp, swap to
ToolName + dbt_mcp_tool.
"""

from enum import Enum
from functools import partial

from dbt_mcp.tools.definitions import generic_dbt_mcp_tool


class ArtifactSearchToolName(Enum):
    LOAD_ARTIFACTS = "load_artifacts"
    LIST_ARTIFACT_TABLES = "list_artifact_tables"
    DESCRIBE_ARTIFACT_TABLE = "describe_artifact_table"
    QUERY_ARTIFACTS = "query_artifacts"
    SEARCH_ARTIFACTS = "search_artifacts"
    CLEAR_STORE = "clear_store"


class ArtifactSearchToolset(Enum):
    ARTIFACT_SEARCH = "artifact_search"


# Every artifact search tool belongs to the ARTIFACT_SEARCH toolset
TOOL_TO_TOOLSET: dict[ArtifactSearchToolName, ArtifactSearchToolset] = {
    name: ArtifactSearchToolset.ARTIFACT_SEARCH for name in ArtifactSearchToolName
}

# Decorator: drop-in replacement for dbt_mcp_tool using our local enum
artifact_search_tool = partial(generic_dbt_mcp_tool, name_enum=ArtifactSearchToolName)
