"""dbt-artifacts MCP server — in-memory DuckDB artifact store with query tools."""

from src.mcp_server.server import main, mcp
from src.mcp_server.store import ArtifactStore

__all__ = ["ArtifactStore", "main", "mcp"]
