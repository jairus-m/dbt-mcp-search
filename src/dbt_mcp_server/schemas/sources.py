"""Pydantic schemas for sources.json (freshness) artifacts.

Extends dbt-mcp's existing SourceFreshnessResultSchema with additional
fields needed for DuckDB ingestion.
"""

from typing import Any

from dbt_mcp.dbt_admin.run_artifacts.config import (
    SourceFreshnessResultSchema as DbtMcpSourceFreshnessResultSchema,
)

from src.dbt_mcp_server.schemas._base import ArtifactBaseModel
from src.dbt_mcp_server.schemas.common import ArtifactMetadataSchema


class SourceFreshnessResultSchema(DbtMcpSourceFreshnessResultSchema):
    """Extended source freshness result for DuckDB ingestion.

    Inherits unique_id, status, max_loaded_at, snapshotted_at,
    max_loaded_at_time_ago_in_s, criteria from dbt-mcp.
    """

    execution_time: float = 0.0
    thread_id: str = ""
    error: str = ""
    adapter_response: dict[str, Any] = {}
    timing: list[dict[str, Any]] = []


class SourcesArtifactSchema(ArtifactBaseModel):
    """Top-level schema for sources.json."""

    metadata: ArtifactMetadataSchema = ArtifactMetadataSchema()
    results: list[SourceFreshnessResultSchema] = []
    elapsed_time: float = 0.0
