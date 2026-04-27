"""Pydantic schemas for run_results.json artifacts.

Extends dbt-mcp's existing RunResultSchema with additional fields
needed for DuckDB ingestion.
"""

from typing import Any

from dbt_mcp.dbt_admin.run_artifacts.config import (
    RunResultSchema as DbtMcpRunResultSchema,
    RunResultsArgsSchema,
)

from src.dbt_mcp_server.schemas._base import ArtifactBaseModel
from src.dbt_mcp_server.schemas.common import ArtifactMetadataSchema


class TimingEntrySchema(ArtifactBaseModel):
    """A timing entry within a run result."""

    name: str = ""
    started_at: str = ""
    completed_at: str = ""


class RunResultSchema(DbtMcpRunResultSchema):
    """Extended run result with fields needed for DuckDB ingestion.

    Inherits unique_id, status, message, relation_name, compiled_code
    from dbt-mcp's RunResultSchema.
    """

    execution_time: float = 0.0
    thread_id: str = ""
    adapter_response: dict[str, Any] = {}
    timing: list[TimingEntrySchema] = []


class RunResultsArgsExtendedSchema(RunResultsArgsSchema):
    """Extended args with fields for the invocations table."""

    which: str = ""
    command: str = ""
    select: str = ""


class RunResultsArtifactSchema(ArtifactBaseModel):
    """Top-level schema for run_results.json."""

    metadata: ArtifactMetadataSchema = ArtifactMetadataSchema()
    results: list[RunResultSchema] = []
    elapsed_time: float = 0.0
    args: RunResultsArgsExtendedSchema = RunResultsArgsExtendedSchema()
