"""Artifact search error hierarchy.

Mirrors dbt_mcp/errors/admin_api.py structure.
"""

from dbt_mcp.errors.base import ToolCallError


class ArtifactSearchError(ToolCallError):
    """Base exception for all artifact search errors."""


class ArtifactLoadError(ArtifactSearchError):
    """Raised when fetching or parsing an artifact from the Admin API fails."""


class ArtifactQueryError(ArtifactSearchError):
    """Raised when a SQL query against the artifact store fails."""


class ArtifactNotLoadedError(ArtifactSearchError):
    """Raised when querying before any artifacts have been loaded."""


class ArtifactValidationError(ArtifactSearchError):
    """Raised when Pydantic validation of raw artifact JSON fails."""
