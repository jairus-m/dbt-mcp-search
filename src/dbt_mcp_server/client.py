"""Artifact fetch client — wraps DbtAdminAPIClient for artifact retrieval.

Mirrors the pattern in dbt_mcp/dbt_admin/run_artifacts/parser.py where
ErrorFetcher wraps the admin client for structured artifact access.
"""

import json
import logging
from typing import Any

from dbt_mcp.dbt_admin.client import DbtAdminAPIClient
from dbt_mcp.errors.admin_api import ArtifactRetrievalError
from dbt_mcp.errors.common import NotFoundError

from src.dbt_mcp_server.errors import ArtifactLoadError, ArtifactValidationError

logger = logging.getLogger(__name__)


class ArtifactFetchClient:
    """Client for fetching and validating dbt artifacts from the Admin API.

    Wraps ``DbtAdminAPIClient.get_job_run_artifact()`` with JSON parsing
    and structured error handling.
    """

    def __init__(self, admin_client: DbtAdminAPIClient) -> None:
        self.admin_client = admin_client

    async def fetch_artifact(
        self,
        account_id: int,
        run_id: int,
        artifact_path: str,
        *,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Fetch an artifact from the Admin API and parse the JSON.

        Args:
            account_id: dbt Cloud account ID.
            run_id: Job run ID.
            artifact_path: e.g. ``"manifest.json"``.
            step: Optional step index for multi-step runs.

        Returns:
            Parsed artifact JSON as a dict.

        Raises:
            ArtifactLoadError: If the fetch fails (404, network error, etc.).
            ArtifactValidationError: If the response is not valid JSON.
        """
        try:
            raw_text = await self.admin_client.get_job_run_artifact(
                account_id, run_id, artifact_path, step=step
            )
        except NotFoundError as e:
            raise ArtifactLoadError(
                f"Artifact '{artifact_path}' not found for run {run_id}. "
                f"Use list_job_run_artifacts to see available artifacts."
            ) from e
        except ArtifactRetrievalError as e:
            raise ArtifactLoadError(
                f"Failed to fetch '{artifact_path}' for run {run_id}: {e}"
            ) from e

        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError) as e:
            raise ArtifactValidationError(
                f"Invalid JSON in '{artifact_path}' for run {run_id}: {e}"
            ) from e

        if not isinstance(data, dict):
            raise ArtifactValidationError(
                f"Expected JSON object for '{artifact_path}', got {type(data).__name__}"
            )

        logger.info(
            f"Fetched {artifact_path} for run {run_id} "
            f"({len(raw_text):,} bytes)"
        )
        return data
