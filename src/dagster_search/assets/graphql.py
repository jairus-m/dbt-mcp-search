"""
Dagster asset for extracting dbt models from GraphQL Discovery API.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from dagster import asset, AssetExecutionContext, MetadataValue, Output
from dotenv import load_dotenv

from dbt_mcp.discovery.tools import DiscoveryToolContext
from dbt_mcp.config.config_providers import DiscoveryConfig, ConfigProvider
from dbt_mcp.config.headers import DiscoveryHeadersProvider
from dbt_mcp.oauth.token_provider import StaticTokenProvider

load_dotenv()

DBT_HOST = os.environ["DBT_HOST"]
DBT_TOKEN = os.environ["DBT_TOKEN"]
DBT_PROD_ENV_ID = int(os.environ["DBT_PROD_ENV_ID"])


class StaticConfigProvider(ConfigProvider[DiscoveryConfig]):
    """Simple static config provider (dbt MCP-specific contruct)."""

    def __init__(self, config: DiscoveryConfig):
        self._config = config

    async def get_config(self) -> DiscoveryConfig:
        return self._config


async def fetch_models_from_api() -> list[dict]:
    """Fetch all models from the dbt Cloud Discovery API."""

    token_provider = StaticTokenProvider(token=DBT_TOKEN)
    headers_provider = DiscoveryHeadersProvider(token_provider=token_provider)

    url = f"https://metadata.{DBT_HOST}/graphql"
    config = DiscoveryConfig(
        url=url, headers_provider=headers_provider, environment_id=DBT_PROD_ENV_ID
    )

    config_provider = StaticConfigProvider(config)
    context = DiscoveryToolContext(config_provider)

    models = await context.models_fetcher.fetch_models()
    return models


@asset(
    description="Extract: Fetch dbt models from Discovery API via GraphQL",
    compute_kind="graphql",
)
def graphql_extract(context: AssetExecutionContext) -> Output[List[Dict[str, Any]]]:
    """
    Step 1: Extract dbt models from Discovery API.

    Returns: List of model dictionaries
    """
    models = asyncio.run(fetch_models_from_api())

    output_dir = Path("data/gql_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "all_models.json"

    with open(output_file, "w") as f:
        json.dump(models, f, indent=2)

    context.log.info(f"Extracted {len(models)} models from Discovery API")

    return Output(
        value=models,
        metadata={
            "num_models": MetadataValue.int(len(models)),
            "output_path": MetadataValue.path(str(output_file)),
            "file_size_kb": MetadataValue.float(output_file.stat().st_size / 1024),
            "sample_model": MetadataValue.json(models[0] if models else {}),
        },
    )
