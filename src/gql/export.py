"""
V1 - Export GQL result from Discovery API
Currently, this is specific to get_all_models , but would like to 
generalize to more tool calls
"""

import asyncio
import json
import logging
import os

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
    """Simple static config provider for testing."""

    def __init__(self, config: DiscoveryConfig):
        self._config = config

    async def get_config(self) -> DiscoveryConfig:
        return self._config


async def get_all_models(context: DiscoveryToolContext) -> list[dict]:
    """
    Fetch all models from the dbt Cloud Discovery API.

    Args:
        context: DiscoveryToolContext containing the models_fetcher instance

    Returns:
        List of dictionaries containing model data
    """
    return await context.models_fetcher.fetch_models()


logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


async def main():
    token_provider = StaticTokenProvider(token=DBT_TOKEN)
    headers_provider = DiscoveryHeadersProvider(token_provider=token_provider)

    # Discovery API GraphQL endpoint URL
    url = f'https://metadata.{DBT_HOST}/graphql'

    config = DiscoveryConfig(
        url=url,
        headers_provider=headers_provider,
        environment_id=DBT_PROD_ENV_ID
    )

    config_provider = StaticConfigProvider(config)
    context = DiscoveryToolContext(config_provider)

    models = await get_all_models(context)

    print(f"Fetched {len(models)} models")

    output_dir = "data/gql_output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_models.json")

    with open(output_file, "w") as f:
        json.dump(models, f, indent=2)

    print(f"Saved models to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())