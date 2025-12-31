"""
V1 - Export GQL result from Discovery API
Currently, this is specific to get_all_models , but would like to
generalize to more tool calls
"""

import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

from dbt_mcp.discovery.tools import DiscoveryToolContext
from dbt_mcp.config.config_providers import DiscoveryConfig, ConfigProvider
from dbt_mcp.config.headers import DiscoveryHeadersProvider
from dbt_mcp.oauth.token_provider import StaticTokenProvider

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# Validate required env vars
required_vars = ["DBT_HOST", "DBT_TOKEN", "DBT_PROD_ENV_ID"]
missing = [v for v in required_vars if not os.environ.get(v)]
if missing:
    logger.error(f"Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)

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


async def main():
    """Main function to export dbt models from Discovery API to JSON."""
    token_provider = StaticTokenProvider(token=DBT_TOKEN)
    headers_provider = DiscoveryHeadersProvider(token_provider=token_provider)

    url = f"https://metadata.{DBT_HOST}/graphql"
    config = DiscoveryConfig(
        url=url, headers_provider=headers_provider, environment_id=DBT_PROD_ENV_ID
    )

    config_provider = StaticConfigProvider(config)
    context = DiscoveryToolContext(config_provider)

    models = await get_all_models(context)
    logger.info(f"Fetched {len(models)} models")

    output_dir = "data/gql_output"
    output_file = os.path.join(output_dir, "all_models.json")

    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(models, f, indent=2)

    logger.info(f"Saved models to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
