"""
Export run_results.json artifact from dbt Cloud Admin API.
"""

import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

from dbt_mcp.config.config_providers import AdminApiConfig, ConfigProvider
from dbt_mcp.config.headers import AdminApiHeadersProvider
from dbt_mcp.dbt_admin.client import DbtAdminAPIClient
from dbt_mcp.oauth.token_provider import StaticTokenProvider

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

required_vars = ["DBT_HOST", "DBT_TOKEN", "DBT_ACCOUNT_ID", "DBT_RUN_ID"]
missing = [v for v in required_vars if not os.environ.get(v)]
if missing:
    logger.error(f"Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)

DBT_HOST = os.environ["DBT_HOST"].rstrip("/")
if not DBT_HOST.startswith("http"):
    DBT_HOST = f"https://{DBT_HOST}"
DBT_TOKEN = os.environ["DBT_TOKEN"]
DBT_ACCOUNT_ID = int(os.environ["DBT_ACCOUNT_ID"])
DBT_RUN_ID = int(os.environ["DBT_RUN_ID"])


class StaticConfigProvider(ConfigProvider[AdminApiConfig]):
    """Simple static config provider for testing."""

    def __init__(self, config: AdminApiConfig):
        self._config = config

    async def get_config(self) -> AdminApiConfig:
        return self._config


async def main():
    """Fetch run_results.json from dbt Cloud Admin API and save locally."""
    token_provider = StaticTokenProvider(token=DBT_TOKEN)
    headers_provider = AdminApiHeadersProvider(token_provider=token_provider)

    config = AdminApiConfig(
        url=DBT_HOST,
        headers_provider=headers_provider,
        account_id=DBT_ACCOUNT_ID,
    )

    config_provider = StaticConfigProvider(config)
    client = DbtAdminAPIClient(config_provider)

    logger.info(f"Fetching run_results.json for run {DBT_RUN_ID}")
    raw_text = await client.get_job_run_artifact(
        account_id=DBT_ACCOUNT_ID,
        run_id=DBT_RUN_ID,
        artifact_path="run_results.json",
    )

    data = json.loads(raw_text)
    results_count = len(data.get("results", []))
    logger.info(f"Fetched run_results.json with {results_count} results")

    output_dir = "data/admin_output"
    output_file = os.path.join(output_dir, "run_results.json")

    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved run_results.json to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
