"""Entry point for `python -m src.dbt_mcp_server.main`:

Test config in .mcp.json:

```json
{
  "mcpServers": {
    "dbt-artifacts": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.dbt_mcp_server.main"],
      "env": {
        "DBT_HOST": "YOUR-HOST",
        "DBT_TOKEN": "YOUR-TOKEN",
        "DBT_ACCOUNT_ID": "YOUR-ACCOUNT-ID"
      }
    }
  }
}
```


"""

from src.dbt_mcp_server.server import main

if __name__ == "__main__":
    main()
