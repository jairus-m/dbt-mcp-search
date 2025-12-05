# Hybrid Search for Token-Inefficient Tools in the dbt MCP

https://github.com/dbt-labs/dbt-mcp/issues/413

Goals:
- Learn more about search and the implementation of LanceDB / DuckDB
- See how dbt MCP tool outputs can integrate
- Explore questions raised in the issue

Plan:
- Stage tool outputs locally as JSON objects
- Store locally in DBs
  - Vector embed text for semantic search
  - Store raw text for FTS (full text search)
  - Semantic search + FTS = hybrid searcg
- Query DBs

TODO: Will document more on how this project/experiment works and how it can be ran locally