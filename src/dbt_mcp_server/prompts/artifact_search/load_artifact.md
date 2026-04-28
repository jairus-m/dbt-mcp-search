Load a dbt Cloud job run artifact into the searchable in-memory database.

Fetches the artifact from the dbt Cloud Admin API and makes it queryable via `search_artifacts` and `query_artifacts`. This is the first step — you must load artifacts before you can search or query them.

## Parameters

- **run_id** (required): The dbt Cloud job run ID. Get this from `list_jobs_runs` or `get_job_run_details`.
- **artifact_type** (optional, default: "manifest.json"): Which artifact file to load. One of:
  - `manifest.json` — Project metadata: models, tests, seeds, snapshots, sources, macros, exposures, metrics, groups. Produces 8 tables (nodes, node_columns, edges, test_metadata, exposures, metrics, groups, macros).
  - `catalog.json` — Warehouse metadata: table types, column types, row counts. Produces 2 tables (catalog_tables, catalog_stats) and enriches node_columns with authoritative types.
  - `run_results.json` — Execution results: status, timing, messages per node. Produces 2 tables (invocations, run_results).
  - `sources.json` — Source freshness check results. Produces 1 table (source_freshness).

## Returns

A dict with:
- `status`: "loaded"
- `artifact`: The artifact file name
- `run_id`: The run ID
- `tables_loaded`: Dict of {table_name: row_count} for each table populated

## Recommended Loading Order

1. Start with `manifest.json` — gives you the project structure
2. Then `catalog.json` — enriches columns with warehouse types
3. Then `run_results.json` and/or `sources.json` as needed

## Example Usage

```json
{"run_id": 477826677, "artifact_type": "manifest.json"}
```
