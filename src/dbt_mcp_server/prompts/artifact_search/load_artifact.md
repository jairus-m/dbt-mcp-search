Load one or more dbt Cloud job run artifacts into the searchable in-memory database.

Fetches the requested artifacts from the dbt Cloud Admin API and makes them queryable
via `search_artifacts` and `query_artifacts`. **Automatically clears any previously
loaded run first** — only one run is held in memory at a time.

## Parameters

- **run_id** (required): The dbt Cloud job run ID. Get this from `list_jobs_runs` or
  `get_job_run_details`.
- **artifact_types** (optional, default: all four): List of artifact files to load.
  Pass a subset to load only what you need (faster for targeted workflows).
  - `manifest.json` — Project metadata: models, tests, seeds, snapshots, sources,
    macros, exposures, metrics, groups. Produces 8 tables (nodes, node_columns, edges,
    test_metadata, exposures, metrics, groups, macros).
  - `catalog.json` — Warehouse metadata: table types, column types, row counts.
    Produces 2 tables (catalog_tables, catalog_stats) and enriches node_columns with
    authoritative types.
  - `run_results.json` — Execution results: status, timing, messages per node.
    Produces 2 tables (invocations, run_results).
  - `sources.json` — Source freshness check results. Produces 1 table
    (source_freshness).

## Returns

A dict with:
- `status`: "loaded"
- `run_id`: The run ID
- `tables_loaded`: Dict of `{table_name: row_count}` for each table populated
- `indexes_built`: List of table names that had indexes built
- `timing_ms`: Per-artifact timing breakdown `{artifact: {validate_ms, extract_ms, insert_ms, index_ms, total_ms}}`
- `index_build_ms`: Time spent building all indexes after loading (milliseconds)
- `errors`: Dict of `{artifact: error_message}` for any failed artifacts

## Recommended loading order

1. `manifest.json` — gives you the full project structure
2. `catalog.json` — enriches columns with warehouse types
3. `run_results.json` and/or `sources.json` as needed

## Example usage

Load all artifacts (default):
```json
{"run_id": 477826677}
```

Load a subset for quick failure diagnosis:
```json
{"run_id": 477826677, "artifact_types": ["manifest.json", "run_results.json"]}
```
