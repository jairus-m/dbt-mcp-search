"""Artifact definitions for loading dbt artifacts into DuckDB.

Replicates the dbt-index approach: parse dbt artifacts (manifest, catalog,
run_results, sources) into normalized, richly-typed DuckDB tables.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class TableConfig:
    """Schema definition for a DuckDB table."""

    table_name: str
    table_ddl: str
    fts_columns: list[str] = field(default_factory=list)
    index_columns: list[str] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────


def _json(data: Any) -> str:
    """Serialize to JSON string, empty string for falsy values."""
    return json.dumps(data) if data else ""


def _checksum(node: dict) -> str:
    """Extract checksum string from node."""
    cs = node.get("checksum", {})
    if isinstance(cs, dict):
        return cs.get("checksum", "")
    return str(cs) if cs else ""


# ── Table Definitions ────────────────────────────────────────────────────

NODES = TableConfig(
    table_name="nodes",
    table_ddl="""
        CREATE TABLE nodes (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR NOT NULL,
            name VARCHAR,
            resource_type VARCHAR,
            package_name VARCHAR,
            file_path VARCHAR,
            original_file_path VARCHAR,
            fqn TEXT,
            alias VARCHAR,
            checksum VARCHAR,
            description TEXT,
            node_language VARCHAR,
            raw_code TEXT,
            database_name VARCHAR,
            schema_name VARCHAR,
            relation_name VARCHAR,
            identifier VARCHAR,
            enabled BOOLEAN,
            materialized VARCHAR,
            incremental_strategy VARCHAR,
            on_schema_change VARCHAR,
            unique_key TEXT,
            full_refresh BOOLEAN,
            config TEXT,
            access_level VARCHAR,
            group_name VARCHAR,
            contract_enforced BOOLEAN,
            version VARCHAR,
            latest_version VARCHAR,
            deprecation_date VARCHAR,
            primary_key TEXT,
            tags TEXT,
            meta TEXT,
            source_name VARCHAR,
            source_description TEXT,
            loader VARCHAR,
            loaded_at_field VARCHAR,
            freshness TEXT,
            compiled_code TEXT,
            compiled_path VARCHAR,
            extra_ctes TEXT,
            patch_path VARCHAR,
            docs_show BOOLEAN,
            quoting TEXT,
            depends_on_nodes TEXT,
            depends_on_macros TEXT
        )
    """,
    fts_columns=["name", "description", "raw_code", "compiled_code"],
    index_columns=["unique_id", "name", "resource_type", "package_name"],
)

NODE_COLUMNS = TableConfig(
    table_name="node_columns",
    table_ddl="""
        CREATE TABLE node_columns (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR NOT NULL,
            column_name VARCHAR NOT NULL,
            column_index INTEGER,
            declared_type VARCHAR,
            catalog_type VARCHAR,
            data_type VARCHAR,
            description TEXT,
            tags TEXT,
            meta TEXT,
            tests TEXT,
            catalog_comment TEXT,
            UNIQUE (unique_id, column_name)
        )
    """,
    fts_columns=["column_name", "description"],
    index_columns=["unique_id"],
)

EDGES = TableConfig(
    table_name="edges",
    table_ddl="""
        CREATE TABLE edges (
            id INTEGER PRIMARY KEY,
            parent_unique_id VARCHAR NOT NULL,
            child_unique_id VARCHAR NOT NULL,
            edge_type VARCHAR
        )
    """,
    index_columns=["parent_unique_id", "child_unique_id"],
)

TEST_METADATA = TableConfig(
    table_name="test_metadata",
    table_ddl="""
        CREATE TABLE test_metadata (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR NOT NULL,
            test_name VARCHAR,
            test_namespace VARCHAR,
            kwargs TEXT,
            column_name VARCHAR,
            attached_node VARCHAR,
            severity VARCHAR,
            warn_if VARCHAR,
            error_if VARCHAR,
            fail_calc VARCHAR,
            store_failures BOOLEAN,
            store_failures_as VARCHAR
        )
    """,
    index_columns=["unique_id", "test_name", "attached_node"],
)

EXPOSURES = TableConfig(
    table_name="exposures",
    table_ddl="""
        CREATE TABLE exposures (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR NOT NULL,
            name VARCHAR,
            exposure_type VARCHAR,
            label VARCHAR,
            owner_name VARCHAR,
            owner_email VARCHAR,
            url VARCHAR,
            maturity VARCHAR,
            description TEXT,
            package_name VARCHAR,
            file_path VARCHAR,
            original_file_path VARCHAR,
            fqn TEXT,
            depends_on_nodes TEXT,
            depends_on_macros TEXT,
            tags TEXT,
            meta TEXT,
            config TEXT
        )
    """,
    fts_columns=["name", "description"],
    index_columns=["unique_id", "name"],
)

METRICS = TableConfig(
    table_name="metrics",
    table_ddl="""
        CREATE TABLE metrics (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR NOT NULL,
            name VARCHAR,
            label VARCHAR,
            metric_type VARCHAR,
            description TEXT,
            package_name VARCHAR,
            file_path VARCHAR,
            original_file_path VARCHAR,
            fqn TEXT,
            type_params TEXT,
            time_granularity VARCHAR,
            semantic_model_name VARCHAR,
            depends_on_nodes TEXT,
            depends_on_macros TEXT,
            group_name VARCHAR,
            tags TEXT,
            meta TEXT,
            config TEXT
        )
    """,
    fts_columns=["name", "label", "description"],
    index_columns=["unique_id", "name"],
)

GROUPS = TableConfig(
    table_name="groups",
    table_ddl="""
        CREATE TABLE groups (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR NOT NULL,
            name VARCHAR,
            description TEXT,
            package_name VARCHAR,
            file_path VARCHAR,
            original_file_path VARCHAR,
            owner_name VARCHAR,
            owner_email VARCHAR
        )
    """,
    index_columns=["unique_id", "name"],
)

MACROS = TableConfig(
    table_name="macros",
    table_ddl="""
        CREATE TABLE macros (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR NOT NULL,
            name VARCHAR,
            package_name VARCHAR,
            file_path VARCHAR,
            original_file_path VARCHAR,
            macro_sql TEXT,
            description TEXT,
            depends_on_macros TEXT,
            arguments TEXT,
            meta TEXT
        )
    """,
    fts_columns=["name", "description", "macro_sql"],
    index_columns=["unique_id", "name", "package_name"],
)

CATALOG_TABLES = TableConfig(
    table_name="catalog_tables",
    table_ddl="""
        CREATE TABLE catalog_tables (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR NOT NULL,
            table_type VARCHAR,
            database_name VARCHAR,
            schema_name VARCHAR,
            table_name VARCHAR,
            table_owner VARCHAR,
            table_comment TEXT
        )
    """,
    fts_columns=["table_name", "table_comment"],
    index_columns=["unique_id", "table_name"],
)

CATALOG_STATS = TableConfig(
    table_name="catalog_stats",
    table_ddl="""
        CREATE TABLE catalog_stats (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR NOT NULL,
            stat_id VARCHAR,
            stat_label VARCHAR,
            stat_value VARCHAR,
            description TEXT,
            include_in_stats BOOLEAN
        )
    """,
    index_columns=["unique_id"],
)

INVOCATIONS = TableConfig(
    table_name="invocations",
    table_ddl="""
        CREATE TABLE invocations (
            id INTEGER PRIMARY KEY,
            invocation_id VARCHAR NOT NULL,
            command VARCHAR,
            selector VARCHAR,
            dbt_version VARCHAR,
            generated_at VARCHAR,
            elapsed_time FLOAT,
            args TEXT,
            node_count INTEGER
        )
    """,
    index_columns=["invocation_id"],
)

RUN_RESULTS = TableConfig(
    table_name="run_results",
    table_ddl="""
        CREATE TABLE run_results (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR NOT NULL,
            invocation_id VARCHAR,
            status VARCHAR,
            execution_time FLOAT,
            thread_id VARCHAR,
            message TEXT,
            relation_name VARCHAR,
            adapter_response TEXT,
            timing TEXT
        )
    """,
    fts_columns=["status", "message"],
    index_columns=["unique_id", "invocation_id", "status"],
)

SOURCE_FRESHNESS = TableConfig(
    table_name="source_freshness",
    table_ddl="""
        CREATE TABLE source_freshness (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR NOT NULL,
            source_name VARCHAR,
            table_name VARCHAR,
            invocation_id VARCHAR,
            status VARCHAR,
            max_loaded_at VARCHAR,
            snapshotted_at VARCHAR,
            max_loaded_at_time_ago FLOAT,
            execution_time FLOAT,
            thread_id VARCHAR,
            error TEXT,
            warn_after_count INTEGER,
            warn_after_period VARCHAR,
            error_after_count INTEGER,
            error_after_period VARCHAR,
            adapter_response TEXT,
            timing TEXT
        )
    """,
    fts_columns=["source_name", "table_name", "status"],
    index_columns=["unique_id", "source_name", "status"],
)

# All table configs, keyed by table name
TABLES: dict[str, TableConfig] = {
    c.table_name: c
    for c in [
        NODES,
        NODE_COLUMNS,
        EDGES,
        TEST_METADATA,
        EXPOSURES,
        METRICS,
        GROUPS,
        MACROS,
        CATALOG_TABLES,
        CATALOG_STATS,
        INVOCATIONS,
        RUN_RESULTS,
        SOURCE_FRESHNESS,
    ]
}


# ── Extraction: manifest.json ───────────────────────────────────────────


def _map_node(idx: int, node: dict[str, Any]) -> tuple:
    """Map a manifest node or source to a nodes table row."""
    config = node.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    depends_on = node.get("depends_on") or {}
    docs = node.get("docs") or {}
    contract = node.get("contract") or {}

    return (
        idx,
        node.get("unique_id", ""),
        node.get("name", ""),
        node.get("resource_type", ""),
        node.get("package_name", ""),
        node.get("path", "") or node.get("file_path", "") or "",
        node.get("original_file_path", "") or "",
        _json(node.get("fqn")),
        node.get("alias", "") or "",
        _checksum(node),
        node.get("description", "") or "",
        node.get("language", "") or "",
        node.get("raw_code", "") or node.get("raw_sql", "") or "",
        node.get("database", "") or "",
        node.get("schema", "") or "",
        node.get("relation_name", "") or "",
        node.get("identifier", "") or node.get("alias", "") or "",
        node.get("enabled", config.get("enabled", True)),
        node.get("materialized", "") or config.get("materialized", "") or "",
        config.get("incremental_strategy") or "",
        config.get("on_schema_change") or "",
        _json(config.get("unique_key")) if config.get("unique_key") else "",
        config.get("full_refresh"),
        _json(config),
        node.get("access", "") or "",
        node.get("group", "") or "",
        contract.get("enforced", False) if isinstance(contract, dict) else False,
        str(node["version"]) if node.get("version") is not None else "",
        str(node["latest_version"])
        if node.get("latest_version") is not None
        else "",
        node.get("deprecation_date", "") or "",
        _json(node.get("constraints")),
        _json(node.get("tags")),
        _json(node.get("meta")),
        node.get("source_name", "") or "",
        node.get("source_description", "") or "",
        node.get("loader", "") or "",
        node.get("loaded_at_field", "") or "",
        _json(node.get("freshness")),
        node.get("compiled_code", "") or node.get("compiled_sql", "") or "",
        node.get("compiled_path", "") or "",
        _json(node.get("extra_ctes")),
        node.get("patch_path", "") or "",
        docs.get("show", True) if isinstance(docs, dict) else True,
        _json(node.get("quoting")),
        _json(depends_on.get("nodes")),
        _json(depends_on.get("macros")),
    )


def _extract_node_columns(node: dict[str, Any]) -> list[tuple]:
    """Extract column rows from a manifest node."""
    columns = node.get("columns") or {}
    if not isinstance(columns, dict):
        return []
    unique_id = node.get("unique_id", "")
    rows = []
    for idx, (col_name, col_data) in enumerate(columns.items()):
        if not isinstance(col_data, dict):
            continue
        rows.append((
            unique_id,
            col_data.get("name", col_name),
            idx,
            col_data.get("data_type", "") or col_data.get("type", "") or "",
            None,  # catalog_type
            None,  # data_type resolved
            col_data.get("description", "") or "",
            _json(col_data.get("tags")),
            _json(col_data.get("meta")),
            _json(col_data.get("tests")),
            None,  # catalog_comment
        ))
    return rows


def _extract_edges(node: dict[str, Any]) -> list[tuple]:
    """Extract dependency edges from a manifest node."""
    unique_id = node.get("unique_id", "")
    depends_on = node.get("depends_on") or {}
    dep_nodes = depends_on.get("nodes") or []
    return [(parent_id, unique_id, "ref") for parent_id in dep_nodes]


def _extract_test_metadata(node: dict[str, Any]) -> tuple | None:
    """Extract test metadata if this is a test node."""
    if node.get("resource_type") != "test":
        return None
    tm = node.get("test_metadata")
    if not tm or not isinstance(tm, dict):
        return None
    config = node.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    dep_nodes = (node.get("depends_on") or {}).get("nodes") or []
    attached = next((n for n in dep_nodes if not n.startswith("test.")), "")

    return (
        node.get("unique_id", ""),
        tm.get("name", ""),
        tm.get("namespace", ""),
        _json(tm.get("kwargs")),
        tm.get("kwargs", {}).get("column_name", "")
        if isinstance(tm.get("kwargs"), dict)
        else "",
        attached,
        config.get("severity", "") or "",
        config.get("warn_if", "") or "",
        config.get("error_if", "") or "",
        config.get("fail_calc", "") or "",
        config.get("store_failures"),
        config.get("store_failures_as", "") or "",
    )


def _map_exposure(idx: int, exp: dict[str, Any]) -> tuple:
    """Map an exposure entry to a row."""
    owner = exp.get("owner") or {}
    depends_on = exp.get("depends_on") or {}
    return (
        idx,
        exp.get("unique_id", ""),
        exp.get("name", ""),
        exp.get("type", ""),
        exp.get("label", "") or "",
        owner.get("name", "") if isinstance(owner, dict) else "",
        owner.get("email", "") if isinstance(owner, dict) else "",
        exp.get("url", "") or "",
        exp.get("maturity", "") or "",
        exp.get("description", "") or "",
        exp.get("package_name", "") or "",
        exp.get("path", "") or "",
        exp.get("original_file_path", "") or "",
        _json(exp.get("fqn")),
        _json(depends_on.get("nodes")),
        _json(depends_on.get("macros")),
        _json(exp.get("tags")),
        _json(exp.get("meta")),
        _json(exp.get("config")),
    )


def _map_metric(idx: int, metric: dict[str, Any]) -> tuple:
    """Map a metric entry to a row."""
    depends_on = metric.get("depends_on") or {}
    type_params = metric.get("type_params") or {}
    return (
        idx,
        metric.get("unique_id", ""),
        metric.get("name", ""),
        metric.get("label", "") or "",
        metric.get("type", "") or metric.get("calculation_method", "") or "",
        metric.get("description", "") or "",
        metric.get("package_name", "") or "",
        metric.get("path", "") or "",
        metric.get("original_file_path", "") or "",
        _json(metric.get("fqn")),
        _json(type_params),
        metric.get("time_granularity", "") or "",
        type_params.get("measure", {}).get("name", "")
        if isinstance(type_params.get("measure"), dict)
        else "",
        _json(depends_on.get("nodes")),
        _json(depends_on.get("macros")),
        metric.get("group", "") or "",
        _json(metric.get("tags")),
        _json(metric.get("meta")),
        _json(metric.get("config")),
    )


def _map_group(idx: int, group: dict[str, Any]) -> tuple:
    """Map a group entry to a row."""
    owner = group.get("owner") or {}
    return (
        idx,
        group.get("unique_id", ""),
        group.get("name", ""),
        group.get("description", "") or "",
        group.get("package_name", "") or "",
        group.get("path", "") or "",
        group.get("original_file_path", "") or "",
        owner.get("name", "") if isinstance(owner, dict) else "",
        owner.get("email", "") if isinstance(owner, dict) else "",
    )


def _map_macro(idx: int, macro: dict[str, Any]) -> tuple:
    """Map a macro entry to a row."""
    depends_on = macro.get("depends_on") or {}
    return (
        idx,
        macro.get("unique_id", ""),
        macro.get("name", ""),
        macro.get("package_name", ""),
        macro.get("path", "") or "",
        macro.get("original_file_path", "") or "",
        macro.get("macro_sql", "") or "",
        macro.get("description", "") or "",
        _json(depends_on.get("macros")),
        _json(macro.get("arguments")),
        _json(macro.get("meta")),
    )


def extract_from_manifest(data: dict[str, Any]) -> dict[str, list[tuple]]:
    """Extract all tables from manifest.json."""
    all_nodes = list((data.get("nodes") or {}).values()) + list(
        (data.get("sources") or {}).values()
    )

    node_rows: list[tuple] = []
    column_rows: list[tuple] = []
    edge_rows: list[tuple] = []
    test_rows: list[tuple] = []

    for idx, node in enumerate(all_nodes):
        node_rows.append(_map_node(idx, node))
        column_rows.extend(_extract_node_columns(node))
        edge_rows.extend(_extract_edges(node))
        tm = _extract_test_metadata(node)
        if tm:
            test_rows.append(tm)

    # Add sequential ids to tables that extracted without them
    column_rows = [(i, *row) for i, row in enumerate(column_rows)]
    edge_rows = [(i, *row) for i, row in enumerate(edge_rows)]
    test_rows = [(i, *row) for i, row in enumerate(test_rows)]

    # Exposures
    exposures = list((data.get("exposures") or {}).values())
    exposure_rows = [_map_exposure(i, e) for i, e in enumerate(exposures)]

    # Metrics
    metrics = list((data.get("metrics") or {}).values())
    metric_rows = [_map_metric(i, m) for i, m in enumerate(metrics)]

    # Groups
    groups = list((data.get("groups") or {}).values())
    group_rows = [_map_group(i, g) for i, g in enumerate(groups)]

    # Macros
    macros = list((data.get("macros") or {}).values())
    macro_rows = [_map_macro(i, m) for i, m in enumerate(macros)]

    return {
        "nodes": node_rows,
        "node_columns": column_rows,
        "edges": edge_rows,
        "test_metadata": test_rows,
        "exposures": exposure_rows,
        "metrics": metric_rows,
        "groups": group_rows,
        "macros": macro_rows,
    }


# ── Extraction: catalog.json ────────────────────────────────────────────


def extract_from_catalog(data: dict[str, Any]) -> dict[str, list[tuple]]:
    """Extract tables from catalog.json.

    Returns catalog_tables, catalog_stats, and _node_columns_update
    (a special key for merging catalog types into node_columns).
    """
    all_entries = list((data.get("nodes") or {}).values()) + list(
        (data.get("sources") or {}).values()
    )

    table_rows: list[tuple] = []
    stat_rows: list[tuple] = []
    column_updates: list[tuple] = []

    stat_idx = 0
    col_idx = 0

    for idx, entry in enumerate(all_entries):
        meta = entry.get("metadata") or {}
        unique_id = entry.get("unique_id", "")

        table_rows.append((
            idx,
            unique_id,
            meta.get("type", ""),
            meta.get("database", ""),
            meta.get("schema", ""),
            meta.get("name", ""),
            meta.get("owner", ""),
            meta.get("comment", "") or "",
        ))

        # Stats
        stats = entry.get("stats") or {}
        for stat_id, stat_data in stats.items():
            if not isinstance(stat_data, dict):
                continue
            stat_rows.append((
                stat_idx,
                unique_id,
                stat_data.get("id", stat_id),
                stat_data.get("label", ""),
                str(stat_data.get("value", "")),
                stat_data.get("description", "") or "",
                stat_data.get("include", True),
            ))
            stat_idx += 1

        # Column updates for merge into node_columns
        columns = entry.get("columns") or {}
        for col_name, col_data in columns.items():
            if not isinstance(col_data, dict):
                continue
            column_updates.append((
                unique_id,
                col_data.get("name", col_name),
                col_data.get("index"),
                col_data.get("type", ""),
                col_data.get("comment", "") or "",
            ))
            col_idx += 1

    return {
        "catalog_tables": table_rows,
        "catalog_stats": stat_rows,
        "_node_columns_update": column_updates,
    }


# ── Extraction: run_results.json ────────────────────────────────────────


def extract_from_run_results(data: dict[str, Any]) -> dict[str, list[tuple]]:
    """Extract tables from run_results.json."""
    metadata = data.get("metadata") or {}
    args = data.get("args") or {}
    results = data.get("results") or []

    invocation_id = metadata.get("invocation_id", "")

    invocation_rows = [(
        0,
        invocation_id,
        args.get("which", "") or args.get("command", "") or "",
        args.get("select", "") or "",
        metadata.get("dbt_version", ""),
        metadata.get("generated_at", ""),
        data.get("elapsed_time", 0.0),
        _json(args),
        len(results),
    )]

    result_rows = []
    for idx, result in enumerate(results):
        result_rows.append((
            idx,
            result.get("unique_id", ""),
            invocation_id,
            result.get("status", ""),
            result.get("execution_time", 0.0),
            result.get("thread_id", ""),
            result.get("message", "") or "",
            result.get("relation_name", "") or "",
            _json(result.get("adapter_response")),
            _json(result.get("timing")),
        ))

    return {
        "invocations": invocation_rows,
        "run_results": result_rows,
    }


# ── Extraction: sources.json ────────────────────────────────────────────


def extract_from_sources(data: dict[str, Any]) -> dict[str, list[tuple]]:
    """Extract tables from sources.json."""
    metadata = data.get("metadata") or {}
    results = data.get("results") or []
    invocation_id = metadata.get("invocation_id", "")

    rows = []
    for idx, result in enumerate(results):
        unique_id = result.get("unique_id", "")
        parts = unique_id.split(".")
        criteria = result.get("criteria") or {}
        warn_after = criteria.get("warn_after") or {}
        error_after = criteria.get("error_after") or {}

        rows.append((
            idx,
            unique_id,
            parts[2] if len(parts) > 2 else "",
            parts[3] if len(parts) > 3 else "",
            invocation_id,
            result.get("status", ""),
            result.get("max_loaded_at", "") or "",
            result.get("snapshotted_at", "") or "",
            result.get("max_loaded_at_time_ago_in_s", 0.0),
            result.get("execution_time", 0.0),
            result.get("thread_id", "") or "",
            result.get("error", "") or "",
            warn_after.get("count"),
            warn_after.get("period", ""),
            error_after.get("count"),
            error_after.get("period", ""),
            _json(result.get("adapter_response")),
            _json(result.get("timing")),
        ))

    return {"source_freshness": rows}


# ── Pipeline ─────────────────────────────────────────────────────────────

ARTIFACT_PIPELINE: list[tuple[str, Callable[[dict[str, Any]], dict[str, list[tuple]]]]] = [
    ("manifest.json", extract_from_manifest),
    ("catalog.json", extract_from_catalog),
    ("run_results.json", extract_from_run_results),
    ("sources.json", extract_from_sources),
]
