"""Artifact definitions for loading dbt Cloud job run artifacts into DuckDB.

Each artifact is defined as an ArtifactConfig that specifies how to extract
entries from the JSON, the table schema, row mapping, and index configuration.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ArtifactConfig:
    """Declarative configuration for loading a dbt artifact into DuckDB."""

    filename: str
    table_name: str
    table_ddl: str
    extract_entries: Callable[[dict[str, Any]], list[dict[str, Any]]]
    map_row: Callable[[int, dict[str, Any]], tuple]
    fts_columns: list[str]
    index_columns: list[str] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────


def _parse_unique_id(unique_id: str) -> tuple[str, str]:
    """Extract (resource_type, name) from 'type.project.name'."""
    parts = unique_id.split(".")
    return (parts[0] if parts else "", parts[-1] if parts else "")


def _extract_row_count(node: dict[str, Any]) -> int | None:
    """Extract row_count from catalog node stats."""
    stats = node.get("stats", {})
    if isinstance(stats, dict):
        row_count_stat = stats.get("row_count", {}) or stats.get("num_rows", {})
        if isinstance(row_count_stat, dict):
            value = row_count_stat.get("value")
            if value is not None:
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    pass
    return None


def _json_field(data: Any) -> str:
    """Serialize a dict/list to JSON string, or return empty string."""
    return json.dumps(data) if data else ""


def _meta_get(node: dict[str, Any], key: str) -> str:
    """Safely extract a field from a catalog node's metadata dict."""
    metadata = node.get("metadata", {}) or {}
    return metadata.get(key, "") if isinstance(metadata, dict) else ""


# ── Row mappers ──────────────────────────────────────────────────────────


def _map_manifest_row(idx: int, node: dict[str, Any]) -> tuple:
    return (
        idx,
        node.get("unique_id", ""),
        node.get("resource_type", ""),
        node.get("name", ""),
        node.get("description", "") or "",
        node.get("raw_code", "") or node.get("raw_sql", "") or "",
        node.get("compiled_code", "") or node.get("compiled_sql", "") or "",
        _json_field(node.get("columns", {})),
        _json_field(node.get("depends_on", {})),
        node.get("database", "") or "",
        node.get("schema", "") or "",
        node.get("package_name", "") or "",
    )


def _map_catalog_row(idx: int, node: dict[str, Any]) -> tuple:
    return (
        idx,
        node.get("unique_id", ""),
        _meta_get(node, "name"),
        _meta_get(node, "type"),
        _meta_get(node, "schema"),
        _meta_get(node, "database"),
        _meta_get(node, "owner"),
        _meta_get(node, "comment"),
        _json_field(node.get("columns", {})),
        _extract_row_count(node),
    )


def _map_run_results_row(idx: int, result: dict[str, Any]) -> tuple:
    unique_id = result.get("unique_id", "")
    resource_type, name = _parse_unique_id(unique_id)
    return (
        idx,
        unique_id,
        resource_type,
        name,
        result.get("status", ""),
        result.get("message", "") or "",
        result.get("relation_name", "") or "",
        result.get("compiled_code", "") or "",
        result.get("execution_time", 0.0),
    )


def _map_sources_row(idx: int, result: dict[str, Any]) -> tuple:
    unique_id = result.get("unique_id", "")
    parts = unique_id.split(".")
    return (
        idx,
        unique_id,
        parts[2] if len(parts) > 2 else "",
        parts[3] if len(parts) > 3 else "",
        result.get("status", ""),
        result.get("max_loaded_at", "") or "",
        result.get("snapshotted_at", "") or "",
        result.get("max_loaded_at_time_ago_in_s", 0.0),
        _json_field(result.get("criteria")),
    )


# ── Entry extractors ────────────────────────────────────────────────────


def _extract_manifest(data: dict[str, Any]) -> list[dict[str, Any]]:
    return list(data.get("nodes", {}).values())


def _extract_catalog(data: dict[str, Any]) -> list[dict[str, Any]]:
    return list(data.get("nodes", {}).values()) + list(
        data.get("sources", {}).values()
    )


def _extract_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    return data.get("results", [])


# ── Configs ──────────────────────────────────────────────────────────────

MANIFEST = ArtifactConfig(
    filename="manifest.json",
    table_name="manifest_nodes",
    table_ddl="""
        CREATE TABLE manifest_nodes (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR,
            resource_type VARCHAR,
            name VARCHAR,
            description TEXT,
            raw_code TEXT,
            compiled_code TEXT,
            columns_json TEXT,
            depends_on_json TEXT,
            database_name VARCHAR,
            schema_name VARCHAR,
            package_name VARCHAR
        )
    """,
    extract_entries=_extract_manifest,
    map_row=_map_manifest_row,
    fts_columns=["name", "description", "raw_code", "compiled_code", "columns_json"],
    index_columns=["name", "unique_id", "resource_type"],
)

CATALOG = ArtifactConfig(
    filename="catalog.json",
    table_name="catalog_nodes",
    table_ddl="""
        CREATE TABLE catalog_nodes (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR,
            name VARCHAR,
            node_type VARCHAR,
            schema_name VARCHAR,
            database_name VARCHAR,
            owner VARCHAR,
            comment TEXT,
            columns_json TEXT,
            row_count BIGINT
        )
    """,
    extract_entries=_extract_catalog,
    map_row=_map_catalog_row,
    fts_columns=["name", "comment", "columns_json"],
    index_columns=["name", "unique_id", "node_type"],
)

RUN_RESULTS = ArtifactConfig(
    filename="run_results.json",
    table_name="run_results",
    table_ddl="""
        CREATE TABLE run_results (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR,
            resource_type VARCHAR,
            name VARCHAR,
            status VARCHAR,
            message TEXT,
            relation_name VARCHAR,
            compiled_code TEXT,
            execution_time FLOAT
        )
    """,
    extract_entries=_extract_results,
    map_row=_map_run_results_row,
    fts_columns=["name", "status", "message", "compiled_code"],
    index_columns=["name", "unique_id", "status"],
)

SOURCES = ArtifactConfig(
    filename="sources.json",
    table_name="source_freshness",
    table_ddl="""
        CREATE TABLE source_freshness (
            id INTEGER PRIMARY KEY,
            unique_id VARCHAR,
            source_name VARCHAR,
            table_name VARCHAR,
            status VARCHAR,
            max_loaded_at VARCHAR,
            snapshotted_at VARCHAR,
            max_loaded_at_time_ago_in_s FLOAT,
            criteria TEXT
        )
    """,
    extract_entries=_extract_results,
    map_row=_map_sources_row,
    fts_columns=["source_name", "table_name", "status"],
    index_columns=["source_name", "table_name", "status"],
)

ALL_ARTIFACTS = [MANIFEST, CATALOG, RUN_RESULTS, SOURCES]
