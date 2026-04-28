"""Extraction functions that convert validated Pydantic models to DuckDB row tuples.

Refactored from src/mcp_server/artifacts.py to accept typed schemas
instead of raw dicts.
"""

import json
from enum import Enum
from typing import Any, Callable

from src.dbt_mcp_server.schemas._base import ArtifactBaseModel
from src.dbt_mcp_server.schemas.catalog import (
    CatalogArtifactSchema,
    CatalogNodeSchema,
)
from src.dbt_mcp_server.schemas.common import (
    ChecksumSchema,
    ContractSchema,
    DependsOnSchema,
    DocsSchema,
    NodeConfigSchema,
    TestMetadataInnerSchema,
)
from src.dbt_mcp_server.schemas.manifest import (
    ManifestArtifactSchema,
    ManifestExposureSchema,
    ManifestGroupSchema,
    ManifestMacroSchema,
    ManifestMetricSchema,
    ManifestNodeSchema,
)
from src.dbt_mcp_server.schemas.run_results import RunResultsArtifactSchema
from src.dbt_mcp_server.schemas.sources import SourcesArtifactSchema


# ── Helpers ─────────────────────────────────────────────────────────────


def _json(data: Any) -> str:
    """Serialize to JSON string, empty string for falsy values."""
    if isinstance(data, ArtifactBaseModel):
        return json.dumps(data.model_dump()) if data else ""
    return json.dumps(data) if data else ""


def _get_config(node: ManifestNodeSchema) -> NodeConfigSchema:
    """Safely extract NodeConfigSchema from a node."""
    if isinstance(node.config, NodeConfigSchema):
        return node.config
    return NodeConfigSchema()


def _get_depends_on(node: ManifestNodeSchema) -> DependsOnSchema:
    """Safely extract DependsOnSchema from a node."""
    if isinstance(node.depends_on, DependsOnSchema):
        return node.depends_on
    return DependsOnSchema()


def _get_contract(node: ManifestNodeSchema) -> ContractSchema:
    """Safely extract ContractSchema from a node."""
    if isinstance(node.contract, ContractSchema):
        return node.contract
    return ContractSchema()


def _get_docs(node: ManifestNodeSchema) -> DocsSchema:
    """Safely extract DocsSchema from a node."""
    if isinstance(node.docs, DocsSchema):
        return node.docs
    return DocsSchema()


def _get_checksum(node: ManifestNodeSchema) -> str:
    """Extract checksum string from a node."""
    if isinstance(node.checksum, ChecksumSchema):
        return node.checksum.checksum
    return str(node.checksum) if node.checksum else ""


def _owner_email_str(email: str | list[str]) -> str:
    """Normalize owner email to string (groups use list[str])."""
    if isinstance(email, list):
        return ", ".join(email)
    return email


# ── Artifact type enum ──────────────────────────────────────────────────


class ArtifactType(str, Enum):
    MANIFEST = "manifest.json"
    CATALOG = "catalog.json"
    RUN_RESULTS = "run_results.json"
    SOURCES = "sources.json"


# ── Manifest extraction ─────────────────────────────────────────────────


def _map_node(idx: int, node: ManifestNodeSchema) -> tuple:
    """Map a manifest node or source to a nodes table row."""
    config = _get_config(node)
    depends_on = _get_depends_on(node)
    contract = _get_contract(node)
    docs = _get_docs(node)

    return (
        idx,
        node.unique_id,
        node.name,
        node.resource_type,
        node.package_name,
        node.path or node.file_path or "",
        node.original_file_path or "",
        _json(node.fqn),
        node.alias or "",
        _get_checksum(node),
        node.description or "",
        node.language or "",
        node.raw_code or node.raw_sql or "",
        node.database or "",
        node.schema_ or "",
        node.relation_name or "",
        node.identifier or node.alias or "",
        node.enabled if node.enabled is not None else config.enabled,
        node.materialized or config.materialized or "",
        config.incremental_strategy or "",
        config.on_schema_change or "",
        _json(config.unique_key) if config.unique_key else "",
        config.full_refresh,
        _json(node.config),
        node.access or "",
        node.group or "",
        contract.enforced,
        str(node.version) if node.version is not None else "",
        str(node.latest_version) if node.latest_version is not None else "",
        node.deprecation_date or "",
        _json(node.constraints),
        _json(node.tags),
        _json(node.meta),
        node.source_name or "",
        node.source_description or "",
        node.loader or "",
        node.loaded_at_field or "",
        _json(node.freshness),
        node.compiled_code or node.compiled_sql or "",
        node.compiled_path or "",
        _json(node.extra_ctes),
        node.patch_path or "",
        docs.show,
        _json(node.quoting),
        _json(depends_on.nodes),
        _json(depends_on.macros),
    )


def _extract_node_columns(node: ManifestNodeSchema) -> list[tuple]:
    """Extract column rows from a manifest node."""
    rows = []
    for idx, (col_name, col) in enumerate(node.columns.items()):
        rows.append((
            node.unique_id,
            col.name or col_name,
            idx,
            col.data_type or col.type or "",
            None,  # catalog_type
            None,  # data_type resolved
            col.description or "",
            _json(col.tags),
            _json(col.meta),
            _json(col.tests),
            None,  # catalog_comment
        ))
    return rows


def _extract_edges(node: ManifestNodeSchema) -> list[tuple]:
    """Extract dependency edges from a manifest node."""
    depends_on = _get_depends_on(node)
    return [(parent_id, node.unique_id, "ref") for parent_id in depends_on.nodes]


def _extract_test_metadata(node: ManifestNodeSchema) -> tuple | None:
    """Extract test metadata if this is a test node."""
    if node.resource_type != "test":
        return None
    tm = node.test_metadata
    if not tm or not isinstance(tm, TestMetadataInnerSchema):
        return None
    config = _get_config(node)
    depends_on = _get_depends_on(node)
    attached = next((n for n in depends_on.nodes if not n.startswith("test.")), "")

    return (
        node.unique_id,
        tm.name,
        tm.namespace,
        _json(tm.kwargs),
        tm.kwargs.get("column_name", "") if isinstance(tm.kwargs, dict) else "",
        attached,
        config.severity or "",
        config.warn_if or "",
        config.error_if or "",
        config.fail_calc or "",
        config.store_failures,
        config.store_failures_as or "",
    )


def _map_exposure(idx: int, exp: ManifestExposureSchema) -> tuple:
    """Map an exposure entry to a row."""
    return (
        idx,
        exp.unique_id,
        exp.name,
        exp.type,
        exp.label or "",
        exp.owner.name,
        _owner_email_str(exp.owner.email),
        exp.url or "",
        exp.maturity or "",
        exp.description or "",
        exp.package_name or "",
        exp.path or "",
        exp.original_file_path or "",
        _json(exp.fqn),
        _json(exp.depends_on.nodes),
        _json(exp.depends_on.macros),
        _json(exp.tags),
        _json(exp.meta),
        _json(exp.config),
    )


def _map_metric(idx: int, metric: ManifestMetricSchema) -> tuple:
    """Map a metric entry to a row."""
    measure = metric.type_params.get("measure", {})
    semantic_model_name = measure.get("name", "") if isinstance(measure, dict) else ""
    return (
        idx,
        metric.unique_id,
        metric.name,
        metric.label or "",
        metric.type or metric.calculation_method or "",
        metric.description or "",
        metric.package_name or "",
        metric.path or "",
        metric.original_file_path or "",
        _json(metric.fqn),
        _json(metric.type_params),
        metric.time_granularity or "",
        semantic_model_name,
        _json(metric.depends_on.nodes),
        _json(metric.depends_on.macros),
        metric.group or "",  # None-safe
        _json(metric.tags),
        _json(metric.meta),
        _json(metric.config),
    )


def _map_group(idx: int, group: ManifestGroupSchema) -> tuple:
    """Map a group entry to a row."""
    return (
        idx,
        group.unique_id,
        group.name,
        group.description or "",
        group.package_name or "",
        group.path or "",
        group.original_file_path or "",
        group.owner.name,
        _owner_email_str(group.owner.email),
    )


def _map_macro(idx: int, macro: ManifestMacroSchema) -> tuple:
    """Map a macro entry to a row."""
    depends_on = (
        macro.depends_on
        if isinstance(macro.depends_on, DependsOnSchema)
        else DependsOnSchema()
    )
    return (
        idx,
        macro.unique_id,
        macro.name,
        macro.package_name,
        macro.path or "",
        macro.original_file_path or "",
        macro.macro_sql or "",
        macro.description or "",
        _json(depends_on.macros),
        _json(macro.arguments),
        _json(macro.meta),
    )


def extract_from_manifest(
    data: ManifestArtifactSchema,
) -> dict[str, list[tuple]]:
    """Extract all tables from a validated manifest."""
    all_nodes = list(data.nodes.values()) + list(data.sources.values())

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

    # Exposure → model edges
    for exp in data.exposures.values():
        depends_on = (
            exp.depends_on if isinstance(exp.depends_on, DependsOnSchema)
            else DependsOnSchema()
        )
        for parent_id in depends_on.nodes:
            edge_rows.append((parent_id, exp.unique_id, "exposure_ref"))

    # Metric → model edges
    for metric in data.metrics.values():
        depends_on = (
            metric.depends_on if isinstance(metric.depends_on, DependsOnSchema)
            else DependsOnSchema()
        )
        for parent_id in depends_on.nodes:
            edge_rows.append((parent_id, metric.unique_id, "metric_ref"))

    # Add sequential ids to tables that extracted without them
    column_rows = [(i, *row) for i, row in enumerate(column_rows)]
    edge_rows = [(i, *row) for i, row in enumerate(edge_rows)]
    test_rows = [(i, *row) for i, row in enumerate(test_rows)]

    # Exposures
    exposure_rows = [
        _map_exposure(i, e) for i, e in enumerate(data.exposures.values())
    ]

    # Metrics
    metric_rows = [_map_metric(i, m) for i, m in enumerate(data.metrics.values())]

    # Groups
    group_rows = [_map_group(i, g) for i, g in enumerate(data.groups.values())]

    # Macros
    macro_rows = [_map_macro(i, m) for i, m in enumerate(data.macros.values())]

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


# ── Catalog extraction ──────────────────────────────────────────────────


def extract_from_catalog(
    data: CatalogArtifactSchema,
) -> dict[str, list[tuple]]:
    """Extract tables from a validated catalog."""
    all_entries = list(data.nodes.values()) + list(data.sources.values())

    table_rows: list[tuple] = []
    stat_rows: list[tuple] = []
    column_updates: list[tuple] = []

    stat_idx = 0

    for idx, entry in enumerate(all_entries):
        table_rows.append((
            idx,
            entry.unique_id,
            entry.metadata.type,
            entry.metadata.database,
            entry.metadata.schema_,
            entry.metadata.name,
            entry.metadata.owner,
            entry.metadata.comment or "",
        ))

        # Stats
        for stat_id, stat in entry.stats.items():
            stat_rows.append((
                stat_idx,
                entry.unique_id,
                stat.id or stat_id,
                stat.label,
                str(stat.value),
                stat.description or "",
                stat.include,
            ))
            stat_idx += 1

        # Column updates for merge into node_columns
        for col_name, col in entry.columns.items():
            column_updates.append((
                entry.unique_id,
                col.name or col_name,
                col.index,
                col.type,
                col.comment or "",
            ))

    return {
        "catalog_tables": table_rows,
        "catalog_stats": stat_rows,
        "_node_columns_update": column_updates,
    }


# ── Run results extraction ──────────────────────────────────────────────


def extract_from_run_results(
    data: RunResultsArtifactSchema,
) -> dict[str, list[tuple]]:
    """Extract tables from validated run_results."""
    invocation_id = data.metadata.invocation_id

    invocation_rows = [(
        0,
        invocation_id,
        data.args.which or data.args.command or "",
        data.args.select or "",
        data.metadata.dbt_version,
        data.metadata.generated_at,
        data.elapsed_time,
        _json(data.args.model_dump()),
        len(data.results),
    )]

    result_rows = []
    for idx, result in enumerate(data.results):
        result_rows.append((
            idx,
            result.unique_id,
            invocation_id,
            result.status,
            result.execution_time,
            result.thread_id,
            result.message or "",
            result.relation_name or "",
            _json(result.adapter_response),
            _json([t.model_dump() for t in result.timing]),
        ))

    return {
        "invocations": invocation_rows,
        "run_results": result_rows,
    }


# ── Sources extraction ──────────────────────────────────────────────────


def extract_from_sources(
    data: SourcesArtifactSchema,
) -> dict[str, list[tuple]]:
    """Extract tables from validated sources (freshness)."""
    invocation_id = data.metadata.invocation_id

    rows = []
    for idx, result in enumerate(data.results):
        unique_id = result.unique_id
        parts = unique_id.split(".")

        rows.append((
            idx,
            unique_id,
            parts[2] if len(parts) > 2 else "",
            parts[3] if len(parts) > 3 else "",
            invocation_id,
            result.status,
            result.max_loaded_at or "",
            result.snapshotted_at or "",
            result.max_loaded_at_time_ago_in_s or 0.0,
            result.execution_time,
            result.thread_id or "",
            result.error or "",
            (result.criteria or {}).get("warn_after", {}).get("count"),
            (result.criteria or {}).get("warn_after", {}).get("period", ""),
            (result.criteria or {}).get("error_after", {}).get("count"),
            (result.criteria or {}).get("error_after", {}).get("period", ""),
            _json(result.adapter_response),
            _json(result.timing),
        ))

    return {"source_freshness": rows}


# ── Pipeline mappings ───────────────────────────────────────────────────

ARTIFACT_VALIDATORS: dict[ArtifactType, type[ArtifactBaseModel]] = {
    ArtifactType.MANIFEST: ManifestArtifactSchema,
    ArtifactType.CATALOG: CatalogArtifactSchema,
    ArtifactType.RUN_RESULTS: RunResultsArtifactSchema,
    ArtifactType.SOURCES: SourcesArtifactSchema,
}

ARTIFACT_EXTRACTORS: dict[ArtifactType, Callable] = {
    ArtifactType.MANIFEST: extract_from_manifest,
    ArtifactType.CATALOG: extract_from_catalog,
    ArtifactType.RUN_RESULTS: extract_from_run_results,
    ArtifactType.SOURCES: extract_from_sources,
}
