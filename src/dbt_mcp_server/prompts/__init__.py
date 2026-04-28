"""Prompt loader for artifact search tools.

Mirrors ``dbt_mcp.prompts.prompts.get_prompt``.
"""

from pathlib import Path


def get_prompt(name: str) -> str:
    """Load a prompt markdown file by name (without extension)."""
    return (Path(__file__).parent / "artifact_search" / f"{name}.md").read_text()
