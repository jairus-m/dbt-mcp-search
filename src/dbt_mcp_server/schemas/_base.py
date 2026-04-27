"""Base model for all dbt artifact schemas."""

from pydantic import BaseModel, ConfigDict


class ArtifactBaseModel(BaseModel):
    """Base model with ``extra="allow"`` so unknown API fields pass through."""

    model_config = ConfigDict(extra="allow")
