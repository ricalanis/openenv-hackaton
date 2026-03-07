"""Data models for the DataSage Enrichment Environment."""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class EnrichmentAction(Action):
    """Action for the Enrichment environment."""

    operation: str = Field(..., description="add_field|lookup|compute_derived|add_category")
    field_name: str = Field(..., description="Name of new field to add")
    source: str = Field(default="", description="Source from domain's enrichment registry")
    logic: str = Field(default="", description="Computation logic or lookup key")
    params: dict = Field(default_factory=dict)


class EnrichmentObservation(Observation):
    """Observation from the Enrichment environment."""

    domain: str = Field(default="")
    data_preview: str = Field(default="")
    schema_info: str = Field(default="", description="Current columns with types")
    available_sources: list[str] = Field(default_factory=list)
    enrichment_coverage: float = Field(default=0.0)
    fields_added: list[str] = Field(default_factory=list)
    possible_enrichments: list[str] = Field(default_factory=list)
    step_number: int = Field(default=0)
    max_steps: int = Field(default=12)
