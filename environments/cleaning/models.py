"""Data models for the DataSage Cleaning Environment."""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional


class CleaningAction(Action):
    """Action for the Cleaning environment - a data cleaning operation."""

    operation: str = Field(
        ...,
        description="Cleaning operation: fill_null|fix_type|remove_duplicate|standardize|trim|correct_typo",
    )
    column: str = Field(..., description="Target column name")
    value: Optional[str] = Field(
        None,
        description="Replacement value or rule (e.g., 'median', 'mode', a specific value)",
    )
    params: dict = Field(default_factory=dict)


class CleaningObservation(Observation):
    """Observation from the Cleaning environment - data quality state."""

    domain: str = Field(default="", description="Current domain: hr|sales|pm|it_ops")
    data_preview: str = Field(default="", description="First 5 rows as text table")
    dq_report: str = Field(
        default="",
        description="Completeness, consistency, uniqueness breakdown",
    )
    dq_score: float = Field(default=0.0, description="Overall DQ score 0-1")
    columns_info: str = Field(
        default="", description="Column names, types, null counts"
    )
    step_number: int = Field(default=0)
    max_steps: int = Field(default=15)
