"""Data models for the DataSage Answering Environment."""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AnsweringAction(Action):
    """Action for the Answering environment - a generated answer."""

    answer: str = Field(..., description="The generated answer text")
    cited_columns: list[str] = Field(default_factory=list, description="Data columns cited")
    reasoning: str = Field(default="", description="Chain-of-thought reasoning")


class AnsweringObservation(Observation):
    """Observation from the Answering environment - context for generating an answer."""

    domain: str = Field(default="")
    dataset_summary: str = Field(default="", description="Statistical summary of enriched data")
    persona: str = Field(default="", description="Executive|Manager|Individual Contributor")
    persona_description: str = Field(default="", description="What this persona cares about")
    question: str = Field(default="", description="The question to answer")
    available_columns: list[str] = Field(default_factory=list)
    column_stats: str = Field(default="", description="Relevant column statistics")
