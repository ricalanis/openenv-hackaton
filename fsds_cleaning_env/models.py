from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetSnapshot(BaseModel):
    row_count: int
    column_count: int
    missing_cells: int
    duplicate_rows: int
    invalid_type_cells: int
    schema_consistency: float = Field(ge=0.0, le=1.0)


class QualityGateResult(BaseModel):
    name: str
    passed: bool
    message: str


class Observation(BaseModel):
    task_id: str
    stage: str
    summary: str
    available_tools: List[str]
    snapshot: DatasetSnapshot
    recent_events: List[str] = Field(default_factory=list)
    quality_gates: List[QualityGateResult] = Field(default_factory=list)
    done: bool = False


class FSDSAction(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)


class StepRequest(BaseModel):
    action: FSDSAction


class ResetResponse(BaseModel):
    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    truncated: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
