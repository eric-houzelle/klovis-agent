from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

StepType = Literal["plan", "execute", "check", "replan", "finish"]
StepStatus = Literal["pending", "running", "done", "failed"]


class StepSpec(BaseModel):
    """Specification of a step in the execution plan."""

    step_id: str
    step_type: Literal["execute", "check", "replan", "finish"]

    title: str
    objective: str

    inputs: dict[str, object] = Field(default_factory=dict)
    expected_outputs: dict[str, object] = Field(default_factory=dict)

    success_criteria: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)

    retry_limit: int = 0
    depends_on: list[str] = Field(default_factory=list)

    status: StepStatus = "pending"


class StepResult(BaseModel):
    """Result of a step execution."""

    step_id: str
    status: Literal["success", "failed", "retry"]

    outputs: dict[str, object] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)

    tool_used: str | None = None
    next_action: str | None = None
