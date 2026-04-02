from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from klovis_agent.models.step import StepSpec


class PlanStepResult(BaseModel):
    """Planning phase result: generated steps and reasoning."""

    generated_steps: list[StepSpec]
    reasoning_summary: str


class ExecutionPlan(BaseModel):
    """Versioned execution plan for a given goal."""

    plan_id: str
    version: int = 1
    goal: str
    steps: list[StepSpec] = Field(default_factory=list)
    status: Literal["active", "completed", "failed", "replanned"] = "active"
