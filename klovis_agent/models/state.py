from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from klovis_agent.models.plan import ExecutionPlan
from klovis_agent.models.step import StepResult
from klovis_agent.models.task import Task


class AgentState(BaseModel):
    """Complete state of an agent run, passed through the LangGraph graph."""

    run_id: str

    task: Task
    plan: ExecutionPlan | None = None

    current_step_id: str | None = None

    step_results: list[StepResult] = Field(default_factory=list)
    artifacts: dict[str, object] = Field(default_factory=dict)

    iteration_count: int = 0
    max_iterations: int = 25
    verbose: bool = False
    status: Literal["running", "completed", "failed"] = "running"
