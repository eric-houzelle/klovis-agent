from __future__ import annotations

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A goal to be achieved by the agent."""

    task_id: str
    goal: str
    context: dict[str, object] = Field(default_factory=dict)
    constraints: dict[str, object] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
