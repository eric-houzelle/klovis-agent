from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StepLog(BaseModel):
    """Log entry for a step or LLM call."""

    run_id: str
    step_id: str
    event_type: Literal["step_start", "step_end", "llm_call", "tool_call", "error"]
    llm_model: str | None = None
    tool_name: str | None = None
    duration_ms: int | None = None
    status: str = "pending"
    metadata: dict[str, object] = Field(default_factory=dict)
