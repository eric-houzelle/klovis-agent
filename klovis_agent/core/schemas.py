"""Pydantic models for LLM structured outputs of each graph node.

Each model uses ``extra="forbid"`` so the generated JSON Schema includes
``additionalProperties: false`` at every level — improving compliance even
when the provider ignores the ``strict`` flag.

The public ``*_OUTPUT_SCHEMA`` constants are derived via
``.model_json_schema()`` so they stay in sync with the models automatically.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared step schema (used by Plan and Replan)
# ---------------------------------------------------------------------------

class StepOutputItem(BaseModel):
    model_config = {"extra": "forbid"}

    step_id: str = Field(description="Simple integer as string: '1', '2', '3', etc.")
    step_type: Literal["execute", "check", "replan", "finish"]
    title: str
    objective: str
    success_criteria: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class PlanOutput(BaseModel):
    model_config = {"extra": "forbid"}

    reasoning_summary: str
    steps: list[StepOutputItem]


PLAN_OUTPUT_SCHEMA: dict[str, object] = PlanOutput.model_json_schema()


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

class ExecuteOutput(BaseModel):
    model_config = {"extra": "forbid"}

    action: Literal["tool_call", "direct_response"]
    tool_name: str = ""
    tool_input: dict[str, object] = Field(default_factory=dict)
    direct_response: str = ""


EXECUTE_OUTPUT_SCHEMA: dict[str, object] = ExecuteOutput.model_json_schema()


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

class CheckOutput(BaseModel):
    model_config = {"extra": "forbid"}

    status: Literal["success", "failed", "retry"]
    observations: list[str]
    next_action: Literal["continue", "retry", "replan", "finish"]


CHECK_OUTPUT_SCHEMA: dict[str, object] = CheckOutput.model_json_schema()


# ---------------------------------------------------------------------------
# Replan
# ---------------------------------------------------------------------------

class ReplanOutput(BaseModel):
    model_config = {"extra": "forbid"}

    reasoning_summary: str
    updated_steps: list[StepOutputItem]


REPLAN_OUTPUT_SCHEMA: dict[str, object] = ReplanOutput.model_json_schema()


# ---------------------------------------------------------------------------
# Finish
# ---------------------------------------------------------------------------

class FinishOutput(BaseModel):
    model_config = {"extra": "forbid"}

    summary: str
    artifacts_produced: list[str] = Field(default_factory=list)
    overall_status: Literal["success", "partial_success", "failure"]
    limitations: list[str] = Field(default_factory=list)


FINISH_OUTPUT_SCHEMA: dict[str, object] = FinishOutput.model_json_schema()
