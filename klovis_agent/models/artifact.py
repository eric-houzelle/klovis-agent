from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CodeArtifact(BaseModel):
    """Code artifact produced by the agent."""

    artifact_id: str
    language: str
    entrypoint: str
    files: dict[str, str] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Result of a sandbox execution."""

    status: Literal["success", "error", "timeout"]
    stdout: str = ""
    stderr: str = ""
    artifacts: list[dict[str, object]] = Field(default_factory=list)
