"""Agent state models with Pydantic v2 validation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AgentStatus(str, Enum):
    """Valid agent status values."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


class AgentState(BaseModel):
    """Pydantic model for agent state with validation."""

    id: str = Field(..., description="Unique state identifier")
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Current agent status")
    current_goal: str | None = Field(default=None, description="Active goal being pursued")
    iteration_count: int = Field(default=0, ge=0, description="Number of iterations completed")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="State creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    error_message: str | None = Field(default=None, description="Error message if status is ERROR")

    @field_validator("iteration_count")
    @classmethod
    def validate_iteration_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError("iteration_count must be non-negative")
        return v

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, v: datetime, info) -> datetime:
        # Ensure updated_at is not before created_at
        if "created_at" in info.data and v < info.data["created_at"]:
            raise ValueError("updated_at cannot be before created_at")
        return v

    def mark_updated(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()

    def set_error(self, message: str) -> None:
        """Set error status with message."""
        self.status = AgentStatus.ERROR
        self.error_message = message
        self.mark_updated()

    def complete(self) -> None:
        """Mark state as completed."""
        self.status = AgentStatus.COMPLETED
        self.mark_updated()
