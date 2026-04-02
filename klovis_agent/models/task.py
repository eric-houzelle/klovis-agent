"""Task models with Pydantic v2 validation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskPriority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Task status values."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Pydantic model for task with validation."""

    id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., min_length=1, max_length=500, description="Task title")
    description: str | None = Field(default=None, description="Detailed task description")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority level")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    goal: str = Field(..., min_length=1, description="Goal this task contributes to")
    success_criteria: list[str] = Field(default_factory=list, description="Criteria for task completion")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Task creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    completed_at: datetime | None = Field(default=None, description="Task completion timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional task metadata")
    parent_task_id: str | None = Field(default=None, description="Parent task ID if subtask")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def validate_completion_timestamp(self) -> Task:
        if self.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            if self.completed_at is None:
                self.completed_at = datetime.utcnow()
        elif self.completed_at is not None:
            # If status is not completed/cancelled, completed_at should be None
            self.completed_at = None
        return self

    def start(self) -> None:
        """Mark task as in progress."""
        self.status = TaskStatus.IN_PROGRESS
        self.updated_at = datetime.utcnow()

    def complete_task(self) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at

    def fail(self) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.updated_at = datetime.utcnow()

    def cancel(self) -> None:
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
