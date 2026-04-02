from __future__ import annotations

import asyncio
import sys
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """Declarative specification of a tool."""

    name: str
    description: str
    input_schema: dict[str, object] = Field(default_factory=dict)
    output_schema: dict[str, object] = Field(default_factory=dict)
    requires_sandbox: bool = False


class ToolResult(BaseModel):
    """Normalized result of a tool invocation."""

    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class BaseTool(ABC):
    """Base class for all tools.

    Tools that perform potentially dangerous operations can set
    ``requires_confirmation`` to prompt the user before execution.
    The flag defaults to *False* and can be toggled per-instance.
    """

    requires_confirmation: bool = False

    def __init__(self, *, requires_confirmation: bool | None = None) -> None:
        if requires_confirmation is not None:
            self.requires_confirmation = requires_confirmation

    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool specification."""
        ...

    @abstractmethod
    async def execute(self, inputs: dict[str, Any]) -> ToolResult:
        """Execute the tool with validated inputs."""
        ...

    def describe_action(self, inputs: dict[str, Any]) -> str:
        """Human-readable summary of what this call will do.

        Override in subclasses for richer descriptions.  The default
        shows the tool name and input keys.
        """
        spec = self.spec()
        keys = ", ".join(f"{k}={_short(v)}" for k, v in inputs.items())
        return f"{spec.name}({keys})"


async def ask_confirmation(description: str) -> bool:
    """Prompt the user on stdin and return True if they approve."""
    prompt = f"\n⚠️  Confirmation required:\n  {description}\n  Proceed? [y/N] "
    loop = asyncio.get_running_loop()
    sys.stdout.write(prompt)
    sys.stdout.flush()
    answer = await loop.run_in_executor(None, sys.stdin.readline)
    return answer.strip().lower() in ("y", "yes")


def _short(val: Any) -> str:
    s = str(val)
    return s[:80] + "…" if len(s) > 80 else s
