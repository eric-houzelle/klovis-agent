from __future__ import annotations

from typing import Any

import structlog

from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec, ask_confirmation

logger = structlog.get_logger(__name__)


class ToolRegistry:
    """Central registry of available tools, with per-agent allowlist."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        spec = tool.spec()
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = tool
        logger.info("tool_registered", name=spec.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_specs(self, allowed: list[str] | None = None) -> list[ToolSpec]:
        if allowed is None:
            return [t.spec() for t in self._tools.values()]
        return [t.spec() for name, t in self._tools.items() if name in allowed]

    async def invoke(
        self,
        name: str,
        inputs: dict[str, Any],
        allowed_tools: list[str] | None = None,
    ) -> ToolResult:
        if allowed_tools is not None and name not in allowed_tools:
            logger.warning("tool_not_allowed", name=name, allowed=allowed_tools)
            return ToolResult(success=False, error=f"Tool '{name}' not in allowlist")

        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Tool '{name}' not found")

        if tool.requires_confirmation:
            description = tool.describe_action(inputs)
            approved = await ask_confirmation(description)
            if not approved:
                logger.info("tool_declined_by_user", name=name)
                return ToolResult(
                    success=False,
                    error="Action declined by user",
                )

        logger.info("tool_invoke", name=name)
        try:
            return await tool.execute(inputs)
        except Exception as exc:
            logger.error("tool_error", name=name, error=str(exc))
            return ToolResult(success=False, error=str(exc))
