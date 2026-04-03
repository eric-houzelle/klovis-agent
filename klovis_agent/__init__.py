"""klovis-agent — Composable autonomous agent library for Python."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klovis_agent.agent import Agent
    from klovis_agent.config import AgentConfig, EmbeddingConfig, LLMConfig, SandboxConfig
    from klovis_agent.daemon import AgentDaemon
    from klovis_agent.models.task import Task
    from klovis_agent.perception.base import PerceptionSource
    from klovis_agent.perception.inbox import InboxPerceptionSource
    from klovis_agent.result import AgentResult
    from klovis_agent.tools.base import BaseTool, ToolResult, ToolSpec
    from klovis_agent.tools.builtin.github import GitHubPerceptionSource
    from klovis_agent.tools.builtin.moltbook import MoltbookPerceptionSource
    from klovis_agent.tools.registry import ToolRegistry

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentDaemon",
    "AgentResult",
    "BaseTool",
    "EmbeddingConfig",
    "GitHubPerceptionSource",
    "InboxPerceptionSource",
    "LLMConfig",
    "MoltbookPerceptionSource",
    "PerceptionSource",
    "SandboxConfig",
    "Task",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
]

__version__ = "0.2.0"

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Agent": ("klovis_agent.agent", "Agent"),
    "AgentConfig": ("klovis_agent.config", "AgentConfig"),
    "AgentDaemon": ("klovis_agent.daemon", "AgentDaemon"),
    "AgentResult": ("klovis_agent.result", "AgentResult"),
    "BaseTool": ("klovis_agent.tools.base", "BaseTool"),
    "EmbeddingConfig": ("klovis_agent.config", "EmbeddingConfig"),
    "InboxPerceptionSource": ("klovis_agent.perception.inbox", "InboxPerceptionSource"),
    "LLMConfig": ("klovis_agent.config", "LLMConfig"),
    "GitHubPerceptionSource": ("klovis_agent.tools.builtin.github", "GitHubPerceptionSource"),
    "MoltbookPerceptionSource": ("klovis_agent.tools.builtin.moltbook", "MoltbookPerceptionSource"),
    "PerceptionSource": ("klovis_agent.perception.base", "PerceptionSource"),
    "SandboxConfig": ("klovis_agent.config", "SandboxConfig"),
    "Task": ("klovis_agent.models.task", "Task"),
    "ToolRegistry": ("klovis_agent.tools.registry", "ToolRegistry"),
    "ToolResult": ("klovis_agent.tools.base", "ToolResult"),
    "ToolSpec": ("klovis_agent.tools.base", "ToolSpec"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'klovis_agent' has no attribute {name!r}")
