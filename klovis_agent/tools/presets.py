"""Tool presets for common configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from klovis_agent.tools.base import BaseTool
from klovis_agent.tools.builtin.code_execution import CodeExecutionTool
from klovis_agent.tools.builtin.file_tools import FileReadTool, FileWriteTool
from klovis_agent.tools.builtin.memory import MemoryTool
from klovis_agent.tools.builtin.semantic_memory import SemanticMemoryTool
from klovis_agent.tools.builtin.shell import ShellCommandTool
from klovis_agent.tools.builtin.skills import ListSkillsTool, ReadSkillTool, SkillStore
from klovis_agent.tools.builtin.web import HttpRequestTool, WebSearchTool

if TYPE_CHECKING:
    from klovis_agent.llm.embeddings import EmbeddingClient
    from klovis_agent.sandbox.service import SandboxExecutionService
    from klovis_agent.tools.workspace import AgentWorkspace


def default_tools(
    workspace: AgentWorkspace,
    sandbox: SandboxExecutionService,
    embedder: EmbeddingClient,
    skill_store: SkillStore | None = None,
) -> list[BaseTool]:
    """Standard set of tools for a full-featured agent."""
    return [
        CodeExecutionTool(sandbox),
        FileReadTool(workspace),
        FileWriteTool(workspace),
        ShellCommandTool(workspace.scratch),
        HttpRequestTool(skill_store=skill_store),
        WebSearchTool(),
        MemoryTool(),
        SemanticMemoryTool(embedder),
        *(
            [ListSkillsTool(skill_store), ReadSkillTool(skill_store)]
            if skill_store
            else []
        ),
    ]


def minimal_tools() -> list[BaseTool]:
    """Minimal tools without sandbox or filesystem access."""
    return [WebSearchTool(), MemoryTool()]
