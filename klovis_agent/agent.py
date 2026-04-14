"""Main Agent class -- the public entry point for klovis-agent.

Replaces the old ``AgentRuntime`` with a composable, library-friendly API.
Tools, perceptions, and memory backends are injected rather than hardcoded.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from ulid import ULID

from klovis_agent.config import EmbeddingConfig, LLMConfig, SandboxConfig
from klovis_agent.console import Console
from klovis_agent.consolidation import consolidate_run
from klovis_agent.core.graph import build_agent_graph
from klovis_agent.llm.embeddings import EmbeddingClient
from klovis_agent.llm.router import LLMRouter
from klovis_agent.models.state import AgentState
from klovis_agent.models.task import Task
from klovis_agent.paths import skills_home
from klovis_agent.recall import recall_for_task
from klovis_agent.result import AgentResult
from klovis_agent.sandbox.service import (
    LocalSandbox,
    OpenSandboxService,
    SandboxExecutionService,
)
from klovis_agent.tools.base import BaseTool
from klovis_agent.tools.builtin.code_execution import CodeExecutionTool
from klovis_agent.tools.builtin.file_tools import FileEditTool, FileReadTool, FileWriteTool
from klovis_agent.tools.builtin.filesystem import (
    FsCopyTool,
    FsDeleteTool,
    FsListTool,
    FsMkdirTool,
    FsMoveTool,
    FsReadTool,
    FsWriteTool,
)
from klovis_agent.tools.builtin.github import register_github_tools
from klovis_agent.tools.builtin.memory import MemoryTool
from klovis_agent.tools.builtin.memory_introspection import MemoryIntrospectionTool
from klovis_agent.tools.builtin.moltbook import bootstrap_moltbook
from klovis_agent.tools.builtin.semantic_memory import (
    SemanticMemoryStore,
    SemanticMemoryTool,
)
from klovis_agent.tools.builtin.shell import ShellCommandTool
from klovis_agent.tools.builtin.skills import (
    InstallSkillTool,
    ListSkillsTool,
    ReadSkillTool,
    SearchRemoteSkillsTool,
    SkillIndex,
    SkillStore,
)
from klovis_agent.tools.builtin.web import HttpRequestTool, WebSearchTool
from klovis_agent.tools.registry import ToolRegistry

try:
    from klovis_agent.tools.builtin.browser import BrowserTool as _BrowserTool

    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False

from klovis_agent.tools.workspace import AgentWorkspace

if TYPE_CHECKING:
    from klovis_agent.perception.base import PerceptionSource

logger = structlog.get_logger(__name__)


def _configure_structlog(verbose: bool) -> None:
    """Configure structlog output level.

    In normal mode, structlog is silenced on stdout so only the Console
    output is visible.  In verbose mode, structlog prints its usual
    key=value lines alongside the Console output.
    """
    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        )
    else:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        )


def _create_sandbox(
    config: SandboxConfig,
    scratch_root: Path,
) -> LocalSandbox | OpenSandboxService:
    if config.backend == "opensandbox":
        return OpenSandboxService(
            domain=config.domain,
            api_key=config.api_key,
            protocol=config.protocol,
            image=config.image,
            timeout_minutes=config.timeout_minutes,
            python_version=config.python_version,
            keep_alive=config.keep_alive,
        )
    return LocalSandbox(
        timeout=config.timeout,
        max_output_bytes=config.max_output_bytes,
        workspace_root=scratch_root,
    )


class Agent:
    """Composable autonomous agent.

    Usage::

        from klovis_agent import Agent, LLMConfig

        agent = Agent(
            llm=LLMConfig(api_key="sk-...", default_model="gpt-4o"),
            tools=[WebSearchTool(), MemoryTool()],
        )
        result = await agent.run("Explain quantum computing")
        print(result.summary)
    """

    def __init__(
        self,
        llm: LLMConfig,
        *,
        embedding: EmbeddingConfig | None = None,
        tools: list[BaseTool] | None = None,
        perceptions: list[PerceptionSource] | None = None,
        sandbox: SandboxConfig | SandboxExecutionService | None = None,
        soul: str | Path | None = None,
        max_iterations: int = 25,
        verbose: bool = False,
        data_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        skills_dirs: list[str | Path] | None = None,
        github_auth: Any | None = None,
        ephemeral: bool = False,
    ) -> None:
        self._verbose = verbose
        self._console = Console(verbose=verbose)

        _configure_structlog(verbose)

        self._llm = LLMRouter(
            api_key=llm.api_key,
            base_url=llm.base_url,
            default_model=llm.default_model,
            default_max_tokens=llm.max_tokens,
            default_temperature=llm.temperature,
            policy=llm.routing_policy,
        )

        emb = embedding or EmbeddingConfig()
        self._embedder = EmbeddingClient(
            api_key=emb.api_key or llm.api_key,
            base_url=emb.base_url or llm.base_url,
            model=emb.model,
        )

        self._workspace = AgentWorkspace(
            data_dir=data_dir,
            cache_dir=cache_dir,
            ephemeral=ephemeral,
        )

        if isinstance(sandbox, SandboxExecutionService):
            self._sandbox: SandboxExecutionService = sandbox
        else:
            self._sandbox = _create_sandbox(
                sandbox if isinstance(sandbox, SandboxConfig) else SandboxConfig(),
                self._workspace.scratch.root,
            )

        self._perceptions = perceptions or []
        self._soul = self._load_soul(soul)
        self._skills_dirs = [Path(p) for p in (skills_dirs or [])]

        self._semantic_store: SemanticMemoryStore | None = None
        self._skill_store: SkillStore | None = None
        self._skill_index: SkillIndex | None = None
        self._skills_indexed: bool = False
        self._github_auth = github_auth
        self._tool_registry = self._build_registry(tools)
        self._max_iterations = max_iterations

        graph = build_agent_graph(self._llm, self._tool_registry)
        self._compiled_graph = graph.compile()

    @staticmethod
    def _load_soul(soul: str | Path | None) -> str:
        if soul is None:
            return ""
        if isinstance(soul, Path):
            if soul.is_file():
                return soul.read_text(encoding="utf-8").strip()
            return ""
        return soul

    def _build_registry(self, tools: list[BaseTool] | None) -> ToolRegistry:
        registry = ToolRegistry()

        if tools is not None:
            for tool in tools:
                registry.register(tool)
            sem_tools = [
                t for t in tools if isinstance(t, SemanticMemoryTool)
            ]
            if sem_tools:
                self._semantic_store = sem_tools[0].store
            if _HAS_PLAYWRIGHT:
                browser_tools = [t for t in tools if isinstance(t, _BrowserTool)]
                self._browser_tool = browser_tools[0] if browser_tools else None
            else:
                self._browser_tool = None
            return registry

        return self._default_registry()

    def _default_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(CodeExecutionTool(self._sandbox))
        registry.register(FileReadTool(self._workspace))
        registry.register(FileWriteTool(self._workspace))
        registry.register(FileEditTool(self._workspace))
        registry.register(ShellCommandTool(self._workspace.scratch))

        registry.register(FsReadTool())
        registry.register(FsListTool())
        registry.register(FsMkdirTool())
        registry.register(FsWriteTool())
        registry.register(FsDeleteTool())
        registry.register(FsMoveTool())
        registry.register(FsCopyTool())

        workspace_skills = Path.cwd() / ".skills"
        user_skills = skills_home()
        if self._skills_dirs:
            skill_dirs = self._skills_dirs
        else:
            skill_dirs = [workspace_skills, user_skills]
        self._skill_store = SkillStore(skill_dirs)
        registry.register(HttpRequestTool(skill_store=self._skill_store))

        registry.register(WebSearchTool())

        if _HAS_PLAYWRIGHT:
            self._browser_tool = _BrowserTool()
            registry.register(self._browser_tool)
        else:
            self._browser_tool = None

        registry.register(MemoryTool())
        sem_tool = SemanticMemoryTool(self._embedder)
        self._semantic_store = sem_tool.store
        registry.register(sem_tool)
        registry.register(MemoryIntrospectionTool(sem_tool.store))

        if self._semantic_store:
            self._skill_index = SkillIndex(self._semantic_store, self._embedder)

        registry.register(ListSkillsTool(self._skill_store))
        registry.register(ReadSkillTool(self._skill_store))
        registry.register(InstallSkillTool(self._skill_store, self._skill_index))
        registry.register(SearchRemoteSkillsTool(self._skill_store))

        bootstrap_moltbook(
            registry, self._llm, workspace=self._workspace,
        )

        if self._github_auth is not None:
            register_github_tools(
                registry,
                self._github_auth,
                scratch_dir=self._workspace.scratch.root,
            )

        return registry

    async def _index_existing_skills(self) -> None:
        """Index all currently installed skills into semantic memory (once)."""
        if not self._skill_index or not self._skill_store:
            return
        for meta in self._skill_store.list_skills():
            content = self._skill_store.get_content(meta.name)
            if content:
                with contextlib.suppress(Exception):
                    await self._skill_index.index_skill(meta, content)

    async def run(self, goal: str, **kwargs: Any) -> AgentResult:
        """Run the full agentic loop for a goal.

        Args:
            goal: The objective to achieve.
            **kwargs: Extra fields forwarded to ``Task`` (task_id, context,
                constraints, success_criteria). Pass ``show_goal=False`` to
                suppress the goal banner (e.g. when the daemon already
                displayed a decision narration).

        Returns:
            An ``AgentResult`` wrapping the final state.
        """
        show_goal = kwargs.pop("show_goal", True)
        run_id = str(ULID())

        task = Task(
            task_id=kwargs.pop("task_id", run_id),
            goal=goal,
            context=kwargs.pop("context", {}),
            constraints=kwargs.pop("constraints", {}),
            success_criteria=kwargs.pop("success_criteria", []),
        )

        self._console.run_start(goal, run_id, show_goal=show_goal)

        if self._skill_index and not self._skills_indexed:
            await self._index_existing_skills()
            self._skills_indexed = True

        if self._semantic_store:
            recalled = await recall_for_task(
                goal=task.goal,
                embedder=self._embedder,
                store=self._semantic_store,
                skill_index=self._skill_index,
            )
            if recalled:
                task = task.model_copy(update={
                    "context": {
                        **task.context,
                        "recalled_memories": recalled,
                    },
                })
                self._console.recall(recalled)

        if self._skill_store is not None:
            skill_names = [s.name for s in self._skill_store.list_skills()]
            if skill_names:
                task = task.model_copy(update={
                    "context": {
                        **task.context,
                        "available_skills": skill_names,
                    },
                })

        initial_state = AgentState(
            run_id=run_id,
            task=task,
            max_iterations=self._max_iterations,
            verbose=self._verbose,
        )

        state_dict = initial_state.model_dump()
        state_dict["_console"] = self._console
        state_dict["soul"] = self._soul
        state_dict["_token_usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

        logger.info("agent_run_start", run_id=run_id, goal=task.goal)

        try:
            final_state_dict = await self._compiled_graph.ainvoke(state_dict)
        finally:
            await self._sandbox.cleanup()
            if self._browser_tool is not None:
                await self._browser_tool.cleanup()

        if "_token_usage" in final_state_dict:
            artifacts = final_state_dict.get("artifacts", {})
            if isinstance(artifacts, dict):
                artifacts["_token_usage"] = final_state_dict["_token_usage"]
                final_state_dict["artifacts"] = artifacts

        final_state_dict.pop("_console", None)
        final_state = AgentState(**final_state_dict)

        logger.info(
            "agent_run_end",
            run_id=run_id,
            status=final_state.status,
            iterations=final_state.iteration_count,
            num_results=len(final_state.step_results),
        )

        if self._semantic_store:
            try:
                n = await consolidate_run(
                    state=final_state,
                    llm=self._llm,
                    embedder=self._embedder,
                    store=self._semantic_store,
                )
                self._console.consolidation(n)
            except Exception as exc:
                logger.warning("consolidation_failed", error=str(exc))

        return AgentResult(final_state)

    async def run_task(self, task: Task) -> AgentResult:
        """Run with a pre-built Task object (for backward compatibility)."""
        return await self.run(
            goal=task.goal,
            task_id=task.task_id,
            context=task.context,
            constraints=task.constraints,
            success_criteria=task.success_criteria,
        )

    def as_daemon(
        self,
        max_cycles: int = 0,
    ) -> "AgentDaemon":  # noqa: F821
        """Create a reactive daemon that uses this agent for execution."""
        from klovis_agent.daemon import AgentDaemon

        return AgentDaemon(
            agent=self,
            max_cycles=max_cycles,
            verbose=self._verbose,
            sources=self._perceptions,
        )

    @property
    def console(self) -> Console:
        return self._console

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def workspace(self) -> AgentWorkspace:
        return self._workspace

    @property
    def workspace_path(self) -> Path:
        return self._workspace.root

    @property
    def llm_router(self) -> LLMRouter:
        return self._llm

    @property
    def soul(self) -> str:
        return self._soul

    @property
    def embedder(self) -> EmbeddingClient:
        return self._embedder
