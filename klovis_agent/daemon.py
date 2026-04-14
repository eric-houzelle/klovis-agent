"""Daemon mode — reactive, event-driven agent existence.

Instead of a fixed-interval OODA cycle, the daemon runs each perception
source in its own async listener task.  Events are pushed into a shared
EventBus.  A single reactive loop drains the bus and decides whether to
act.  While the agent executes a goal, the listeners keep running so
nothing is lost — accumulated events are processed as soon as the action
finishes.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from klovis_agent.console import DIM, RESET, Console
from klovis_agent.decision import decide
from klovis_agent.llm.types import ModelRequest
from klovis_agent.perception.base import (
    Event,
    EventKind,
    PerceptionResult,
    PerceptionSource,
)
from klovis_agent.perception.bus import EventBus
from klovis_agent.perception.inbox import InboxPerceptionSource
from klovis_agent.recall import recall_for_task
from klovis_agent.tools.builtin.discord_bot import (
    DiscordPerceptionSource,
    format_discord_reply,
)
from klovis_agent.tools.builtin.github import GitHubPerceptionSource
from klovis_agent.tools.builtin.moltbook import MoltbookPerceptionSource
from klovis_agent.tools.builtin.semantic_memory import SemanticMemoryStore

if TYPE_CHECKING:
    from klovis_agent.agent import Agent
    from klovis_agent.result import AgentResult

logger = structlog.get_logger(__name__)

_BUS_DRAIN_TIMEOUT = 5.0
_FOLLOWUP_DRAIN_TIMEOUT = 1.0
_ACTION_COOLDOWN_SECONDS = 300


async def _stream_narration(
    llm, con: Console, system: str, user: str, *, max_tokens: int = 100,
) -> None:
    """Stream a short LLM-generated summary to the console."""
    request = ModelRequest(
        purpose="narration",
        system_prompt=system,
        user_prompt=user,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    try:
        async for token in llm.invoke_stream(request):
            con.stream_token(token)
    except Exception:
        logger.debug("narration_stream_failed", exc_info=True)


class AgentDaemon:
    """Reactive daemon: permanent listeners + event bus + decision loop.

    Perception sources are injected by the caller — the daemon does not
    auto-discover anything.  Pass them via ``Agent(perceptions=[...])``
    or directly to this constructor.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        max_cycles: int = 0,
        verbose: bool = False,
        sources: list[PerceptionSource] | None = None,
    ) -> None:
        self._agent = agent
        self._max_cycles = max_cycles
        self._verbose = verbose

        self._sources = sources or []

        self._cycle_count = 0
        self._last_action_time = 0.0

        self._con = Console(verbose=verbose)
        self._bus = EventBus()
        self._shutdown = asyncio.Event()

    # ------------------------------------------------------------------
    # Memory helpers (unchanged)
    # ------------------------------------------------------------------

    async def _recall_context(self, goal: str) -> str:
        try:
            store = SemanticMemoryStore()
            if store.count() == 0:
                return ""
            return await recall_for_task(
                goal=goal,
                embedder=self._agent.embedder,
                store=store,
            )
        except Exception as exc:
            logger.warning("daemon_recall_failed", error=str(exc))
            return ""

    async def _recall_persistent_directives(self) -> str:
        try:
            store = SemanticMemoryStore()
            if store.count(zone="semantic") == 0:
                return ""
            items = store.list_recent(
                limit=8,
                zone="semantic",
                memory_types=["mission", "state", "preference", "strategy"],
            )
            if not items:
                return ""
            lines = []
            for item in items:
                mtype = item.get("metadata", {}).get("type", "other")
                lines.append(f"- ({mtype}) {item.get('content', '')}")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("daemon_directives_recall_failed", error=str(exc))
            return ""

    # ------------------------------------------------------------------
    # Token usage helpers (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    @staticmethod
    def _merge_usage(dst: dict[str, int], src: dict[str, int] | None) -> None:
        if not src:
            return
        dst["prompt_tokens"] += int(src.get("prompt_tokens", 0) or 0)
        dst["completion_tokens"] += int(src.get("completion_tokens", 0) or 0)
        dst["total_tokens"] += int(src.get("total_tokens", 0) or 0)
        dst["calls"] += int(src.get("calls", 0) or 0)

    @staticmethod
    def _usage_from_result(result: "AgentResult") -> dict[str, int]:
        raw = result.artifacts.get("_token_usage", {})
        if not isinstance(raw, dict):
            return AgentDaemon._empty_usage()
        return {
            "prompt_tokens": int(raw.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(raw.get("completion_tokens", 0) or 0),
            "total_tokens": int(raw.get("total_tokens", 0) or 0),
            "calls": int(raw.get("calls", 0) or 0),
        }

    # ------------------------------------------------------------------
    # Source lookup helpers (unchanged)
    # ------------------------------------------------------------------

    def _find_moltbook_source(self) -> MoltbookPerceptionSource | None:
        for s in self._sources:
            if isinstance(s, MoltbookPerceptionSource):
                return s
        return None

    def _find_discord_source(self) -> DiscordPerceptionSource | None:
        for s in self._sources:
            if isinstance(s, DiscordPerceptionSource):
                return s
        return None

    def _find_github_source(self) -> GitHubPerceptionSource | None:
        for s in self._sources:
            if isinstance(s, GitHubPerceptionSource):
                return s
        return None

    def _find_inbox_source(self) -> InboxPerceptionSource | None:
        for s in self._sources:
            if isinstance(s, InboxPerceptionSource):
                return s
        return None

    # ------------------------------------------------------------------
    # Post-action hooks (unchanged)
    # ------------------------------------------------------------------

    async def _run_task(
        self,
        goal: str,
        acted_events: list[Event] | None = None,
        usage_accumulator: dict[str, int] | None = None,
        show_goal: bool = True,
    ) -> None:
        try:
            result = await self._agent.run(goal, show_goal=show_goal)
            self._con.run_result(
                result.status,
                result.iteration_count,
                len(result.steps),
                result.summary or "",
            )
            if usage_accumulator is not None:
                self._merge_usage(usage_accumulator, self._usage_from_result(result))
        except Exception as exc:
            self._con.step_failed(f"Run crashed: {exc}")
            logger.error("daemon_run_failed", error=str(exc))
            self._last_action_time = time.time()
            return

        self._last_action_time = time.time()

        moltbook_src = self._find_moltbook_source()
        if moltbook_src and acted_events:
            for ev in acted_events:
                post_id = ev.metadata.get("post_id", "")
                if post_id:
                    moltbook_src.record_acted_post(post_id)
            try:
                await moltbook_src.mark_notifications_read()
            except Exception as exc:
                logger.warning("daemon_mark_read_failed", error=str(exc))

        github_src = self._find_github_source()
        if github_src and acted_events:
            github_events = [e for e in acted_events if e.source == "github"]
            if github_events:
                try:
                    await github_src.mark_notifications_read()
                except Exception as exc:
                    logger.warning("daemon_github_mark_read_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Request handling (unchanged)
    # ------------------------------------------------------------------

    async def _handle_requests(
        self,
        events: list[Event],
        usage_accumulator: dict[str, int] | None = None,
    ) -> bool:
        requests = [e for e in events if e.kind == EventKind.REQUEST]
        if not requests:
            return False

        inbox = self._find_inbox_source()
        discord_src = self._find_discord_source()

        for req in requests:
            goal = req.detail or req.title
            if req.source == "discord":
                history_context = self._format_discord_recent_context(req.metadata)
                goal = (
                    f"{goal}\n\n"
                    f"{history_context}"
                    "Discord reply requirements:\n"
                    "- Provide a clear, final user-facing answer.\n"
                    "- Use the same language as the user's message.\n"
                    "- If you create a file, still include the actual content in the reply."
                )
            self._con._ts_print(f"Request from {req.source}: {goal[:120]}")

            try:
                result = await self._agent.run(goal)
                self._con.run_result(
                    result.status,
                    result.iteration_count,
                    len(result.steps),
                    result.summary or "",
                )
                if usage_accumulator is not None:
                    self._merge_usage(usage_accumulator, self._usage_from_result(result))

                if discord_src and req.source == "discord":
                    await discord_src.send_reply(
                        req.metadata, format_discord_reply(result),
                    )
            except Exception as exc:
                self._con.step_failed(f"Run crashed: {exc}")
                logger.error("daemon_request_run_failed", error=str(exc))
                if discord_src and req.source == "discord":
                    await discord_src.send_reply(
                        req.metadata, f"Erreur lors de l'exécution : {exc}",
                    )

            if inbox and req.metadata.get("file"):
                inbox.archive(req.metadata["file"])

        return True

    @staticmethod
    def _format_discord_recent_context(metadata: dict[str, object]) -> str:
        raw = metadata.get("recent_messages", [])
        if not isinstance(raw, list) or not raw:
            return ""

        lines = ["Recent conversation context (oldest -> newest):"]
        for item in raw[-8:]:
            if not isinstance(item, dict):
                continue
            author = str(item.get("author_name", "unknown"))
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"- {author}: {content}")

        if len(lines) == 1:
            return ""
        return "\n".join(lines) + "\n\n"

    # ------------------------------------------------------------------
    # Decision helpers (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_decision_events(events: list[Event]) -> tuple[list[Event], int]:
        """Pass through all perceived events to the decision model."""
        return list(events), 0

    @staticmethod
    def _build_recall_query(events: list[Event]) -> str:
        parts: list[str] = []
        for ev in events[:10]:
            post_id = ev.metadata.get("post_id", "")
            from_agent = ev.metadata.get("from", "")
            snippet = ev.title[:100]
            if post_id:
                parts.append(f"post {post_id}: {snippet}")
            elif from_agent:
                parts.append(f"from {from_agent}: {snippet}")
            else:
                parts.append(snippet)

        if not parts:
            return "recent activity and interactions"

        return "Actions already taken on: " + "; ".join(parts)

    # ------------------------------------------------------------------
    # Perception listeners
    # ------------------------------------------------------------------

    async def _run_listener(self, source: PerceptionSource) -> None:
        """Poll a single source in a loop, pushing events into the bus."""
        interval = getattr(source, "poll_interval", 30.0)
        logger.info("listener_started", source=source.name, interval=interval)

        while not self._shutdown.is_set():
            try:
                events = await source.poll()
                for event in events:
                    await self._bus.put(event)
                if events:
                    logger.info(
                        "listener_events",
                        source=source.name,
                        count=len(events),
                    )
            except Exception as exc:
                logger.warning(
                    "listener_poll_error",
                    source=source.name,
                    error=str(exc),
                )

            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=interval,
                )
                break
            except asyncio.TimeoutError:
                pass

        logger.info("listener_stopped", source=source.name)

    # ------------------------------------------------------------------
    # Reactive loop (replaces _cycle)
    # ------------------------------------------------------------------

    async def _reactive_loop(self) -> None:
        """Drain the event bus, decide, act, repeat."""
        while not self._shutdown.is_set():
            events = await self._bus.drain(timeout=_BUS_DRAIN_TIMEOUT)

            if not events:
                if self._shutdown.is_set():
                    break
                continue

            self._cycle_count += 1
            self._con.cycle_start(self._cycle_count)
            cycle_usage = self._empty_usage()

            event_count = len(events)
            by_source: dict[str, int] = {}
            for e in events:
                by_source[e.source] = by_source.get(e.source, 0) + 1
            summary = " | ".join(
                f"{src}: {n} event(s)" for src, n in by_source.items()
            )
            self._con.perceive_result(summary, event_count)

            # Narrate what was perceived
            if events and not self._con.quiet:
                events_lines = "\n".join(
                    f"- [{e.source}/{e.kind.value}] {e.title}"
                    for e in events[:15]
                )
                self._con.perceive_narration_start()
                await _stream_narration(
                    self._agent.llm_router, self._con,
                    system=(
                        "You are a concise narrator for an autonomous agent. "
                        "Summarize the events the agent just perceived in 1-2 short sentences. "
                        "Be specific about sources and what happened. "
                        "Do NOT include technical IDs or tokens. "
                        "Write in the same language as the events."
                    ),
                    user=f"Perceived events:\n{events_lines}",
                    max_tokens=120,
                )
                self._con.stream_end()

            # Handle explicit requests first
            handled_requests = await self._handle_requests(
                events, usage_accumulator=cycle_usage,
            )

            # After handling requests, drain anything that arrived meanwhile
            new_events = await self._bus.drain(timeout=_FOLLOWUP_DRAIN_TIMEOUT)
            other_events = [
                e for e in (events + new_events) if e.kind != EventKind.REQUEST
            ]
            # Handle any new requests that arrived during the previous runs
            if new_events:
                new_requests = [e for e in new_events if e.kind == EventKind.REQUEST]
                if new_requests:
                    await self._handle_requests(
                        new_requests, usage_accumulator=cycle_usage,
                    )

            # Decision on non-request events
            if other_events:
                await self._decide_and_act(
                    other_events, cycle_usage,
                )
            elif not handled_requests:
                self._con._ts_print("Nothing to do. Staying silent.")

            if cycle_usage["calls"] > 0:
                self._con.cycle_usage(
                    cycle_usage["prompt_tokens"],
                    cycle_usage["completion_tokens"],
                    cycle_usage["total_tokens"],
                    cycle_usage["calls"],
                )

            if self._max_cycles and self._cycle_count >= self._max_cycles:
                self._con.daemon_stop(
                    f"max cycles ({self._max_cycles}) reached",
                )
                self._shutdown.set()
                break

    async def _decide_and_act(
        self,
        events: list[Event],
        cycle_usage: dict[str, int],
    ) -> None:
        """Run the decision model on non-request events and optionally act."""
        decision_events, dropped_events = self._filter_decision_events(events)
        if dropped_events:
            self._con._ts_print(
                f"Filtered {dropped_events} non-actionable event(s) before decision."
            )

        directives = await self._recall_persistent_directives()

        if not decision_events:
            if directives:
                self._con._ts_print(
                    "No actionable external events; evaluating persistent directives."
                )
                decision_events = [
                    Event(
                        source="memory",
                        kind=EventKind.SCHEDULE,
                        title="Periodic review of persistent directives",
                        detail="No actionable external events this cycle.",
                    )
                ]
            else:
                self._con._ts_print("No actionable events after filtering. Staying silent.")
                return

        perception = PerceptionResult(events=decision_events)

        recall_query = self._build_recall_query(decision_events)
        if self._verbose:
            self._con._print(f"   {DIM}Recall query: {recall_query}{RESET}")
        recalled = await self._recall_context(recall_query)

        self._con.decision_context(
            events_text=perception.as_text(),
            recalled_memories=recalled,
            persistent_directives=directives,
        )
        self._con.deciding()
        decision_usage = self._empty_usage()
        decision = await decide(
            perception, recalled, self._agent.llm_router,
            soul=self._agent.soul,
            persistent_directives=directives,
            usage_out=decision_usage,
        )
        if decision_usage["total_tokens"] > 0:
            self._con.llm_usage(
                "decision",
                decision_usage["prompt_tokens"],
                decision_usage["completion_tokens"],
                decision_usage["total_tokens"],
            )
            decision_usage["calls"] = 1
            self._merge_usage(cycle_usage, decision_usage)

        # Reasoning narration
        if not self._con.quiet:
            memory_ctx = recalled.strip()[:1000] if recalled else "(no memories)"
            directive_ctx = directives.strip()[:800] if directives else "(no directives)"

            outcome = (
                f"Chosen action: {decision.goal}"
                if decision.should_act
                else f"Decision: stay idle. Reason: {decision.reasoning}"
            )

            self._con.reasoning_narration_start()
            await _stream_narration(
                self._agent.llm_router, self._con,
                system=(
                    "You are a transparent narrator explaining an autonomous agent's thought process. "
                    "Structure your explanation by CITING the source of each piece of information:\n"
                    "- Start with what the agent OBSERVED (from perceived events)\n"
                    "- Then what its MEMORY told it (quote specific memories if provided, "
                    "e.g. 'Memory says: already followed wuya on 2025-04-07')\n"
                    "- Then what its MISSION/DIRECTIVES push it to do (quote the directive)\n"
                    "- End with the logical CONCLUSION: why the agent chose to act OR stay idle\n"
                    "Use 3-4 sentences. Be specific: quote memory content, name agents/posts, "
                    "reference directive types. If there are no memories, say so. "
                    "Write in the same language as the goal or events."
                ),
                user=(
                    f"Perceived events:\n{perception.as_text()}\n\n"
                    f"Recalled memories:\n{memory_ctx}\n\n"
                    f"Persistent directives:\n{directive_ctx}\n\n"
                    f"Agent's reasoning: {decision.reasoning}\n"
                    f"{outcome}"
                ),
                max_tokens=280,
            )
            self._con.stream_end()

        self._con.decision(
            decision.should_act,
            decision.goal,
            decision.reasoning,
            decision.priority,
        )

        if not self._con.quiet:
            self._con.decision_narration_start()
            if decision.should_act and decision.goal:
                await _stream_narration(
                    self._agent.llm_router, self._con,
                    system=(
                        "You are a concise narrator for an autonomous agent. "
                        "Summarize in ONE short sentence (max 20 words) the concrete action the agent will take. "
                        "Be friendly and clear. Write in the same language as the goal."
                    ),
                    user=f"Goal: {decision.goal}",
                )
            else:
                await _stream_narration(
                    self._agent.llm_router, self._con,
                    system=(
                        "You are a concise narrator for an autonomous agent. "
                        "Summarize in ONE short sentence (max 20 words) why the agent chose to do nothing right now. "
                        "Be friendly and clear. Write in the same language as the reasoning."
                    ),
                    user=f"Reasoning: {decision.reasoning}",
                )
            self._con.stream_end()

        if decision.should_act and decision.goal:
            cooldown_remaining = (
                (self._last_action_time + _ACTION_COOLDOWN_SECONDS) - time.time()
            )
            if cooldown_remaining > 0:
                self._con.cooldown(int(cooldown_remaining))
                await asyncio.sleep(cooldown_remaining)
            await self._run_task(
                decision.goal,
                acted_events=decision_events,
                usage_accumulator=cycle_usage,
                show_goal=False,
            )

    # ------------------------------------------------------------------
    # Source lifecycle
    # ------------------------------------------------------------------

    async def _start_sources(self) -> None:
        """Initialize perception sources that require async startup."""
        for source in self._sources:
            if hasattr(source, "start") and callable(source.start):
                try:
                    await source.start()
                except Exception as exc:
                    logger.error(
                        "perception_source_start_failed",
                        source=source.name,
                        error=str(exc),
                    )

    async def _stop_sources(self) -> None:
        """Gracefully shut down perception sources."""
        for source in self._sources:
            if hasattr(source, "stop") and callable(source.stop):
                try:
                    await source.stop()
                except Exception as exc:
                    logger.warning(
                        "perception_source_stop_failed",
                        source=source.name,
                        error=str(exc),
                    )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        source_names = [s.name for s in self._sources]
        self._con.daemon_start(source_names, self._max_cycles)

        if not self._sources:
            self._con._ts_print(
                "WARNING: No perception sources. Daemon has nothing to observe."
            )

        await self._start_sources()

        listener_tasks: list[asyncio.Task] = []
        try:
            for source in self._sources:
                task = asyncio.create_task(
                    self._run_listener(source),
                    name=f"listener-{source.name}",
                )
                listener_tasks.append(task)

            await self._reactive_loop()

        except KeyboardInterrupt:
            self._con.daemon_stop("user")
        except Exception as exc:
            logger.error("daemon_fatal", error=str(exc))
            self._con.daemon_stop(f"crash: {exc}")
            raise
        finally:
            self._shutdown.set()
            for task in listener_tasks:
                task.cancel()
            await asyncio.gather(*listener_tasks, return_exceptions=True)
            await self._stop_sources()
