"""Daemon mode -- the agent's continuous existence.

Implements the OODA loop: Observe -> Orient -> Decide -> Act.
The agent periodically perceives its environment through registered
PerceptionSources, decides whether to act, and launches autonomous
runs when warranted. Source-agnostic by design.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from klovis_agent.console import Console
from klovis_agent.decision import decide
from klovis_agent.perception.base import (
    Event,
    EventKind,
    PerceptionResult,
    PerceptionSource,
    perceive,
)
from klovis_agent.perception.inbox import InboxPerceptionSource
from klovis_agent.recall import recall_for_task
from klovis_agent.tools.builtin.moltbook import MoltbookPerceptionSource
from klovis_agent.tools.builtin.semantic_memory import SemanticMemoryStore

if TYPE_CHECKING:
    from klovis_agent.agent import Agent

logger = structlog.get_logger(__name__)


class AgentDaemon:
    """Runs the agent in a continuous perception-decision-action loop.

    Perception sources are injected by the caller — the daemon does not
    auto-discover anything.  Pass them via ``Agent(perceptions=[...])``
    or directly to this constructor.
    """

    def __init__(
        self,
        agent: Agent,
        interval_minutes: float = 30,
        max_cycles: int = 0,
        verbose: bool = False,
        sources: list[PerceptionSource] | None = None,
    ) -> None:
        self._agent = agent
        self._interval = interval_minutes * 60
        self._max_cycles = max_cycles
        self._verbose = verbose

        self._sources = sources or []

        self._cycle_count = 0
        self._last_action_time = 0.0

        self._con = Console(verbose=verbose)

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

    async def _run_task(
        self,
        goal: str,
        acted_events: list[Event] | None = None,
    ) -> None:
        try:
            result = await self._agent.run(goal)
            self._con.run_result(
                result.status,
                result.iteration_count,
                len(result.steps),
                result.summary or "",
            )
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

    def _find_moltbook_source(self) -> MoltbookPerceptionSource | None:
        for s in self._sources:
            if isinstance(s, MoltbookPerceptionSource):
                return s
        return None

    def _find_inbox_source(self) -> InboxPerceptionSource | None:
        for s in self._sources:
            if isinstance(s, InboxPerceptionSource):
                return s
        return None

    async def _handle_requests(
        self, perception: PerceptionResult,
    ) -> bool:
        requests = [
            e for e in perception.events if e.kind == EventKind.REQUEST
        ]
        if not requests:
            return False

        inbox = self._find_inbox_source()

        for req in requests:
            goal = req.detail or req.title
            self._con._ts_print(f"Request from {req.source}: {goal[:120]}")
            await self._run_task(goal)

            if inbox and req.metadata.get("file"):
                inbox.archive(req.metadata["file"])

        return True

    @staticmethod
    def _build_recall_query(events: list[Event]) -> str:
        """Build a recall query from perceived events.

        Instead of the generic "recent activity and interactions", we
        summarize the actual events so the semantic search can find
        memories related to these specific interactions.
        """
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

    async def _cycle(self) -> None:
        self._cycle_count += 1
        self._con.cycle_start(self._cycle_count)

        self._con.perceive_start(len(self._sources))
        perception = await perceive(self._sources)
        if perception.errors:
            self._con.perceive_errors(perception.errors)

        event_count = len(perception.events)
        self._con.perceive_result(perception.summary(), event_count)

        handled_requests = await self._handle_requests(perception)

        other_events = [
            e for e in perception.events if e.kind != EventKind.REQUEST
        ]
        if other_events:
            other_perception = PerceptionResult(
                events=other_events,
                errors=perception.errors,
                timestamp=perception.timestamp,
            )

            recall_query = self._build_recall_query(other_events)
            recalled = await self._recall_context(recall_query)

            self._con.deciding()
            decision = await decide(
                other_perception, recalled, self._agent.llm_router,
                soul=self._agent.soul,
            )
            self._con.decision(
                decision.should_act,
                decision.goal,
                decision.reasoning,
                decision.priority,
            )

            if decision.should_act and decision.goal:
                cooldown_remaining = (
                    (self._last_action_time + 300) - time.time()
                )
                if cooldown_remaining > 0:
                    self._con.cooldown(int(cooldown_remaining))
                    await asyncio.sleep(cooldown_remaining)
                await self._run_task(decision.goal, acted_events=other_events)
        elif not handled_requests:
            self._con._ts_print("Nothing to do. Staying silent.")

    async def run(self) -> None:
        source_names = [s.name for s in self._sources]
        self._con.daemon_start(
            self._interval / 60, source_names, self._max_cycles,
        )

        if not self._sources:
            self._con._ts_print(
                "WARNING: No perception sources. Daemon has nothing to observe."
            )

        try:
            while True:
                await self._cycle()

                if (
                    self._max_cycles
                    and self._cycle_count >= self._max_cycles
                ):
                    self._con.daemon_stop(
                        f"max cycles ({self._max_cycles}) reached",
                    )
                    break

                next_check = time.strftime(
                    "%H:%M:%S",
                    time.localtime(time.time() + self._interval),
                )
                self._con.next_cycle(next_check)
                await asyncio.sleep(self._interval)
        except KeyboardInterrupt:
            self._con.daemon_stop("user")
        except Exception as exc:
            logger.error("daemon_fatal", error=str(exc))
            self._con.daemon_stop(f"crash: {exc}")
            raise
