"""Perception module — the agent's sensory system.

Defines a generic PerceptionSource interface and Event model.
Any external service (Moltbook, email, RSS, cron, filesystem watcher…)
can implement PerceptionSource to feed events into the daemon loop.
The daemon doesn't know or care where events come from.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class EventKind(str, Enum):
    """Broad categories of events the agent can perceive."""

    NOTIFICATION = "notification"
    MESSAGE = "message"
    MENTION = "mention"
    REACTION = "reaction"
    NEW_CONTENT = "new_content"
    REQUEST = "request"
    SCHEDULE = "schedule"
    SYSTEM = "system"
    OTHER = "other"


@dataclass
class Event:
    """A single thing that happened in the agent's environment."""

    source: str
    kind: EventKind
    title: str
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def summary_line(self) -> str:
        return f"[{self.source}:{self.kind.value}] {self.title}"


@dataclass
class PerceptionResult:
    """Aggregated result of polling one or more sources."""

    events: list[Event] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def has_events(self) -> bool:
        return len(self.events) > 0

    def summary(self) -> str:
        if not self.events:
            return "nothing new"
        by_source: dict[str, int] = {}
        for e in self.events:
            by_source[e.source] = by_source.get(e.source, 0) + 1
        parts = [f"{src}: {n} event(s)" for src, n in by_source.items()]
        return " | ".join(parts)

    def as_text(self) -> str:
        """Full textual representation for the LLM decision prompt."""
        if not self.events:
            return "No events detected."
        lines = []
        for e in self.events:
            lines.append(f"- [{e.source}] ({e.kind.value}) {e.title}")
            if e.detail:
                lines.append(f"  {e.detail}")
        return "\n".join(lines)


class PerceptionSource(ABC):
    """Interface for anything that can produce events for the agent.

    Subclasses must implement ``name`` and ``poll()``.  The reactive daemon
    runs each source in its own async task, calling ``poll()`` in a loop
    with ``poll_interval`` seconds between iterations.

    Override ``poll_interval`` to control how often a source is checked
    (default 30 s).  Sources that are push-based (e.g. Discord websocket)
    can set a very long interval and push events from their own callbacks.
    """

    poll_interval: float = 30.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this source (e.g. 'moltbook', 'email')."""
        ...

    @abstractmethod
    async def poll(self) -> list[Event]:
        """Check for new events. Return an empty list if nothing happened."""
        ...


async def perceive(sources: list[PerceptionSource]) -> PerceptionResult:
    """Poll all registered sources and aggregate events."""
    result = PerceptionResult()

    for source in sources:
        try:
            events = await source.poll()
            result.events.extend(events)
            logger.info("perception_source_polled", source=source.name, events=len(events))
        except Exception as exc:
            msg = f"{source.name}: {exc}"
            result.errors.append(msg)
            logger.warning("perception_source_error", source=source.name, error=str(exc))

    logger.info(
        "perception_complete",
        total_events=len(result.events),
        sources=len(sources),
        errors=len(result.errors),
    )
    return result
