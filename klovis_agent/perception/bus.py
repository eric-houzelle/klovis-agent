"""Event bus — async queue connecting perception listeners to the decision loop.

Each PerceptionSource pushes events into the bus independently.
The reactive loop drains the bus and decides what to do.
"""

from __future__ import annotations

import asyncio

from klovis_agent.perception.base import Event


class EventBus:
    """Thread-safe async queue of perception events."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue()

    async def put(self, event: Event) -> None:
        await self._queue.put(event)

    async def get(self) -> Event:
        """Block until an event is available."""
        return await self._queue.get()

    async def drain(self, timeout: float = 2.0) -> list[Event]:
        """Wait for the first event, then collect any others that arrive
        within *timeout* seconds.  Returns at least one event (blocks on
        the first) or an empty list if *initial_timeout* expires first.
        """
        events: list[Event] = []

        # Block for the first event
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            events.append(first)
        except asyncio.TimeoutError:
            return events

        # Greedily collect follow-up events with a short window
        followup_timeout = 0.5
        while True:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(), timeout=followup_timeout,
                )
                events.append(event)
            except asyncio.TimeoutError:
                break

        return events

    @property
    def qsize(self) -> int:
        return self._queue.qsize()
