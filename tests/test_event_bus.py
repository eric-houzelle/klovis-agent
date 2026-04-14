"""Unit tests for the EventBus."""

from __future__ import annotations

import asyncio

import pytest

from klovis_agent.perception.base import Event, EventKind
from klovis_agent.perception.bus import EventBus


def _make_event(title: str = "test", source: str = "test") -> Event:
    return Event(source=source, kind=EventKind.NOTIFICATION, title=title)


@pytest.mark.asyncio
async def test_put_and_get():
    bus = EventBus()
    ev = _make_event("hello")
    await bus.put(ev)
    assert bus.qsize == 1
    got = await bus.get()
    assert got.title == "hello"
    assert bus.qsize == 0


@pytest.mark.asyncio
async def test_drain_returns_empty_on_timeout():
    bus = EventBus()
    events = await bus.drain(timeout=0.1)
    assert events == []


@pytest.mark.asyncio
async def test_drain_collects_available_events():
    bus = EventBus()
    for i in range(5):
        await bus.put(_make_event(f"ev-{i}"))

    events = await bus.drain(timeout=0.5)
    assert len(events) == 5
    assert [e.title for e in events] == [f"ev-{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_drain_collects_late_arrivals():
    """Events pushed shortly after the first should be included in the batch."""
    bus = EventBus()

    async def delayed_push():
        await asyncio.sleep(0.05)
        await bus.put(_make_event("late"))

    await bus.put(_make_event("first"))
    asyncio.create_task(delayed_push())

    events = await bus.drain(timeout=2.0)
    assert len(events) == 2
    assert events[0].title == "first"
    assert events[1].title == "late"


@pytest.mark.asyncio
async def test_qsize():
    bus = EventBus()
    assert bus.qsize == 0
    await bus.put(_make_event())
    await bus.put(_make_event())
    assert bus.qsize == 2
